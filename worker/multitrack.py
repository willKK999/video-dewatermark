"""由跨时段共识建立多水印先验，不使用影片坐标或字幕位置规则。"""

import json
from pathlib import Path
import cv2
import numpy as np
from .process import PipelineError


def highpass(image):
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if image.ndim == 3
        else image.astype(np.float32)
    )
    return gray - cv2.GaussianBlur(gray, (0, 0), 4)


def temporal_consensus(images, groups=5, check=lambda: None):
    if len(images) < groups * 3:
        raise PipelineError(
            "TEMPORAL_UNCERTAIN", "全片时域采样不足，不能可靠排除字幕", review=True
        )
    medians = []
    for indices in np.array_split(np.arange(len(images)), groups):
        values = []
        for i in indices:
            check()
            image = cv2.imread(str(images[i]))
            values.append(highpass(image))
        medians.append(np.median(np.stack(values), axis=0))
    medians = np.stack(medians)
    median = np.median(medians, axis=0)
    noise = float(np.median(np.abs(median)))
    threshold = max(0.5, noise * 5)
    agreement = np.sum(medians * np.sign(median)[None] > 0.3, axis=0)
    signal = ((np.abs(median) > threshold) & (agreement >= groups - 1)).astype(
        np.uint8
    ) * 255
    return median, signal, dict(groups=groups, noise=noise, threshold=threshold)


def connected_regions(signal, boxes):
    """时域连通部件补齐检测框旁的图标；只有模型印证的部件才纳入。"""
    joined = cv2.morphologyEx(
        signal, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 7))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    regions = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if area < 60 or min(w, h) < 8:
            continue
        matching = []
        for p in boxes:
            a, b, c, d = p["box"]
            intersection = max(0, min(x + w, c) - max(x, a)) * max(
                0, min(y + h, d) - max(y, b)
            )
            model_area = max(1, (c - a) * (d - b))
            if w * h >= 0.15 * model_area and intersection >= 0.2 * min(
                w * h, model_area
            ):
                matching.append(p)
        if not matching:
            continue
        mask = np.where(labels == index, 255, 0).astype(np.uint8)
        regions.append(
            dict(
                id=len(regions) + 1,
                box=[x, y, x + w, y + h],
                confidence=max(p["confidence"] for p in matching),
                class_ids=sorted({p["class_id"] for p in matching if "class_id" in p}),
                mask=mask,
            )
        )
    return regions


def fill_support(mask, radius=3):
    """封闭轮廓内部属于同一标识，保留不同标识之间的真实画面。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    cv2.drawContours(result, contours, -1, 255, cv2.FILLED)
    return cv2.dilate(
        result,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    )


def template_candidates(image, template, support, threshold=0.45, limit=12):
    """全画面匹配同一自动标识的独立副本，可同时返回多处。"""
    ih, iw = image.shape
    th, tw = template.shape
    if th > ih or tw > iw:
        return []
    response = cv2.matchTemplate(
        image.astype(np.float32),
        template.astype(np.float32),
        cv2.TM_CCORR_NORMED,
        mask=(support > 0).astype(np.uint8),
    )
    response = np.nan_to_num(response, nan=-1, posinf=-1, neginf=-1)
    items = []
    for _ in range(limit):
        _, score, _, (x, y) = cv2.minMaxLoc(response)
        if score < threshold:
            break
        items.append(dict(box=[x, y, x + tw, y + th], confidence=float(score)))
        response[
            max(0, y - th // 2) : min(response.shape[0], y + th // 2 + 1),
            max(0, x - tw // 2) : min(response.shape[1], x + tw // 2 + 1),
        ] = -1
    return items


def merge_masks(masks, shape):
    output = np.zeros(shape, np.uint8)
    for mask in masks:
        if mask.shape != shape:
            raise PipelineError("MASK_INVALID", "多轨迹掩膜尺寸不一致")
        output = cv2.bitwise_or(output, mask)
    return output


def save_prior(path, median, signal, regions, variants, metrics):
    metadata = dict(
        version=1,
        shape=list(median.shape),
        regions=[{k: v for k, v in r.items() if k != "mask"} for r in regions],
        variants=variants,
        metrics=metrics,
    )
    arrays = dict(median=median, signal=signal, metadata=np.array(json.dumps(metadata)))
    arrays.update({f'mask_{r["id"]}': r["mask"] for r in regions})
    np.savez_compressed(path, **arrays)


def load_prior(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        median = data["median"].copy()
        signal = data["signal"].copy()
        if (
            median.ndim != 2
            or list(median.shape) != metadata["shape"]
            or not np.isfinite(median).all()
            or signal.shape != median.shape
        ):
            raise PipelineError("DETECTION_INVALID", "多轨迹先验数据损坏")
        regions = [
            dict(r, mask=data[f'mask_{r["id"]}'].copy()) for r in metadata["regions"]
        ]
    return median, signal, regions, metadata["variants"], metadata["metrics"]


def bank(median, regions):
    items = []
    for region in regions:
        x, y, r, b = region["box"]
        for flip in [False, True]:
            for scale in [0.85, 1.0, 1.15]:
                patch = median[y:b, x:r]
                support = region["mask"][y:b, x:r]
                if flip:
                    patch, support = cv2.flip(patch, 1), cv2.flip(support, 1)
                size = (max(2, round((r - x) * scale)), max(2, round((b - y) * scale)))
                items.append(
                    dict(
                        region=region["id"],
                        flip=flip,
                        scale=scale,
                        template=cv2.resize(patch, size),
                        support=cv2.resize(
                            support, size, interpolation=cv2.INTER_NEAREST
                        ),
                    )
                )
    return items


def outside_fixed(box, regions):
    # 只抑制同一位置的固定副本；相交的包围框不代表两个标识相同。
    for r in regions:
        a = r["box"]
        intersection = max(0, min(box[2], a[2]) - max(box[0], a[0])) * max(
            0, min(box[3], a[3]) - max(box[1], a[1])
        )
        union = (
            (box[2] - box[0]) * (box[3] - box[1])
            + (a[2] - a[0]) * (a[3] - a[1])
            - intersection
        )
        if intersection / max(1, union) > 0.6:
            return False
    return True


def discover_variants(files, median, regions, check=lambda: None):
    variants = bank(median, regions)
    observations = [[] for _ in variants]
    for frame, path in enumerate(files[::5]):
        check()
        image = highpass(
            cv2.resize(cv2.imread(str(path)), (median.shape[1], median.shape[0]))
        )
        for i, v in enumerate(variants):
            for p in template_candidates(image, v["template"], v["support"], 0.55, 5):
                if outside_fixed(p["box"], regions):
                    observations[i].append(dict(p, frame=frame * 5))
    chosen = []
    # 同一个副本在不同尺度上只保留最有力的模板，不把背景偶合当成新轨迹。
    for region in regions:
        for flip in [False, True]:
            candidates = [
                (i, o)
                for i, o in enumerate(observations)
                if variants[i]["region"] == region["id"]
                and variants[i]["flip"] == flip
                and len({p["frame"] for p in o}) >= 3
            ]
            if candidates:
                i, obs = max(
                    candidates, key=lambda p: sum(x["confidence"] for x in p[1])
                )
                chosen.append({k: variants[i][k] for k in ["region", "flip", "scale"]})
    return chosen


def make_tracks(files, scenes, prior_path, root, check=lambda: None, fps=30):
    import shutil

    median, signal, regions, variants, metrics = load_prior(prior_path)
    shutil.copyfile(prior_path, root / "multitrack-prior.npz")
    source = cv2.imread(str(files[0]))
    height, width = source.shape[:2]
    sx, sy = width / median.shape[1], height / median.shape[0]
    tracks = []
    for region in regions:
        box = [v * (sx if i % 2 == 0 else sy) for i, v in enumerate(region["box"])]
        tracks.append(
            dict(
                id=region["id"],
                kind="fixed",
                region=region["id"],
                hints=[
                    dict(
                        frame=i,
                        box=box,
                        confidence=region["confidence"],
                        track_id=region["id"],
                    )
                    for i in range(len(files))
                ],
            )
        )
    available = bank(median, regions)
    for variant_index, spec in enumerate(variants):
        variant = next(v for v in available if all(v[k] == spec[k] for k in spec))
        instances = []
        for i, path in enumerate(files):
            check()
            image = highpass(
                cv2.resize(cv2.imread(str(path)), (median.shape[1], median.shape[0]))
            )
            candidates = [
                p
                for p in template_candidates(
                    image, variant["template"], variant["support"], 0.45, 12
                )
                if outside_fixed(p["box"], regions)
            ]
            assigned = set()
            for p in sorted(candidates, key=lambda p: p["confidence"], reverse=True):
                center = np.array(
                    [(p["box"][0] + p["box"][2]) / 2, (p["box"][1] + p["box"][3]) / 2]
                )
                eligible = [
                    (float(np.linalg.norm(center - t["center"])), k)
                    for k, t in enumerate(instances)
                    if k not in assigned and i - t["last"] <= fps
                ]
                distance, k = min(eligible, default=(float("inf"), -1))
                if distance > 128 * 30 / fps:
                    k = len(instances)
                    instances.append(
                        dict(
                            id=len(regions) + 1 + variant_index + k * len(variants),
                            kind="moving",
                            **spec,
                            hints=[],
                        )
                    )
                t = instances[k]
                assigned.add(k)
                t["center"] = center
                t["last"] = i
                t["hints"].append(
                    dict(
                        frame=i,
                        box=[
                            v * (sx if j % 2 == 0 else sy)
                            for j, v in enumerate(p["box"])
                        ],
                        confidence=p["confidence"],
                        track_id=t["id"],
                    )
                )
        for t in instances:
            if len(t["hints"]) >= max(1, round(fps * 0.1)):
                t.pop("center")
                t.pop("last")
                t["hints"] = bridge_gaps(t["hints"], len(files), max_gap=round(fps * 2))
                tracks.append(t)
    result = dict(
        hls=True,
        multitrack=True,
        scenes=scenes,
        tracks=tracks,
        hints=sorted(
            [p for t in tracks for p in t["hints"]],
            key=lambda p: (p["frame"], p["track_id"]),
        ),
        prior=str(root / "multitrack-prior.npz"),
        metrics=metrics,
    )
    (root / "detections.json").write_text(json.dumps(result))
    return result


def prior_masks(detections, index, shape):
    median, signal, regions, variants, metrics = load_prior(detections["prior"])
    return prior_masks_loaded(detections, index, shape, median, regions)


def prior_masks_loaded(detections, index, shape, median, regions):
    height, width = shape
    result = {}
    for track in detections["tracks"]:
        region = next(r for r in regions if r["id"] == track["region"])
        if track["kind"] == "fixed":
            mask = fill_support(region["mask"])
        else:
            observed = {p["frame"]: p for p in track["hints"]}
            if index not in observed:
                continue
            p = observed[index]
            x, y, r, b = region["box"]
            support = region["mask"][y:b, x:r]
            if track["flip"]:
                support = cv2.flip(support, 1)
            a, c, d, e = map(
                int,
                [
                    p["box"][0] * median.shape[1] / width,
                    p["box"][1] * median.shape[0] / height,
                    p["box"][2] * median.shape[1] / width,
                    p["box"][3] * median.shape[0] / height,
                ],
            )
            mask = np.zeros_like(median, dtype=np.uint8)
            patch = fill_support(
                cv2.resize(
                    support,
                    (max(1, d - a), max(1, e - c)),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            mask[
                max(0, c) : min(mask.shape[0], e), max(0, a) : min(mask.shape[1], d)
            ] = patch[
                max(0, -c) : min(e - c, mask.shape[0] - c),
                max(0, -a) : min(d - a, mask.shape[1] - a),
            ]
        result[track["id"]] = cv2.resize(
            mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
    return result


def prepare(source, root, check=lambda: None):
    import os, time
    from .process import run

    started = time.monotonic()
    samples = root / "consensus-samples"
    samples.mkdir()
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            source,
            "-vf",
            "fps=1,scale=1280:720:force_original_aspect_ratio=decrease",
            "-q:v",
            "2",
            samples / "%06d.jpg",
        ],
        check,
    )
    files = sorted(samples.glob("*.jpg"))
    median, signal, metrics = temporal_consensus(files, check=check)
    image = root / "consensus.png"
    cv2.imwrite(str(image), np.clip(median * 12 + 128, 0, 255).astype(np.uint8))
    output = root / "consensus-boxes.json"
    run(
        [
            os.environ["DETECT_PYTHON"],
            str(Path(__file__).with_name("consensus_stage.py")),
            "--image",
            image,
            "--weights",
            os.environ["YOLO_WEIGHTS"],
            "--output",
            output,
        ],
        check,
    )
    boxes = json.loads(output.read_text())
    regions = connected_regions(signal, boxes)
    if not regions:
        raise PipelineError(
            "DETECTION_UNCERTAIN", "全片共识没有获得模型印证的标识", review=True
        )
    variants = discover_variants(files, median, regions, check)
    metrics.update(
        sample_count=len(files),
        fixed_tracks=len(regions),
        moving_templates=len(variants),
        seconds=time.monotonic() - started,
        model_candidates=len(boxes),
    )
    path = root / "multitrack-prior.npz"
    save_prior(path, median, signal, regions, variants, metrics)
    (root / "multitrack-metrics.json").write_text(json.dumps(metrics))
    return path


def combine_tracks(detections, prior_path):
    median, signal, regions, variants, metrics = load_prior(prior_path)
    tracks = []
    for identity in sorted({p["track_id"] for p in detections["hints"]}):
        hints = [p for p in detections["hints"] if p["track_id"] == identity]
        # 分块使用确定的模板身份，固定轨迹编号始终等于共识编号。
        if identity <= len(regions):
            spec = dict(kind="fixed", region=identity)
        else:
            spec = dict(
                kind="moving", **variants[(identity - len(regions) - 1) % len(variants)]
            )
        tracks.append(dict(id=identity, hints=hints, **spec))
    return dict(
        detections,
        multitrack=True,
        tracks=tracks,
        prior=str(prior_path),
        metrics=metrics,
    )


def component_scores(template, result, support):
    """完整图案及自适应网格分别核验，防止只剩图标时被文字均值稀释。"""
    h, w = template.shape
    records = []
    full_energy = float(np.sum(template[support > 0] ** 2))
    bounds = [(0, 0, w, h)] + [
        (
            round(x * w / 4),
            round(y * h / 2),
            round((x + 1) * w / 4),
            round((y + 1) * h / 2),
        )
        for y in range(2)
        for x in range(4)
    ]
    for x, y, r, b in bounds:
        active = support[y:b, x:r] > 0
        a = template[y:b, x:r][active]
        z = result[y:b, x:r][active]
        energy = float(np.dot(a, a))
        if len(a) < 64 or energy < full_energy * 0.03:
            continue
        records.append(
            dict(
                part=[x, y, r, b],
                correlation=float(
                    np.dot(a, z) / max(1e-6, np.linalg.norm(a) * np.linalg.norm(z))
                ),
                relative_amplitude=float(np.dot(a, z) / max(1e-6, energy)),
            )
        )
    return records


def residual_quality(root, files, detections, check=lambda: None):
    """独立检查完整模板（包括图标），避免仅按单个文字框宣称零残留。"""
    from .process import PipelineError

    median, signal, regions, variants, metrics = load_prior(detections["prior"])
    templates = bank(median, regions)
    samples = []
    moving_evidence = []
    scores = []
    indices = range(len(files))
    for i in indices:
        check()
        image = cv2.imread(str(root / "composite" / files[i].name))
        feature = highpass(cv2.resize(image, (median.shape[1], median.shape[0])))
        if i % 30 == 0:
            samples.append(feature)
        for track in detections["tracks"]:
            if track["kind"] != "moving":
                continue
            hint = next((p for p in track["hints"] if p["frame"] == i), None)
            if hint is None:
                continue
            v = next(
                v
                for v in templates
                if all(v[k] == track[k] for k in ["region", "flip", "scale"])
            )
            x, y, r, b = [
                round(
                    q
                    * (
                        median.shape[1] / image.shape[1]
                        if j % 2 == 0
                        else median.shape[0] / image.shape[0]
                    )
                )
                for j, q in enumerate(hint["box"])
            ]
            patch = feature[
                max(0, y - 6) : min(feature.shape[0], b + 6),
                max(0, x - 6) : min(feature.shape[1], r + 6),
            ]
            candidates = template_candidates(
                patch, v["template"], v["support"], 0.35, 1
            )
            exact = feature[
                max(0, y) : min(feature.shape[0], b),
                max(0, x) : min(feature.shape[1], r),
            ]
            partial = []
            if exact.shape == v["template"].shape:
                partial = component_scores(v["template"], exact, v["support"])
            strong_parts = [
                p
                for p in partial
                if p["correlation"] >= 0.55 and p["relative_amplitude"] >= 0.15
            ]
            if candidates or strong_parts:
                confidence = max(
                    [p["confidence"] for p in candidates]
                    + [p["correlation"] for p in strong_parts]
                )
                moving_evidence.append(
                    dict(
                        frame=i,
                        track_id=track["id"],
                        confidence=confidence,
                        components=strong_parts,
                        box=hint["box"],
                    )
                )
    fixed = []
    if samples:
        consensus = np.median(np.stack(samples), axis=0)
        for region in regions:
            x, y, r, b = region["box"]
            support = region["mask"][y:b, x:r] > 0
            components = component_scores(
                median[y:b, x:r], consensus[y:b, x:r], support
            )
            scores.append(
                dict(
                    track_id=region["id"],
                    components=components,
                    correlation=max((p["correlation"] for p in components), default=0),
                    relative_amplitude=max(
                        (p["relative_amplitude"] for p in components), default=0
                    ),
                )
            )
            if any(
                p["correlation"] >= 0.35 and p["relative_amplitude"] >= 0.15
                for p in components
            ):
                fixed.append(region["id"])
    count = len(moving_evidence) + len(fixed)
    quality = dict(
        sample_count=len(files),
        residual_count=count,
        needs_review=bool(count),
        fixed_template_scores=scores,
        moving_evidence=moving_evidence,
        method="全帧移动完整模板与固定轨迹时域共识复检",
    )
    (root / "quality.json").write_text(json.dumps(quality))
    (root / "quality-detections.json").write_text(json.dumps(moving_evidence))
    if count:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "多轨迹完整标识复检发现残留证据", review=True
        )


def refine(root, files, detections, engine, low_memory, fps, stage, check):
    """多轨迹复修只扩展有证据的标识支持，保留其它目标并完整重跑上下文。"""
    import shutil, time
    from . import vision

    records = []
    median, signal, regions, variants, metrics = load_prior(detections["prior"])
    for iteration in range(2):
        quality = json.loads((root / "quality.json").read_text())
        if not quality["residual_count"]:
            break
        started = time.monotonic()
        fixed = {
            x["track_id"]
            for x in quality["fixed_template_scores"]
            if any(
                p["correlation"] >= 0.35 and p["relative_amplitude"] >= 0.15
                for p in x.get("components", [x])
            )
        }
        moving = {(p["frame"], p["track_id"]) for p in quality["moving_evidence"]}
        affected = {
            i for i in range(len(files)) if fixed or any(f == i for f, t in moving)
        }
        ranges = [
            (s, e) for s, e in detections["scenes"] if any(s <= i < e for i in affected)
        ]
        work = root / f"multitrack-refine-{iteration}"
        work.mkdir()
        (work / "masks").mkdir()
        (work / "frames").symlink_to(
            (root / "frames").resolve(), target_is_directory=True
        )
        for i, path in enumerate(files):
            check()
            original = cv2.imread(str(root / "masks" / path.name), 0)
            components = prior_masks_loaded(
                detections, i, original.shape, median, regions
            )
            for identity, component in components.items():
                if identity in fixed or any(t == identity for f, t in moving):
                    radius = 6 + iteration * 3
                    original = cv2.bitwise_or(
                        original,
                        cv2.dilate(
                            component,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
                            ),
                        ),
                    )
            if np.count_nonzero(original) / original.size > 0.2:
                raise PipelineError(
                    "MASK_UNCERTAIN", "多目标复修掩膜超过画面 20%", review=True
                )
            cv2.imwrite(str(work / "masks" / path.name), original)
        stage("inpaint")
        vision.inpaint(work, files, ranges, engine, low_memory, check)
        for path in (work / "composite").glob("*.png"):
            shutil.copyfile(path, root / "composite" / path.name)
        stage("quality")
        try:
            residual_quality(root, files, detections, check)
        except PipelineError as exc:
            if exc.code != "RESIDUAL_UNCERTAIN":
                raise
        after = json.loads((root / "quality.json").read_text())
        records.append(
            dict(
                round=iteration + 1,
                residual_before=quality["residual_count"],
                residual_after=after["residual_count"],
                context_ranges=ranges,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
        )
        shutil.rmtree(work)
    (root / "hls-refinements.json").write_text(json.dumps(records, ensure_ascii=False))
    if json.loads((root / "quality.json").read_text())["residual_count"]:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "多目标复修后仍有完整标识残留", review=True
        )
    return records


def bridge_gaps(hints, count, max_gap=60):
    """只在同一身份的可靠前后观测之间插值，禁止跨越长时间消失。"""
    observed = {p["frame"]: p for p in hints}
    for left, right in zip(hints, hints[1:]):
        gap = right["frame"] - left["frame"]
        if 1 < gap <= max_gap:
            for frame in range(left["frame"] + 1, right["frame"]):
                fraction = (frame - left["frame"]) / gap
                observed[frame] = dict(
                    left,
                    frame=frame,
                    box=[
                        a + (b - a) * fraction
                        for a, b in zip(left["box"], right["box"])
                    ],
                    confidence=min(left["confidence"], right["confidence"]),
                    interpolated=True,
                )
    return [observed[i] for i in sorted(observed)]
