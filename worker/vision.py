"""Detection, manifest validation and bounded compositing. Torch runs in child environments."""

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from .process import PipelineError, run
from .restoration import soft_composite, CleanReferences
from .workflow import interpolated_box, bounded_mask, abrupt_area, setting


def read(path, gray=False):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if image is None:
        raise PipelineError("OUTPUT_INVALID", "missing or corrupt frame")
    return image


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_models(path):
    """No implicit downloads. Every file and upstream checkout is pinned by operator."""
    manifest = json.loads(Path(path).read_text())
    for entry in manifest["files"]:
        p = Path(entry["path"])
        if not p.is_file() or sha256(p) != entry["sha256"]:
            raise PipelineError("MODEL_INVALID", "model artifact checksum differs")
    for entry in manifest["repositories"]:
        commit = run(["git", "-C", entry["path"], "rev-parse", "HEAD"]).decode().strip()
        dirty = (
            run(
                [
                    "git",
                    "-C",
                    entry["path"],
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ]
            )
            .decode()
            .strip()
        )
        if commit != entry["commit"] or dirty:
            raise PipelineError(
                "MODEL_INVALID", "model repository differs from locked revision"
            )
    required = (
        {os.environ["SAM_CHECKPOINT"], os.environ["YOLO_WEIGHTS"]}
        if os.environ.get("DETECTOR") == "yolo"
        else {os.environ["SAM_CHECKPOINT"]}
    )
    required |= {
        str(Path(os.environ["PROPAINTER_ROOT"]) / "weights" / name)
        for name in [
            "ProPainter.pth",
            "raft-things.pth",
            "recurrent_flow_completion.pth",
        ]
    }
    locked = {str(Path(e["path"]).resolve()) for e in manifest["files"]}
    repos = {str(Path(e["path"]).resolve()) for e in manifest["repositories"]}
    if (
        not {str(Path(p).resolve()) for p in required} <= locked
        or not {
            str(Path(os.environ[k]).resolve()) for k in ("SAM_ROOT", "PROPAINTER_ROOT")
        }
        <= repos
    ):
        raise PipelineError(
            "MODEL_INVALID",
            "manifest does not cover all required models and source repositories",
        )
    return sha256(path)


def shots(files, check=lambda: None):
    cuts = [0]
    previous = None
    for i, f in enumerate(files):
        check()
        image = cv2.resize(read(f), (160, 90))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image], [0, 1], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if previous is not None:
            old_hist, old_gray = previous
            distance = cv2.compareHist(old_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            difference = np.abs(gray.astype(float) - old_gray)
            delta = difference.mean() / 255
            changed = np.mean(difference > 32)
            if distance > setting("SHOT_HIST_THRESHOLD", 0.5) or (
                delta > setting("SHOT_PIXEL_THRESHOLD", 0.18)
                and changed > setting("SHOT_CHANGED_FRACTION", 0.5)
            ):
                cuts.append(i)
        previous = hist, gray
    return list(zip(cuts, cuts[1:] + [len(files)]))


def detect(files, scenes, options, root, check=lambda: None, dense=False):
    if options.get("multitrack_prior"):
        from .multitrack import make_tracks

        return make_tracks(files, scenes, options["multitrack_prior"], root, check)
    hints = {
        int(p["frame"]): dict(frame=int(p["frame"]), box=p["box"], confidence=1.0)
        for p in options.get("prompts", [])
    }
    seeds = options.get("hls_reference", [])
    for seed in seeds:
        hints.setdefault(int(seed["frame"]), dict(seed))
    if seeds:
        (root / "hls-seeds.json").write_text(json.dumps(seeds))
    if any(i >= len(files) for i in hints):
        raise PipelineError(
            "INVALID_PROMPT", "prompt frame is outside normalized video"
        )
    template_path = os.environ.get("TEMPLATE_PATH")
    detector = os.environ.get("DETECTOR")
    template = (
        read(template_path, True) if template_path and detector != "yolo" else None
    )
    if template is not None and template.std() < 2:
        raise PipelineError("INVALID_TEMPLATE", "template needs visible texture")

    def sample(indices):
        if detector == "yolo":
            (root / "detect-indices.json").write_text(json.dumps(sorted(indices)))
            run(
                [
                    os.environ["DETECT_PYTHON"],
                    str(Path(__file__).with_name("yolo_stage.py")),
                    "--frames",
                    root / "frames",
                    "--output",
                    root / "detector-output.json",
                    "--indices",
                    root / "detect-indices.json",
                    "--weights",
                    os.environ["YOLO_WEIGHTS"],
                    *(["--seeds", root / "hls-seeds.json"] if seeds else []),
                    *(["--continuous-track"] if options.get("hls") else []),
                ],
                check,
            )
            candidates = json.loads((root / "detector-output.json").read_text())
            audit_path = root / "metrics.detector-output.json"
            if options.get("hls") and audit_path.exists():
                audit = json.loads(audit_path.read_text())
                if audit.get("uncovered_watermark_frames", 0) > max(
                    5, len(indices) * 0.05
                ):
                    raise PipelineError(
                        "DETECTION_UNCERTAIN",
                        "检测到多个独立水印候选，当前单轨迹修复不能保证完整覆盖",
                        review=True,
                    )
            if isinstance(candidates, dict):
                raise PipelineError(
                    "DETECTION_UNCERTAIN",
                    "multiple plausible watermark tracks",
                    review=True,
                )
            for candidate in candidates:
                hints.setdefault(candidate["frame"], candidate)
        elif template is not None:
            th, tw = template.shape
            for i in sorted(indices):
                check()
                image = read(files[i], True)
                if image.shape[0] < th or image.shape[1] < tw:
                    continue
                _, confidence, _, (x, y) = cv2.minMaxLoc(
                    cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
                )
                if confidence >= 0.85:
                    hints.setdefault(
                        i,
                        dict(
                            frame=i, box=[x, y, x + tw, y + th], confidence=confidence
                        ),
                    )

    indices = set()
    for start, end in scenes:
        indices.update(range(start, end, 1 if dense or detector == "yolo" else 5))
        indices.add(end - 1)
    sample(indices)
    missing = [
        (start, end)
        for start, end in scenes
        if not any(start <= i < end for i in hints)
    ]
    if missing and not dense:
        sample({i for start, end in missing for i in range(start, end)})
    for start, end in scenes:
        if not any(start <= i < end for i in hints):
            raise PipelineError(
                "DETECTION_UNCERTAIN",
                "a shot has no reliable detection; provide normalized-frame box prompts",
                review=True,
            )
    if detector == "yolo":
        for start, end in scenes:
            observed = sorted(i for i in hints if start <= i < end)
            positions = [start] + observed + [end - 1]
            if max((b - a for a, b in zip(positions, positions[1:])), default=0) > int(
                setting("DETECTION_MAX_GAP", 10)
            ):
                raise PipelineError(
                    "DETECTION_UNCERTAIN",
                    "watermark detection gap too long",
                    review=True,
                )
    image = read(files[0])
    h, w = image.shape[:2]
    for hint in hints.values():
        x1, y1, x2, y2 = hint["box"]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise PipelineError(
                "INVALID_PROMPT", "prompt exceeds normalized frame dimensions"
            )
    ordered = sorted(hints.values(), key=lambda p: p["frame"])
    if options.get("hls"):
        from .workflow import stabilize_hls_hints

        ordered = stabilize_hls_hints(ordered, scenes, seeds)
    result = {
        "scenes": scenes,
        "hints": ordered,
        "dense": dense or bool(missing),
        **({"hls": True} if options.get("hls") else {}),
    }
    (root / "detections.json").write_text(json.dumps(result))
    return result


def masks(root, files, detections, engine, check=lambda: None, budget=None):
    dest = root / "masks"
    dest.mkdir()
    if engine == "propainter":
        execution = ({"timeout": budget["timeout_seconds"], "stage": "mask",
                      "event_path": root / "process-mask.json", "workload": budget["workload"]}
                     if budget is not None else {})
        run(
            [
                os.environ["TRACK_PYTHON"],
                str(Path(__file__).with_name("sam_stage.py")),
                "--root",
                root,
            ],
            check,
            **execution,
        )
    else:
        # Explicit CPU smoke mode: interpolate operator boxes, never advertised as SAM.
        for start, end in detections["scenes"]:
            hints = [p for p in detections["hints"] if start <= p["frame"] < end]
            for i in range(start, end):
                check()
                image = read(files[i])
                mask = np.zeros(image.shape[:2], np.uint8)
                box = [
                    np.interp(
                        i, [p["frame"] for p in hints], [p["box"][j] for p in hints]
                    )
                    for j in range(4)
                ]
                x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
                mask[y1:y2, x1:x2] = 255
                cv2.imwrite(str(dest / files[i].name), mask)
    return validate_masks(root, files, detections, check)


def validate_masks(root, files, detections, check=lambda: None):
    """对已完成的 SAM 输出执行同一套完整覆盖及面积校验，供失败阶段续跑。"""
    if detections.get("schema") == "yolo-model-tracks-1":
        from .model_tracks import validate
        return validate(root, files, detections, check)
    dest = root / "masks"
    if len(list(dest.glob("*.png"))) != len(files):
        raise PipelineError("MASK_INVALID", "mask count differs")
    prior = None
    if detections.get("multitrack"):
        from .multitrack import load_prior

        prior = load_prior(detections["prior"])
    manifest = []
    old_area = None
    for i, f in enumerate(files):
        check()
        mask = read(dest / f.name, True)
        image = read(f)
        if mask.shape != image.shape[:2] or not set(np.unique(mask)) <= {0, 255}:
            raise PipelineError("MASK_INVALID", "mask geometry or values differ")
        scene_hints = [
            p
            for p in detections["hints"]
            if any(s <= i < e and s <= p["frame"] < e for s, e in detections["scenes"])
        ]
        if detections.get("multitrack"):
            from .multitrack import prior_masks_loaded, merge_masks

            components = prior_masks_loaded(
                detections, i, image.shape[:2], prior[0], prior[2]
            )
            supports = merge_masks(components.values(), image.shape[:2])
            # 各目标的时域先验补齐淡色笔画和相邻图标；SAM 仅在支持邻域内扩展。
            allowed = cv2.dilate(supports, np.ones((7, 7), np.uint8))
            mask = cv2.bitwise_or(supports, cv2.bitwise_and(mask, allowed))
        elif os.environ.get("DETECTOR") == "yolo":
            if not np.any(mask):
                raise PipelineError(
                    "TRACKING_DRIFT", "empty SAM watermark segmentation", review=True
                )
            mask = bounded_mask(mask, interpolated_box(scene_hints, i), image)
        else:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        area = np.count_nonzero(mask) / mask.size
        # Conservative stop, rather than silently repairing an uncertain large region.
        if area > 0.2:
            raise PipelineError(
                "MASK_UNCERTAIN", "mask covers over 20% of frame", review=True
            )
        if i in [s[0] for s in detections["scenes"]]:
            old_area = None
        if abrupt_area(old_area, area):
            raise PipelineError(
                "MASK_UNCERTAIN", "mask area changes abruptly", review=True
            )
        old_area = area
        cv2.imwrite(str(dest / f.name), mask)
        yy, xx = np.where(mask > 0)
        bbox = (
            [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
            if len(xx)
            else None
        )
        key_hint = next((p for p in detections["hints"] if p["frame"] == i), None)
        if key_hint and bbox and not detections.get("multitrack"):
            a, b = bbox, key_hint["box"]
            intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                0, min(a[3], b[3]) - max(a[1], b[1])
            )
            union = (
                (a[2] - a[0]) * (a[3] - a[1])
                + (b[2] - b[0]) * (b[3] - b[1])
                - intersection
            )
            if intersection / max(union, 1) < 0.15:
                raise PipelineError(
                    "TRACKING_DRIFT",
                    "mask differs from reliable detection",
                    review=True,
                )
        scene = next((s, e) for s, e in detections["scenes"] if s <= i < e)
        nearest = min(
            (p for p in detections["hints"] if scene[0] <= p["frame"] < scene[1]),
            key=lambda p: abs(p["frame"] - i),
        )
        manifest.append(
            dict(
                frame_index=i,
                shot_id=next(
                    k for k, (s, e) in enumerate(detections["scenes"]) if s <= i < e
                ),
                bbox_xyxy=bbox,
                confidence=nearest["confidence"],
                mask_key=f"masks/{f.name}",
                sha256=sha256(dest / f.name),
            )
        )
    if not any(m["bbox_xyxy"] for m in manifest):
        raise PipelineError(
            "MASK_UNCERTAIN", "tracker produced no watermark mask", review=True
        )
    return manifest


def windows(start, end, size=60, overlap=12):
    if size <= overlap or overlap < 0:
        raise ValueError("invalid window configuration")
    while start < end:
        stop = min(start + size, end)
        yield start, stop
        if stop == end:
            break
        start = stop - overlap


def _composite_window(
    root, files, start, end, overlap, window, check, references, performance
):
    dest = root / "composite"
    predicted = sorted((window / "out" / "frames" / "frames").glob("*.png"))
    if len(predicted) != end - start:
        raise PipelineError("OUTPUT_INVALID", "ProPainter did not return every frame")
    conditioned_path = window / "conditioned.json"
    conditioned = (
        set(json.loads(conditioned_path.read_text()))
        if conditioned_path.exists()
        else set()
    )
    for j, p in enumerate(predicted, start):
        check()
        started = time.perf_counter()
        original = read(files[j])
        mask = read(root / "masks" / files[j].name, True)
        repair = read(p)
        if repair.shape != original.shape:
            raise PipelineError("OUTPUT_INVALID", "repair geometry differs")
        performance["composite_io_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        if j not in conditioned:
            repair = soft_composite(
                original, references.restore(original, mask, repair), mask
            )
        performance["restoration_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        old = dest / files[j].name
        if old.exists():
            previous = read(old)
            region = mask > 0
            disagreement = (
                float(
                    np.abs(previous.astype(float) - repair.astype(float))[region].mean()
                )
                if region.any()
                else 0.0
            )
            performance["overlap_max_mae"] = max(
                performance.get("overlap_max_mae", 0), disagreement
            )
            performance["overlap_samples"] = performance.get("overlap_samples", 0) + 1
            from .hls_settings import HLSSettings

            if disagreement > HLSSettings.from_env().overlap_mae_limit:
                failure = dict(
                    frame=j, window_start=start, window_end=end,
                    disagreement_mae=disagreement,
                    limit=HLSSettings.from_env().overlap_mae_limit,
                    conditioned_overlap=performance.get("conditioned_overlap",False),
                    mask_sha256=sha256(root / "masks" / files[j].name))
                (root / "overlap-failure.json").write_text(json.dumps(failure))
                raise PipelineError(
                    "SEAM_UNCERTAIN",
                    "overlap disagreement requires review",
                    review=True,
                    details=failure,
                )
            weight = (j - start + 1) / (overlap + 1)
            repair = cv2.addWeighted(repair, weight, previous, 1 - weight, 0)
        cv2.imwrite(str(old), repair)
        performance["composite_io_seconds"] += time.perf_counter() - started
    shutil.rmtree(window)


def inpaint(
    root,
    files,
    scenes,
    engine,
    low_memory=False,
    check=lambda: None,
    condition_overlap=False,
):
    dest = root / "composite"
    dest.mkdir()
    if engine == "telea-dev":
        for f in files:
            check()
            image = read(f)
            mask = read(root / "masks" / f.name, True)
            repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
            image[mask > 0] = repaired[mask > 0]
            cv2.imwrite(str(dest / f.name), image)
        return
    processed_indices = [i for start, end in scenes for i in range(start, end)]
    sampled_masks = [
        read(root / "masks" / files[processed_indices[int(i)]].name, True)
        for i in np.linspace(
            0, len(processed_indices) - 1, min(8, len(processed_indices)), dtype=int
        )
    ]
    mask_fraction = max(np.count_nonzero(mask) / mask.size for mask in sampled_masks)
    # 多目标联合掩膜增加稀疏注意力的活动区域，自动缩短窗口而不降分辨率。
    low_memory = low_memory or mask_fraction > 0.05
    from .hls_settings import HLSSettings

    settings = HLSSettings.from_env()
    size, overlap = (
        (settings.window_frames, settings.overlap_frames) if low_memory else (60, 12)
    )
    restoration_stats = []
    performance = dict(
        propainter_seconds=0.0,
        restoration_seconds=0.0,
        composite_io_seconds=0.0,
        window_frames=size,
        overlap_frames=overlap,
        mask_fraction=mask_fraction,
        conditioned_overlap=condition_overlap,
        low_memory=low_memory,
    )

    def save_performance():
        (root / "performance.json").write_text(json.dumps(performance))

    # 单一合成线程严格保留重叠顺序；至多一个窗口合成与下一窗口推理重叠。
    with ThreadPoolExecutor(max_workers=1) as compositor:
        pending = None
        for scene_start, scene_end in scenes:
            # 固定上限控制内存与 CPU 成本，参考始终来自同一镜头原图。
            refs = np.unique(
                np.linspace(
                    scene_start,
                    scene_end - 1,
                    min(24, scene_end - scene_start),
                    dtype=int,
                )
            )
            references = CleanReferences(
                [read(files[j]) for j in refs],
                [read(root / "masks" / files[j].name, True) for j in refs],
            )
            for index, (start, end) in enumerate(
                windows(scene_start, scene_end, size, overlap)
            ):
                check()
                window = root / f"window-{scene_start}-{index}"
                window.mkdir()
                frames = window / "frames"
                frames.mkdir()
                masks_dir = window / "masks"
                masks_dir.mkdir()
                conditioned = []
                if condition_overlap and pending is not None:
                    pending.result()
                    pending = None
                for j in range(start, end):
                    name = f"{j - start:06d}.png"
                    previous = dest / files[j].name
                    if condition_overlap and previous.exists():
                        # 同时刻已完成画面作为干净上下文，不重新遮住它；真实模型输出仍走35阈值核验。
                        os.link(previous, frames / name)
                        cv2.imwrite(
                            str(masks_dir / name),
                            np.zeros(
                                read(root / "masks" / files[j].name, True).shape,
                                np.uint8,
                            ),
                        )
                        conditioned.append(j)
                    else:
                        os.link(files[j], frames / name)
                        os.link(root / "masks" / files[j].name, masks_dir / name)
                if conditioned:
                    (window / "conditioned.json").write_text(json.dumps(conditioned))
                singleton = end - start == 1
                if singleton:
                    # 单帧镜头没有跨帧光流；复制同帧，绝不借用转场另一侧的画面。
                    os.link(files[start], frames / "000001.png")
                    os.link(
                        root / "masks" / files[start].name, masks_dir / "000001.png"
                    )
                args = [
                    os.environ["INPAINT_PYTHON"],
                    str(Path(__file__).with_name("torch_stage.py")),
                    "--metrics",
                    root / f"metrics.inpaint-{scene_start}-{index}.json",
                    "--frames-only",
                    "--script",
                    str(
                        Path(os.environ["PROPAINTER_ROOT"]) / "inference_propainter.py"
                    ),
                    "--video",
                    frames,
                    "--mask",
                    masks_dir,
                    "--output",
                    window / "out",
                    "--save_frames",
                    "--mask_dilation",
                    "0",
                    "--subvideo_length",
                    str(size),
                    "--neighbor_length",
                    "6" if low_memory else "10",
                    "--ref_stride",
                    "20" if low_memory else "10",
                ]
                if os.environ.get("PROPAINTER_FP16") == "true":
                    args.append("--fp16")
                started = time.perf_counter()
                run(args, check, cwd=os.environ["PROPAINTER_ROOT"])
                if singleton:
                    emitted = sorted(
                        (window / "out" / "frames" / "frames").glob("*.png")
                    )
                    if len(emitted) != 2:
                        raise PipelineError(
                            "OUTPUT_INVALID", "单帧镜头的模型上下文输出数量不一致"
                        )
                    emitted[-1].unlink()
                performance["propainter_seconds"] += time.perf_counter() - started
                if pending is not None:
                    pending.result()
                pending = compositor.submit(
                    _composite_window,
                    root,
                    files,
                    start,
                    end,
                    overlap,
                    window,
                    check,
                    references,
                    performance,
                )
            if pending is not None:
                pending.result()
            for key, value in getattr(references, "timings", {}).items():
                performance[key] = performance.get(key, 0.0) + value
            save_performance()
            restoration_stats.append(
                dict(scene=[scene_start, scene_end], **references.stats)
            )
            (root / "restoration.json").write_text(json.dumps(restoration_stats))


def residual_quality(root, files, detections, check=lambda: None):
    """独立复检输出；只统计与原水印轨迹相交的高置信残留。"""
    if os.environ.get("DETECTOR") != "yolo":
        return
    if detections.get("multitrack"):
        from .multitrack import residual_quality

        return residual_quality(root, files, detections, check)
    indices = list(range(len(files)))
    hls = detections.get("hls", False)
    confidence = setting("RESIDUAL_CONFIDENCE", 0.35)
    if hls:
        confidence = min(confidence, setting("HLS_RESIDUAL_CONFIDENCE", 0.15))
    (root / "quality-indices.json").write_text(json.dumps(indices))
    run(
        [
            os.environ["DETECT_PYTHON"],
            str(Path(__file__).with_name("yolo_stage.py")),
            "--frames",
            root / "composite",
            "--output",
            root / "quality-detections.json",
            "--indices",
            root / "quality-indices.json",
            "--weights",
            os.environ["YOLO_WEIGHTS"],
            "--confidence",
            str(confidence),
            "--reference",
            root / "detections.json",
            *(["--reference-margin", "1.5"] if hls else []),
        ],
        check,
    )
    candidates = json.loads((root / "quality-detections.json").read_text())
    from .yolo_stage import residual_match

    count = 0
    if isinstance(candidates, list):
        for p in candidates:
            scene = next((s, e) for s, e in detections["scenes"] if s <= p["frame"] < e)
            hints = [
                h for h in detections["hints"] if scene[0] <= h["frame"] < scene[1]
            ]
            expected = interpolated_box(hints, p["frame"])
            count += residual_match(p["box"], expected, 1.5 if hls else 0)
    uncertain = isinstance(candidates, dict) or count >= max(
        int(setting("RESIDUAL_MIN_FRAMES", 3)),
        len(indices) * setting("RESIDUAL_FRACTION", 0.05),
    )
    (root / "quality.json").write_text(
        json.dumps(
            dict(
                sample_count=len(indices), residual_count=count, needs_review=uncertain
            )
        )
    )
    if uncertain:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "persistent watermark evidence in output", review=True
        )
