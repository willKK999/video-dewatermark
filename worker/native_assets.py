"""七版原生交付的自动素材；只从源片、模型与时域先验提取，不接受人工框。"""

import json
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from .branding import learn_palette
from .matte import fit_overlay
from .multitrack import load_prior
from .process import PipelineError, run
from .speedup_cache import StageCache, fingerprint


def prepare_assets(
    source, root, source_hash, check=lambda: None, prior=None, profile=None
):
    """全片只准备一次原生模板和透明度类别，不进行透明反解。"""
    from .media import probe
    from .multitrack import prepare
    from scripts.hls_template import build

    root.mkdir(parents=True, exist_ok=True)
    cache = StageCache(root / "stages")
    from .roi_inpaint import digest

    from .hls_settings import reusable_profile

    explicit_profile = profile is not None
    profile = profile or os.getenv("HLS_V7_ASSETS")
    if profile and reusable_profile(profile, source_hash, 60):
        from .hls_v7 import load_assets

        supplied = load_assets(Path(profile), source_hash)
        names = ("prior", "alpha", "template", "palette", "projection")
        if supplied.get("low_band_template"):
            if digest(supplied["low_band_template"]) != supplied.get("low_band_sha256"):
                raise PipelineError("OUTPUT_INVALID", "低频素材摘要不一致")
            supplied["sha256"]["low_band_template"] = supplied["low_band_sha256"]
            names += ("low_band_template",)
        graphics = (
            "color_seed_image",
            "color_seed_mask",
            "variant_seed_image",
            "variant_seed_mask",
        )
        if any(supplied.get(name) for name in graphics):
            if not all(supplied.get(name) for name in graphics):
                raise PipelineError("OUTPUT_INVALID", "完整图形模板必须成组提供")
            names += graphics
        # 所有复用项必须有内容校验，不能只验证清单中碰巧列出的文件。
        if any(name not in supplied.get("sha256", {}) for name in names):
            raise PipelineError("OUTPUT_INVALID", "自动素材清单缺少必要内容摘要")
        filenames = dict(
            prior="multitrack-prior.npz",
            alpha="alpha.npz",
            template="template.npz",
            palette="palette.npy",
            projection="projection.json",
            low_band_template="low-band.npz",
            color_seed_image="color-seed-image.png",
            color_seed_mask="color-seed-mask.png",
            variant_seed_image="variant-seed-image.png",
            variant_seed_mask="variant-seed-mask.png",
        )
        targets = {name: root / filenames[name] for name in names}
        if len(set(targets.values())) != len(names):
            raise PipelineError("OUTPUT_INVALID", "自动素材文件名冲突")
        identity = fingerprint(
            [Path(profile), *[Path(supplied[n]) for n in names]], [source_hash]
        )

        def copy_profile():
            for name in names:
                shutil.copyfile(supplied[name], targets[name])

        cache.run(
            "import_profile", identity, copy_profile, lambda: list(targets.values())
        )
        return targets
    if explicit_profile:
        raise PipelineError("OUTPUT_INVALID", "指定的自动素材清单与冻结源片不一致")

    code = [
        digest(Path(__file__).with_name(n))
        for n in ("native_assets.py", "matte.py", "multitrack.py", "branding.py")
    ]
    key = fingerprint(
        [prior] if prior else [], [source_hash, os.getenv("MODEL_LOCK_HASH"), code]
    )
    target = root / "multitrack-prior.npz"

    def make_prior():
        if prior:
            shutil.copyfile(prior, target)
        else:
            prepare(source, root, check)

    cache.run("prior", key, make_prior, lambda: [target])
    duration = float(probe(source, check)["format"]["duration"])
    count = min(60, int(duration))
    if count < 15:
        raise PipelineError(
            "TEMPORAL_UNCERTAIN", "全片原生校准不足十五个独立秒级样本", review=True
        )
    samples = root / "native-samples"

    def sample():
        if samples.exists():
            shutil.rmtree(samples)
        samples.mkdir()
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                source,
                "-vf",
                f"fps={count / duration}",
                "-frames:v",
                str(count),
                "-q:v",
                "2",
                samples / "%06d.jpg",
            ],
            check,
            timeout=7200,
        )
        if len(list(samples.glob("*.jpg"))) != count:
            raise PipelineError("OUTPUT_INVALID", "全片原生采样帧数不足")

    key = fingerprint([], [key, cache.tokens["prior"], count])
    cache.run("samples", key, sample, lambda: [samples])

    def calibrate():
        median, signal, regions, _, _ = load_prior(target)
        mask = np.zeros(median.shape, np.uint8)
        for region in regions:
            mask |= region["mask"]
        images = [
            cv2.resize(cv2.imread(str(p)), (median.shape[1], median.shape[0]))
            for p in sorted(samples.glob("*.jpg"))
        ]
        alpha, _ = fit_overlay(images, mask)
        np.savez_compressed(root / "alpha.npz", alpha=alpha)
        # 颜色只用于确认完整品牌的前景支持，无法学到时不伪造颜色。
        try:
            palette = learn_palette(images, mask)
        except ValueError:
            palette = np.empty((0, 3), np.float32)
        np.save(root / "palette.npy", palette)
        build(samples, root / "template.npz", 3)
        from .low_frequency import band

        np.savez_compressed(
            root / "low-band.npz",
            template=np.median(np.stack([band(image) for image in images]), axis=0),
        )

    cache.run(
        "calibrate",
        fingerprint([], [key, cache.tokens["samples"]]),
        calibrate,
        lambda: [
            root / "alpha.npz",
            root / "template.npz",
            root / "palette.npy",
            root / "low-band.npz",
        ],
    )
    return dict(
        prior=target,
        alpha=root / "alpha.npz",
        template=root / "template.npz",
        palette=root / "palette.npy",
        low_band_template=root / "low-band.npz",
    )


def brand_references(prior):
    from .branding import recognize

    median, _, regions, _, _ = load_prior(prior)
    references = []
    for region in regions:
        x, y, a, b = region["box"]
        patch = np.clip(median[y:b, x:a] * 12 + 128, 0, 255).astype(np.uint8)
        references.append(recognize(cv2.resize(patch, None, fx=3, fy=3), 11))
    return references


def primary_covers(box, points):
    from .short_brand import fraction_inside

    return any(
        fraction_inside(box, p["box"]) >= 0.85 and fraction_inside(p["box"], box) >= 0.5
        for p in points
    )


def confirmed_template(image, box, assets, banks):
    from .brand_templates import variants, match
    from .short_brand import fraction_inside

    if not banks:
        for prefix, scales, threshold in (
            ("color", [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1], 0.55),
            ("variant", [0.95, 1, 1.05], 0.5),
        ):
            key = prefix + "_seed_"
            if assets.get(key + "image") and assets.get(key + "mask"):
                bank = variants(
                    cv2.imread(str(assets[key + "image"])),
                    cv2.imread(str(assets[key + "mask"]), 0),
                    scales,
                )
                banks.append((bank, threshold))
    return any(
        fraction_inside(box, point["box"]) >= 0.85
        for bank, threshold in banks
        for point in match(image, bank, threshold, 2)
    )


def graphic_prompt(image, box, palette):
    """只用自动检测范围内、源片可验证的品牌颜色引导分割，不能把矩形当作前景。"""
    from .branding import palette_support, grow_brand_components

    h, w = image.shape[:2]
    x, y, a, b = map(round, box)
    x, y, a, b = max(0, x), max(0, y), min(w, a), min(h, b)
    mask = np.zeros((h, w), np.uint8)
    if a > x and b > y:
        # 颜色支持在整帧计算，不能把检测框边界误当作原片画面边界。
        support = palette_support(image, palette)
        mask[y:b, x:a] = support[y:b, x:a]
        # 检测器可能只框住图标的一部分；复用七版短时品牌的相邻笔画补全。
        mask |= grow_brand_components(support, [box])
    if np.count_nonzero(mask) < 64:
        raise PipelineError(
            "MASK_UNCERTAIN",
            "图形缺少源片可验证的像素提示，不能用检测框代替前景",
            review=True,
        )
    return mask


def bounded_graphic_mask(mask, box):
    """图形只作范围限制；沿用七版后续固定核补边，不能叠加文字框高度外扩。"""
    from .workflow import expand_box

    h, w = mask.shape
    x, y, a, b = expand_box(box, w, h)
    bounded = np.zeros_like(mask)
    bounded[int(y) : int(np.ceil(b)), int(x) : int(np.ceil(a))] = mask[
        int(y) : int(np.ceil(b)), int(x) : int(np.ceil(a))
    ]
    return bounded


def prepare_graphics(work, detections, assets, fast, check=lambda: None):
    """完整保留候选；短时文字须经自动品牌文本确认，其他字幕不会加入掩膜。"""
    from .short_brand import discover, fraction_inside
    from .brand_templates import variants, match, mask_for

    files = sorted((work / "frames").glob("*.png"))
    graphic = work / "graphic"
    graphic.mkdir(exist_ok=True)
    (graphic / "masks").mkdir(exist_ok=True)
    (graphic / "seeds").mkdir(exist_ok=True)
    indices = graphic / "indices.json"
    indices.write_text(json.dumps(list(range(len(files)))))
    run(
        [
            os.environ["DETECT_PYTHON"],
            Path(__file__).with_name("yolo_stage.py"),
            "--frames",
            work / "frames",
            "--weights",
            os.environ["YOLO_WEIGHTS"],
            "--indices",
            indices,
            "--all-candidates",
            "--output",
            graphic / "raw-candidates.json",
        ],
        check,
    )
    rows = json.loads((graphic / "raw-candidates.json").read_text())
    if len(rows) != len(files) or [p["frame"] for p in rows] != list(range(len(files))):
        raise PipelineError(
            "DETECTION_UNCERTAIN", "完整品牌检测帧数不一致", review=True
        )
    palette = np.load(assets["palette"])
    seeds = []
    primary_points = {}
    for point in detections["hints"]:
        primary_points.setdefault(point["frame"], []).append(point)
    covered_points = []
    # 每个镜头独立发现短时品牌；同一镜头的多个位置逐个去重发现。
    for start, end in detections["scenes"]:
        remaining = [
            dict(row, boxes=list(row["boxes"]))
            for row in rows
            if start <= row["frame"] < end
        ]
        for _ in range(32):
            target = graphic / f"short-{len(seeds)}"
            target.mkdir(exist_ok=True)
            found = discover(
                files,
                dict(detections, scenes=[[start, end]]),
                assets["prior"],
                remaining,
                palette,
                target,
            )
            if not found:
                shutil.rmtree(target)
                break
            seeds.append(found)
            box = found["hints"][0]["box"]
            for row in remaining:
                row["boxes"] = [
                    p for p in row["boxes"] if fraction_inside(p["box"], box) <= 0.5
                ]
            found["hints"][0]["mask_prompt"] = str(
                (target / "prompts/seed.png").relative_to(graphic)
            )
        else:
            raise PipelineError(
                "DETECTION_UNCERTAIN", "单镜头品牌候选过多，需要复核", review=True
            )
    # 原始观测先做已有主轨迹排重，再按对象、镜头和形态生命周期关联。
    from .graphic_tracks import associate

    remaining = []
    for row in rows:
        check()
        known = cv2.imread(str(work / "tracking/masks" / files[row["frame"]].name), 0)
        candidates = []
        for point in row["boxes"]:
            if point["class_id"] not in (0, 1) or point["confidence"] < 0.65:
                continue
            x, y, a, b = map(round, point["box"])
            x, y, a, b = (
                max(0, x),
                max(0, y),
                min(known.shape[1], a),
                min(known.shape[0], b),
            )
            if a <= x or b <= y:
                continue
            if primary_covers(point["box"], primary_points.get(row["frame"], [])):
                covered_points.append(
                    dict(
                        frame=row["frame"],
                        box=point["box"],
                        依据="同帧主轨迹双向覆盖；不重复初始化",
                    )
                )
                continue
            if np.count_nonzero(known[y:b, x:a]) > (a - x) * (b - y) * 0.85:
                continue
            if any(
                seed["scenes"][0][0] <= row["frame"] < seed["scenes"][0][1]
                and fraction_inside(point["box"], seed["hints"][0]["box"]) > 0.5
                for seed in seeds
            ):
                continue
            candidates.append(point)
        remaining.append(dict(frame=row["frame"], boxes=candidates))
    seeds.extend(associate(remaining, detections["scenes"]))
    if len(seeds) > 64:
        raise PipelineError(
            "DETECTION_UNCERTAIN", "品牌轨迹数超出有界调度上限", review=True
        )
    h, w = cv2.imread(str(files[0])).shape[:2]
    for file in files:
        cv2.imwrite(str(graphic / "masks" / file.name), np.zeros((h, w), np.uint8))
    records, text_review, accepted_intervals = [], [], []
    lifecycle_transitions = []
    prompt_review = []
    references = None
    known_banks = []
    for identity, seed in enumerate(seeds):
        check()
        from .branding import recognize, readable_text, same_brand

        point = max(seed["hints"], key=lambda p: p["confidence"])
        source = cv2.imread(str(files[point["frame"]]))
        x, y, a, b = map(round, point["box"])
        crop = source[max(0, y) : min(h, b), max(0, x) : min(w, a)]
        if crop.size and not seed.get("automatic_short_brand"):
            text = recognize(cv2.resize(crop, None, fx=3, fy=3))
            if readable_text(text) or crop.shape[1] >= crop.shape[0] * 2:
                if references is None:
                    references = brand_references(assets["prior"])
                if not same_brand(text, references) and not confirmed_template(
                    source, point["box"], assets, known_banks
                ):
                    text_review.append(
                        dict(
                            frame=point["frame"],
                            box=point["box"],
                            原因="图标类别包含未匹配品牌的文字，暂不修复，保留原始候选待字幕复核",
                        )
                    )
                    continue
        tracker = graphic / f"track-{identity}"
        tracker.mkdir(exist_ok=True)
        (tracker / "frames").symlink_to(
            (work / "frames").resolve(), target_is_directory=True
        )
        (tracker / "masks").mkdir(exist_ok=True)
        hints = []
        observations = seed["hints"]
        last_observation = max(p["frame"] for p in observations)
        from .sam_schedule import adaptive_mask_hints

        def load_prompt(p):
            if p.get("mask_prompt"):
                from .sam_schedule import load_mask_prompt

                return (
                    load_mask_prompt(graphic, p["mask_prompt"], (h, w)).astype(np.uint8)
                    * 255
                )
            try:
                return graphic_prompt(
                    cv2.imread(str(files[p["frame"]])), p["box"], palette
                )
            except PipelineError as error:
                if error.code != "MASK_UNCERTAIN":
                    raise
                prompt_review.append(
                    dict(
                        track=identity,
                        frame=p["frame"],
                        box=p["box"],
                        依据="颜色像素不足，不注入伪前景；保留原观测及完整生命周期，由可靠提示传播覆盖",
                    )
                )
                (graphic / "prompt-review.json").write_text(
                    json.dumps(prompt_review, ensure_ascii=False, indent=2)
                )
                return None

        for p, prompt in adaptive_mask_hints(
            observations, *seed["scenes"][0], load_prompt, fast=fast
        ):
            p = dict(p, track_id=identity + 1)
            (tracker / "prompts").mkdir(exist_ok=True)
            prompt_name = f"prompts/{p['frame']:06d}.png"
            cv2.imwrite(str(tracker / prompt_name), prompt)
            # 原始像素提示单独留证，跟踪临时目录清理后仍可审计。
            evidence = graphic / "prompts" / str(identity)
            evidence.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tracker / prompt_name, evidence / Path(prompt_name).name)
            p["mask_prompt"] = prompt_name
            yy, xx = np.where(prompt > 0)
            p["source_box"] = p["box"]
            p["box"] = [
                int(xx.min()),
                int(yy.min()),
                int(xx.max() + 1),
                int(yy.max() + 1),
            ]
            hints.append(p)
        if not hints:
            raise PipelineError(
                "MASK_UNCERTAIN", "整条图形轨迹没有可靠像素提示", review=True
            )
        (tracker / "detections.json").write_text(
            json.dumps(dict(scenes=seed["scenes"], hints=hints))
        )
        run(
            [
                os.environ["TRACK_PYTHON"],
                Path(__file__).with_name("sam_stage.py"),
                "--root",
                tracker,
                "--speedup" if fast else "--no-speedup",
            ],
            check,
        )
        from .workflow import interpolated_box, abrupt_area
        from .graphic_tracks import valid_area

        for start, end in seed["scenes"]:
            previous_area = None
            for index in range(start, end):
                check()
                file = tracker / "masks" / files[index].name
                component = cv2.imread(str(file), 0)
                if (
                    component is None
                    or component.shape != (h, w)
                    or not set(np.unique(component)) <= {0, 255}
                ):
                    raise PipelineError("MASK_INVALID", "图形分割输出无效")
                component = bounded_graphic_mask(
                    component, interpolated_box(hints, index)
                )
                area = np.count_nonzero(component) / component.size
                if not valid_area(previous_area, area, index, last_observation):
                    raise PipelineError(
                        "MASK_UNCERTAIN", "图形分割超过面积或漂移门槛", review=True
                    )
                if abrupt_area(previous_area, area):
                    lifecycle_transitions.append(
                        dict(
                            track=identity,
                            frame=index,
                            previous_area=previous_area,
                            area=area,
                            last_observation=max(p["frame"] for p in hints),
                            依据="模型最后观测后的尾部缩小，仍须核验原片与编码成品的残留和闪烁",
                        )
                    )
                previous_area = area
                cv2.imwrite(str(file), component)
        point = max(hints, key=lambda p: p["confidence"])
        accepted_intervals.extend(seed["scenes"])
        mask = cv2.imread(str(tracker / "masks" / f"{point['frame']:06d}.png"), 0)
        if mask is None or not mask.any():
            raise PipelineError(
                "TRACKING_DRIFT", "完整品牌分割没有可靠前景", review=True
            )
        image = cv2.imread(str(files[point["frame"]]))
        from .brand_foreground import foreground as source_foreground

        foreground, foreground_basis = source_foreground(image, mask, palette)
        seed_image = graphic / "seeds" / f"{identity}-image.png"
        seed_mask = graphic / "seeds" / f"{identity}-mask.png"
        cv2.imwrite(str(seed_image), image)
        cv2.imwrite(str(seed_mask), foreground)
        bank = variants(image, foreground, [0.95, 1, 1.05])
        from .speedup import bounded_map

        def render_match(item):
            index, file = item
            check()
            # 每个新镜头或新出现的标识已独立初始化；只扩展该次确认的生命周期。
            # 不把短期轨迹的模板无依据传播到整块，原始全帧检测仍完整保留。
            if fast and not any(a <= index < b for a, b in seed["scenes"]):
                return []
            current = cv2.imread(str(file))
            points = match(current, bank, 0.5, 2, allow_partial=True)
            shape = mask_for((h, w), bank, points)
            tracked = cv2.imread(str(tracker / "masks" / file.name), 0)
            if tracked is not None:
                # 保留模型确认的完整图标，不因颜色或模板漏掉淡出边缘。
                shape |= tracked
            target = graphic / "masks" / file.name
            union = cv2.imread(str(target), 0) | cv2.dilate(
                shape, np.ones((5, 5), np.uint8)
            )
            cv2.imwrite(str(target), union)
            return points

        workers = 4 if fast else 1
        matches = list(
            bounded_map(render_match, enumerate(files), depth=workers, workers=workers)
        )
        records.append(
            dict(
                id=identity,
                seed_frame=point["frame"],
                matches=matches,
                foreground_basis=foreground_basis,
            )
        )
    for index, file in enumerate(files):
        mask = cv2.imread(str(graphic / "masks" / file.name), 0)
        if np.count_nonzero(mask) > mask.size * 0.2:
            raise PipelineError(
                "MASK_UNCERTAIN", "品牌联合支持超过原有面积门槛", review=True
            )
        cv2.imwrite(str(graphic / "masks" / f"{index:06d}.png"), mask)
    scenes = accepted_intervals
    (graphic / "lifecycle-transitions.json").write_text(
        json.dumps(lifecycle_transitions, ensure_ascii=False, indent=2)
    )
    (graphic / "covered-observations.json").write_text(
        json.dumps(covered_points, ensure_ascii=False, indent=2)
    )
    (graphic / "text-review.json").write_text(
        json.dumps(text_review, ensure_ascii=False, indent=2)
    )
    supplied = assets.get("graphic")
    if supplied:
        for index, file in enumerate(files):
            path = supplied / "masks" / file.name
            if path.exists():
                mask = cv2.imread(str(path), 0)
                if (
                    mask is None
                    or mask.shape != (h, w)
                    or not set(np.unique(mask)) <= {0, 255}
                ):
                    raise PipelineError("MASK_INVALID", "外部七版自动图形支持无效")
                target = graphic / "masks" / file.name
                union = cv2.imread(str(target), 0) | mask
                if np.count_nonzero(union) > union.size * 0.2:
                    raise PipelineError(
                        "MASK_UNCERTAIN", "完整品牌联合支持超出原门槛", review=True
                    )
                cv2.imwrite(str(target), union)
        if (supplied / "detections.json").exists():
            scenes.extend(
                json.loads((supplied / "detections.json").read_text())["scenes"]
            )
    (graphic / "detections.json").write_text(json.dumps(dict(scenes=scenes)))
    (graphic / "brands.json").write_text(json.dumps(records))
    # 临时分割输入可重建，保留证据、种子和掩膜即可。
    for path in graphic.glob("track-*"):
        shutil.rmtree(path)
    return graphic


def residual_candidates(audit, work, assets, check=lambda: None):
    """对全部编码帧保留原始模型候选；字幕仍须品牌文本印证，不能直接计作水印。"""
    from .branding import recognize, same_brand
    from .workflow import setting

    files = sorted((audit / "composite").glob("*.png"))
    indices = audit / "raw-indices.json"
    indices.write_text(json.dumps(list(range(len(files)))))
    threshold = min(
        setting("RESIDUAL_CONFIDENCE", 0.35), setting("HLS_RESIDUAL_CONFIDENCE", 0.15)
    )
    target = audit / "raw-residual-candidates.json"
    run(
        [
            os.environ["DETECT_PYTHON"],
            Path(__file__).with_name("yolo_stage.py"),
            "--frames",
            audit / "composite",
            "--weights",
            os.environ["YOLO_WEIGHTS"],
            "--indices",
            indices,
            "--all-candidates",
            "--confidence",
            str(threshold),
            "--output",
            target,
        ],
        check,
        timeout=7200,
    )
    rows = json.loads(target.read_text())
    if len(rows) != len(files) or [r["frame"] for r in rows] != list(range(len(files))):
        raise PipelineError("OUTPUT_INVALID", "编码后全画面复检候选帧数不符")
    references = brand_references(assets["prior"])
    confirmed, review = [], []
    cache = {}
    import hashlib

    for row in rows:
        check()
        image = None
        for candidate in row["boxes"]:
            if image is None:
                image = cv2.imread(str(files[row["frame"]]))
            x, y, a, b = map(round, candidate["box"])
            patch = image[
                max(0, y) : min(image.shape[0], b),
                max(0, x) : min(image.shape[1], a),
            ]
            if not patch.size:
                raise PipelineError("OUTPUT_INVALID", "文字候选超出编码画面")
            key = hashlib.sha256(patch.tobytes()).hexdigest()
            if key not in cache:
                if len(cache) >= 256:
                    cache.pop(next(iter(cache)))
                cache[key] = recognize(cv2.resize(patch, None, fx=3, fy=3))
            from .branding import readable_text

            if candidate["class_id"] in (0, 1) and not readable_text(cache[key]):
                confirmed.append(
                    dict(
                        candidate,
                        frame=row["frame"],
                        basis="专用模型非文字图标候选，须与原始证据复核",
                    )
                )
                continue
            is_brand = same_brand(cache[key], references)
            if is_brand:
                confirmed.append(
                    dict(candidate, frame=row["frame"], basis="自动品牌文本一致")
                )
            else:
                review.append(
                    dict(
                        candidate,
                        frame=row["frame"],
                        basis="文字不匹配已知品牌，不能把字幕直接计作水印",
                        original_logo_class=candidate["class_id"] in (0, 1),
                    )
                )
    result = dict(
        原始候选文件=target.name,
        原始候选总数=sum(len(row["boxes"]) for row in rows),
        需要复核=any(p.get("original_logo_class") for p in review),
        全帧数=len(files),
        检测置信度=threshold,
        残留候选=confirmed,
        文字复核依据=review,
    )
    (audit / "raw-residual-review.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    return result
