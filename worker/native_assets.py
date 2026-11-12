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


def prepare_graphics(work, detections, assets, fast, check=lambda: None):
    """只调用七版完整模板与逐镜头SAM，不建立新的图形判定。"""
    from .hls_v7 import graphic_masks
    from .colour_tracks import complete

    files = sorted((work / "frames").glob("*.png"))
    graphic_masks(
        files, {key: str(value) for key, value in assets.items()}, work, check
    )
    complete(
        work, {key: str(value) for key, value in assets.items()}, detections, check
    )
    if not (work / "graphic").exists():
        (work / "graphic").symlink_to(work / "graphics", target_is_directory=True)


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
