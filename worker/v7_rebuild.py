"""按七版原始生成顺序，在本次运行内从原片建立九项输入。"""

import json
import os
import shutil
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from . import media, multitrack, vision
from .branding import learn_palette, palette_support
from .fine_masks import Supports
from .matte import fit_overlay
from .process import PipelineError, run
from .roi_inpaint import digest
from .short_brand import discover
from .v7_equivalent import REQUIRED_ASSETS


def calibrate(samples, prior, root):
    """原透明度与颜色配方；不使用此前提速版的缩小、均匀六十帧配方。"""
    median, signal, regions, _, _ = multitrack.load_prior(prior)
    files = sorted(samples.glob("*.jpg"))
    first = cv2.imread(str(files[0]))
    height, width = first.shape[:2]
    mh, mw = median.shape
    mask = np.zeros((height, width), np.uint8)
    for region in regions:
        actual = multitrack.fill_support(signal & region["mask"], 2)
        mask |= cv2.resize(actual, (width, height), interpolation=cv2.INTER_NEAREST)
    anchor = max(regions, key=lambda r: r["confidence"])
    x, y, a, b = anchor["box"]
    support = anchor["mask"][y:b, x:a] > 0
    template = median[y:b, x:a][support]
    selected, colors, color_indices = [], [], []
    for i, file in enumerate(files):
        image = cv2.imread(str(file))
        small = cv2.resize(image, (mw, mh))
        feature = multitrack.highpass(small)[y:b, x:a][support]
        correlation = float(
            template
            @ feature
            / max(1e-6, np.linalg.norm(template) * np.linalg.norm(feature))
        )
        if correlation > 0.35:
            selected.append(file)
            if i % 5 == 0:
                colors.append(small[y:b, x:a])
                color_indices.append(i)
    alpha, offset = fit_overlay((cv2.imread(str(p)) for p in selected), mask)
    np.savez_compressed(root / "alpha.npz", alpha=alpha, offset=offset, mask=mask)
    palette = learn_palette(
        colors, ((median[y:b, x:a] > 2) & support).astype(np.uint8) * 255
    )
    np.save(root / "palette.npy", palette)
    (root / "calibration-samples.json").write_text(
        json.dumps(
            dict(
                透明度样本=[p.name for p in selected],
                颜色样本=color_indices,
                锚点区域=anchor["id"],
                原生尺寸=[width, height],
                可见相关阈值=0.35,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


def fine_masks(work, files, detected, prior, alpha, graphic, check):
    """原脚本的 Supports 联合及原百分之二十保护，直接使用七版函数。"""
    work.mkdir()
    (work / "frames").symlink_to(files[0].parent.resolve(), target_is_directory=True)
    (work / "masks").mkdir()
    support = Supports(prior, detected, np.load(alpha)["alpha"], graphic)
    areas = []
    for index, file in enumerate(files):
        check()
        image = cv2.imread(str(file))
        mask = np.zeros(image.shape[:2], np.uint8)
        for part in support.at(index, image):
            mask |= part["mask"]
        area = float(np.count_nonzero(mask) / mask.size)
        if area > 0.2:
            raise PipelineError(
                "MASK_UNCERTAIN", "稀疏联合掩膜超过原20%门槛", review=True
            )
        cv2.imwrite(str(work / "masks" / file.name), mask)
        areas.append(area)
    (work / "supports.json").write_text(json.dumps(dict(areas=areas)))


def remainder(coarse, graphic, palette, work, files, check):
    """原七版图形补残：颜色交集、原片未变像素及原近邻范围。"""
    work.mkdir()
    (work / "frames").symlink_to(
        (coarse / "composite").resolve(), target_is_directory=True
    )
    (work / "masks").mkdir()
    scenes = json.loads((graphic / "detections.json").read_text())["scenes"]
    targets = []
    for index, file in enumerate(files):
        check()
        image = cv2.imread(str(file))
        mask = np.zeros(image.shape[:2], np.uint8)
        if any(a <= index < b for a, b in scenes):
            sam = cv2.imread(str(graphic / "masks" / file.name), 0)
            if sam is not None and sam.any():
                yy, xx = np.where(sam > 0)
                radius = max(3, round((yy.max() - yy.min() + 1) * 0.1))
                near = cv2.dilate(
                    sam,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
                    ),
                )
                result = cv2.imread(str(coarse / "composite" / file.name))
                possible = (
                    palette_support(image, palette)
                    & palette_support(result, palette)
                    & near
                )
                unchanged = (
                    np.max(np.abs(image.astype(float) - result.astype(float)), axis=2)
                    < 18
                )
                mask = np.where((possible > 0) & unchanged, 255, 0).astype(np.uint8)
                mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
                if mask.any():
                    targets.append(index)
        cv2.imwrite(str(work / "masks" / file.name), mask)
    if targets:
        left, right = max(0, min(targets) - 30), min(len(files), max(targets) + 31)
        vision.inpaint(
            work,
            sorted((work / "frames").glob("*.png")),
            [(left, right)],
            "propainter",
            True,
            check,
            roi=False,
        )
        for path in (work / "composite").glob("*.png"):
            shutil.copyfile(path, coarse / "composite" / path.name)
        for index in targets:
            name = files[index].name
            extra = cv2.imread(str(work / "masks" / name), 0)
            for directory in (coarse / "masks", graphic / "masks"):
                path = directory / name
                if path.exists():
                    cv2.imwrite(str(path), cv2.imread(str(path), 0) | extra)
    (work / "repair-complete.json").write_text(json.dumps(dict(target_frames=targets)))


def rebuild(source, root, source_hash, check=lambda: None, seconds=27):
    """拒绝旧目录；不接收冻结清单、旧先验、旧种子或阶段结果。"""
    from scripts.hls_template import build
    from scripts.hls_align_reference import align
    from scripts.hls_fit_projection import fit

    if root.exists() and any(root.iterdir()):
        raise PipelineError("OUTPUT_INVALID", "九项输入重建必须使用全新空目录")
    root.mkdir(parents=True, exist_ok=True)
    stages = root / "stages"
    stages.mkdir()
    timings = {}

    def perform(name, operation):
        check()
        start = time.monotonic()
        complete = False
        print("原片输入重建阶段：" + name, flush=True)
        try:
            result = operation()
            complete = True
            return result
        finally:
            timings[name] = dict(
                seconds=time.monotonic() - start, reused=False, completed=complete
            )
            (stages / "timings.json").write_text(json.dumps(timings))
            if complete:
                (stages / (name + ".json")).write_text("{}")

    # 生产先验的原入口与独立原生秒级采样分别保留原解码相位。
    prior_root = root / "prior-work"
    prior_root.mkdir()
    prior = perform("prior", lambda: multitrack.prepare(source, prior_root, check))
    samples = root / "source-samples"
    samples.mkdir()
    video = next(
        v for v in media.probe(source)["streams"] if v["codec_type"] == "video"
    )
    rate = Fraction(video["avg_frame_rate"])
    if rate.denominator != 1:
        raise PipelineError(
            "MODEL_INVALID", "原七版秒级抽样配方仅适用整数帧率，不能擅自更换相位"
        )
    perform(
        "samples",
        lambda: run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                source,
                "-vf",
                f"select=not(mod(n\\,{rate.numerator}))",
                "-fps_mode",
                "passthrough",
                "-q:v",
                "2",
                "-start_number",
                "0",
                samples / "%06d.jpg",
            ],
            check,
            timeout=7200,
        ),
    )
    if len(list(samples.glob("*.jpg"))) < 15:
        raise PipelineError("TEMPORAL_UNCERTAIN", "原生秒级样本不足十五帧", review=True)
    perform("calibration", lambda: calibrate(samples, prior, root))
    perform("template", lambda: build(samples, root / "template.npz", 3))

    bootstrap = root / "bootstrap"
    bootstrap.mkdir()
    clip = bootstrap / "input.mp4"

    def decode():
        # 保留原品牌种子生成时的两次编码；去掉会改变其输入像素。
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                source,
                "-t",
                str(seconds),
                "-an",
                "-r",
                "30",
                "-c:v",
                "libx264",
                "-crf",
                "16",
                clip,
            ],
            check,
            timeout=1800,
        )
        media.normalize(clip, bootstrap, 720, check)

    perform("seed_decode", decode)
    files = sorted((bootstrap / "frames").glob("*.png"))
    detected = perform(
        "seed_detect",
        lambda: multitrack.make_tracks(
            files, vision.shots(files, check), prior, bootstrap, check
        ),
    )
    candidates = bootstrap / "all-yolo.json"
    perform(
        "seed_candidates",
        lambda: run(
            [
                os.environ["DETECT_PYTHON"],
                "-m",
                "scripts.hls_candidates",
                "--frames",
                bootstrap / "frames",
                "--output",
                candidates,
            ],
            check,
            timeout=7200,
        ),
    )
    color = root / "color-bootstrap"
    color.mkdir()
    (color / "frames").symlink_to(
        (bootstrap / "frames").resolve(), target_is_directory=True
    )
    (color / "masks").mkdir()

    def seed():
        result = discover(
            files,
            detected,
            prior,
            json.loads(candidates.read_text()),
            np.load(root / "palette.npy"),
            color,
        )
        if result is None:
            raise PipelineError(
                "MASK_UNCERTAIN",
                "原七版短品牌发现没有生成种子，禁止改用其他提示",
                review=True,
            )
        index = result["hints"][0]["frame"]
        shutil.copyfile(files[index], root / "color-seed-image.png")
        shutil.copyfile(color / "prompts/seed.png", root / "color-seed-mask.png")

    perform("color_seed", seed)
    perform(
        "color_sam",
        lambda: run(
            [
                os.environ["TRACK_PYTHON"],
                Path(__file__).with_name("sam_stage.py"),
                "--root",
                color,
            ],
            check,
            timeout=7200,
        ),
    )
    coarse = root / "fine-bootstrap"
    perform(
        "fine_masks",
        lambda: fine_masks(
            coarse, files, detected, prior, root / "alpha.npz", color, check
        ),
    )
    perform(
        "fine_inpaint",
        lambda: vision.inpaint(
            coarse, files, detected["scenes"], "propainter", False, check, roi=False
        ),
    )
    perform(
        "graphic_remainder",
        lambda: remainder(
            coarse,
            color,
            np.load(root / "palette.npy"),
            root / "graphic-remainder",
            files,
            check,
        ),
    )
    variant = root / "variant-bootstrap"
    perform(
        "variant_seed",
        lambda: run(
            [
                os.environ["INPAINT_PYTHON"],
                "-m",
                "scripts.hls_brand_variants",
                "--frames",
                bootstrap / "frames",
                "--candidates",
                candidates,
                "--known-masks",
                coarse / "masks",
                "--confirmed-brand",
                color,
                "--palette",
                root / "palette.npy",
                "--output",
                variant,
            ],
            check,
            timeout=7200,
        ),
    )
    repaired = root / "variant-repair"
    repaired.mkdir()
    (repaired / "frames").symlink_to(
        (coarse / "composite").resolve(), target_is_directory=True
    )
    (repaired / "masks").symlink_to(
        (variant / "masks").resolve(), target_is_directory=True
    )
    perform(
        "variant_inpaint",
        lambda: vision.inpaint(
            repaired,
            sorted((repaired / "frames").glob("*.png")),
            detected["scenes"],
            "propainter",
            True,
            check,
            roi=False,
        ),
    )
    combined = root / "combined"
    combined.mkdir()
    (combined / "frames").symlink_to(
        (bootstrap / "frames").resolve(), target_is_directory=True
    )
    (combined / "composite").symlink_to(
        (repaired / "composite").resolve(), target_is_directory=True
    )
    (combined / "masks").mkdir()
    graphics = root / "graphics"
    (graphics / "masks").mkdir(parents=True)
    shutil.copyfile(color / "detections.json", graphics / "detections.json")

    def combine():
        for file in files:
            check()
            extra = cv2.imread(str(variant / "masks" / file.name), 0)
            union = cv2.imread(str(coarse / "masks" / file.name), 0) | extra
            if np.count_nonzero(union) / union.size > 0.2:
                raise PipelineError(
                    "MASK_UNCERTAIN", "新联合掩膜超出原20%门槛", review=True
                )
            cv2.imwrite(str(combined / "masks" / file.name), union)
            old_path = color / "masks" / file.name
            old = (
                cv2.imread(str(old_path), 0)
                if old_path.exists()
                else np.zeros_like(extra)
            )
            cv2.imwrite(str(graphics / "masks" / file.name), old | extra)

    perform("combine", combine)
    reference = root / "aligned-reference"
    perform(
        "reference", lambda: align(source, bootstrap / "frames", reference, check=check)
    )
    perform(
        "projection",
        lambda: fit(
            combined,
            reference,
            bootstrap / "detections.json",
            prior,
            root / "alpha.npz",
            root / "template.npz",
            graphics,
            root / "projection.json",
            3,
        ),
    )
    shutil.copyfile(prior, root / "multitrack-prior.npz")
    for kind in ("image", "mask"):
        shutil.copyfile(variant / f"seed-{kind}.png", root / f"variant-seed-{kind}.png")
    assets = dict(
        prior=root / "multitrack-prior.npz",
        alpha=root / "alpha.npz",
        template=root / "template.npz",
        palette=root / "palette.npy",
        projection=root / "projection.json",
        color_seed_image=root / "color-seed-image.png",
        color_seed_mask=root / "color-seed-mask.png",
        variant_seed_image=root / "variant-seed-image.png",
        variant_seed_mask=root / "variant-seed-mask.png",
    )
    assert set(assets) == set(REQUIRED_ASSETS)
    (root / "rebuilt-assets.json").write_text(
        json.dumps(
            dict(
                method="v7",
                source_sha256=source_hash,
                本次从原片重建=True,
                **{key: str(value.resolve()) for key, value in assets.items()},
                sha256={key: digest(value) for key, value in assets.items()},
                源码摘要={
                    name: digest(Path(__file__).with_name(name))
                    for name in (
                        "v7_rebuild.py",
                        "fine_masks.py",
                        "branding.py",
                        "matte.py",
                        "short_brand.py",
                        "sam_stage.py",
                    )
                },
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return assets
