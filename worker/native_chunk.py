"""整集与基准共用的七版原生分块执行器；只处理传入边界，不调度后续块。"""

import json
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
from . import vision
from .fine_masks import Supports
from .process import PipelineError, run
from .speedup_cache import StageCache, fingerprint
from .native_assets import prepare_graphics


def decode_chunk(source, work, plan, check=lambda: None):
    frames = work / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    # 用原生帧网格截取，先归零时间戳再作模型采样，不作中间视频转码。
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            source,
            "-an",
            "-vf",
            f"trim=start_frame={plan['left']}:end_frame={plan['right']},setpts=PTS-STARTPTS,fps=30:start_time=0,scale=1280:720,setsar=1",
            "-start_number",
            "0",
            frames / "%06d.png",
        ],
        check,
        timeout=7200,
    )
    files = sorted(frames.glob("*.png"))
    expected = round((plan["right"] - plan["left"]) / 2)
    if not files or abs(len(files) - expected) > 1 or len(files) > 900:
        raise PipelineError("OUTPUT_INVALID", "原生块的模型帧数或上下文不一致")
    return files


def fine_masks(work, detections, assets, check=lambda: None):
    supports = Supports(
        assets["prior"], detections, np.load(assets["alpha"])["alpha"], work / "graphic"
    )
    dest = work / "masks"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir()
    for index, file in enumerate(sorted((work / "frames").glob("*.png"))):
        check()
        image = cv2.imread(str(file))
        mask = np.zeros(image.shape[:2], np.uint8)
        for part in supports.at(index, image):
            component = part["mask"]
            if part["opaque"]:
                component = cv2.dilate(component, np.ones((5, 5), np.uint8))
            mask |= component
        if np.count_nonzero(mask) > mask.size * 0.2:
            raise PipelineError(
                "MASK_UNCERTAIN", "稀疏联合支持超过原有百分之二十门槛", review=True
            )
        cv2.imwrite(str(dest / file.name), mask)


def process_chunk(
    source,
    work,
    plan,
    assets,
    config,
    source_hash,
    check=lambda: None,
    stage=lambda _: None,
):
    from scripts.hls_align_reference import align
    from scripts.hls_fit_projection import fit
    from scripts.hls_native_chunk import render

    work.mkdir(parents=True, exist_ok=True)
    cache = StageCache(work / "stages")
    settings = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(
            (
                "BOX_",
                "MASK_",
                "SHOT_",
                "DETECTION_",
                "YOLO_",
                "RESIDUAL_",
                "REFERENCE_",
                "RESTORE_",
                "SAM_",
                "PROPAINTER_",
            )
        )
    }
    import sys
    import importlib.metadata

    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    key = fingerprint(
        [],
        [
            source_hash,
            plan,
            config["mode"],
            os.getenv("MODEL_LOCK_HASH"),
            list(sys.version_info[:3]),
            cv2.__version__,
            np.__version__,
            torch_version,
            "stage-dependencies-2",
        ],
    )
    dependencies = {
        "decode": ([], ["process.py"]),
        "detect": (
            ["prior"],
            [
                "vision.py",
                "multitrack.py",
                "workflow.py",
                "yolo_stage.py",
                "hls_settings.py",
            ],
        ),
        "sam": (
            ["prior"],
            [
                "vision.py",
                "sam_stage.py",
                "sam_schedule.py",
                "workflow.py",
                "multitrack.py",
            ],
        ),
        "graphics": (
            [
                "prior",
                "palette",
                "color_seed_image",
                "color_seed_mask",
                "variant_seed_image",
                "variant_seed_mask",
            ],
            [
                "native_assets.py",
                "branding.py",
                "short_brand.py",
                "brand_templates.py",
                "brand_foreground.py",
                "graphic_tracks.py",
                "yolo_stage.py",
                "sam_stage.py",
                "sam_schedule.py",
                "workflow.py",
            ],
        ),
        "fine_masks": (
            ["prior", "alpha"],
            ["fine_masks.py", "matte.py", "multitrack.py"],
        ),
        "inpaint": (
            [],
            [
                "vision.py",
                "roi.py",
                "roi_inpaint.py",
                "restoration.py",
                "torch_stage.py",
                "propainter_offline.py",
                "speedup.py",
            ],
        ),
        "align": ([], ["scripts/hls_align_reference.py"]),
        "projection": (
            ["prior", "alpha", "template", "projection"],
            ["scripts/hls_fit_projection.py", "matte.py", "fine_masks.py"],
        ),
        "render": (
            ["prior", "alpha", "template"],
            [
                "scripts/hls_native_chunk.py",
                "scripts/hls_recompose_detail.py",
                "matte.py",
                "fine_masks.py",
                "texture_refs.py",
                "torch_composite.py",
                "speedup.py",
                "gpu_lock.py",
            ],
        ),
    }
    prefixes = {
        "decode": (),
        "detect": ("SHOT_", "DETECTION_", "YOLO_"),
        "sam": ("BOX_", "MASK_", "SAM_"),
        "graphics": ("BOX_", "MASK_", "SAM_", "YOLO_", "DETECTION_"),
        "fine_masks": ("BOX_", "MASK_"),
        "inpaint": ("PROPAINTER_", "REFERENCE_", "RESTORE_", "MASK_"),
        "align": (),
        "projection": ("REFERENCE_", "RESTORE_"),
        "render": ("REFERENCE_", "RESTORE_", "MASK_"),
    }

    def step(name, identity, operation, outputs, clean=lambda: None):
        check()
        asset_names, modules = dependencies[name]
        base = Path(__file__).parent
        code_paths = [
            Path(__file__),
            base / "speedup_cache.py",
            base / "process.py",
            base / "media.py",
        ] + [
            base.parent / module if module.startswith("scripts/") else base / module
            for module in modules
        ]
        paths = [assets[item] for item in asset_names if assets.get(item)] + code_paths
        identity = fingerprint(
            paths,
            [
                identity,
                {k: v for k, v in settings.items() if k.startswith(prefixes[name])},
            ],
        )
        stage(
            {
                "sam": "mask",
                "graphics": "detect",
                "fine_masks": "mask",
                "align": "encode",
                "projection": "encode",
                "render": "encode",
                "decode": "probe",
            }.get(name, name)
        )
        cache.run(name, identity, operation, outputs, clean)

    def remove(name):
        target = work / name
        if target.exists():
            shutil.rmtree(target)

    step(
        "decode",
        key,
        lambda: decode_chunk(source, work, plan, check),
        lambda: [work / "frames"],
    )
    files = sorted((work / "frames").glob("*.png"))
    detect_key = fingerprint([], [key, cache.tokens["decode"]])
    step(
        "detect",
        detect_key,
        lambda: vision.detect(
            files,
            vision.shots(files, check),
            dict(
                hls=True,
                multitrack_prior=str(assets["prior"]),
                detection_workers=4 if config["fast"] else 1,
            ),
            work,
            check,
        ),
        lambda: [work / "detections.json", work / "multitrack-prior.npz"],
    )
    detections = json.loads((work / "detections.json").read_text())
    tracking = work / "tracking"

    def segment():
        tracking.mkdir()
        (tracking / "frames").symlink_to(
            (work / "frames").resolve(), target_is_directory=True
        )
        (tracking / "detections.json").write_text(json.dumps(detections))
        vision.masks(
            tracking, files, detections, "propainter", check, speedup=config["fast"]
        )

    sam_key = fingerprint([], [detect_key, cache.tokens["detect"]])
    step(
        "sam",
        sam_key,
        segment,
        lambda: [
            tracking / "masks",
            tracking / "object-masks",
            tracking / "metrics.sam.json",
        ],
        lambda: remove("tracking"),
    )
    graphic_key = fingerprint([], [sam_key, cache.tokens["sam"]])
    step(
        "graphics",
        graphic_key,
        lambda: prepare_graphics(work, detections, assets, config["fast"], check),
        lambda: [work / "graphic"],
        lambda: remove("graphic"),
    )
    mask_key = fingerprint([], [graphic_key, cache.tokens["graphics"]])
    step(
        "fine_masks",
        mask_key,
        lambda: fine_masks(work, detections, assets, check),
        lambda: [work / "masks"],
    )
    inpaint_key = fingerprint([], [mask_key, cache.tokens["fine_masks"]])
    step(
        "inpaint",
        inpaint_key,
        lambda: vision.inpaint(
            work,
            files,
            detections["scenes"],
            "propainter",
            True,
            check,
            roi=config["fast"],
        ),
        lambda: [work / "composite", work / "performance.json"],
        (lambda: None) if config["fast"] else lambda: remove("composite"),
    )
    reference = work / "native-reference"
    step(
        "align",
        detect_key,
        lambda: align(
            source,
            work / "frames",
            reference,
            source_start_frame=plan["left"],
            check=check,
            source_end_frame=plan["right"],
        ),
        lambda: [reference],
        lambda: remove("native-reference"),
    )
    projection_key = fingerprint(
        [], [inpaint_key, cache.tokens["inpaint"], cache.tokens["align"]]
    )
    step(
        "projection",
        projection_key,
        lambda: shutil.copyfile(assets["projection"], work / "projection.json")
        if assets.get("projection")
        else fit(
            work,
            reference,
            work / "detections.json",
            assets["prior"],
            assets["alpha"],
            assets["template"],
            work / "graphic",
            work / "projection.json",
            1.2,
        ),
        lambda: [work / "projection.json"],
    )
    rendered = work / "rendered"
    render_key = fingerprint(
        [], [projection_key, cache.tokens["projection"], config["backend"]]
    )

    def render_native():
        from contextlib import nullcontext
        from .gpu_lock import device_lock

        lock = (
            device_lock(check=check)
            if config["backend"] == "torch-cuda"
            else nullcontext()
        )
        with lock:
            render(
                source,
                work,
                work / "detections.json",
                assets["prior"],
                assets["alpha"],
                assets["template"],
                work / "graphic",
                rendered,
                (plan["right"] - plan["left"]) / 60,
                reference,
                work / "projection.json",
                60,
                1.2,
                10,
                config["backend"],
                True,
                "libx264",
                frames_only=True,
                source_start_frame=plan["left"],
                check=check,
            )

    step(
        "render",
        render_key,
        render_native,
        lambda: [rendered / "frames", rendered / "native-report.json"],
        lambda: remove("rendered"),
    )
    summary = dict(
        plan=plan,
        mode=config["mode"],
        stages=cache.timings,
        native=json.loads((rendered / "native-report.json").read_text()),
        performance=json.loads((work / "performance.json").read_text()),
    )
    (work / "chunk.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def audit_chunk(source, result, work, plan, assets, check=lambda: None, enforce=True):
    """在最终编码后的核心帧上重做残留、品牌部件和全部相邻帧清晰度核验。"""
    from functools import lru_cache
    from scripts.hls_native_verify import remap_references
    from scripts.hls_brand_verify import clipped_scores
    from .brand_templates import variants
    from .multitrack import highpass
    from .hls_settings import is_residual_score
    from scripts.hls_detail_verify import occluder_mask
    from .matte import temporal_detail, stable_detail_mask

    native = json.loads((work / "rendered/native-report.json").read_text())
    lo, hi = plan["start"] - plan["left"], plan["end"] - plan["left"]
    mapping = native["帧映射"][lo:hi]
    detections = json.loads((work / "detections.json").read_text())
    detections["prior"] = str(assets["prior"])
    refs = remap_references(detections, mapping)
    # 同时恢复场景、总提示和所有轨迹，不能只映射部分提示。
    refs["hints"] = sorted(
        [p for t in refs["tracks"] for p in t["hints"]],
        key=lambda p: (p["frame"], p["track_id"]),
    )
    refs["scenes"] = [[0, len(mapping)]]
    audit = work / "audit"
    if audit.exists():
        shutil.rmtree(audit)
    (audit / "composite").mkdir(parents=True)
    (audit / "detections.json").write_text(json.dumps(refs))
    supports = Supports(
        assets["prior"], detections, np.load(assets["alpha"])["alpha"], work / "graphic"
    )
    brands = json.loads((work / "graphic/brands.json").read_text())
    banks = [
        variants(
            cv2.imread(str(work / "graphic/seeds" / f"{b['id']}-image.png")),
            cv2.imread(str(work / "graphic/seeds" / f"{b['id']}-mask.png"), 0),
            [0.95, 1, 1.05],
        )
        for b in brands
    ]

    @lru_cache(maxsize=8)
    def model_source(index):
        return cv2.imread(str(work / "frames" / f"{index:06d}.png"))

    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(result))
    a.set(cv2.CAP_PROP_POS_FRAMES, plan["start"])
    b.set(cv2.CAP_PROP_POS_FRAMES, plan["start"])
    samples, evidence, flicker = [], [], []
    previous = None
    try:
        for n, model_index in enumerate(mapping):
            check()
            ok, original = a.read()
            good, final = b.read()
            if not ok or not good:
                raise PipelineError("OUTPUT_INVALID", "成品核验核心帧不足")
            small = cv2.resize(final, (1280, 720))
            cv2.imwrite(str(audit / "composite" / f"{n:06d}.png"), small)
            feature = highpass(small)
            for brand, bank in zip(brands, banks):
                for point in brand["matches"][model_index]:
                    variant = bank[point["variant"]]
                    scores = clipped_scores(
                        feature, variant["template"], variant["mask"], point["box"]
                    )
                    bad = [p for p in scores if is_residual_score(p)]
                    if bad:
                        evidence.append(
                            dict(
                                frame=plan["start"] + n,
                                model_frame=model_index,
                                brand=brand["id"],
                                components=bad,
                            )
                        )
            graphic_mask = cv2.imread(
                str(work / "graphic/masks" / f"{model_index:06d}.png"), 0
            )
            if graphic_mask is not None and graphic_mask.any():
                from .multitrack import component_scores

                palette = np.load(assets["palette"])
                source_small = model_source(model_index)
                from .brand_foreground import foreground as source_foreground

                try:
                    foreground, _ = source_foreground(
                        source_small, graphic_mask, palette
                    )
                except PipelineError as error:
                    if error.code != "MASK_UNCERTAIN":
                        raise
                    foreground = np.zeros_like(graphic_mask)
                scores = component_scores(highpass(source_small), feature, foreground)
                bad = [p for p in scores if is_residual_score(p)]
                if bad or not scores:
                    evidence.append(
                        dict(
                            frame=plan["start"] + n,
                            model_frame=model_index,
                            kind="完整图形前景",
                            components=bad,
                            support_insufficient=not scores,
                        )
                    )
            faint = {}
            other = occluder_mask([work / "graphic"], model_index, original.shape[:2])
            occupied = other.copy()
            for part in supports.at(model_index, model_source(model_index)):
                mask = cv2.resize(
                    part["mask"],
                    (original.shape[1], original.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                occupied |= mask
                if part["kind"] == "fixed" and not part["opaque"]:
                    faint[part["id"]] = mask
                else:
                    other |= mask
            if previous is not None:
                old_x, old_y, old_faint, old_other, old_occupied = previous
                for identity, mask in faint.items():
                    if identity in old_faint:
                        stable = stable_detail_mask(
                            mask & old_faint[identity], other, old_other
                        )
                        metric = temporal_detail(
                            old_x,
                            original,
                            old_y,
                            final,
                            stable,
                            occupied | old_occupied,
                        )
                        samples.append(
                            dict(
                                frame=plan["start"] + n,
                                time=(plan["start"] + n) / 60,
                                track_id=identity,
                                **metric,
                            )
                        )
                repair_delta = (final.astype(np.float32) - original) - (
                    old_y.astype(np.float32) - old_x
                )
                active = (occupied | old_occupied) > 0
                flicker.append(
                    dict(
                        frame=plan["start"] + n,
                        source_mae=float(np.abs(original.astype(float) - old_x).mean()),
                        repair_delta_mae=float(np.abs(repair_delta)[active].mean())
                        if active.any()
                        else 0.0,
                    )
                )
            previous = original, final, faint, other, occupied
    finally:
        a.release()
        b.release()
    # 先保留全部原始证据，再执行硬门槛；中断不丢失复核材料。
    summary = dict(
        清晰度统计=samples,
        低于零点七的样本=[
            s for s in samples if s["measurable"] and s["retention_ratio"] < 0.7
        ],
        品牌残留候选=evidence,
        闪烁指标=flicker,
        闪烁目检="未进行",
        通过=None,
    )
    (audit / "native-quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    residual_error = None
    try:
        vision.residual_quality(
            audit, sorted((audit / "composite").glob("*.png")), refs, check
        )
    except PipelineError as error:
        if error.code != "RESIDUAL_UNCERTAIN":
            raise
        residual_error = error
    finally:
        if (audit / "quality.json").exists():
            summary["原始残留检查"] = json.loads((audit / "quality.json").read_text())
            (audit / "native-quality.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2)
            )
    from .native_assets import residual_candidates

    raw = residual_candidates(audit, work, assets, check)
    summary["全画面模型复检"] = raw
    (audit / "native-quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    low = None
    if assets.get("low_band_template"):
        from .hls_graphic_repair import verify_low
        from .multitrack import load_prior

        # 成品核验独立运行，沿用主流程的低频填色复检，不能只检查高频笔画。
        low = verify_low(
            result,
            np.load(assets["low_band_template"])["template"],
            load_prior(assets["prior"])[2],
            start_frame=plan["start"],
            frame_count=len(mapping),
            check=check,
        )
    summary["完整图形低频复检"] = low
    low_bad = low and low["残留区域数"] > 0
    summary["硬门槛通过"] = not (
        residual_error
        or raw["残留候选"]
        or raw.get("需要复核", False)
        or evidence
        or low_bad
    )
    (audit / "native-quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    if enforce and residual_error:
        raise residual_error
    if enforce and raw.get("需要复核"):
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "原始图标类文字候选须做字幕复核", review=True
        )
    if enforce and raw["残留候选"]:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "全画面模型复检存在原始残留候选", review=True
        )
    if enforce and low_bad:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "完整图形低频复检存在残留候选", review=True
        )
    if enforce and evidence:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "完整品牌部件复检存在残留候选", review=True
        )
    return summary
