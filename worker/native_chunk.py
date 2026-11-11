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
from .roi_inpaint import digest
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
    code = [
        digest(Path(__file__).with_name(n))
        for n in (
            "native_chunk.py",
            "native_assets.py",
            "vision.py",
            "sam_stage.py",
            "sam_schedule.py",
            "roi.py",
            "roi_inpaint.py",
            "torch_composite.py",
            "matte.py",
            "texture_refs.py",
        )
    ]
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
    key = fingerprint(
        list(assets.values()),
        [
            source_hash,
            plan,
            config["mode"],
            os.getenv("MODEL_LOCK_HASH"),
            code,
            settings,
        ],
    )

    def step(name, identity, operation, outputs, clean=lambda: None):
        check()
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
            dict(hls=True, multitrack_prior=str(assets["prior"])),
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
    step(
        "render",
        render_key,
        lambda: render(
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
        ),
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


def audit_chunk(source, result, work, plan, assets, check=lambda: None):
    """在最终编码后的核心帧上重做残留、品牌部件和全部相邻帧清晰度核验。"""
    from functools import lru_cache
    from scripts.hls_native_verify import remap_references
    from scripts.hls_brand_verify import clipped_scores
    from .brand_templates import variants
    from .multitrack import highpass
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
                    bad = [
                        p
                        for p in scores
                        if abs(p["correlation"]) >= 0.55
                        and abs(p["relative_amplitude"]) >= 0.15
                    ]
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
                from .branding import palette_support
                from .multitrack import component_scores

                palette = np.load(assets["palette"])
                source_small = model_source(model_index)
                foreground = (
                    graphic_mask & palette_support(source_small, palette)
                    if len(palette)
                    else graphic_mask
                )
                scores = component_scores(highpass(source_small), feature, foreground)
                bad = [
                    p
                    for p in scores
                    if abs(p["correlation"]) >= 0.55
                    and abs(p["relative_amplitude"]) >= 0.15
                ]
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
            faint, other = {}, np.zeros(original.shape[:2], np.uint8)
            occupied = np.zeros_like(other)
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
    try:
        vision.residual_quality(
            audit, sorted((audit / "composite").glob("*.png")), refs, check
        )
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
    if raw["残留候选"]:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "全画面模型复检存在原始残留候选", review=True
        )
    if evidence:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "完整品牌部件复检存在残留候选", review=True
        )
    return summary
