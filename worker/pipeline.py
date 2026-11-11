import json
import os
import shutil
import time
from pathlib import Path
from . import media, vision
from .process import PipelineError


def execute(
    source,
    root,
    options,
    engine="propainter",
    low_memory=False,
    stage=lambda _: None,
    check=lambda: None,
):
    root = Path(root).resolve()
    source = Path(source).resolve()
    timings = {}

    def step(name, operation):
        check()
        if name != "quality" or options.get("hls"):
            stage(name)
        start = time.monotonic()
        try:
            return operation()
        finally:
            timings[name] = timings.get(name, 0) + round(
                (time.monotonic() - start) * 1000
            )
            (root / "phase_ms.json").write_text(json.dumps(timings))

    short = options.get("max_short_edge", 720)
    if low_memory and options.get("allow_downgrade"):
        short = min(short, 540)
    normalizer = (
        media.normalize_direct if options.get("speedup_v2") else media.normalize
    )
    info = step("probe", lambda: normalizer(source, root, short, check))
    source_w, source_h = info["source_width"], info["source_height"]
    baseline_scale = min(
        1,
        1280 / max(source_w, source_h),
        options.get("max_short_edge", 720) / min(source_w, source_h),
    )
    baseline_w = max(2, int(source_w * baseline_scale) // 2 * 2)
    baseline_h = max(2, int(source_h * baseline_scale) // 2 * 2)
    info["downgraded"] = info["width"] < baseline_w or info["height"] < baseline_h
    files = sorted((root / "frames").glob("*.png"))
    detections = step(
        "detect",
        lambda: vision.detect(files, vision.shots(files, check), options, root, check),
    )
    mask_options = {"speedup": True} if options.get("speedup_v2") else {}
    inpaint_options = {"roi": True} if options.get("speedup_v2") else {}
    try:
        manifest = step(
            "mask",
            lambda: vision.masks(
                root, files, detections, engine, check, **mask_options
            ),
        )
    except PipelineError as exc:
        if exc.code not in ("TRACKING_DRIFT", "MASK_UNCERTAIN") or not (
            os.getenv("TEMPLATE_PATH") or os.getenv("DETECTOR") == "yolo"
        ):
            raise
        # A fresh SAM subprocess reinitializes tracking with dense detections.
        shutil.rmtree(root / "masks")
        detections = step(
            "detect",
            lambda: vision.detect(
                files, detections["scenes"], options, root, check, dense=True
            ),
        )
        manifest = step(
            "mask",
            lambda: vision.masks(
                root, files, detections, engine, check, **mask_options
            ),
        )
    for frame in manifest:
        frame["pts_ms"] = round(frame["frame_index"] * 1000 / info["fps"], 3)
    metadata = {
        "pipeline_version": os.environ.get("PIPELINE_VERSION", "mvp-1"),
        "engine": engine,
        "model_lock_hash": os.getenv("MODEL_LOCK_HASH", "unrecorded"),
        "image_digest": os.getenv("IMAGE_DIGEST", "unrecorded"),
        "media": info,
        "frames": manifest,
    }
    (root / "manifest.json").write_text(json.dumps(metadata, indent=2))
    step(
        "inpaint",
        lambda: vision.inpaint(
            root,
            files,
            detections["scenes"],
            engine,
            low_memory,
            check,
            **inpaint_options,
        ),
    )
    dest = step("encode", lambda: media.encode(root, info, check))
    if os.environ.get("DETECTOR") == "yolo":
        step("quality", lambda: vision.residual_quality(root, files, detections, check))
    return (
        dest,
        {
            k: info[k]
            for k in (
                "width",
                "height",
                "duration_ms",
                "frame_count",
                "fps",
                "downgraded",
            )
        },
        timings,
    )
