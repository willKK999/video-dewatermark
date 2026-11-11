"""局部补画执行器：单卡串行推理，有界准备与合成，逐窗口校验检查点。"""

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
from .restoration import CleanReferences
from .roi import plan_rois, paste_roi
from .speedup import bounded_map


def digest(path):
    path = Path(path)
    if path.resolve().name == "key.bin" or path.resolve().name.endswith(".env"):
        raise ValueError("禁止读取密钥或环境配置作为检查点内容")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def checkpoint_valid(folder, identity):
    marker = folder / "complete.json"
    if not marker.exists():
        return False
    try:
        saved = json.loads(marker.read_text())
        return (
            saved["identity"] == identity
            and bool(saved["outputs"])
            and all(
                (folder / name).is_file() and digest(folder / name) == value
                for name, value in saved["outputs"].items()
            )
        )
    except (ValueError, KeyError, OSError):
        return False


def inpaint_roi(root, files, scenes, low_memory=False, check=lambda: None):
    from .vision import read, windows, _composite_window

    dest = root / "composite"
    dest.mkdir(exist_ok=True)
    cache = root / "roi-cache"
    cache.mkdir(exist_ok=True)
    margin = int(os.environ.get("PROPAINTER_ROI_MARGIN", "96"))
    # 保留七版窗口和重叠；不根据区域缩小扩大时间窗口或同时运行模型。
    size, overlap = 36, 8
    performance = dict(
        propainter_seconds=0.0,
        restoration_seconds=0.0,
        composite_io_seconds=0.0,
        window_frames=size,
        overlap_frames=overlap,
        roi_windows=[],
        cached_windows=0,
    )
    upstream = Path(os.environ["PROPAINTER_ROOT"]) / "inference_propainter.py"
    parameters = dict(
        upstream=digest(upstream) if upstream.is_file() else None,
        margin=margin,
        size=size,
        overlap=overlap,
        fp16=os.getenv("PROPAINTER_FP16"),
        model_lock=os.getenv("MODEL_LOCK_HASH"),
        model_root=os.getenv("PROPAINTER_ROOT"),
        python=os.getenv("INPAINT_PYTHON"),
        code=[
            digest(Path(__file__).with_name(name))
            for name in (
                "roi.py",
                "roi_inpaint.py",
                "torch_stage.py",
                "propainter_offline.py",
            )
        ],
    )
    # 一次散列；重叠窗口不重复读入全部视频内容。
    signatures = {}
    for a, b in scenes:
        for j in range(a, b):
            check()
            if j not in signatures:
                signatures[j] = [
                    digest(files[j]),
                    digest(root / "masks" / files[j].name),
                ]
            (dest / files[j].name).unlink(missing_ok=True)

    def prepare(plan):
        start, end = plan
        check()
        masks = [read(root / "masks" / files[j].name, True) for j in range(start, end)]
        boxes = plan_rois(masks, margin=margin)
        identity = hashlib.sha256(
            json.dumps(
                [
                    parameters,
                    start,
                    end,
                    boxes,
                    [signatures[j] for j in range(start, end)],
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        folder = cache / identity
        if checkpoint_valid(folder, identity):
            return start, end, folder, boxes, identity, True
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        for k, box in enumerate(boxes):
            x, y, a, b = box
            task = folder / f"roi-{k}"
            (task / "frames").mkdir(parents=True)
            (task / "masks").mkdir()
            for n, j in enumerate(range(start, end)):
                check()
                image = read(files[j])
                crop, mask = image[y:b, x:a], masks[n][y:b, x:a]
                # 仅填充到八的倍数，绝不缩放原像素。
                dh, dw = (-crop.shape[0]) % 8, (-crop.shape[1]) % 8
                cv2.imwrite(
                    str(task / "frames" / f"{n:06d}.png"),
                    cv2.copyMakeBorder(crop, 0, dh, 0, dw, cv2.BORDER_REFLECT_101),
                )
                cv2.imwrite(
                    str(task / "masks" / f"{n:06d}.png"),
                    cv2.copyMakeBorder(mask, 0, dh, 0, dw, cv2.BORDER_CONSTANT),
                )
            if end - start == 1:
                for name in ("frames", "masks"):
                    os.link(task / name / "000000.png", task / name / "000001.png")
        return start, end, folder, boxes, identity, False

    with ThreadPoolExecutor(max_workers=1) as compositor:
        pending = None
        for scene_start, scene_end in scenes:
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
            for start, end, folder, boxes, identity, reused in bounded_map(
                prepare, windows(scene_start, scene_end, size, overlap), depth=1
            ):
                check()
                if reused:
                    performance["cached_windows"] += 1
                else:
                    started = time.perf_counter()
                    for k, box in enumerate(boxes):
                        task = folder / f"roi-{k}"
                        args = [
                            os.environ["INPAINT_PYTHON"],
                            Path(__file__).with_name("torch_stage.py"),
                            "--metrics",
                            task / "metrics.json",
                            "--frames-only",
                            "--script",
                            Path(os.environ["PROPAINTER_ROOT"])
                            / "inference_propainter.py",
                            "--video",
                            task / "frames",
                            "--mask",
                            task / "masks",
                            "--output",
                            task / "out",
                            "--save_frames",
                            "--mask_dilation",
                            "0",
                            "--subvideo_length",
                            str(size),
                            "--neighbor_length",
                            "6",
                            "--ref_stride",
                            "20",
                        ]
                        if os.getenv("PROPAINTER_FP16") == "true":
                            args.append("--fp16")
                        run(args, check, cwd=os.environ["PROPAINTER_ROOT"])
                    performance["propainter_seconds"] += time.perf_counter() - started
                    predicted = [
                        sorted(
                            (folder / f"roi-{k}" / "out/frames/frames").glob("*.png")
                        )
                        for k in range(len(boxes))
                    ]
                    expected = max(2, end - start)
                    if any(len(paths) != expected for paths in predicted):
                        raise PipelineError("OUTPUT_INVALID", "局部补画未返回全部帧")
                    output = folder / "out/frames/frames"
                    output.mkdir(parents=True)
                    hashes = {}
                    for n, j in enumerate(range(start, end)):
                        image = read(files[j])
                        mask = read(root / "masks" / files[j].name, True)
                        for k, box in enumerate(boxes):
                            crop = read(predicted[k][n])
                            x, y, a, b = box
                            dh, dw = (-(b - y)) % 8, (-(a - x)) % 8
                            if crop.shape != (b - y + dh, a - x + dw, 3):
                                raise PipelineError(
                                    "OUTPUT_INVALID", "局部补画输出尺寸不一致"
                                )
                            paste_roi(image, crop[: b - y, : a - x], mask, box)
                        path = output / f"{n:06d}.png"
                        cv2.imwrite(str(path), image)
                        hashes[str(path.relative_to(folder))] = digest(path)
                    marker = folder / "complete.tmp"
                    marker.write_text(
                        json.dumps(dict(identity=identity, outputs=hashes))
                    )
                    marker.replace(folder / "complete.json")
                    for k in range(len(boxes)):
                        # 保留模型指标，删除可重建的裁剪帧以限制磁盘使用。
                        task = folder / f"roi-{k}"
                        for name in ("frames", "masks", "out"):
                            shutil.rmtree(task / name)
                window_metrics = []
                for metrics in sorted(folder.glob("roi-*/metrics.json")):
                    window_metrics.append(json.loads(metrics.read_text()))
                shape = read(files[start]).shape
                performance["roi_windows"].append(
                    dict(
                        start=start,
                        end=end,
                        boxes=boxes,
                        reused=reused,
                        area_ratio=sum((a - x) * (b - y) for x, y, a, b in boxes)
                        / (shape[0] * shape[1]),
                        gpu_metrics=window_metrics,
                    )
                )
                if pending:
                    pending.result()
                pending = compositor.submit(
                    _composite_window,
                    root,
                    files,
                    start,
                    end,
                    overlap,
                    folder,
                    check,
                    references,
                    performance,
                    False,
                )
            if pending:
                pending.result()
            (root / "performance.json").write_text(json.dumps(performance))
    return performance
