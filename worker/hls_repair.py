"""视频去水印系统：以自动残留证据触发带上下文的局部复修。"""

import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from . import vision
from .process import PipelineError, run
from .workflow import expand_box


def intervals(indices, scenes, count, radius=30):
    ranges = []
    for index in sorted(set(indices)):
        if not 0 <= index < count:
            raise PipelineError("OUTPUT_INVALID", "残留帧号超出处理块")
        start, end = next((s, e) for s, e in scenes if s <= index < e)
        left, right = max(start, index - radius), min(end, index + radius + 1)
        if ranges and left <= ranges[-1][1] and ranges[-1][0] >= start:
            ranges[-1] = (ranges[-1][0], max(right, ranges[-1][1]))
        else:
            ranges.append((left, right))
    return ranges


def component_support(mask, watermark_box, residual_box):
    """允许残留框超出原始漏检框，同时限制 SAM 传播泄漏。"""
    height, width = mask.shape
    envelope = [
        min(watermark_box[0], residual_box[0]),
        min(watermark_box[1], residual_box[1]),
        max(watermark_box[2], residual_box[2]),
        max(watermark_box[3], residual_box[3]),
    ]
    a, b, c, d = expand_box(envelope, width, height)
    limit = np.zeros_like(mask)
    limit[int(b) : int(np.ceil(d)), int(a) : int(np.ceil(c))] = 255
    clipped = cv2.bitwise_and(mask, limit)
    if not clipped.any():
        return clipped
    y, x = np.where(clipped > 0)
    box = [
        max(int(a), int(min(x.min(), residual_box[0]))),
        max(int(b), int(min(y.min(), residual_box[1]))),
        min(int(np.ceil(c)), int(np.ceil(max(x.max() + 1, residual_box[2])))),
        min(int(np.ceil(d)), int(np.ceil(max(y.max() + 1, residual_box[3])))),
    ]
    support = np.zeros_like(mask)
    support[box[1] : box[3], box[0] : box[2]] = 255
    return support


def reseed_masks(work, root, files, detections, candidates, affected, ranges, check):
    from .yolo_stage import reference_box

    target = work / "sam-reseed"
    target.mkdir()
    (target / "frames").symlink_to(
        (root / "frames").resolve(), target_is_directory=True
    )
    (target / "masks").mkdir()
    hints = [
        dict(p)
        for p in candidates
        if any(start <= p["frame"] < end for start, end in ranges)
    ]
    if detections.get("hls"):
        # 同一镜头以最清晰的残留部件提示，避免模糊碎片提示抢占身份。
        hints = [
            max(
                (p for p in hints if start <= p["frame"] < end),
                key=lambda p: p["confidence"],
            )
            for start, end in ranges
        ]
    (target / "detections.json").write_text(
        json.dumps(dict(scenes=ranges, hints=hints, dense=True))
    )
    run(
        [
            os.environ["TRACK_PYTHON"],
            str(Path(__file__).with_name("sam_stage.py")),
            "--root",
            target,
        ],
        check,
    )
    result = {}
    targets = {p["frame"] for p in candidates}
    for frame in sorted(affected):
        mask = cv2.imread(str(target / "masks" / files[frame].name), 0)
        if mask is None:
            raise PipelineError("MASK_INVALID", "局部上下文分割帧缺失", review=True)
        if frame in targets and not mask.any():
            raise PipelineError("TRACKING_DRIFT", "残留部件重新分割失败", review=True)
        if mask.any():
            # SAM 已确认部件；自动残留框补足半透明细轮廓，整段上下文同时遮住该部件。
            scene = next(
                (start, end)
                for start, end in detections["scenes"]
                if start <= frame < end
            )
            part = min(
                (p for p in hints if scene[0] <= p["frame"] < scene[1]),
                key=lambda p: abs(p["frame"] - frame),
            )
            before = reference_box(detections, part["frame"])
            now = reference_box(detections, frame)
            dx = (now[0] + now[2] - before[0] - before[2]) / 2
            dy = (now[1] + now[3] - before[1] - before[3]) / 2
            shifted = [
                part["box"][0] + dx,
                part["box"][1] + dy,
                part["box"][2] + dx,
                part["box"][3] + dy,
            ]
            result[frame] = component_support(mask, now, shifted)
        else:
            result[frame] = mask
    return result


def refine(root, files, detections, engine, low_memory, fps, stage, check, rounds=2):
    """只使用模型生成的掩膜，不以人工框或整幅画面覆盖作为回退。"""
    records = []
    for iteration in range(rounds):
        quality = json.loads((root / "quality.json").read_text())
        if quality["residual_count"] == 0 and not quality["needs_review"]:
            break
        candidates = json.loads((root / "quality-detections.json").read_text())
        if not isinstance(candidates, list) or not candidates:
            raise PipelineError(
                "RESIDUAL_UNCERTAIN", "残留轨迹存在歧义，无法安全自动复修", review=True
            )
        from .yolo_stage import reference_box, residual_match

        hls = detections.get("hls", False)

        targets = sorted(
            {
                p["frame"]
                for p in candidates
                if residual_match(
                    p["box"],
                    reference_box(detections, p["frame"]),
                    1.5 if hls else 0,
                )
            }
        )
        if not targets:
            raise PipelineError(
                "RESIDUAL_UNCERTAIN", "没有可安全复修的残留轨迹", review=True
            )
        ranges = intervals(
            targets, detections["scenes"], len(files), max(1, round(fps))
        )
        if hls:
            # 片段化轮廓可能低于检测阈值；同镜头传播部件，覆盖整个镜头的漏修。
            ranges = sorted(
                {
                    (start, end)
                    for start, end in detections["scenes"]
                    if any(start <= i < end for i in targets)
                }
            )
        affected = {i for start, end in ranges for i in range(start, end)}
        work = root / f"refine-{iteration}"
        work.mkdir()
        masks = work / "masks"
        masks.mkdir()
        (work / "frames").symlink_to(
            (root / "composite").resolve(), target_is_directory=True
        )
        current = [work / "frames" / p.name for p in files]
        started = time.monotonic()
        stage("mask")
        resegmented = (
            reseed_masks(
                work, root, files, detections, candidates, affected, ranges, check
            )
            if engine == "propainter"
            else {}
        )
        for i in {i for start, end in ranges for i in range(start, end)}:
            check()
            original = cv2.imread(str(root / "masks" / files[i].name), 0)
            mask = np.zeros_like(original)
            if i in affected:
                radius = 3 + iteration * 3
                if engine == "propainter":
                    original = resegmented[i]
                mask = cv2.dilate(
                    original,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
                    ),
                )
            if np.count_nonzero(mask) / mask.size > 0.2:
                raise PipelineError(
                    "MASK_UNCERTAIN", "局部复修掩膜超过画面 20%", review=True
                )
            cv2.imwrite(str(masks / files[i].name), mask)
        stage("inpaint")
        vision.inpaint(work, current, ranges, engine, low_memory, check)
        for path in (work / "composite").glob("*.png"):
            check()
            distance = min(abs(int(path.stem) - index) for index in targets)
            radius = max(4, round(fps))
            weight = (
                1.0 if distance <= 3 else max(0.0, (radius - distance) / (radius - 3))
            )
            if hls:
                weight = 1.0
            target = root / "composite" / path.name
            if weight >= 1:
                shutil.copyfile(path, target)
            elif weight > 0:
                previous = cv2.imread(str(target))
                updated = cv2.imread(str(path))
                cv2.imwrite(
                    str(target),
                    cv2.addWeighted(previous, 1 - weight, updated, weight, 0),
                )
        shutil.rmtree(work)
        stage("quality")
        try:
            vision.residual_quality(root, files, detections, check)
        except PipelineError as exc:
            if exc.code != "RESIDUAL_UNCERTAIN":
                raise
        after = json.loads((root / "quality.json").read_text())
        records.append(
            dict(
                round=iteration + 1,
                residual_before=quality["residual_count"],
                residual_after=after["residual_count"],
                target_frames=len(targets),
                context_ranges=ranges,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
        )
    (root / "hls-refinements.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )
    final = json.loads((root / "quality.json").read_text())
    if final["residual_count"] or final["needs_review"]:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "带上下文的局部复修后仍存在残留证据", review=True
        )
    return records
