"""Only dedicated watermark weights; generic COCO models are not acceptable."""

import argparse
import json
import os
from pathlib import Path
import re
import time


def watermark_class(name):
    name = str(name).strip().lower()
    return (
        "watermark" in name
        or "水印" in name
        or "logo" in re.split(r"[^a-z0-9]+", name)
        or name in {"corner_logo", "float_logo", "float_text"}
    )


def dedicated_classes(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    items = list(items)
    # Mixed generic object models are not dedicated watermark detectors.
    if not items or not all(watermark_class(name) for _, name in items):
        raise ValueError("dedicated watermark classes required")
    return {int(index) for index, _ in items}


def overlaps(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return smaller > 0 and intersection >= 0.1 * smaller


def compatible_parts(a, b):
    ah, bh = a[3] - a[1], b[3] - b[1]
    return (
        min(ah, bh) > 0
        and max(ah, bh) / min(ah, bh) <= 2
        and abs((a[1] + a[3]) - (b[1] + b[3])) / 2 <= min(ah, bh) * 0.6
    )


def compatible_scale(a, b):
    aa, ba = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return min(aa, ba) > 0 and max(aa, ba) / min(aa, ba) <= 2.5


def merge_boxes(
    boxes, classes, frame, previous_box=None, continuous=False, anchor_confidence=0
):
    relevant = [b for b in boxes if int(b.cls[0]) in classes]
    if not relevant:
        return None
    candidates = [(b.xyxy[0].cpu().tolist(), float(b.conf[0])) for b in relevant]
    anchors = [item for item in candidates if item[1] >= anchor_confidence]
    if not anchors:
        return None
    # Prefer temporal continuity to a distant subtitle with higher confidence.
    nearby = [
        item
        for item in anchors
        if previous_box is not None
        and overlaps(item[0], previous_box)
        and compatible_scale(item[0], previous_box)
    ]
    if continuous and previous_box is not None and not nearby:
        # 短暂漏检保留轨迹身份，交给前后可靠观测插值；不跳到远处字幕。
        return None
    anchor = max(nearby or anchors, key=lambda item: item[1])

    def same_component(b):
        if overlaps(b, anchor[0]):
            return True
        if not continuous or previous_box is None:
            return False
        a = anchor[0]
        gap = max(0, max(a[0], b[0]) - min(a[2], b[2]))
        union = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        return gap <= min(a[3] - a[1], b[3] - b[1]) * 0.5 and compatible_scale(
            union, previous_box
        )

    selected = [
        item
        for item in candidates
        if same_component(item[0]) and compatible_parts(item[0], anchor[0])
    ]
    if previous_box is not None:
        area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        limit = max(area(previous_box), area(anchor[0])) * 2.5
        selected = [item for item in selected if area(item[0]) <= limit]
    if continuous:
        area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        bounded = [anchor]
        union = list(anchor[0])
        for item in sorted(selected, key=lambda p: p[1], reverse=True):
            if item is anchor:
                continue
            if item[1] < anchor_confidence and area(item[0]) > area(anchor[0]):
                continue
            b = item[0]
            combined = [
                min(union[0], b[0]),
                min(union[1], b[1]),
                max(union[2], b[2]),
                max(union[3], b[3]),
            ]
            if area(combined) <= area(anchor[0]) * 3:
                bounded.append(item)
                union = combined
        selected = bounded
    coords = [item[0] for item in selected]
    return dict(
        frame=frame,
        box=[
            min(b[0] for b in coords),
            min(b[1] for b in coords),
            max(b[2] for b in coords),
            max(b[3] for b in coords),
        ],
        confidence=max(item[1] for item in selected),
    )


def reference_box(reference, index):
    start, end = next((s, e) for s, e in reference["scenes"] if s <= index < e)
    hints = [p for p in reference["hints"] if start <= p["frame"] < end]
    left = max(
        (p for p in hints if p["frame"] <= index),
        key=lambda p: p["frame"],
        default=hints[0],
    )
    right = min(
        (p for p in hints if p["frame"] >= index),
        key=lambda p: p["frame"],
        default=hints[-1],
    )
    fraction = (index - left["frame"]) / max(1, right["frame"] - left["frame"])
    return [a + (b - a) * fraction for a, b in zip(left["box"], right["box"])]


def reference_match(box, expected):
    area = (box[2] - box[0]) * (box[3] - box[1])
    target_area = (expected[2] - expected[0]) * (expected[3] - expected[1])
    intersection = max(0, min(box[2], expected[2]) - max(box[0], expected[0])) * max(
        0, min(box[3], expected[3]) - max(box[1], expected[1])
    )
    return area > 0 and intersection / area >= 0.5 and area <= target_area * 2.5


def reference_region(box, margin=0):
    """复检包含邻接图标，避免原检测只覆盖文字时把真实残留过滤掉。"""
    pad = (box[3] - box[1]) * margin
    return [box[0] - pad, box[1] - pad / 5, box[2] + pad, box[3] + pad / 5]


def residual_match(box, expected, margin=0):
    if margin and box[3] - box[1] > 2 * (expected[3] - expected[1]):
        return False
    return reference_match(box, reference_region(expected, margin))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--indices", required=True)
    p.add_argument(
        "--confidence", type=float, default=float(os.getenv("YOLO_CONFIDENCE", "0.35"))
    )
    p.add_argument("--reference")
    p.add_argument("--seeds")
    p.add_argument("--continuous-track", action="store_true")
    p.add_argument("--reference-margin", type=float, default=0)
    p.add_argument(
        "--all-candidates",
        action="store_true",
        help="保留专用模型的全部原始候选，供完整品牌核验",
    )
    args = p.parse_args()
    if not 0 < args.confidence <= 1:
        p.error("confidence must be in (0, 1]")
    # Require an existing local artifact; never implicitly download generic weights.
    if not Path(args.weights).is_file():
        raise ValueError("local dedicated watermark weights required")
    started = time.perf_counter()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    classes = dedicated_classes(model.names)
    results = []
    files = sorted(Path(args.frames).glob("*.png"))
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    seeds = (
        {p["frame"]: p for p in json.loads(Path(args.seeds).read_text())}
        if args.seeds
        else {}
    )
    previous_box = None
    uncertain = []
    uncovered = []
    for i in sorted(json.loads(Path(args.indices).read_text())):
        threshold = (
            min(args.confidence, 0.15) if args.continuous_track else args.confidence
        )
        prediction = model.predict(
            str(files[i]), conf=threshold, verbose=False, device=0
        )[0]
        boxes = prediction.boxes
        if args.all_candidates:
            results.append(
                dict(
                    frame=i,
                    boxes=[
                        dict(
                            box=b.xyxy[0].cpu().tolist(),
                            confidence=float(b.conf[0]),
                            class_id=int(b.cls[0]),
                        )
                        for b in boxes
                        if int(b.cls[0]) in classes
                    ],
                )
            )
            continue
        if i in seeds:
            previous_box = seeds[i]["box"]
        if reference is not None:
            expected = reference_box(reference, i)
            boxes = [
                b
                for b in boxes
                if residual_match(
                    b.xyxy[0].cpu().tolist(), expected, args.reference_margin
                )
            ]
            previous_box = expected
        candidate = merge_boxes(
            boxes,
            classes,
            i,
            previous_box,
            args.continuous_track,
            args.confidence if args.continuous_track else 0,
        )
        if i in seeds and (
            candidate is None or not overlaps(candidate["box"], seeds[i]["box"])
        ):
            candidate = dict(seeds[i])
        if candidate is not None:
            if args.continuous_track:
                for b in boxes:
                    name = str(model.names[int(b.cls[0])]).lower()
                    if (
                        "logo" in name
                        and float(b.conf[0]) >= 0.65
                        and not overlaps(b.xyxy[0].cpu().tolist(), candidate["box"])
                    ):
                        uncovered.append(
                            dict(
                                frame=i,
                                box=b.xyxy[0].cpu().tolist(),
                                confidence=float(b.conf[0]),
                                class_id=int(b.cls[0]),
                            )
                        )
            # 多个互不关联的高置信候选不能安全地当成同一水印。
            rivals = [
                b
                for b in boxes
                if int(b.cls[0]) in classes
                and float(b.conf[0]) >= max(0.65, candidate["confidence"] - 0.03)
                and not overlaps(b.xyxy[0].cpu().tolist(), candidate["box"])
            ]
            if rivals and (
                previous_box is None or not overlaps(previous_box, candidate["box"])
            ):
                uncertain.append(i)
            results.append(candidate)
            previous_box = candidate["box"]
    Path(args.output).with_name("metrics." + Path(args.output).name).write_text(
        json.dumps(
            dict(
                model_load_seconds=load_seconds,
                detection_seconds=time.perf_counter() - started,
                uncovered_watermark_frames=len({p["frame"] for p in uncovered}),
                uncovered_nontext_candidates=uncovered[:32],
            )
        )
    )
    if len(uncertain) > max(2, len(results) * 0.1):
        Path(args.output).write_text(json.dumps({"uncertain": True}))
    else:
        Path(args.output).write_text(json.dumps(results))


if __name__ == "__main__":
    try:
        from .gpu_lock import device_lock
    except ImportError:
        from gpu_lock import device_lock
    with device_lock():
        main()
