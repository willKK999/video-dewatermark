"""Only dedicated watermark weights; generic COCO models are not acceptable."""

import argparse
import json
import os
from pathlib import Path
import re


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
    smaller = min((a[2] - a[0]) * (a[3] - a[1]),
                  (b[2] - b[0]) * (b[3] - b[1]))
    return smaller > 0 and intersection >= .1 * smaller


def compatible_parts(a, b):
    ah, bh = a[3]-a[1], b[3]-b[1]
    return (min(ah, bh) > 0 and max(ah, bh)/min(ah, bh) <= 2
            and abs((a[1]+a[3])-(b[1]+b[3]))/2 <= min(ah, bh)*.6)


def compatible_scale(a, b):
    aa, ba = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return min(aa, ba) > 0 and max(aa, ba)/min(aa, ba) <= 2.5


def merge_boxes(boxes, classes, frame, previous_box=None):
    relevant = [b for b in boxes if int(b.cls[0]) in classes]
    if not relevant:
        return None
    candidates = [(b.xyxy[0].cpu().tolist(), float(b.conf[0])) for b in relevant]
    # Prefer temporal continuity to a distant subtitle with higher confidence.
    nearby = [item for item in candidates if previous_box is not None and overlaps(item[0], previous_box) and compatible_scale(item[0], previous_box)]
    anchor = max(nearby or candidates, key=lambda item: item[1])
    selected = [item for item in candidates if overlaps(item[0], anchor[0]) and compatible_parts(item[0], anchor[0])]
    coords = [item[0] for item in selected]
    return dict(
        frame=frame,
        box=[min(b[0] for b in coords), min(b[1] for b in coords),
             max(b[2] for b in coords), max(b[3] for b in coords)],
        confidence=max(item[1] for item in selected),
    )


def reference_box(reference, index):
    start, end = next((s, e) for s, e in reference['scenes'] if s <= index < e)
    hints = [p for p in reference['hints'] if start <= p['frame'] < end]
    left = max((p for p in hints if p['frame'] <= index), key=lambda p: p['frame'], default=hints[0])
    right = min((p for p in hints if p['frame'] >= index), key=lambda p: p['frame'], default=hints[-1])
    fraction = (index-left['frame']) / max(1, right['frame']-left['frame'])
    return [a+(b-a)*fraction for a, b in zip(left['box'], right['box'])]


def reference_match(box, expected):
    area = (box[2]-box[0])*(box[3]-box[1])
    target_area = (expected[2]-expected[0])*(expected[3]-expected[1])
    intersection = max(0, min(box[2], expected[2])-max(box[0], expected[0])) * max(
        0, min(box[3], expected[3])-max(box[1], expected[1]))
    return area > 0 and intersection/area >= .5 and area <= target_area*2.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--indices", required=True)
    p.add_argument("--confidence", type=float, default=float(os.getenv("YOLO_CONFIDENCE", "0.35")))
    p.add_argument("--reference")
    args = p.parse_args()
    if not 0 < args.confidence <= 1:
        p.error("confidence must be in (0, 1]")
    # Require an existing local artifact; never implicitly download generic weights.
    if not Path(args.weights).is_file():
        raise ValueError("local dedicated watermark weights required")
    from ultralytics import YOLO

    model = YOLO(args.weights)
    classes = dedicated_classes(model.names)
    results = []
    files = sorted(Path(args.frames).glob("*.png"))
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    previous_box = None
    uncertain = []
    for i in sorted(json.loads(Path(args.indices).read_text())):
        prediction = model.predict(str(files[i]), conf=args.confidence, verbose=False, device=0)[0]
        boxes = prediction.boxes
        if reference is not None:
            expected = reference_box(reference, i)
            boxes = [b for b in boxes if reference_match(b.xyxy[0].cpu().tolist(), expected)]
            previous_box = expected
        candidate = merge_boxes(boxes, classes, i, previous_box)
        if candidate is not None:
            # 多个互不关联的高置信候选不能安全地当成同一水印。
            rivals = [b for b in boxes if int(b.cls[0]) in classes
                      and float(b.conf[0]) >= max(.65, candidate['confidence']-.03)
                      and not overlaps(b.xyxy[0].cpu().tolist(), candidate['box'])]
            if rivals and (previous_box is None or not overlaps(previous_box, candidate['box'])):
                uncertain.append(i)
            results.append(candidate)
            previous_box = candidate["box"]
    if len(uncertain) > max(2, len(results)*.1):
        Path(args.output).write_text(json.dumps({'uncertain': True}))
    else:
        Path(args.output).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
