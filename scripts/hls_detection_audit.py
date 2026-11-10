"""统计全部水印候选；增强只用于检测，不改变待处理视频。"""

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
import cv2


def detection_views(source, augment):
    views = [("原图", source, False)]
    if augment:
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(
            lab[:, :, 0]
        )
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        views += [
            ("翻转", cv2.flip(source, 1), True),
            ("局部对比增强", enhanced, False),
            ("增强后翻转", cv2.flip(enhanced, 1), True),
        ]
    return views


def source_box(coords, width, flipped):
    return (
        [width - coords[2], coords[1], width - coords[0], coords[3]]
        if flipped
        else list(coords)
    )


def main():
    p = argparse.ArgumentParser(description="逐秒统计专用模型的多尺度水印候选")
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--preview-times", default="5,60,120,200")
    a = p.parse_args()
    from ultralytics import YOLO

    root = Path(a.frames)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = YOLO(os.environ["YOLO_WEIGHTS"])
    from worker.yolo_stage import dedicated_classes

    dedicated_classes(model.names)
    records = []
    sizes = (640, 1280, 1920)
    categories = {}
    for i, name in model.names.items():
        categories[int(i)] = (
            "角标" if "corner" in name else "漂浮图标" if "logo" in name else "漂浮文字"
        )

    def iou(a, b):
        inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1])
        )
        return inter / max(
            1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        )

    preview_times = {int(x) for x in a.preview_times.split(",") if x}
    for index, path in enumerate(sorted(root.glob("*.jpg"))):
        source = cv2.imread(str(path))
        raw = []
        ts = int(path.stem)
        views = detection_views(source, a.augment)
        for variant, image, flipped in views:
            for size in sizes:
                result = model.predict(
                    image, imgsz=size, conf=0.05, iou=0.7, verbose=False, device=0
                )[0]
                for b in result.boxes:
                    coords = source_box(
                        b.xyxy[0].cpu().tolist(), source.shape[1], flipped
                    )
                    raw.append(
                        dict(
                            box=coords,
                            confidence=float(b.conf[0]),
                            class_id=int(b.cls[0]),
                            size=size,
                            variant=variant,
                        )
                    )
        merged = []
        for b in sorted(raw, key=lambda b: b["confidence"], reverse=True):
            if not any(iou(b["box"], old["box"]) >= 0.5 for old in merged):
                merged.append(b)
        records.append(dict(second=ts, raw=raw, merged=merged))
        if ts in preview_times:
            for cutoff in (0.05, 0.15, 0.35):
                image = source.copy()
                for b in merged:
                    if b["confidence"] < cutoff:
                        continue
                    box = list(map(round, b["box"]))
                    color = (0, 255, 0) if b["confidence"] >= 0.35 else (0, 220, 255)
                    cv2.rectangle(image, box[:2], box[2:], color, 2)
                    cv2.putText(
                        image,
                        f"{b['class_id']} {b['confidence']:.3f}",
                        (box[0], max(14, box[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                cv2.imwrite(str(out / f"检测-{ts:03d}秒-{cutoff:.2f}.jpg"), image)
        if (index + 1) % 25 == 0:
            print("已统计秒级样本：" + str(index + 1), flush=True)
    (out / "detections.json").write_text(json.dumps(records, ensure_ascii=False))
    summary = {
        "样本数量": len(records),
        "检测尺度": sizes,
        "使用翻转与对比增强": a.augment,
        "检测阈值": 0.05,
        "检测耗时秒": round(time.monotonic() - started, 3),
        "类别": categories,
        "各阈值统计": {},
    }
    for cutoff in (0.05, 0.15, 0.35, 0.65):
        boxes = [b for r in records for b in r["merged"] if b["confidence"] >= cutoff]
        counts = [sum(b["confidence"] >= cutoff for b in r["merged"]) for r in records]
        summary["各阈值统计"][str(cutoff)] = {
            "融合候选总数": len(boxes),
            "每秒最少候选数": min(counts),
            "每秒最多候选数": max(counts),
            "平均候选数": round(sum(counts) / max(1, len(counts)), 3),
            "类别数量": dict(Counter(categories[b["class_id"]] for b in boxes)),
        }
    summary["置信度分布"] = {}
    for low, high in ((0.05, 0.15), (0.15, 0.35), (0.35, 0.65), (0.65, 1.01)):
        summary["置信度分布"][f"{low:.2f}至{min(high,1):.2f}"] = sum(
            low <= b["confidence"] < high for r in records for b in r["merged"]
        )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("秒级多尺度统计完成", flush=True)


if __name__ == "__main__":
    main()
