"""在时域共识图上运行专用模型，多尺度验证淡色与斜向标识。"""

import argparse, json
from pathlib import Path
from ultralytics import YOLO
from yolo_stage import dedicated_classes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image")
    p.add_argument("--weights")
    p.add_argument("--output")
    a = p.parse_args()
    model = YOLO(a.weights)
    classes = dedicated_classes(model.names)
    items = []
    for size in [640, 1280, 1920]:
        result = model.predict(a.image, imgsz=size, conf=0.05, verbose=False)[0]
        for b in result.boxes:
            if int(b.cls[0]) not in classes:
                continue
            items.append(
                dict(
                    box=b.xyxy[0].cpu().tolist(),
                    confidence=float(b.conf[0]),
                    class_id=int(b.cls[0]),
                )
            )
    Path(a.output).write_text(json.dumps(items))


if __name__ == "__main__":
    main()
