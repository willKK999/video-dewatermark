"""Only dedicated watermark weights; generic COCO models are not acceptable."""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--indices", required=True)
    args = p.parse_args()
    model = YOLO(args.weights)
    if not any("watermark" in str(n).lower() for n in model.names.values()):
        raise ValueError("dedicated watermark class required")
    results = []
    files = sorted(Path(args.frames).glob("*.png"))
    for i in json.loads(Path(args.indices).read_text()):
        prediction = model.predict(str(files[i]), conf=0.85, verbose=False, device=0)[0]
        boxes = [
            b
            for b in prediction.boxes
            if "watermark" in model.names[int(b.cls[0])].lower()
        ]
        if len(boxes) == 1:
            b = boxes[0]
            results.append(
                dict(frame=i, box=b.xyxy[0].cpu().tolist(), confidence=float(b.conf[0]))
            )
    Path(args.output).write_text(json.dumps(results))


if __name__ == "__main__":
    main()
