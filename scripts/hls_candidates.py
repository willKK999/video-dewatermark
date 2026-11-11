"""视频去水印系统：逐帧保存专用模型的全部候选，不把字幕直接作为修复对象。"""

import argparse, json, os, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from ultralytics import YOLO
    from worker.yolo_stage import dedicated_classes
    import torch

    torch.set_num_threads(4)
    model = YOLO(os.environ["YOLO_WEIGHTS"])
    dedicated_classes(model.names)
    rows = []
    started = time.monotonic()
    for index, path in enumerate(sorted(a.frames.glob("*.png"))):
        result = model.predict(
            str(path), imgsz=1280, conf=0.15, iou=0.7, verbose=False, device=0
        )[0]
        rows.append(
            dict(
                frame=index,
                boxes=[
                    dict(
                        box=b.xyxy[0].cpu().tolist(),
                        confidence=float(b.conf[0]),
                        class_id=int(b.cls[0]),
                    )
                    for b in result.boxes
                ],
            )
        )
    a.output.write_text(json.dumps(rows))
    a.output.with_suffix(".timing.json").write_text(
        json.dumps(
            dict(耗时秒=time.monotonic() - started, 类别=model.names),
            ensure_ascii=False,
        )
    )
    print("逐帧多候选检测完成，帧数", len(rows), flush=True)


if __name__ == "__main__":
    from worker.gpu_lock import device_lock

    with device_lock():
        main()
