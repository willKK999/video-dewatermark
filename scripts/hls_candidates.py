"""视频去水印系统：逐帧保存专用模型的全部候选，不把字幕直接作为修复对象。"""

import argparse, json, os, time
from pathlib import Path


def box_iou(a,b):
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def detection_views(image,enhance=False,tiles=False,flips=False):
    import cv2
    views=[(image,0,0,False)]
    if enhance:
        lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
        lab[:,:,0]=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(lab[:,:,0])
        augmented=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR);views.append((augmented,0,0,False))
    else:augmented=image
    if tiles:
        h,w=image.shape[:2];cw=max(1,round(w*.625));ch=max(1,round(h*.625))
        views += [(augmented[y:y+ch,x:x+cw],x,y,False) for y in [0,h-ch] for x in [0,w-cw]]
    if flips:views += [(cv2.flip(v,1),0,0,True) for v in ([image,augmented] if enhance else [image])]
    return views


def mapped_box(box,view):
    image,dx,dy,flipped=view
    if flipped:box=[image.shape[1]-box[2],box[1],image.shape[1]-box[0],box[3]]
    return [box[0]+dx,box[1]+dy,box[2]+dx,box[3]+dy]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--enhance",action="store_true")
    p.add_argument("--tiles",action="store_true")
    p.add_argument("--flips",action="store_true")
    a = p.parse_args()
    from ultralytics import YOLO
    from worker.yolo_stage import dedicated_classes
    import torch
    import cv2,numpy as np

    torch.set_num_threads(4)
    model = YOLO(os.environ["YOLO_WEIGHTS"])
    dedicated_classes(model.names)
    rows = []
    started = time.monotonic()
    for index, path in enumerate(sorted(a.frames.glob("*.png"))):
        image=cv2.imread(str(path));views=detection_views(image,a.enhance,a.tiles,a.flips)
        results = model.predict([v[0] for v in views],imgsz=1280,conf=0.15,iou=0.7,verbose=False,device=0)
        boxes=[]
        for k,result in enumerate(results):
            for b in result.boxes:
                box=b.xyxy[0].cpu().tolist()
                boxes.append(dict(box=mapped_box(box,views[k]),
                    confidence=float(b.conf[0]),class_id=int(b.cls[0]),detection_view=k))
        selected=[]
        for b in sorted(boxes,key=lambda b:-b['confidence']):
            if not any(b['class_id']==q['class_id'] and box_iou(b['box'],q['box'])>.7 for q in selected):selected.append(b)
        rows.append(
            dict(
                frame=index,
                boxes=selected,
            )
        )
    a.output.write_text(json.dumps(rows))
    a.output.with_suffix(".timing.json").write_text(
        json.dumps(
            dict(耗时秒=time.monotonic() - started, 类别=model.names,低对比增强=a.enhance,重叠分区检测=a.tiles,翻转检测=a.flips),
            ensure_ascii=False,
        )
    )
    print("逐帧多候选检测完成，帧数", len(rows), flush=True)


if __name__ == "__main__":
    from worker.gpu_lock import device_lock

    with device_lock():
        main()
