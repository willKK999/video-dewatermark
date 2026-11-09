"""无干净真值的成对质量审计；指标仅作伪影线索，不能代替目检。"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def align_previous(original, previous_original, previous, mask, previous_mask):
    """只从水印以外估计相邻帧相机运动，不用修复结果估计光流。"""
    h, w = mask.shape
    excluded = cv2.dilate(cv2.bitwise_or(mask, previous_mask), np.ones((9, 9), np.uint8))
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    oldgray = cv2.cvtColor(previous_original, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(gray, 200, .01, 6, mask=255-excluded)
    if points is None or len(points) < 16:
        return None
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray, oldgray, points, None)
    if tracked is None or status.sum() < 16:
        return None
    valid = status.ravel() > 0
    matrix, inliers = cv2.estimateAffinePartial2D(points[valid], tracked[valid], method=cv2.RANSAC, ransacReprojThreshold=1)
    if matrix is None or inliers.mean() < .7:
        return None
    return [cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) for frame in previous]


def audit(root, baseline, output, times):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    caps = [cv2.VideoCapture(str(path)) for path in [baseline, root.parent/(root.name+'-result.mp4')]]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not all(cap.isOpened() for cap in caps):
        raise ValueError('视频无法读取')
    samples = {round(t*fps): t for t in times}
    outside_changed = 0
    rows, previous, previous_mask, previous_original = [], None, None, None
    detections = json.loads((root.parent/(root.name+'-detections.json')).read_text())
    cuts = {scene[0] for scene in detections['scenes']}
    paths = sorted((root/'frames').glob('*.png'))
    for i, path in enumerate(paths):
        original = cv2.imread(str(path))
        mask = cv2.imread(str(root/'masks'/path.name), 0)
        frames = [cap.read()[1] for cap in caps]
        if any(f is None for f in frames):
            raise ValueError('视频帧数不匹配')
        core = mask > 0
        support = mask > 0
        composite = cv2.imread(str(root/'composite'/path.name))
        outside_changed += int(np.count_nonzero(np.any(composite != original, axis=2) & ~support))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17,17))
        ring = (cv2.dilate(mask, kernel) > 0) & ~core
        if not core.any() or not ring.any():
            previous, previous_mask, previous_original = frames, mask, original
            continue
        row = {'frame': i, 'seconds': i/fps}
        aligned = None
        if previous is not None and i not in cuts:
            aligned = align_previous(original, previous_original, previous, mask, previous_mask)
        # 边界上修复引入的颜色不连续；原图差分抵消真实场景边缘。
        boundary = (mask > 0) & (cv2.erode(mask, np.ones((3,3), np.uint8)) == 0)
        for label, frame in zip(['before','after'], frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            delta = frame.astype(np.float32)-original.astype(np.float32)
            gradient = np.maximum(np.abs(delta-np.roll(delta,1,axis=0)).mean(axis=2), np.abs(delta-np.roll(delta,1,axis=1)).mean(axis=2))
            row[label+'_boundary'] = float(gradient[boundary].mean())
            row[label+'_background_gap'] = float(np.abs(frame[core].mean(axis=0)-frame[ring].mean(axis=0)).mean())
            lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
            row[label+'_texture_ratio'] = float(lap[core].mean()/max(lap[ring].mean(),.01))
            if aligned is not None:
                prev = aligned[0 if label == 'before' else 1]
                diff = np.abs(frame.astype(float)-prev.astype(float)).mean(axis=2)
                row[label+'_flicker_mae'] = float(diff[core].mean())
                row[label+'_background_flicker'] = float(diff[ring].mean())
                row[label+'_flicker_excess'] = float(diff[core].mean()-diff[ring].mean())
        rows.append(row)
        if i in samples:
            t = samples[i]
            panels = []
            for label, frame in zip(['1','2','3'], [original]+frames):
                panel = frame.copy()
                cv2.putText(panel,label,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                panels.append(panel)
            cv2.imwrite(str(output/f'{root.name}-{t:g}s.png'), np.hstack(panels))
            yy, xx = np.where(core)
            x1,x2=max(0,xx.min()-18),min(mask.shape[1],xx.max()+19)
            y1,y2=max(0,yy.min()-18),min(mask.shape[0],yy.max()+19)
            crop=np.hstack([f[y1:y2,x1:x2] for f in [original]+frames])
            cv2.imwrite(str(output/f'{root.name}-{t:g}s-crop.png'),cv2.resize(crop,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
        previous, previous_mask, previous_original = frames, mask, original
    for cap in caps: cap.release()
    keys = sorted(set().union(*(r.keys() for r in rows))-{'frame', 'seconds'})
    summary = {key: float(np.mean([r[key] for r in rows if key in r])) for key in keys}
    summary['frames'] = len(rows)
    summary['aligned_pairs'] = sum('after_flicker_mae' in r for r in rows)
    summary['outside_support_changed_pixels'] = outside_changed
    (output/(root.name+'-metrics.json')).write_text(json.dumps({'summary':summary,'frames':rows},ensure_ascii=False,indent=2))
    print(json.dumps({'clip':root.name,**summary},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--times',type=float,nargs='+',required=True)
    args=parser.parse_args()
    audit(args.root,args.baseline,args.output,args.times)
