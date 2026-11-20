"""仅以同镜头连续原图模型观测兜底文字检测，不推测相邻图标。"""
from .model_tracks_v7 import intersection_fraction


def restore_confirmed_text(candidates, rows, scenes, labels, cfg):
    if cfg.fallback_observations < 3 or not 0 < cfg.fallback_confidence <= 1:
        raise ValueError('文字兜底至少需要三帧高置信度观测')
    if not 0 < cfg.fallback_overlap <= 1 or cfg.fallback_scale_ratio < 1:
        raise ValueError('文字兜底几何约束无效')

    def compatible(a, b):
        if a['class_id'] != b['class_id']:
            return False
        x, y = a['box'], b['box']
        sizes = [(z[2]-z[0], z[3]-z[1]) for z in (x, y)]
        if min(*sizes[0], *sizes[1]) <= 0:
            return False
        ratio = max(max(u/v, v/u) for u, v in zip(*sizes))
        return ratio <= cfg.fallback_scale_ratio and min(
            intersection_fraction(x, y), intersection_fraction(y, x)
        ) >= cfg.fallback_overlap

    # 只从本输入原模板路径已确认的同类实例取锚点；不把连续字幕当新品牌。
    confirmed = [dict(raw, confirmed_prototype=p['prototype'])
                 for i, row in enumerate(rows) for raw in row['boxes'] for p in candidates[i]
                 if raw['confidence'] >= cfg.seed_confidence and p['kind'] == 'prototype'
                 and p['model_box'] == raw['box']]

    for start, end in scenes:
        active = []
        runs = []
        for i in range(start, end):
            current = []
            used = set()
            for raw in sorted(rows[i]['boxes'], key=lambda p: p['confidence'], reverse=True):
                if ('text' not in labels.get(raw['class_id'], '') or raw['confidence'] < cfg.fallback_confidence
                        or not any(compatible(anchor, raw) for anchor in confirmed)):
                    continue
                eligible = [(min(intersection_fraction(t[-1][1]['box'], raw['box']),
                                 intersection_fraction(raw['box'], t[-1][1]['box'])), k)
                            for k, t in enumerate(active) if k not in used and compatible(t[-1][1], raw)]
                if eligible:
                    _, k = max(eligible)
                    run = active[k]
                    used.add(k)
                else:
                    run = []
                    runs.append(run)
                run.append((i, raw))
                current.append(run)
            active = current
        for run in runs:
            if len(run) < cfg.fallback_observations:
                continue
            for i, raw in run:
                # 保持旧模板及图形路径优先；只恢复原流程没有覆盖的直接观测。
                if any(intersection_fraction(raw['box'], p['box']) > cfg.overlap_fraction for p in candidates[i]):
                    continue
                anchor=max((a for a in confirmed if compatible(a, raw)), key=lambda a:
                    min(intersection_fraction(a['box'],raw['box']),intersection_fraction(raw['box'],a['box'])))
                candidates[i].append(dict(box=list(raw['box']), model_box=list(raw['box']),
                    confidence=raw['confidence'], kind='confirmed_text', prototype=anchor['confirmed_prototype'],
                    evidence='同镜头连续高置信度原图YOLO观测且与本输入已确认同类实例几何一致', observed_frames=len(run)))


def confirmed_text_support(proto, size):
    """只截取已确认原型内的模型文字锚点，不把相邻完整图标扩进新框。"""
    import cv2
    import numpy as np
    mask=proto['mask']; box=proto['box']; anchor=proto['anchor']
    sx=mask.shape[1]/max(1,box[2]-box[0]);sy=mask.shape[0]/max(1,box[3]-box[1])
    a,b,c,d=[round((v-box[j%2])*(sx if j%2==0 else sy)) for j,v in enumerate(anchor)]
    a,b,c,d=max(0,a),max(0,b),min(mask.shape[1],c),min(mask.shape[0],d)
    if c<=a or d<=b:
        return np.zeros((size[1],size[0]),np.uint8)
    return cv2.resize(mask[b:d,a:c],size,interpolation=cv2.INTER_NEAREST)
