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
    confirmed = [raw for i, row in enumerate(rows) for raw in row['boxes']
                 if raw['confidence'] >= cfg.seed_confidence and
                 any(p['kind'] == 'prototype' and p['model_box'] == raw['box'] for p in candidates[i])]

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
                candidates[i].append(dict(box=list(raw['box']), model_box=list(raw['box']),
                    confidence=raw['confidence'], kind='confirmed_text', prototype=-1,
                    evidence='同镜头连续高置信度原图YOLO观测且与本输入已确认同类实例几何一致', observed_frames=len(run)))
