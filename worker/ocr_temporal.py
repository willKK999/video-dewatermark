"""从v9移植的OCR位置关联、文字变化排除和镜头内有界补全。"""
from dataclasses import dataclass, asdict
from collections import Counter
import re
import numpy as np


@dataclass(frozen=True)
class ExperimentalSettings:
    stride: int = 5
    ocr_detection_model: str = 'PP-OCRv5_server_det'
    ocr_recognition_model: str = 'PP-OCRv5_server_rec'
    ocr_score: float = .65
    association_iou: float = .4
    maximum_gap_seconds: float = .6
    minimum_observations: int = 3
    minimum_duration_seconds: float = .4
    ocr_minimum_duration_seconds: float = 2.
    ocr_dominant_text_fraction: float = .75
    ocr_position_jitter: float = .3
    mask_fraction_limit: float = .2

    def __post_init__(self):
        if self.stride < 1 or self.minimum_observations < 2:
            raise ValueError('采样间隔和最少观测数无效')
        if self.mask_fraction_limit != .2:
            raise ValueError('实验不能改变每层20%面积保护')
        for v in (self.ocr_score,
                  self.association_iou, self.ocr_dominant_text_fraction):
            if not 0 < v <= 1:
                raise ValueError('阈值须在(0,1]之间')
        if min(self.maximum_gap_seconds, self.minimum_duration_seconds,
               self.ocr_minimum_duration_seconds, self.ocr_position_jitter) <= 0:
            raise ValueError('时域参数必须为正数')

    def to_dict(self):
        return asdict(self)


def iou(a, b):
    x = max(0, min(a[2], b[2])-max(a[0], b[0]))
    y = max(0, min(a[3], b[3])-max(a[1], b[1]))
    intersection = x*y
    return intersection / max(1e-9, (a[2]-a[0])*(a[3]-a[1]) +
                              (b[2]-b[0])*(b[3]-b[1])-intersection)


def normalize_text(text):
    return re.sub(r'[^\w]', '', text.casefold())


def associate_candidates(rows, scenes, fps, settings=None):
    """位置先关联再检查文字变化，防止按句建轨把对白误判成稳定品牌。"""
    cfg = settings or ExperimentalSettings()
    accepted, rejected = [], []
    for scene_index, (start, end) in enumerate(scenes):
        tracks = []
        for frame in sorted(int(k) for k in rows if start <= int(k) < end):
            used = set()
            for raw in sorted(rows.get(str(frame), rows.get(frame, [])),
                              key=lambda p: p['confidence'], reverse=True):
                p = dict(raw, frame=frame)
                if not np.isfinite(p['box']).all() or p['box'][2] <= p['box'][0] or p['box'][3] <= p['box'][1]:
                    rejected.append(dict(reason='无效模型框', observations=[p])); continue
                choices = [(iou(t[-1]['box'], p['box']), k) for k, t in enumerate(tracks)
                           if k not in used and t[-1]['source'] == p['source']
                           and 0 < frame-t[-1]['frame'] <= cfg.maximum_gap_seconds*fps]
                score, k = max(choices, default=(0., -1))
                if score >= cfg.association_iou:
                    tracks[k].append(p); used.add(k)
                else:
                    tracks.append([p]); used.add(len(tracks)-1)
        for observations in tracks:
            reason = None
            span = (observations[-1]['frame']-observations[0]['frame'])/fps
            source = observations[0]['source']
            if len(observations) < cfg.minimum_observations or span < cfg.minimum_duration_seconds:
                reason = '持续性不足'
            if source == 'OCR':
                words = [normalize_text(p.get('text', '')) for p in observations]
                dominant, count = Counter(words).most_common(1)[0]
                boxes = np.array([p['box'] for p in observations], dtype=float)
                centers = (boxes[:, :2]+boxes[:, 2:])/2
                size = np.maximum(1., np.median(boxes[:, 2:]-boxes[:, :2], axis=0))
                jitter = float(np.max(np.abs(centers-np.median(centers, axis=0))/size))
                if not dominant or count/len(words) < cfg.ocr_dominant_text_fraction:
                    reason = '同位置文字变化或识别不稳定'
                elif span < cfg.ocr_minimum_duration_seconds or jitter > cfg.ocr_position_jitter:
                    reason = '文字位置或持续时间不稳定'
            track = dict(id=len(accepted)+1, source=source, scene=scene_index,
                         observations=observations)
            if reason:
                rejected.append(dict(track, reason=reason)); continue
            # 按镜头内轨迹的局部中位框抑制抖动；只插值缺口，不向生命周期外扩展。
            hints = []
            for index, p in enumerate(observations):
                box = np.median([q['box'] for q in observations[max(0,index-1):index+2]], axis=0).tolist()
                hints.append(dict(p, box=box, temporal=False))
            dense = []
            for left, right in zip(hints, hints[1:]):
                dense.append(left)
                for frame in range(left['frame']+1, right['frame']):
                    weight = (frame-left['frame'])/(right['frame']-left['frame'])
                    dense.append(dict(left, frame=frame, temporal=True,
                                      box=((1-weight)*np.array(left['box'])+weight*np.array(right['box'])).tolist()))
            dense.append(hints[-1]); track['hints'] = dense
            accepted.append(track)
    return accepted, rejected


def guarded_mask(mask, limit=.2):
    if limit != .2 or mask.ndim != 2 or not set(np.unique(mask)) <= {0, 255}:
        raise ValueError('掩膜格式或20%保护配置无效')
    fraction = float(np.count_nonzero(mask)/mask.size)
    if fraction > limit:
        raise ValueError('单条实验轨迹超过20%面积保护')
    return fraction
