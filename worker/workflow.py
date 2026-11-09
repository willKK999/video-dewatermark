"""视频无关的检测与掩膜策略；坐标仅用于几何计算。"""
import os
import numpy as np
import cv2


def setting(name, default):
    return float(os.environ.get(name, default))


def expand_box(box, width, height):
    x1, y1, x2, y2 = box
    px = max(setting('BOX_PAD_MIN', 4), (x2-x1)*setting('BOX_PAD_RATIO', .08))
    py = max(setting('BOX_PAD_MIN', 4), (y2-y1)*setting('BOX_PAD_RATIO', .08))
    return [max(0, x1-px), max(0, y1-py), min(width, x2+px), min(height, y2+py)]


def interpolated_box(hints, index):
    return [float(np.interp(index, [p['frame'] for p in hints],
                            [p['box'][j] for p in hints])) for j in range(4)]


def bounded_mask(mask, box):
    """限制分割泄漏，并覆盖检测器确认的半透明文字及图标边缘。"""
    h, w = mask.shape
    x1, y1, x2, y2 = expand_box(box, w, h)
    region = np.zeros_like(mask)
    region[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 255
    # 检测框是覆盖下限；SAM 提供框附近的形状细节。
    radius = max(1, int(round(min(w, h)*setting('MASK_PAD_RATIO', .004))))
    grown = cv2.dilate(mask, np.ones((radius*2+1, radius*2+1), np.uint8))
    outer = cv2.dilate(region, np.ones((radius*2+1, radius*2+1), np.uint8))
    return cv2.bitwise_or(cv2.bitwise_and(grown, outer), region)


def abrupt_area(previous, area):
    return (previous is not None and min(previous, area) > .001
            and abs(previous-area) > setting('MASK_JUMP_ABSOLUTE', .015)
            and max(previous, area)/min(previous, area) > setting('MASK_JUMP_RATIO', 3))
