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


def bounded_mask(mask, box, image=None):
    """限制分割泄漏，并覆盖检测器确认的半透明文字及图标边缘。"""
    h, w = mask.shape
    x1, y1, x2, y2 = expand_box(box, w, h)
    region = np.zeros_like(mask)
    region[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 255
    if image is not None:
        # 分割器可能仅选中图标；框内边缘补充细文字，不推断固定文字颜色。
        edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                         setting('MASK_EDGE_LOW', 24), setting('MASK_EDGE_HIGH', 72))
        text_region = np.zeros_like(mask)
        a, b, c, d = box
        text_region[max(0, int(b)):min(h, int(np.ceil(d))),
                    max(0, int(a)):min(w, int(np.ceil(c)))] = 255
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(edges, text_region))
    # 只外扩分割形状，保留字间及图标周围的真实背景。
    radius = max(int(setting('MASK_PAD_MIN', 3)),
                 int(round(min(w, h)*setting('MASK_PAD_RATIO', .004))),
                 int(round((box[3]-box[1])*setting('MASK_PAD_BOX_RATIO', .12))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    grown = cv2.dilate(mask, kernel)
    outer = cv2.dilate(region, kernel)
    return cv2.bitwise_and(grown, outer)



def abrupt_area(previous, area):
    return (previous is not None and min(previous, area) > .001
            and abs(previous-area) > setting('MASK_JUMP_ABSOLUTE', .015)
            and max(previous, area)/min(previous, area) > setting('MASK_JUMP_RATIO', 3))
