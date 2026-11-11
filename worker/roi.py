"""按窗口全部掩膜规划固定局部区域，保持像素尺度与镜头边界。"""

import math
import cv2
import numpy as np


def plan_rois(masks, margin=96, minimum=128, alignment=8, full_ratio=0.8):
    if margin < 16 or minimum < 8 or alignment < 1 or not 0 < full_ratio <= 1:
        raise ValueError("局部补画参数无效")
    union = None
    for mask in masks:
        if mask is None or mask.ndim != 2:
            raise ValueError("局部补画掩膜无效")
        if union is None:
            union = mask > 0
        elif union.shape != mask.shape:
            raise ValueError("局部补画掩膜尺寸不一致")
        else:
            union |= mask > 0
    if union is None or not union.any():
        return []
    h, w = union.shape

    def axis(a, b, limit):
        length = min(
            limit, math.ceil(max(b - a + 2 * margin, minimum) / alignment) * alignment
        )
        start = max(0, min(limit - length, (a + b - length) // 2))
        start = start // alignment * alignment
        end = min(limit, math.ceil((start + length) / alignment) * alignment)
        # 贴边时只增加上下文，不缩小有效支持。
        if end < b:
            end = limit
        return start, end

    count, _, stats, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), 8)
    boxes = []
    for x, y, width, height, _ in stats[1:count]:
        a, b = axis(int(x), int(x + width), w)
        c, d = axis(int(y), int(y + height), h)
        boxes.append((a, c, b, d))
    # 将字形周边相交的区域传递合并，防止覆盖顺序影响结果。
    changed = True
    while changed:
        changed = False
        result = []
        for box in boxes:
            for k, other in enumerate(result):
                if (
                    box[0] < other[2]
                    and other[0] < box[2]
                    and box[1] < other[3]
                    and other[1] < box[3]
                ):
                    result[k] = (
                        min(box[0], other[0]),
                        min(box[1], other[1]),
                        max(box[2], other[2]),
                        max(box[3], other[3]),
                    )
                    changed = True
                    break
            else:
                result.append(box)
        boxes = result
    if sum((b - x) * (d - y) for x, y, b, d in boxes) >= w * h * full_ratio:
        return [(0, 0, w, h)]
    return sorted(boxes)


def paste_roi(original, repair, mask, box):
    """只回填原掩膜支持，不扩大实际修改范围。"""
    x, y, a, b = box
    if repair.shape != (b - y, a - x, 3):
        raise ValueError("局部补画结果尺寸不一致")
    active = mask[y:b, x:a] > 0
    original[y:b, x:a][active] = repair[active]
    return original
