"""不同品牌可使用不同颜色；颜色先验缺失时只复用模型确认范围内的原片笔画。"""

import numpy as np
from .branding import palette_support
from .multitrack import highpass
from .process import PipelineError


def foreground(image, mask, palette):
    if mask is None or mask.shape != image.shape[:2] or not np.any(mask):
        raise PipelineError("MASK_UNCERTAIN", "完整品牌缺少模型确认的前景", review=True)
    selected = mask & palette_support(image, palette) if len(palette) else mask.copy()
    if np.any(selected):
        return selected, "已有颜色先验" if len(palette) else "模型确认的完整前景"
    # 一个品牌的颜色不能否决其他颜色的标识；不扩张模型范围，不猜测新颜色。
    feature = np.abs(highpass(image))
    threshold = max(0.5, float(np.median(feature[mask > 0])))
    selected = ((mask > 0) & (feature >= threshold)).astype(np.uint8) * 255
    if np.count_nonzero(selected) < 64:
        raise PipelineError(
            "MASK_UNCERTAIN", "新品牌的模型前景没有足够真实笔画", review=True
        )
    return selected, "模型掩膜内的原片明暗笔画"
