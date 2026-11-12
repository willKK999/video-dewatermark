"""全局颜色先验不能把其他品牌颜色误判为没有笔画；没有证据时仍拒绝修复。"""

import cv2
import numpy as np
import pytest
from worker.brand_foreground import foreground
from worker.process import PipelineError


def test_unseen_dark_brand_uses_only_real_ink_inside_confirmed_mask():
    image = np.full((100, 180, 3), 240, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.putText(mask, "WM", (30, 70), 0, 1.6, 255, 3)
    image[mask > 0] = 20
    palette = cv2.cvtColor(np.uint8([[[0, 255, 0]]]), cv2.COLOR_BGR2LAB).reshape(1, 3)
    actual, basis = foreground(image, mask, palette)
    assert np.count_nonzero(actual) >= 64 and not actual[mask == 0].any()
    assert basis == "模型掩膜内的原片明暗笔画"


def test_palette_mismatch_without_measurable_ink_keeps_review_gate():
    image = np.full((80, 120, 3), 70, np.uint8)
    mask = np.full(image.shape[:2], 255, np.uint8)
    palette = cv2.cvtColor(np.uint8([[[0, 255, 0]]]), cv2.COLOR_BGR2LAB).reshape(1, 3)
    with pytest.raises(PipelineError, match="真实笔画"):
        foreground(image, mask, palette)
