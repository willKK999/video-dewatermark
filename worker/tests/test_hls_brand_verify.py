"""完整图案的编码后核验必须识别边界裁切及反色图标。"""

import numpy as np
from scripts.hls_brand_verify import clipped_scores


def test_clipped_and_inverted_visible_icon_is_not_hidden_by_removed_text():
    rng = np.random.default_rng(3)
    template = rng.normal(0, 8, (60, 180)).astype(np.float32)
    mask = np.full((60, 180), 255, np.uint8)
    feature = np.zeros((100, 200), np.float32)
    feature[20:80, :40] = -template[:, 140:]
    scores = clipped_scores(feature, template, mask, [-140, 20, 40, 80])
    assert any(
        p["correlation"] < -0.99 and p["relative_amplitude"] < -0.99 for p in scores
    )
    empty = clipped_scores(np.zeros_like(feature), template, mask, [-140, 20, 40, 80])
    assert all(p["relative_amplitude"] == 0 for p in empty)
