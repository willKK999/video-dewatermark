"""不透明遮挡不能冒充纹理损失；真正的剩余皮肤模糊仍可测。"""

import cv2
import numpy as np
from scripts.hls_detail_verify import occluder_mask
from worker.matte import stable_detail_mask, temporal_detail


def test_moving_caption_inside_fixed_faint_prior_is_not_detail_loss(tmp_path):
    rng = np.random.default_rng(876)
    a = rng.integers(80, 180, (160, 240, 3), dtype=np.uint8)
    b = np.roll(a, 1, axis=1)
    mask = np.zeros(a.shape[:2], np.uint8)
    mask[30:130, 35:205] = 255
    blocker = np.zeros_like(mask)
    blocker[35:75, 50:180] = 255
    root = tmp_path / "layer"
    (root / "masks").mkdir(parents=True)
    for i in (0, 1):
        cv2.imwrite(str(root / f"masks/{i:06d}.png"), blocker)
    source_a, source_b = a.copy(), b.copy()
    source_a[blocker > 0] = 255
    source_b[blocker > 0] = 0
    exclusion = occluder_mask([root], 1, mask.shape)
    stable = stable_detail_mask(mask, exclusion, exclusion)
    assert not stable[50, 100] and stable[110, 100]
    metric = temporal_detail(source_a, source_b, a, b, stable, mask)
    assert metric["measurable"] and metric["passed"]
    x, y = [cv2.GaussianBlur(frame, (0, 0), 3) for frame in (a, b)]
    x[mask == 0], y[mask == 0] = a[mask == 0], b[mask == 0]
    assert not temporal_detail(source_a, source_b, x, y, stable, mask)["passed"]


def test_full_occlusion_is_unmeasurable_and_missing_optional_layer_is_empty(tmp_path):
    mask = np.full((80, 120), 255, np.uint8)
    frame = np.zeros((80, 120, 3), np.uint8)
    assert not occluder_mask([tmp_path], 0, mask.shape).any()
    stable = stable_detail_mask(mask, mask, mask)
    assert not temporal_detail(frame, frame, frame, frame, stable, mask)["measurable"]
