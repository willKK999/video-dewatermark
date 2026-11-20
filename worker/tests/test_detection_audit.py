"""增强检测必须正确映射原帧坐标，且不能更改输入。"""

import numpy as np
from scripts.hls_detection_audit import detection_views, source_box


def test_flipped_box_round_trip_preserves_original_position():
    original = [10, 20, 50, 60]
    mirrored = source_box(original, 200, True)
    assert mirrored == [150, 20, 190, 60]
    assert source_box(mirrored, 200, True) == original
    assert source_box(original, 200, False) == original


def test_detection_enhancement_preserves_source_and_geometry():
    image = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
    before = image.copy()
    views = detection_views(image, True)
    assert len(views) == 4 and all(v.shape == image.shape for _, v, _ in views)
    assert np.array_equal(views[1][1], image[:, ::-1])
    assert np.array_equal(image, before)
    assert len(detection_views(image, False)) == 1
