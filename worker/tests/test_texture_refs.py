"""局部光流回填只能使用其它帧未被水印覆盖的真实像素。"""

import cv2, numpy as np
import pytest
from worker.texture_refs import NativeReferences


@pytest.mark.parametrize("blocked_taps", [False, True])
def test_reference_restores_clean_texture_and_rejects_covered_pixels(
    tmp_path, blocked_taps
):
    for folder in ["composite", "masks", "source"]:
        (tmp_path / folder).mkdir()
    rng = np.random.default_rng(1)
    image = rng.integers(95, 115, (60, 100, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(image, (0, 0), 3)
    for i in range(11):
        mask = np.zeros((60, 100), np.uint8)
        if i == 5:
            mask[20:40, 30:70] = 255
        elif blocked_taps:
            for x, y in [(38, 30), (62, 30), (50, 18), (50, 42)]:
                mask[y - 1 : y + 2, x - 1 : x + 2] = 255
        cv2.imwrite(str(tmp_path / "composite" / f"{i:06d}.png"), base)
        cv2.imwrite(str(tmp_path / "source" / f"{i:06d}.png"), image)
        cv2.imwrite(str(tmp_path / "masks" / f"{i:06d}.png"), mask)
    restore = NativeReferences(tmp_path, tmp_path / "source", [[0, 11]])

    class Zero:
        def calc(self, a, b, c):
            return np.zeros((*a.shape, 2), np.float32)

    restore.flow = Zero()
    result, filled = restore.restore(5)
    assert filled[30, 50] and np.array_equal(result[30, 50], image[30, 50])
    mask = np.full((60, 100), 255, np.uint8)
    for i in [0, 10]:
        cv2.imwrite(str(tmp_path / "masks" / f"{i:06d}.png"), mask)
    restore = NativeReferences(tmp_path, tmp_path / "source", [[0, 11]])
    restore.flow = Zero()
    _, filled = restore.restore(5)
    assert not filled.any()


def test_reference_texture_keeps_target_tone_without_exposure_patch(tmp_path):
    for folder in ["composite", "masks", "source"]:
        (tmp_path / folder).mkdir()
    rng = np.random.default_rng(4)
    image = rng.integers(85, 115, (60, 100, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(image, (0, 0), 3)
    for i in range(11):
        mask = np.zeros((60, 100), np.uint8)
        if i == 5:
            mask[20:40, 30:70] = 255
        cv2.imwrite(str(tmp_path / "composite" / f"{i:06d}.png"), base)
        cv2.imwrite(str(tmp_path / "masks" / f"{i:06d}.png"), mask)
        cv2.imwrite(
            str(tmp_path / "source" / f"{i:06d}.png"), image if i == 5 else image + 7
        )
    restore = NativeReferences(
        tmp_path, tmp_path / "source", [[0, 11]], match_low_frequency=True
    )

    class Zero:
        def calc(self, a, b, c):
            return np.zeros((*a.shape, 2), np.float32)

    restore.flow = Zero()
    out, filled = restore.restore(5)
    assert filled[30, 50]
    assert abs(out[25:35, 40:60].mean() - base[25:35, 40:60].mean()) < 2
    assert out[25:35, 40:60].std() > base[25:35, 40:60].std() * 3
