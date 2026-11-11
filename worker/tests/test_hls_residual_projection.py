"""残留投影必须清除正负字形，并保留原像素的时变差和掩膜外区域。"""

import cv2, numpy as np, pytest
from worker.hls_residual_projection import fit_correction, apply_correction


@pytest.mark.parametrize("sign", [1, -1])
def test_temporal_glyph_projection_removes_both_signs_without_smoothing(sign):
    glyph = np.zeros((100, 220), np.float32)
    cv2.putText(glyph, "Brand", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 10, 3)
    glyph -= cv2.GaussianBlur(glyph, (0, 0), 3)
    mask = cv2.dilate(
        (np.abs(glyph) > 0.05).astype(np.uint8) * 255, np.ones((7, 7), np.uint8)
    )
    common = np.repeat((glyph * sign * 0.6)[:, :, None], 3, axis=2)
    field = fit_correction(glyph, common, mask)
    before = np.full((100, 220, 3), 128, np.float32) + common
    source = np.rint(before).astype(np.uint8)
    out = apply_correction(source, field, mask)
    assert (
        np.mean(np.abs(out.astype(float) - 128)[mask > 0])
        < np.mean(np.abs(source.astype(float) - 128)[mask > 0]) * 0.2
    )
    assert np.array_equal(out[mask == 0], source[mask == 0])
    other = source + np.uint8(3)
    out2 = apply_correction(other, field, mask)
    assert np.array_equal(out2.astype(int) - out.astype(int), np.full_like(out, 3))
