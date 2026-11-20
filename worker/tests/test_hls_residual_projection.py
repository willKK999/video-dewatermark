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


def test_smoothed_residual_side_lobes_are_removed_without_changing_temporal_texture():
    glyph = np.zeros((120, 240), np.float32)
    cv2.putText(glyph, 'Mark', (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 15, 3)
    glyph -= cv2.GaussianBlur(glyph, (0, 0), 3)
    smooth = cv2.GaussianBlur(glyph, (0, 0), 3)
    mask = cv2.dilate((np.abs(glyph) > .05).astype(np.uint8)*255, np.ones((25,25),np.uint8))
    common = np.repeat((glyph * .2 + smooth * .8)[:,:,None],3,axis=2)
    correction = fit_correction(glyph, common, mask)
    source = np.rint(128 + common).astype(np.uint8)
    result = apply_correction(source, correction, mask)
    assert np.abs(result.astype(float)-128)[mask>0].mean() < np.abs(source.astype(float)-128)[mask>0].mean() * .25
    shifted = apply_correction(source + np.uint8(2), correction, mask)
    assert np.array_equal(shifted.astype(int)-result.astype(int),np.full_like(result,2))


def test_projection_cannot_create_inverse_stamp_after_source_stamp_disappears():
    from worker.hls_residual_projection import visible_support
    from worker.multitrack import highpass
    image = np.full((120,240,3),100,np.uint8)
    cv2.putText(image,'Brand',(20,75),cv2.FONT_HERSHEY_SIMPLEX,1.5,(200,200,200),3)
    prior=highpass(image)
    mask=(np.abs(prior)>.1).astype(np.uint8)*255
    regions=[dict(id=1,mask=mask)]
    assert visible_support(image,prior,regions,{1:mask}).any()
    assert not visible_support(np.full_like(image,100),prior,regions,{1:mask}).any()


def test_subpixel_correction_is_not_rounded_away_and_has_no_temporal_randomness():
    image = np.full((192,256,3),128,np.uint8)
    mask = np.full(image.shape[:2],255,np.uint8)
    correction = np.full(image.shape,.2,np.float32)
    result = apply_correction(image,correction,mask)
    assert abs(result.mean()-127.8)<.015
    assert np.array_equal(result,apply_correction(image,correction,mask))
    assert np.array_equal(apply_correction(image+np.uint8(3),correction,mask).astype(int)-result.astype(int),np.full_like(result,3))
