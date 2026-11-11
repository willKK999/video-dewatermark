"""透明反解不能以涂抹背景代替原始细节，质量检查必须能拦住糊化。"""

import cv2, numpy as np
from worker.matte import fit_overlay, invert_overlay, temporal_detail


def test_overlay_inverse_preserves_texture_and_reduces_stamp():
    rng = np.random.default_rng(14)
    mask = np.zeros((80, 120), np.uint8)
    cv2.putText(mask, "LOGO", (10, 45), 0, 0.8, 255, 1)
    backgrounds = []
    images = []
    for i in range(40):
        x = np.clip(rng.normal(100 + i * 2, 12, (80, 120, 3)), 0, 255).astype(np.uint8)
        y = x.copy()
        y[mask > 0] = np.rint(x[mask > 0] * 0.88 + 255 * 0.12).astype(np.uint8)
        backgrounds.append(x)
        images.append(y)
    alpha, offset = fit_overlay(images, mask)
    clean = invert_overlay(images[20], alpha, offset)
    before = np.mean(
        np.abs(images[20][mask > 0].astype(float) - backgrounds[20][mask > 0])
    )
    after = np.mean(np.abs(clean[mask > 0].astype(float) - backgrounds[20][mask > 0]))
    assert after < before * 0.5
    assert np.array_equal(clean[mask == 0], images[20][mask == 0])
    assert np.std(clean[mask > 0]) >= np.std(images[20][mask > 0]) * 0.95


def test_detail_check_rejects_blurred_region_and_accepts_original():
    rng = np.random.default_rng(9)
    a = rng.integers(40, 200, (80, 120, 3), dtype=np.uint8)
    b = np.roll(a, 2, axis=1)
    mask = np.zeros((80, 120), np.uint8)
    mask[20:60, 30:90] = 255
    x = a.copy()
    y = b.copy()
    x[mask > 0] = cv2.GaussianBlur(a, (0, 0), 3)[mask > 0]
    y[mask > 0] = cv2.GaussianBlur(b, (0, 0), 3)[mask > 0]
    assert not temporal_detail(a, b, x, y, mask)["passed"]
    assert temporal_detail(a, b, a, b, mask)["passed"]


def test_sparse_ink_does_not_close_skin_between_letters():
    from worker.matte import ink_support

    signal = np.zeros((80, 140), np.uint8)
    signal[20:50, 20:24] = 255
    signal[20:50, 70:74] = 255
    template = (signal > 0).astype(np.float32)
    mask = ink_support(template, signal, np.full_like(signal, 255), 1)
    assert mask[30, 21] and mask[30, 71] and not mask[30, 45]


def test_texture_fusion_preserves_non_stamp_high_frequency():
    from worker.matte import texture_composite

    rng = np.random.default_rng(3)
    original = rng.integers(90, 140, (60, 100, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(original, (0, 0), 3)
    mask = np.zeros((60, 100), np.uint8)
    mask[10:50, 15:85] = 255
    result, beta = texture_composite(
        original, base, mask, np.zeros((60, 100), np.float32)
    )
    assert np.std(result[15:45, 20:80]) > np.std(base[15:45, 20:80]) * 3
    assert np.array_equal(result[mask == 0], original[mask == 0])


def test_channel_projection_preserves_texture_without_returning_colored_stamp():
    from worker.matte import projected_texture_composite

    rng = np.random.default_rng(17)
    background = np.clip(rng.normal(130, 8, (96, 160, 3)), 0, 255).astype(np.uint8)
    stamp = np.zeros((96, 160), np.float32)
    cv2.putText(stamp, "WM", (20, 60), 0, 1.2, 1, 2)
    original = background.astype(np.float32)
    original += stamp[:, :, None] * np.array([25, 15, 8], np.float32)
    original = np.clip(original, 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(background, (0, 0), 2)
    mask = cv2.dilate((stamp > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    template = stamp - cv2.GaussianBlur(stamp, (0, 0), 1.2)
    output = projected_texture_composite(original, base, mask, template)
    active = stamp > 0
    before = np.mean(np.abs(original[active].astype(float) - background[active]))
    after = np.mean(np.abs(output[active].astype(float) - background[active]))
    assert after < before * 0.6
    assert np.array_equal(output[mask == 0], original[mask == 0])


def test_detail_ring_excludes_other_processed_pixels():
    rng = np.random.default_rng(2)
    a = rng.integers(70, 150, (120, 200, 3), dtype=np.uint8)
    b = np.roll(a, 1, axis=1)
    mask = np.zeros((120, 200), np.uint8)
    mask[30:60, 30:60] = 255
    other = np.zeros_like(mask)
    other[30:60, 64:78] = 255
    x = a.copy()
    y = b.copy()
    x[other > 0] = 0
    y[other > 0] = 255
    assert temporal_detail(a, b, x, y, mask, mask | other)["passed"]
    x[mask > 0] = cv2.GaussianBlur(a, (0, 0), 3)[mask > 0]
    y[mask > 0] = cv2.GaussianBlur(b, (0, 0), 3)[mask > 0]
    assert not temporal_detail(a, b, x, y, mask, mask | other)["passed"]


def test_joint_projection_does_not_return_overlapping_watermarks():
    from worker.matte import projected_texture_composite

    rng = np.random.default_rng(12)
    background = np.clip(rng.normal(130, 6, (100, 180, 3)), 0, 255).astype(np.uint8)
    one = np.zeros((100, 180), np.float32)
    two = one.copy()
    cv2.putText(one, "AB", (25, 55), 0, 1.1, 1, 2)
    cv2.putText(two, "CD", (45, 65), 0, 1.1, 1, 2)
    original = np.clip(
        background.astype(float) + one[:, :, None] * 20 + two[:, :, None] * 15, 0, 255
    ).astype(np.uint8)
    mask = cv2.dilate(
        ((one + two) > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8)
    )
    templates = [v - cv2.GaussianBlur(v, (0, 0), 1.2) for v in [one, two]]
    out = projected_texture_composite(
        original, cv2.GaussianBlur(background, (0, 0), 2), mask, templates
    )
    before = np.abs(original.astype(float) - background)[mask > 0].mean()
    after = np.abs(out.astype(float) - background)[mask > 0].mean()
    assert after < before * 0.7


def test_confirmed_ink_does_not_return_when_template_has_subpixel_error():
    from worker.matte import projected_texture_composite

    rng = np.random.default_rng(26)
    background = rng.integers(95, 140, (100, 180, 3), dtype=np.uint8)
    ink = np.zeros((100, 180), np.uint8)
    cv2.putText(ink, "WM", (30, 60), 0, 1.2, 255, 2)
    mask = cv2.dilate(ink, np.ones((9, 9), np.uint8))
    source = background.copy()
    source[ink > 0] = np.clip(source[ink > 0].astype(float) + 30, 0, 255)
    base = cv2.GaussianBlur(background, (0, 0), 2)
    shifted = np.roll(ink.astype(np.float32) / 255, 1, axis=1)
    g = shifted - cv2.GaussianBlur(shifted, (0, 0), 1.2)
    out = projected_texture_composite(source, base, mask, g, ink)
    error = np.abs(out[ink > 0].astype(float) - background[ink > 0]).mean()
    assert (
        error < np.abs(source[ink > 0].astype(float) - background[ink > 0]).mean() * 0.5
    )
    assert np.array_equal(out[mask == 0], source[mask == 0])


def test_temporally_calibrated_projection_preserves_moving_texture():
    from worker.matte import projected_texture_composite, solve_projection

    rng = np.random.default_rng(88)
    stamp = np.zeros((96, 160), np.float32)
    cv2.putText(stamp, "WM", (20, 60), 0, 1.2, 1, 2)
    mask = cv2.dilate((stamp > 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    g = stamp - cv2.GaussianBlur(stamp, (0, 0), 1.2)
    acc = {}

    def pair():
        clean = np.clip(rng.normal(120, 8, (96, 160, 3)), 0, 255).astype(np.uint8)
        source = np.clip(clean.astype(float) + stamp[:, :, None] * 18, 0, 255).astype(
            np.uint8
        )
        return clean, source, cv2.GaussianBlur(clean, (0, 0), 2)

    for _ in range(30):
        _, source, base = pair()
        projected_texture_composite(source, base, mask, g, accumulator=acc)
    coefficients = solve_projection(acc)
    a = []
    b = []
    for _ in range(2):
        clean, source, base = pair()
        out = projected_texture_composite(
            source, base, mask, g, calibration=coefficients
        )
        assert np.abs(out.astype(float) - clean)[stamp > 0].mean() < 8
        a.append(source)
        b.append(out)
    assert temporal_detail(*a, *b, mask)["passed"]


def test_moving_stamp_is_excluded_without_hiding_real_blur():
    from worker.matte import stable_detail_mask

    rng = np.random.default_rng(77)
    a = rng.integers(90, 140, (120, 200, 3), dtype=np.uint8)
    b = np.roll(a, 1, axis=1)
    mask = np.zeros((120, 200), np.uint8)
    mask[30:90, 40:160] = 255
    moving_a = np.zeros_like(mask)
    moving_a[40:60, 55:100] = 255
    moving_b = np.zeros_like(mask)
    moving_b[40:60, 65:110] = 255
    source_a = a.copy()
    source_b = b.copy()
    source_a[moving_a > 0] = 240
    source_b[moving_b > 0] = 240
    stable = stable_detail_mask(mask, moving_a, moving_b)
    assert stable[70, 80] and not stable[50, 100]
    assert temporal_detail(source_a, source_b, a, b, stable, mask)["passed"]
    x = cv2.GaussianBlur(a, (0, 0), 3)
    y = cv2.GaussianBlur(b, (0, 0), 3)
    # 周边保持原片，实际掩膜内的糊化仍必须被拒绝。
    x[mask == 0] = a[mask == 0]
    y[mask == 0] = b[mask == 0]
    assert not temporal_detail(source_a, source_b, x, y, stable, mask)["passed"]


def test_intermediate_frame_uses_real_temporal_change_without_returning_static_mark():
    from worker.matte import restore_intermediate_frame

    rng = np.random.default_rng(12)
    a = rng.integers(80, 150, (120, 200, 3), dtype=np.uint8)
    b = np.roll(a, 1, axis=1)
    mask = np.zeros((120, 200), np.uint8)
    mask[25:95, 30:170] = 255
    stamp = np.zeros_like(mask)
    cv2.putText(stamp, "WM", (60, 75), 0, 1.3, 255, 2)

    def overlay(image):
        return np.clip(image.astype(float) + stamp[:, :, None] * 0.08, 0, 255).astype(
            np.uint8
        )

    x, y = overlay(a), overlay(b)
    frozen = b.copy()
    frozen[mask > 0] = a[mask > 0]
    assert not temporal_detail(x, y, a, frozen, mask)["passed"]
    restored = restore_intermediate_frame(frozen, a, y, x, mask)
    assert temporal_detail(x, y, a, restored, mask)["passed"]
    assert (
        np.abs(restored[35:85, 40:160].astype(float) - b[35:85, 40:160]).mean() < 0.01
    )
    assert np.array_equal(restored[mask == 0], frozen[mask == 0])


def test_multiband_cutoff_preserves_real_mid_frequency_skin_texture():
    from worker.matte import projected_texture_composite

    yy, xx = np.indices((100, 180))
    gray = (125 + 12 * np.sin(xx * 2 * np.pi / 8)).astype(np.uint8)
    a = np.repeat(gray[:, :, None], 3, axis=2)
    b = np.roll(a, 1, axis=1)
    mask = np.zeros((100, 180), np.uint8)
    mask[20:80, 25:155] = 255
    g = np.zeros(mask.shape, np.float32)
    result = []
    old = []
    for x in [a, b]:
        base = cv2.GaussianBlur(x, (0, 0), 3)
        old.append(projected_texture_composite(x, base, mask, g, detail_sigma=1.2))
        result.append(projected_texture_composite(x, base, mask, g, detail_sigma=3))
    assert not temporal_detail(a, b, *old, mask)["passed"]
    assert temporal_detail(a, b, *result, mask)["passed"]
    assert np.array_equal(result[0][mask == 0], a[mask == 0])


def test_template_rescaling_cannot_change_restored_background_detail():
    from worker.matte import projected_texture_composite

    rng = np.random.default_rng(80)
    background = rng.integers(80, 150, (100, 180, 3), dtype=np.uint8)
    stamp = np.zeros((100, 180), np.float32)
    cv2.putText(stamp, "WM", (25, 65), 0, 1.3, 1, 2)
    source = np.clip(background.astype(float) + stamp[:, :, None] * 20, 0, 255).astype(
        np.uint8
    )
    base = cv2.GaussianBlur(background, (0, 0), 3)
    mask = cv2.dilate((stamp > 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    g = stamp - cv2.GaussianBlur(stamp, (0, 0), 1.2)
    small = projected_texture_composite(source, base, mask, g)
    large = projected_texture_composite(source, base, mask, g * 100)
    assert np.abs(small.astype(float) - large).max() <= 1
