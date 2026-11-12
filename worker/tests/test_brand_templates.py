"""多个同品牌标识必须分别覆盖，翻转缩放后也不能只跟踪其中一个。"""

import cv2, numpy as np
from worker.brand_templates import variants, match, mask_for


def test_multiple_scaled_mirrored_brand_instances_keep_full_icon():
    seed = np.full((80, 160, 3), 60, np.uint8)
    mask = np.zeros((80, 160), np.uint8)
    cv2.circle(mask, (30, 40), 19, 255, -1)
    cv2.putText(mask, "WM", (60, 55), 0, 1.1, 255, 2)
    seed[mask > 0] = 220
    bank = variants(seed, mask, [0.5, 1])
    image = np.full((180, 400, 3), 60, np.uint8)
    glyph = seed[21:60, 11:114]
    # 使用自动产生的完整模板范围，保持翻转后的图标与文字一起移动。
    yy, xx = np.where(mask > 0)
    glyph = seed[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    small = cv2.flip(cv2.resize(glyph, None, fx=0.5, fy=0.5), 1)
    image[20 : 20 + small.shape[0], 30 : 30 + small.shape[1]] = small
    image[110 : 110 + glyph.shape[0], 210 : 210 + glyph.shape[1]] = glyph
    found = match(image, bank, 0.8)
    out = mask_for(image.shape[:2], bank, found)
    assert len(found) == 2
    assert out[30, 30 + small.shape[1] - 12] and out[130, 230]
    assert not out[80, 150]


def test_partial_brand_at_edge_keeps_visible_icon_and_never_wraps_mask():
    seed = np.full((80, 160, 3), 60, np.uint8)
    mask = np.zeros((80, 160), np.uint8)
    cv2.putText(mask, "BRAND", (2, 55), 0, 0.8, 255, 2)
    cv2.circle(mask, (140, 30), 14, 255, -1)
    seed[mask > 0] = 220
    bank = variants(seed, mask, [1])
    yy, xx = np.where(mask > 0)
    patch = seed[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    image = np.full((160, 260, 3), 60, np.uint8)
    cut = 40
    image[40 : 40 + patch.shape[0], : patch.shape[1] - cut] = patch[:, cut:]
    points = match(image, bank, 0.6, allow_partial=True)
    out = mask_for(image.shape[:2], bank, points)
    assert any(p["box"][0] < 0 for p in points)
    assert out[55, 100] and not out[:, -1].any()


def test_icon_only_remains_detectable_after_most_text_leaves_frame():
    seed = np.full((80, 220, 3), 60, np.uint8)
    mask = np.zeros((80, 220), np.uint8)
    cv2.putText(mask, "LONG BRAND", (0, 55), 0, 0.75, 255, 2)
    cv2.circle(mask, (195, 30), 18, 255, -1)
    seed[mask > 0] = 220
    yy, xx = np.where(mask > 0)
    patch = seed[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    bank = variants(seed, mask, [1])
    cut = patch.shape[1] - 45
    image = np.full((150, 260, 3), 60, np.uint8)
    image[35 : 35 + patch.shape[0], :45] = patch[:, cut:]
    points = match(image, bank, 0.8, allow_partial=True)
    out = mask_for(image.shape[:2], bank, points)
    assert points and out[50, 20]


def test_constant_frame_never_creates_numeric_brand_candidates():
    image = np.full((120, 220, 3), 137, np.uint8)
    seed = image[:60, :160].copy()
    mask = np.zeros((60, 160), np.uint8)
    cv2.putText(mask, "WM", (10, 45), 0, 1, 255, 2)
    seed[mask > 0] = 220
    bank = variants(seed, mask, [1])
    assert not match(image, bank, 0.5, allow_partial=True)


def test_temporal_brand_template_excludes_background_without_hiding_remnant():
    from worker.brand_templates import temporal_brand_consensus
    from worker.multitrack import component_scores

    rng = np.random.default_rng(42)
    stamp = np.zeros((80, 180), np.float32)
    cv2.putText(stamp, "WM", (25, 55), 0, 1.2, 12, 2)
    backgrounds = [rng.normal(0, 10, stamp.shape).astype(np.float32) for _ in range(40)]
    mask = np.full(stamp.shape, 255, np.uint8)
    template, support = temporal_brand_consensus([b + stamp for b in backgrounds], mask)
    clean = component_scores(template, backgrounds[0], support)
    assert not any(
        abs(q["correlation"]) >= 0.55 and abs(q["relative_amplitude"]) >= 0.15
        for q in clean
    )
    residual = component_scores(template, backgrounds[0] + stamp, support)
    assert any(
        q["correlation"] >= 0.55 and q["relative_amplitude"] >= 0.15 for q in residual
    )


def test_peak_rechecks_share_exact_source_feature_without_repeated_filtering(
    monkeypatch,
):
    from worker import brand_templates as brand

    seed = np.full((80, 160, 3), 60, np.uint8)
    mask = np.zeros(seed.shape[:2], np.uint8)
    cv2.putText(mask, "BRAND", (2, 55), 0, 0.8, 255, 2)
    seed[mask > 0] = 220
    bank = brand.variants(seed, mask, [1])
    image = np.tile(seed, (2, 3, 1))
    original_filter = brand.highpass
    calls = []
    monkeypatch.setattr(
        brand, "highpass", lambda im: calls.append(1) or original_filter(im)
    )
    points = brand.match(image, bank, 0.8, allow_partial=True)
    assert len(points) > 1 and len(calls) == 1
    feature = original_filter(image)
    for point in points:
        variant = bank[point["variant"]]
        assert brand.verified_score(
            image, variant, point["box"], feature
        ) == brand.verified_score(image, variant, point["box"])
