"""不透明品牌面板依赖模型定位及真实闭合轮廓，不能把字幕或皮肤当面板。"""

import cv2, numpy as np
from worker.opaque_panels import panel_support, pair_score


def test_complete_closed_panels_include_dark_fill_not_surrounding_skin():
    im = np.full((160, 400, 3), (140, 170, 190), np.uint8)
    cv2.rectangle(im, (20, 20), (95, 95), (10, 10, 10), -1)
    cv2.rectangle(im, (20, 20), (95, 95), (240, 240, 240), 2)
    cv2.circle(im, (57, 57), 23, (0, 220, 250), -1)
    cv2.rectangle(im, (100, 20), (370, 95), (10, 10, 10), -1)
    cv2.rectangle(im, (100, 20), (370, 95), (240, 240, 240), 2)
    cv2.putText(im, "Brand", (120, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 250), 2)
    mask = panel_support(im, [17, 15, 375, 105])
    assert mask[30, 30] and mask[80, 200]
    assert not mask[130, 200] and not mask[10, 10]
    assert not panel_support(
        np.full_like(im, (140, 170, 190)), [17, 15, 375, 105]
    ).any()


def test_unchanged_or_inverted_opaque_watermark_is_detected():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    mask = np.full((100, 100), 255, np.uint8)
    assert pair_score(a, a, mask)["correlation"] > 0.99
    assert pair_score(a, 255 - a, mask)["correlation"] < -0.99


def test_panel_clipped_by_video_edge_still_has_complete_support():
    im = np.full((160, 400, 3), (140, 170, 190), np.uint8)
    cv2.rectangle(im, (200, 0), (430, 80), (10, 10, 10), -1)
    cv2.rectangle(im, (200, 0), (430, 80), (240, 240, 240), 2)
    cv2.putText(im, "Brand", (215, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 250), 2)
    mask = panel_support(im, [198, 0, 400, 100])
    assert mask[30, 390] and mask[65, 250]
    assert not mask[130, 250]


def test_colour_caption_strip_is_completed_beside_closed_brand_panel():
    im = np.full((160, 400, 3), (140, 170, 190), np.uint8)
    cv2.rectangle(im, (20, 20), (370, 75), (10, 10, 10), -1)
    cv2.rectangle(im, (20, 20), (370, 75), (240, 240, 240), 2)
    cv2.rectangle(im, (20, 76), (370, 100), (0, 220, 250), -1)
    cv2.putText(
        im, "changing caption", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )
    mask = panel_support(im, [18, 15, 375, 110])
    assert mask[85, 200] and mask[40, 200]
    assert not mask[140, 200]


def test_separate_icon_is_joined_across_small_detector_gap(tmp_path):
    from worker.opaque_panels import observations

    im = np.full((160, 400, 3), (140, 170, 190), np.uint8)
    for a, b in [(20, 100), (105, 370)]:
        cv2.rectangle(im, (a, 20), (b, 95), (10, 10, 10), -1)
        cv2.rectangle(im, (a, 20), (b, 95), (240, 240, 240), 2)
    cv2.circle(im, (60, 57), 23, (0, 220, 250), -1)
    cv2.putText(im, "Brand", (120, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 250), 2)
    file = tmp_path / "000000.png"
    cv2.imwrite(str(file), im)
    rows = [
        dict(
            frame=0,
            boxes=[
                dict(box=[105, 20, 370, 95], class_id=0, confidence=0.8),
                dict(box=[20, 20, 100, 95], class_id=0, confidence=0.3),
                dict(box=[80, 125, 300, 145], class_id=2, confidence=0.99),
            ],
        )
    ]
    output = observations(rows, [file])[0]["color_matches"]
    assert len(output) == 1 and output[0]["box"] == [20, 20, 370, 95]


def test_partial_detection_cannot_clip_previously_confirmed_icon():
    from worker.opaque_panels import track_envelope

    points = [dict(box=[20, 20, 370, 100]), dict(box=[105, 20, 370, 100])]
    assert track_envelope(points) == [20, 20, 370, 100]
    im = np.full((160, 400, 3), (140, 170, 190), np.uint8)
    cv2.rectangle(im, (20, 20), (100, 95), (10, 10, 10), -1)
    cv2.rectangle(im, (20, 20), (100, 95), (240, 240, 240), 2)
    cv2.circle(im, (60, 57), 23, (0, 220, 250), -1)
    cv2.rectangle(im, (105, 20), (370, 95), (10, 10, 10), -1)
    cv2.rectangle(im, (105, 20), (370, 95), (240, 240, 240), 2)
    cv2.putText(im, "Brand", (120, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 250), 2)
    mask = panel_support(im, track_envelope(points))
    assert mask[57, 60] and mask[50, 300]


def test_icon_residue_is_not_diluted_by_cleaned_text_panel():
    rng = np.random.default_rng(44)
    source = rng.integers(0, 256, (100, 400, 3), dtype=np.uint8)
    result = np.full_like(source, 128)
    result[:50, :100] = source[:50, :100]
    score = pair_score(source, result, np.full((100, 400), 255, np.uint8))
    assert score["correlation"] > 0.99 and score["relative_amplitude"] > 0.9
    assert any(p["part"] == [0, 0, 100, 50] for p in score["components"])


def test_background_in_expansion_is_reported_separately_but_not_silently_cleared():
    from worker.opaque_panels import measured_supports
    from worker.hls_settings import is_residual_score
    rng = np.random.default_rng(56)
    source = rng.integers(0, 256, (160, 400, 3), dtype=np.uint8)
    core = np.zeros((160, 400), np.uint8)
    core[40:120, 140:260] = 255
    expanded = np.full_like(core, 255)
    result = source.copy()
    result[15:145, 115:285] = 128
    score = measured_supports(source, result, expanded, core)
    assert is_residual_score(score) and score['仅外扩区触发']
    assert not is_residual_score(score['完整轮廓测量'])


def test_only_icon_left_inside_actual_support_remains_hard_candidate():
    from worker.opaque_panels import measured_supports
    from worker.hls_settings import is_residual_score
    rng = np.random.default_rng(66)
    source = rng.integers(0, 256, (100, 400, 3), dtype=np.uint8)
    result = np.full_like(source, 128)
    result[:50, :100] = source[:50, :100]
    mask = np.full((100, 400), 255, np.uint8)
    score = measured_supports(source, result, mask, mask)
    assert is_residual_score(score) and not score['仅外扩区触发']
    assert is_residual_score(score['完整轮廓测量'])


def test_icon_without_its_own_yolo_box_is_included_by_adjacent_panel_geometry(tmp_path):
    from worker.opaque_panels import observations
    image = np.full((160, 400, 3), (140, 170, 190), np.uint8)
    for left, right in [(20, 100), (105, 370)]:
        cv2.rectangle(image, (left, 20), (right, 95), (10, 10, 10), -1)
        cv2.rectangle(image, (left, 20), (right, 95), (240, 240, 240), 3)
    cv2.circle(image, (60, 57), 23, (0, 220, 250), -1)
    cv2.putText(image, 'Brand', (120, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 250), 2)
    cv2.putText(image, 'caption', (30, 145), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    file = tmp_path / '000000.png'
    cv2.imwrite(str(file), image)
    rows = observations([dict(frame=0, boxes=[dict(box=[105, 20, 370, 95], class_id=0, confidence=.8)])], [file])
    box = rows[0]['color_matches'][0]['box']
    mask = panel_support(image, box)
    assert box[0] < 25 and mask[57, 60]
    assert not mask[140, 60] and not mask[120, 200]


def test_nearby_skin_or_plain_coloured_object_cannot_grow_brand_box():
    from worker.opaque_panels import adjacent_panel_box
    image = np.full((160, 400, 3), (140, 170, 190), np.uint8)
    cv2.rectangle(image, (20, 20), (100, 95), (0, 220, 250), -1)
    box = [105, 20, 370, 95]
    assert adjacent_panel_box(image, box) == box


def test_footer_connected_to_tall_icon_is_completed_as_one_colour_component():
    im=np.full((180,400,3),(140,170,190),np.uint8)
    cv2.rectangle(im,(20,20),(370,75),(10,10,10),-1)
    cv2.rectangle(im,(20,20),(370,75),(240,240,240),2)
    cv2.circle(im,(57,57),30,(0,220,250),-1)
    cv2.rectangle(im,(20,76),(370,114),(0,220,250),-1)
    cv2.rectangle(im,(20,40),(30,100),(0,220,250),-1)
    cv2.putText(im,'Brand',(125,65),0,1,(0,220,250),2)
    mask=panel_support(im,[18,15,375,122])
    assert mask[108,300] and mask[57,57]
    assert not mask[150,300]
