import numpy as np
from worker.branding import same_brand, merge_text_rows, grow_brand_components
import cv2


def test_brand_identity_is_learned_not_literal_and_rejects_subtitle():
    assert same_brand("ExampleBrand.com", ["examplebrand.com"])
    assert not same_brand("a subtitle sentence", ["examplebrand.com"])
    assert not same_brand("AI", ["examplebrand.com"])


def test_split_domain_boxes_merge_but_subtitle_row_does_not():
    boxes = [
        dict(box=b, confidence=0.8)
        for b in [[50, 40, 100, 50], [98, 40, 160, 50], [20, 90, 170, 100]]
    ]
    rows = merge_text_rows(boxes)
    assert len(rows) == 2 and rows[0]["box"] == [50, 40, 160, 50]


def test_adjacent_graphic_and_title_join_domain_but_not_distant_subtitle():
    signal = np.zeros((180, 240), np.uint8)
    signal[70:90, 90:200] = 255
    signal[30:60, 90:200] = 255
    signal[30:90, 30:75] = 255
    signal[145:160, 40:210] = 255
    result = grow_brand_components(signal, [[90, 70, 200, 90]])
    assert result[50, 40] and result[45, 120] and result[80, 120]
    assert not result[150, 120]


def test_unrelated_ocr_lines_do_not_hide_confirmed_brand_token():
    assert same_brand("examplebrand.com", ["noise\nsymbol\nexamplebrand.com"])


def test_brand_palette_keeps_graphic_and_rejects_skin_and_border_background():
    from worker.branding import learn_palette, palette_support
    import cv2

    images = []
    mask = np.zeros((80, 120), np.uint8)
    mask[10:30, 10:30] = 255
    for i in range(5):
        im = np.full((80, 120, 3), (70 + i * 10, 110 + i * 10, 140 + i * 10), np.uint8)
        im[10:30, 10:30] = (0, 220, 255)
        images.append(im)
    palette = learn_palette(images, mask, 2)
    test = np.full((80, 120, 3), (110, 150, 180), np.uint8)
    test[30:50, 40:60] = (0, 220, 255)
    test[:20, :10] = (0, 220, 255)
    result = palette_support(test, palette)
    assert result[40, 50] and not result[60, 90] and not result[5, 5]


def test_play_icon_requires_triangle_inside_circle():
    import cv2
    from worker.branding import play_icons

    image = np.full((240, 320, 3), 100, np.uint8)
    cv2.circle(image, (160, 120), 60, (0, 220, 255), -1)
    cv2.fillPoly(
        image,
        [np.array([[145, 90], [145, 150], [190, 120]], np.int32)],
        (255, 255, 255),
    )
    assert play_icons(image)
    cv2.circle(image, (160, 120), 60, (0, 220, 255), -1)
    assert not play_icons(image)


def test_changed_color_residual_is_checked_without_original_pixel_match():
    from worker.branding import residual_components

    image = np.full((120, 200, 3), 80, np.uint8)
    image[45:65, 75:85] = 240
    image[5:20, 160:180] = 240
    target = np.zeros((120, 200), np.uint8)
    target[40:80, 40:75] = 255
    palette = cv2.cvtColor(np.array([[[255, 255, 255]]], np.uint8), cv2.COLOR_BGR2LAB)[
        0
    ]
    mask = residual_components(image, palette, target)
    assert mask[50, 80] and not mask[10, 170]
