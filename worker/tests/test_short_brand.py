import cv2, numpy as np
from worker import short_brand
from worker.multitrack import save_prior


def setup_case(tmp_path, monkeypatch, subtitle=False):
    files = []
    for i in range(5):
        p = tmp_path / f"{i:06d}.png"
        cv2.imwrite(str(p), np.full((80, 160, 3), 100, np.uint8))
        files.append(p)
    mask = np.zeros((80, 160), np.uint8)
    mask[5:15, 5:35] = 255
    prior = tmp_path / "prior.npz"
    save_prior(
        prior,
        np.zeros((80, 160), np.float32),
        mask,
        [dict(id=1, box=[5, 5, 35, 15], mask=mask, confidence=0.9)],
        [],
        {},
    )
    d = dict(
        scenes=[[0, 5]], hints=[dict(frame=i, box=[5, 5, 35, 15]) for i in range(5)]
    )
    rows = [
        dict(
            frame=i,
            boxes=[dict(box=[90, 45, 145, 55], confidence=0.7 + i * 0.03, class_id=2)],
        )
        for i in range(5)
    ]
    monkeypatch.setattr(
        short_brand,
        "recognize",
        lambda image, psm=7: (
            "examplebrand.com" if psm == 11 or not subtitle else "ordinary subtitle"
        ),
    )
    selected = np.zeros((80, 160), np.uint8)
    selected[30:55, 70:145] = 255
    monkeypatch.setattr(short_brand, "palette_support", lambda *a: selected)
    monkeypatch.setattr(short_brand, "grow_brand_components", lambda *a: selected)
    return files, d, prior, rows


def test_short_brand_adds_model_derived_mask_without_manual_box(tmp_path, monkeypatch):
    args = setup_case(tmp_path, monkeypatch)
    result = short_brand.discover(
        *args, np.zeros((1, 3), np.float32), tmp_path, stride=1
    )
    assert result["automatic_short_brand"] and result["scenes"] == [[0, 5]]
    assert (
        result["hints"][0]["frame"] == 4
        and result["hints"][0]["mask_prompt"] == "prompts/seed.png"
    )
    assert (tmp_path / "prompts/seed.png").exists()


def test_subtitle_text_without_confirmed_brand_is_not_a_watermark(
    tmp_path, monkeypatch
):
    args = setup_case(tmp_path, monkeypatch, True)
    assert (
        short_brand.discover(*args, np.zeros((1, 3), np.float32), tmp_path, stride=1)
        is None
    )


def test_graphic_tail_outlives_text_observation_without_crossing_cut():
    assert short_brand.tracking_interval((0, 424), 119) == [0, 150]
    assert short_brand.tracking_interval((100, 130), 119) == [100, 130]
