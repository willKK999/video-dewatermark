import cv2, numpy as np
from worker.fine_masks import Supports
from worker.multitrack import save_prior


def test_absent_global_watermarks_do_not_erase_skin_and_opaque_is_automatic(tmp_path):
    shape = (100, 200)
    median = np.zeros(shape, np.float32)
    signal = np.zeros(shape, np.uint8)
    regions = []
    tracks = []
    for identity, box in enumerate([[10, 10, 50, 30], [120, 50, 165, 75]], 1):
        x, y, a, b = box
        mask = np.zeros(shape, np.uint8)
        cv2.putText(mask, "WM", (x + 2, b - 3), 0, 0.4, 255, 1)
        median[mask > 0] = 8
        signal[mask > 0] = 255
        regions.append(dict(id=identity, box=box, mask=mask, confidence=0.9))
        tracks.append(
            dict(
                id=identity,
                region=identity,
                kind="fixed",
                hints=[dict(frame=0, box=box)],
            )
        )
    prior = tmp_path / "prior.npz"
    save_prior(prior, median, signal, regions, [], {})
    alpha = np.zeros(shape, np.float32)
    alpha[regions[0]["mask"] > 0] = 0.9
    import json

    (tmp_path / "detections.json").write_text(json.dumps(dict(scenes=[[0, 1]])))
    support = Supports(prior, dict(tracks=tracks), alpha, tmp_path)
    assert support.opaque == {1}
    image = cv2.cvtColor(
        np.clip(128 + median * 8, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )
    assert len(support.at(0, image)) == 2
    assert support.at(0, np.full((100, 200, 3), 128, np.uint8)) == []


def test_short_graphic_is_kept_even_when_global_style_is_absent(tmp_path):
    shape = (60, 100)
    mask = np.zeros(shape, np.uint8)
    mask[10:30, 20:40] = 255
    prior = tmp_path / "prior.npz"
    save_prior(
        prior,
        np.zeros(shape, np.float32),
        mask,
        [dict(id=1, box=[20, 10, 40, 30], mask=mask, confidence=0.9)],
        [],
        {},
    )
    (tmp_path / "masks").mkdir()
    cv2.imwrite(str(tmp_path / "masks/000000.png"), mask)
    support = Supports(prior, dict(tracks=[]), np.zeros(shape, np.float32), tmp_path)
    parts = support.at(0, np.full((60, 100, 3), 128, np.uint8))
    assert len(parts) == 1 and parts[0]["kind"] == "graphic"
