"""多目标与时域共识必须独立覆盖图标，且不能把变化字幕纳入。"""

import pytest
import cv2, numpy as np
from worker.multitrack import (
    temporal_consensus,
    connected_regions,
    fill_support,
    template_candidates,
    merge_masks,
)


def test_temporal_consensus_retains_two_overlays_and_rejects_changing_subtitle(
    tmp_path,
):
    rng = np.random.default_rng(7)
    files = []
    for i in range(25):
        image = rng.integers(80, 160, (100, 160, 3), dtype=np.uint8)
        cv2.rectangle(image, (8, 8), (25, 24), (255, 255, 255), 2)
        cv2.putText(image, "ABC", (65, 45), 0, 0.5, (255, 255, 255), 2)
        cv2.putText(image, str(i), (30 + i * 3, 90), 0, 0.4, (255, 255, 255), 1)
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), image)
        files.append(p)
    median, signal, _ = temporal_consensus(files)
    assert np.count_nonzero(signal[5:28, 5:28]) > 70
    assert np.count_nonzero(signal[70:]) < np.count_nonzero(signal[:60])


def test_adjacent_icon_is_recovered_by_consensus_component():
    signal = np.zeros((100, 180), np.uint8)
    signal[30:50, 25:45] = 255
    signal[32:48, 49:105] = 255
    regions = connected_regions(signal, [dict(box=[49, 32, 105, 48], confidence=0.9)])
    assert len(regions) == 1 and regions[0]["box"][0] <= 25
    assert fill_support(regions[0]["mask"])[40, 30] == 255


def test_template_finds_multiple_instances_without_single_track_rejection():
    rng = np.random.default_rng(2)
    template = rng.normal(size=(12, 20)).astype(np.float32)
    image = np.zeros((80, 130), np.float32)
    image[10:22, 15:35] = template
    image[45:57, 85:105] = template
    candidates = template_candidates(image, template, np.ones_like(template), 0.95)
    assert {tuple(p["box"]) for p in candidates} == {
        (15, 10, 35, 22),
        (85, 45, 105, 57),
    }


def test_masks_are_union_of_every_track_and_preserve_outside():
    a = np.zeros((30, 40), np.uint8)
    b = a.copy()
    a[2:6, 2:8] = 255
    b[20:24, 25:32] = 255
    result = merge_masks([a, b], a.shape)
    assert np.array_equal(result > 0, (a > 0) | (b > 0)) and result[15, 15] == 0


def test_subtitle_fragments_do_not_become_logo_from_large_model_box():
    signal = np.zeros((100, 180), np.uint8)
    signal[40:48, 60:69] = 255
    assert (
        connected_regions(signal, [dict(box=[10, 10, 170, 90], confidence=0.8)]) == []
    )


def test_prior_roundtrip_and_invalid_geometry(tmp_path):
    from worker.multitrack import save_prior, load_prior

    median = np.zeros((30, 40), np.float32)
    signal = np.zeros((30, 40), np.uint8)
    signal[3:9, 2:12] = 255
    regions = [dict(id=1, box=[2, 3, 12, 9], confidence=0.8, mask=signal)]
    path = tmp_path / "prior.npz"
    save_prior(path, median, signal, regions, [], dict(sample_count=25))
    loaded = load_prior(path)
    assert np.array_equal(loaded[2][0]["mask"], signal)
    assert loaded[4]["sample_count"] == 25


def test_multiple_moving_copies_have_independent_tracks(tmp_path, monkeypatch):
    from worker import multitrack

    image = np.zeros((80, 120, 3), np.uint8)
    files = []
    for i in range(4):
        path = tmp_path / f"{i:06d}.png"
        cv2.imwrite(str(path), image)
        files.append(path)
    median = np.zeros((80, 120), np.float32)
    signal = np.zeros_like(median, np.uint8)
    signal[:10, :10] = 255
    regions = [dict(id=1, box=[0, 0, 10, 10], confidence=0.8, mask=signal)]
    prior = tmp_path / "prior.npz"
    multitrack.save_prior(
        prior, median, signal, regions, [dict(region=1, flip=False, scale=1.0)], {}
    )
    monkeypatch.setattr(
        multitrack,
        "template_candidates",
        lambda *args: [
            dict(box=[30, 30, 40, 40], confidence=0.9),
            dict(box=[80, 45, 90, 55], confidence=0.85),
        ],
    )
    result = multitrack.make_tracks(files, [[0, 4]], prior, tmp_path)
    assert len(result["tracks"]) == 3
    assert all(len(t["hints"]) == 4 for t in result["tracks"])
    parts = multitrack.prior_masks(result, 2, (80, 120))
    assert set(parts) == {1, 2, 3}
    assert merge_masks(parts.values(), (80, 120))[48, 85] == 255


@pytest.mark.parametrize("polarity", [1, -1])
def test_full_icon_remnant_is_detected_without_text_detector(tmp_path, polarity):
    from worker import multitrack

    rng = np.random.default_rng(6)
    median = np.zeros((80, 120), np.float32)
    median[20:40, 30:55] = rng.normal(0, 10, (20, 25))
    signal = np.zeros_like(median, np.uint8)
    signal[20:40, 30:55] = 255
    region = dict(id=1, box=[30, 20, 55, 40], confidence=0.8, mask=signal)
    prior = tmp_path / "prior.npz"
    multitrack.save_prior(prior, median, signal, [region], [], {})
    (tmp_path / "composite").mkdir()
    files = []
    for i in range(3):
        path = tmp_path / f"{i:06d}.png"
        files.append(path)
        image = np.clip(polarity * median + 128, 0, 255).astype(np.uint8)
        cv2.imwrite(str(tmp_path / "composite" / path.name), image)
    detections = dict(
        prior=str(prior), tracks=[dict(id=1, kind="fixed", region=1, hints=[])]
    )
    import pytest
    from worker.process import PipelineError

    with pytest.raises(PipelineError, match="多轨迹完整标识"):
        multitrack.residual_quality(tmp_path, files, detections)
    assert (
        __import__("json").loads((tmp_path / "quality.json").read_text())[
            "residual_count"
        ]
        > 0
    )
    for path in files:
        cv2.imwrite(
            str(tmp_path / "composite" / path.name), np.full((80, 120), 128, np.uint8)
        )
    multitrack.residual_quality(tmp_path, files, detections)
    assert (
        __import__("json").loads((tmp_path / "quality.json").read_text())[
            "residual_count"
        ]
        == 0
    )


def test_temporal_gap_bridging_preserves_identity_and_rejects_long_gap():
    from worker.multitrack import bridge_gaps

    hints = [
        dict(frame=0, box=[10, 10, 20, 20], confidence=0.8, track_id=2),
        dict(frame=4, box=[14, 10, 24, 20], confidence=0.9, track_id=2),
        dict(frame=100, box=[20, 10, 30, 20], confidence=0.9, track_id=2),
    ]
    result = bridge_gaps(hints, 101)
    assert [p["frame"] for p in result] == [0, 1, 2, 3, 4, 100]
    assert result[2]["box"] == [12, 10, 22, 20] and result[2]["track_id"] == 2


def test_overlapping_boxes_do_not_hide_independent_watermarks():
    from worker.multitrack import outside_fixed

    regions = [dict(box=[40, 10, 90, 65])]
    assert outside_fixed([10, 50, 60, 80], regions)
    assert not outside_fixed([40, 10, 90, 65], regions)


def test_small_icon_residual_cannot_be_diluted_by_removed_text():
    from worker.multitrack import component_scores

    rng = np.random.default_rng(12)
    template = rng.normal(0, 8, (60, 160)).astype(np.float32)
    result = np.zeros_like(template)
    result[:30, :40] = template[:30, :40]
    scores = component_scores(template, result, np.ones_like(template, np.uint8))
    assert scores[0]["relative_amplitude"] < 0.15
    assert any(
        p["correlation"] > 0.99 and p["relative_amplitude"] > 0.99 for p in scores[1:]
    )


def test_resumed_sam_validation_covers_every_track_and_keeps_area_limit(
    tmp_path, monkeypatch
):
    from worker.multitrack import save_prior
    from worker.vision import validate_masks
    from worker.process import PipelineError
    import pytest

    monkeypatch.setenv("DETECTOR", "yolo")
    for folder in ["frames", "masks"]:
        (tmp_path / folder).mkdir()
    image = np.full((100, 200, 3), 100, np.uint8)
    file = tmp_path / "frames/000000.png"
    cv2.imwrite(str(file), image)
    cv2.imwrite(str(tmp_path / "masks/000000.png"), np.zeros((100, 200), np.uint8))
    median = np.zeros((100, 200), np.float32)
    regions = []
    tracks = []
    hints = []
    for identity, box in enumerate([[10, 10, 30, 20], [140, 60, 160, 70]], 1):
        mask = np.zeros((100, 200), np.uint8)
        x, y, r, b = box
        mask[y:b, x:r] = 255
        regions.append(dict(id=identity, box=box, mask=mask, confidence=0.9))
        hint = dict(frame=0, box=box, confidence=0.9, track_id=identity)
        hints.append(hint)
        tracks.append(dict(id=identity, region=identity, kind="fixed", hints=[hint]))
    prior = tmp_path / "prior.npz"
    save_prior(prior, median, np.zeros_like(median, np.uint8), regions, [], {})
    d = dict(
        multitrack=True, prior=str(prior), tracks=tracks, hints=hints, scenes=[[0, 1]]
    )
    manifest = validate_masks(tmp_path, [file], d)
    mask = cv2.imread(str(tmp_path / "masks/000000.png"), 0)
    assert mask[15, 20] and mask[65, 150] and not mask[45, 90]
    assert len(manifest) == 1
    regions[0]["mask"][:80, :150] = 255
    save_prior(prior, median, np.zeros_like(median, np.uint8), regions, [], {})
    with pytest.raises(PipelineError, match="20%"):
        validate_masks(tmp_path, [file], d)
