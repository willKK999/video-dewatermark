"""自动残留证据只触发局部带上下文复修。"""

import json
import cv2
import numpy as np
import pytest
from worker import hls_repair
from worker.process import PipelineError


def test_context_ranges_do_not_cross_scene_cuts():
    assert hls_repair.intervals([29, 30], [(0, 30), (30, 90)], 90, 10) == [
        (19, 30),
        (30, 41),
    ]
    assert hls_repair.intervals([20, 22, 70], [(0, 100)], 100, 10) == [
        (10, 33),
        (60, 81),
    ]


@pytest.mark.parametrize("hls", [False, True])
def test_refinement_masks_only_affected_frames_with_temporal_context(
    tmp_path, monkeypatch, hls
):
    root = tmp_path
    for folder in ("frames", "masks", "composite"):
        (root / folder).mkdir()
    files = []
    image = np.full((20, 30, 3), 100, np.uint8)
    mask = np.zeros((20, 30), np.uint8)
    mask[8:10, 12:15] = 255
    for i in range(80):
        name = f"{i:06d}.png"
        files.append(root / "frames" / name)
        cv2.imwrite(str(files[-1]), image)
        cv2.imwrite(str(root / "composite" / name), image)
        cv2.imwrite(str(root / "masks" / name), mask)
    (root / "quality.json").write_text(
        json.dumps(dict(residual_count=1, needs_review=False))
    )
    (root / "quality-detections.json").write_text(
        json.dumps([dict(frame=40, box=[12, 8, 15, 10])])
    )
    called = []
    expected_ranges = [(0, 80)] if hls else [(30, 51)]
    affected = {i for start, end in expected_ranges for i in range(start, end)}

    def inpaint(work, current, ranges, engine, low_memory, check):
        assert ranges == expected_ranges
        assert current[0].resolve() == (root / "composite" / "000000.png").resolve()
        for i in sorted(affected):
            actual = cv2.imread(str(work / "masks" / f"{i:06d}.png"), 0)
            assert actual.any()
        (work / "composite").mkdir()
        for i in sorted(affected):
            updated = image.copy()
            updated[8:10, 12:15] = 0
            cv2.imwrite(str(work / "composite" / f"{i:06d}.png"), updated)
        called.append(1)

    def quality(*args):
        (root / "quality.json").write_text(
            json.dumps(dict(residual_count=0, needs_review=False))
        )

    monkeypatch.setattr(
        hls_repair,
        "reseed_masks",
        lambda *args: {i: mask.copy() for i in affected},
    )
    monkeypatch.setattr(hls_repair.vision, "inpaint", inpaint)
    monkeypatch.setattr(hls_repair.vision, "residual_quality", quality)
    stages = []
    result = hls_repair.refine(
        root,
        files,
        dict(hls=hls, scenes=[(0, 80)], hints=[dict(frame=0, box=[12, 8, 15, 10])]),
        "propainter",
        False,
        10,
        stages.append,
        lambda: None,
    )
    assert len(called) == 1 and stages == ["mask", "inpaint", "quality"]
    assert result[0]["residual_before"] == 1 and result[0]["residual_after"] == 0
    assert cv2.imread(str(root / "composite" / "000040.png"))[8, 12, 0] == 0
    if hls:
        assert cv2.imread(str(root / "composite" / "000000.png"))[8, 12, 0] == 0
        assert cv2.imread(str(root / "composite" / "000079.png"))[8, 12, 0] == 0
    else:
        assert 0 < cv2.imread(str(root / "composite" / "000031.png"))[8, 12, 0] < 100
        assert cv2.imread(str(root / "composite" / "000029.png"))[8, 12, 0] == 100
        assert np.array_equal(cv2.imread(str(root / "composite" / "000000.png")), image)


def test_ambiguous_residual_does_not_silently_expand_mask(tmp_path):
    (tmp_path / "quality.json").write_text(
        json.dumps(dict(residual_count=2, needs_review=True))
    )
    (tmp_path / "quality-detections.json").write_text(json.dumps(dict(uncertain=True)))
    with pytest.raises(PipelineError, match="歧义"):
        hls_repair.refine(
            tmp_path, [], {}, "propainter", False, 30, lambda _: None, lambda: None
        )


def test_refinement_rejects_invalid_frame_numbers():
    with pytest.raises(PipelineError, match="帧号"):
        hls_repair.intervals([100], [(0, 80)], 80)


def test_residual_support_covers_original_detector_omission_and_bounds_leak():
    mask = np.full((100, 100), 255, np.uint8)
    support = hls_repair.component_support(mask, [50, 50, 80, 65], [40, 48, 55, 66])
    assert support[50, 41] == 255
    assert not support[:40].any() and not support[80:].any()
    assert not support[:, :30].any() and not support[:, 90:].any()
