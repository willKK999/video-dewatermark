"""接缝比较必须使用完成回填及羽化后的相同阶段画面。"""

from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from worker.vision import _composite_window
from worker.restoration import soft_composite
from worker.process import PipelineError


def setup_window(root, previous, raw, restored, mask):
    for name in ("frames", "masks", "composite"):
        (root / name).mkdir()
    original = np.full_like(previous, 20)
    file = root / "frames" / "000000.png"
    cv2.imwrite(str(file), original)
    cv2.imwrite(str(root / "masks" / file.name), mask)
    cv2.imwrite(str(root / "composite" / file.name), previous)
    window = root / "window"
    predicted = window / "out" / "frames" / "frames"
    predicted.mkdir(parents=True)
    cv2.imwrite(str(predicted / file.name), raw)
    references = SimpleNamespace(restore=lambda *args: restored.copy())
    return file, window, references


def test_reference_corrected_overlap_does_not_compare_raw_prediction(tmp_path):
    shape = (12, 12, 3)
    previous = np.full(shape, 50, np.uint8)
    file, window, references = setup_window(
        tmp_path,
        previous,
        np.full(shape, 200, np.uint8),
        previous,
        np.full(shape[:2], 255, np.uint8),
    )
    _composite_window(
        tmp_path,
        [file],
        0,
        1,
        12,
        window,
        lambda: None,
        references,
        dict(composite_io_seconds=0, restoration_seconds=0),
    )
    assert np.array_equal(cv2.imread(str(tmp_path / "composite" / file.name)), previous)


def test_actual_restored_overlap_disagreement_still_requires_review(tmp_path):
    shape = (12, 12, 3)
    previous = np.full(shape, 50, np.uint8)
    file, window, references = setup_window(
        tmp_path,
        previous,
        previous,
        np.full(shape, 160, np.uint8),
        np.full(shape[:2], 255, np.uint8),
    )
    with pytest.raises(PipelineError, match="overlap disagreement"):
        _composite_window(
            tmp_path,
            [file],
            0,
            1,
            12,
            window,
            lambda: None,
            references,
            dict(composite_io_seconds=0, restoration_seconds=0),
        )


def test_overlap_does_not_feather_same_frame_twice(tmp_path):
    original = np.full((20, 20, 3), 20, np.uint8)
    restored = np.full_like(original, 200)
    mask = np.zeros((20, 20), np.uint8)
    mask[4:16, 4:16] = 255
    previous = soft_composite(original, restored, mask)
    file, window, references = setup_window(
        tmp_path, previous, restored, restored, mask
    )
    _composite_window(
        tmp_path,
        [file],
        0,
        1,
        12,
        window,
        lambda: None,
        references,
        dict(composite_io_seconds=0, restoration_seconds=0),
    )
    actual = cv2.imread(str(tmp_path / "composite" / file.name))
    assert np.array_equal(actual, previous)
    assert np.array_equal(actual[mask == 0], original[mask == 0])


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("large_mask", [False, True])
def test_single_frame_scene_gets_same_frame_context_without_extra_output(
    tmp_path, monkeypatch, count, large_mask
):
    import sys
    from worker import vision

    for name in ("frames", "masks"):
        (tmp_path / name).mkdir()
    image = np.full((20, 20, 3), 80, np.uint8)
    mask = np.zeros((20, 20), np.uint8)
    mask[8:12, 8:12] = 255
    if large_mask:
        mask[5:15, 5:15] = 255
    files = []
    for i in range(count):
        path = tmp_path / "frames" / f"{i:06d}.png"
        files.append(path)
        cv2.imwrite(str(path), image)
        cv2.imwrite(str(tmp_path / "masks" / path.name), mask)
    monkeypatch.setenv("INPAINT_PYTHON", sys.executable)
    monkeypatch.setenv("PROPAINTER_ROOT", str(tmp_path / "model"))
    monkeypatch.setattr(
        vision,
        "CleanReferences",
        lambda *args: SimpleNamespace(restore=lambda a, m, b: b, stats={}, timings={}),
    )
    calls = []

    def model(args, check, cwd):
        assert args[args.index("--subvideo_length") + 1] == (
            "36" if large_mask else "60"
        )
        assert args[args.index("--neighbor_length") + 1] == (
            "6" if large_mask else "10"
        )
        frames = Path(args[args.index("--video") + 1])
        paths = sorted(frames.glob("*.png"))
        assert len(paths) == 2
        assert np.array_equal(cv2.imread(str(paths[0])), cv2.imread(str(paths[1])))
        dest = Path(args[args.index("--output") + 1]) / "frames" / "frames"
        dest.mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(dest / f"{i:06d}.png"), image)
        calls.append(1)

    from pathlib import Path

    monkeypatch.setattr(vision, "run", model)
    vision.inpaint(tmp_path, files, [(0, count)], "propainter")
    assert (
        len(calls) == 1 and len(list((tmp_path / "composite").glob("*.png"))) == count
    )


def test_overlap_records_measured_disagreement_without_changing_limit(tmp_path):
    previous = np.full((12, 12, 3), 50, np.uint8)
    file, window, references = setup_window(
        tmp_path,
        previous,
        previous,
        np.full_like(previous, 60),
        np.full((12, 12), 255, np.uint8),
    )
    performance = dict(composite_io_seconds=0, restoration_seconds=0)
    _composite_window(
        tmp_path, [file], 0, 1, 8, window, lambda: None, references, performance
    )
    assert performance["overlap_samples"] == 1
    assert performance["overlap_max_mae"] == 10


def test_partial_scene_inpaint_does_not_read_masks_outside_requested_range(
    tmp_path, monkeypatch
):
    from worker import vision
    from pathlib import Path
    import sys

    for folder in ["frames", "masks"]:
        (tmp_path / folder).mkdir()
    image = np.full((20, 20, 3), 80, np.uint8)
    files = []
    for i in range(6):
        path = tmp_path / "frames" / f"{i:06d}.png"
        cv2.imwrite(str(path), image)
        files.append(path)
    for i in [2, 3]:
        cv2.imwrite(
            str(tmp_path / "masks" / files[i].name), np.zeros((20, 20), np.uint8)
        )
    monkeypatch.setenv("INPAINT_PYTHON", sys.executable)
    monkeypatch.setenv("PROPAINTER_ROOT", str(tmp_path))
    monkeypatch.setattr(
        vision,
        "CleanReferences",
        lambda *a: SimpleNamespace(restore=lambda x, m, y: y, stats={}, timings={}),
    )

    def model(args, check, cwd):
        dest = Path(args[args.index("--output") + 1]) / "frames/frames"
        dest.mkdir(parents=True)
        for i in range(2):
            cv2.imwrite(str(dest / f"{i:06d}.png"), image)

    monkeypatch.setattr(vision, "run", model)
    vision.inpaint(tmp_path, files, [(2, 4)], "propainter")
    assert {p.name for p in (tmp_path / "composite").glob("*.png")} == {
        "000002.png",
        "000003.png",
    }
