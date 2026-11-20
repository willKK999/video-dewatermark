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
