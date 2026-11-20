import cv2
import numpy as np
import pytest


def test_cuda_pyramid_tracks_forward_and_backward():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA 设备")

    from worker.roi_lk_cuda import track

    random = np.random.default_rng(42)
    source = random.integers(0, 256, (320, 400), dtype=np.uint8)
    references = [np.roll(source, (2, 1), axis=(0, 1)),
                  np.roll(source, (-3, 2), axis=(0, 1))]
    points = cv2.goodFeaturesToTrack(source, 180, .01, 5)
    forward, valid, backward, reverse_valid = track(source, references, points, bidirectional=True)

    for idx, reference in enumerate(references):
        expected, expected_valid, _ = cv2.calcOpticalFlowPyrLK(source, reference, points, None)
        expected_back, expected_back_valid, _ = cv2.calcOpticalFlowPyrLK(reference, source, expected, None)
        np.testing.assert_array_equal(valid[idx], expected_valid)
        np.testing.assert_array_equal(reverse_valid[idx], expected_back_valid)
        np.testing.assert_array_equal(forward[idx], expected)
        np.testing.assert_array_equal(backward[idx], expected_back)
