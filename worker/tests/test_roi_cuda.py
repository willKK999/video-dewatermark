import cv2
import numpy as np
import pytest

from worker import roi_cuda


@pytest.fixture
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA 显卡")


def test_donor_consensus_matches_cpu(cuda):
    rng = np.random.default_rng(18)
    values = rng.integers(10, 32, size=(9, 43, 3)).astype(np.float32)
    values[3:, ::2] = np.nan
    values[6:, 1::2] = np.nan
    expected = np.nanmedian(values, axis=0)
    agreement = np.nanmax(np.abs(values - expected), axis=(0, 2)) <= 10
    actual, actual_agreement = roi_cuda.donor_consensus(values)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_agreement, agreement)


@pytest.mark.parametrize("size", [1, 3, 15, 67])
def test_box_count_matches_opencv_at_edges(cuda, size):
    rng = np.random.default_rng(83)
    values = rng.integers(0, 2, size=(55, 127))[:, ::2].astype(bool, order="F")
    expected = cv2.boxFilter(
        values.astype(np.float32), -1, (size, size),
        normalize=False, borderType=cv2.BORDER_CONSTANT,
    )
    np.testing.assert_array_equal(roi_cuda.box_count(values, size), expected)
