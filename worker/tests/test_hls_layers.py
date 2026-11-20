"""多个真实水印联合可超过20%，每遍施工保护及来源验证必须保持。"""

import cv2, numpy as np, pytest
from worker.hls_layers import validate_union, track_bins, write_manifest, VerifiedLayers
from worker.process import PipelineError
import json


def test_multiple_verified_layers_cover_more_than_twenty_percent():
    a = np.zeros((100, 100), np.uint8)
    b = a.copy()
    a[:15] = 255
    b[50:65] = 255
    actual = a | b
    assert np.count_nonzero(actual) / actual.size == 0.3
    assert validate_union(actual, [a, b])
    with pytest.raises(PipelineError):
        validate_union(actual, [actual])
    wrong = actual.copy()
    wrong[90:, 90:] = 255
    with pytest.raises(PipelineError):
        validate_union(wrong, [a, b])


def test_track_partition_keeps_every_track_and_same_per_pass_limit():
    sizes = {1: 0.12, 2: 0.08, 3: 0.09, 4: 0.04}
    bins = track_bins(sizes)
    assert sorted(v for group in bins for v in group) == list(sizes)
    assert all(sum(sizes[v] for v in group) <= 0.2 for group in bins)
    with pytest.raises(PipelineError):
        track_bins({1: 0.21})


def test_layer_proof_is_bound_to_detection_and_every_mask(tmp_path):
    mask = np.zeros((20, 20), np.uint8)
    mask[:3] = 255
    directory = tmp_path / "masks"
    directory.mkdir()
    file = directory / "000000.png"
    cv2.imwrite(str(file), mask)
    detected = tmp_path / "detections.json"
    detected.write_text(json.dumps(dict(hints=[])))
    write_manifest(tmp_path, detected, [directory], [file])
    proof = VerifiedLayers(tmp_path / "layers.json", detected)
    assert proof.validate(0, mask)
    cv2.imwrite(str(file), np.zeros_like(mask))
    with pytest.raises(PipelineError):
        VerifiedLayers(tmp_path / "layers.json", detected).validate(0, mask)
