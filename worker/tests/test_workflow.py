import cv2
import numpy as np
from worker.workflow import expand_box, interpolated_box, bounded_mask, abrupt_area
from worker.vision import shots


def test_moving_box_and_edge_padding():
    hints = [dict(frame=0, box=[0, 10, 40, 30]), dict(frame=4, box=[20, 10, 60, 30])]
    assert interpolated_box(hints, 2) == [10, 10, 50, 30]
    assert expand_box([0, 10, 40, 30], 100, 100) == [0, 6, 44, 34]
    mask = np.full((100, 100), 255, np.uint8)
    result = bounded_mask(mask, [10, 10, 50, 30])
    assert np.all(result[6:34, 6:54] == 255)
    assert np.all(result[60:] == 0)  # 不允许 SAM 泄漏到独立字幕或气泡


def test_area_jump_requires_material_change():
    assert not abrupt_area(.002, .009)
    assert not abrupt_area(.02, .025)
    assert abrupt_area(.005, .04)
    assert not abrupt_area(None, .04)


def test_same_histogram_structural_cut(tmp_path):
    a = np.zeros((90, 160, 3), np.uint8)
    a[:, :80] = 255
    b = 255-a
    paths = []
    for i, image in enumerate([a, a, b, b]):
        path = tmp_path / f'{i}.png'
        cv2.imwrite(str(path), image)
        paths.append(path)
    assert shots(paths) == [(0, 2), (2, 4)]


def test_residual_requires_persistent_overlapping_evidence(tmp_path, monkeypatch):
    import json
    import pytest
    import worker.vision as vision
    from worker.process import PipelineError
    monkeypatch.setenv('DETECTOR', 'yolo')
    monkeypatch.setenv('DETECT_PYTHON', 'python')
    monkeypatch.setenv('YOLO_WEIGHTS', 'local.pt')
    files = [tmp_path / f'{i:06d}.png' for i in range(16)]
    detections = dict(scenes=[(0, 16)], hints=[dict(frame=0, box=[10, 10, 40, 30])])
    candidates = [dict(frame=i, box=[10, 10, 40, 30]) for i in range(3)]
    def fake_run(*args):
        (tmp_path / 'quality-detections.json').write_text(json.dumps(candidates))
    monkeypatch.setattr(vision, 'run', fake_run)
    vision.residual_quality(tmp_path, files, detections)
    candidates.append(dict(frame=3, box=[10, 10, 40, 30]))
    with pytest.raises(PipelineError) as exc:
        vision.residual_quality(tmp_path, files, detections)
    assert exc.value.review and exc.value.code == 'RESIDUAL_UNCERTAIN'
    for p in candidates:
        p['box'] = [100, 100, 200, 120]  # 独立字幕不计入原水印残留
    vision.residual_quality(tmp_path, files, detections)


def test_shape_padding_preserves_background_between_characters():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:30, 20:25] = 255
    mask[20:30, 50:55] = 255
    result = bounded_mask(mask, [15, 15, 60, 35])
    assert result[25, 17] == 255
    assert result[25, 40] == 0
    assert not bounded_mask(np.zeros_like(mask), [15, 15, 60, 35]).any()
