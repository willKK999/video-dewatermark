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
    candidates = [dict(frame=i, box=[10, 10, 40, 30]) for i in range(2)]
    def fake_run(*args):
        (tmp_path / 'quality-detections.json').write_text(json.dumps(candidates))
    monkeypatch.setattr(vision, 'run', fake_run)
    vision.residual_quality(tmp_path, files, detections)
    candidates.append(dict(frame=2, box=[10, 10, 40, 30]))
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


def test_yolo_samples_every_frame_and_rejects_long_blind_gap(tmp_path, monkeypatch):
    import json
    import pytest
    import worker.vision as vision
    from worker.process import PipelineError
    monkeypatch.setenv('DETECTOR', 'yolo')
    monkeypatch.setenv('DETECT_PYTHON', 'python')
    monkeypatch.setenv('YOLO_WEIGHTS', 'local.pt')
    files = [tmp_path / f'{i:06d}.png' for i in range(13)]
    def fake_run(*args):
        indices = json.loads((tmp_path / 'detect-indices.json').read_text())
        assert indices == list(range(13))
        (tmp_path / 'detector-output.json').write_text(json.dumps([
            dict(frame=0, box=[2, 2, 10, 8], confidence=.8)
        ]))
    monkeypatch.setattr(vision, 'run', fake_run)
    with pytest.raises(PipelineError) as exc:
        vision.detect(files, [(0, 13)], {}, tmp_path)
    assert exc.value.review and exc.value.code == 'DETECTION_UNCERTAIN'


def test_edge_support_recovers_text_omitted_by_segmentation():
    image = np.full((100, 200, 3), 60, np.uint8)
    cv2.putText(image, 'ABC', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (220, 220, 220), 2)
    mask = np.zeros((100, 200), np.uint8)
    mask[25:50, 15:25] = 255
    result = bounded_mask(mask, [10, 20, 110, 55], image)
    assert np.count_nonzero(result[20:55, 40:105]) > 300
    assert not result[80:].any()


def test_hls_geometry_stabilizes_spike_without_smoothing_motion_or_seeds():
    from worker.workflow import stabilize_hls_hints
    hints=[dict(frame=i,box=[i,10,i+100,30],confidence=.8) for i in range(21)]
    hints[10]['box']=[0,0,200,50]
    result=stabilize_hls_hints(hints,[(0,21)])
    assert result[10]['box']==[10,10,110,30]
    assert result[9]==hints[9] and result[11]==hints[11]
    assert stabilize_hls_hints(hints,[(0,21)],[hints[10]])[10]==hints[10]
    assert stabilize_hls_hints(hints,[(0,10),(10,21)])[10]==hints[10]
