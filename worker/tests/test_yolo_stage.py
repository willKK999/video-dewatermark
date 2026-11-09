from types import SimpleNamespace
import numpy as np
import pytest
from worker.yolo_stage import dedicated_classes, merge_boxes


class Coordinates:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return np.array(self.values)


def box(cls, coords, confidence):
    return SimpleNamespace(cls=[cls], xyxy=[Coordinates(coords)], conf=[confidence])


def test_dedicated_watermark_classes():
    assert dedicated_classes({0: "corner_logo", 1: "float_logo", 2: "float_text"}) == {0, 1, 2}
    assert dedicated_classes(["watermark", "brand-logo", "水印"]) == {0, 1, 2}
    for names in ({0: "person", 1: "car"}, {0: "watermark", 1: "person"}, {0: "text"}, {}):
        with pytest.raises(ValueError):
            dedicated_classes(names)


def test_multiple_parts_cover_button_and_text():
    result = merge_boxes([box(0, [10, 20, 30, 40], .7), box(2, [28, 18, 100, 42], .8),
                          box(9, [0, 0, 400, 400], .99)], {0, 2}, 5)
    assert result == dict(frame=5, box=[10, 18, 100, 42], confidence=.8)
    assert merge_boxes([], {0}, 0) is None
    assert merge_boxes([box(9, [0, 0, 10, 10], .9)], {0}, 0) is None


def test_stage_uses_specialist_threshold_and_local_weights(tmp_path, monkeypatch):
    import json
    import sys
    from worker.yolo_stage import main

    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "000000.png").touch()
    weights = tmp_path / "best.pt"
    weights.touch()
    indices = tmp_path / "indices.json"
    indices.write_text("[0]")
    output = tmp_path / "output.json"
    calls = []

    class Model:
        names = {0: "corner_logo", 1: "float_text"}

        def predict(self, image, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=[box(0, [5, 10, 20, 30], .5),
                                            box(1, [18, 10, 60, 30], .6)])]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=lambda path: Model()))
    monkeypatch.delenv("YOLO_CONFIDENCE", raising=False)
    argv = ["yolo_stage", "--frames", str(frames), "--weights", str(weights),
            "--indices", str(indices), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert calls[0]["conf"] == .35
    assert json.loads(output.read_text())[0]["box"] == [5, 10, 60, 30]
    monkeypatch.setenv("YOLO_CONFIDENCE", ".45")
    main()
    assert calls[1]["conf"] == .45
    weights.unlink()
    with pytest.raises(ValueError, match="local dedicated"):
        main()


def test_temporal_match_keeps_distant_subtitles_out():
    detections = [box(0, [12, 21, 80, 42], .8), box(0, [90, 760, 350, 790], .95),
                  box(1, [13, 22, 25, 41], .6)]
    result = merge_boxes(detections, {0, 1}, 5, [10, 20, 78, 41])
    assert result['box'] == [12, 21, 80, 42]
    assert merge_boxes(detections, {0, 1}, 0)['box'] == [90, 760, 350, 790]


def test_grazing_false_positive_does_not_expand_watermark():
    result = merge_boxes([box(0, [94, 329, 247, 358], .84),
                          box(0, [244, 354, 377, 407], .46)], {0}, 120)
    assert result['box'] == [94, 329, 247, 358]
