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


def test_large_overlapping_scene_object_cannot_absorb_logo_track():
    result = merge_boxes([box(0, [11, 11, 111, 31], .8),
                          box(0, [5, 5, 220, 90], .95)], {0}, 5, [10, 10, 110, 30])
    assert result['box'] == [11, 11, 111, 31]


def test_reference_gate_keeps_weak_residue_despite_stronger_subtitle(tmp_path, monkeypatch):
    import json
    import sys
    from worker.yolo_stage import main
    frames = tmp_path / 'frames'
    frames.mkdir()
    (frames / '000000.png').touch()
    weights = tmp_path / 'best.pt'
    weights.touch()
    (tmp_path / 'indices.json').write_text('[0]')
    (tmp_path / 'reference.json').write_text(json.dumps(dict(scenes=[[0, 1]], hints=[
        dict(frame=0, box=[10, 10, 60, 30])
    ])))
    class Model:
        names = {0: 'corner_logo'}
        def predict(self, *args, **kwargs):
            return [SimpleNamespace(boxes=[box(0, [10, 10, 60, 30], .4),
                                           box(0, [10, 90, 180, 110], .95)])]
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=lambda path: Model()))
    monkeypatch.setattr(sys, 'argv', ['yolo_stage', '--frames', str(frames), '--weights', str(weights),
        '--indices', str(tmp_path/'indices.json'), '--output', str(tmp_path/'out.json'),
        '--reference', str(tmp_path/'reference.json')])
    main()
    result = json.loads((tmp_path/'out.json').read_text())
    assert result[0]['box'] == [10, 10, 60, 30] and result[0]['confidence'] == .4


def test_hls_overlap_seed_prevents_subtitle_reinitialization(tmp_path,monkeypatch):
    import json,sys
    from worker.yolo_stage import main
    frames=tmp_path/'frames';frames.mkdir()
    for i in range(2):(frames/f'{i:06d}.png').touch()
    weights=tmp_path/'best.pt';weights.touch()
    (tmp_path/'indices.json').write_text('[0,1]')
    (tmp_path/'seeds.json').write_text(json.dumps([dict(frame=0,box=[10,10,60,30],confidence=.8)]))
    class Model:
        names={0:'corner_logo'}
        def predict(self,image,**kwargs):
            i=int(Path(image).stem)
            return [SimpleNamespace(boxes=[box(0,[12+i*3,10,62+i*3,30],.8),box(0,[10,90,180,110],.95)])]
    from pathlib import Path
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda path:Model()))
    monkeypatch.setattr(sys,'argv',['yolo_stage','--frames',str(frames),'--weights',str(weights),'--indices',str(tmp_path/'indices.json'),'--output',str(tmp_path/'out.json'),'--seeds',str(tmp_path/'seeds.json')])
    main()
    result=json.loads((tmp_path/'out.json').read_text())
    assert [p['box'] for p in result]==[[12,10,62,30],[15,10,65,30]]


def test_overlapping_large_component_cannot_expand_existing_track():
    result=merge_boxes([box(0,[50,50,150,75],.8),box(0,[5,35,155,85],.7)],{0},10,[50,50,150,75])
    assert result['box']==[50,50,150,75]
    result=merge_boxes([box(0,[10,20,30,40],.7),box(2,[28,18,100,42],.8)],{0,2},5,[10,18,100,42])
    assert result['box']==[10,18,100,42]


def test_hls_missing_logo_does_not_reinitialize_on_subtitle():
    assert merge_boxes([box(0,[10,90,180,110],.95)],{0},5,[10,10,60,30],continuous=True) is None
    result=merge_boxes([box(0,[13,10,63,30],.6),box(0,[10,90,180,110],.95)],{0},6,[10,10,60,30],continuous=True)
    assert result['box']==[13,10,63,30]


def test_hls_adjacent_logo_parts_keep_whole_overlay():
    result=merge_boxes([box(0,[10,20,90,45],.8),box(0,[93,20,130,45],.6)],{0},5,[10,20,130,45],continuous=True)
    assert result['box']==[10,20,130,45]
    result=merge_boxes([box(0,[10,20,90,45],.8),box(0,[160,20,200,45],.6)],{0},5,[10,20,130,45],continuous=True)
    assert result['box']==[10,20,90,45]


def test_hls_weak_adjacent_part_cannot_initialize_or_steal_track():
    parts=[box(0,[50,50,130,75],.8),box(0,[13,50,47,75],.2),box(0,[10,90,200,120],.3)]
    result=merge_boxes(parts,{0},5,[10,50,130,75],continuous=True,anchor_confidence=.35)
    assert result['box']==[13,50,130,75]
    assert merge_boxes(parts[1:],{0},5,[10,50,130,75],continuous=True,anchor_confidence=.35) is None


def test_residual_region_includes_omitted_neighboring_icon():
    from worker.yolo_stage import reference_match, reference_region
    expected=[100,200,180,230];icon=[58,195,99,235]
    assert not reference_match(icon,expected)
    assert reference_match(icon,reference_region(expected,1.5))
    assert not reference_match([20,700,280,740],reference_region(expected,1.5))


def test_hls_weak_background_cannot_expand_track_and_cause_later_dropout():
    previous=[10,10,110,30]
    result=merge_boxes([box(0,previous,.9),box(0,[0,5,230,35],.2)],{0},0,previous,continuous=True,anchor_confidence=.35)
    assert result['box']==previous
    following=merge_boxes([box(0,[12,10,112,30],.9)],{0},1,result['box'],continuous=True,anchor_confidence=.35)
    assert following['box']==[12,10,112,30]


def test_hls_residual_geometry_retains_icon_and_rejects_nearby_foreground():
    from worker.yolo_stage import residual_match
    expected=[100,200,180,230]
    assert residual_match([58,195,99,235],expected,1.5)
    assert not residual_match([100,220,170,270],expected,1.5)
    assert not residual_match([100,175,170,245],expected,1.5)


def test_hls_detection_records_uncovered_logo_but_not_subtitles(tmp_path,monkeypatch):
    import json,sys
    from worker.yolo_stage import main
    frames=tmp_path/'frames';frames.mkdir();(frames/'000000.png').touch();weights=tmp_path/'weights.pt';weights.touch();(tmp_path/'indices.json').write_text('[0]')
    class Model:
        names={0:'corner_logo',1:'float_logo',2:'float_text'}
        def predict(self,*args,**kwargs):return [SimpleNamespace(boxes=[box(0,[10,10,60,40],.95),box(1,[120,50,190,80],.8),box(2,[10,100,190,120],.9)])]
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=lambda *args:Model()))
    monkeypatch.setattr(sys,'argv',['stage','--frames',str(frames),'--weights',str(weights),'--indices',str(tmp_path/'indices.json'),'--output',str(tmp_path/'out.json'),'--continuous-track'])
    main();audit=json.loads((tmp_path/'metrics.out.json').read_text())
    assert audit['uncovered_watermark_frames']==1
    assert [p['class_id'] for p in audit['uncovered_nontext_candidates']]==[1]
