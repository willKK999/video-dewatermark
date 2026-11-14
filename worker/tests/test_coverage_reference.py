"""同源历史覆盖不能流入其他输入；复用不重复SAM，且覆盖退步必须报错。"""
import json
import cv2,numpy as np,pytest
from worker.coverage_reference import merge,verify_coverage
from worker.process import PipelineError
from worker.sam_schedule import groups_for_scene


def test_other_source_is_rejected_before_mask_or_template_reuse(tmp_path):
    current=tmp_path/'current';reference=tmp_path/'reference';current.mkdir();reference.mkdir()
    (current/'frames-binding.json').write_text(json.dumps(dict(source='a',sha256={})))
    (reference/'frames-binding.json').write_text(json.dumps(dict(source='b',sha256={})))
    with pytest.raises(PipelineError) as caught:merge(current,reference,{})
    assert caught.value.code=='SOURCE_CHANGED'


def test_precomputed_models_do_not_trigger_new_sam_initialization():
    d=dict(multitrack=True,precomputed_track_ids=[7],hints=[dict(frame=0,track_id=7),dict(frame=1,track_id=8)])
    groups=groups_for_scene(d,0,3)
    assert len(groups)==1 and {p['track_id'] for _,_,h in groups for p in h}=={8}


def test_missing_baseline_pixel_is_reported_without_changing_original(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';(a/'masks').mkdir(parents=True);(b/'masks').mkdir(parents=True)
    old=np.zeros((40,60),np.uint8);old[10:20,12:30]=255
    cv2.imwrite(str(b/'masks/000000.png'),old)
    missing=old.copy();missing[11,13]=0;cv2.imwrite(str(a/'masks/000000.png'),missing)
    with pytest.raises(PipelineError) as caught:verify_coverage(a,b)
    assert caught.value.code=='COVERAGE_REGRESSION'
    cv2.imwrite(str(a/'masks/000000.png'),cv2.dilate(old,np.ones((3,3),np.uint8)))
    assert verify_coverage(a,b)['缺失像素总数']==0
    assert np.array_equal(cv2.imread(str(b/'masks/000000.png'),0),old)


def test_reliable_yolo_graphics_are_not_discarded_by_template_confirmation():
    from worker.model_tracks import reliable_graphic,ModelTrackSettings
    cfg=ModelTrackSettings()
    def track(values):return dict(kind='logo',hints=[dict(confidence=v) for v in values])
    assert reliable_graphic(track([.7,.8]),cfg)
    assert not reliable_graphic(track([.8]),cfg)
    assert not reliable_graphic(track([.3,.4]),cfg)
