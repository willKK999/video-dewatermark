"""使用合成检测验证模板缺失兜底及其防误报边界。"""
from worker.model_tracks_v7 import ModelTrackSettings, associate
from worker.model_text_confirmation import restore_confirmed_text


def observed(confidence=.85, box=None, class_id=1):
    return dict(confidence=confidence, box=box or [10, 10, 50, 20], class_id=class_id)


def restore(rows, scenes=None, existing=None, confirmed=True):
    candidates = existing or [[] for _ in rows]
    scenes = scenes or [[0,len(rows)]]
    if confirmed:
        anchor=observed()
        rows=[[anchor]]+rows
        candidates=[[dict(box=anchor['box'],model_box=anchor['box'],kind='prototype',prototype=0)]]+candidates
        scenes=[[0,1]]+[[a+1,b+1] for a,b in scenes]
    restore_confirmed_text(candidates, [dict(boxes=r) for r in rows], scenes,
                           {1:'corner_text',2:'float_text',3:'corner_logo'}, ModelTrackSettings())
    return candidates[1:] if confirmed else candidates


def test_high_confidence_continuous_text_survives_without_template():
    result=restore([[observed()] for _ in range(5)])
    assert all(len(r)==1 and r[0]['box']==[10,10,50,20] for r in result)
    assert all(r[0]['observed_frames']==5 for r in result)
    tracks=associate(result,[[0,5]],30,ModelTrackSettings())
    assert len(tracks)==1 and len(tracks[0]['hints'])==5


def test_isolated_low_confidence_and_graphic_are_not_restored():
    assert restore([[observed()],[],[observed()]])==[[],[],[]]
    for p in [observed(.79),observed(class_id=3)]:
        assert restore([[p] for _ in range(4)])==[[],[],[],[]]


def test_cuts_gaps_class_and_geometry_changes_break_confirmation():
    assert restore([[observed()] for _ in range(4)],[[0,2],[2,4]])==[[],[],[],[]]
    for middle in [[],[observed(class_id=2)],[observed(box=[80,10,120,20])],
                   [observed(box=[10,10,100,30])]]:
        assert restore([[observed()],middle,[observed()]])==[[],[],[]]


def test_existing_template_coverage_and_other_detections_remain_unchanged():
    old=dict(box=[5,5,60,25],model_box=[10,10,50,20],kind='prototype',prototype=0)
    result=restore([[observed()] for _ in range(3)],existing=[[dict(old)] for _ in range(3)])
    assert result==[[old],[old],[old]]
    assert not ModelTrackSettings().temporal_text_fallback


def test_continuous_subtitle_without_confirmed_brand_instance_is_rejected():
    assert restore([[observed()] for _ in range(8)],confirmed=False)==[[] for _ in range(8)]
    assert restore([[observed(box=[80,80,120,90])] for _ in range(8)])==[[] for _ in range(8)]


def test_v8_detection_uses_frozen_single_view_detector(tmp_path,monkeypatch):
    from scripts import hls_single_check as entry
    calls=[]
    monkeypatch.setenv('DETECT_PYTHON','模型Python')
    monkeypatch.setattr(entry,'run',lambda args,**kwargs:calls.append(args))
    entry.detect_all(tmp_path,tmp_path/'missing.json',mask_source='v8')
    assert calls[0][1:3]==['-m','scripts.hls_candidates_v7']
    assert not {'--enhance','--tiles','--flips'} & set(calls[0])


def test_confirmed_text_support_only_crops_original_anchor_without_adjacent_icon():
    import numpy as np
    from worker.model_text_confirmation import confirmed_text_support
    mask=np.zeros((40,80),np.uint8)
    mask[:20,:20]=255  # 相邻图标
    mask[25:30,30:60]=255  # 模型已确认文字笔画
    proto=dict(mask=mask,box=[100,200,180,240],anchor=[120,220,170,235])
    result=confirmed_text_support(proto,(50,15))
    expected=np.zeros((15,50),np.uint8);expected[5:10,10:40]=255
    assert np.array_equal(result,expected)
