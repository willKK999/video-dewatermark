"""通用规则行为测试，不使用影片坐标或轨迹特例。"""
from dataclasses import replace
import numpy as np
import pytest
from worker.v10_masks import Settings, rectangle, text_tracks, subtract_subtitles, scene_object, temporal_fill
from worker.ocr_temporal import guarded_mask


def observation(text,box):
    return dict(source='OCR',text=text,box=box,confidence=.99)


def test_changing_dialogue_not_brand():
    cfg=replace(Settings(),fps=10,stride=2)
    rows={str(i):[observation('句子'+str(i//8),[20,40,100,50])] for i in range(0,60,2)}
    tracks,subs,audit=text_tracks(rows,[(0,60)],cfg)
    assert not tracks and subs


def test_stable_brand_and_moving_copy():
    cfg=replace(Settings(),fps=10,stride=2,canonical_min_observations=10)
    rows={str(i):[observation('测试品牌',[10,10,40,20]),observation('测试品牌',[100+i,70,140+i,80])] for i in range(0,60,2)}
    tracks,subs,_=text_tracks(rows,[(0,60)],cfg)
    assert len(tracks)==2 and not subs
    assert any(h['temporal'] for t in tracks for h in t['hints'])


def test_subtitle_only_subtract_overlap():
    m=np.zeros((80,120),np.uint8);m[10:20,10:40]=255;m[50:60,20:80]=255
    r=rectangle([15,48,85,62],m.shape)
    kept,removed=subtract_subtitles(m,r,Settings())
    assert np.array_equal(kept[10:20,10:40],m[10:20,10:40])
    assert not kept[50:60,20:80].any() and removed.any()
    assert np.array_equal(kept|removed,m)


def test_scene_motion_and_screen_overlay():
    p=dict(background_motion=4.,centroid_error=.2,inner_error=.4,shrink=0.)
    assert scene_object([p]*6,Settings())['rejected']
    assert not scene_object([dict(p,centroid_error=4.)]*6,Settings())['rejected']
    assert not scene_object([dict(p,background_motion=0.)]*6,Settings())['rejected']


def test_occlusion_requires_motion_consistency():
    p=dict(background_motion=3.,centroid_error=.2,inner_error=.2,shrink=.5)
    bad=dict(p,centroid_error=10.,shrink=0.)
    assert scene_object([p]*3+[bad]*3,Settings())['rejected']
    assert not scene_object([p]+[bad]*5,Settings())['rejected']


def test_temporal_no_scene_crossing_or_extrapolation():
    cfg=replace(Settings(),fps=10,temporal_gap_seconds=.3)
    mask=np.ones((8,8),np.uint8)*255
    assert set(temporal_fill({0:mask,3:mask,5:mask},[(0,4),(4,8)],cfg))=={1,2}
    assert not temporal_fill({0:mask,5:mask},[(0,8)],cfg)


def test_twenty_percent_not_relaxed():
    with pytest.raises(ValueError): Settings(mask_fraction_limit=.21)
    with pytest.raises(ValueError): Settings(ocr_device='cpu')
    with pytest.raises(ValueError): guarded_mask(np.ones((10,10),np.uint8)*255)


def test_short_candidate_not_kept():
    rows={'0':[observation('临时标题',[10,10,40,20])], '5':[observation('临时标题',[10,10,40,20])]}
    tracks,_,_=text_tracks(rows,[(0,20)],Settings())
    assert not tracks


def test_rectangle_clips_bounds():
    m=rectangle([-2,-2,4,5],(10,10),2)
    assert np.count_nonzero(m)==42


def test_measured_flow_distinguishes_moving_material_from_screen_overlay():
    from worker.v10_masks import flow_evidence
    cfg=replace(Settings(),flow_scale=1.,flow_min_pixels=4)
    a=rectangle([20,20,30,30],(80,100));b=rectangle([24,20,34,30],a.shape)
    flow=np.zeros((80,100,2),np.float32);flow[:,:,0]=4
    material=flow_evidence(a,b,flow,cfg)
    overlay=flow_evidence(a,a,flow,cfg)
    assert material['centroid_error'] < 1e-6
    assert overlay['centroid_error'] == 4.
