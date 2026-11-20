"""v11通用行为验证，全部合成输入，不使用影片特例。"""
from dataclasses import replace
import numpy as np
import pytest
from worker.v11_masks import (Settings,canonical_tracks,strict_subtract,rectangle,shape_metrics,
    decide_shape,ocr_protection,group_nodes,background_matches)


def obs(text,box):return dict(source='OCR',text=text,box=box,confidence=.99)


def test_variants_normalized_before_stability_and_propagation():
    cfg=replace(Settings(),fps=10,stride=2,canonical_min_observations=10,canonical_min_seconds=2.,propagation_min_seconds=1.)
    rows={str(i):[obs('测试品牌' if i%6 else '测试品牌A',[20+i,30,100+i,50])] for i in range(2,50,2)}
    # 长稳定同内容区域提供共识，移动副本包含识别变体。
    for i in range(0,60,2):rows.setdefault(str(i),[]).append(obs('测试品牌',[200,10,280,30]))
    tracks,_,audit=canonical_tracks(rows,[(0,60)],cfg)
    moving=next(t for t in tracks if t['observations'][0]['box'][0]<100)
    assert moving['hints'][0]['frame']==0
    assert moving['hints'][-1]['frame']>48
    assert all(0<=h['frame']<60 for h in moving['hints'])
    assert all(h['text']=='测试品牌' for h in moving['hints'])


def test_subtitles_override_floor_and_mixed_graphic():
    shape=(100,140);raw=rectangle([10,20,120,80],shape)
    floor=rectangle([20,50,100,70],shape)
    subtitle=rectangle([0,60,130,75],shape)
    final=strict_subtract(raw|floor,subtitle)
    assert not np.any(final&subtitle)
    assert final[30,30] and final[55,55]


def test_ocr_motion_relation_protects_transparent_icon():
    cfg=replace(Settings(),flow_scale=1.)
    samples={i:rectangle([10+i,20,30+i,40],(100,200)) for i in [0,5,10,15]}
    track=dict(id=1,hints=[dict(frame=i,box=[35+i,20,100+i,40]) for i in samples])
    assert ocr_protection(samples,[track],cfg)
    track['hints']=[dict(frame=i,box=[130-i*4,20,190-i*4,40]) for i in samples]
    assert not ocr_protection(samples,[track],cfg)


def pair(**changes):
    p=dict(valid=True,screen_iou=.2,template_iou=.5,background_edges=.9,background_containment=.95,
           background_motion=4.,occlusion_fraction=.3,occlusion_component=.2,occlusion_cover_fraction=.8,
           occlusion_foreground_points=0,acceleration=0.)
    p.update(changes);return p


def test_background_alignment_alone_cannot_delete():
    cfg=Settings()
    assert not decide_shape([pair(occlusion_cover_fraction=0.)]*6,[],cfg)['rejected']
    assert not decide_shape([pair(occlusion_fraction=0.)]*6,[],cfg)['rejected']
    assert decide_shape([pair()]*6,[],cfg)['rejected']
    assert not decide_shape([pair()]*6,[dict(track=1)],cfg)['rejected']


def test_screen_and_independent_smooth_template_protected():
    assert not decide_shape([pair(screen_iou=.95)]*6,[],Settings())['rejected']
    assert not decide_shape([pair(template_iou=.95,background_edges=.2)]*6,[],Settings())['rejected']


def test_shape_warp_uses_geometry_and_occluder():
    cfg=replace(Settings(),flow_scale=1.)
    a=rectangle([15,20,55,60],(100,120));matrix=np.array([[1.,0,5.],[0,1,0.]])
    b=rectangle([20,20,60,60],a.shape);cut=rectangle([40,20,60,50],a.shape);b&=~cut
    p=shape_metrics(a,b,matrix,a,matrix,cfg,occluders=cut)
    assert p['background_containment']>.9 and p['occlusion_fraction']>.1
    assert p['occlusion_cover_fraction']>.9
    assert 'inner_error' not in p


def test_cross_window_links_same_scene_only():
    cfg=replace(Settings(),flow_scale=1.)
    m=rectangle([10,10,30,30],(70,80))
    nodes={str(k):dict(scene=scene,frames=indices,samples={i:m for i in indices},first=m,last=m)
           for k,(scene,indices) in enumerate([(0,[0,5]),(0,[6,10]),(1,[11,15])])}
    groups,links=group_nodes(nodes,cfg)
    assert sorted(map(len,groups))==[1,2] and len(links)==1


def test_no_dialogue_promoted_by_short_lifecycle():
    rows={str(i):[obs('对白'+str(i//8),[20,20,100,40])] for i in range(0,60,2)}
    t,sub,_=canonical_tracks(rows,[(0,60)],replace(Settings(),fps=10,stride=2))
    assert not t and sub


def test_background_features_exclude_internal_pixels():
    cfg=replace(Settings(),background_feature_distance=2)
    rng=np.random.default_rng(5);im=rng.integers(0,255,(100,120),dtype=np.uint8)
    mask=rectangle([25,25,85,80],im.shape)
    x,y=background_matches(im,im,mask,mask,cfg)
    assert len(x)>0
    coords=np.rint(x).astype(int)
    assert not np.any(mask[coords[:,1],coords[:,0]])


def test_area_and_tight_box_config():
    with pytest.raises(ValueError):Settings(mask_fraction_limit=.21)
    with pytest.raises(ValueError):Settings(text_padding=5)
    with pytest.raises(ValueError):Settings(background_pair_fraction=0.)


def test_high_adjacent_overlap_does_not_mean_fixed_screen_position():
    evidence=[pair(screen_iou=.95,screen_position_delta=20.)]*6
    assert decide_shape(evidence,[],Settings())['rejected']
