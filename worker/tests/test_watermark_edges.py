from dataclasses import replace
import cv2,numpy as np,pytest
from worker.watermark_edges import EdgeSettings,proposal,radius,align


def example():
    image=np.full((160,240,3),(45,75,100),np.uint8);mask=np.zeros((160,240),np.uint8)
    cv2.circle(image,(85,80),36,(245,180,15),-1);cv2.circle(mask,(85,80),30,255,-1)
    return image,mask


def test_missing_colored_outer_ring_is_recovered_without_uniform_growth():
    image,seed=example();added,info=proposal(image,seed,np.zeros_like(seed),EdgeSettings())
    assert np.count_nonzero(added)>100 and added[80,120]>0
    assert added[80,140]==0 and not np.any(added&seed)


def test_scene_and_subtitle_guards_always_win():
    image,seed=example();guard=np.zeros_like(seed);guard[:,110:]=255
    added,_=proposal(image,seed,guard,EdgeSettings())
    assert not np.any(added&guard)


def test_radius_tracks_component_scale_and_frame_cap():
    _,seed=example();large=cv2.resize(seed,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    cfg=EdgeSettings();assert radius(large,cfg)>radius(seed,cfg)
    assert radius(seed,cfg,motion=10000)<=np.ceil(min(seed.shape)*cfg.maximum_radius_frame_fraction)


def test_temporal_alignment_uses_candidate_motion_and_scale():
    seed=np.zeros((100,150),np.uint8);seed[20:40,10:30]=255
    result=align(seed,[10,20,30,40],[50,40,90,80],seed.shape)
    assert result[50,60]==255 and result[25,15]==0


def test_no_palette_does_not_expand_flat_background():
    image=np.full((120,180,3),90,np.uint8);seed=np.zeros((120,180),np.uint8);seed[40:80,50:90]=255
    added,_=proposal(image,seed,np.zeros_like(seed),EdgeSettings())
    assert not np.any(added)


def test_area_limit_cannot_be_relaxed():
    with pytest.raises(ValueError):EdgeSettings(mask_fraction_limit=.21)


def test_translucent_white_text_is_not_calibrated_as_brown_foreground():
    background=np.full((120,200,3),(30,60,90),np.uint8);foreground=background.copy()
    cv2.putText(foreground,'AB',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)
    image=cv2.addWeighted(foreground,.3,background,.7,0);seed=np.zeros((120,200),np.uint8);seed[42:83,28:60]=255
    added,info=proposal(image,seed,np.zeros_like(seed),EdgeSettings(),[[28,40,70,85]])
    assert [255.,255.,255.] in info['palette']
    assert all(p==[255.,255.,255.] or max(p)-min(p)>150 for p in info['palette'])
    assert np.count_nonzero(added)>0


def test_background_motion_rejects_only_when_it_beats_own_alignment():
    from worker.watermark_edges import reject_background
    candidate=np.zeros((60,80),np.uint8);candidate[20:30,20:40]=255
    empty=np.zeros_like(candidate)
    result,n=reject_background(candidate,[empty],[candidate],EdgeSettings())
    assert n==200 and not result.any()
    result,n=reject_background(candidate,[candidate],[candidate],EdgeSettings())
    assert n==0 and np.array_equal(result,candidate)
    result,n=reject_background(candidate,[empty],[],EdgeSettings())
    assert n==0 and np.array_equal(result,candidate)


def test_tiny_isolated_seed_does_not_inherit_large_component_radius():
    image,seed=example();seed[130,190]=255
    cv2.line(image,(180,130),(205,130),(245,180,15),3)
    added,_=proposal(image,seed,np.zeros_like(seed),EdgeSettings())
    assert not added[120:140,175:215].any()


def test_background_registration_uses_only_excluded_mask_complement():
    from worker.watermark_edges import background_transform
    rng=np.random.default_rng(13);image=rng.integers(0,256,(160,240,3),dtype=np.uint8)
    moved=cv2.warpAffine(image,np.float32([[1,0,8],[0,1,4]]),(240,160))
    excluded=np.zeros((160,240),np.uint8);excluded[40:120,60:160]=255
    matrix=background_transform(image,moved,excluded,EdgeSettings())
    assert matrix is not None and np.allclose(matrix[:,2],[8,4],atol=1)
