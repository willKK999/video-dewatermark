"""补漏只学习本输入稳定图形，不能沿镜头传播或突破面积保护。"""
import cv2
import numpy as np
from worker.detection_recall import RecallSettings, connected_support, template_recall
from worker.model_tracks import associate, temporal_bins
from worker.process import PipelineError
import pytest


def test_connected_icon_is_included_but_remote_background_is_excluded():
    rng=np.random.default_rng(72)
    signal=np.zeros((100,200),np.float32)
    cv2.circle(signal,(48,50),13,20,2)
    signal[44:58,70:128]=20
    signal[10:20,175:195]=20
    features=[signal+rng.normal(0,1,signal.shape).astype(np.float32) for _ in range(24)]
    feature,mask=connected_support(features,[70,40,130,62],RecallSettings())
    assert mask[50,48]>0 and mask[50,90]>0
    assert not mask[15,185]
    assert feature.shape==mask.shape


def test_inconsistent_scene_texture_does_not_become_template_support():
    rng=np.random.default_rng(51)
    features=[rng.normal(0,3,(80,140)).astype(np.float32) for _ in range(24)]
    _,mask=connected_support(features,[50,30,90,45],RecallSettings())
    assert np.count_nonzero(mask)/mask.size<.01


def test_multiscale_recall_finds_moved_brand_and_rejects_unrelated_texture():
    rng=np.random.default_rng(22)
    mark=np.zeros((24,50,3),np.uint8)
    cv2.circle(mark,(12,12),8,(220,220,220),-1)
    cv2.putText(mark,'AB',(24,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(230,230,230),1)
    from worker.multitrack import highpass
    proto=dict(id=0,whole_feature=highpass(mark),mask=np.full((24,50),255,np.uint8))
    cfg=RecallSettings(recall_search_ratio=1.,recall_scales=(1.,),recall_correlation=.85)
    image=np.zeros((100,180,3),np.uint8);image[50:74,85:135]=mark
    hits=template_recall(image,[proto],cfg)
    assert len(hits)==1 and hits[0]['box']==[85,50,135,74]
    noise=rng.integers(0,255,image.shape,dtype=np.uint8)
    assert template_recall(noise,[proto],cfg)==[]
    assert template_recall(image,[],cfg)==[]


def test_recall_tracks_restart_at_cut_and_keep_twenty_percent_guard():
    p=dict(box=[10,10,25,20],kind='prototype',prototype=0,confidence=.9,template_recall=True)
    tracks=associate([[p]]*12,[[0,6],[6,12]],30,RecallSettings())
    assert len(tracks)==2
    assert [t['hints'][0]['frame'] for t in tracks]==[0,6]
    with pytest.raises(PipelineError):temporal_bins({1:{0:.201}})
