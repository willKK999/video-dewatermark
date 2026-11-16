"""v12叠加层保护和无OCR字幕保护的合成行为测试。"""
from dataclasses import replace
import cv2
import numpy as np
import pytest
from worker.v12_masks import (Settings,decide_shape,strict_ocr_relation,detect_text_rows,
                             associate_subtitle_rows,rectangle,strict_subtract)


def evidence(**updates):
    p=dict(valid=True,background_motion=5.,background_edges=.95,background_containment=.95,
           background_center_error=.5,background_center_tolerance=2.,occlusion_fraction=.3,
           occlusion_component=.2,occlusion_cover_fraction=.9,occlusion_foreground_points=0,
           occlusion_template_iou=.7,template_iou=.95,acceleration=0.,screen_iou=.5,screen_position_delta=100.)
    p.update(updates);return p


def test_ocr_nearby_text_cannot_override_scene_and_occlusion():
    result=decide_shape([evidence()]*8,[dict(track=1)],Settings())
    assert result['rejected'] and not result['ocr_protected']


def test_stable_shape_not_enough_when_following_scene():
    result=decide_shape([evidence(template_iou=.99)]*8,[dict(track=1)],Settings())
    assert result['rejected'] and not result['overlay_evidence']


def test_false_cumulative_occlusion_of_unchanged_text_does_not_delete():
    result=decide_shape([evidence(occlusion_template_iou=.99)]*8,[dict(track=1)],Settings())
    assert not result['rejected'] and result['occlusion_samples']==0


def test_independent_stable_translucent_shape_can_be_protected():
    result=decide_shape([evidence(background_center_error=8.,occlusion_template_iou=.99)]*8,[dict(track=1)],Settings())
    assert result['ocr_protected'] and result['own_motion_protected'] and not result['rejected']


def test_fixed_screen_overlay_protected():
    result=decide_shape([evidence(screen_position_delta=0.,screen_iou=.99,background_center_error=10.)]*8,[dict(track=1)],Settings())
    assert result['ocr_protected'] and result['screen_protected']


def test_uncertain_nearby_ocr_is_not_protection():
    result=decide_shape([evidence(occlusion_fraction=0.,template_iou=.2)]*8,[dict(track=1)],Settings())
    assert not result['ocr_protected'] and not result['rejected']


def test_strict_relative_motion_rejects_accidental_neighborhood():
    cfg=replace(Settings(),flow_scale=1.,ocr_comotion_seconds=.3)
    samples={i:rectangle([40+i,40,60+i,60],(180,320)) for i in [0,5,10,15]}
    track=dict(id=1,hints=[dict(frame=i,box=[90,30,260,100]) for i in samples])
    assert not strict_ocr_relation(samples,[track],cfg)
    track['hints']=[dict(frame=i,box=[90+i,30,260+i,100]) for i in samples]
    assert strict_ocr_relation(samples,[track],cfg)


def caption_image(background=135,color=(245,245,245),outline=True):
    im=np.full((480,720,3),background,np.uint8)
    for text,y in [('SCHOOL IS CLOSED',350),('AFTER THE CLASS',382)]:
        scale=.8;size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0]
        origin=((720-size[0])//2,y)
        if outline:cv2.putText(im,text,origin,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),5,cv2.LINE_AA)
        cv2.putText(im,text,origin,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    return im


def test_caption_rows_detect_without_any_ocr():
    cfg=Settings();rows=detect_text_rows(caption_image(),cfg)
    assert len(rows)>=2
    detections={i:rows for i in range(20)}
    tracks,rejected=associate_subtitle_rows(detections,[(0,30)],(480,720),cfg)
    assert len(tracks)>=2
    assert all(t['rows'][0]['frame']==0 and t['rows'][-1]['frame']==19 for t in tracks)


def test_low_contrast_transparent_brand_not_caption():
    assert not detect_text_rows(caption_image(color=(175,175,175),outline=False),Settings())


def test_colored_logo_not_white_caption():
    assert not detect_text_rows(caption_image(color=(255,170,0)),Settings())


def test_persistent_printed_text_not_typical_dialogue():
    cfg=replace(Settings(),fps=10,subtitle_max_seconds=2.)
    rows=detect_text_rows(caption_image(),cfg)
    tracks,_=associate_subtitle_rows({i:rows for i in range(40)},[(0,40)],(480,720),cfg)
    assert not tracks


def test_short_flash_not_dialogue():
    rows=detect_text_rows(caption_image(),Settings())
    tracks,_=associate_subtitle_rows({0:rows,1:rows},[(0,30)],(480,720),Settings())
    assert not tracks


def test_independent_caption_priority_over_watermark_without_ocr():
    cfg=Settings();rows=detect_text_rows(caption_image(),cfg)
    tracks,_=associate_subtitle_rows({i:rows for i in range(20)},[(0,30)],(480,720),cfg)
    protection=np.zeros((480,720),np.uint8)
    for t in tracks:protection|=rectangle(t['box'],protection.shape,cfg.subtitle_region_padding)
    watermark=rectangle([200,330,520,400],protection.shape)
    result=strict_subtract(watermark,protection)
    assert np.any(protection) and not np.any(result&protection)


def test_no_scene_crossing_of_subtitle_lifetime():
    cfg=replace(Settings(),fps=10,subtitle_min_seconds=.4)
    rows=detect_text_rows(caption_image(),cfg)
    tracks,_=associate_subtitle_rows({i:rows for i in range(6)},[(0,3),(3,6)],(480,720),cfg)
    assert not tracks


def test_limits_not_relaxed():
    with pytest.raises(ValueError):Settings(mask_fraction_limit=.21)
    with pytest.raises(ValueError):Settings(overlap_mae_limit=36.)
    with pytest.raises(ValueError):Settings(subtitle_roi_top=.95)


def test_measured_geometric_independence_not_internal_flow():
    from worker.v12_masks import shape_metrics
    cfg=replace(Settings(),flow_scale=1.)
    first=rectangle([20,20,80,40],(100,140));second=rectangle([24,20,84,40],first.shape)
    background=np.array([[1.,0.,-5.],[0.,1.,0.]])
    result=shape_metrics(first,second,background,first,background,cfg)
    assert result['background_center_error']==9.
    assert result['occlusion_template_iou']>.99
    assert 'inner_error' not in result


def test_audit_writer_handles_numpy_scalars_without_changing_values(tmp_path):
    import json
    from worker.v12_masks import write_json
    path=tmp_path/'audit.json'
    write_json(path,dict(flag=np.bool_(False),count=np.int64(3),error=np.float64(1.25),nested=[np.bool_(True)]))
    assert json.loads(path.read_text())==dict(flag=False,count=3,error=1.25,nested=[True])
