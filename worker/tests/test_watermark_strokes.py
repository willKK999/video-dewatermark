import cv2
import numpy as np
import pytest
from worker.watermark_strokes import StrokeSettings,stroke_support,refine_extra,same_text_member


def test_cached_rectangle_is_reduced_to_text_strokes():
    image=np.full((120,180,3),80,np.uint8)
    cv2.putText(image,'AB',(30,75),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,220,0),3)
    raw=np.full(image.shape[:2],255,np.uint8)
    result=stroke_support(image,raw,[20,30,130,90],StrokeSettings())
    assert result[55,35]>0 and result[35,120]==0
    assert 100<np.count_nonzero(result)<2000


def test_all_new_scene_overlap_is_removed_even_inside_stroke():
    extra=np.full((30,40),255,np.uint8);rect=extra.copy();ink=extra.copy();scene=np.zeros_like(extra);scene[8:15,10:20]=255
    subtitle=np.zeros_like(extra);subtitle[20:25]=255
    result=refine_extra(extra,rect,ink,scene,subtitle)
    assert not np.any(result&scene) and not np.any(result&subtitle)
    assert result[0,0]==255


def test_small_dilation_does_not_fill_letter_holes_or_entire_box():
    image=np.full((100,160,3),50,np.uint8);cv2.putText(image,'O',(40,75),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
    result=stroke_support(image,np.full((100,160),255,np.uint8),[30,20,110,85],StrokeSettings())
    assert result[50,63]==0 and np.any(result)


def test_dialogue_cannot_be_reclassified_by_position_alone():
    cfg=StrokeSettings()
    assert same_text_member('牌','品牌',cfg)
    assert not same_text_member('他说了很多遍','品牌',cfg)


def test_guards_cannot_be_relaxed():
    with pytest.raises(ValueError):StrokeSettings(dilation_pixels=8)
    with pytest.raises(ValueError):StrokeSettings(block_scene_additions=False)
