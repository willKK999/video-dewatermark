"""独立品牌小字须完整覆盖，不能把肤色或框外字幕纳入。"""
import cv2,numpy as np
from worker.branding import opaque_brand_ink


def test_disconnected_opaque_footer_is_selected_without_a_sam_object():
    image=np.full((180,360,3),(150,180,200),np.uint8)
    cv2.putText(image,'Brand',(30,70),0,1.4,(255,255,255),3)
    cv2.putText(image,'example.com',(25,112),0,.7,(255,255,255),2)
    cv2.putText(image,'caption',(25,165),0,.8,(255,255,255),2)
    palette=cv2.cvtColor(np.array([[[255,255,255],[0,220,250]]],np.uint8),cv2.COLOR_BGR2LAB)[0]
    mask=opaque_brand_ink(image,[15,20,320,125],palette)
    assert np.all(mask[(np.min(image,axis=2)>245)&(np.indices(mask.shape)[0]<125)]>0)
    assert not mask[150:180].any() and not mask[30,300]


def test_light_neutral_skin_is_not_opaque_white_ink():
    image=np.full((100,200,3),(215,222,230),np.uint8)
    palette=cv2.cvtColor(np.array([[[255,255,255]]],np.uint8),cv2.COLOR_BGR2LAB)[0]
    assert not opaque_brand_ink(image,[10,10,190,90],palette).any()


def test_verification_covers_footer_that_was_not_in_the_sam_object():
    from scripts.hls_brand_verify import graphic_support
    from worker.multitrack import component_scores,highpass
    image=np.full((180,360,3),100,np.uint8)
    cv2.putText(image,'Brand',(25,70),0,1.4,(255,255,255),3)
    cv2.putText(image,'example.com',(25,125),0,1,(255,255,255),2)
    sam=np.zeros(image.shape[:2],np.uint8);sam[:85]=255
    palette=cv2.cvtColor(np.array([[[255,255,255]]],np.uint8),cv2.COLOR_BGR2LAB)[0]
    extra=opaque_brand_ink(image,[15,20,320,140],palette)
    mask=graphic_support(image,sam,palette,extra)
    result=image.copy();result[:85]=100
    scores=component_scores(highpass(image),highpass(result),mask)
    assert mask[110:130].any()
    assert any(abs(p['correlation'])>=.55 and abs(p['relative_amplitude'])>=.15 for p in scores)
