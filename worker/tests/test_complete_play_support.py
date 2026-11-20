"""没有独立检测框的播放图形由通用几何确认，完整内部也必须被遮住。"""
import cv2,numpy as np
from worker.branding import complete_play_support


def test_dark_play_icon_is_completed_near_automatic_text_track():
    image=np.full((240,420,3),180,np.uint8)
    cv2.circle(image,(300,120),45,(230,230,230),-1)
    cv2.fillPoly(image,[np.array([[286,96],[286,144],[326,120]],np.int32)],(20,20,20))
    mask,evidence=complete_play_support(image,[90,70,345,175])
    assert evidence and mask[120,300] and mask[100,315]
    assert not mask[20,20]


def test_plain_circle_or_distant_play_icon_is_not_assumed_to_be_the_known_brand():
    image=np.full((240,420,3),180,np.uint8)
    cv2.circle(image,(300,120),45,(230,230,230),-1)
    mask,evidence=complete_play_support(image,[90,70,345,175])
    assert not evidence and not mask.any()
