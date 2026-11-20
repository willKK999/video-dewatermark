"""冻结SAM文字提示内的局部笔画形状；不使用整块文字矩形下限。"""
from dataclasses import dataclass
from difflib import SequenceMatcher
import cv2
import numpy as np
from worker.ocr_temporal import normalize_text


@dataclass(frozen=True)
class StrokeSettings:
    dilation_pixels:int=2
    background_radius_fraction:float=.2
    minimum_contrast:int=12
    minimum_radius:int=3
    text_similarity:float=.6
    block_scene_additions:bool=True
    mask_fraction_limit:float=.2

    def __post_init__(self):
        if self.mask_fraction_limit!=.2 or not self.block_scene_additions:raise ValueError('20%和实物新增像素保护不得放宽')
        if not 0<=self.dilation_pixels<=3 or self.minimum_contrast<=0 or self.background_radius_fraction<=0:raise ValueError('笔画配置无效')


def stroke_support(image,raw,box,cfg):
    h,w=image.shape[:2];x,y,a,b=np.rint(box).astype(int);x,y,a,b=max(0,x),max(0,y),min(w,a),min(h,b)
    result=np.zeros((h,w),np.uint8)
    if a<=x or b<=y:return result
    value=cv2.cvtColor(image[y:b,x:a],cv2.COLOR_BGR2HSV)[:,:,2]
    radius=max(cfg.minimum_radius,round((b-y)*cfg.background_radius_fraction))
    background=cv2.morphologyEx(value,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)))
    # 值域局部凸起与已缓存SAM形状同时支持，避免矩形吞入同色背景。
    core=((value.astype(np.int16)-background)>=cfg.minimum_contrast)&(raw[y:b,x:a]>0)
    result[y:b,x:a]=core.astype(np.uint8)*255
    r=cfg.dilation_pixels
    if r:result=cv2.dilate(result,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
    return result


def same_text_member(text,canonical,cfg):
    a,b=normalize_text(text),normalize_text(canonical)
    return bool(a and b and SequenceMatcher(None,a,b).ratio()>=cfg.text_similarity)


def refine_extra(extra,rectangles,ink,scene,protection):
    return extra & (~rectangles|ink) & ~scene & ~protection
