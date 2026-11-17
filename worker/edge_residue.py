"""确认形状约束的主色细边：不向全局颜色或高亮边搜索。"""
from dataclasses import dataclass
import cv2
import numpy as np
from .process import PipelineError


@dataclass(frozen=True)
class EdgeSettings:
    shape_radius:int=2
    hue_radius:int=12
    minimum_saturation:int=60
    minimum_value:int=60
    minimum_template_pixels:int=32
    template_contrast_ratio:float=2.0
    template_minimum_fraction:float=.1
    template_reference_radius:int=8
    residue_hue_ranges:tuple=((85,130),)
    clean_reference_tolerance:int=0
    maximum_residue_pixels:int=0
    composite_mode:str='confirmed_replace'

    def __post_init__(self):
        if not 0<=self.shape_radius<=3 or not 0<self.hue_radius<=15:raise ValueError('形状收边范围无效')
        if not 1<=self.minimum_saturation<=255 or not 1<=self.minimum_value<=255 or self.minimum_template_pixels<1:raise ValueError('主色配置无效')
        if self.template_contrast_ratio<2 or not 0<self.template_minimum_fraction<=1 or self.template_reference_radius<3:raise ValueError('模板区别背景保护无效')
        if not self.residue_hue_ranges or any(len(x)!=2 or not 0<=x[0]<=x[1]<=179 for x in self.residue_hue_ranges):raise ValueError('复检色相范围无效')
        if self.clean_reference_tolerance!=0:raise ValueError('背景举证必须逐通道精确相等，不得放宽')
        if self.maximum_residue_pixels!=0:raise ValueError('残留门槛不得放宽')
        if self.composite_mode!='confirmed_replace':raise ValueError('禁止将源水印混回确认域')


def highlight_guard(image):
    # 白衣、栏杆和纸张等低饱和高亮实物边界，保护其一像素邻域。
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gradient=cv2.morphologyEx(gray,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    bright=(hsv[...,1]<60)&(hsv[...,2]>=180)&(gradient>=12)
    return cv2.dilate(bright.astype(np.uint8),np.ones((3,3),np.uint8))>0


def color_template(image,mask,cfg):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    eligible=(mask>0)&(hsv[...,1]>=cfg.minimum_saturation)&(hsv[...,2]>=cfg.minimum_value)
    if eligible.sum()<cfg.minimum_template_pixels:return None
    hist=np.bincount(hsv[...,0][eligible],minlength=180)
    smooth=sum(np.roll(hist,k) for k in range(-cfg.hue_radius,cfg.hue_radius+1))
    return int(np.argmax(smooth))


def matches(image,hue,cfg):
    if hue is None:return np.zeros(image.shape[:2],bool)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);delta=np.abs(hsv[...,0].astype(int)-hue)
    return (np.minimum(delta,180-delta)<=cfg.hue_radius)&(hsv[...,1]>=cfg.minimum_saturation)&(hsv[...,2]>=cfg.minimum_value)


def refine_group(image,group,confirmed,protected,cfg):
    """只在本组形状的窄带寻找自身主色，禁跨越场景/字幕/高亮保护。"""
    if group.shape!=confirmed.shape or protected.shape!=confirmed.shape:raise PipelineError('MASK_INVALID','收边几何不一致')
    hue=color_template(image,group & ~protected,cfg)
    if hue is not None:
        colored=matches(image,hue,cfg);radius=cfg.template_reference_radius
        outer=(cv2.dilate(group,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0)&(group==0)&(protected==0)
        inner=(group>0)&(protected==0)
        inside=float(colored[inner].mean()) if inner.any() else 0.
        outside=float(colored[outer].mean()) if outer.any() else 1.
        if inside<cfg.template_minimum_fraction or inside<cfg.template_contrast_ratio*max(outside,1/max(1,int(outer.sum()))):hue=None
    r=cfg.shape_radius
    shape=cv2.dilate(group,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))>0
    candidate=shape&(confirmed==0)&matches(image,hue,cfg)
    guard=(protected>0)|highlight_guard(image)
    added=candidate&~guard
    if np.any(added&guard):raise PipelineError('MASK_UNCERTAIN','收边侵入保护区')
    return added.astype(np.uint8)*255,dict(hue=hue,候选像素=int(candidate.sum()),新增像素=int(added.sum()),保护拒绝像素=int((candidate&guard).sum()),保护交集像素=int((added&guard).sum()))


def composite(original,repaired,mask):
    # 已确认水印的内侧边缘也不可复用原水印颜色。施工域外由调用方逐像素保持。
    result=original.copy();result[mask>0]=repaired[mask>0];return result


def measure(source,painted,mask,band,cfg,reference_evidence=None):
    """全施工域固定颜色段复检；同色背景只能由至少三份可信源参考举证。"""
    if source is None or painted is None or mask is None or band is None:
        raise PipelineError('RESIDUE_INPUT_INVALID','残留复检输入缺失')
    if source.ndim!=3 or source.shape[2]!=3 or painted.shape!=source.shape or mask.ndim!=2 or band.shape!=mask.shape or mask.shape!=source.shape[:2]:
        raise PipelineError('RESIDUE_INPUT_INVALID','残留复检要求对齐的三通道BGR和二维掩膜')
    if source.dtype!=np.uint8 or painted.dtype!=np.uint8 or mask.dtype!=np.uint8 or band.dtype!=np.uint8 or not set(np.unique(mask))<={0,255} or not set(np.unique(band))<={0,255}:
        raise PipelineError('RESIDUE_INPUT_INVALID','残留复检输入类型或掩膜取值无效')
    if not np.any(mask):raise PipelineError('RESIDUE_INPUT_INVALID','残留复检施工样本为空，禁止按零残留通过')
    if np.any((band>0)&(mask==0)):raise PipelineError('RESIDUE_INPUT_INVALID','收边带超出当前施工域')
    hsv=cv2.cvtColor(painted,cv2.COLOR_BGR2HSV);hues=np.zeros(mask.shape,bool)
    for lo,hi in cfg.residue_hue_ranges:hues|=(hsv[...,0]>=lo)&(hsv[...,0]<=hi)
    colored=hues&(hsv[...,1]>=cfg.minimum_saturation)&(hsv[...,2]>=cfg.minimum_value)
    proved=np.zeros(mask.shape,bool)
    if reference_evidence is not None:
        if not isinstance(reference_evidence,dict) or not {'background','confidence','counts'}<=reference_evidence.keys():
            raise PipelineError('RESIDUE_INPUT_INVALID','干净参考证据字段缺失')
        background=reference_evidence['background'];confidence=reference_evidence['confidence'];counts=reference_evidence['counts']
        if background.shape!=source.shape or confidence.shape!=mask.shape or counts.shape!=mask.shape or background.dtype!=np.uint8 or confidence.dtype!=np.uint8 or counts.dtype!=np.uint8:
            raise PipelineError('RESIDUE_INPUT_INVALID','干净参考证据未与施工裁块对齐')
        proved=(confidence>0)&(counts>=3)&(np.abs(painted.astype(np.int16)-background.astype(np.int16)).max(axis=2)<=cfg.clean_reference_tolerance)
    uncertain=colored&~proved;inner=int(np.count_nonzero(uncertain&(mask>0)));edge=int(np.count_nonzero(uncertain&(band>0)))
    sx=cv2.cvtColor(source,cv2.COLOR_BGR2HSV);source_hues=np.zeros(mask.shape,bool)
    for lo,hi in cfg.residue_hue_ranges:source_hues|=(sx[...,0]>=lo)&(sx[...,0]<=hi)
    source_colored=source_hues&(sx[...,1]>=cfg.minimum_saturation)&(sx[...,2]>=cfg.minimum_value)
    return dict(源片同色像素=int(np.count_nonzero(source_colored&(mask>0))),颜色范围=[list(x) for x in cfg.residue_hue_ranges],有效施工像素=int(np.count_nonzero(mask)),边缘带像素=int(np.count_nonzero(band)),统计形状=list(painted.shape),
                遮罩内颜色候选=int(np.count_nonzero(colored&(mask>0))),收边带颜色候选=int(np.count_nonzero(colored&(band>0))),
                干净参考证实的背景候选=int(np.count_nonzero(colored&proved&(mask>0))),遮罩内残留=inner,收边带残留=edge,
                阈值=cfg.maximum_residue_pixels,阈值单位='像素/施工帧',通过=max(inner,edge)<=cfg.maximum_residue_pixels,
                口径='残留字段为未获可信背景证明的颜色候选，不等同已确认水印真值；同色实物及未消除水印均保留原始颜色数。')
