"""模型可信区与写回域分离；冻结水印并集中的像素不可充当已知背景。"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json
import cv2,numpy as np
from worker.process import PipelineError


@dataclass(frozen=True)
class ContextSettings:
    mask_fraction_limit:float=.2
    strategy:str='stable_union_or_stop'
    model_sample_count:int=3
    retain_stage_sample_count:int=1
    boundary_exclusion_pixels:int=3

    def __post_init__(self):
        if self.mask_fraction_limit!=.2:raise ValueError('20%保护不得放宽')
        if self.strategy!='stable_union_or_stop':raise ValueError('不支持隐式跨层回退')
        if not 1<=self.model_sample_count<=3 or not 0<=self.retain_stage_sample_count<=1:raise ValueError('抽样配置超出磁盘预算')
        if not 0<=self.boundary_exclusion_pixels<=8:raise ValueError('可信区边界配置无效')


def exclusion_mask(confirmed,cfg):
    # 与CleanReferences的供体可信区边界一致；仅模型排除，不改变施工掩膜。
    from worker.restoration import context_mask
    return context_mask(confirmed,cfg.boundary_exclusion_pixels)


def verify_input(expected,actual,construction,frame_pixels,cfg):
    if actual is None or expected is None or construction is None or expected.shape!=actual.shape or actual.shape!=construction.shape or not set(np.unique(actual))<={0,255}:
        raise PipelineError('CONTEXT_MASK_INVALID','模型排除掩膜形状或取值不符')
    missing=int(np.count_nonzero((expected>0)&(actual==0)))
    outside=int(np.count_nonzero((construction>0)&(expected==0)))
    area=int(np.count_nonzero(actual))
    row=dict(已知水印像素=int(np.count_nonzero(expected)),模型排除像素=area,源水印未排除像素=missing,施工域超出确认水印像素=outside,模型掩膜占整帧比例=area/frame_pixels)
    if missing or outside:raise PipelineError('UNTRUSTED_CONTEXT','已知水印未被模型排除，禁止推理',details=row)
    if area/frame_pixels>cfg.mask_fraction_limit:raise PipelineError('MASK_UNCERTAIN','模型排除区超过20%保护，必须重规划',details=row)
    return row


def stable_union_layers(frames,limit=.2):
    """同一固定层包含全生命周期的全部确认像素；不能合并则明确停止。"""
    if limit!=.2:raise ValueError('20%保护不得放宽')
    maximum=0.;count=0
    for mask in frames:
        count+=1
        if mask is None or not set(np.unique(mask))<={0,255}:raise PipelineError('MASK_INVALID','确认水印并集无效')
        maximum=max(maximum,np.count_nonzero(mask)/mask.size)
        if maximum>limit:raise PipelineError('MASK_UNCERTAIN','稳定联合层超过20%，不得迁移部件或拆层绕过可信区保护')
    if not count:raise PipelineError('MASK_INVALID','冻结确认水印输入为空，禁止生成空计划')
    return dict(layers=1,maximum_fraction=maximum,all_components_fixed_layer=0)


def read_confirmed(path):
    mask=cv2.imread(str(path),0)
    if mask is None or mask.ndim!=2 or not set(np.unique(mask))<={0,255}:
        raise PipelineError('MASK_INVALID','确认水印掩膜缺失或无效')
    return mask


def verify_from_confirmed(path,box,actual,construction,frame_pixels,cfg):
    # 独立读取冻结真值并独立形态计算，不接受写入侧 model_mask/exclusion。
    confirmed=read_confirmed(path)
    r=cfg.boundary_exclusion_pixels
    expected=cv2.dilate(confirmed,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
    if np.count_nonzero(expected)/expected.size>cfg.mask_fraction_limit:
        raise PipelineError('MASK_UNCERTAIN','全确认水印排除区超过20%')
    a,b,c,d=box;expected=expected[b:d,a:c]
    expected=cv2.copyMakeBorder(expected,0,(-expected.shape[0])%8,0,(-expected.shape[1])%8,cv2.BORDER_REPLICATE)
    return verify_input(expected,actual,construction,frame_pixels,cfg)


def overlap_weight(frame,start,end,completed):
    overlap=sorted(j for j in completed if start<=j<end)
    return (overlap.index(frame)+1)/(len(overlap)+1) if frame in overlap else 1.
