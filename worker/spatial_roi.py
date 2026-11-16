"""按空间连通区域和像素帧预算规划独立施工域；不改变输入掩膜。"""
from dataclasses import dataclass,replace
import math
import cv2
import numpy as np
from worker import roi_inpaint as roi


@dataclass(frozen=True)
class SpatialSettings:
    dilation_pixels:int=6
    merge_gap_pixels:int=24
    maximum_crop_pixels:int=708480
    maximum_pixel_frames:int=90685440
    minimum_width:int=512
    minimum_height:int=512
    fixed_context_pixels:int=96

    def __post_init__(self):
        if min(self.dilation_pixels,self.merge_gap_pixels,self.minimum_width,self.minimum_height,self.fixed_context_pixels)<0 or min(self.maximum_crop_pixels,self.maximum_pixel_frames)<=0:raise ValueError('空间ROI配置无效')


def padded_area(box):
    a,b,c,d=box;return math.ceil((c-a)/8)*8*math.ceil((d-b)/8)*8


def context_box(rect,shape,cfg,settings):
    box=roi.padded_box(rect,shape,replace(cfg,minimum_context=max(cfg.minimum_context,settings.fixed_context_pixels)))
    h,w=shape
    def expand(a,b,minimum,bound):
        size=min(bound,max(b-a,math.ceil(minimum/8)*8))
        first=max(0,min(math.floor(((a+b-size)/2)/8)*8,bound-size))
        return first,min(bound,first+size)
    x,a=expand(box[0],box[2],settings.minimum_width,w);y,b=expand(box[1],box[3],settings.minimum_height,h)
    return [x,y,a,b]


def intersects(a,b):return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])


def plan(footprint,frames,cfg,settings):
    limit=min(settings.maximum_crop_pixels,settings.maximum_pixel_frames//frames)
    if limit<1:raise ValueError('像素帧预算不足')
    radius=settings.dilation_pixels
    grown=cv2.dilate(footprint,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))) if radius else footprint
    count,labels,_,_=cv2.connectedComponentsWithStats((grown>0).astype(np.uint8));parts=[]
    for k in range(1,count):
        support=((labels==k)&(footprint>0)).astype(np.uint8)*255
        if np.any(support):parts.append(support)
    def gap(a,b):
        return max(0,max(a[0],b[0])-min(a[2],b[2]),max(a[1],b[1])-min(a[3],b[3]))
    # 只合并近邻，而且合并后的上下文裁块必须满足预算。
    changed=True
    while changed:
        changed=False
        for i in range(len(parts)):
            for j in range(i+1,len(parts)):
                a,b=roi.bbox(parts[i]),roi.bbox(parts[j])
                if gap(a,b)>settings.merge_gap_pixels and not intersects(context_box(a,footprint.shape,cfg,settings),context_box(b,footprint.shape,cfg,settings)):continue
                joined=parts[i]|parts[j];box=context_box(roi.bbox(joined),footprint.shape,cfg,settings)
                if padded_area(box)<=limit:
                    parts[i]=joined;parts.pop(j);changed=True;break
            if changed:break
    result=[]
    def split(support):
        rect=roi.bbox(support);box=context_box(rect,footprint.shape,cfg,settings)
        if padded_area(box)<=limit:
            result.append(dict(box=box,ownership=support));return
        a,b,c,d=rect
        if c-a<=1 and d-b<=1:raise ValueError('保留既有最小上下文时无法满足显存预算')
        first=support.copy();second=support.copy()
        if c-a>=d-b and c-a>1:
            middle=(a+c)//2;first[:,middle:]=0;second[:,:middle]=0
        else:
            middle=(b+d)//2;first[middle:]=0;second[:middle]=0
        for item in [first,second]:
            if np.any(item):split(item)
    for support in parts:split(support)
    # 面积限制导致拆分后，仍在预算内的扩展重叠块再次尽量合并。
    changed=True
    while changed:
        changed=False
        for i in range(len(result)):
            for j in range(i+1,len(result)):
                if not intersects(result[i]['box'],result[j]['box']) and gap(roi.bbox(result[i]['ownership']),roi.bbox(result[j]['ownership']))>settings.merge_gap_pixels:continue
                ownership=result[i]['ownership']|result[j]['ownership'];box=context_box(roi.bbox(ownership),footprint.shape,cfg,settings)
                if padded_area(box)<=limit:
                    result[i]=dict(box=box,ownership=ownership);result.pop(j);changed=True;break
            if changed:break
    union=np.zeros_like(footprint)
    for row in result:
        if np.any(union&row['ownership']):raise ValueError('施工域重复分配')
        union|=row['ownership']
    if not np.array_equal(union,footprint):raise ValueError('空间切块遗漏掩膜像素')
    return result
