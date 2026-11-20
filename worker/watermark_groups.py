"""内容绑定的水印成员联动关系与动画窗口；无影片专用规则。"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GroupSettings:
    minimum_samples:int=3
    minimum_seconds:float=.25
    maximum_neighbor_size:float=3.
    offset_residual:float=.30
    size_residual:float=.25
    shared_fraction:float=.65
    fixed_pixels:float=3.
    fixed_size_fraction:float=.05
    scale_correlation:float=.75
    motion_correlation:float=.75
    minimum_scale_change:float=.12
    minimum_motion_pixels:float=5.
    propagation_seconds:float=.4
    propagation_template_correlation:float=.25
    subtitle_member_overlap:float=.65
    mask_fraction_limit:float=.2

    def __post_init__(self):
        if self.mask_fraction_limit!=.2:raise ValueError('每层20%保护不得放宽')
        if self.minimum_samples<3 or self.minimum_seconds<=0:raise ValueError('成员关联证据不足')


def geometry(box):
    b=np.array(box,float);size=np.maximum(1,b[2:]-b[:2])
    return (b[:2]+b[2:])/2,size,float(np.sqrt(np.prod(size)))


def relation(first,second,fps,cfg):
    common=sorted(set(first)&set(second))
    if len(common)<cfg.minimum_samples or (common[-1]-common[0])/fps<cfg.minimum_seconds:return None
    a=np.array([first[i] for i in common]);b=np.array([second[i] for i in common])
    ac=(a[:,:2]+a[:,2:])/2;bc=(b[:,:2]+b[:,2:])/2
    az=np.maximum(1,a[:,2:]-a[:,:2]);bz=np.maximum(1,b[:,2:]-b[:,:2])
    sa=np.sqrt(az.prod(axis=1));sb=np.sqrt(bz.prod(axis=1))
    distance=np.linalg.norm(ac-bc,axis=1)/np.maximum(sa,sb)
    near=distance<=cfg.maximum_neighbor_size
    if near.mean()<cfg.shared_fraction:return None
    normalized=(bc-ac)/np.maximum(sa,sb)[:,None]
    residual=float(np.quantile(np.linalg.norm(normalized-np.median(normalized,axis=0),axis=1),.75))
    ratios=sb/sa;size_error=float(np.median(np.abs(ratios-np.median(ratios)))/max(.01,np.median(ratios)))
    if residual>cfg.offset_residual or size_error>cfg.size_residual:return None
    fixed=max(float(np.quantile(np.linalg.norm(ac-np.median(ac,axis=0),axis=1),cfg.shared_fraction)),float(np.quantile(np.linalg.norm(bc-np.median(bc,axis=0),axis=1),cfg.shared_fraction)))<=cfg.fixed_pixels+cfg.fixed_size_fraction*max(np.median(sa),np.median(sb))
    scale_change=min(float(np.ptp(np.log(sa))),float(np.ptp(np.log(sb))))
    scale_corr=float(np.corrcoef(np.log(sa),np.log(sb))[0,1]) if scale_change>=cfg.minimum_scale_change else 0.
    motion_change=min(float(np.ptp(ac,axis=0).max()),float(np.ptp(bc,axis=0).max()))
    motion_corr=0.
    if motion_change>=cfg.minimum_motion_pixels:
        axis=int(np.argmax(np.ptp(ac,axis=0)));x=ac[:,axis];y=bc[:,axis]
        if min(np.std(x),np.std(y))>1e-6:motion_corr=float(np.corrcoef(x,y)[0,1])
    if not (fixed or scale_corr>=cfg.scale_correlation or motion_corr>=cfg.motion_correlation):return None
    return dict(samples=len(common),first=common[0],last=common[-1],offset_residual=residual,size_residual=size_error,
                fixed=bool(fixed),scale_correlation=scale_corr,motion_correlation=motion_corr)


def connected_groups(nodes,fps,cfg):
    keys=sorted(nodes);parent={k:k for k in keys};links=[]
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    candidates=[];pair_ok={}
    for pos,key in enumerate(keys):
        for other in keys[pos+1:]:
            a,b=nodes[key],nodes[other]
            if a['scene']!=b['scene'] or a.get('rejected') or b.get('rejected'):continue
            evidence=relation(a['boxes'],b['boxes'],fps,cfg)
            pair_ok[frozenset([key,other])]=evidence
            if evidence:
                candidates.append(dict(left=key,right=other,**evidence))
    # 不能由一条含多个覆盖物的混合SAM轨迹把独立水印串成一组。
    # 任何具有足够共现样本的跨组成员都必须相互符合联动关系。
    for edge in sorted(candidates,key=lambda e:e['offset_residual']+e['size_residual']):
        key,other=edge['left'],edge['right'];a=find(key);b=find(other)
        if a==b:continue
        left=[k for k in keys if find(k)==a];right=[k for k in keys if find(k)==b]
        valid=True
        for x in left:
            for y in right:
                common=set(nodes[x]['boxes'])&set(nodes[y]['boxes'])
                if len(common)>=cfg.minimum_samples and not pair_ok.get(frozenset([x,y])):valid=False
        if valid:parent[b]=a;links.append(edge)
    result={}
    for k in keys:result.setdefault(find(k),[]).append(k)
    return list(result.values()),links


@dataclass(frozen=True)
class AnimationSettings:
    minimum_scale_ratio:float=1.3
    minimum_center_travel:float=.6
    minimum_frames:int=12
    minimum_step_scale:float=.015
    minimum_step_motion:float=.025
    settle_frames:int=8
    context_frames:int=8
    maximum_window_frames:int=120
    long_event_overlap_fraction:float=.75

    def __post_init__(self):
        if self.maximum_window_frames<36 or not .5<=self.long_event_overlap_fraction<1:raise ValueError('动画窗口配置无效')


def animation_interval(boxes,cfg):
    indices=sorted(boxes)
    if len(indices)<cfg.minimum_frames:return None
    b=np.array([boxes[i] for i in indices]);c=(b[:,:2]+b[:,2:])/2;s=np.sqrt(np.maximum(1,b[:,2:]-b[:,:2]).prod(axis=1))
    if s.max()/s.min()<cfg.minimum_scale_ratio and np.ptp(c,axis=0).max()/np.median(s)<cfg.minimum_center_travel:return None
    active=[]
    for k in range(1,len(indices)):
        gap=indices[k]-indices[k-1]
        if gap!=1:continue
        if abs(np.log(s[k]/s[k-1]))>=cfg.minimum_step_scale or np.linalg.norm(c[k]-c[k-1])/max(s[k],s[k-1])>=cfg.minimum_step_motion:active.extend([indices[k-1],indices[k]])
    if not active:return None
    return [max(indices[0],min(active)-cfg.settle_frames),min(indices[-1]+1,max(active)+cfg.settle_frames+1)]


def planned_windows(start,end,base_window,overlap,events,cfg):
    from worker.vision import windows
    intervals=[]
    for a,b in sorted(events):
        a=max(start,a-cfg.context_frames);b=min(end,b+cfg.context_frames)
        if a>=b:continue
        if intervals and a<=intervals[-1][1]:intervals[-1][1]=max(b,intervals[-1][1])
        else:intervals.append([a,b])
    result=[];cursor=start
    for a,b in intervals:
        if cursor<a:result.extend(windows(cursor,min(end,a+overlap),base_window,overlap))
        if b-a<=cfg.maximum_window_frames:result.append((a,b))
        else:
            large_overlap=max(overlap,round(cfg.maximum_window_frames*cfg.long_event_overlap_fraction))
            result.extend(windows(a,b,cfg.maximum_window_frames,large_overlap))
        cursor=max(cursor,end if b==end else b-overlap)
    if cursor<end:result.extend(windows(cursor,end,base_window,overlap))
    return list(dict.fromkeys(result))
