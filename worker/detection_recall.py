"""本输入自动模板补漏：多背景连通支持、全画面多尺度复查与镜头隔离。"""
from dataclasses import dataclass
import cv2
import numpy as np
from .model_tracks import ModelTrackSettings, clip_box, patch, projected_box, intersection_fraction
from .multitrack import highpass, fill_support


@dataclass(frozen=True)
class RecallSettings(ModelTrackSettings):
    complete_component_support: bool = True
    component_context_heights: float = 3.
    component_gap_heights: float = .65
    component_agreement: float = .7
    recall_stride: int = 5
    recall_scales: tuple = (.65, .85, 1., 1.25, 1.5, 1.8, 2.2, 2.7)
    recall_correlation: float = .82
    recall_min_observations: int = 2
    recall_search_ratio: float = .5

    def __post_init__(self):
        if self.recall_stride<1 or self.recall_min_observations<2 or not 0<self.recall_search_ratio<=1:
            raise ValueError('模板补漏采样及确认参数无效')
        if not 0<self.recall_correlation<=1 or any(s<=0 for s in self.recall_scales):
            raise ValueError('模板补漏尺度或相关阈值无效')


def connected_support(features, anchor, cfg):
    """只补模型锚点邻近、跨背景同号稳定的连通区域，闭合图标内部。"""
    stack=np.stack(features)
    median=np.median(stack,axis=0)
    magnitude=np.abs(median)
    threshold=max(.8,float(np.median(magnitude))*5)
    votes=np.mean(stack*np.sign(median)[None]>.3,axis=0)
    signal=((magnitude>threshold)&(votes>=cfg.component_agreement)).astype(np.uint8)*255
    joined=cv2.morphologyEx(signal,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(joined)
    selected=[];a,b,c,d=anchor;gap=max(2,(d-b)*cfg.component_gap_heights)
    pending=[(i,tuple(map(int,stats[i][:4]))) for i in range(1,n) if stats[i,4]>=6]
    frontier=[anchor]
    while pending:
        rest=[];added=False
        for i,(x,y,w,h) in pending:
            box=[x,y,x+w,y+h]
            distance=lambda q: np.hypot(max(0,q[0]-box[2],box[0]-q[2]),max(0,q[1]-box[3],box[1]-q[3]))
            if min(map(distance,frontier))<=gap:
                selected.append(i);frontier.append(box);added=True
            else:rest.append((i,(x,y,w,h)))
        if not added:break
        pending=rest
    mask=np.where(np.isin(labels,selected),255,0).astype(np.uint8)
    # 每个连通部件分别封闭；不把图文间的背景整体填成面板。
    return median,fill_support(mask,cfg.support_radius)


def complete_prototype(files,hints,seed,cfg):
    """以模型框对齐多背景样本，从连通区域推导图文整体范围。"""
    anchor=seed['model_box'];height=anchor[3]-anchor[1]
    context=[anchor[0]-height*cfg.component_context_heights,anchor[1]-height,
             anchor[2]+height*cfg.component_context_heights,anchor[3]+height]
    image=cv2.imread(str(files[seed['frame']]))
    context=list(map(round,clip_box(context,image.shape[:2])))
    a,b,c,d=context;size=(max(2,c-a),max(2,d-b))
    geometry=dict(anchor=anchor,box=context)
    features=[]
    for p in hints:
        box=projected_box(geometry,p['model_box'])
        # 边缘截断样本不能挤压成完整模板。
        im=cv2.imread(str(files[p['frame']]))
        if box[0]<0 or box[1]<0 or box[2]>im.shape[1] or box[3]>im.shape[0]:continue
        features.append(highpass(patch(im,box,size)))
    if len(features)<cfg.minimum_template_frames:return None
    feature,mask=connected_support(features,[anchor[0]-a,anchor[1]-b,anchor[2]-a,anchor[3]-b],cfg)
    yy,xx=np.where(mask>0)
    if len(xx)<16:return None
    x,y,r,s=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
    return dict(box=[a+x,b+y,a+r,b+s],whole_feature=feature[y:s,x:r],mask=mask[y:s,x:r])


def template_recall(image,prototypes,cfg):
    """高阈值全画面复查；模板仅来自当前源片自动确认的实例。"""
    from .brand_templates import match
    ratio=cfg.recall_search_ratio
    small=cv2.resize(image,None,fx=ratio,fy=ratio)
    bank=[];owners=[]
    seen=set()
    for p in prototypes:
        family=p.get('family',p['id'])
        if family in seen or 'whole_feature' not in p or p.get('component_only'):continue
        seen.add(family)
        for scale in cfg.recall_scales:
            f=p['whole_feature'];m=p['mask'];size=(max(2,round(f.shape[1]*ratio*scale)),max(2,round(f.shape[0]*ratio*scale)))
            if size[0]>small.shape[1] or size[1]>small.shape[0]:continue
            bank.append(dict(template=cv2.resize(f,size),mask=cv2.resize(m,size,interpolation=cv2.INTER_NEAREST)))
            owners.append(p)
    if not bank:return []
    result=[]
    for hit in match(small,bank,cfg.recall_correlation,limit=2):
        p=owners[hit['variant']];box=[v/ratio for v in hit['box']]
        result.append(dict(box=box,template_box=box,model_box=box,kind='prototype',prototype=p['id'],
                           family=p.get('family',p['id']),confidence=hit['confidence'],template_score=hit['confidence'],
                           evidence='本输入高置信度实例全画面多尺度复核',template_recall=True))
    return result


def build(*args,**kwargs):
    from .model_tracks import build as build_tracks
    kwargs['settings']=RecallSettings()
    return build_tracks(*args,**kwargs)
