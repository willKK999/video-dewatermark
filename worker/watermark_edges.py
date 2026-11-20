"""确认水印组的尺度自适应色/alpha边缘候选；不包含影片位置规则。"""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class EdgeSettings:
    antialias_pixels:int=2
    component_radius_fraction:float=.15
    maximum_radius_frame_fraction:float=.06
    text_margin_height_fraction:float=.75
    motion_exposure_fraction:float=.5
    hue_tolerance:float=9.
    minimum_saturation:float=192.
    minimum_value:float=160.
    palette_background_odds:float=1.5
    palette_minimum_fraction:float=.025
    palette_colors:int=3
    neutral_saturation:float=128.
    neutral_minimum_value:float=0.
    contrast_noise:float=4.
    alpha_minimum:float=.02
    alpha_residual_noise:float=6.
    alpha_residual_fraction:float=.08
    gradient_minimum:float=1.5
    minimum_component_area:int=5
    minimum_seed_area_fraction:float=.002
    colored_text_fraction:float=.25
    temporal_stride_seconds:float=.25
    background_features:int=400
    background_minimum_features:int=8
    background_inlier_fraction:float=.6
    background_match_fraction:float=.7
    background_advantage:float=.1
    temporal_stride_frames:int=3
    temporal_support:int=3
    rapid_scale_log:float=.08
    mask_fraction_limit:float=.2

    def __post_init__(self):
        if self.mask_fraction_limit!=.2:raise ValueError('20%保护不得放宽')
        if not 1<=self.antialias_pixels<=3 or not 0<self.component_radius_fraction<.5 or not 0<self.maximum_radius_frame_fraction<.1:raise ValueError('边缘尺度配置无效')
        if not 0<self.alpha_minimum<1 or self.temporal_support<1 or self.temporal_stride_frames<1:raise ValueError('alpha或时域配置无效')


def box(mask):
    y,x=np.where(mask>0)
    return None if not len(x) else np.array([x.min(),y.min(),x.max()+1,y.max()+1],float)


def radius(mask,cfg,motion=0.):
    n,_,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8));sizes=[min(w,h) for x,y,w,h,a in stats[1:] if a>=cfg.minimum_component_area]
    scale=max(sizes,default=1);cap=cfg.maximum_radius_frame_fraction*min(mask.shape)
    return int(math.ceil(max(cfg.antialias_pixels,min(cap,scale*cfg.component_radius_fraction+motion*cfg.motion_exposure_fraction))))


def align(mask,first_box,second_box,shape):
    a=np.asarray(first_box);b=np.asarray(second_box);scale=(b[2:]-b[:2])/np.maximum(1,a[2:]-a[:2]);matrix=np.array([[scale[0],0,b[0]-a[0]*scale[0]],[0,scale[1],b[1]-a[1]*scale[1]]],np.float32)
    return cv2.warpAffine(mask,matrix,(shape[1],shape[0]),flags=cv2.INTER_NEAREST)


def background_transform(first,second,excluded,cfg):
    scale=.5;shape=(round(first.shape[1]*scale),round(first.shape[0]*scale));a=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),shape);b=cv2.resize(cv2.cvtColor(second,cv2.COLOR_BGR2GRAY),shape);valid=cv2.resize(255-excluded,shape,interpolation=cv2.INTER_NEAREST)
    points=cv2.goodFeaturesToTrack(a,cfg.background_features,.01,4,mask=valid)
    if points is None or len(points)<cfg.background_minimum_features:return None
    nxt,status,error=cv2.calcOpticalFlowPyrLK(a,b,points,None)
    if nxt is None:return None
    keep=(status.ravel()>0)&(error.ravel()<30)
    if keep.sum()<cfg.background_minimum_features:return None
    matrix,inliers=cv2.estimateAffinePartial2D(points[keep]/scale,nxt[keep]/scale,method=cv2.RANSAC,ransacReprojThreshold=max(1.,min(first.shape[:2])*.003))
    if matrix is None or not np.isfinite(matrix).all() or np.mean(inliers)<cfg.background_inlier_fraction:return None
    return matrix


def reject_background(candidate,own_aligned,background_aligned,cfg):
    if not background_aligned:return candidate.copy(),0
    count,labels,_,_=cv2.connectedComponentsWithStats((candidate>0).astype(np.uint8));result=candidate.copy();removed=0
    for k in range(1,count):
        part=labels==k
        if part.sum()<cfg.minimum_component_area:continue
        own=float(np.mean([np.mean(m[part]>0) for m in own_aligned]));bg=float(np.mean([np.mean(m[part]>0) for m in background_aligned]))
        if bg>=cfg.background_match_fraction and bg>own+cfg.background_advantage:removed+=int(part.sum());result[part]=0
    return result,removed


def proposal(image,seed,protection,cfg,ocr_boxes=(),motion=0.):
    original=seed;result=np.zeros(seed.shape,np.uint8);n,labels,stats,_=cv2.connectedComponentsWithStats((seed>0).astype(np.uint8));minimum=max(cfg.minimum_component_area,np.count_nonzero(seed)*cfg.minimum_seed_area_fraction);seed=np.where(np.isin(labels,[k for k in range(1,n) if stats[k,cv2.CC_STAT_AREA]>=minimum]),255,0).astype(np.uint8);bounds=box(seed)
    if bounds is None:return result,dict(radius=0,palette=[])
    r=radius(seed,cfg,motion);margin=max(r*3,max((b[3]-b[1])*cfg.text_margin_height_fraction*2 for b in ocr_boxes) if ocr_boxes else 0)
    h,w=seed.shape;x,y,a,b=np.rint(bounds+[-margin,-margin,margin,margin]).astype(int);x,y,a,b=max(0,x),max(0,y),min(w,a),min(h,b)
    im=image[y:b,x:a];m=seed[y:b,x:a];guard=protection[y:b,x:a];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);distance=cv2.distanceTransform(255-m,cv2.DIST_L2,5);search=np.zeros(m.shape,bool);envelope=np.zeros_like(m);neutral_envelope=np.zeros_like(m);font_height=0.
    count,components,properties,_=cv2.connectedComponentsWithStats((m>0).astype(np.uint8))
    for k in range(1,count):
        cx,cy,cw,ch,area=properties[k];rr=int(math.ceil(max(cfg.antialias_pixels,min(cfg.maximum_radius_frame_fraction*min(seed.shape),min(cw,ch)*cfg.component_radius_fraction+motion*cfg.motion_exposure_fraction))));lx,ly,la,lb=max(0,cx-rr),max(0,cy-rr),min(m.shape[1],cx+cw+rr),min(m.shape[0],cy+ch+rr);part=(components[ly:lb,lx:la]==k).astype(np.uint8)*255;search[ly:lb,lx:la]|=cv2.distanceTransform(255-part,cv2.DIST_L2,5)<=rr
    for tb in ocr_boxes:
        height=tb[3]-tb[1];font_height=max(font_height,height);tx,ty,ta,tb2=np.rint(np.array(tb)-[x,y,x,y]).astype(int);region=m[max(0,ty):min(m.shape[0],tb2),max(0,tx):min(m.shape[1],ta)]>0;colors=hsv[max(0,ty):min(m.shape[0],tb2),max(0,tx):min(m.shape[1],ta)];color_fraction=float(np.count_nonzero(region&(colors[:,:,1]>=cfg.minimum_saturation)&(colors[:,:,2]>=cfg.minimum_value))/max(1,np.count_nonzero(region))) if region.size else 0.;colored=color_fraction>=cfg.colored_text_fraction;pad=height*(cfg.component_radius_fraction if colored else cfg.text_margin_height_fraction)
        tx,ty,ta,tb2=np.rint(np.array(tb)+[-pad,-pad,pad,pad]-[x,y,x,y]).astype(int);envelope[max(0,ty):min(m.shape[0],tb2),max(0,tx):min(m.shape[1],ta)]=255
        if not colored:neutral_envelope[max(0,ty):min(m.shape[0],tb2),max(0,tx):min(m.shape[1],ta)]=255
    search|=envelope>0;inner=cv2.erode(m,np.ones((3,3),np.uint8))>0
    if not inner.any():inner=m>0
    saturated=(hsv[:,:,1]>=cfg.minimum_saturation)&(hsv[:,:,2]>=cfg.minimum_value)
    hist=np.bincount(hsv[:,:,0][inner&saturated],minlength=180).astype(float);ring=(distance>r)&(distance<=r*2)&(guard==0);background_hist=np.bincount(hsv[:,:,0][ring&saturated],minlength=180).astype(float)
    hist/=max(1,hist.sum());background_hist/=max(1,background_hist.sum());palette=[];color=np.zeros(m.shape,bool);score=hist/(background_hist+.01)
    for _ in range(cfg.palette_colors):
        k=int(np.argmax(score));near=np.minimum(np.abs(np.arange(180)-k),180-np.abs(np.arange(180)-k))<=cfg.hue_tolerance
        if hist[near].sum()<cfg.palette_minimum_fraction or score[k]<cfg.palette_background_odds:break
        selected=inner&saturated&near[hsv[:,:,0]]
        if selected.any():
            foreground=np.median(im[selected],axis=0);palette.append(foreground.tolist());dh=np.abs(hsv[:,:,0].astype(float)-k);color|=(np.minimum(dh,180-dh)<=cfg.hue_tolerance)&(hsv[:,:,1]>=cfg.minimum_saturation*.25)&(hsv[:,:,2]>=cfg.minimum_value)
        score[near]=0
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1));opened=cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel);contrast=gray.astype(float)-opened
    neutral_seed=inner&(hsv[:,:,1]<=cfg.neutral_saturation)&(hsv[:,:,2]>=cfg.neutral_minimum_value)&(contrast>=cfg.contrast_noise)
    if np.count_nonzero(neutral_seed)>=cfg.minimum_component_area:palette.append([255.,255.,255.])
    outside=(~search)&(m==0)&(guard==0);weight=outside.astype(np.float32);sigma=max(1.,r*.75,font_height*cfg.text_margin_height_fraction*.75);den=cv2.GaussianBlur(weight,(0,0),sigma);bg=cv2.GaussianBlur(im.astype(np.float32)*weight[:,:,None],(0,0),sigma)/np.maximum(den[:,:,None],1e-5)
    delta=im.astype(float)-bg;alpha_evidence=np.zeros(m.shape,bool)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1);edge=cv2.magnitude(gx,gy)>=cfg.gradient_minimum
    has_colored=any(max(p)-min(p)>100 for p in palette)
    for p in palette:
        direction=np.array(p)-bg;power=np.sum(direction*direction,axis=2);alpha=np.sum(delta*direction,axis=2)/np.maximum(power,1.);residual=np.linalg.norm(delta-alpha[:,:,None]*direction,axis=2);limit=np.maximum(cfg.alpha_residual_noise,np.sqrt(power)*cfg.alpha_residual_fraction)
        accepted=(alpha>=cfg.alpha_minimum)&(alpha<=1.1)&(residual<=limit)&(power>=20**2)&(den>.01)&edge
        if has_colored and max(p)-min(p)<1:accepted&=(neutral_envelope>0)|(distance<=cfg.antialias_pixels)
        alpha_evidence|=accepted
    neutral=(contrast>=cfg.contrast_noise)&edge&alpha_evidence
    eligible=(color|alpha_evidence|neutral)&search&(guard==0);eligible|=m>0
    n,labels,stats,_=cv2.connectedComponentsWithStats(eligible.astype(np.uint8));contact=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*cfg.antialias_pixels+1,)*2))>0;kept=np.zeros_like(m)
    for k in range(1,n):
        part=labels==k;cx,cy,cw,ch,area=stats[k]
        glyph=font_height>0 and area>=cfg.minimum_component_area and ch<=font_height*1.5 and cw<=font_height*2.5 and np.any(part&(envelope>0))
        if np.any(part&contact) or glyph:kept[part]=255
    kept[(original[y:b,x:a]>0)|(guard>0)]=0;result[y:b,x:a]=kept
    return result,dict(radius=r,palette=palette,color_candidates=int(np.count_nonzero(color&search&(m==0))),alpha_candidates=int(np.count_nonzero(alpha_evidence&search&(m==0))))
