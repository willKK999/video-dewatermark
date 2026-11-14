"""复用v7标准差3的联合纹理投影；不透明区域禁止回灌源水印细节。"""
import json
from pathlib import Path
import cv2,numpy as np
from .matte import projected_texture_composite,solve_projection
from .hls_settings import HLSSettings
from .process import PipelineError


def place_field(field,box,shape):
    a,b,c,d=map(round,box);h,w=shape;out=np.zeros(shape,field.dtype)
    if c<=a or d<=b:return out
    scaled=cv2.resize(field,(c-a,d-b),interpolation=cv2.INTER_LINEAR)
    x,y,r,s=max(0,a),max(0,b),min(w,c),min(h,d)
    if r>x and s>y:out[y:s,x:r]=scaled[y-b:s-b,x-a:r-a]
    return out


def fields(points,templates,shape,return_support=False):
    glyph=np.zeros(shape,np.float32);exclude=np.zeros(shape,np.uint8);known=np.zeros(shape,np.uint8)
    for p in points:
        data=templates[p['track_id']];box=p.get('template_box',p['box'])
        support=data['mask']>0
        glyph+=place_field(np.where(support,data['feature'],0),box,shape)
        known|=(place_field(support.astype(np.uint8)*255,box,shape)>0).astype(np.uint8)*255
        opaque=np.where(support & (data['alpha']>.45),255,0).astype(np.uint8)
        exclude|=(place_field(opaque,box,shape)>0).astype(np.uint8)*255
    return (glyph,exclude,known) if return_support else (glyph,exclude)


def render(work,files,detected,fps):
    templates={t['id']:dict(np.load(t['template'],allow_pickle=False)) for t in detected['tracks']}
    byframe={}
    for p in detected['hints']:byframe.setdefault(p['frame'],[]).append(p)
    dest=work/'detail';dest.mkdir();shape=cv2.imread(str(files[0])).shape[:2]
    cfg=HLSSettings.from_env();acc={}
    selected=np.unique(np.linspace(0,len(files)-1,min(24,len(files)),dtype=int))
    for i in selected:
        source=cv2.imread(str(files[i]));base=cv2.imread(str(work/'composite'/files[i].name))
        mask=cv2.imread(str(work/'masks'/files[i].name),0)
        glyph,exclude,known=fields(byframe.get(int(i),[]),templates,shape,True)
        exclude|=mask&~known
        projected_texture_composite(source,base,mask,glyph,exclude,accumulator=acc,detail_sigma=cfg.detail_sigma)
    calibration=solve_projection(acc)
    (work/'detail-calibration.json').write_text(json.dumps(calibration))
    opaque_pixels=0;masked_pixels=0
    for i,file in enumerate(files):
        source=cv2.imread(str(file));base=cv2.imread(str(work/'composite'/file.name))
        if base is None:raise PipelineError('OUTPUT_INCOMPLETE','细节合成缺少补画结果，禁止原帧补齐')
        mask=cv2.imread(str(work/'masks'/file.name),0)
        glyph,exclude,known=fields(byframe.get(i,[]),templates,shape,True)
        exclude|=mask&~known
        result=projected_texture_composite(source,base,mask,glyph,exclude,
                calibration=calibration,detail_sigma=cfg.detail_sigma)
        cv2.imwrite(str(dest/file.name),result)
        masked_pixels+=np.count_nonzero(mask);opaque_pixels+=np.count_nonzero(exclude&mask)
    stats=dict(标准差=cfg.detail_sigma,校准样本数=len(selected),处理帧数=len(files),
               掩膜像素=int(masked_pixels),不透明排除像素=int(opaque_pixels),透明反解=False)
    (work/'detail-summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    return dest,stats
