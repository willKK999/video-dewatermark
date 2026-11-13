"""逐帧专用YOLO印证的多目标轨迹与内容绑定支持；旧v7路径不调用本模块。"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,math
import cv2,numpy as np
from .multitrack import load_prior,highpass,fill_support,bridge_gaps,component_scores
from .hls_settings import is_residual_score
from .process import PipelineError
from .sam_schedule import sampled_hints


@dataclass(frozen=True)
class ModelTrackSettings:
    anchor_confidence: float = .35
    seed_confidence: float = .65
    template_correlation: float = .55
    overlap_fraction: float = .6
    maximum_gap_seconds: float = .35
    logo_horizontal_padding: float = 1.2
    logo_vertical_padding: float = .2
    template_samples: int = 24
    support_radius: int = 3


def intersection_fraction(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))/max(1,(a[2]-a[0])*(a[3]-a[1]))


def clip_box(box,shape):
    h,w=shape
    a,b,c,d=box
    return [max(0.,min(w-1.,a)),max(0.,min(h-1.,b)),max(1.,min(float(w),c)),max(1.,min(float(h),d))]


def patch(image,box,size=None):
    a,b,c,d=map(round,clip_box(box,image.shape[:2]))
    result=image[b:d,a:c]
    if not result.size:raise PipelineError('DETECTION_INVALID','自动目标区域为空')
    return cv2.resize(result,size) if size else result


def feature_score(a,b):
    b=cv2.resize(b,(a.shape[1],a.shape[0]))
    support=np.abs(a)>max(.5,float(np.median(np.abs(a)))*2)
    if support.sum()<16:return 0.
    x,y=a[support],b[support]
    return float(x@y/max(1e-6,np.linalg.norm(x)*np.linalg.norm(y)))


def learn_text_prototypes(files,rows,prior,labels,settings):
    median,signal,regions,_,_=load_prior(prior)
    image=cv2.imread(str(files[0]));h,w=image.shape[:2];mh,mw=median.shape
    result=[]
    for region in regions:
        if region['confidence']<settings.seed_confidence:continue
        rbox=[v*(w/mw if j%2==0 else h/mh) for j,v in enumerate(region['box'])]
        candidates=[dict(p,frame=row['frame']) for row in rows for p in row['boxes']
                    if 'text' in labels.get(p['class_id'],'') and p['confidence']>=settings.anchor_confidence
                    and intersection_fraction(p['box'],rbox)>.9]
        if not candidates:continue
        # 只从模型与跨时段共识共同确认的位置学习图形，不把共识坐标固定到新位置。
        chosen=sorted(candidates,key=lambda p:p['confidence'],reverse=True)[:settings.template_samples]
        seed=chosen[0]
        im=cv2.imread(str(files[seed['frame']]))
        feature=highpass(patch(im,seed['box']))
        mask=cv2.resize(fill_support(region['mask']), (w,h),interpolation=cv2.INTER_NEAREST)
        a,b,c,d=map(round,rbox)
        # 对本输入中稳定的模型印证图形补齐填色；不使用其他影片的颜色或固定框。
        from .branding import learn_palette,palette_support
        spread=sorted(candidates,key=lambda p:p['frame'])
        chosen_indices=np.unique(np.linspace(0,len(spread)-1,min(settings.template_samples,len(spread)),dtype=int))
        images=[patch(cv2.imread(str(files[spread[k]['frame']])),rbox,(c-a,d-b)) for k in chosen_indices]
        if len(images)>=3:
            try:
                palette=learn_palette(images,mask[b:d,a:c])
                colors=np.stack(images).astype(np.float32)
                middle=np.median(colors,axis=0)
                stable=np.median(np.abs(colors-middle),axis=0).mean(axis=2)<10
                colored=palette_support(middle.astype(np.uint8),palette)
                mask[b:d,a:c] |= fill_support(np.where(stable,colored,0).astype(np.uint8),settings.support_radius)
            except ValueError:
                pass  # 无可靠颜色证据时保留已有模型/时域支持，不能编造填色。
        result.append(dict(id=len(result),anchor=seed['box'],box=rbox,
                           feature=feature,mask=mask[b:d,a:c],samples=len(chosen),
                           examples=[highpass(patch(cv2.imread(str(files[p['frame']])),p['box'])) for p in chosen[::max(1,len(chosen)//5)]]))
    return result


def candidates_for_frame(image,row,prototypes,labels,cfg):
    blue=[];logos=[]
    for raw in row['boxes']:
        if raw['confidence']<cfg.anchor_confidence:continue
        label=labels.get(raw['class_id'],'')
        if 'text' in label or 'logo' in label:
            feature=highpass(patch(image,raw['box']))
            scored=[(max(feature_score(z,feature) for z in p['examples']),p) for p in prototypes
                    if max((raw['box'][2]-raw['box'][0])/(raw['box'][3]-raw['box'][1]),
                           (p['anchor'][2]-p['anchor'][0])/(p['anchor'][3]-p['anchor'][1])) /
                       max(.01,min((raw['box'][2]-raw['box'][0])/(raw['box'][3]-raw['box'][1]),
                                   (p['anchor'][2]-p['anchor'][0])/(p['anchor'][3]-p['anchor'][1])))<1.8]
            score,proto=max(scored,key=lambda x:x[0],default=(0,None))
            if proto is not None and score>=cfg.template_correlation:
                scale=(raw['box'][2]-raw['box'][0])/max(1,proto['anchor'][2]-proto['anchor'][0])
                dx=raw['box'][0]-proto['anchor'][0]*scale;dy=raw['box'][1]-proto['anchor'][1]*scale
                raw_box=[proto['box'][0]*scale+dx,proto['box'][1]*scale+dy,
                         proto['box'][2]*scale+dx,proto['box'][3]*scale+dy]
                box=clip_box(raw_box,image.shape[:2])
                blue.append(dict(box=box,template_box=raw_box,confidence=raw['confidence'],kind='prototype',prototype=proto['id'],model_box=raw['box'],template_score=score))
                continue
            if 'text' in label:continue
        if 'logo' in label:
            if 'corner' in label and raw['confidence']<cfg.seed_confidence:continue
            a,b,c,d=raw['box'];height=d-b
            box=clip_box([a-height*cfg.logo_horizontal_padding,b-height*cfg.logo_vertical_padding,
                          c+height*cfg.logo_horizontal_padding,d+height*cfg.logo_vertical_padding],image.shape[:2])
            logos.append(dict(box=box,confidence=raw['confidence'],kind='logo',prototype=-1,model_box=raw['box']))
    selected=[]
    for p in sorted(blue,key=lambda p:p['confidence'],reverse=True)+sorted(logos,key=lambda p:p['confidence'],reverse=True):
        if any(min(intersection_fraction(p['box'],q['box']),intersection_fraction(q['box'],p['box']))>cfg.overlap_fraction
               or (intersection_fraction(p['box'],q['box'])>.9 and q['kind']=='prototype') for q in selected):continue
        selected.append(p)
    return selected


def associate(rows,scenes,fps,cfg):
    tracks=[];next_id=1
    for start,end in scenes:
        active=[]
        for i in range(start,end):
            used=set()
            for p in rows[i]:
                eligible=[]
                for t in active:
                    if t['id'] in used or i-t['last']>max(1,round(fps*cfg.maximum_gap_seconds)):continue
                    if t['kind']!=p['kind']:continue
                    old=t['hints'][-1]['box'];box=p['box']
                    overlap=min(intersection_fraction(old,box),intersection_fraction(box,old))
                    distance=math.hypot((old[0]+old[2]-box[0]-box[2])/2,(old[1]+old[3]-box[1]-box[3])/2)
                    size=max(old[2]-old[0],old[3]-old[1])
                    if overlap>.1 or distance<size*.5:eligible.append((overlap-distance/max(1,size),t))
                if eligible:t=max(eligible,key=lambda x:x[0])[1]
                else:
                    t=dict(id=next_id,kind=p['kind'],prototype=p['prototype'],hints=[],last=i)
                    next_id+=1;active.append(t)
                t['hints'].append(dict(p,frame=i,track_id=t['id']));t['last']=i;used.add(t['id'])
        for t in active:
            t.pop('last');t['hints']=bridge_gaps(t['hints'],end,max_gap=max(1,round(fps*cfg.maximum_gap_seconds)))
            tracks.append(t)
    return tracks


def build(files,scenes,rows,prior,labels,root,fps=30,settings=None,check=lambda:None):
    cfg=settings or ModelTrackSettings()
    prototypes=learn_text_prototypes(files,rows,prior,labels,cfg)
    candidates=[]
    for i,file in enumerate(files):
        check();candidates.append(candidates_for_frame(cv2.imread(str(file)),rows[i],prototypes,labels,cfg))
    tracks=associate(candidates,scenes,fps,cfg)
    if not tracks:raise PipelineError('DETECTION_UNCERTAIN','逐帧YOLO未得到可验证的品牌轨迹')
    prompts=root/'prompts';prompts.mkdir(exist_ok=True)
    templates=root/'templates';templates.mkdir(exist_ok=True)
    image=cv2.imread(str(files[0]));shape=image.shape[:2]
    for t in tracks:
        hints=t['hints'];seed=max(hints,key=lambda p:p['confidence'])
        a,b,c,d=map(round,seed['box']);size=(max(2,c-a),max(2,d-b))
        selected=np.unique(np.linspace(0,len(hints)-1,min(cfg.template_samples,len(hints)),dtype=int))
        features=[highpass(patch(cv2.imread(str(files[hints[k]['frame']])),hints[k]['box'],size)) for k in selected]
        feature=np.median(np.stack(features),axis=0)
        if t['kind']=='prototype':
            proto=prototypes[t['prototype']];base=cv2.resize(proto['mask'],size,interpolation=cv2.INTER_NEAREST)
        else:
            noise=float(np.median(np.abs(feature)))
            signal=(np.abs(feature)>max(.5,noise*5)).astype(np.uint8)*255
            signal=cv2.morphologyEx(signal,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            base=fill_support(signal,cfg.support_radius)
        if np.count_nonzero(base)<16:raise PipelineError('MASK_UNCERTAIN','自动品牌轮廓支持不足')
        np.savez_compressed(templates/f'{t["id"]}.npz',feature=feature,mask=base)
        t['template']=str(templates/f'{t["id"]}.npz')
        # 提示来自同一输入的自动模型与模板，供SAM分割细化；无人工坐标或帧号。
        selected_prompts={p['frame'] for p in sampled_hints(hints,hints[0]['frame'],hints[-1]['frame']+1)}
        for p in hints:
            if p['frame'] not in selected_prompts:continue
            mask=place(base,p.get('template_box',p['box']),shape)
            path=prompts/f'{t["id"]}-{p["frame"]}.png';cv2.imwrite(str(path),mask)
            p['mask_prompt']=str(path.relative_to(root))
    result=dict(schema='yolo-model-tracks-1',multitrack=True,hls=True,scenes=scenes,tracks=tracks,
                hints=sorted([p for t in tracks for p in t['hints']],key=lambda p:(p['frame'],p['track_id'])),
                settings=asdict(cfg),raw_model_boxes=sum(len(r['boxes']) for r in rows),
                evidence='目标由逐帧YOLO可靠观测建立，短缺口仅在可靠前后观测间有界插值；文本需本输入模型/共识模板印证；不以全画面模板峰新增轨迹')
    (root/'detections.json').write_text(json.dumps(result))
    return result


def place(base,box,shape):
    h,w=shape;a,b,c,d=map(round,box)
    result=np.zeros(shape,np.uint8)
    if c<=a or d<=b:return result
    scaled=cv2.resize(base,(c-a,d-b),interpolation=cv2.INTER_NEAREST)
    x,y,r,s=max(0,a),max(0,b),min(w,c),min(h,d)
    if r>x and s>y:result[y:s,x:r]=scaled[y-b:s-b,x-a:r-a]
    return result


def validate(root,files,detections,check=lambda:None):
    from .vision import sha256
    from .hls_layers import track_bins,write_manifest,VerifiedLayers
    byframe={};maximum={};object_masks=root/'validated-objects';object_masks.mkdir(exist_ok=True)
    for track in detections['tracks']:
        (object_masks/str(track['id'])).mkdir(exist_ok=True)
        data=np.load(track['template'],allow_pickle=False);base=data['mask']
        for p in track['hints']:byframe.setdefault(p['frame'],[]).append((track,p,base))
    manifests=[]
    for i,file in enumerate(files):
        check();image=cv2.imread(str(file));shape=image.shape[:2];joint=np.zeros(shape,np.uint8)
        for track,p,base in byframe.get(i,[]):
            support=place(base,p.get('template_box',p['box']),shape)
            raw=root/'object-masks'/str(track['id'])/file.name
            if not raw.exists():raise PipelineError('MASK_INVALID','SAM未输出预期的对象帧，禁止伪装为完成')
            sam=cv2.imread(str(raw),0)
            if sam is None:raise PipelineError('MASK_INVALID','SAM对象掩膜不可读')
            mask=support | (sam & cv2.dilate(support,np.ones((7,7),np.uint8)))
            fraction=float(np.count_nonzero(mask)/mask.size)
            if fraction>.2:raise PipelineError('MASK_UNCERTAIN','单条模型轨迹超过20%面积保护')
            maximum[track['id']]=max(maximum.get(track['id'],0),fraction)
            cv2.imwrite(str(object_masks/str(track['id'])/file.name),mask);joint|=mask
        cv2.imwrite(str(root/'masks'/file.name),joint)
        manifests.append(dict(frame_index=i,mask_key='masks/'+file.name,sha256=sha256(root/'masks'/file.name)))
    bins=track_bins(maximum);layers=[]
    for k,identities in enumerate(bins):
        directory=root/f'layer-{k:02d}'/'masks';directory.mkdir(parents=True,exist_ok=True)
        for file in files:
            image=cv2.imread(str(root/'masks'/file.name),0);joint=np.zeros_like(image)
            for identity in identities:
                p=object_masks/str(identity)/file.name
                if p.exists():joint|=cv2.imread(str(p),0)
            cv2.imwrite(str(directory/file.name),joint)
        layers.append(directory)
    write_manifest(root,root/'detections.json',layers,files)
    verified=VerifiedLayers(root/'layers.json',root/'detections.json')
    for i,file in enumerate(files):verified.validate(i,cv2.imread(str(root/'masks'/file.name),0))
    (root/'model-mask-summary.json').write_text(json.dumps(dict(层数=len(layers),轨迹数=len(detections['tracks']),逐帧=manifests,最大轨迹面积=maximum),ensure_ascii=False))
    return manifests
