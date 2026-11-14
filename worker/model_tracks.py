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
    logo_vertical_padding: float = .5
    template_samples: int = 24
    support_radius: int = 3
    minimum_template_frames: int = 15
    maximum_prototypes: int = 12


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


def projected_box(proto,anchor):
    scale=(anchor[2]-anchor[0])/max(1,proto['anchor'][2]-proto['anchor'][0])
    dx=anchor[0]-proto['anchor'][0]*scale;dy=anchor[1]-proto['anchor'][1]*scale
    return [proto['box'][0]*scale+dx,proto['box'][1]*scale+dy,
            proto['box'][2]*scale+dx,proto['box'][3]*scale+dy]


def prototype_score(image,raw,proto):
    local=highpass(patch(image,raw['box']))
    score=0. if proto.get('require_whole') else max(feature_score(z,local) for z in proto['examples'])
    if 'whole_feature' in proto:
        full=highpass(patch(image,projected_box(proto,raw['box']),
                            (proto['mask'].shape[1],proto['mask'].shape[0])))
        support=proto['mask']>0
        score=max(score,feature_score(np.where(support,proto['whole_feature'],0),np.where(support,full,0)))
    return score


def temporal_bins(areas,limit=.2):
    """按逐帧实际施工面积装箱；非同时出现的轨迹不重复占用面积预算。"""
    groups=[];totals=[]
    for identity,values in sorted(areas.items(),key=lambda p:(-max(p[1].values(),default=0),p[0])):
        if max(values.values(),default=0)>limit:raise PipelineError('MASK_UNCERTAIN','单轨迹超过20%保护')
        k=next((i for i,t in enumerate(totals) if all(t.get(f,0)+a<=limit for f,a in values.items())),len(groups))
        if k==len(groups):groups.append([]);totals.append({})
        groups[k].append(identity)
        for f,a in values.items():totals[k][f]=totals[k].get(f,0)+a
    return groups


def disjoint_masks(masks):
    assigned=np.zeros_like(masks[0]);result=[]
    for mask in masks:
        result.append(mask & ~assigned);assigned|=mask
    return result


def learn_opacity(images,mask,minimum_frames=15):
    """仅作细节回灌保护，不执行透明反解；稳定彩色前景也按不透明处理。"""
    if len(images)<minimum_frames:return np.ones(mask.shape,np.float32)
    from .matte import fit_overlay
    alpha,_=fit_overlay(images,mask)
    stack=np.stack(images).astype(np.float32);middle=np.median(stack,axis=0)
    stable=np.median(np.abs(stack-middle),axis=0).mean(axis=2)<10
    saturation=cv2.cvtColor(middle.astype(np.uint8),cv2.COLOR_BGR2HSV)[:,:,1]
    alpha[(mask>0)&stable&(saturation>64)]=1.
    return alpha


def learn_text_prototypes(files,rows,prior,labels,settings,fps=None,scenes=None):
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
        a,b,c,d=map(round,rbox)
        # 对本输入中稳定的模型印证图形补齐填色；不使用其他影片的颜色或固定框。
        from .branding import learn_palette,palette_support
        spread=sorted(candidates,key=lambda p:p['frame'])
        chosen_indices=np.unique(np.linspace(0,len(spread)-1,min(settings.template_samples,len(spread)),dtype=int))
        images=[patch(cv2.imread(str(files[spread[k]['frame']])),rbox,(c-a,d-b)) for k in chosen_indices]
        full=np.median(np.stack([highpass(x) for x in images]),axis=0)
        from .matte import ink_support
        signal=(np.abs(full)>max(.5,float(np.median(np.abs(full)))*5)).astype(np.uint8)*255
        ink=ink_support(full,signal,np.full(signal.shape,255,np.uint8),settings.support_radius)
        mask=np.zeros((h,w),np.uint8);mask[b:d,a:c]=ink
        if len(images)>=3:
            try:
                colors=np.stack(images).astype(np.float32)
                middle=np.median(colors,axis=0)
                stable=np.median(np.abs(colors-middle),axis=0).mean(axis=2)<10
                saturation=cv2.cvtColor(middle.astype(np.uint8),cv2.COLOR_BGR2HSV)[:,:,1]
                colored=np.where(stable&(saturation>64),255,0).astype(np.uint8)
                mask[b:d,a:c] |= ink_support(full,colored,np.full(signal.shape,255,np.uint8),settings.support_radius)
            except ValueError:
                pass  # 无可靠颜色证据时保留已有模型/时域支持，不能编造填色。
        result.append(dict(id=len(result),family=len(result),anchor=seed['box'],box=rbox,
                           feature=feature,mask=mask[b:d,a:c],samples=len(chosen),
                           whole_feature=full,alpha=learn_opacity(images,mask[b:d,a:c]),
                           examples=[highpass(patch(cv2.imread(str(files[p['frame']])),p['box'])) for p in chosen[::max(1,len(chosen)//5)]]))
        canonical=result[-1]
        # 替代文字锚点继承完整图形几何，避免高分局部框抢占后截掉图标。
        for raw in rows[seed['frame']]['boxes']:
            if len(result)>=settings.maximum_prototypes:break
            if ('text' not in labels.get(raw['class_id'],'') or raw['confidence']<settings.anchor_confidence
                    or intersection_fraction(raw['box'],rbox)<.95):continue
            z=highpass(patch(im,raw['box']))
            result.append(dict(canonical,id=len(result),anchor=raw['box'],feature=z,examples=[z],require_whole=True))
        # 已确认完整图形中的圆形彩色图标可独立成为模型图标框的几何模板。
        n,component,stats,_=cv2.connectedComponentsWithStats(canonical['mask'])
        for identity in range(1,n):
            x,y,cw,ch,area=stats[identity]
            if len(result)>=settings.maximum_prototypes:break
            if area<canonical['mask'].size*.05 or not .65<cw/max(1,ch)<1.4:continue
            isolated=(component[y:y+ch,x:x+cw]==identity).astype(np.uint8)*255
            colors=np.median(np.stack(images),axis=0)[y:y+ch,x:x+cw].astype(np.uint8)
            if np.median(cv2.cvtColor(colors,cv2.COLOR_BGR2HSV)[:,:,1][isolated>0])<64:continue
            f=full[y:y+ch,x:x+cw]
            result.append(dict(id=len(result),family=len(result),anchor=[a+x,b+y,a+x+cw,b+y+ch],box=[a+x,b+y,a+x+cw,b+y+ch],
                feature=f,whole_feature=f,mask=isolated,alpha=canonical['alpha'][y:y+ch,x:x+cw],
                examples=[f],samples=len(images),model_logo_only=True,require_whole=True))
            result[-1]['component_only']=True
    flipped_families={}
    for proto in list(result):
        if len(result)>=settings.maximum_prototypes:break
        family=proto['family'];flipped_families.setdefault(family,len(result))
        a,_,c,_=proto['box'];x,y,r,s=proto['anchor']
        result.append(dict(proto,id=len(result),family=flipped_families[family],
             anchor=[a+c-r,y,a+c-x,s],feature=cv2.flip(proto['feature'],1),
             mask=cv2.flip(proto['mask'],1),alpha=cv2.flip(proto['alpha'],1),
             whole_feature=cv2.flip(proto['whole_feature'],1),examples=[cv2.flip(z,1) for z in proto['examples']],
             require_whole=True))
    from .hls_settings import HLSSettings
    return learn_logo_prototypes(files,rows,labels,result,settings,fps or HLSSettings.from_env().model_fps,scenes)


def learn_logo_prototypes(files,rows,labels,prototypes,cfg,fps,scenes=None):
    """从多帧YOLO图形观测学习移动叠加层，文字本身不能创建品牌模板。"""
    observed=[candidates_for_frame(cv2.imread(str(file)),rows[i],prototypes,labels,cfg)
              for i,file in enumerate(files)]
    tracks=associate(observed,[[0,len(files)]],fps,cfg)
    eligible=[t for t in tracks if t['kind']=='logo' and len(t['hints'])>=cfg.minimum_template_frames]
    for t in sorted(eligible,key=lambda t:-len(t['hints'])):
        if len(prototypes)>=cfg.maximum_prototypes:break
        seed=max(t['hints'],key=lambda p:p['confidence'])
        if seed['confidence']<cfg.seed_confidence:continue
        aspect=lambda b:(b[2]-b[0])/max(1,b[3]-b[1])
        hints=[p for p in t['hints'] if not p.get('interpolated') and max(aspect(p['model_box']),aspect(seed['model_box']))/
               min(aspect(p['model_box']),aspect(seed['model_box']))<1.3]
        if len(hints)<cfg.minimum_template_frames:continue
        if scenes and len({j for p in hints for j,(a,b) in enumerate(scenes) if a<=p['frame']<b})<2:continue
        chosen=[hints[k] for k in np.unique(np.linspace(0,len(hints)-1,min(cfg.template_samples,len(hints)),dtype=int))]
        anchor=seed['model_box'];box=seed['box'];w,h=map(lambda x:max(2,round(x)),[box[2]-box[0],box[3]-box[1]])
        aw,ah=map(lambda x:max(2,round(x)),[anchor[2]-anchor[0],anchor[3]-anchor[1]])
        examples=[];whole=[]
        for p in chosen:
            image=cv2.imread(str(files[p['frame']]))
            examples.append(highpass(patch(image,p['model_box'],(aw,ah))))
            whole.append(highpass(patch(image,p['box'],(w,h))))
        feature=np.median(np.stack(examples),axis=0)
        if any(feature_score(p['examples'][0],feature)>.8 for p in prototypes):continue
        full=np.median(np.stack(whole),axis=0)
        noise=float(np.median(np.abs(full)))
        signal=(np.abs(full)>max(.5,noise*5)).astype(np.uint8)*255
        from .matte import ink_support
        mask=ink_support(full,signal,np.full(signal.shape,255,np.uint8),cfg.support_radius)
        if np.count_nonzero(mask)<16:continue
        proto=dict(id=len(prototypes),family=len(prototypes),anchor=anchor,box=box,feature=feature,mask=mask,
                   whole_feature=full,examples=examples[::max(1,len(examples)//5)],
                   samples=len(chosen),evidence='多帧YOLO图形印证的本输入时域模板')
        images=[patch(cv2.imread(str(files[p['frame']])),p['box'],(w,h)) for p in chosen]
        proto['alpha']=learn_opacity(images,mask,cfg.minimum_template_frames)
        prototypes.append(proto)
        # 同一图形中的模型文字部件可作为替代锚点；仍需完整图形模板复核。
        seed_image=cv2.imread(str(files[seed['frame']]))
        for raw in rows[seed['frame']]['boxes']:
            if len(prototypes)>=cfg.maximum_prototypes:break
            if ('text' not in labels.get(raw['class_id'],'') or raw['confidence']<cfg.seed_confidence
                    or intersection_fraction(raw['box'],box)<.95):continue
            crop=patch(mask,[raw['box'][0]-box[0],raw['box'][1]-box[1],raw['box'][2]-box[0],raw['box'][3]-box[1]])
            if np.count_nonzero(crop)/crop.size<.2:continue
            z=highpass(patch(seed_image,raw['box']))
            prototypes.append(dict(proto,id=len(prototypes),anchor=raw['box'],feature=z,examples=[z],
                                   require_whole=True))
    return prototypes


def candidates_for_frame(image,row,prototypes,labels,cfg):
    blue=[];logos=[]
    for raw in row['boxes']:
        if raw['confidence']<cfg.anchor_confidence:continue
        label=labels.get(raw['class_id'],'')
        if 'text' in label or 'logo' in label:
            feature=highpass(patch(image,raw['box']))
            scored=[(prototype_score(image,raw,p),p) for p in prototypes
                    if not p.get('model_logo_only') or 'logo' in label
                    if max((raw['box'][2]-raw['box'][0])/(raw['box'][3]-raw['box'][1]),
                           (p['anchor'][2]-p['anchor'][0])/(p['anchor'][3]-p['anchor'][1])) /
                       max(.01,min((raw['box'][2]-raw['box'][0])/(raw['box'][3]-raw['box'][1]),
                                   (p['anchor'][2]-p['anchor'][0])/(p['anchor'][3]-p['anchor'][1])))<1.8]
            score,proto=max(scored,key=lambda x:x[0],default=(0,None))
            if proto is not None and score>=cfg.template_correlation:
                raw_box=projected_box(proto,raw['box'])
                box=clip_box(raw_box,image.shape[:2])
                blue.append(dict(box=box,template_box=raw_box,confidence=raw['confidence'],kind='prototype',prototype=proto['id'],family=proto.get('family',proto['id']),component_only=proto.get('component_only',False),model_box=raw['box'],template_score=score))
                continue
            if 'text' in label:continue
        if 'logo' in label:
            if 'corner' in label and raw['confidence']<cfg.seed_confidence:continue
            a,b,c,d=raw['box'];height=d-b
            box=clip_box([a-height*cfg.logo_horizontal_padding,b-height*cfg.logo_vertical_padding,
                          c+height*cfg.logo_horizontal_padding,d+height*cfg.logo_vertical_padding],image.shape[:2])
            logos.append(dict(box=box,confidence=raw['confidence'],kind='logo',prototype=-1,model_box=raw['box']))
    selected=[]
    for p in sorted(blue,key=lambda p:(p.get('component_only',False),-p['confidence']))+sorted(logos,key=lambda p:p['confidence'],reverse=True):
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
                    if p['kind']=='prototype' and t.get('family')!=p.get('family',p['prototype']):continue
                    old=t['hints'][-1]['box'];box=p['box']
                    overlap=min(intersection_fraction(old,box),intersection_fraction(box,old))
                    distance=math.hypot((old[0]+old[2]-box[0]-box[2])/2,(old[1]+old[3]-box[1]-box[3])/2)
                    size=max(old[2]-old[0],old[3]-old[1])
                    if overlap>.1 or distance<size*.5:eligible.append((overlap-distance/max(1,size),t))
                if eligible:t=max(eligible,key=lambda x:x[0])[1]
                else:
                    t=dict(id=next_id,kind=p['kind'],prototype=p['prototype'],family=p.get('family',p['prototype']),hints=[],last=i)
                    next_id+=1;active.append(t)
                t['hints'].append(dict(p,frame=i,track_id=t['id']));t['last']=i;used.add(t['id'])
        for t in active:
            t.pop('last');t['hints']=bridge_gaps(t['hints'],end,max_gap=max(1,round(fps*cfg.maximum_gap_seconds)))
            tracks.append(t)
    return tracks


def build(files,scenes,rows,prior,labels,root,fps=30,settings=None,check=lambda:None):
    cfg=settings or ModelTrackSettings()
    prototypes=learn_text_prototypes(files,rows,prior,labels,cfg,fps,scenes)
    candidates=[]
    for i,file in enumerate(files):
        check();candidates.append(candidates_for_frame(cv2.imread(str(file)),rows[i],prototypes,labels,cfg))
    tracks=associate(candidates,scenes,fps,cfg)
    unconfirmed=[t for t in tracks if t['kind']!='prototype']
    tracks=[t for t in tracks if t['kind']=='prototype']
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
        alpha=cv2.resize(proto['alpha'],size,interpolation=cv2.INTER_LINEAR)
        np.savez_compressed(templates/f'{t["id"]}.npz',feature=feature,mask=base,alpha=alpha)
        t['template']=str(templates/f'{t["id"]}.npz')
        # 提示来自同一输入的自动模型与模板，供SAM分割细化；无人工坐标或帧号。
        selected_prompts={p['frame'] for p in sampled_hints(hints,hints[0]['frame'],hints[-1]['frame']+1)}
        for p in hints:
            if p['frame'] not in selected_prompts:continue
            mask=place(base,p.get('template_box',p['box']),shape)
            path=prompts/f'{t["id"]}-{p["frame"]}.png';cv2.imwrite(str(path),mask)
            p['mask_prompt']=str(path.relative_to(root))
    result=dict(schema='yolo-model-tracks-1',multitrack=True,hls=True,scenes=scenes,tracks=tracks,
                unconfirmed_model_tracks=unconfirmed,
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
    byframe={};maximum={};areas={};object_masks=root/'validated-objects';object_masks.mkdir(exist_ok=True)
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
            areas.setdefault(track['id'],{})[i]=fraction
            cv2.imwrite(str(object_masks/str(track['id'])/file.name),mask);joint|=mask
        cv2.imwrite(str(root/'masks'/file.name),joint)
        manifests.append(dict(frame_index=i,mask_key='masks/'+file.name,sha256=sha256(root/'masks'/file.name)))
    bins=temporal_bins(areas);layers=[]
    for k,identities in enumerate(bins):
        directory=root/f'layer-{k:02d}'/'masks';directory.mkdir(parents=True,exist_ok=True)
        layers.append(directory)
    for file in files:
        image=cv2.imread(str(root/'masks'/file.name),0);raw=[]
        for identities in bins:
            joint=np.zeros_like(image)
            for identity in identities:
                p=object_masks/str(identity)/file.name
                if p.exists():joint|=cv2.imread(str(p),0)
            raw.append(joint)
        for directory,joint in zip(layers,disjoint_masks(raw)):
            cv2.imwrite(str(directory/file.name),joint)
    write_manifest(root,root/'detections.json',layers,files)
    verified=VerifiedLayers(root/'layers.json',root/'detections.json')
    for i,file in enumerate(files):verified.validate(i,cv2.imread(str(root/'masks'/file.name),0))
    (root/'model-mask-summary.json').write_text(json.dumps(dict(层数=len(layers),轨迹数=len(detections['tracks']),逐帧=manifests,最大轨迹面积=maximum,
          分层方式='逐帧实际面积装箱，图层像素互斥，联合掩膜保持不变'),ensure_ascii=False))
    return manifests
