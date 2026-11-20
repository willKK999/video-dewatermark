"""v11：形状边缘判据、跨窗口关联、字幕优先与OCR双向短传播。只出掩膜。"""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
import time
from pathlib import Path
import cv2
import numpy as np
from worker.v10_masks import Settings as V10Settings, rectangle, text_tracks as v10_text_tracks, temporal_fill, write_json
from worker.ocr_temporal import ExperimentalSettings, associate_candidates, normalize_text
from worker.sam31_masks import digest, write_png
from worker.hls_layers import track_bins, write_manifest, VerifiedLayers


@dataclass(frozen=True)
class Settings(V10Settings):
    text_padding: int = 0
    text_outer_padding: int = 1
    propagation_seconds: float = .6
    propagation_min_seconds: float = 2.
    propagation_fit_observations: int = 4
    propagation_residual_fraction: float = .25
    shape_size: int = 64
    screen_iou: float = .85
    screen_position_pixels: float = 2.
    template_iou: float = .78
    smooth_acceleration_pixels: float = 2.
    shape_edge_tolerance: int = 1
    background_edge_fraction: float = .65
    background_containment: float = .75
    background_pair_fraction: float = .6
    shape_minimum_pairs: int = 4
    occlusion_interior_fraction: float = .08
    occlusion_component_fraction: float = .04
    occlusion_erosion_pixels: int = 2
    occlusion_minimum_samples: int = 1
    occlusion_cover_fraction: float = .3
    occlusion_foreground_points: int = 2
    occluder_motion_pixels: float = 3.
    link_gap_frames: int = 6
    link_iou: float = .35
    duplicate_iou: float = .65
    link_template_iou: float = .5
    background_feature_count: int = 1800
    background_feature_quality: float = .01
    background_feature_distance: int = 5
    background_exclusion_pixels: int = 4
    background_context_pixels: int = 64
    background_context_fraction: float = .5
    background_minimum_features: int = 6
    background_inlier_fraction: float = .55
    background_ransac_pixels: float = 2.
    background_lk_window: int = 21
    background_lk_levels: int = 3
    background_lk_error: float = 30.
    ocr_protect_samples: int = 3
    ocr_protect_fraction: float = .6
    ocr_neighbor_heights: float = 2.5
    ocr_offset_residual: float = .4
    ocr_size_variation: float = .25

    def __post_init__(self):
        super().__post_init__()
        for name in ('screen_iou','template_iou','background_edge_fraction','background_containment',
                     'background_pair_fraction','occlusion_interior_fraction','occlusion_component_fraction',
                     'link_iou','duplicate_iou','link_template_iou','background_feature_quality',
                     'background_inlier_fraction','ocr_protect_fraction'):
            if not 0 < getattr(self,name) <= 1: raise ValueError('无效比例：'+name)
        for name in ('propagation_seconds','propagation_min_seconds','propagation_fit_observations',
                     'propagation_residual_fraction','shape_size','shape_minimum_pairs','link_gap_frames',
                     'background_minimum_features','ocr_protect_samples','ocr_neighbor_heights','ocr_offset_residual'):
            if getattr(self,name)<=0: raise ValueError('无效配置：'+name)
        if self.text_padding>1 or self.text_outer_padding>2: raise ValueError('文字紧框不得大幅膨胀')


def read_mask(path, shape=None):
    m=cv2.imread(str(path),0)
    if m is None or (shape is not None and m.shape!=shape) or np.any((m!=0)&(m!=255)):
        raise ValueError('缺失、非二值或尺寸不符的掩膜：'+str(path))
    return m


def check_area(m):
    f=float(np.count_nonzero(m)/m.size)
    if f>.2: raise ValueError('单对象/单层超过20%保护')
    return f


def strict_subtract(mask, subtitle):
    return mask & ~subtitle


def canonical_tracks(rows, scenes, cfg):
    _,_,old=v10_text_tracks(rows,scenes,cfg)
    seeds=old['seeds'];brands={};subtitles=defaultdict(list)
    for key,observations in rows.items():
        brands[key]=[];i=int(key)
        end=next(b for a,b in scenes if a<=i<b)
        for p in observations:
            word=normalize_text(p.get('text',''))
            score,canonical=max([(SequenceMatcher(None,word,s).ratio(),s) for s in seeds],default=(0.,''))
            if word and score>=cfg.canonical_similarity:
                # 先归一识别变体再检查逐轨内容稳定，避免相似品牌变体互相稀释。
                brands[key].append(dict(p,text=canonical,original_text=p.get('text','')))
            else:
                for j in range(i,min(i+cfg.stride,end)): subtitles[j].append(p)
    tracking=ExperimentalSettings(stride=cfg.stride,association_iou=cfg.association_iou,
        maximum_gap_seconds=cfg.maximum_gap_seconds,minimum_observations=cfg.minimum_observations,
        minimum_duration_seconds=cfg.minimum_track_seconds,ocr_minimum_duration_seconds=cfg.minimum_track_seconds,
        ocr_position_jitter=cfg.moving_position_jitter,ocr_dominant_text_fraction=cfg.dominant_text_fraction)
    tracks,rejected=associate_candidates(brands,scenes,cfg.fps,tracking)
    for t in tracks:
        obs=t['observations'];a,b=scenes[t['scene']]
        if (obs[-1]['frame']-obs[0]['frame'])/cfg.fps < cfg.propagation_min_seconds: continue
        extra=[]
        for front in (True,False):
            samples=obs[:cfg.propagation_fit_observations] if front else obs[-cfg.propagation_fit_observations:]
            frames=np.array([p['frame'] for p in samples],float)
            boxes=np.array([p['box'] for p in samples],float)
            x=np.column_stack([frames-frames.mean(),np.ones(len(frames))])
            coef=np.linalg.lstsq(x,boxes,rcond=None)[0]
            residual=np.max(np.abs(x@coef-boxes))
            height=max(1.,np.median(boxes[:,3]-boxes[:,1]))
            if residual/height>cfg.propagation_residual_fraction: continue
            edge=obs[0]['frame'] if front else obs[-1]['frame'];gap=round(cfg.propagation_seconds*cfg.fps)
            indices=range(max(a,edge-gap),edge) if front else range(edge+1,min(b,edge+gap+1))
            for i in indices:
                box=(np.array([i-frames.mean(),1.])@coef).tolist()
                if box[2]<=box[0] or box[3]<=box[1]:continue
                extra.append(dict(samples[0],frame=i,box=box,temporal=True,propagated=True))
        t['hints']=sorted(t['hints']+extra,key=lambda h:h['frame'])
    return tracks,dict(subtitles),dict(seeds=seeds,rejected=rejected,normalization_before_stability=True)


def bbox(mask):
    y,x=np.where(mask>0)
    if not len(x):return np.array([0.,0.,0.,0.])
    return np.array([x.min(),y.min(),x.max()+1,y.max()+1],float)


def overlap(a,b):
    aa=a>0;bb=b>0
    return float(np.count_nonzero(aa&bb)/max(1,np.count_nonzero(aa|bb)))


def template(mask,size):
    x,y,a,b=bbox(mask).astype(int)
    if a<=x or b<=y:return np.zeros((size,size),np.uint8)
    return cv2.resize(mask[y:b,x:a],(size,size),interpolation=cv2.INTER_NEAREST)


def edge(mask):
    return cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))


def shape_metrics(first,second,matrix,reference,reference_matrix,cfg,occluders=None,correspondences=None):
    h,w=first.shape
    warped=cv2.warpAffine(first,matrix,(w,h),flags=cv2.INTER_NEAREST)
    ref=cv2.warpAffine(reference,reference_matrix,(w,h),flags=cv2.INTER_NEAREST)
    k=np.ones((2*cfg.shape_edge_tolerance+1,)*2,np.uint8)
    near=cv2.dilate(edge(warped),k)>0;edges=edge(second)>0
    edgefit=np.count_nonzero(near&edges)/max(1,np.count_nonzero(edges))
    containment=np.count_nonzero((second>0)&(cv2.dilate(warped,k)>0))/max(1,np.count_nonzero(second))
    interior=cv2.erode(ref,np.ones((2*cfg.occlusion_erosion_pixels+1,)*2,np.uint8))>0
    missing=interior & ~(cv2.dilate(second,k)>0)
    n,_,components,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8))
    largest=max(components[1:,cv2.CC_STAT_AREA],default=0)
    denominator=max(1,np.count_nonzero(interior))
    cover=float(np.count_nonzero(missing & (occluders>0))/max(1,np.count_nonzero(missing))) if occluders is not None else 0.
    foreground=0
    if correspondences is not None:
        px,py=correspondences
        predicted=px@matrix[:,:2].T+matrix[:,2]
        different=np.linalg.norm(py-predicted,axis=1)>=cfg.occluder_motion_pixels
        ids=np.where(different)[0];coords=np.rint(py[ids]).astype(int)
        good=(coords[:,0]>=0)&(coords[:,0]<w)&(coords[:,1]>=0)&(coords[:,1]<h)
        coords=coords[good]
        foreground=int(np.count_nonzero(missing[coords[:,1],coords[:,0]]))
    box=bbox(first);center=np.r_[(box[:2]+box[2:])/2,1.]
    background_motion=float(np.linalg.norm(matrix@center-center[:2])/cfg.flow_scale)
    return dict(screen_iou=overlap(first,second),template_iou=overlap(template(first,cfg.shape_size),template(second,cfg.shape_size)),
                background_edges=float(edgefit),background_containment=float(containment),
                occlusion_fraction=float(np.count_nonzero(missing)/denominator),
                occlusion_component=float(largest/denominator),occlusion_cover_fraction=cover,occlusion_foreground_points=foreground,background_motion=background_motion)


def ocr_protection(samples, tracks, cfg):
    evidence=[]
    for t in tracks:
        hints={p['frame']:p for p in t['hints']};relations=[]
        for i,mask in samples.items():
            if i not in hints:continue
            box=np.array(hints[i]['box']);mb=bbox(mask)/cfg.flow_scale
            center=(mb[:2]+mb[2:])/2;tc=(box[:2]+box[2:])/2
            height=max(1.,box[3]-box[1]);size=np.maximum(1,mb[2:]-mb[:2])
            # 只用候选轮廓和OCR几何关系；没有掩膜内部图像运动。
            near=(center[0]>=box[0]-cfg.ocr_neighbor_heights*height and center[0]<=box[2]+cfg.ocr_neighbor_heights*height
                  and abs(center[1]-tc[1])<=cfg.ocr_neighbor_heights*height)
            relations.append((near,(center-tc)/height,size/height))
        close=[p for p in relations if p[0]]
        if len(close)<cfg.ocr_protect_samples or len(close)/max(1,len(relations))<cfg.ocr_protect_fraction:continue
        offsets=np.array([p[1] for p in close]);sizes=np.array([p[2] for p in close])
        residual=float(np.max(np.median(np.abs(offsets-np.median(offsets,axis=0)),axis=0)))
        variation=float(np.max(np.median(np.abs(sizes-np.median(sizes,axis=0)),axis=0)/np.maximum(1.,np.median(sizes,axis=0))))
        if residual<=cfg.ocr_offset_residual and variation<=cfg.ocr_size_variation:
            evidence.append(dict(track=t['id'],samples=len(close),offset_residual=residual,size_variation=variation))
    return evidence


def decide_shape(pairs, protected, cfg):
    valid=[p for p in pairs if p.get('valid')]
    moving=[p for p in valid if p['background_motion']>=cfg.background_min_motion]
    aligned=[p for p in moving if p['background_edges']>=cfg.background_edge_fraction
             and p['background_containment']>=cfg.background_containment]
    occlusion=[p for p in valid if p['occlusion_fraction']>=cfg.occlusion_interior_fraction
               and p['occlusion_component']>=cfg.occlusion_component_fraction
               and (p.get('occlusion_cover_fraction',0)>=cfg.occlusion_cover_fraction or p.get('occlusion_foreground_points',0)>=cfg.occlusion_foreground_points)]
    fixed=bool(valid) and max(p.get('screen_position_delta',0.) for p in valid)<=cfg.screen_position_pixels and sum(p['screen_iou']>=cfg.screen_iou for p in valid)/len(valid)>=cfg.ocr_protect_fraction
    own=sum(p['template_iou']>=cfg.template_iou and p['background_edges']<cfg.background_edge_fraction
            and p.get('acceleration',0)<=cfg.smooth_acceleration_pixels for p in valid)/max(1,len(valid))>=cfg.ocr_protect_fraction
    reject=not (protected or fixed or own) and len(moving)>=cfg.shape_minimum_pairs and len(aligned)/max(1,len(moving))>=cfg.background_pair_fraction and len(occlusion)>=cfg.occlusion_minimum_samples
    return dict(rejected=bool(reject),ocr_protected=bool(protected),screen_protected=bool(fixed),own_motion_protected=bool(own),
                valid_pairs=len(valid),moving_pairs=len(moving),aligned_pairs=len(aligned),occlusion_samples=len(occlusion),
                consistency=len(aligned)/max(1,len(moving)),ocr_evidence=protected,pairs=pairs)


def group_nodes(nodes,cfg):
    keys=list(nodes);parent={k:k for k in keys}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    links=[]
    for index,k in enumerate(keys):
        n=nodes[k]
        for q in keys[index+1:]:
            other=nodes[q]
            if n['scene']!=other['scene']:continue
            common=sorted(set(n['samples'])&set(other['samples']))
            linked=False;score=0.
            if common:
                selected=[common[0],common[len(common)//2],common[-1]]
                score=float(np.median([overlap(n['samples'][i],other['samples'][i]) for i in selected]))
                linked=score>=cfg.duplicate_iou
            else:
                left,right=(n,other) if max(n['frames'])<min(other['frames']) else (other,n)
                gap=min(right['frames'])-max(left['frames'])
                if 0<gap<=cfg.link_gap_frames:
                    a=left['last'];b=right['first'];score=overlap(a,b)
                    linked=score>=cfg.link_iou and overlap(template(a,cfg.shape_size),template(b,cfg.shape_size))>=cfg.link_template_iou
            if linked:
                parent[find(q)]=find(k);links.append(dict(left=k,right=q,iou=score))
    groups=defaultdict(list)
    for k in keys:groups[find(k)].append(k)
    return list(groups.values()),links


def background_matches(first,second,excluded_first,excluded_second,cfg):
    allowed=255-cv2.dilate(excluded_first,np.ones((2*cfg.background_exclusion_pixels+1,)*2,np.uint8))
    points=cv2.goodFeaturesToTrack(first,maxCorners=cfg.background_feature_count,qualityLevel=cfg.background_feature_quality,
                                 minDistance=cfg.background_feature_distance,mask=allowed)
    if points is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
    next_points,status,error=cv2.calcOpticalFlowPyrLK(first,second,points,None,
        winSize=(cfg.background_lk_window,)*2,maxLevel=cfg.background_lk_levels)
    if next_points is None:return np.empty((0,2),np.float32),np.empty((0,2),np.float32)
    x=points[:,0];y=next_points[:,0];height,width=first.shape
    okay=(status[:,0]>0)&(error[:,0]<=cfg.background_lk_error)&np.isfinite(y).all(axis=1)
    okay&=(y[:,0]>=0)&(y[:,0]<width)&(y[:,1]>=0)&(y[:,1]<height)
    ids=np.where(okay)[0];nearest=np.rint(y[ids]).astype(int)
    nearest[:,0]=np.clip(nearest[:,0],0,width-1);nearest[:,1]=np.clip(nearest[:,1],0,height-1)
    ids=ids[excluded_second[nearest[:,1],nearest[:,0]]==0]
    return x[ids],y[ids]


def analyze_shapes(args,cfg,data,metrics,files):
    started=time.monotonic();shape=read_mask(args.sam/'masks'/files[0].name).shape
    scaled=(round(shape[1]*cfg.flow_scale),round(shape[0]*cfg.flow_scale));nodes={}
    for key,t in metrics['tracks'].items():
        if not t['retained']:continue
        for scene,(a,b) in enumerate(data['scenes']):
            indices=[i for i in t['frames'] if a<=i<b]
            if not indices:continue
            selected=sorted(set([indices[0],indices[-1]]+[i for i in indices if i%cfg.stride==0]))
            samples={i:cv2.resize(read_mask(args.sam/'raw-object-masks'/key/files[i].name,shape),scaled,interpolation=cv2.INTER_NEAREST) for i in selected}
            nodes[f'{key}@{scene}']=dict(key=key,scene=scene,frames=indices,samples=samples,first=samples[indices[0]],last=samples[indices[-1]])
    groups,links=group_nodes(nodes,cfg)
    unified=defaultdict(set)
    for node in nodes.values(): unified[node['scene']].update(node['samples'])
    for node in nodes.values():
        present=set(node['frames'])
        for i in sorted(unified[node['scene']] & present):
            if i not in node['samples']:
                node['samples'][i]=cv2.resize(read_mask(args.sam/'raw-object-masks'/node['key']/files[i].name,shape),scaled,interpolation=cv2.INTER_NEAREST)
    group_data=[];exclusions=defaultdict(lambda:np.zeros((scaled[1],scaled[0]),np.uint8))
    for identities in groups:
        samples={}
        for key in identities:
            for i,m in nodes[key]['samples'].items():
                samples.setdefault(i,np.zeros_like(m));samples[i]|=m;exclusions[i]|=m
        scene=nodes[identities[0]]['scene']
        group_data.append(dict(members=identities,scene=scene,samples=samples,pairs=[],reference_matrix=np.eye(3),reference=None,last_velocity=None))
    # 同一镜头统一采样；窗口边界的端点也进入，背景对应只在候选掩膜之外提取。
    for scene,(a,b) in enumerate(data['scenes']):
        indices=sorted(i for i in exclusions if a<=i<b)
        for i,j in zip(indices,indices[1:]):
            if j-i>cfg.link_gap_frames:continue
            first=cv2.resize(cv2.imread(str(files[i]),0),scaled);second=cv2.resize(cv2.imread(str(files[j]),0),scaled)
            x,y=background_matches(first,second,exclusions[i],exclusions[j],cfg)
            for group in group_data:
                samples=group['samples']
                if group['scene']!=scene or i not in samples or j not in samples:continue
                m,n=samples[i],samples[j];box=bbox(m)
                pad=max(cfg.background_context_pixels*cfg.flow_scale,cfg.background_context_fraction*max(box[2:]-box[:2]))
                near=(x[:,0]>=box[0]-pad)&(x[:,0]<=box[2]+pad)&(x[:,1]>=box[1]-pad)&(x[:,1]<=box[3]+pad)
                px,py=x[near],y[near];record=dict(frame=i,next_frame=j,valid=False,features=len(px))
                if len(px)>=cfg.background_minimum_features:
                    matrix,inliers=cv2.estimateAffine2D(px,py,method=cv2.RANSAC,ransacReprojThreshold=cfg.background_ransac_pixels)
                    if matrix is not None and np.mean(inliers)>=cfg.background_inlier_fraction and np.isfinite(matrix).all():
                        if group['reference'] is None:group['reference']=m
                        full=np.vstack([matrix,[0,0,1]])
                        group['reference_matrix']=full@group['reference_matrix']
                        record.update(shape_metrics(m,n,matrix,group['reference'],group['reference_matrix'][:2],cfg,occluders=exclusions[j]&~n,correspondences=(px,py)),valid=True,inlier_fraction=float(np.mean(inliers)))
                if not record['valid']:
                    # 无可靠背景模型时仍可给屏幕/自身模板保护，绝不据此拒绝。
                    record.update(screen_iou=overlap(m,n),template_iou=overlap(template(m,cfg.shape_size),template(n,cfg.shape_size)))
                    group['reference']=None;group['reference_matrix']=np.eye(3)
                velocity=((bbox(n)[:2]+bbox(n)[2:]-bbox(m)[:2]-bbox(m)[2:])/2)/max(1,j-i)/cfg.flow_scale
                record['screen_position_delta']=float(np.max(np.abs(bbox(n)-bbox(samples[min(samples)])))/cfg.flow_scale)
                record['acceleration']=float(np.linalg.norm(velocity-group['last_velocity'])) if group['last_velocity'] is not None else 0.
                group['last_velocity']=velocity;group['pairs'].append(record)
        print(f'形状与外部背景关联完成 镜头{scene} {a}:{b}',flush=True)
    audit={};decisions={}
    for index,group in enumerate(group_data):
        protection=ocr_protection(group['samples'],[t for t in data['tracks'] if t['scene']==group['scene']],cfg)
        decision=decide_shape(group['pairs'],protection,cfg);decision.update(members=group['members'],scene=group['scene'])
        audit[str(index)]=decision
        for key in group['members']:decisions[key]=str(index)
    write_json(args.output/'motion-audit.json',audit);write_json(args.output/'cross-window-links.json',links)
    return audit,decisions,time.monotonic()-started


def prepare(args,cfg):
    started=time.monotonic();root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'detections.json').exists():raise ValueError('禁止覆盖检查点')
    files=sorted((args.v10/'frames').glob('*.png'))
    if not files:raise ValueError('缺少解码缓存')
    source=json.loads((args.sam/'source.json').read_text())
    if {f.name:digest(f) for f in files}!=source['frames']:raise ValueError('缓存输入绑定不一致')
    old=json.loads((args.v10/'detections.json').read_text());rows=json.loads(args.ocr.read_text())['rows']
    tracks,subs,audit=canonical_tracks(rows,old['scenes'],cfg)
    (root/'frames').symlink_to((args.v10/'frames').resolve(),target_is_directory=True)
    write_json(root/'detections.json',dict(tracks=tracks,subtitles=subs,scenes=old['scenes'],settings=asdict(cfg),audit=audit,
        hints=[p for t in tracks for p in t['hints']],bindings=dict(ocr_sha256=digest(args.ocr),sam_source_sha256=digest(args.sam/'source.json'),
            sam_metrics_sha256=digest(args.sam/'metrics.json'),v10_detections_sha256=digest(args.v10/'detections.json'))))
    write_json(root/'prepare-timing.json',dict(seconds=time.monotonic()-started,frames=len(files),ocr_cache_reused=True))
    print(f'准备完成：{len(files)}帧、{len(tracks)}条文字轨迹',flush=True)


def segment(args,cfg):
    started=time.monotonic();root=args.output;data=json.loads((root/'detections.json').read_text());old=json.loads((args.v10/'detections.json').read_text())
    def identity(i,box):return (i,tuple(round(float(x),6) for x in box))
    cache={identity(p['frame'],p['box']):args.v10/'ocr-sam-raw'/str(t['id'])/f'{p["frame"]:06d}.png' for t in old['tracks'] for p in t['hints']}
    needed=defaultdict(list);reused=0
    for t in data['tracks']:
        for h in t['hints']:
            target=root/'ocr-sam-raw'/str(t['id'])/f'{h["frame"]:06d}.png';source=cache.get(identity(h['frame'],h['box']))
            if source is not None and source.exists():
                target.parent.mkdir(parents=True,exist_ok=True);target.hardlink_to(source);reused+=1
            else:needed[h['frame']].append((t,h))
    loaded=0.
    if needed:
        import os,torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from worker.gpu_lock import device_lock
        load=time.monotonic()
        with device_lock(),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor=SAM2ImagePredictor(build_sam2(os.environ['SAM_CONFIG'],os.environ['SAM_CHECKPOINT'],device='cuda'));loaded=time.monotonic()-load
            for index,items in sorted(needed.items()):
                im=cv2.imread(str(root/'frames'/f'{index:06d}.png'));predictor.set_image(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
                for t,h in items:
                    masks,scores,_=predictor.predict(box=np.array(h['box']),multimask_output=True)
                    write_png(root/'ocr-sam-raw'/str(t['id'])/f'{index:06d}.png',masks[int(np.argmax(scores))].astype(np.uint8)*255)
                if index%50==0:print(f'仅新增框提示 {index}',flush=True)
    write_json(root/'sam2-summary.json',dict(seconds=time.monotonic()-started,load_seconds=loaded,reused=reused,new=sum(map(len,needed.values())),failures=[]))


def merge(args,cfg):
    root=args.output;started=time.monotonic();files=sorted((root/'frames').glob('*.png'));shape=cv2.imread(str(files[0]),0).shape
    data=json.loads((root/'detections.json').read_text());metrics=json.loads((args.sam/'metrics.json').read_text())
    audit,decisions,analysis_seconds=analyze_shapes(args,cfg,data,metrics,files)
    subregions={}
    def subtitle(i):
        if i not in subregions:
            m=np.zeros(shape,np.uint8)
            for p in data['subtitles'].get(str(i),[]):m|=rectangle(p['box'],shape,cfg.subtitle_padding)
            subregions[i]=m
        return subregions[i]
    # 按轨迹逐个过滤后存盘，保持逐对象20%保护，不保存全片RGB副本。
    maxima={};byframe=defaultdict(list);filters={};stage=time.monotonic()
    for key,t in metrics['tracks'].items():
        if not t['retained']:continue
        masks={i:read_mask(args.sam/'raw-object-masks'/key/files[i].name,shape) for i in t['frames']}
        valid=len(masks)>=cfg.minimum_observations and (max(masks)-min(masks))/cfg.fps>=cfg.minimum_track_seconds
        valid=valid and all(np.count_nonzero(m)/m.size<=.2 for m in masks.values())
        observed={};removed=0;rejected_frames=0
        for i,m in masks.items():
            scene=next(j for j,(a,b) in enumerate(data['scenes']) if a<=i<b);group=decisions[f'{key}@{scene}']
            reject=audit[group]['rejected'] or not valid
            kept=np.zeros(shape,np.uint8) if reject else strict_subtract(m,subtitle(i))
            rejected_frames+=bool(reject);removed+=int(np.count_nonzero(m&~kept))
            if np.any(kept):observed[i]=kept
        extra=temporal_fill(observed,data['scenes'],cfg)
        for kind,items,folder in [('sam',observed,'sam-objects'),('temporal',extra,'temporal-objects')]:
            maximum=0.
            for i,m in items.items():
                m=strict_subtract(m,subtitle(i));maximum=max(maximum,check_area(m))
                if np.any(m):write_png(root/folder/key/files[i].name,m);byframe[i].append((kind+'/'+key,folder,kind=='temporal'))
            if maximum:maxima[kind+'/'+key]=maximum
        filters[key]=dict(removed=removed,retained_frames=len(observed),rejected_frames=rejected_frames,temporal_frames=len(extra))
        print(f'过滤对象 {key}',flush=True)
    failures=[];summary=json.loads((root/'sam2-summary.json').read_text())
    for t in data['tracks']:
        raw={h['frame']:read_mask(root/'ocr-sam-raw'/str(t['id'])/files[h['frame']].name,shape) for h in t['hints']}
        if any(np.count_nonzero(m)/m.size>.2 for m in raw.values()):
            failures.append(dict(track=t['id'],reason='原始SAM2对象超过20%'));continue
        maximum=0.
        for h in t['hints']:
            i=h['frame'];support=rectangle(h['box'],shape,cfg.text_padding);outer=rectangle(h['box'],shape,cfg.text_outer_padding)
            kept=strict_subtract((raw[i]&outer)|support,subtitle(i));maximum=max(maximum,check_area(kept))
            write_png(root/'ocr-objects'/str(t['id'])/files[i].name,kept)
            byframe[i].append(('ocr/'+str(t['id']),'ocr-objects',h['temporal']))
        maxima['ocr/'+str(t['id'])]=maximum
    summary['failures']=failures;write_json(root/'sam2-summary.json',summary)
    bins=track_bins({k:v for k,v in maxima.items() if v},limit=.2) or [[]];directories=[root/'layers'/f'{j:02d}' for j in range(len(bins))]
    rows=[]
    for i,file in enumerate(files):
        sources={s:np.zeros(shape,np.uint8) for s in ['SAM','OCR','temporal']};components={}
        for key,folder,is_temporal in byframe[i]:
            m=read_mask(root/folder/key.split('/')[1]/file.name,shape);components[key]=m
            sources[{'sam':'SAM','ocr':'OCR','temporal':'temporal'}[key.split('/')[0]]]|=m
            if is_temporal:sources['temporal']|=m
        union=np.zeros(shape,np.uint8)
        for keys,directory in zip(bins,directories):
            m=np.zeros(shape,np.uint8)
            for key in keys:
                if key in components:m|=components[key]
            check_area(m);union|=m;write_png(directory/file.name,m)
        if np.any(union&subtitle(i)):raise ValueError('字幕优先级违规')
        original=read_mask(args.sam/'masks'/file.name,shape);v10=read_mask(args.v10/'masks'/file.name,shape)
        removed=original&~sources['SAM'];write_png(root/'removed'/file.name,removed)
        for name,m in sources.items():write_png(root/name/file.name,m)
        write_png(root/'masks'/file.name,union)
        rows.append(dict(frame=i,pixels=int(np.count_nonzero(union)),sources={k:int(np.count_nonzero(v)) for k,v in sources.items()},
            sam_added=int(np.count_nonzero(union&~original)),sam_removed=int(np.count_nonzero(original&~union)),
            v10_added=int(np.count_nonzero(union&~v10)),v10_removed=int(np.count_nonzero(v10&~union)),filtered=int(np.count_nonzero(removed)),
            subtitle_overlap=int(np.count_nonzero(union&subtitle(i)))))
        if i%100==0:print(f'合并与字幕交集核验 {i}/{len(files)}',flush=True)
    write_manifest(root,root/'detections.json',directories,files);verified=VerifiedLayers(root/'layers.json',root/'detections.json')
    for i,file in enumerate(files):
        layers=verified.masks(i);expected=np.bitwise_or.reduce(layers)
        if not np.array_equal(expected,read_mask(root/'masks'/file.name,shape)):raise ValueError('严格并集核验失败')
    write_json(root/'statistics.json',dict(frames=rows,filters=filters,layers=len(directories),area_protection_passed=True,
        timings=dict(shape_and_background=analysis_seconds,filter_merge_verify=time.monotonic()-stage,total=time.monotonic()-started)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','segment','merge'])
    for name in ['config','output','v10']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sam',type=Path);p.add_argument('--ocr',type=Path)
    args=p.parse_args();cfg=Settings(**json.loads(args.config.read_text()))
    {'prepare':prepare,'segment':segment,'merge':merge}[args.stage](args,cfg)


if __name__=='__main__':main()
