"""v12：叠加层证据约束OCR保护，独立于OCR的对白候选；仅掩膜。"""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
import cv2
import numpy as np
from worker import v11_masks as prior
from worker.v11_masks import (read_mask, check_area, strict_subtract, rectangle, bbox, overlap, template,
    group_nodes, background_matches, ocr_protection, temporal_fill, write_json, write_png, digest,
    track_bins, write_manifest, VerifiedLayers)

from worker.json_types import json_native

def write_json(path,data):
    Path(path).write_text(json.dumps(json_native(data),ensure_ascii=False,indent=2))


@dataclass(frozen=True)
class Settings(prior.Settings):
    overlap_mae_limit: float = 35.
    background_center_pixels: float = 1.5
    background_center_height_fraction: float = .03
    overlay_independence_fraction: float = .5
    overlay_stable_fraction: float = .75
    overlay_occlusion_template_ceiling: float = .90
    ocr_comotion_pixels: float = 2.
    ocr_comotion_seconds: float = .5
    subtitle_roi_top: float = .52
    subtitle_roi_bottom: float = .94
    subtitle_core_luma: int = 205
    subtitle_core_saturation: int = 85
    subtitle_outline_luma: int = 105
    subtitle_outline_radius: int = 3
    subtitle_outline_fraction: float = .16
    subtitle_min_char_height: float = .010
    subtitle_max_char_height: float = .045
    subtitle_max_char_width: float = .10
    subtitle_min_char_area: int = 8
    subtitle_min_characters: int = 4
    subtitle_min_line_width: float = .18
    subtitle_max_line_width: float = .93
    subtitle_center_tolerance: float = .20
    subtitle_line_y_tolerance: float = .50
    subtitle_max_line_height: float = .06
    subtitle_signature_width: int = 128
    subtitle_signature_height: int = 32
    subtitle_signature_iou: float = .43
    subtitle_min_seconds: float = .25
    subtitle_max_seconds: float = 7.
    subtitle_min_observation_fraction: float = .6
    subtitle_max_gap_seconds: float = .20
    subtitle_max_y_jitter: float = .012
    subtitle_multiline_gap: float = .10
    subtitle_band_tolerance: float = .015
    subtitle_region_padding: int = 4

    def __post_init__(self):
        super().__post_init__()
        if not 0<self.overlap_mae_limit<=35:raise ValueError('接缝上限不得超过35')
        for n in ('overlay_independence_fraction','overlay_stable_fraction','overlay_occlusion_template_ceiling',
                  'subtitle_outline_fraction','subtitle_signature_iou','subtitle_min_observation_fraction'):
            if not 0<getattr(self,n)<=1:raise ValueError('无效比例：'+n)
        if not 0<self.subtitle_roi_top<self.subtitle_roi_bottom<=1:raise ValueError('无效字幕相对位置范围')
        if not 0<self.subtitle_min_seconds<self.subtitle_max_seconds:raise ValueError('无效句级时长范围')
        for n in ('background_center_pixels','ocr_comotion_pixels','subtitle_outline_radius','subtitle_min_characters',
                  'subtitle_min_char_area','subtitle_max_gap_seconds','subtitle_signature_width','subtitle_signature_height'):
            if getattr(self,n)<=0:raise ValueError('配置必须为正：'+n)


def shape_metrics(first,second,matrix,reference,reference_matrix,cfg,occluders=None,correspondences=None):
    result=prior.shape_metrics(first,second,matrix,reference,reference_matrix,cfg,occluders,correspondences)
    a=bbox(first);b=bbox(second);ca=(a[:2]+a[2:])/2;cb=(b[:2]+b[2:])/2
    error=float(np.linalg.norm(matrix@np.r_[ca,1.]-cb)/cfg.flow_scale)
    height=max(a[3]-a[1],b[3]-b[1])/cfg.flow_scale
    result['background_center_error']=error
    result['background_center_tolerance']=cfg.background_center_pixels+cfg.background_center_height_fraction*height
    ref=cv2.warpAffine(reference,reference_matrix,(first.shape[1],first.shape[0]),flags=cv2.INTER_NEAREST)
    result['occlusion_template_iou']=overlap(template(ref,cfg.shape_size),template(second,cfg.shape_size))
    return result


def strict_ocr_relation(samples,tracks,cfg):
    loose=ocr_protection(samples,tracks,cfg);accepted=[]
    for evidence in loose:
        track=next(t for t in tracks if t['id']==evidence['track']);hints={h['frame']:h for h in track['hints']}
        rows=[]
        for i,m in sorted(samples.items()):
            if i in hints:
                mb=bbox(m)/cfg.flow_scale;tb=np.array(hints[i]['box'])
                rows.append((i,(mb[:2]+mb[2:])/2-(tb[:2]+tb[2:])/2))
        if len(rows)<cfg.ocr_protect_samples or (rows[-1][0]-rows[0][0])/cfg.fps<cfg.ocr_comotion_seconds:continue
        errors=[float(np.linalg.norm(b-a)*cfg.stride/max(1,j-i)) for (i,a),(j,b) in zip(rows,rows[1:])]
        median=float(np.median(errors)) if errors else float('inf')
        if median<=cfg.ocr_comotion_pixels:accepted.append(dict(evidence,comotion_error_pixels=median))
    return accepted


def decide_shape(pairs,protected,cfg):
    valid=[p for p in pairs if p.get('valid')];moving=[p for p in valid if p['background_motion']>=cfg.background_min_motion]
    def coupled(p):
        return (p['background_edges']>=cfg.background_edge_fraction and p['background_containment']>=cfg.background_containment
                and p['background_center_error']<=p['background_center_tolerance'])
    aligned=[p for p in moving if coupled(p)]
    occlusion=[p for p in valid if coupled(p) and p['occlusion_fraction']>=cfg.occlusion_interior_fraction
               and p['occlusion_component']>=cfg.occlusion_component_fraction
               and p['occlusion_template_iou']<=cfg.overlay_occlusion_template_ceiling
               and (p.get('occlusion_cover_fraction',0)>=cfg.occlusion_cover_fraction
                    or p.get('occlusion_foreground_points',0)>=cfg.occlusion_foreground_points)]
    stable=sum(p['template_iou']>=cfg.template_iou and p.get('acceleration',0)<=cfg.smooth_acceleration_pixels for p in valid)/max(1,len(valid))
    fixed=bool(valid) and max(p.get('screen_position_delta',float('inf')) for p in valid)<=cfg.screen_position_pixels and sum(p['screen_iou']>=cfg.screen_iou for p in valid)/len(valid)>=cfg.overlay_stable_fraction
    independence=sum(p['background_center_error']>p['background_center_tolerance'] for p in moving)/max(1,len(moving))
    coherent=len(moving)>=cfg.shape_minimum_pairs and len(aligned)/max(1,len(moving))>=cfg.background_pair_fraction
    scene=coherent and len(occlusion)>=cfg.occlusion_minimum_samples
    independent=stable>=cfg.overlay_stable_fraction and independence>=cfg.overlay_independence_fraction
    overlay=(fixed or independent) and not scene
    ocr=bool(protected) and overlay
    return dict(rejected=bool(scene),ocr_protected=ocr,screen_protected=bool(fixed and not scene),own_motion_protected=bool(independent and not scene),
        overlay_evidence=bool(overlay),scene_evidence=bool(scene),valid_pairs=len(valid),moving_pairs=len(moving),aligned_pairs=len(aligned),
        occlusion_samples=len(occlusion),template_stability=stable,independence=independence,
        consistency=len(aligned)/max(1,len(moving)),ocr_evidence=protected if ocr else [],ocr_rejected_evidence=protected if not ocr else [],pairs=pairs)


def detect_text_rows(image,cfg):
    """仅图像形态：亮芯、暗描边、字符尺寸、下部居中行；不读OCR结果。"""
    h,w=image.shape[:2];gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    sat=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,1] if image.ndim==3 else np.zeros_like(gray)
    core=((gray>=cfg.subtitle_core_luma)&(sat<=cfg.subtitle_core_saturation)).astype(np.uint8)
    top=round(h*cfg.subtitle_roi_top);bottom=round(h*cfg.subtitle_roi_bottom);core[:top]=0;core[bottom:]=0
    count,labels,stats,centers=cv2.connectedComponentsWithStats(core)
    characters=[];radius=cfg.subtitle_outline_radius
    for k in range(1,count):
        x,y,cw,ch,area=stats[k]
        if not (h*cfg.subtitle_min_char_height<=ch<=h*cfg.subtitle_max_char_height and cw<=w*cfg.subtitle_max_char_width and area>=cfg.subtitle_min_char_area):continue
        xa=max(0,x-radius);ya=max(0,y-radius);xb=min(w,x+cw+radius);yb=min(h,y+ch+radius)
        component=(labels[ya:yb,xa:xb]==k).astype(np.uint8)
        ring=(cv2.dilate(component,np.ones((radius*2+1,)*2,np.uint8))>0)&~(component>0)
        dark=float(np.count_nonzero(ring&(gray[ya:yb,xa:xb]<=cfg.subtitle_outline_luma))/max(1,np.count_nonzero(ring)))
        if dark>=cfg.subtitle_outline_fraction:characters.append(dict(box=[int(x),int(y),int(x+cw),int(y+ch)],height=int(ch),cy=float(centers[k,1]),outline=dark))
    groups=[]
    for char in sorted(characters,key=lambda q:q['cy']):
        chosen=next((g for g in groups if abs(np.median([p['cy'] for p in g])-char['cy'])<=cfg.subtitle_line_y_tolerance*max(char['height'],np.median([p['height'] for p in g]))),None)
        if chosen is None:groups.append([char])
        else:chosen.append(char)
    rows=[]
    for group in groups:
        if len(group)<cfg.subtitle_min_characters:continue
        boxes=np.array([p['box'] for p in group]);box=[int(boxes[:,0].min()),int(boxes[:,1].min()),int(boxes[:,2].max()),int(boxes[:,3].max())]
        x,y,a,b=box;line_width=a-x
        if not (w*cfg.subtitle_min_line_width<=line_width<=w*cfg.subtitle_max_line_width and b-y<=h*cfg.subtitle_max_line_height and abs((x+a)/2-w/2)<=w*cfg.subtitle_center_tolerance):continue
        signature=cv2.resize(core[y:b,x:a],(cfg.subtitle_signature_width,cfg.subtitle_signature_height),interpolation=cv2.INTER_NEAREST)*255
        rows.append(dict(box=box,characters=len(group),outline=float(np.median([p['outline'] for p in group])),signature=signature))
    return rows


def associate_subtitle_rows(detections,scenes,shape,cfg):
    tracks=[];h,w=shape
    for scene,(start,end) in enumerate(scenes):
        active=[]
        for frame in range(start,end):
            used=set()
            for row in detections.get(frame,[]):
                cy=(row['box'][1]+row['box'][3])/2;choices=[]
                for index,t in enumerate(active):
                    last=t['rows'][-1];gap=frame-last['frame']
                    if index in used or not 0<gap<=cfg.subtitle_max_gap_seconds*cfg.fps:continue
                    other=last['box'];height=max(row['box'][3]-row['box'][1],other[3]-other[1])
                    if abs(cy-(other[1]+other[3])/2)>height*cfg.subtitle_line_y_tolerance:continue
                    similarity=overlap(row['signature'],last['signature'])
                    if similarity>=cfg.subtitle_signature_iou:choices.append((similarity,index))
                score,index=max(choices,default=(0,-1))
                item=dict(row,frame=frame)
                if index<0:active.append(dict(scene=scene,rows=[item]));used.add(len(active)-1)
                else:active[index]['rows'].append(item);used.add(index)
        tracks.extend(active)
    accepted=[];rejected=[]
    for t in tracks:
        rows=t['rows'];duration=(rows[-1]['frame']-rows[0]['frame']+1)/cfg.fps
        boxes=np.array([p['box'] for p in rows]);centers=(boxes[:,1]+boxes[:,3])/2
        reason=None
        if not cfg.subtitle_min_seconds<=duration<=cfg.subtitle_max_seconds:reason='句级时长不符'
        elif len(rows)/(rows[-1]['frame']-rows[0]['frame']+1)<cfg.subtitle_min_observation_fraction:reason='出现比例不足'
        elif np.max(np.abs(centers-np.median(centers)))>cfg.subtitle_max_y_jitter*h:reason='行基线随画面移动'
        if reason:rejected.append(dict(scene=t['scene'],first=rows[0]['frame'],last=rows[-1]['frame'],reason=reason));continue
        t['box']=np.median(boxes,axis=0).tolist();accepted.append(t)
    # 与同镜头其他句/多行的字幕带一致，避免仅凭下部单行把网站或标牌收入。
    final=[]
    for t in accepted:
        first=t['rows'][0]['frame'];last=t['rows'][-1]['frame'];cy=(t['box'][1]+t['box'][3])/2
        consistent=False
        for other in accepted:
            if other is t or other['scene']!=t['scene']:continue
            oy=(other['box'][1]+other['box'][3])/2
            concurrent=max(first,other['rows'][0]['frame'])<=min(last,other['rows'][-1]['frame'])
            if (abs(cy-oy)<=cfg.subtitle_band_tolerance*h or
                (concurrent and abs(cy-oy)<=cfg.subtitle_multiline_gap*h)):
                consistent=True;break
        if consistent:final.append(t)
        else:rejected.append(dict(scene=t['scene'],first=first,last=last,reason='缺少其他句或多行字幕带支持'))
    return final,rejected


def subtitles_stage(args,cfg):
    started=time.monotonic();root=args.output;data=json.loads((root/'detections.json').read_text());files=sorted((root/'frames').glob('*.png'))
    shape=cv2.imread(str(files[0]),0).shape;rows={}
    for i,file in enumerate(files):
        rows[i]=detect_text_rows(cv2.imread(str(file)),cfg)
        if i%100==0:print(f'独立文字形态扫描 {i}/{len(files)}',flush=True)
    tracks,rejected=associate_subtitle_rows(rows,data['scenes'],shape,cfg);byframe=defaultdict(list)
    audit=[]
    for index,t in enumerate(tracks):
        first=t['rows'][0]['frame'];last=t['rows'][-1]['frame']
        for i in range(first,last+1):byframe[i].append(t['box'])
        audit.append(dict(id=index,scene=t['scene'],first=first,last=last,observations=len(t['rows']),box=t['box'],
                          characters=float(np.median([r['characters'] for r in t['rows']]))))
    counts=[]
    for i,file in enumerate(files):
        independent=np.zeros(shape,np.uint8);ocr=np.zeros(shape,np.uint8)
        for box in byframe[i]:independent|=rectangle(box,shape,cfg.subtitle_region_padding)
        for p in data['subtitles'].get(str(i),[]):ocr|=rectangle(p['box'],shape,cfg.subtitle_padding)
        write_png(root/'subtitle-independent'/file.name,independent);write_png(root/'subtitle-ocr'/file.name,ocr)
        write_png(root/'subtitle-protection'/file.name,independent|ocr)
        counts.append(dict(frame=i,independent=int(np.count_nonzero(independent)),ocr=int(np.count_nonzero(ocr)),union=int(np.count_nonzero(independent|ocr))))
    write_json(root/'subtitle-audit.json',dict(tracks=audit,rejected=rejected,frames=counts,seconds=time.monotonic()-started,uses_ocr_for_independent_detection=False))


def prepare(args,cfg):
    started=time.monotonic();root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'detections.json').exists():raise ValueError('禁止覆盖检查点')
    files=sorted((args.v11/'frames').glob('*.png'));source=json.loads((args.sam/'source.json').read_text())
    if not files or {f.name:digest(f) for f in files}!=source['frames']:raise ValueError('输入缓存摘要不匹配')
    data=json.loads((args.v11/'detections.json').read_text());data['settings']=asdict(cfg)
    data['bindings']['v11_detections_sha256']=digest(args.v11/'detections.json')
    (root/'frames').symlink_to((args.v11/'frames').resolve(),target_is_directory=True)
    write_json(root/'detections.json',data)
    count=0
    for t in data['tracks']:
        for p in t['hints']:
            source=args.v11/'ocr-sam-raw'/str(t['id'])/f'{p["frame"]:06d}.png';target=root/'ocr-sam-raw'/str(t['id'])/source.name
            read_mask(source);target.parent.mkdir(parents=True,exist_ok=True);target.hardlink_to(source);count+=1
    write_json(root/'sam2-summary.json',dict(reused=count,new=0,failures=[],seconds=0.,load_seconds=0.))
    write_json(root/'prepare-timing.json',dict(seconds=time.monotonic()-started,frames=len(files),sam2_reused=count,ocr_reused=True))
    print(f'准备完成：{len(files)}帧，复用{count}张SAM2框提示，无模型重跑',flush=True)


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
        protection=strict_ocr_relation(group['samples'],[t for t in data['tracks'] if t['scene']==group['scene']],cfg)
        decision=decide_shape(group['pairs'],protection,cfg);decision.update(members=group['members'],scene=group['scene'])
        audit[str(index)]=decision
        for key in group['members']:decisions[key]=str(index)
    write_json(args.output/'motion-audit.json',audit);write_json(args.output/'cross-window-links.json',links)
    return audit,decisions,time.monotonic()-started


def merge(args,cfg):
    root=args.output;started=time.monotonic();files=sorted((root/'frames').glob('*.png'));shape=cv2.imread(str(files[0]),0).shape
    data=json.loads((root/'detections.json').read_text());metrics=json.loads((args.sam/'metrics.json').read_text())
    audit,decisions,analysis_seconds=analyze_shapes(args,cfg,data,metrics,files)
    subtitle_cache={}
    def subtitle(i):
        if i not in subtitle_cache:subtitle_cache[i]=read_mask(root/'subtitle-protection'/files[i].name,shape)
        return subtitle_cache[i]
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
        original=read_mask(args.sam/'masks'/file.name,shape);v10=read_mask(args.v10/'masks'/file.name,shape);v11=read_mask(args.v11/'masks'/file.name,shape)
        removed=original&~sources['SAM'];write_png(root/'removed'/file.name,removed)
        for name,m in sources.items():write_png(root/name/file.name,m)
        write_png(root/'masks'/file.name,union)
        rows.append(dict(frame=i,pixels=int(np.count_nonzero(union)),sources={k:int(np.count_nonzero(v)) for k,v in sources.items()},
            sam_added=int(np.count_nonzero(union&~original)),sam_removed=int(np.count_nonzero(original&~union)),
            v10_added=int(np.count_nonzero(union&~v10)),v10_removed=int(np.count_nonzero(v10&~union)),filtered=int(np.count_nonzero(removed)),v11_added=int(np.count_nonzero(union&~v11)),v11_removed=int(np.count_nonzero(v11&~union)),
            subtitle_overlap=int(np.count_nonzero(union&subtitle(i)))))
        if i%100==0:print(f'合并与字幕交集核验 {i}/{len(files)}',flush=True)
    write_manifest(root,root/'detections.json',directories,files);verified=VerifiedLayers(root/'layers.json',root/'detections.json')
    for i,file in enumerate(files):
        layers=verified.masks(i);expected=np.bitwise_or.reduce(layers)
        if not np.array_equal(expected,read_mask(root/'masks'/file.name,shape)):raise ValueError('严格并集核验失败')
    write_json(root/'statistics.json',dict(frames=rows,filters=filters,layers=len(directories),area_protection_passed=True,
        timings=dict(shape_and_background=analysis_seconds,filter_merge_verify=time.monotonic()-stage,total=time.monotonic()-started)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','subtitles','merge'])
    for name in ['config','output','v11']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--v10',type=Path);p.add_argument('--sam',type=Path)
    args=p.parse_args();cfg=Settings(**json.loads(args.config.read_text()))
    {'prepare':prepare,'subtitles':subtitles_stage,'merge':merge}[args.stage](args,cfg)


if __name__=='__main__':main()
