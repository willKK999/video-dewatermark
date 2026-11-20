"""SAM3.1原始掩膜 + OCR框提示 + 通用过滤；仅掩膜，显式离线入口。"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import json
from pathlib import Path
import time
import cv2
import numpy as np
from worker.ocr_temporal import ExperimentalSettings, associate_candidates, normalize_text, guarded_mask
from worker.sam31_masks import binary, digest, write_png
from worker.hls_layers import track_bins, write_manifest, VerifiedLayers


@dataclass(frozen=True)
class Settings:
    fps: float = 30.
    stride: int = 5
    ocr_device: str = 'gpu:0'
    ocr_score: float = .65
    ocr_detection_model: str = 'PP-OCRv5_server_det'
    ocr_recognition_model: str = 'PP-OCRv5_server_rec'
    canonical_min_observations: int = 18
    canonical_min_seconds: float = 3.
    canonical_similarity: float = .8
    canonical_min_characters: int = 3
    minimum_track_seconds: float = .4
    minimum_observations: int = 3
    maximum_gap_seconds: float = .6
    association_iou: float = .4
    dominant_text_fraction: float = .75
    static_text_seconds: float = 2.
    static_position_jitter: float = .3
    moving_position_jitter: float = 1e6
    text_padding: int = 2
    subtitle_padding: int = 4
    subtitle_overlap_fraction: float = .05
    temporal_gap_seconds: float = .2
    flow_scale: float = .5
    flow_window: int = 21
    flow_levels: int = 3
    flow_pyr_scale: float = .5
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2
    ring_radius: int = 24
    flow_min_pixels: int = 12
    background_min_motion: float = .8
    motion_error_pixels: float = 1.8
    background_consistency_fraction: float = .65
    minimum_motion_pairs: int = 4
    occlusion_shrink_fraction: float = .35
    occlusion_consistency_fraction: float = .5
    mask_fraction_limit: float = .2

    def __post_init__(self):
        if self.mask_fraction_limit != .2 or self.fps <= 0 or self.stride < 1:
            raise ValueError('帧率、步长或20%保护配置无效')
        if not self.ocr_device.startswith('gpu:'):
            raise ValueError('新增OCR只能使用GPU')
        for name in ('canonical_similarity', 'subtitle_overlap_fraction', 'flow_scale',
                     'background_consistency_fraction', 'occlusion_shrink_fraction', 'occlusion_consistency_fraction'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError('比例配置无效：'+name)
        if min(self.canonical_min_observations, self.canonical_min_seconds, self.minimum_track_seconds,
               self.minimum_observations, self.maximum_gap_seconds, self.temporal_gap_seconds,
               self.flow_window, self.flow_levels, self.ring_radius, self.flow_min_pixels,
               self.background_min_motion, self.motion_error_pixels, self.minimum_motion_pairs) <= 0:
            raise ValueError('时域或运动配置必须为正')
        if min(self.text_padding, self.subtitle_padding) < 0:
            raise ValueError('膨胀不能为负')


def rectangle(box, shape, padding=0):
    m = np.zeros(shape, np.uint8)
    x,y,a,b = np.array(box, float)
    x,y = np.floor([x-padding,y-padding]).astype(int)
    a,b = np.ceil([a+padding,b+padding]).astype(int)
    if a <= x or b <= y or not np.isfinite(box).all():
        raise ValueError('无效文字框')
    m[max(0,y):min(shape[0],b), max(0,x):min(shape[1],a)] = 255
    return m


def text_tracks(rows, scenes, cfg):
    base = ExperimentalSettings(stride=cfg.stride, maximum_gap_seconds=cfg.maximum_gap_seconds,
                                minimum_duration_seconds=cfg.minimum_track_seconds,
                                minimum_observations=cfg.minimum_observations, association_iou=cfg.association_iou,
                                ocr_dominant_text_fraction=cfg.dominant_text_fraction,
                                ocr_minimum_duration_seconds=cfg.static_text_seconds,
                                ocr_position_jitter=cfg.static_position_jitter)
    accepted, rejected = associate_candidates(rows, scenes, cfg.fps, base)
    # 已通过v9严格稳定门控的文字为种子；长轨迹允许自身平滑移动。
    seeds = {normalize_text(p['text']) for t in accepted for p in t['observations']}
    for t in rejected:
        obs = t['observations']; words = [normalize_text(p.get('text','')) for p in obs]
        word, count = Counter(words).most_common(1)[0]
        if (len(word) >= cfg.canonical_min_characters and count >= cfg.canonical_min_observations
            and (obs[-1]['frame']-obs[0]['frame'])/cfg.fps >= cfg.canonical_min_seconds
            and count/len(obs) >= base.ocr_dominant_text_fraction):
            seeds.add(word)
    def known(text):
        word=normalize_text(text)
        return bool(word) and any(SequenceMatcher(None,word,s).ratio() >= cfg.canonical_similarity for s in seeds)
    # 用同一关联器再次生成稠密框；只移除位置抖动/两秒门槛，保留逐轨文本稳定和最低生命周期。
    moving = ExperimentalSettings(stride=cfg.stride, maximum_gap_seconds=cfg.maximum_gap_seconds,
        minimum_duration_seconds=cfg.minimum_track_seconds, minimum_observations=cfg.minimum_observations,
        ocr_minimum_duration_seconds=cfg.minimum_track_seconds, ocr_position_jitter=cfg.moving_position_jitter,
        association_iou=cfg.association_iou,ocr_dominant_text_fraction=cfg.dominant_text_fraction)
    brand_rows={k:[p for p in v if known(p.get('text',''))] for k,v in rows.items()}
    tracks, short = associate_candidates(brand_rows, scenes, cfg.fps, moving)
    # 非稳定词汇按实际OCR观测存活；仅填相邻采样间隔，不跨句或向镜头外延伸。
    subtitle=defaultdict(list)
    for a,b in scenes:
        for k,obs in rows.items():
            i=int(k)
            if not a <= i < b: continue
            for p in obs:
                if known(p.get('text','')): continue
                for j in range(i,min(i+cfg.stride,b)):
                    subtitle[j].append(p)
    return tracks, dict(subtitle), dict(seeds=sorted(seeds), rejected=rejected, short_brand_tracks=short)


def subtract_subtitles(mask, regions, cfg):
    if not np.any(mask): return mask.copy(), np.zeros_like(mask)
    overlap = (mask>0)&(regions>0)
    # 分量内显著重叠才扣除该文字区域；保留混合轨迹其余图标/网址。
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask>0).astype('uint8'))
    removed=np.zeros_like(mask)
    for k in range(1,n):
        part=labels==k
        if np.count_nonzero(overlap&part)/stats[k,cv2.CC_STAT_AREA] >= cfg.subtitle_overlap_fraction:
            removed[overlap&part]=255
    return mask & ~removed, removed


def scene_object(evidence, cfg):
    moving=[p for p in evidence if p['background_motion'] >= cfg.background_min_motion]
    consistent=[p for p in moving if p['centroid_error'] <= cfg.motion_error_pixels
                and p['inner_error'] <= cfg.motion_error_pixels]
    fraction=len(consistent)/max(1,len(moving))
    occlusion=any(p['shrink'] >= cfg.occlusion_shrink_fraction for p in consistent)
    reject=(len(moving)>=cfg.minimum_motion_pairs and
            (fraction>=cfg.background_consistency_fraction or
             (occlusion and fraction>=cfg.occlusion_consistency_fraction)))
    return dict(rejected=reject, moving_pairs=len(moving), consistent_pairs=len(consistent),
                consistency=fraction, occlusion=occlusion, pairs=evidence)


def flow_evidence(first, second, flow, cfg):
    size=(flow.shape[1],flow.shape[0])
    a=cv2.resize(first,size,interpolation=cv2.INTER_NEAREST)>0
    b=cv2.resize(second,size,interpolation=cv2.INTER_NEAREST)>0
    radius=max(1,round(cfg.ring_radius*cfg.flow_scale))
    ring=(cv2.dilate(a.astype('uint8'),np.ones((radius*2+1,)*2,np.uint8))>0)&~a
    if min(np.count_nonzero(a),np.count_nonzero(b),np.count_nonzero(ring)) < cfg.flow_min_pixels:
        return None
    inner=np.median(flow[a],axis=0)/cfg.flow_scale
    bg=np.median(flow[ring],axis=0)/cfg.flow_scale
    ya,xa=np.where(a); yb,xb=np.where(b)
    displacement=np.array([xb.mean()-xa.mean(),yb.mean()-ya.mean()])/cfg.flow_scale
    return dict(background_motion=float(np.linalg.norm(bg)), inner_error=float(np.linalg.norm(inner-bg)),
                centroid_error=float(np.linalg.norm(displacement-bg)),
                shrink=float(max(0,1-np.count_nonzero(b)/np.count_nonzero(a))))


def temporal_fill(observed, scenes, cfg):
    additions={}
    for a,b in scenes:
        indices=sorted(i for i in observed if a<=i<b)
        for left,right in zip(indices,indices[1:]):
            if 1 < right-left <= cfg.temporal_gap_seconds*cfg.fps:
                for i in range(left+1,right):
                    # 两端交集避免扩散到新背景；不外推，不跨镜头。
                    additions[i]=observed[left]&observed[right]
    return additions


def gpu_ocr(files, cfg, output):
    from paddleocr import PaddleOCR
    import paddle
    if not paddle.is_compiled_with_cuda(): raise RuntimeError('OCR环境没有GPU支持，禁止回退CPU')
    paddle.set_device(cfg.ocr_device)
    model=PaddleOCR(text_detection_model_name=cfg.ocr_detection_model,
        text_recognition_model_name=cfg.ocr_recognition_model, use_doc_orientation_classify=False,
        use_doc_unwarping=False,use_textline_orientation=False,device=cfg.ocr_device,enable_mkldnn=False)
    started=time.monotonic(); rows={}
    for i in range(0,len(files),cfg.stride):
        result=next(iter(model.predict(str(files[i])))); rows[str(i)]=[]
        for poly,text,score in zip(result['rec_polys'],result['rec_texts'],result['rec_scores']):
            if score < cfg.ocr_score: continue
            poly=np.asarray(poly)
            rows[str(i)].append(dict(source='OCR',text=str(text),confidence=float(score),
                box=[float(poly[:,0].min()),float(poly[:,1].min()),float(poly[:,0].max()),float(poly[:,1].max())]))
        if i%50==0: print(f'GPU OCR {i}/{len(files)}',flush=True)
    Path(output).write_text(json.dumps(dict(rows=rows,seconds=time.monotonic()-started,device=cfg.ocr_device),ensure_ascii=False))


def segment_ocr(root, cfg):
    import os, torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from worker.gpu_lock import device_lock
    data=json.loads((root/'detections.json').read_text()); byframe=defaultdict(list)
    for t in data['tracks']:
        for h in t['hints']: byframe[h['frame']].append((t,h))
    started=time.monotonic(); maximum={}; failures=[]
    with device_lock(),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor=SAM2ImagePredictor(build_sam2(os.environ['SAM_CONFIG'],os.environ['SAM_CHECKPOINT'],device='cuda'))
        loaded=time.monotonic()-started
        for i, items in sorted(byframe.items()):
            image=cv2.imread(str(root/'frames'/f'{i:06d}.png')); predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            for t,h in items:
                masks,scores,_=predictor.predict(box=np.array(h['box']),multimask_output=True)
                support=rectangle(h['box'],image.shape[:2],cfg.text_padding)
                # OCR确认框作为掩膜下限；SAM支持限制在其膨胀框内，避免框提示吞入人物。
                raw=masks[int(np.argmax(scores))].astype(np.uint8)*255
                write_png(root/'ocr-sam-raw'/str(t['id'])/f'{i:06d}.png',raw)
                mask=(raw & support)|support
                try:
                    fraction=guarded_mask(mask); maximum[str(t['id'])]=max(maximum.get(str(t['id']),0),fraction)
                except ValueError:
                    failures.append(dict(track=t['id'],frame=i,reason='OCR框提示超过20%'))
                write_png(root/'ocr-objects'/str(t['id'])/f'{i:06d}.png',mask)
            if i%100==0: print(f'OCR框提示SAM2.1 {i}',flush=True)
    write_json(root/'sam2-summary.json',dict(seconds=time.monotonic()-started,load_seconds=loaded,maximum=maximum,failures=failures))


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2))


def prepare(args,cfg):
    root=args.output; root.mkdir(parents=True,exist_ok=True)
    if (root/'detections.json').exists(): raise ValueError('禁止覆盖已有检测检查点')
    files=sorted(args.frames.glob('*.png'))
    source=json.loads((args.sam/'source.json').read_text())
    started=time.monotonic()
    if {f.name:digest(f) for f in files}!=source['frames']: raise ValueError('解码帧与SAM缓存不匹配')
    from worker.vision import shots
    scenes=shots(files)
    rows=json.loads(args.ocr.read_text())['rows']
    tracks,subs,audit=text_tracks(rows,scenes,cfg)
    (root/'frames').symlink_to(args.frames.resolve(),target_is_directory=True)
    write_json(root/'detections.json',dict(tracks=tracks,subtitles=subs,scenes=scenes,settings=asdict(cfg),audit=audit,
        hints=[h for t in tracks for h in t['hints']],bindings=dict(ocr_sha256=digest(args.ocr),sam_source_sha256=digest(args.sam/'source.json'))))
    write_json(root/'prepare-timing.json',dict(seconds=time.monotonic()-started,frames=len(files)))
    print(f'准备完成：{len(files)}帧、{len(tracks)}条OCR轨迹、{len(scenes)}个镜头',flush=True)


def merge(args,cfg):
    started=time.monotonic(); root=args.output; files=sorted((root/'frames').glob('*.png'))
    shape=cv2.imread(str(files[0]),0).shape
    data=json.loads((root/'detections.json').read_text()); scenes=data['scenes']
    metrics=json.loads((args.sam/'metrics.json').read_text())
    # 核验原始缓存文件；每对象保留独立来源，拒绝过大对象而非切碎规避面积保护。
    tracks={k:v for k,v in metrics['tracks'].items() if v['retained']}
    evidence={k:[] for k in tracks}; sample=time.monotonic()
    def image(i):
        im=cv2.imread(str(files[i]),0)
        return cv2.resize(im,None,fx=cfg.flow_scale,fy=cfg.flow_scale)
    for a,b in scenes:
        for i in range(a,b-cfg.stride,cfg.stride):
            j=i+cfg.stride
            flow=cv2.calcOpticalFlowFarneback(image(i),image(j),None,cfg.flow_pyr_scale,cfg.flow_levels,cfg.flow_window,cfg.flow_iterations,cfg.flow_poly_n,cfg.flow_poly_sigma,0)
            for k,t in tracks.items():
                if i not in t['frames'] or j not in t['frames']: continue
                x=cv2.imread(str(args.sam/'raw-object-masks'/k/files[i].name),0)
                y=cv2.imread(str(args.sam/'raw-object-masks'/k/files[j].name),0)
                p=flow_evidence(x,y,flow,cfg)
                if p: evidence[k].append(dict(p,frame=i,next_frame=j))
        print(f'背景运动检查 {a}:{b}',flush=True)
    decisions={k:scene_object(v,cfg) for k,v in evidence.items()}
    write_json(root/'motion-audit.json',decisions)
    flow_seconds=time.monotonic()-sample
    maxima={}; filter_audit={}; stage=time.monotonic()
    subs={int(k):v for k,v in data['subtitles'].items()}
    def subtitle_mask(i):
        region=np.zeros(shape,np.uint8)
        for p in subs.get(i,[]): region|=rectangle(p['box'],shape,cfg.subtitle_padding)
        return region
    # 过滤结果为独立对象，便于逐层摘要和20%检查；原始对象永远不改写。
    for k,t in tracks.items():
        observed={}; removed_count=0; maximum=0.
        valid=(len(t['frames'])>=cfg.minimum_observations and
               (max(t['frames'])-min(t['frames']))/cfg.fps>=cfg.minimum_track_seconds)
        valid=valid and all(np.count_nonzero(cv2.imread(str(args.sam/'raw-object-masks'/k/files[i].name),0))/np.prod(shape)<=.2 for i in t['frames'])
        for i in t['frames']:
            m=binary(cv2.imread(str(args.sam/'raw-object-masks'/k/files[i].name),0),shape)
            if np.count_nonzero(m)/m.size>.2: valid=False
            if decisions[k]['rejected'] or not valid: filtered=np.zeros(shape,np.uint8)
            else: filtered,_=subtract_subtitles(m,subtitle_mask(i),cfg)
            removed_count+=int(np.count_nonzero((m>0)&(filtered==0)))
            if np.any(filtered): observed[i]=filtered
        extra=temporal_fill(observed,scenes,cfg)
        for i,m in extra.items():
            m,_=subtract_subtitles(m,subtitle_mask(i),cfg)
            if np.any(m): write_png(root/'temporal-objects'/k/files[i].name,m)
        for i,m in observed.items():
            write_png(root/'sam-objects'/k/files[i].name,m)
        maxima['sam/'+k]=max([np.count_nonzero(m)/m.size for m in observed.values()]+[0.])
        maxima['temporal/'+k]=max([np.count_nonzero(m)/m.size for m in extra.values()]+[0.])
        filter_audit[k]=dict(removed=removed_count,retained_frames=len(observed),temporal_frames=len(extra),
                            rejected_motion=decisions[k]['rejected'],persistent=valid)
    summary=json.loads((root/'sam2-summary.json').read_text()); failed={p['track'] for p in summary['failures']}
    for t in data['tracks']:
        k=str(t['id'])
        if t['id'] in failed: continue
        for h in t['hints']:
            m=cv2.imread(str(root/'ocr-objects'/k/files[h['frame']].name),0); guarded_mask(m)
            maxima['ocr/'+k]=max(maxima.get('ocr/'+k,0),np.count_nonzero(m)/m.size)
    bins=track_bins({k:v for k,v in maxima.items() if v>0},limit=.2) or [[]]
    dirs=[root/'layers'/f'{i:02d}' for i in range(len(bins))]
    allrows=[]; totals=Counter()
    for i,f in enumerate(files):
        sources={s:np.zeros(shape,np.uint8) for s in ('SAM','OCR','temporal')}; components={}
        for key in maxima:
            kind,k=key.split('/')
            folder={'sam':'sam-objects','ocr':'ocr-objects','temporal':'temporal-objects'}[kind]
            p=root/folder/k/f.name
            if not p.exists(): continue
            m=binary(cv2.imread(str(p),0),shape); components[key]=m
            temporal_ocr=kind=='ocr' and next(h['temporal'] for t in data['tracks'] if str(t['id'])==k for h in t['hints'] if h['frame']==i)
            sources[{'sam':'SAM','ocr':'OCR','temporal':'temporal'}[kind]]|=m
            if temporal_ocr: sources['temporal']|=m
        union=np.zeros(shape,np.uint8)
        for identities,directory in zip(bins,dirs):
            m=np.zeros(shape,np.uint8)
            for key in identities:
                if key in components: m|=components[key]
            guarded_mask(m); union|=m; write_png(directory/f.name,m)
        original=cv2.imread(str(args.sam/'masks'/f.name),0)
        removed=original & ~sources['SAM']
        write_png(root/'removed'/f.name,removed)
        for s,m in sources.items(): write_png(root/s/f.name,m)
        write_png(root/'masks'/f.name,union)
        row=dict(frame=i,sources={s:int(np.count_nonzero(m)) for s,m in sources.items()},
                 sam_added=int(np.count_nonzero((union>0)&(original==0))),
                 sam_removed=int(np.count_nonzero((original>0)&(union==0))),filtered=int(np.count_nonzero(removed)))
        if args.v9:
            v9=cv2.imread(str(args.v9/f.name),0)
            row.update(v9_added=int(np.count_nonzero((union>0)&(v9==0))),v9_removed=int(np.count_nonzero((v9>0)&(union==0))))
        allrows.append(row)
        if i%100==0: print(f'逐帧合并 {i}/{len(files)}',flush=True)
    write_manifest(root,root/'detections.json',dirs,files)
    verified=VerifiedLayers(root/'layers.json',root/'detections.json')
    for i,f in enumerate(files):
        expected=np.bitwise_or.reduce(verified.masks(i))
        if not np.array_equal(expected,cv2.imread(str(root/'masks'/f.name),0)): raise ValueError('最终联合不是图层严格并集')
    write_json(root/'statistics.json',dict(frames=allrows,filters=filter_audit,layers=len(dirs),area_protection_passed=True,
        timings=dict(flow=flow_seconds,filter_merge_verify=time.monotonic()-stage,total=time.monotonic()-started)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','segment','merge','ocr'])
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=Path);p.add_argument('--sam',type=Path);p.add_argument('--ocr',type=Path);p.add_argument('--v9',type=Path)
    args=p.parse_args();cfg=Settings(**json.loads(args.config.read_text()))
    if args.stage=='prepare': prepare(args,cfg)
    elif args.stage=='segment': segment_ocr(args.output,cfg)
    elif args.stage=='merge': merge(args,cfg)
    else: gpu_ocr(sorted(args.frames.glob('*.png')),cfg,args.output)


if __name__=='__main__': main()
