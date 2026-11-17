"""v14：从冻结确认组补全色/alpha边缘，保护实物与对白。"""
import argparse,json,time,shutil
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
from dataclasses import asdict
import cv2,numpy as np
from worker.watermark_edges import EdgeSettings,proposal,align,box,background_transform,reject_background
from worker.v12_masks import write_json,write_png,digest
from worker.hls_layers import track_bins,write_manifest


def read(p):return json.loads(Path(p).read_text())


def execute(args):
    root=args.output;root.mkdir(parents=True,exist_ok=True);cfg=EdgeSettings(**read(args.config)['edges']);base=args.base;plan=read(base/'group-plan.json');data=read(base/'detections.json');files=sorted((base/'frames').glob('*.png'));shape=cv2.imread(str(files[0]),0).shape;fps=data['settings']['fps'];groups=[g for g in plan['groups'] if g['active']];hints=defaultdict(list)
    for p in plan['prompts']:hints[p['group'],p['frame']].append(p['box'])
    selected={round(float(t)*fps) for t in args.times.split(',')};started=time.monotonic()
    @lru_cache(maxsize=32)
    def image(i):return cv2.imread(str(files[i]))
    @lru_cache(maxsize=32)
    def guard(i):return cv2.imread(str(base/'scene-protection'/files[i].name),0)|cv2.imread(str(base/'subtitle-protection'/files[i].name),0)
    def groupmask(gid,i):
        p=base/'group-objects'/f'group-{gid}'/files[i].name
        return cv2.imread(str(p),0) if p.exists() else None
    if args.stage=='diagnose':
        cap=cv2.VideoCapture(str(args.video));renders={}
        for i in range(max(selected)+1):
            ok,x=cap.read()
            if not ok:raise ValueError('v13e成片指定帧解码失败')
            if i in selected:renders[i]=x
        cap.release();rows=[];e=root/'diagnosis';e.mkdir(exist_ok=True)
        for i in sorted(selected):
            union=np.zeros(shape,np.uint8);old=cv2.imread(str(base/'masks'/files[i].name),0);protection=guard(i)
            for g in groups:
                seed=groupmask(g['id'],i)
                if seed is None:continue
                candidate,info=proposal(image(i),seed,protection,cfg,hints[g['id'],i]);distance=cv2.distanceTransform(255-seed,cv2.DIST_L2,5);valid=(old==0)&(protection==0);bands=[]
                delta=np.abs(image(i).astype(float)-renders[i]).mean(axis=2)
                for lo,hi in zip([0,4,8,16],[4,8,16,24]):
                    ring=(distance>lo)&(distance<=hi)&valid;found=ring&(candidate>0)
                    bands.append(dict(inner=lo,outer=hi,ring_pixels=int(ring.sum()),watermark_color_alpha_candidates=int(found.sum()),still_similar_in_v13e=int(np.count_nonzero(found&(delta<=8))),source_result_mae=float(delta[found].mean()) if found.any() else None))
                union|=candidate&~old;rows.append(dict(frame=i,time=i/fps,group=g['id'],bands=bands,**info))
            for label,m,color in [('v13b',old,(0,180,255)),('outside',union,(0,0,255))]:
                x=image(i).copy();x[m>0]=x[m>0]*.4+np.array(color)*.6;cv2.imwrite(str(e/f'{i:06d}-{label}.jpg'),x,[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(e/f'{i:06d}-source.jpg'),image(i),[cv2.IMWRITE_JPEG_QUALITY,95]);cv2.imwrite(str(e/f'{i:06d}-v13e.jpg'),renders[i],[cv2.IMWRITE_JPEG_QUALITY,95])
        write_json(root/'diagnosis.json',dict(rows=rows,seconds=time.monotonic()-started,说明='统计为扣除保护区后的同色/alpha候选，非人工水印真值；外侧与v13e相似支持存在漏盖，不证明所有掩膜内残影均由此产生'))
        print(f'环带诊断完成，{len(rows)}个组/时点记录',flush=True);return
    if (root/'masks').exists():raise ValueError('禁止覆盖已有v14掩膜')
    (root/'frames').symlink_to((base/'frames').resolve(),target_is_directory=True);shutil.copy2(base/'group-plan.json',root/'group-plan.json');data['edge_settings']=asdict(cfg);data['bindings']['v13b_layers_sha256']=digest(base/'layers.json');write_json(root/'detections.json',data)
    stride=max(1,round(fps*cfg.temporal_stride_seconds))
    @lru_cache(maxsize=64)
    def background(i,j):
        old=cv2.imread(str(base/'masks'/files[i].name),0);excluded=cv2.dilate(old,np.ones((5,5),np.uint8))
        return background_transform(image(i),image(j),excluded,cfg)
    active_keys={f"group-{g['id']}" for g in groups};maxima={};presence=defaultdict(list);statistics=[];pergroup=[]
    for directory in sorted((base/'group-objects').iterdir()):
        if not directory.is_dir():continue
        key=directory.name;gid=int(key.split('-')[1]) if key in active_keys else None;paths={int(p.stem):p for p in directory.glob('*.png')};scene=next((g['scene'] for g in groups if g['id']==gid),None)
        @lru_cache(maxsize=32)
        def raw(i):return cv2.imread(str(paths[i]),0)
        @lru_cache(maxsize=32)
        def proposed(i):
            seed=raw(i);neighbors=[j for j in [i-stride,i+stride] if j in paths and data['scenes'][scene][0]<=j<data['scenes'][scene][1]] if gid is not None else [];b=box(seed);motion=max((float(np.linalg.norm((box(raw(j))[:2]+box(raw(j))[2:]-b[:2]-b[2:])/2))/abs(j-i) for j in neighbors),default=0.)
            return proposal(image(i),seed,guard(i),cfg,hints[gid,i],motion)
        added_pixels=0;background_removed=0;radius_values=[]
        for i,path in sorted(paths.items()):
            old=raw(i);added=np.zeros(shape,np.uint8)
            if gid is not None:
                candidate,info=proposed(i);radius_values.append(info['radius']);votes=(candidate>0).astype(np.uint8);b=box(old);rapid=False
                available=1;own_support=[];background_support=[]
                for j in [i-stride,i+stride]:
                    if j not in paths or not data['scenes'][scene][0]<=j<data['scenes'][scene][1]:continue
                    available+=1
                    nb=box(raw(j));other,_=proposed(j);own=align(raw(j)|other,nb,b,shape);own_support.append(own);votes+=(own>0).astype(np.uint8)
                    matrix=background(i,j)
                    if matrix is not None:background_support.append(cv2.warpAffine(raw(j)|other,cv2.invertAffineTransform(matrix),(shape[1],shape[0]),flags=cv2.INTER_NEAREST))
                    rapid|=bool(np.max(np.abs(np.log(np.maximum(1,b[2:]-b[:2])/np.maximum(1,nb[2:]-nb[:2]))))>=cfg.rapid_scale_log)
                candidate,rejected=reject_background(candidate,own_support,background_support,cfg);background_removed+=rejected
                required=1 if rapid else min(cfg.temporal_support,available)
                added=np.where((candidate>0)&(votes>=required),255,0).astype(np.uint8);old_union=cv2.imread(str(base/'masks'/files[i].name),0);added&=~old_union;added&=~guard(i);added_pixels+=int(np.count_nonzero(added))
            mask=old|added;area=np.count_nonzero(mask)/mask.size
            if area>.2:raise ValueError(f'已确认组{key}外扩后超过20%保护，停止')
            target=root/'group-objects'/key/files[i].name;write_png(target,mask);presence[i].append((key,str(target)));maxima[key]=max(maxima.get(key,0.),area)
            if i%50==0:print(f'边缘组={key} 帧={i} 新增累计={added_pixels}',flush=True)
        pergroup.append(dict(group=key,confirmed=gid is not None,added=added_pixels,background_removed=background_removed,radius_min=min(radius_values,default=0),radius_max=max(radius_values,default=0)))
        proposed.cache_clear();raw.cache_clear();image.cache_clear();guard.cache_clear()
    bins=track_bins(maxima,limit=.2);directories=[root/'layers'/f'{k:02d}' for k in range(len(bins))]
    for i,file in enumerate(files):
        components={k:cv2.imread(p,0) for k,p in presence[i]};union=np.zeros(shape,np.uint8)
        for keys,directory in zip(bins,directories):
            m=np.zeros(shape,np.uint8)
            for key in keys:
                if key in components:m|=components[key]
            if np.count_nonzero(m)/m.size>.2:raise ValueError('图层面积超过20%')
            write_png(directory/file.name,m);union|=m
        old=cv2.imread(str(base/'masks'/file.name),0);scene_mask=cv2.imread(str(base/'scene-protection'/file.name),0);subtitle=cv2.imread(str(base/'subtitle-protection'/file.name),0);added=union&~old;sc=int(np.count_nonzero(added&scene_mask));sub=int(np.count_nonzero(union&subtitle))
        if sc or sub or np.any(old&~union):raise ValueError('实物/字幕/原掩膜保护回退')
        write_png(root/'masks'/file.name,union);write_png(root/'added'/file.name,added)
        for name in ['scene-protection','subtitle-protection']:
            target=root/name/file.name;target.parent.mkdir(parents=True,exist_ok=True);target.hardlink_to(base/name/file.name)
        statistics.append(dict(frame=i,added=int(np.count_nonzero(added)),scene_added=sc,subtitle_overlap=sub,removed=0))
        if i%100==0:print(f'全帧掩膜检查 {i}/{len(files)}',flush=True)
    write_manifest(root,root/'detections.json',directories,files);old_animation=read(base/'animation-events.json');events=old_animation['events'];write_json(root/'animation-events.json',dict(settings=old_animation['settings'],events=events,layers={str(k):[e for e in events if f"group-{e['group']}" in keys] for k,keys in enumerate(bins)}))
    write_json(root/'statistics.json',dict(frames=statistics,groups=pergroup,layers=len(bins),bins=bins,added=sum(x['added'] for x in statistics),scene_added=sum(x['scene_added'] for x in statistics),subtitle_overlap=sum(x['subtitle_overlap'] for x in statistics),removed=0,seconds=time.monotonic()-started,model_reruns=0))
    print('v14全部780帧边缘掩膜核验完成',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['diagnose','merge'])
    for n in ['base','output','config']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--video',type=Path);p.add_argument('--times',default='0,3.4,3.97,4.5,6.8,13.7');execute(p.parse_args())
