"""只重算v13受影响的掩膜合并；所有模型结果均从冻结缓存复用。"""
import argparse,json,time,shutil
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
import cv2
import numpy as np
from worker.watermark_strokes import StrokeSettings,stroke_support,same_text_member,refine_extra
from worker.v13_masks import read,bounding
from worker.v12_masks import write_json,write_png,rectangle,digest
from worker.hls_layers import track_bins,write_manifest
from worker.watermark_groups import AnimationSettings,animation_interval


def execute(args):
    started=time.monotonic();root=args.output;root.mkdir();base=read(args.base/'detections.json');plan=read(args.cache/'group-plan.json');cfg=StrokeSettings(**read(args.config)['strokes']);motion=read(args.base/'motion-audit.json')
    files=sorted((args.cache/'frames').glob('*.png'));(root/'frames').symlink_to((args.cache/'frames').resolve(),target_is_directory=True)
    shutil.copy2(args.cache/'group-plan.json',root/'group-plan.json')
    data=read(args.cache/'detections.json');data['stroke_settings']=asdict(cfg);data['bindings']['v13_group_plan_sha256']=digest(args.cache/'group-plan.json');write_json(root/'detections.json',data)
    byframe=defaultdict(list)
    for p in plan['prompts']:byframe[p['frame']].append(p)
    canonical={str(t['id']):t['observations'][0]['text'] for t in base['tracks']};stats=[];presence=defaultdict(list);maxima={};trajectories=defaultdict(dict)
    for i,file in enumerate(files):
        original=cv2.imread(str(args.base/'masks'/file.name),0);previous=cv2.imread(str(args.cache/'masks'/file.name),0);extra=previous&~original
        independent=cv2.imread(str(args.base/'subtitle-independent'/file.name),0);ocr=cv2.imread(str(args.base/'subtitle-ocr'/file.name),0);reclassified=np.zeros_like(original)
        for row in base['subtitles'].get(str(i),[]):
            region=rectangle(row['box'],original.shape)
            for p in byframe[i]:
                node=plan['nodes'][p['id']];word=canonical.get(str(node.get('track')))
                if word is None and node.get('observations'):word=node['observations'][0]['text']
                if not word or not same_text_member(row['text'],word,cfg):continue
                support=rectangle(p['box'],original.shape)
                if np.count_nonzero(region&support)/max(1,np.count_nonzero(region))>=plan['settings']['subtitle_member_overlap']:
                    reclassified|=rectangle(row['box'],original.shape,base['settings']['subtitle_padding']);break
        protection=independent|(ocr&~reclassified);scene=np.zeros_like(original)
        for g in motion.values():
            if not g['rejected'] or not base['scenes'][g['scene']][0]<=i<base['scenes'][g['scene']][1]:continue
            for member in g['members']:
                p=args.sam/'raw-object-masks'/member.rsplit('@',1)[0]/file.name
                if p.exists():scene|=cv2.imread(str(p),0)
        rectangles=np.zeros_like(original);ink=np.zeros_like(original);image=None
        for p in byframe[i]:
            region=rectangle(p['box'],original.shape,base['settings']['text_outer_padding']);rectangles|=region
            if not np.any(region&extra):continue
            if image is None:image=cv2.imread(str(root/'frames'/file.name))
            raw=cv2.imread(str(args.cache/'group-sam-raw'/p['id']/file.name),0)
            ink|=stroke_support(image,raw,p['box'],cfg)
        kept=refine_extra(extra,rectangles,ink,scene,protection);actual=original|kept
        if np.any(kept&scene) or np.any(actual&independent) or np.any(original&~actual):raise ValueError('实物/字幕/已有水印保护核验失败')
        for path in sorted((args.cache/'group-objects').glob('*/'+file.name)):
            old=cv2.imread(str(path),0);mask=(old&original)|(old&kept)
            area=np.count_nonzero(mask)/mask.size
            if area>.2:raise ValueError('单组超过20%')
            if not area:continue
            key=path.parent.name;dest=root/'group-objects'/key/file.name;write_png(dest,mask);presence[i].append((key,str(dest)));maxima[key]=max(maxima.get(key,0),area)
            if key.startswith('group-'):trajectories[int(key.split('-')[1])][i]=bounding(mask)
        for folder,mask in [('masks',actual),('added',kept),('subtitle-protection',protection),('scene-protection',scene),('stroke-support',ink)]:write_png(root/folder/file.name,mask)
        stats.append(dict(frame=i,added=int(np.count_nonzero(kept)),removed=0,scene_added_overlap=0,independent_overlap=0,pixels=int(np.count_nonzero(actual))))
        if i%100==0:print(f'笔画合并核验 {i}/{len(files)}',flush=True)
    bins=track_bins(maxima,limit=.2);directories=[root/'layers'/f'{j:02d}' for j in range(len(bins))]
    for i,file in enumerate(files):
        components={k:cv2.imread(p,0) for k,p in presence[i]};union=np.zeros_like(original)
        for keys,directory in zip(bins,directories):
            mask=np.zeros_like(original)
            for k in keys:
                if k in components:mask|=components[k]
            if np.count_nonzero(mask)/mask.size>.2:raise ValueError('图层超过20%')
            write_png(directory/file.name,mask);union|=mask
        if not np.array_equal(union,cv2.imread(str(root/'masks'/file.name),0)):raise ValueError('严格并集核验失败')
    write_manifest(root,root/'detections.json',directories,files)
    events=[];animation=AnimationSettings(**plan['animation'])
    for gid,boxes in trajectories.items():
        interval=animation_interval(boxes,animation)
        if interval:events.append(dict(group=gid,scene=next(g['scene'] for g in plan['groups'] if g['id']==gid),interval=interval))
    write_json(root/'animation-events.json',dict(settings=asdict(animation),events=events,layers={str(j):[e for e in events if 'group-'+str(e['group']) in keys] for j,keys in enumerate(bins)}))
    write_json(root/'statistics.json',dict(frames=stats,added=sum(p['added'] for p in stats),removed=0,layers=len(bins),bins=bins,maximum_fraction=maxima,seconds=time.monotonic()-started,model_reruns=0))
    print('笔画掩膜合并完成，尚待目检，不自动启动补画',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['output','base','cache','sam','config']:p.add_argument('--'+name,type=Path,required=True)
    execute(p.parse_args())
