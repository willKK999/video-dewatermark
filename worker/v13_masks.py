"""v13离线掩膜：复用v12过滤，仅按水印组增加成员和重分层。"""
import argparse,json,time
from collections import defaultdict,Counter
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
from worker.watermark_groups import GroupSettings,AnimationSettings,connected_groups,relation,animation_interval
from worker.v12_masks import write_json,read_mask,rectangle,digest,write_png,write_manifest
from worker.ocr_temporal import normalize_text


def read(p):return json.loads(Path(p).read_text())
def bounding(mask):
    y,x=np.where(mask>0)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None
def union_box(boxes):
    a=np.array(boxes);return [float(a[:,0].min()),float(a[:,1].min()),float(a[:,2].max()),float(a[:,3].max())]
def dense(observations):
    out={p['frame']:p['box'] for p in observations}
    for a,b in zip(observations,observations[1:]):
        for i in range(a['frame']+1,b['frame']):
            t=(i-a['frame'])/(b['frame']-a['frame']);out[i]=((1-t)*np.array(a['box'])+t*np.array(b['box'])).tolist()
    return out


def prepare(args,cfg,animation):
    t=time.monotonic();root=args.output;root.mkdir(parents=True,exist_ok=True)
    if (root/'group-plan.json').exists():raise ValueError('禁止覆盖已有计划')
    base=read(args.base/'detections.json');geo=read(args.geometries);motion=read(args.base/'motion-audit.json');nodes={};fps=base['settings']['fps']
    for k,row in motion.items():
        if row['rejected']:continue
        a,b=base['scenes'][row['scene']];boxes={}
        members=[p.rsplit('@',1)[0] for p in row['members']]
        for key in members:
            for i,p in geo[key]['frames'].items():
                i=int(i)
                if a<=i<b:boxes.setdefault(i,[]).append(p['box'])
        nodes['sam:'+k]=dict(kind='sam',scene=row['scene'],members=members,boxes={i:union_box(v) for i,v in boxes.items()},confirmed=row['overlay_evidence'])
    for tr in base['tracks']:
        nodes['ocr:'+str(tr['id'])]=dict(kind='ocr',scene=tr['scene'],track=tr['id'],boxes={h['frame']:h['box'] for h in tr['hints']},confirmed=True)
    for k,tr in enumerate(base['audit']['rejected']):
        obs=tr.get('observations',[])
        if len(obs)<cfg.minimum_samples or 'scene' not in tr:continue
        word,count=Counter(normalize_text(p['text']) for p in obs).most_common(1)[0]
        if word not in base['audit']['seeds'] or count/len(obs)<base['settings']['dominant_text_fraction']:continue
        nodes['short:'+str(k)]=dict(kind='ocr',scene=tr['scene'],track=None,boxes=dense(obs),confirmed=False,observations=obs)
    groups,links=connected_groups(nodes,fps,cfg);prompts=[];rows=[]
    files=sorted((args.base/'frames').glob('*.png'));source=read(args.sam/'source.json')
    if {p.name:digest(p) for p in files}!=source['frames']:raise ValueError('源解码帧摘要不符')
    (root/'frames').symlink_to((args.base/'frames').resolve(),target_is_directory=True)
    image_cache={}
    def image(i):
        if i not in image_cache:
            if len(image_cache)>40:image_cache.clear()
            image_cache[i]=cv2.imread(str(files[i]))
        return image_cache[i]
    for gid,keys in enumerate(groups):
        sam=[k for k in keys if nodes[k]['kind']=='sam'];ocr=[k for k in keys if nodes[k]['kind']=='ocr']
        active=bool(sam and len(keys)>1 and any(nodes[k]['confirmed'] for k in keys))
        rows.append(dict(id=gid,members=keys,scene=nodes[keys[0]]['scene'],active=active))
        if not active:continue
        for key in ocr:
            node=nodes[key];boxes=node['boxes'];choices=[]
            for anchor in sam:
                evidence=relation(nodes[anchor]['boxes'],boxes,fps,cfg)
                if evidence:choices.append((evidence['offset_residual'],anchor))
            if not choices:continue
            _,anchor=min(choices);ab=nodes[anchor]['boxes'];common=sorted(set(ab)&set(boxes));rels=[]
            for i in common:
                a=np.array(ab[i]);size=np.maximum(1,a[2:]-a[:2]);rels.append((np.array(boxes[i])-a[[0,1,0,1]])/size[[0,1,0,1]])
            rel=np.median(rels,axis=0);extras={};gap=round(cfg.propagation_seconds*fps)
            for i,a in ab.items():
                if i in boxes or not min(boxes)-gap<=i<=max(boxes)+gap:continue
                a=np.array(a);size=np.maximum(1,a[2:]-a[:2]);box=(a[[0,1,0,1]]+rel*size[[0,1,0,1]]).tolist()
                ref=min(boxes,key=lambda j:abs(j-i))
                def patch(i,box):
                    h,w=image(i).shape[:2];x,y,c,d=np.rint(box).astype(int);x,c=max(0,x),min(w,c);y,d=max(0,y),min(h,d)
                    if c<=x or d<=y:return None
                    return cv2.resize(cv2.cvtColor(image(i)[y:d,x:c],cv2.COLOR_BGR2GRAY),(96,48))
                x,y=patch(i,box),patch(ref,boxes[ref])
                if x is None or y is None:continue
                score=float(cv2.matchTemplate(x,y,cv2.TM_CCOEFF_NORMED)[0,0])
                if score>=cfg.propagation_template_correlation:extras[i]=box
            for i,box in sorted(boxes.items()):
                prompts.append(dict(id=key,group=gid,frame=i,box=box,track=node['track'],propagated=False))
            for i,box in sorted(extras.items()):prompts.append(dict(id=key,group=gid,frame=i,box=box,track=None,propagated=True))
    serial={k:dict(v,boxes={str(i):b for i,b in v['boxes'].items()}) for k,v in nodes.items()}
    write_json(root/'group-plan.json',dict(groups=rows,links=links,nodes=serial,prompts=prompts,settings=asdict(cfg),animation=asdict(animation),seconds=time.monotonic()-t))
    data=dict(base);data['bindings']=dict(base['bindings'],v12_detections_sha256=digest(args.base/'detections.json'));data['group_settings']=asdict(cfg)
    write_json(root/'detections.json',data)
    print(f'分组完成：{len(nodes)}个节点，{len(links)}条联动，{sum(r["active"] for r in rows)}个有效组，{len(prompts)}次文字框提示',flush=True)


def segment(args,cfg,animation):
    import os,shutil
    root=args.output;plan=read(root/'group-plan.json');needed=defaultdict(list);reused=0;t=time.monotonic()
    for p in plan['prompts']:
        target=root/'group-sam-raw'/p['id']/f'{p["frame"]:06d}.png'
        if p['track'] is not None:
            old=args.base/'ocr-sam-raw'/str(p['track'])/target.name
            if old.exists():target.parent.mkdir(parents=True,exist_ok=True);target.hardlink_to(old);reused+=1;continue
        needed[p['frame']].append((p,target))
    if needed:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from worker.gpu_lock import device_lock
        with device_lock(),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            model=SAM2ImagePredictor(build_sam2(os.environ['SAM_CONFIG'],os.environ['SAM_CHECKPOINT'],device='cuda'))
            for i,items in sorted(needed.items()):
                model.set_image(cv2.cvtColor(cv2.imread(str(root/'frames'/f'{i:06d}.png')),cv2.COLOR_BGR2RGB))
                for p,target in items:
                    masks,scores,_=model.predict(box=np.array(p['box']),multimask_output=True)
                    write_png(target,masks[int(np.argmax(scores))].astype(np.uint8)*255)
                if i%50==0:print(f'新增组内框提示 {i}',flush=True)
    write_json(root/'segment-summary.json',dict(reused=reused,new=sum(map(len,needed.values())),seconds=time.monotonic()-t))


def merge(args,cfg,animation):
    t=time.monotonic();root=args.output;plan=read(root/'group-plan.json');base=read(args.base/'detections.json');files=sorted((root/'frames').glob('*.png'));shape=read_mask(args.base/'masks'/files[0].name).shape
    groups=[g for g in plan['groups'] if g['active']];prompts=defaultdict(list)
    for p in plan['prompts']:prompts[p['frame']].append(p)
    nodes=plan['nodes'];maxima={};presence=defaultdict(list);stats=[];events=[];trajectories=defaultdict(dict)
    for i,file in enumerate(files):
        if args.reuse is not None and all((args.reuse/folder/file.name).exists() for folder in ['masks','added','subtitle-protection','reclassified-ocr']):
            if digest(args.reuse/'group-plan.json')!=digest(root/'group-plan.json'):raise ValueError('恢复计划摘要不一致')
            expected=np.zeros(shape,np.uint8);cached_groups={}
            for path in sorted((args.reuse/'group-objects').glob('*/'+file.name)):
                mask=read_mask(path);area=np.count_nonzero(mask)/mask.size
                if area>.2:raise ValueError('缓存图层面积违规')
                key=path.parent.name;target=root/'group-objects'/key/file.name;target.parent.mkdir(parents=True,exist_ok=True);target.hardlink_to(path)
                presence[i].append((key,str(target)));maxima[key]=max(maxima.get(key,0),area);expected|=mask
                if key.startswith('group-'):
                    gid=int(key.split('-')[1]);cached_groups[str(gid)]=int(np.count_nonzero(mask));trajectories[gid][i]=bounding(mask)
            original=read_mask(args.base/'masks'/file.name);actual=read_mask(args.reuse/'masks'/file.name);independent=read_mask(args.base/'subtitle-independent'/file.name);protection=read_mask(args.reuse/'subtitle-protection'/file.name)
            if not np.array_equal(expected,actual) or np.any(original&~actual) or np.any(actual&protection) or np.any(actual&independent):raise ValueError('缓存严格并集或保护核验失败')
            for folder in ['masks','added','subtitle-protection','reclassified-ocr']:
                target=root/folder/file.name;target.parent.mkdir(parents=True,exist_ok=True);target.hardlink_to(args.reuse/folder/file.name)
            stats.append(dict(frame=i,added=int(np.count_nonzero(actual&~original)),removed=0,independent_overlap=0,pixels=int(np.count_nonzero(actual)),groups=cached_groups,reused_checkpoint=True))
            if i%100==0:print(f'复核已完成掩膜缓存 {i}/{len(files)}',flush=True)
            continue
        original=read_mask(args.base/'masks'/file.name);independent=read_mask(args.base/'subtitle-independent'/file.name);ocr_sub=read_mask(args.base/'subtitle-ocr'/file.name);reclassified=np.zeros(shape,np.uint8)
        # OCR行必须与已联动确认的组内文字框显著重叠，才能改分类；独立对白保护从不扣掉。
        for row in base['subtitles'].get(str(i),[]):
            region=rectangle(row['box'],shape)
            for p in prompts[i]:
                b=rectangle(p['box'],shape)
                if np.count_nonzero(region&b)/max(1,np.count_nonzero(region))>=cfg.subtitle_member_overlap:
                    reclassified|=rectangle(row['box'],shape,base['settings']['subtitle_padding']);break
        protection=independent|(ocr_sub&~reclassified);group_masks={g['id']:np.zeros(shape,np.uint8) for g in groups};raw_sam={}
        for g in groups:
            if not base['scenes'][g['scene']][0]<=i<base['scenes'][g['scene']][1]:continue
            mask=group_masks[g['id']]
            for key in g['members']:
                n=nodes[key]
                if n['kind']=='sam':
                    for member in n['members']:
                        path=args.sam/'raw-object-masks'/member/file.name
                        if path.exists():
                            if member not in raw_sam:raw_sam[member]=read_mask(path)
                            mask|=raw_sam[member]
                elif n['track'] is not None:
                    path=args.base/'ocr-objects'/str(n['track'])/file.name
                    if path.exists():mask|=read_mask(path)
            mask&=~protection
        for p in prompts[i]:
            support=rectangle(p['box'],shape);outer=rectangle(p['box'],shape,base['settings']['text_outer_padding']);raw=read_mask(root/'group-sam-raw'/p['id']/file.name)
            # 原始模型对象也受20%约束，不用裁框隐藏模型越界。
            if np.count_nonzero(raw)/raw.size>.2:raise ValueError('组内SAM原始对象超过20%，停止')
            group_masks[p['group']]|=((raw&outer)|support)&~protection
        all_groups=np.zeros(shape,np.uint8)
        for gid,mask in group_masks.items():
            area=np.count_nonzero(mask)/mask.size
            if area>.2:raise ValueError('完整水印组超过20%，不得拆组绕过保护')
            if not area:continue
            key='group-'+str(gid);path=root/'group-objects'/key/file.name;write_png(path,mask);presence[i].append((key,str(path)));maxima[key]=max(maxima.get(key,0),area);all_groups|=mask
            trajectories[gid][i]=bounding(mask)
        # 原v12像素全部保留；已被有效组接管的像素从旧层取走，保证组内同一遍施工。
        for j,layer in enumerate(read(args.base/'layers.json')['layers']):
            mask=read_mask(Path(layer['directory'])/file.name)&~all_groups
            if not np.any(mask):continue
            key='base-'+str(j);path=root/'group-objects'/key/file.name;write_png(path,mask);presence[i].append((key,str(path)));maxima[key]=max(maxima.get(key,0),np.count_nonzero(mask)/mask.size)
        actual=original|all_groups;added=actual&~original
        if np.any(actual&independent):raise ValueError('独立对白保护回退')
        if np.any(actual&protection):raise ValueError('最终对白保护违规')
        write_png(root/'masks'/file.name,actual);write_png(root/'added'/file.name,added);write_png(root/'subtitle-protection'/file.name,protection);write_png(root/'reclassified-ocr'/file.name,reclassified)
        stats.append(dict(frame=i,added=int(np.count_nonzero(added)),removed=int(np.count_nonzero(original&~actual)),independent_overlap=int(np.count_nonzero(actual&independent)),pixels=int(np.count_nonzero(actual)),groups={str(k):int(np.count_nonzero(m)) for k,m in group_masks.items() if np.any(m)}))
        if i%100==0:print(f'组掩膜与字幕核验 {i}/{len(files)}',flush=True)
    from worker.hls_layers import track_bins
    bins=track_bins(maxima,limit=.2);directories=[root/'layers'/f'{j:02d}' for j in range(len(bins))]
    for i,file in enumerate(files):
        components={k:read_mask(Path(p)) for k,p in presence[i]};union=np.zeros(shape,np.uint8)
        for keys,directory in zip(bins,directories):
            mask=np.zeros(shape,np.uint8)
            for k in keys:
                if k in components:mask|=components[k]
            if np.count_nonzero(mask)/mask.size>.2:raise ValueError('图层超过20%')
            write_png(directory/file.name,mask);union|=mask
        if not np.array_equal(union,read_mask(root/'masks'/file.name)):raise ValueError('图层严格并集不一致')
    write_manifest(root,root/'detections.json',directories,files)
    for gid,boxes in trajectories.items():
        interval=animation_interval(boxes,animation)
        if interval:events.append(dict(group=gid,scene=next(g['scene'] for g in groups if g['id']==gid),interval=interval))
    bylayer={str(j):[e for e in events if 'group-'+str(e['group']) in keys] for j,keys in enumerate(bins)}
    write_json(root/'animation-events.json',dict(settings=asdict(animation),layers=bylayer,events=events))
    write_json(root/'statistics.json',dict(frames=stats,layers=len(bins),bins=bins,group_maximum_fraction=maxima,seconds=time.monotonic()-t,added=sum(p['added'] for p in stats),removed=sum(p['removed'] for p in stats)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','segment','merge'])
    for n in ['output','base','sam','geometries','config']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--reuse',type=Path)
    a=p.parse_args();c=read(a.config);globals()[a.stage](a,GroupSettings(**c['groups']),AnimationSettings(**c['animation']))
