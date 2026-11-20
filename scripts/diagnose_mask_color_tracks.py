"""只读追踪源帧蓝色部件，区分掩膜缺失和存储桶换号，不调用检测/补画模型。"""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from scripts.diagnose_watermark_residue import color_blue,intervals,write,read


def overlap(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));v=x*y;return v/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-v)


def execute(args):
    root=args.masks;out=args.output;data=read(root/'detections.json');n=max(b for a,b in data['scenes']);source=np.memmap(args.source,mode='r',dtype=np.uint8,shape=(n,1280,720,3));result=np.memmap(args.result,mode='r',dtype=np.uint8,shape=source.shape);tracks=[];started=time.monotonic()
    for scene,(a,b) in enumerate(data['scenes']):
        alive=[]
        for i in range(a,b):
            name=f'{i:06d}.png';mask=cv2.imread(str(root/'masks'/name),0)>0;blue=color_blue(source[i]);protection=cv2.imread(str(root/'scene-protection'/name),0)>0;blue&=~protection
            count,labels,stats,_=cv2.connectedComponentsWithStats(blue.astype(np.uint8));parts=[];by=color_blue(result[i]);delta=np.abs(source[i].astype(np.int16)-result[i]).mean(axis=2)
            for k in range(1,count):
                x,y,w,h,area=stats[k]
                if area<40:continue
                selected=labels==k;parts.append(dict(bbox=[int(x),int(y),int(x+w),int(y+h)],frame=i,pixels=int(area),masked=int(np.count_nonzero(selected&mask)),result_still_blue=int(np.count_nonzero(selected&by)),near_original=int(np.count_nonzero(selected&by&(delta<=12)))))
            links=[]
            for t in alive:
                if i-t['rows'][-1]['frame']>3:continue
                for k,p in enumerate(parts):
                    score=overlap(t['rows'][-1]['bbox'],p['bbox'])
                    if score>=.15:links.append((score,t['id'],k))
            used_t=set();used_p=set()
            for score,tid,k in sorted(links,reverse=True):
                if tid in used_t or k in used_p:continue
                tracks[tid]['rows'].append(parts[k]);used_t.add(tid);used_p.add(k)
            for k,p in enumerate(parts):
                if k not in used_p:
                    t=dict(id=len(tracks),scene=scene,rows=[p]);tracks.append(t);alive.append(t)
            alive=[t for t in alive if i-t['rows'][-1]['frame']<=3]
            if i%60==0:print(f'源帧颜色部件与实际遮罩追踪 {i}/{n}',flush=True)
    selected=[]
    for t in tracks:
        rows=t['rows']
        if len(rows)<5 or max(r['masked']/r['pixels'] for r in rows)<.5:continue
        low=[r['frame'] for r in rows if r['masked']/r['pixels']<.5];empty=[r['frame'] for r in rows if r['masked']==0];part=[r['frame'] for r in rows if r['masked']/r['pixels']<.95]
        t.update(first=rows[0]['frame'],last=rows[-1]['frame'],source_observed_frames=len(rows),source_observation_gaps=intervals(set(range(rows[0]['frame'],rows[-1]['frame']+1))-{r['frame'] for r in rows}),mask_empty_while_source_visible=intervals(empty),mask_below_half_while_source_visible=intervals(low),mask_below_95_percent=intervals(part),minimum_coverage=min(r['masked']/r['pixels'] for r in rows),maximum_source_pixels=max(r['pixels'] for r in rows));selected.append(t)
    write(out/'源帧蓝色部件轨迹.json',dict(tracks=selected,seconds=time.monotonic()-started,settings=dict(minimum_color_component=40,minimum_lifetime=5,minimum_ever_masked_fraction=.5,association_bbox_iou=.15,maximum_observation_gap=2),说明='源帧颜色部件是诊断候选，不是语义真值。只保留至少一帧被实际水印掩膜覆盖过半且持续≥5帧的部件，排除冻结实物保护。源部件形变/分裂会影响身份关联，不能把源观察断档自动判作漏检。'))
    print(f'颜色部件追踪完成，{len(selected)}条候选轨迹',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['masks','source','result','output']:p.add_argument('--'+k,type=Path,required=True)
    execute(p.parse_args())
