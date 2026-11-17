"""只读诊断冻结掩膜、颜色残留和ROI上下文；不调用检测或补画模型。"""
import argparse,csv,hashlib,json,time
from pathlib import Path
from functools import lru_cache
import cv2,numpy as np


def intervals(indices):
    result=[]
    for i in sorted(set(indices)):
        if result and i==result[-1][1]+1:result[-1][1]=int(i)
        else:result.append([int(i),int(i)])
    return result


def bounds(mask):
    y,x=np.where(mask)
    return None if not len(x) else [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]


def color_blue(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    return (hsv[:,:,0]>=85)&(hsv[:,:,0]<=130)&(hsv[:,:,1]>=100)&(hsv[:,:,2]>=80)


def write(path,data):
    from worker.json_types import json_native
    path.write_text(json.dumps(json_native(data),ensure_ascii=False,indent=2)+'\n')
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def execute(args):
    from worker.spatial_roi import SpatialSettings,plan
    from worker.roi_inpaint import ROISettings
    tick=time.monotonic();out=args.output;out.mkdir(exist_ok=True,parents=True);cache=args.masks;profile=read(args.run/'profile.json');report=read(args.run/'run/roi-report.json');data=read(cache/'detections.json');frozen=read(args.run/'input-binding.json');manifest=read(cache/'layers.json');gp=read(cache/'group-plan.json');fps=data['settings']['fps'];n=len(frozen['frames']);h,w=args.height,args.width
    source=np.memmap(args.source,mode='r',dtype=np.uint8,shape=(n,h,w,3));result=np.memmap(args.result,mode='r',dtype=np.uint8,shape=(n,h,w,3));groups=sorted(p.name for p in (cache/'group-objects').iterdir() if p.is_dir());active={f"group-{g['id']}":g for g in gp['groups'] if g['active']};area={g:np.zeros(n,int) for g in groups};boxes={g:[None]*n for g in groups};group_layer={g:np.zeros(len(manifest['layers']),int) for g in groups};samples=set(round(float(t)*fps) for t in args.times.split(','));e=out/'evidence';e.mkdir(exist_ok=True)
    @lru_cache(maxsize=12)
    def masks(i):
        name=f'{i:06d}.png';union=cv2.imread(str(cache/'masks'/name),0);layers=[cv2.imread(str(cache/'layers'/f'{k:02d}'/name),0) for k in range(len(manifest['layers']))]
        return union,layers
    owners=[];spatial=SpatialSettings(**read(args.run/'run/spatial-roi.json'));cfg=ROISettings(**profile['ROI']);plans={}
    for t in report['任务']:
        key=t['图层'],t['开始帧'],t['结束帧']
        if key not in plans:
            k,a,b=key;footprint=np.zeros((h,w),np.uint8)
            for i in range(a,b):footprint|=masks(i)[1][k]
            plans[key]=iter(plan(footprint,b-a+3*profile['重叠'],cfg,spatial))
        p=next(plans[key]);assert p['box']==t['ROI'] and np.count_nonzero(p['ownership'])==t['施工面积像素'];owners.append(p['ownership'])
    for it in plans.values():assert next(it,None) is None
    print('全部任务施工域与实际ROI记录重建一致',flush=True)
    taskrows=[dict(task=t['任务'],layer=t['图层'],start=t['开始帧'],end_exclusive=t['结束帧'],roi=t['ROI'],attempts=[{k:a[k] for k in ['尝试','开始帧','结束帧','通过','重叠最大差异']} for a in t['尝试']],frames=[]) for t in report['任务']];rows=[];grows=[];layerrows=[];checks=[]
    with (out/'逐帧水印组统计.csv').open('w') as f:
        fields=['frame','seconds','group','area','bbox','source_blue','source_blue_inside','source_blue_outside','result_blue_inside','result_blue_outside','near_original_inside','near_original_outside','deep_inside_blue'];writer=csv.DictWriter(f,fields);writer.writeheader()
        for i in range(n):
            name=f'{i:06d}.png';m,layers=masks(i);check=dict(frame=i,union_sha_match=sha(cache/'masks'/name)==frozen['frames'][i]['mask'],layers_union_match=np.array_equal(np.bitwise_or.reduce(layers),m),group_union_match=True)
            for k,l in enumerate(layers):
                if sha(cache/'layers'/f'{k:02d}'/name)!=manifest['layers'][k]['sha256'][name]:raise ValueError('图层摘要与原清单不符')
            if not check['union_sha_match'] or not check['layers_union_match']:raise ValueError('实际补画绑定与掩膜不符')
            x,y=source[i],result[i];sx,sy=color_blue(x),color_blue(y);delta=np.abs(x.astype(np.int16)-y).mean(axis=2);allgroup=np.zeros((h,w),np.uint8);domain=np.zeros((h,w),bool);groupvalues={}
            for g in groups:
                p=cache/'group-objects'/g/name
                gm=cv2.imread(str(p),0) if p.exists() else np.zeros_like(m);allgroup|=gm;area[g][i]=np.count_nonzero(gm);boxes[g][i]=bounds(gm>0)
                if not gm.any():continue
                for k,l in enumerate(layers):group_layer[g][k]+=np.count_nonzero(gm&l)
                # 颜色定义为通用HSV蓝/青蓝区间，限定在冻结确认组附近，未按影片坐标删选。
                dom=cv2.dilate(gm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(49,49)))>0
                protection=cache/'scene-protection'/name
                if protection.exists():dom&=cv2.imread(str(protection),0)==0
                seed=sx&(gm>0)
                if seed.sum()<20:continue
                domain|=dom;dist=cv2.distanceTransform((m>0).astype(np.uint8),cv2.DIST_L2,5);preserved=dom&sx&sy;blue=dom&sy;inside=m>0;near=preserved&(delta<=12)
                row=dict(frame=i,seconds=i/fps,group=g,area=int(area[g][i]),bbox=boxes[g][i],source_blue=int(np.count_nonzero(dom&sx)),source_blue_inside=int(np.count_nonzero(dom&sx&inside)),source_blue_outside=int(np.count_nonzero(dom&sx&~inside)),result_blue_inside=int(np.count_nonzero(blue&inside)),result_blue_outside=int(np.count_nonzero(blue&~inside)),near_original_inside=int(np.count_nonzero(near&inside)),near_original_outside=int(np.count_nonzero(near&~inside)),deep_inside_blue=int(np.count_nonzero(blue&(dist>3))))
                writer.writerow(row);grows.append(row);groupvalues[g]=row
            check['group_union_match']=np.array_equal(allgroup,m)
            if not check['group_union_match']:raise ValueError('水印组并集与实际补画掩膜不同')
            checks.append(check);dist=cv2.distanceTransform((m>0).astype(np.uint8),cv2.DIST_L2,5);blue=sy&domain;retained=blue&sx;near=retained&(delta<=12);inside=m>0
            row=dict(frame=i,seconds=i/fps,mask_area=int(inside.sum()),source_blue=int(np.count_nonzero(sx&domain)),source_blue_inside=int(np.count_nonzero(sx&domain&inside)),source_blue_outside=int(np.count_nonzero(sx&domain&~inside)),blue_inside=int(np.count_nonzero(blue&inside)),blue_outside=int(np.count_nonzero(blue&~inside)),retained_blue_inside=int(np.count_nonzero(retained&inside)),near_original_inside=int(np.count_nonzero(near&inside)),near_original_outside=int(np.count_nonzero(near&~inside)),blue_deep3=int(np.count_nonzero(blue&(dist>3))),blue_deep8=int(np.count_nonzero(blue&(dist>8))),blue_deep16=int(np.count_nonzero(blue&(dist>16))))
            rows.append(row)
            for k,l in enumerate(layers):layerrows.append(dict(frame=i,layer=k,area=int(np.count_nonzero(l)),blue=int(np.count_nonzero(blue&(l>0)))))
            for num,t in enumerate(report['任务']):
                trial=next(a for a in reversed(t['尝试']) if a['通过']);left,right=trial['开始帧'],trial['结束帧']
                if not left<=i<right:continue
                k=t['图层'];a,b,c,d=t['ROI'];lm=layers[k]>0;prior=np.bitwise_or.reduce(layers[:k])>0 if k else np.zeros_like(inside);future=inside&~lm&~prior;sl=np.s_[b:d,a:c];own=owners[num]>0
                activeblue=blue&lm&own;source_scope=sx&domain
                # 未来图层未施工的像素仍是原帧；此处只计颜色候选，不认定已发生模型传播。
                q=dict(frame=i,context=not(t['开始帧']<=i<t['结束帧']),edge_context=i<left+8 or i>=right-8,source_blue_unmasked_by_layer=int(np.count_nonzero(source_scope[sl]&~lm[sl])),source_blue_unmasked_by_union=int(np.count_nonzero(source_scope[sl]&~inside[sl])),source_blue_future_layer=int(np.count_nonzero(source_scope[sl]&future[sl])),result_blue_owned=int(np.count_nonzero(activeblue)),masked_owned_pixels=int(np.count_nonzero(lm&own)))
                # 本任务施工域在裁块外的像素必须为0；与其他独立块无关。
                outside=own.copy();outside[sl]=False;q['owned_pixels_outside_roi']=int(np.count_nonzero(outside&lm));q['owned_near_internal_roi_edge']=0
                band=np.zeros_like(inside)
                if a>0:band[b:d,a:min(c,a+3)]=True
                if c<w:band[b:d,max(a,c-3):c]=True
                if b>0:band[b:min(d,b+3),a:c]=True
                if d<h:band[max(b,d-3):d,a:c]=True
                q['owned_near_internal_roi_edge']=int(np.count_nonzero(band&lm&own));taskrows[num]['frames'].append(q)
            if i in samples:
                overlay=x.copy();overlay[inside]=overlay[inside]*.4+np.array([0,180,255])*.6;red=y.copy();red[blue&inside]=red[blue&inside]*.25+np.array([0,0,255])*.75;red[blue&~inside]=red[blue&~inside]*.25+np.array([255,0,255])*.75
                for label,im in [('source',x),('mask',overlay),('result',y),('residual',red)]:cv2.imwrite(str(e/f'{i:06d}-{label}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,97])
                cv2.imwrite(str(e/f'{i:06d}-blue-inside.png'),(blue&inside).astype(np.uint8)*255);cv2.imwrite(str(e/f'{i:06d}-blue-outside.png'),(blue&~inside).astype(np.uint8)*255)
            if i%30==0:print(f'逐帧掩膜/残留/上下文诊断 {i}/{n}',flush=True)
    events=[]
    for g in groups:
        a=area[g];present=np.where(a>0)[0]
        if not len(present):continue
        first,last=int(present[0]),int(present[-1]);low=[];jumps=[];abrupt=[]
        for i in range(first,last+1):
            local=a[max(first,i-8):min(last+1,i+9)];median=float(np.median(local[local>0])) if np.any(local>0) else 0
            if 0<a[i]<median*.5:low.append(i)
            if i>first and a[i] and a[i-1]:
                if min(a[i],a[i-1])/max(a[i],a[i-1])<.5:abrupt.append(i)
                bb=np.array(boxes[g][i]);old=np.array(boxes[g][i-1]);shift=np.linalg.norm((bb[:2]+bb[2:]-old[:2]-old[2:])/2);scale=np.linalg.norm(old[2:]-old[:2])
                if shift>max(8,scale*.3):jumps.append(i)
        events.append(dict(group=g,confirmed=g in active,scene=active.get(g,{}).get('scene'),first=first,last=last,active_frames=len(present),missing=intervals(np.where((a==0)&(np.arange(n)>=first)&(np.arange(n)<=last))[0]),small=intervals(low),abrupt_area=intervals(abrupt),jump=intervals(jumps),minimum=int(a[present].min()),maximum=int(a.max()),layer_overlaps=group_layer[g].tolist(),areas=a.tolist(),boxes=boxes[g],说明='base编号是原图层剩余桶，不是跨镜头对象ID；面积/跳变仅为审计候选，缩放/遮挡/桶内容切换也会触发'))
    write(out/'逐帧诊断.json',dict(fps=fps,frames=rows,groups=events,layers=layerrows,checks=checks,tasks=taskrows,group_residual=grows,seconds=time.monotonic()-tick,thresholds=dict(blue_hue_opencv=[85,130],minimum_saturation=100,minimum_value=80,near_original_rgb_mae=12,diagnostic_ring=24,small_area_ratio=.5,local_median_half_window=8,centroid_jump_diagonal_ratio=.3),说明='颜色候选不是水印真值；掩膜内蓝色可被模型改色后仍保留，不要求与原帧像素完全相同。所有分析只读。'))
    print('全部780帧诊断完成',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['masks','run','source','result','output']:p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--width',type=int,default=720);p.add_argument('--height',type=int,default=1280);p.add_argument('--times',default='0,3.4,3.97,4.5,6.8,13.7,17.1,25.73');execute(p.parse_args())
