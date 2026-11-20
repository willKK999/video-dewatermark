from pathlib import Path
import json,csv
import argparse
p=argparse.ArgumentParser(description='汇总只读诊断的异常区间与逐层输入缺口');p.add_argument('--output',type=Path,required=True);args=p.parse_args();r=args.output;read=lambda p:json.loads(p.read_text());d=read(r/'逐帧诊断.json');mapping=read(r/'主要蓝色部件图层归属.json');source_tracks={x['id']:x for x in read(r/'源帧蓝色部件轨迹.json')['tracks']};scenes=read(r/'输入缓存/冻结掩膜/detections.json')['scenes'];events=[]
def ranges(a):
 out=[]
 for i in sorted(set(a)):
  if out and i==out[-1][1]+1:out[-1][1]=i
  else:out.append([i,i])
 return out
for g in d['groups']:
 for scene,(a,b) in enumerate(scenes):
  present=[i for i in range(a,b) if g['areas'][i]>0]
  if not present:continue
  first,last=present[0],present[-1]
  for kind in ['missing','small','abrupt_area','jump']:
   if kind=='missing':indices=[i for i in range(first,last+1) if g['areas'][i]==0]
   else:indices=[i for x,y in g[kind] for i in range(max(a,x),min(b,y+1)) if not(kind in ['jump','abrupt_area'] and i==a)]
   for x,y in ranges(indices):events.append(dict(group=g['group'],scene=scene,kind=kind,start=x,end_inclusive=y,minimum_area=min(g['areas'][x:y+1]),maximum_area=max(g['areas'][x:y+1]),note='base为原图层剩余桶，不能把换桶当对象漏检；面积/框跳变需结合源图核对'))
with (r/'遮罩异常区间.csv').open('w') as f:
 w=csv.DictWriter(f,['group','scene','kind','start','end_inclusive','minimum_area','maximum_area','note']);w.writeheader();w.writerows(events)
exposure=[]
for tr in mapping['tracks']:
 original={x['frame']:x for x in source_tracks[tr['track']]['rows']}
 for t in d['tasks']:
  k=t['layer'];a,b,c,z=t['roi'];rows=[];trial=next(x for x in reversed(t['attempts']) if x['通过'])
  for x in tr['rows']:
   i=x['frame']
   if not trial['开始帧']<=i<trial['结束帧']:continue
   box=original[i]['bbox']
   if not(a<=box[0] and b<=box[1] and c>=box[2] and z>=box[3]):continue
   rows.append(dict(frame=i,pixels=x['source_blue'],covered=x['layers'][k],future_layer_pixels=sum(x['layers'][k+1:])))
  if not rows or not any(x['covered']>=x['pixels']*.5 for x in rows):continue
  low=[x for x in rows if x['covered']<x['pixels']*.5];future=[x for x in low if x['future_layer_pixels']>=x['pixels']*.5]
  if low:exposure.append(dict(track=tr['track'],task=t['task'],layer=k,input=[trial['开始帧'],trial['结束帧']],construction=[t['start'],t['end_exclusive']],roi=t['roi'],current_layer_below_half=ranges(x['frame'] for x in low),future_layer_original_exposure=ranges(x['frame'] for x in future),future_exposure_frames=len(future),future_exposure_pixels=sum(x['pixels']-x['covered'] for x in future),rows=rows))
summary=dict(events=events,per_layer_object_exposure=exposure,total_mask_internal_missing_intervals=sum(x['kind']=='missing' for x in events),roi_outside_owned_pixels=sum(x['owned_pixels_outside_roi'] for t in d['tasks'] for x in t['frames']),roi_owned_near_internal_edge_pixels=sum(x['owned_near_internal_roi_edge'] for t in d['tasks'] for x in t['frames']),all_rois_divisible_by_8=all((t['roi'][2]-t['roi'][0])%8==0 and (t['roi'][3]-t['roi'][1])%8==0 for t in d['tasks']))
(r/'诊断摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('内部缺图区间',summary['total_mask_internal_missing_intervals']);print('对象跨层模型输入缺口',[(x['track'],x['task'],x['layer'],x['current_layer_below_half'],x['future_layer_original_exposure'],x['future_exposure_frames']) for x in exposure]);print('面积变化候选',events)
