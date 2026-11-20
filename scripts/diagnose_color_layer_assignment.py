from pathlib import Path
import json,cv2,numpy as np,time
from scripts.diagnose_watermark_residue import color_blue,intervals,write
import argparse
p=argparse.ArgumentParser(description='只读统计源颜色部件跨图层归属');p.add_argument('--output',type=Path,required=True);args=p.parse_args();r=args.output;root=r/'输入缓存/冻结掩膜';d=json.loads((r/'源帧蓝色部件轨迹.json').read_text());src=np.memmap(r/'输入缓存/source.bgr',dtype=np.uint8,mode='r',shape=(len(json.loads((r/'逐帧诊断.json').read_text())['frames']),*cv2.imread(str(root/'masks/000000.png')).shape,3));records=[]
for tr in d['tracks']:
 if tr['maximum_source_pixels']<3000:continue
 rows=[]
 for row in tr['rows']:
  i=row['frame'];a,b,c,z=row['bbox'];part=color_blue(src[i])[b:z,a:c];counts=[];bin_counts={}
  for k in range(3):
   m=cv2.imread(str(root/'layers'/f'{k:02d}'/f'{i:06d}.png'),0)[b:z,a:c]>0;counts.append(int(np.count_nonzero(part&m)))
  for p in (root/'group-objects').glob('*/'+f'{i:06d}.png'):
   m=cv2.imread(str(p),0)[b:z,a:c]>0;n=int(np.count_nonzero(part&m))
   if n:bin_counts[p.parent.name]=n
  rows.append(dict(frame=i,source_blue=int(part.sum()),layers=counts,bins=bin_counts))
 keys={g for x in rows for g in x['bins']};records.append(dict(track=tr['id'],scene=tr['scene'],first=tr['first'],last=tr['last'],maximum_source_pixels=tr['maximum_source_pixels'],rows=rows,layer_majority_intervals={str(k):intervals(x['frame'] for x in rows if x['layers'][k]>=x['source_blue']*.5) for k in range(3)},bin_majority_intervals={g:intervals(x['frame'] for x in rows if x['bins'].get(g,0)>=x['source_blue']*.5) for g in sorted(keys)}))
 print('主要蓝色部件图层归属完成',tr['id'],flush=True)
write(r/'主要蓝色部件图层归属.json',dict(tracks=records,说明='按源颜色部件包围框内蓝色像素统计实际各层/各桶覆盖；不是修改或重分配图层。'))
