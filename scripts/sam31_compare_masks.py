"""对完整同规格掩膜作差异统计，并生成指定抽样时点的对比证据。"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compare(frames, versions, output, samples, fps):
    started=time.perf_counter()
    output=Path(output); (output/'evidence').mkdir(parents=True,exist_ok=True)
    files=sorted(Path(frames).glob('*.png'))
    if not files: raise ValueError('原帧目录为空')
    names=[p.name for p in files]
    shape=cv2.imread(str(files[0])).shape[:2]
    for label,directory in versions.items():
        if sorted(p.name for p in Path(directory).glob('*.png')) != names:
            raise ValueError(f'{label}掩膜帧集合与原帧不一致')
    rows=[]
    for file in files:
        masks={}
        for label,directory in versions.items():
            mask=cv2.imread(str(Path(directory)/file.name),0)
            if mask is None or mask.shape!=shape or not set(np.unique(mask))<={0,255}:
                raise ValueError(f'{label}掩膜形状或二值错误')
            masks[label]=mask>0
        row={'frame':file.name,'pixels':{k:int(m.sum()) for k,m in masks.items()},'differences':{}}
        for label,mask in masks.items():
            for base in ['v6','v8','v9']:
                if base not in masks or label==base: continue
                row['differences'][label+'相对'+base]={'added':int((mask&~masks[base]).sum()),'removed':int((masks[base]&~mask).sum())}
        rows.append(row)
    columns=['原帧']+list(versions)
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    width=shape[1];height=shape[0]
    thumb_w=300;thumb_h=round(height*thumb_w/width)
    overview=Image.new('RGB',(len(columns)*thumb_w,len(samples)*(thumb_h+40)),'#202020')
    for r,second in enumerate(samples):
        idx=round(second*fps)
        file=files[idx]
        image=cv2.imread(str(file))
        panels=[]
        for label in columns:
            panel=image.copy()
            if label!='原帧':
                mask=cv2.imread(str(Path(versions[label])/file.name),0)>0
                panel[mask]=(panel[mask]*.45+np.array([0,0,255])*.55).astype(np.uint8)
            panels.append(Image.fromarray(cv2.cvtColor(panel,cv2.COLOR_BGR2RGB)))
        large=Image.new('RGB',(len(columns)*width,height+55),'#202020')
        draw=ImageDraw.Draw(large)
        for c,(label,panel) in enumerate(zip(columns,panels)):
            draw.text((c*width+5,8),f'{second:.1f}秒 {label}',font=font,fill='white')
            large.paste(panel,(c*width,55))
            tile=Image.new('RGB',(thumb_w,thumb_h+40),'#202020')
            ImageDraw.Draw(tile).text((3,5),f'{second:.1f}秒 {label}',font=font,fill='white')
            tile.paste(panel.resize((thumb_w,thumb_h)),(0,40))
            overview.paste(tile,(c*thumb_w,r*(thumb_h+40)))
        large.save(output/'evidence'/f'掩膜对比-{second:.1f}秒.jpg',quality=96)
    overview.save(output/'掩膜对比图.jpg',quality=94)
    totals={}
    for row in rows:
        for pair,delta in row['differences'].items():
            dest=totals.setdefault(pair,{'added_pixel_frames':0,'removed_pixel_frames':0})
            dest['added_pixel_frames']+=delta['added'];dest['removed_pixel_frames']+=delta['removed']
    data={'frame_count':len(files),'shape':shape,'fps':fps,'totals':totals,'frames':rows,'seconds':time.perf_counter()-started}
    (output/'mask-comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',required=True);p.add_argument('--versions',required=True)
    p.add_argument('--output',required=True);p.add_argument('--fps',type=float,required=True)
    p.add_argument('--samples',type=float,nargs='+',default=[0,3.4,6.8,10.3,13.7,17.1])
    args=p.parse_args()
    compare(args.frames,json.loads(Path(args.versions).read_text()),args.output,args.samples,args.fps)

if __name__=='__main__':main()
