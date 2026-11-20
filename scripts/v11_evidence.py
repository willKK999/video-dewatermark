"""v11固定抽样与来源证据；只读掩膜，不据审查ID修改算法。"""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlay(image,mask,color=(255,50,30),alpha=.55):
    result=image.copy();active=mask>0
    result[active]=(image[active]*(1-alpha)+np.array(color)*alpha).astype('uint8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','sam','v10','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();root=args.root;dest=args.output;ev=dest/'evidence';ev.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();files=sorted((root/'frames').glob('*.png'))
    shape=cv2.imread(str(files[0]),0).shape;h,w=shape
    fontpath=next((p for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'] if Path(p).exists()))
    font=ImageFont.truetype(fontpath,24)
    def image(i):return cv2.cvtColor(cv2.imread(str(files[i])),cv2.COLOR_BGR2RGB)
    def mask(folder,i):
        path=folder/files[i].name
        return cv2.imread(str(path),0) if path.exists() else np.zeros(shape,np.uint8)
    def panel(images,labels):
        canvas=Image.new('RGB',(w*len(images),h+52),'#202020');draw=ImageDraw.Draw(canvas)
        for j,(im,label) in enumerate(zip(images,labels)):
            canvas.paste(Image.fromarray(im),(j*w,52));draw.text((j*w+8,10),label,font=font,fill='white')
        return canvas
    times=[0,3.4,6.8,10.3,13.7,17.1];tiles=[];sources=[]
    for second in times:
        i=round(second*30);im=image(i)
        images=[im]+[overlay(im,mask(f,i)) for f in [args.sam/'masks',args.v10/'masks',root/'masks']]
        labels=[f'{second:.1f}秒 {x}' for x in ['原帧','SAM3.1','v10','v11']]
        big=panel(images,labels);big.save(ev/f'四列对比-{second:.1f}秒.jpg',quality=96)
        tiles.append(big.resize((1200,round(big.height*1200/big.width))))
        for name,color in [('SAM',(0,200,255)),('OCR',(0,255,80)),('temporal',(255,210,0)),('removed',(255,0,210))]:
            sources.append(dict(second=second,source=name,pixels=int(np.count_nonzero(mask(root/name,i)))))
        ims=[im]+[overlay(im,mask(root/name,i),color) for name,color in [('SAM',(0,200,255)),('OCR',(0,255,80)),('temporal',(255,210,0)),('removed',(255,0,210))]]
        panel(ims,['原帧','SAM保留（青）','OCR紧框（绿）','时域补全（黄）','SAM被过滤（紫）']).save(ev/f'来源分解-{second:.1f}秒.jpg',quality=96)
        colored=im.copy()
        for name,color in [('SAM',(0,200,255)),('OCR',(0,255,80)),('temporal',(255,210,0))]: colored=overlay(colored,mask(root/name,i),color,.45)
        panel([im,colored,cv2.cvtColor(mask(root/'masks',i),cv2.COLOR_GRAY2RGB)],['原帧','v11来源叠加','v11二值']).save(ev/f'细节-{second:.1f}秒.jpg',quality=97)
    overview=Image.new('RGB',(1200,sum(im.height for im in tiles)))
    y=0
    for im in tiles:overview.paste(im,(0,y));y+=im.height
    overview.save(dest/'掩膜对比图.jpg',quality=95)
    # 每两秒固定抽样，全画幅保留，避免通过裁剪隐藏误盖。
    subtitles=json.loads((root/'detections.json').read_text())['subtitles'];sampling=[];thumbs=[]
    for i in range(0,len(files),60):
        im=image(i);m=mask(root/'masks',i);sub=np.zeros(shape,np.uint8)
        from worker.v11_masks import rectangle
        settings=json.loads((root/'detections.json').read_text())['settings']
        for obs in subtitles.get(str(i),[]): sub|=rectangle(obs['box'],shape,settings['subtitle_padding'])
        sampling.append(dict(frame=i,second=i/30,ocr_dialogue_pixels=int(np.count_nonzero(sub)),
                             overlap=int(np.count_nonzero((sub>0)&(m>0)))))
        big=panel([im,overlay(im,m),overlay(im,sub,(80,120,255))],[f'{i/30:.1f}秒 原帧','v11掩膜叠加','OCR非共识文字区'])
        big.save(ev/f'固定间隔-{i/30:.1f}秒.jpg',quality=95)
        thumbs.append(big.resize((900,round(big.height*900/big.width))))
    grid=Image.new('RGB',(1800,sum(t.height for t in thumbs[::2])))
    for j,im in enumerate(thumbs):grid.paste(im,((j%2)*900,(j//2)*im.height))
    grid.save(ev/'全长每2秒对白审查.jpg',quality=94)
    # 已知误盖仅作为结果核验索引，绝不参与过滤。
    known={'公告栏早段':('w000000-p1-o1',111),'归因纠正-黄果图标':('w000120-p1-o0',172),
           '公告栏后段':('w000120-p1-o3',171),'对白混合一':('w000480-p0-o0',542),'对白混合二':('w000720-p0-o1',772)}
    checks=[]
    for label,(key,i) in known.items():
        raw=mask(args.sam/'raw-object-masks'/key,i);kept=mask(root/'sam-objects'/key,i)
        im=image(i);panel([im,overlay(im,raw),overlay(im,kept),overlay(im,raw&~kept,(255,0,210))],
            [label+' 原帧','原始SAM对象','过滤后同对象','扣除部分']).save(ev/f'已知问题-{label}.jpg',quality=97)
        checks.append(dict(label=label,object=key,frame=i,original=int(np.count_nonzero(raw)),retained=int(np.count_nonzero(kept)),
                           removed=int(np.count_nonzero(raw&~kept))))
    motion=json.loads((root/'motion-audit.json').read_text())
    trackdata=json.loads((args.sam/'metrics.json').read_text())['tracks']
    motion_checks=[]
    previous=json.loads((args.v10/'motion-audit.json').read_text())
    for group,decision in motion.items():
        keys=sorted(set(member.split('@')[0] for member in decision['members']))
        if not decision['rejected'] and not any(previous.get(k,{}).get('rejected') for k in keys):continue
        scene=decision['scene'];a,b=json.loads((root/'detections.json').read_text())['scenes'][scene]
        indices=sorted({i for k in keys for i in trackdata[k]['frames'] if a<=i<b})
        def raw_union(i):
            m=np.zeros(shape,np.uint8)
            for k in keys:m|=mask(args.sam/'raw-object-masks'/k,i)
            return m
        i=max(indices,key=lambda j:np.count_nonzero(raw_union(j)))
        im=image(i);raw=raw_union(i)
        panel([im,overlay(im,raw),overlay(im,mask(args.v10/'masks',i)),overlay(im,mask(root/'masks',i))],
              [f'组{group} {i/30:.2f}秒 原帧','原始候选','v10最终联合','v11最终联合']).save(ev/f'形状判据-组{group}.jpg',quality=96)
        motion_checks.append(dict(group=group,objects=keys,frame=i,rejected=decision['rejected'],ocr_protected=decision['ocr_protected']))
    for i in [542,600,772]:
        im=image(i)
        panel([im,overlay(im,mask(args.v10/'masks',i)),overlay(im,mask(root/'masks',i))],
            [f'{i/30:.2f}秒 原帧','v10最终联合','v11最终联合']).save(ev/f'对白最终联合-{i/30:.2f}秒.jpg',quality=97)
    (dest/'evidence-statistics.json').write_text(json.dumps(dict(sources=sources,sampling=sampling,known=checks,motion_checks=motion_checks,seconds=time.monotonic()-started),ensure_ascii=False,indent=2))
    print('证据生成完成',flush=True)


if __name__=='__main__':main()
