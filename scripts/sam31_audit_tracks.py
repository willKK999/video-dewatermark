"""生成SAM3.1所有候选轨迹的最大面积代表帧，供独立语义审查。"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',required=True);p.add_argument('--result',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.result);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data=json.loads((root/'metrics.json').read_text());rows=[];tiles=[]
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    bigfont=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',22)
    files=sorted(Path(args.frames).glob('*.png'))
    for key,track in data['tracks'].items():
        best=None
        for i in track['frames']:
            mask=cv2.imread(str(root/'raw-object-masks'/key/files[i].name),0)
            count=int(np.count_nonzero(mask))
            if best is None or count>best[0]:best=(count,i,mask)
        count,i,mask=best;frame=cv2.imread(str(files[i]));overlay=frame.copy()
        overlay[mask>0]=(overlay[mask>0]*.35+np.array([0,0,255])*.65).astype(np.uint8)
        row=dict(identity=key,prompt=track['prompt'],retained=track['retained'],frame=i,
                 pixels=count,fraction=count/mask.size,first=min(track['frames']),last=max(track['frames']),
                 observations=len(track['frames']),rejection=track.get('rejection'))
        rows.append(row)
        a=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));b=Image.fromarray(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
        pair=Image.new('RGB',(a.width*2,a.height+55),'#202020');d=ImageDraw.Draw(pair)
        d.text((8,8),f'{key} 帧{i} {track["prompt"]} 保留={track["retained"]} 面积{count/mask.size:.2%}',font=bigfont,fill='white')
        pair.paste(a,(0,55));pair.paste(b,(a.width,55));pair.save(out/f'轨迹-{key}.jpg',quality=93)
        tile=Image.new('RGB',(240,490),'#202020');d=ImageDraw.Draw(tile)
        d.text((3,3),key+f' F{i}',font=font,fill='white')
        d.text((3,22),f'{track["prompt"]} 保留={track["retained"]}',font=font,fill='white')
        d.text((3,41),f'{count/mask.size:.2%} 帧{min(track["frames"])}—{max(track["frames"])}',font=font,fill='white')
        tile.paste(b.resize((240,427)),(0,63));tiles.append(tile)
    for start in range(0,len(tiles),24):
        canvas=Image.new('RGB',(1440,1960),'#202020')
        for j,tile in enumerate(tiles[start:start+24]):canvas.paste(tile,((j%6)*240,(j//6)*490))
        canvas.save(out/f'轨迹总览-{start//24+1}.jpg',quality=94)
    (out/'轨迹代表帧.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    print('已生成全部候选轨迹代表帧:',len(rows))

if __name__=='__main__':main()
