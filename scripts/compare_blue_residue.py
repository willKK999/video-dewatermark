"""在同一冻结掩膜/源颜色候选范围内，只读比较两份既有成片。"""
import argparse,json,subprocess,time
from pathlib import Path
import cv2,numpy as np
from scripts.diagnose_watermark_residue import color_blue,write


def execute(args):
    base=args.masks;out=args.output;out.mkdir(parents=True,exist_ok=True);e=out/'蓝色统计证据';e.mkdir(exist_ok=True);files=sorted((base/'masks').glob('*.png'));n=len(files);shape=cv2.imread(str(files[0])).shape+(3,);fps=args.fps;selected={round(float(t)*fps) for t in args.times.split(',')}|set(range(0,n,round(fps*2)))|set(range(round(3.4*fps),round(5*fps),3));processes=[];rows=[];started=time.monotonic()
    for quality_path in args.quality:
        quality=json.loads(quality_path.read_text())
        for row in quality.get('实际掩膜全长闪烁候选',[]):
            selected.update(j for j in range(row['frame']-2,row['frame']+3) if 0<=j<n)
    for path in [args.source,args.reference,args.video]:
        p=subprocess.Popen(['timeout','600','ffmpeg','-v','error','-nostdin','-threads','1','-i',str(path),'-an','-frames:v',str(n),'-vsync','0','-f','rawvideo','-pix_fmt','bgr24','-threads','1','pipe:1'],stdout=subprocess.PIPE);processes.append(p)
    try:
        for i,file in enumerate(files):
            images=[]
            for p in processes:
                data=p.stdout.read(int(np.prod(shape)))
                if len(data)!=int(np.prod(shape)):raise ValueError('颜色对比解码帧不完整')
                images.append(np.frombuffer(data,np.uint8).reshape(shape))
            source,reference,video=images;m=cv2.imread(str(file),0)>0;source_blue=color_blue(source);domain=np.zeros(m.shape,bool);guard_path=base/'scene-protection'/file.name;guard=cv2.imread(str(guard_path),0)>0 if guard_path.exists() else np.zeros_like(m)
            for p in (base/'group-objects').glob('*/'+file.name):
                gm=cv2.imread(str(p),0)
                if np.count_nonzero(source_blue&(gm>0))<20:continue
                domain|=(cv2.dilate(gm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(49,49)))>0)&~guard
            row=dict(frame=i,seconds=i/fps);residuals=[]
            for label,x in [('reference',reference),('new',video)]:
                blue=color_blue(x)&domain;near=blue&source_blue&(np.abs(x.astype(np.int16)-source).mean(axis=2)<=12);distance=cv2.distanceTransform(m.astype(np.uint8),cv2.DIST_L2,5);row[label]=dict(blue_inside=int(np.count_nonzero(blue&m)),blue_outside=int(np.count_nonzero(blue&~m)),near_original_inside=int(np.count_nonzero(near&m)),near_original_outside=int(np.count_nonzero(near&~m)),deep16=int(np.count_nonzero(blue&(distance>16))))
                residual=x.copy();residual[blue&m]=residual[blue&m]*.25+np.array([0,0,255])*.75;residual[blue&~m]=residual[blue&~m]*.25+np.array([255,0,255])*.75;residuals.append(residual)
            rows.append(row)
            if i in selected:
                for label,x in zip(['source','v13e','v16','v13e-residual','v16-residual'],images+residuals):cv2.imwrite(str(e/f'{i:06d}-{label}.jpg'),x,[cv2.IMWRITE_JPEG_QUALITY,97])
            if i%60==0:print(f'同域蓝色残留对比 {i}/{n}',flush=True)
        for p in processes:
            if p.wait(timeout=30):raise ValueError('颜色对比ffmpeg退出异常')
    finally:
        for p in processes:
            if p.poll() is None:p.kill();p.wait(timeout=10)
    summary={label:{key:sum(row[label][key] for row in rows) for key in rows[0][label]} for label in ['reference','new']};write(out/'蓝色残留统计.json',dict(frames=rows,summary=summary,seconds=time.monotonic()-started,scope='同一冻结确认掩膜及全部基础/联动对象24像素邻域，扣已知场景保护；HSV H85—130,S100+,V80+。颜色候选不是人工真值，不能把阈值数等同残留已清零。'))
    print('780帧同域蓝色统计完成',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['masks','source','reference','video','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--quality',type=Path,action='append',default=[]);p.add_argument('--fps',type=float,default=30);p.add_argument('--times',default='0,3.4,3.97,4.5,6.8,13.7,17.1,20.5,25.73');execute(p.parse_args())
