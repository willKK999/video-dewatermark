"""自动轨迹ROI补画：原生裁块、默认半精度、逐像素覆盖核验，不回退失败原帧。"""
from dataclasses import dataclass,asdict
from pathlib import Path
import json,math,os,time,shutil
import cv2,numpy as np
from .process import PipelineError,run
from .restoration import CleanReferences,soft_composite
from .hls_settings import HLSSettings
from .vision import windows,sha256


@dataclass(frozen=True)
class ROISettings:
    maximum_frames:int=120
    minimum_frames:int=32
    reference_pixels:int=640*360
    context_ratio:float=.25
    minimum_context:int=32
    overlap_frames:int=16
    maximum_retries:int=2
    fp16:bool=True

    @classmethod
    def from_env(cls):
        return cls(fp16=os.getenv('PROPAINTER_FP16','true').lower() not in {'false','0','no'})


def bbox(mask):
    y,x=np.where(mask>0)
    return None if not len(x) else [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]


def union_box(a,b):return [min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3])]


def intersects(a,b):return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])


def padded_box(box,shape,cfg):
    h,w=shape;a,b,c,d=box
    margin=max(cfg.minimum_context,round(max(c-a,d-b)*cfg.context_ratio))
    return [max(0,math.floor((a-margin)/8)*8),max(0,math.floor((b-margin)/8)*8),
            min(w,math.ceil((c+margin)/8)*8),min(h,math.ceil((d+margin)/8)*8)]


def crop(image,box,mask=False):
    a,b,c,d=box;part=image[b:d,a:c]
    if not part.size:raise PipelineError('ROI_INVALID','自动ROI为空')
    ph=(-part.shape[0])%8;pw=(-part.shape[1])%8
    return cv2.copyMakeBorder(part,0,ph,0,pw,cv2.BORDER_REPLICATE)


def frame_budget(area,cfg):
    return min(cfg.maximum_frames,max(cfg.minimum_frames,round(100*math.sqrt(cfg.reference_pixels/max(1,area)))))


def atomic_png(path,image):
    temporary=path.with_name(path.stem+'.writing.png')
    if not cv2.imwrite(str(temporary),image):raise PipelineError('OUTPUT_INVALID','ROI帧写入失败')
    temporary.replace(path)


def merge_groups(groups):
    out=[]
    for group in groups:
        joined=dict(group);again=True
        while again:
            again=False
            for old in list(out):
                if intersects(joined['box'],old['box']):
                    joined=dict(box=union_box(joined['box'],old['box']),tracks=joined['tracks']|old['tracks'])
                    out.remove(old);again=True
        out.append(joined)
    return out


def assert_coverage(work,files):
    missing=[];pixels=0
    for i,file in enumerate(files):
        mask=cv2.imread(str(work/'masks'/file.name),0);covered=cv2.imread(str(work/'roi-coverage'/file.name),0)
        if mask is None or covered is None:raise PipelineError('ROI_INCOMPLETE','ROI覆盖清单缺失')
        n=int(np.count_nonzero((mask>0)&(covered==0)))
        if n:missing.append(dict(frame=i,pixels=n));pixels+=n
    if missing:raise PipelineError('ROI_INCOMPLETE','存在未补画的水印像素，禁止原帧补齐',details=dict(missing=missing,missing_pixels=pixels))


def execute(work,files,detected,log=lambda message:None,cfg=None):
    cfg=cfg or ROISettings.from_env();started=time.monotonic();shape=cv2.imread(str(files[0])).shape[:2]
    source_pixels=shape[0]*shape[1];dest=work/'composite';dest.mkdir();coverage=work/'roi-coverage';coverage.mkdir()
    for file in files:
        os.link(file,dest/file.name);atomic_png(coverage/file.name,np.zeros(shape,np.uint8))
    masks=[cv2.imread(str(work/'masks'/p.name),0) for p in files]
    if any(m is None for m in masks):raise PipelineError('MASK_INVALID','ROI缺少全量掩膜')
    layers=json.loads((work/'layers.json').read_text())['layers'];records=[];counter=0
    settings=HLSSettings.from_env();real_pixels=0;model_pixels=0
    def process(box,owned,start,end,scene,depth=0,planning_depth=0):
        nonlocal counter,real_pixels,model_pixels
        active=[i for i in range(start,end) if np.any(owned[i])]
        if not active:return
        start,end=min(active),max(active)+1
        footprint=np.zeros(shape,np.uint8)
        for i in active:footprint|=owned[i]
        box=padded_box(bbox(footprint),shape,cfg)
        if box==[0,0,shape[1],shape[0]]:
            if planning_depth>=8:raise PipelineError('ROI_TOO_LARGE','无法形成有上下文的局部ROI，禁止回退整帧补画')
            if end-start>16:
                mid=(start+end)//2
                process(box,owned,start,min(end,mid+4),scene,depth,planning_depth+1)
                process(box,owned,max(start,mid-4),end,scene,depth,planning_depth+1)
            else:
                axis=0 if shape[0]>=shape[1] else 1;middle=shape[axis]//2
                halves=[{},{}]
                for i,mask in owned.items():
                    a=mask.copy();b=mask.copy()
                    if axis==0:a[middle:]=0;b[:middle]=0
                    else:a[:,middle:]=0;b[:,:middle]=0
                    halves[0][i]=a;halves[1][i]=b
                for half in halves:process(box,half,start,end,scene,depth,planning_depth+1)
            return
        # 仅源片自身提供时间上下文，不跨镜头；上下文帧没有施工目标时仍作为真实参考。
        left=max(scene[0],start-2);right=min(scene[1],end+2)
        counter+=1;task=work/f'roi-{counter:04d}';task.mkdir();(task/'frames').mkdir();(task/'masks').mkdir()
        a,b,c,d=box;native_h,native_w=d-b,c-a
        before=[];model_masks=[]
        for j in range(left,right):
            image=cv2.imread(str(dest/files[j].name));done=cv2.imread(str(coverage/files[j].name),0)
            effective=masks[j]&~done
            clipped=effective[b:d,a:c]
            if np.count_nonzero(clipped)/source_pixels>settings.mask_fraction_limit:
                raise PipelineError('MASK_UNCERTAIN','ROI模型输入掩膜超过20%原帧面积保护')
            before.append(image);model_masks.append(crop(effective,box,True))
            atomic_png(task/'frames'/f'{j-left:06d}.png',crop(image,box))
            atomic_png(task/'masks'/f'{j-left:06d}.png',model_masks[-1])
        singleton=right-left==1
        if singleton:
            shutil.copy2(task/'frames/000000.png',task/'frames/000001.png')
            shutil.copy2(task/'masks/000000.png',task/'masks/000001.png')
        area=crop(before[0],box).shape[0]*crop(before[0],box).shape[1]
        length=min(frame_budget(area,cfg),max(2,right-left))
        row=dict(任务=counter,ROI=box,开始帧=left,结束帧=right,原生裁块尺寸=[native_w,native_h],
                 模型面积=area,subvideo_length=length,fp16=cfg.fp16,降分辨率=False,重试深度=depth,通过=False)
        records.append(row);begin=time.monotonic()
        args=[os.environ['INPAINT_PYTHON'],str(Path(__file__).with_name('torch_stage.py')),
              '--metrics',task/'metrics.json','--frames-only','--script',Path(os.environ['PROPAINTER_ROOT'])/'inference_propainter.py',
              '--video',task/'frames','--mask',task/'masks','--output',task/'out','--save_frames','--mask_dilation','0',
              '--subvideo_length',str(length),'--neighbor_length','10','--ref_stride','10']
        if cfg.fp16:args.append('--fp16')
        log(f'ROI任务={counter} 开始 帧数={right-left} 裁块={native_w}×{native_h} subvideo={length} fp16={cfg.fp16}')
        try:run(args,cwd=os.environ['PROPAINTER_ROOT'],timeout=900,stage='ROI-ProPainter')
        except PipelineError as exc:
            row.update(错误码=exc.code,耗时秒=time.monotonic()-begin)
            (work/'roi-report.json').write_text(json.dumps(dict(配置=asdict(cfg),任务=records),ensure_ascii=False,indent=2))
            if exc.code=='GPU_OOM' and depth<cfg.maximum_retries and end-start>cfg.minimum_frames:
                log(f'ROI任务={row["任务"]} 显存不足，保留失败记录，仅拆分该任务时间范围重试')
                middle=(start+end)//2;overlap=min(8,max(1,(end-start)//8))
                process(box,owned,start,min(end,middle+overlap),scene,depth+1)
                process(box,owned,max(start,middle-overlap),end,scene,depth+1)
                return
            raise
        predicted=sorted((task/'out/frames/frames').glob('*.png'))
        if len(predicted)!=(2 if singleton else right-left):
            raise PipelineError('ROI_INCOMPLETE','ROI模型输出帧数不完整，禁止原帧回退',details=row)
        refs=np.unique(np.linspace(scene[0],scene[1]-1,min(24,scene[1]-scene[0]),dtype=int))
        references=CleanReferences([crop(cv2.imread(str(files[j])),box) for j in refs],
                                   [crop(masks[j],box,True) for j in refs])
        overlap_max=0.;changed=0
        for offset,j in enumerate(range(left,right)):
            repaired=cv2.imread(str(predicted[offset]))
            if repaired is None or repaired.shape!=crop(before[offset],box).shape:
                raise PipelineError('OUTPUT_INVALID','ROI模型改变了裁块几何')
            done=cv2.imread(str(coverage/files[j].name),0);target=owned[j]&~done
            if np.count_nonzero(owned[j])/source_pixels>settings.mask_fraction_limit:
                raise PipelineError('MASK_UNCERTAIN','单次ROI施工超过20%保护')
            known=(done[b:d,a:c]>0)&(owned[j][b:d,a:c]>0)
            current=cv2.imread(str(dest/files[j].name));old=current[b:d,a:c]
            result=repaired[:native_h,:native_w]
            disagreement=float(np.abs(result.astype(float)-old.astype(float))[known].mean()) if known.any() else 0.
            overlap_max=max(overlap_max,disagreement)
            if disagreement>settings.overlap_mae_limit:raise PipelineError('SEAM_UNCERTAIN','ROI重叠上下文输出不一致',details=dict(row,差异=disagreement,上限=settings.overlap_mae_limit))
            if not np.any(target):continue
            local=crop(target,box,True)
            restored=references.restore(crop(current,box),local,repaired)
            painted=soft_composite(crop(current,box),restored,local)[:native_h,:native_w]
            active=target[b:d,a:c]>0;old[active]=painted[active];changed+=int(np.count_nonzero(active))
            atomic_png(dest/files[j].name,current);atomic_png(coverage/files[j].name,done|target)
        real_pixels+=changed;model_pixels+=area*(right-left)
        row.update(通过=True,耗时秒=time.monotonic()-begin,贴回像素=changed,重叠最大差异=overlap_max)
        if (task/'metrics.json').exists():row['GPU模型指标']=json.loads((task/'metrics.json').read_text())
        # 只清本任务已核验的临时裁块，保留任务报告及完整原生结果。
        for name in ['frames','masks','out']:shutil.rmtree(task/name)
        log(f'ROI任务={row["任务"]} 完成 耗时秒={row["耗时秒"]:.3f} 贴回像素={changed}')
        (work/'roi-report.json').write_text(json.dumps(dict(配置=asdict(cfg),任务=records),ensure_ascii=False,indent=2))
    for layer_number,layer in enumerate(layers):
        layer_masks=[cv2.imread(str(Path(layer['directory'])/p.name),0) for p in files]
        for scene in detected['scenes']:
            for start,end in windows(*scene,cfg.maximum_frames,cfg.overlap_frames):
                groups=[]
                for track in detected['tracks']:
                    footprint=np.zeros(shape,np.uint8)
                    for hint in track['hints']:
                        i=hint['frame']
                        if start<=i<end:
                            obj=cv2.imread(str(work/'validated-objects'/str(track['id'])/files[i].name),0)
                            if obj is None:raise PipelineError('MASK_INVALID','ROI缺少已验证对象掩膜')
                            footprint|=obj&layer_masks[i]
                    rect=bbox(footprint)
                    if rect:groups.append(dict(box=rect,tracks={track['id']}))
                for group in merge_groups(groups):
                    owned={i:np.zeros(shape,np.uint8) for i in range(start,end)}
                    for identity in group['tracks']:
                        for i in range(start,end):
                            p=work/'validated-objects'/str(identity)/files[i].name
                            if p.exists():owned[i]|=cv2.imread(str(p),0)&layer_masks[i]
                    box=padded_box(group['box'],shape,cfg)
                    safe=frame_budget((box[2]-box[0])*(box[3]-box[1]),cfg)
                    for a,b in windows(start,end,safe,min(cfg.overlap_frames,safe//4)):
                        # 上下文可能引用窗口以外帧，但永远没有窗口外施工目标。
                        local={i:owned.get(i,np.zeros(shape,np.uint8)) for i in range(max(scene[0],a-2),min(scene[1],b+2))}
                        process(box,local,a,b,scene)
    assert_coverage(work,files)
    summary=dict(配置=asdict(cfg),任务=records,任务数=len(records),实际图层数=len(layers),
        总耗时秒=time.monotonic()-started,施工像素=real_pixels,模型像素帧=model_pixels,
        原整帧六层像素帧=source_pixels*len(files)*6,原帧回退数=0,覆盖核验通过=True)
    (work/'roi-report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary
