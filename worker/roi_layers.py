"""原生ROI逐层补画：冻结前层输入，保持v7窗口、重叠混合及每层面积保护。"""
from dataclasses import asdict
from pathlib import Path
import json,os,shutil,time
import cv2
import numpy as np
from . import roi_inpaint as roi
from .hls_settings import HLSSettings
from .json_types import json_native
from .process import PipelineError,run
from .restoration import CleanReferences,soft_composite
from .vision import windows


def seam_variants(start,end,scene_start,scene_end,settings,cfg):
    """最多三次恢复：平移起点增加重叠、双向上下文、缩短模型内部子窗口。"""
    count=cfg.seam_retries
    if not 0<=count<=3:raise ValueError('接缝重试次数必须在0到3之间')
    step=settings.overlap_frames
    return [(start,end,settings.window_frames),
            (max(scene_start,start-step),end,settings.window_frames),
            (max(scene_start,start-2*step),min(scene_end,end+step),settings.window_frames),
            (max(scene_start,start-step),min(scene_end,end+step),max(2,settings.window_frames//2))][:count+1]


def execute(work,files,detected,log,cfg):
    settings=HLSSettings.from_env();started=time.monotonic()
    shape=cv2.imread(str(files[0])).shape[:2];pixels=shape[0]*shape[1]
    layers=json.loads((work/'layers.json').read_text())['layers']
    coverage=work/'roi-coverage';coverage.mkdir()
    for p in files:roi.atomic_png(coverage/p.name,np.zeros(shape,np.uint8))
    current=files;records=[];layer_records=[]
    def save():
        summary=dict(配置=asdict(cfg),任务=records,图层=layer_records,实际图层数=len(layers),
                     窗口=settings.window_frames,重叠=settings.overlap_frames,
                     总耗时秒=time.monotonic()-started,原帧回退数=0)
        (work/'roi-report.json').write_text(json.dumps(json_native(summary),ensure_ascii=False,indent=2))
        return summary
    for k,layer in enumerate(layers):
        begin=time.monotonic();dest=work/f'roi-layer-{k:02d}';dest.mkdir()
        masks=[cv2.imread(str(Path(layer['directory'])/p.name),0) for p in files]
        for i,m in enumerate(masks):
            if m is None or m.shape!=shape or not set(np.unique(m))<={0,255}:
                raise PipelineError('MASK_INVALID','逐层ROI掩膜缺失或几何无效')
            if np.count_nonzero(m)/pixels>settings.mask_fraction_limit:
                raise PipelineError('MASK_UNCERTAIN','逐层ROI单遍超过20%保护',details=dict(layer=k,frame=i))
        # 每层独立输出；本层任何窗口都只读取上一层，不能把当前层施工反馈进模型。
        for p,source in zip(files,current):os.link(source,dest/p.name)
        completed=set();log(f'逐层ROI 图层={k+1}/{len(layers)} 开始')
        for scene_start,scene_end in detected['scenes']:
            refs=np.unique(np.linspace(scene_start,scene_end-1,min(24,scene_end-scene_start),dtype=int))
            for start,end in windows(scene_start,scene_end,settings.window_frames,settings.overlap_frames):
                footprint=np.bitwise_or.reduce(masks[start:end]);rect=roi.bbox(footprint)
                if rect is None:continue
                box=roi.padded_box(rect,shape,cfg);a,b,c,d=box
                row=dict(任务=len(records)+1,图层=k,开始帧=start,结束帧=end,ROI=box,fp16=cfg.fp16,
                         输入图层=k-1,窗口=settings.window_frames,重叠=settings.overlap_frames,通过=False,尝试=[])
                records.append(row);tick=time.monotonic();save()
                variants=seam_variants(start,end,scene_start,scene_end,settings,cfg)
                oom_used=False
                for attempt,(left,right,length) in enumerate(variants):
                    task=work/f'roi-{len(records):04d}-attempt-{attempt:02d}'
                    task.mkdir();(task/'frames').mkdir();(task/'masks').mkdir()
                    trial=dict(尝试=attempt,开始帧=left,结束帧=right,subvideo_length=length,
                               前向重叠帧数=sum(j in completed for j in range(left,start+settings.overlap_frames)),
                               通过=False,重叠最大差异=None)
                    row['尝试'].append(trial);attempt_tick=time.monotonic();save()
                    log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 输入帧={left}:{right} 施工帧={start}:{end} subvideo={length} 裁块={c-a}×{d-b}')
                    try:
                        for j in range(left,right):
                            roi.atomic_png(task/'frames'/f'{j-left:06d}.png',roi.crop(cv2.imread(str(current[j])),box))
                            roi.atomic_png(task/'masks'/f'{j-left:06d}.png',roi.crop(masks[j],box,True))
                        singleton=right-left==1
                        if singleton:
                            for name in ['frames','masks']:shutil.copy2(task/name/'000000.png',task/name/'000001.png')
                        args=[os.environ['INPAINT_PYTHON'],str(Path(__file__).with_name('torch_stage.py')),
                              '--metrics',task/'metrics.json','--frames-only','--script',Path(os.environ['PROPAINTER_ROOT'])/'inference_propainter.py',
                              '--video',task/'frames','--mask',task/'masks','--output',task/'out','--save_frames',
                              '--mask_dilation','0','--subvideo_length',str(length),'--neighbor_length','6','--ref_stride','20']
                        if cfg.fp16:args.append('--fp16')
                        # 按实际像素帧工作量扩展保护，不限制整块总预算。
                        timeout=max(900,900*(right-left)/settings.window_frames)
                        trial['超时秒']=timeout
                        run(args,cwd=os.environ['PROPAINTER_ROOT'],timeout=timeout,stage='逐层ROI-ProPainter')
                        predicted=sorted((task/'out/frames/frames').glob('*.png'))
                        if len(predicted)!=(2 if singleton else right-left):raise PipelineError('ROI_INCOMPLETE','逐层ROI输出帧数缺失，禁止原帧回退')
                        references=CleanReferences([roi.crop(cv2.imread(str(current[j])),box) for j in refs],
                                                   [roi.crop(cv2.imread(str(work/'masks'/files[j].name),0),box,True) for j in refs])
                        maximum=0.;staged=task/'staged';staged.mkdir()
                        # 先完整核验，再提交；失败尝试不得修改覆盖记录或已通过窗口。
                        for j in range(left,right):
                            original=roi.crop(cv2.imread(str(current[j])),box);mask=roi.crop(masks[j],box,True)
                            repair=cv2.imread(str(predicted[j-left]))
                            if repair is None or repair.shape!=original.shape:raise PipelineError('OUTPUT_INVALID','逐层ROI模型改变几何')
                            painted=soft_composite(original,references.restore(original,mask,repair),mask)[:d-b,:c-a]
                            image=cv2.imread(str(dest/files[j].name));old=image[b:d,a:c];active=masks[j][b:d,a:c]>0
                            if j in completed:
                                disagreement=float(np.abs(old.astype(float)-painted.astype(float))[active].mean()) if active.any() else 0.
                                maximum=max(maximum,disagreement)
                            if start<=j<end:
                                if j in completed:
                                    overlap=sum(t in completed for t in range(start,end))
                                    weight=(j-start+1)/(overlap+1)
                                    painted=cv2.addWeighted(painted,weight,old,1-weight,0)
                                old[active]=painted[active];roi.atomic_png(staged/files[j].name,image)
                        trial['重叠最大差异']=maximum
                        if maximum>settings.overlap_mae_limit:
                            raise PipelineError('SEAM_UNCERTAIN','逐层ROI重叠差异超过原有保护',details=dict(差异=maximum,上限=settings.overlap_mae_limit))
                        for j in range(start,end):
                            (staged/files[j].name).replace(dest/files[j].name)
                            done=cv2.imread(str(coverage/files[j].name),0);roi.atomic_png(coverage/files[j].name,done|masks[j])
                            completed.add(j)
                        trial['通过']=True;row.update(通过=True,重叠最大差异=maximum)
                    except Exception as exc:
                        trial['错误码']=getattr(exc,'code',type(exc).__name__)
                        retry=trial['错误码']=='SEAM_UNCERTAIN'
                        if trial['错误码']=='GPU_OOM' and not oom_used:
                            oom_used=True;retry=True
                            if attempt+1<len(variants):
                                variants[attempt+1]=(left,right,max(2,length//2))
                        if not retry or attempt==len(variants)-1:
                            row.update(错误码=trial['错误码'],耗时秒=time.monotonic()-tick)
                            raise
                    finally:
                        trial['耗时秒']=time.monotonic()-attempt_tick;row['耗时秒']=time.monotonic()-tick;save()
                        log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 参数={left}:{right}/{length} 接缝差异={trial["重叠最大差异"]} 上限={settings.overlap_mae_limit} 通过={trial["通过"]} 错误码={trial.get("错误码", "无")} 耗时秒={trial["耗时秒"]:.3f}')
                    if trial['通过']:
                        for name in ['frames','masks','out','staged']:shutil.rmtree(task/name)
                        break
                log(f'逐层ROI 任务={row["任务"]} 完成 耗时秒={row["耗时秒"]:.3f}')
        layer_records.append(dict(图层=k,耗时秒=time.monotonic()-begin,活动帧数=sum(bool(m.any()) for m in masks)))
        current=[dest/p.name for p in files];save()
        log(f'逐层ROI 图层={k+1} 完成 耗时秒={time.monotonic()-begin:.3f}')
    (work/'composite').symlink_to(current[0].parent,target_is_directory=True)
    roi.assert_coverage(work,files)
    summary=save();summary.update(覆盖核验通过=True,任务数=len(records))
    (work/'roi-report.json').write_text(json.dumps(json_native(summary),ensure_ascii=False,indent=2))
    return summary
