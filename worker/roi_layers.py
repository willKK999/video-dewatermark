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
                task=work/f'roi-{len(records)+1:04d}';task.mkdir();(task/'frames').mkdir();(task/'masks').mkdir()
                row=dict(任务=len(records)+1,图层=k,开始帧=start,结束帧=end,ROI=box,fp16=cfg.fp16,
                         输入图层=k-1,窗口=settings.window_frames,重叠=settings.overlap_frames,通过=False)
                records.append(row);tick=time.monotonic();save()
                for j in range(start,end):
                    roi.atomic_png(task/'frames'/f'{j-start:06d}.png',roi.crop(cv2.imread(str(current[j])),box))
                    roi.atomic_png(task/'masks'/f'{j-start:06d}.png',roi.crop(masks[j],box,True))
                singleton=end-start==1
                if singleton:
                    for name in ['frames','masks']:shutil.copy2(task/name/'000000.png',task/name/'000001.png')
                args=[os.environ['INPAINT_PYTHON'],str(Path(__file__).with_name('torch_stage.py')),
                      '--metrics',task/'metrics.json','--frames-only','--script',Path(os.environ['PROPAINTER_ROOT'])/'inference_propainter.py',
                      '--video',task/'frames','--mask',task/'masks','--output',task/'out','--save_frames',
                      '--mask_dilation','0','--subvideo_length',str(settings.window_frames),'--neighbor_length','6','--ref_stride','20']
                if cfg.fp16:args.append('--fp16')
                log(f'逐层ROI 任务={row["任务"]} 图层={k+1} 帧={start}:{end} 裁块={c-a}×{d-b}')
                try:
                    run(args,cwd=os.environ['PROPAINTER_ROOT'],timeout=900,stage='逐层ROI-ProPainter')
                    predicted=sorted((task/'out/frames/frames').glob('*.png'))
                    if len(predicted)!=(2 if singleton else end-start):raise PipelineError('ROI_INCOMPLETE','逐层ROI输出帧数缺失，禁止原帧回退')
                    references=CleanReferences([roi.crop(cv2.imread(str(current[j])),box) for j in refs],
                                               [roi.crop(masks[j],box,True) for j in refs])
                    maximum=0.
                    for offset,j in enumerate(range(start,end)):
                        original=roi.crop(cv2.imread(str(current[j])),box);mask=roi.crop(masks[j],box,True)
                        repair=cv2.imread(str(predicted[offset]))
                        if repair is None or repair.shape!=original.shape:raise PipelineError('OUTPUT_INVALID','逐层ROI模型改变几何')
                        painted=soft_composite(original,references.restore(original,mask,repair),mask)[:d-b,:c-a]
                        image=cv2.imread(str(dest/files[j].name));old=image[b:d,a:c];active=masks[j][b:d,a:c]>0
                        if j in completed:
                            disagreement=float(np.abs(old.astype(float)-painted.astype(float))[active].mean()) if active.any() else 0.
                            maximum=max(maximum,disagreement)
                            if disagreement>settings.overlap_mae_limit:raise PipelineError('SEAM_UNCERTAIN','逐层ROI重叠差异超过原有保护',details=dict(row,差异=disagreement))
                            weight=(j-start+1)/(settings.overlap_frames+1)
                            painted=cv2.addWeighted(painted,weight,old,1-weight,0)
                        old[active]=painted[active];roi.atomic_png(dest/files[j].name,image)
                        done=cv2.imread(str(coverage/files[j].name),0);roi.atomic_png(coverage/files[j].name,done|masks[j])
                        completed.add(j)
                    row.update(通过=True,重叠最大差异=maximum)
                except Exception as exc:
                    row.update(错误码=getattr(exc,'code',type(exc).__name__),耗时秒=time.monotonic()-tick);save();raise
                row['耗时秒']=time.monotonic()-tick;save()
                for name in ['frames','masks','out']:shutil.rmtree(task/name)
                log(f'逐层ROI 任务={row["任务"]} 完成 耗时秒={row["耗时秒"]:.3f}')
        layer_records.append(dict(图层=k,耗时秒=time.monotonic()-begin,活动帧数=sum(bool(m.any()) for m in masks)))
        current=[dest/p.name for p in files];save()
        log(f'逐层ROI 图层={k+1} 完成 耗时秒={time.monotonic()-begin:.3f}')
    (work/'composite').symlink_to(current[0].parent,target_is_directory=True)
    roi.assert_coverage(work,files)
    summary=save();summary.update(覆盖核验通过=True,任务数=len(records))
    (work/'roi-report.json').write_text(json.dumps(json_native(summary),ensure_ascii=False,indent=2))
    return summary
