"""空间连通ROI：冻结前层、独立施工域和上下文接缝核验。"""
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
from .spatial_roi import SpatialSettings,plan,padded_area
from .roi_attempt_cache import compact,restore


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
    spatial=SpatialSettings(**json.loads((work/'spatial-roi.json').read_text()))
    policy_path=work/'inpaint-policy.json'
    policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
    continue_unpassed=policy.get('continue_unpassed',False)
    shape=cv2.imread(str(files[0])).shape[:2];pixels=shape[0]*shape[1]
    trusted_path=work/'trusted-context.json';trusted=None;input_audits=[]
    if trusted_path.exists():
        from .trusted_context import ContextSettings,verify_input,exclusion_mask
        trusted=ContextSettings(**json.loads(trusted_path.read_text()))
    def save_input_audit():
        if trusted is None:return
        (work.parent/'层输入自检.json').write_text(json.dumps(json_native(dict(
            口径='所有冻结确认水印并集与实际写入模型掩膜逐像素核对，覆盖重试和上下文帧；不是未检测水印的真值召回率',
            模型排除区='全部确认水印并集',回写域='当前层与当前ROI施工域交集',
            源水印未排除像素=sum(f.get('源水印未排除像素',0) for a in input_audits for f in a['帧']),
            全部已检查尝试通过=all(a['通过'] for a in input_audits),尝试=input_audits)),ensure_ascii=False,indent=2))
    save_input_audit()
    layers=json.loads((work/'layers.json').read_text())['layers']
    animation_path=work/'animation-events.json'
    animation_plan=json.loads(animation_path.read_text()) if animation_path.exists() else None
    if animation_plan:
        from .watermark_groups import AnimationSettings,planned_windows
        animation_cfg=AnimationSettings(**animation_plan['settings'])
        if animation_cfg.maximum_window_frames>cfg.maximum_frames:raise ValueError('动画窗口不得超过既有ROI最大帧数')
    coverage=work/'roi-coverage';coverage.mkdir()
    for p in files:roi.atomic_png(coverage/p.name,np.zeros(shape,np.uint8))
    current=files;records=[];layer_records=[];unpassed=[];unfillable=[]
    def save():
        summary=dict(空间配置=asdict(spatial),配置=asdict(cfg),任务=records,未过块=unpassed,不可填块=unfillable,全部ROI通过=len(layer_records)==len(layers) and all(r['通过'] for r in records),图层=layer_records,实际图层数=len(layers),
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
        layer_done={};log(f'逐层ROI 图层={k+1}/{len(layers)} 开始')
        for scene_start,scene_end in detected['scenes']:
            refs=np.unique(np.linspace(scene_start,scene_end-1,min(24,scene_end-scene_start),dtype=int))
            schedule=list(windows(scene_start,scene_end,settings.window_frames,settings.overlap_frames))
            if animation_plan:
                events=[e['interval'] for e in animation_plan['layers'].get(str(k),[]) if scene_start<=e['interval'][0]<scene_end]
                schedule=planned_windows(scene_start,scene_end,settings.window_frames,settings.overlap_frames,events,animation_cfg)
            for start,end in schedule:
                footprint=np.bitwise_or.reduce(masks[start:end])
                jobs=plan(footprint,end-start+settings.overlap_frames*(3 if continue_unpassed else 1),cfg,spatial)
                for fragment in jobs:
                    ownership=fragment['ownership'];box=fragment['box'];a,b,c,d=box
                    completed={j for j in range(scene_start,scene_end) if j in layer_done and np.any(layer_done[j]&ownership)}
                    row=dict(任务=len(records)+1,图层=k,开始帧=start,结束帧=end,ROI=box,fp16=cfg.fp16,
                             施工面积像素=int(np.count_nonzero(ownership)),裁块像素=padded_area(box),像素帧预算=spatial.maximum_pixel_frames,输入图层=k-1,窗口=settings.window_frames,重叠=settings.overlap_frames,通过=False,尝试=[])
                    records.append(row);tick=time.monotonic();save()
                    baseline=work/f'roi-{len(records):04d}-baseline'
                    variants=seam_variants(start,end,scene_start,scene_end,settings,cfg)
                    oom_used=False
                    for attempt,(left,right,length) in enumerate(variants):
                        task=work/f'roi-{len(records):04d}-attempt-{attempt:02d}'
                        task.mkdir();(task/'frames').mkdir();(task/'masks').mkdir()
                        trial=dict(尝试=attempt,开始帧=left,结束帧=right,subvideo_length=length,
                                   前向重叠帧数=sum(j in completed for j in range(left,start+settings.overlap_frames)),
                                   通过=False,重叠最大差异=None)
                        row['尝试'].append(trial);attempt_tick=time.monotonic();save()
                        audit=None
                        if trusted:
                            audit=dict(任务=row['任务'],图层=k,尝试=attempt,输入区间=[left,right],ROI=box,通过=False,帧=[]);input_audits.append(audit)
                        log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 输入帧={left}:{right} 施工帧={start}:{end} subvideo={length} 裁块={c-a}×{d-b}')
                        try:
                            if padded_area(box)*(right-left)>spatial.maximum_pixel_frames:raise PipelineError('GPU_BUDGET_EXCEEDED','空间裁块超过配置像素帧预算')
                            for j in range(left,right):
                                roi.atomic_png(task/'frames'/f'{j-left:06d}.png',roi.crop(cv2.imread(str(current[j])),box))
                                exclusion=exclusion_mask(cv2.imread(str(work/'masks'/files[j].name),0),trusted) if trusted else masks[j]
                                if trusted and np.count_nonzero(exclusion)/pixels>trusted.mask_fraction_limit:
                                    raise PipelineError('MASK_UNCERTAIN','全确认水印排除区超过20%，必须重规划')
                                model_mask=roi.crop(exclusion,box,True);mask_path=task/'masks'/f'{j-left:06d}.png';roi.atomic_png(mask_path,model_mask)
                                if trusted:
                                    try:checked=verify_input(model_mask,cv2.imread(str(mask_path),0),roi.crop(masks[j]&ownership,box,True),pixels,trusted)
                                    except PipelineError as exc:
                                        audit['帧'].append(dict(frame=j,**exc.details));audit['错误码']=exc.code;save_input_audit();raise
                                    import hashlib
                                    audit['帧'].append(dict(frame=j,context=not(start<=j<end),模型掩膜sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),**checked))
                            if trusted:audit['通过']=True;save_input_audit();log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 层输入自检通过，未排除像素=0，检查帧数={right-left}')
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
                            if policy.get('model_timeout_seconds'):timeout=min(timeout,policy['model_timeout_seconds'])
                            trial['超时秒']=timeout
                            heartbeat=[time.monotonic()]
                            def check_model():
                                if policy.get('model_timeout_seconds') and time.monotonic()-heartbeat[0]>=30:
                                    log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 模型进程仍存活，正在计算')
                                    heartbeat[0]=time.monotonic()
                            run(args,check=check_model,cwd=os.environ['PROPAINTER_ROOT'],timeout=timeout,stage='逐层ROI-ProPainter')
                            predicted=sorted((task/'out/frames/frames').glob('*.png'))
                            if len(predicted)!=(2 if singleton else right-left):raise PipelineError('ROI_INCOMPLETE','逐层ROI输出帧数缺失，禁止原帧回退')
                            references=CleanReferences([roi.crop(cv2.imread(str(current[j])),box) for j in refs],
                                                       [roi.crop(cv2.imread(str(work/'masks'/files[j].name),0),box,True) for j in refs])
                            maximum=0.;staged=task/'staged';staged.mkdir();frame_scores=[]
                            sample_frames=set();stage_frames=set()
                            if trusted:
                                active_frames=[j for j in range(start,end) if np.any(masks[j]&ownership)]
                                if active_frames:
                                    sample_frames={active_frames[q] for q in np.linspace(0,len(active_frames)-1,min(trusted.model_sample_count,len(active_frames)),dtype=int)}
                                    stage_frames=set(sorted(sample_frames)[:trusted.retain_stage_sample_count])
                            if continue_unpassed:(task/'seam-difference').mkdir();(task/'seam-shared').mkdir()
                            # 先完整核验，再提交；失败尝试不得修改覆盖记录或已通过窗口。
                            for j in range(left,right):
                                original=roi.crop(cv2.imread(str(current[j])),box);mask=roi.crop(masks[j],box,True)
                                repair=cv2.imread(str(predicted[j-left]))
                                if repair is None or repair.shape!=original.shape:raise PipelineError('OUTPUT_INVALID','逐层ROI模型改变几何')
                                restored=references.restore(original,mask,repair)
                                painted=soft_composite(original,restored,mask)[:d-b,:c-a]
                                if j in sample_frames:
                                    sample=work/'model-samples'/f'task-{row["任务"]:04d}'/f'attempt-{attempt:02d}';sample.mkdir(parents=True,exist_ok=True)
                                    roi.atomic_png(sample/f'{j:06d}-raw-model.png',repair)
                                    shutil.copy2(task/'masks'/f'{j-left:06d}.png',sample/f'{j:06d}-exclusion.png')
                                    roi.atomic_png(sample/f'{j:06d}-construction.png',roi.crop(masks[j]&ownership,box,True))
                                    if j in stage_frames:
                                        for label,value in [('source',original),('restored',restored),('composited',painted)]:roi.atomic_png(sample/f'{j:06d}-{label}.png',value)
                                    (sample/'metadata.json').write_text(json.dumps(dict(task=row['任务'],attempt=attempt,roi=box,input=[left,right],construction=[start,end],frames=sorted(sample_frames),说明='raw-model为模型原始预测；restored为参考回填后；composited为羽化后、窗口混合与施工域回写前。'),ensure_ascii=False,indent=2))
                                image=cv2.imread(str(dest/files[j].name));old=image[b:d,a:c];active=(masks[j][b:d,a:c]>0)&(ownership[b:d,a:c]>0)
                                prior_done=layer_done.get(j)
                                shared=(prior_done[b:d,a:c]>0)&(masks[j][b:d,a:c]>0) if prior_done is not None else np.zeros(active.shape,bool)
                                if shared.any():
                                    disagreement=float(np.abs(old.astype(float)-painted.astype(float))[shared].mean())
                                    maximum=max(maximum,disagreement)
                                if start<=j<end:
                                    if continue_unpassed:
                                        difference=np.where(shared,np.abs(old.astype(float)-painted.astype(float)).mean(axis=2),0.)
                                        roi.atomic_png(task/'seam-difference'/files[j].name,np.rint(difference*256).astype(np.uint16))
                                        roi.atomic_png(task/'seam-shared'/files[j].name,shared.astype(np.uint8)*255)
                                        frame_scores.append(dict(frame=j,mae=float(difference[shared].mean()) if shared.any() else 0.,pixels=int(np.count_nonzero(shared))))
                                    if j in completed:
                                        overlap=sum(t in completed for t in range(start,end))
                                        weight=(j-start+1)/(overlap+1)
                                        mixed=cv2.addWeighted(painted,weight,old,1-weight,0)
                                        painted[active&shared]=mixed[active&shared]
                                    old[active]=painted[active];roi.atomic_png(staged/files[j].name,image)
                            trial['重叠最大差异']=maximum
                            if continue_unpassed:
                                trial['施工帧差异']=frame_scores;trial['差图尺度']=256
                                trial['代表施工帧']=max(frame_scores,key=lambda p:p['mae'])['frame']
                            if maximum>settings.overlap_mae_limit:
                                raise PipelineError('SEAM_UNCERTAIN','逐层ROI重叠差异超过原有保护',details=dict(差异=maximum,上限=settings.overlap_mae_limit))
                            for j in range(start,end):
                                (staged/files[j].name).replace(dest/files[j].name)
                                done=cv2.imread(str(coverage/files[j].name),0);roi.atomic_png(coverage/files[j].name,done|(masks[j]&ownership))
                                layer_done.setdefault(j,np.zeros(shape,np.uint8))[:] |= masks[j]&ownership
                            trial['通过']=True;row.update(通过=True,重叠最大差异=maximum)
                        except Exception as exc:
                            trial['错误码']=getattr(exc,'code',type(exc).__name__)
                            if continue_unpassed and trial['重叠最大差异'] is not None:
                                trial['完整暂存']=compact(task,baseline,dest,files,masks,ownership,start,end)
                                trial['暂存路径']=str(task/'staged-delta');trial['基线路径']=str(baseline)
                            retry=trial['错误码']=='SEAM_UNCERTAIN'
                            if trial['错误码']=='GPU_OOM' and not oom_used:
                                oom_used=True;retry=True
                                if attempt+1<len(variants):
                                    variants[attempt+1]=(left,right,max(2,length//2))
                            if not retry or attempt==len(variants)-1:
                                row.update(错误码=trial['错误码'],耗时秒=time.monotonic()-tick)
                                if continue_unpassed:break
                                raise
                        finally:
                            save_input_audit()
                            trial['耗时秒']=time.monotonic()-attempt_tick;row['耗时秒']=time.monotonic()-tick;save()
                            log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 参数={left}:{right}/{length} 接缝差异={trial["重叠最大差异"]} 上限={settings.overlap_mae_limit} 通过={trial["通过"]} 错误码={trial.get("错误码", "无")} 耗时秒={trial["耗时秒"]:.3f}')
                            if continue_unpassed and not trial['通过']:
                                for name in ['frames','masks','out']:
                                    if (task/name).exists():shutil.rmtree(task/name)
                        if trial['通过']:
                            for name in ['frames','masks','out','staged']:shutil.rmtree(task/name)
                            break
                    if not row['通过'] and continue_unpassed:
                        candidates=[p for p in row['尝试'] if p.get('完整暂存') and p.get('重叠最大差异') is not None]
                        unpassed.append(row['任务']);row['未过块']=True
                        if candidates:
                            best=min(candidates,key=lambda p:p['重叠最大差异']);best_task=work/f"roi-{row['任务']:04d}-attempt-{best['尝试']:02d}"
                            for j in range(start,end):
                                roi.atomic_png(dest/files[j].name,restore(best_task,baseline,files[j].name))
                                done=cv2.imread(str(coverage/files[j].name),0);roi.atomic_png(coverage/files[j].name,done|(masks[j]&ownership))
                                layer_done.setdefault(j,np.zeros(shape,np.uint8))[:] |= masks[j]&ownership
                            row.update(最佳尝试=best['尝试'],最佳接缝=best['重叠最大差异'],采用未过结果=True,代表施工帧=best['代表施工帧'])
                            log(f'逐层ROI 任务={row["任务"]} 未过块=True 采用最佳尝试={best["尝试"]} MAE={best["重叠最大差异"]}；继续其余任务，不标通过')
                        else:
                            unfillable.append(row['任务']);row['采用未过结果']=False
                            log(f'逐层ROI 任务={row["任务"]} 未过块=True 无完整模型候选；继续其余任务，禁止原帧伪造成片')
                        save()
                    log(f'逐层ROI 任务={row["任务"]} 完成 耗时秒={row["耗时秒"]:.3f}')
        layer_records.append(dict(图层=k,耗时秒=time.monotonic()-begin,活动帧数=sum(bool(m.any()) for m in masks)))
        current=[dest/p.name for p in files];save()
        log(f'逐层ROI 图层={k+1} 完成 耗时秒={time.monotonic()-begin:.3f}')
    (work/'composite').symlink_to(current[0].parent,target_is_directory=True)
    if unfillable:
        save();raise PipelineError('ROI_UNFILLABLE','其余任务已执行，但部分未过块无完整模型候选，禁止原帧回退',details=dict(tasks=unfillable))
    roi.assert_coverage(work,files)
    summary=save();summary.update(覆盖核验通过=True,任务数=len(records))
    (work/'roi-report.json').write_text(json.dumps(json_native(summary),ensure_ascii=False,indent=2))
    return summary
