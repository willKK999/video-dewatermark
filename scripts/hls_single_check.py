"""通用单段自动模型验证：只运行指定主体和上下文，失败不回退原帧。"""
import argparse,json,os,time,subprocess,hashlib
from pathlib import Path
from fractions import Fraction
import cv2,numpy as np
from worker import media,vision
from worker.process import PipelineError,run
from worker.model_tracks import build,patch,candidates_for_frame,ModelTrackSettings,learn_text_prototypes
from worker.stage_budget import sam_budget
from worker.hls_layers import VerifiedLayers
from worker.hls_settings import HLSSettings,is_residual_score
from worker.multitrack import highpass,component_scores
from worker.low_frequency import band
from worker.hls_acceptance import clarity_summary
from worker.matte import temporal_detail
from scripts.hls_chunk_check import audio_hash


def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
def read(path):return json.loads(path.read_text()) if path.exists() else {}


def prepare(source,work,expected,duration,context):
    if vision.sha256(source)!=expected:raise PipelineError('SOURCE_CHANGED','源摘要不匹配，禁止处理')
    metadata=media.probe(source);video=next(s for s in metadata['streams'] if s['codec_type']=='video')
    fps=float(Fraction(video['avg_frame_rate']));cfg=HLSSettings.from_env()
    if fps>cfg.model_fps:raise PipelineError('UNSUPPORTED_FRAME_RATE','本单段入口不对高帧率源片静默降帧，请使用原有原生合成路径')
    count=int(video.get('nb_frames') or round(float(video['duration'])*fps))
    core=min(count,round(duration*fps));total=min(count,round((duration+context)*fps))
    signature=dict(source_sha256=expected,first_frame=0,frames=total,fps=fps,width=video['width'],height=video['height'])
    binding=work/'frames-binding.json'
    if binding.exists():
        saved=read(binding)
        if saved.get('source')!=signature:raise PipelineError('SOURCE_CHANGED','帧缓存与输入或范围不一致')
        for name,h in saved['sha256'].items():
            if vision.sha256(work/'frames'/name)!=h:raise PipelineError('SOURCE_CHANGED','帧缓存摘要不一致')
    else:
        if (work/'frames').exists():raise PipelineError('SOURCE_CHANGED','已有帧缺少内容绑定，不允许推测复用')
        (work/'frames').mkdir()
        run(['ffmpeg','-v','error','-y','-threads','2','-i',source,'-an','-vf',f'trim=start_frame=0:end_frame={total},setpts=PTS-STARTPTS','-fps_mode','passthrough','-start_number','0','-threads','2',work/'frames/%06d.png'])
        write(binding,dict(source=signature,sha256={p.name:vision.sha256(p) for p in sorted((work/'frames').glob('*.png'))}))
    files=sorted((work/'frames').glob('*.png'))
    if len(files)!=total:raise PipelineError('OUTPUT_INCOMPLETE','输入帧数不完整')
    return metadata,video,fps,core,files


def detect_all(frames,output,enhance=False,tiles=False,flips=False):
    run([os.environ['DETECT_PYTHON'],'-m','scripts.hls_candidates','--frames',frames,'--output',output,
         *(['--enhance'] if enhance else []),*(['--tiles'] if tiles else []),*(['--flips'] if flips else [])],timeout=1800)
    return read(output)


def tracked_brand_candidates(image,row,prototypes,labels,allowed):
    """源片与结果使用同一品牌判据和自动目标包络；原始YOLO输出另行保留。"""
    from worker.model_tracks import intersection_fraction
    items=candidates_for_frame(image,row,prototypes,labels,ModelTrackSettings())
    return [p for p in items if p['kind']=='prototype' or any(
        intersection_fraction(p['model_box'],box)>.5 for box in allowed)]


def verify(work,video,files,detected,prior,labels,core,fps,output,enhance=False,tiles=False,flips=False):
    start=time.monotonic();encoded=work/'encoded';encoded.mkdir()
    run(['ffmpeg','-v','error','-y','-threads','2','-i',video,'-map','0:v:0','-fps_mode','passthrough','-start_number','0','-threads','2',encoded/'%06d.png'])
    decoded=sorted(encoded.glob('*.png'))
    if len(decoded)!=core:raise PipelineError('OUTPUT_INCOMPLETE','编码后帧数不一致')
    rows=detect_all(encoded,work/'encoded-yolo.json',enhance,tiles,flips)
    source_rows=read(work/'all-yolo.json')
    proto=learn_text_prototypes(files,source_rows,prior,labels,ModelTrackSettings(),fps,detected['scenes'])
    brand_candidates=[];v0_candidates=[];unconfirmed=[]
    # 先保存全部YOLO候选；品牌判别与源片使用同一内容绑定模板，字幕不直接算水印。
    for i,p in enumerate(decoded):
        image=cv2.imread(str(p));source=cv2.imread(str(files[i]))
        allowed=[h['box'] for h in detected['hints'] if h['frame']==i]
        items=tracked_brand_candidates(image,rows[i],proto,labels,allowed)
        all_items=candidates_for_frame(image,rows[i],proto,labels,ModelTrackSettings())
        unconfirmed.extend(dict(frame=i,**p) for p in all_items if p['kind']=='logo' and p not in items)
        original=tracked_brand_candidates(source,source_rows[i],proto,labels,allowed)
        brand_candidates.extend(dict(frame=i,**p) for p in items)
        v0_candidates.extend(dict(frame=i,**p) for p in original)
    template_records=[];template_candidates=[]
    for track in detected['tracks']:
        data=np.load(track['template'],allow_pickle=False)
        hp=[];low=[];src_low=[]
        for point in track['hints']:
            i=point['frame']
            if i>=core:continue
            size=(data['feature'].shape[1],data['feature'].shape[0])
            y=patch(cv2.imread(str(decoded[i])),point['box'],size)
            x=patch(cv2.imread(str(files[i])),point['box'],size)
            hp.append(highpass(y));low.append(band(y));src_low.append(band(x))
        if not hp:continue
        for kind,a,b,support in [('高通',data['feature'],np.median(np.stack(hp),axis=0),data['mask']),
                                 ('低频',np.median(np.stack(src_low),axis=0),np.median(np.stack(low),axis=0),np.ones_like(data['mask']))]:
            scores=component_scores(a,b,support)
            row=dict(轨迹=track['id'],类型=kind,全部部件=scores,候选=any(is_residual_score(p) for p in scores))
            template_records.append(row)
            if row['候选']:template_candidates.append(row)
    hints={}
    for p in detected['hints']:hints.setdefault(p['frame'],[]).append(p)
    cuts={a for a,b in detected['scenes']};times=[];flashes=[];clarity=[];old=None
    for i,path in enumerate(decoded):
        x=cv2.imread(str(files[i]));y=cv2.imread(str(path));mask=cv2.imread(str(work/'masks'/files[i].name),0)
        if old is not None:
            ox,oy,om=old;ds=np.abs(x.astype(np.float32)-ox);dy=np.abs(y.astype(np.float32)-oy)
            dr=np.abs((y.astype(np.float32)-x)-(oy.astype(np.float32)-ox))
            for p in hints.get(i,[]):
                a,b,c,d=map(round,p['box']);a,b,c,d=max(0,a),max(0,b),min(x.shape[1],c),min(x.shape[0],d)
                if c<=a or d<=b:continue
                row=dict(frame=i,time=i/fps,track_id=p['track_id'],box=p['box'],source_scene_cut=i in cuts,
                         source_mae=float(ds[b:d,a:c].mean()),result_mae=float(dy[b:d,a:c].mean()),repair_delta_mae=float(dr[b:d,a:c].mean()))
                times.append(row)
                if i not in cuts and row['result_mae']>max(row['source_mae']+6,row['source_mae']*1.8) and row['repair_delta_mae']>12:flashes.append(row)
            if i%max(1,round(fps)) in (0,1):
                metric=temporal_detail(ox,x,oy,y,mask&om,mask|om);metric.pop('passed',None);metric.pop('threshold',None)
                clarity.append(dict(frame=i,time=i/fps,**metric))
        old=x,y,mask
    quality=dict(视频摘要=vision.sha256(video),逐帧YOLO品牌候选=brand_candidates,
                 原片同口径品牌候选数=len(v0_candidates),编码后品牌候选数=len(brand_candidates),
                 未确认图形候选=unconfirmed,未确认图形候选数=len(unconfirmed),
                 全部模板部件=template_records,模板候选=template_candidates,
                 闪烁候选=flashes,时域全部样本=times,清晰度=clarity_summary(clarity),
                 残留候选数=len(brand_candidates)+len(template_candidates)+len(unconfirmed),
                 原始YOLO输出=str(work/'encoded-yolo.json'),
                 耗时秒=time.monotonic()-start,通过=False,目检完成=False)
    write(output/'quality.json',quality)
    return quality


def execute_one(args):
    started=time.monotonic();work=args.work.resolve();output=args.output.resolve()
    work.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    phase='准备';timings={};events=[]
    def log(message):
        line=time.strftime('%Y-%m-%dT%H:%M:%S%z')+' '+message
        with (output/'progress.log').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
    def step(name,callback):
        nonlocal phase
        phase=name;begin=time.monotonic();log(f'第1块 第{args.iteration}轮 阶段开始={name}')
        try:return callback()
        finally:
            timings[name]=time.monotonic()-begin
            log(f'第1块 第{args.iteration}轮 阶段结束={name} 耗时秒={timings[name]:.3f}')
            write(output/'timings.json',timings)
    try:
        metadata,source_video,fps,core,files=step('准备',lambda:prepare(args.source,work,args.sha256,args.duration,args.context))
        prior_record=read(args.prior.parent/'prior-report.json')
        if (prior_record.get('通过') is not True or prior_record.get('源摘要')!=args.sha256
                or prior_record.get('先验摘要')!=vision.sha256(args.prior)):
            raise PipelineError('SOURCE_CHANGED','品牌先验未绑定本输入摘要')
        raw=work/'all-yolo.json';binding=work/'yolo-binding.json'
        signature=vision.sha256(work/'frames-binding.json')
        if raw.exists():
            if read(binding).get('frames_binding_sha256')!=signature or read(binding).get('detected_sha256')!=vision.sha256(raw):raise PipelineError('SOURCE_CHANGED','检测缓存缺少输入帧绑定')
            rows=read(raw)
        else:
            rows=step('YOLO逐帧检测',lambda:detect_all(work/'frames',raw,enhance=args.enhance_detection,tiles=args.tiled_detection,flips=args.flipped_detection))
            write(binding,dict(frames_binding_sha256=signature,detected_sha256=vision.sha256(raw)))
        labels={int(k):str(v).lower() for k,v in read(raw.with_suffix('.timing.json'))['类别'].items()}
        prepared=read(work/'model-prepared.json')
        if prepared:
            if (prepared['frames_binding_sha256']!=signature
                    or prepared['prior_sha256']!=vision.sha256(args.prior)
                    or prepared['detected_sha256']!=vision.sha256(work/'detections.json')):
                raise PipelineError('SOURCE_CHANGED','模型轨迹缓存与当前输入不一致')
            for name,h in prepared['assets'].items():
                if vision.sha256(work/name)!=h:raise PipelineError('SOURCE_CHANGED','模板或提示摘要不一致')
            detected=read(work/'detections.json')
        else:
            detected=step('模型约束轨迹',lambda:build(files,vision.shots(files),rows,args.prior,labels,work,fps))
            write(work/'model-prepared.json',dict(frames_binding_sha256=signature,prior_sha256=vision.sha256(args.prior),
                detected_sha256=vision.sha256(work/'detections.json'),assets={str(p.relative_to(work)):vision.sha256(p)
                for folder in ['templates','prompts'] for p in (work/folder).glob('*') if p.is_file()}))
        budget=sam_budget(detected,len(files),source_video['width']*source_video['height'])
        write(output/'mask-budget.json',budget)
        log(f'第1块 轨迹数={len(detected["tracks"])} 提示数={len(detected["hints"])} SAM初始化={budget["workload"]["initializations"]} 超时上限秒={budget["timeout_seconds"]}')
        if args.prepare_only:
            write(output/'prepared.json',dict(仅准备=True,预算=budget,实际模型补画轮数=0))
            return
        def masks():
            if args.reuse_validated_masks:
                summary=read(work/'model-mask-summary.json')
                verified=VerifiedLayers(work/'layers.json',work/'detections.json')
                if len(summary.get('逐帧',[]))!=len(files):raise PipelineError('MASK_INVALID','分割缓存帧数不完整')
                for i,item in enumerate(summary['逐帧']):
                    path=work/item['mask_key']
                    if vision.sha256(path)!=item['sha256']:raise PipelineError('MASK_INVALID','分割缓存摘要不一致')
                    verified.validate(i,cv2.imread(str(path),0))
                log('第1块 mask复用已校验的本输入SAM检查点；不重新分割')
                return summary['逐帧']
            return vision.masks(work,files,detected,'propainter',budget=budget)
        manifest=step('mask',masks)
        summaries=read(work/'layers.json');current=work/'frames'
        for k,layer in enumerate(summaries['layers']):
            layer_root=Path(layer['directory']).parent
            (layer_root/'frames').symlink_to(current,target_is_directory=True)
            step(f'inpaint-{k+1}',lambda:vision.inpaint(layer_root,sorted((layer_root/'frames').glob('*.png')),detected['scenes'],'propainter',True,condition_overlap=args.condition_overlap,reference_masks=work/'masks'))
            current=layer_root/'composite'
            if len(list(current.glob('*.png')))!=len(files):raise PipelineError('OUTPUT_INCOMPLETE','补画帧不完整，禁止原帧回退')
        if not summaries['layers']:raise PipelineError('MASK_UNCERTAIN','没有可用的已验证掩膜图层')
        (work/'composite').symlink_to(current,target_is_directory=True)
        verification=VerifiedLayers(work/'layers.json',work/'detections.json')
        for i,file in enumerate(files):verification.validate(i,cv2.imread(str(work/'masks'/file.name),0))
        detail=None
        if args.preserve_detail:
            from worker.model_detail import render
            current,detail=step('联合纹理投影',lambda:render(work,files,detected,fps))
        else:current=work/'composite'
        def encode():
            silent=work/'core-video.mp4'
            run(['ffmpeg','-v','error','-y','-framerate',str(fps),'-i',current/'%06d.png','-frames:v',str(core),'-c:v','libx264','-preset','fast','-crf',str(HLSSettings.from_env().video_crf),'-threads','2','-pix_fmt','yuv420p',silent])
            audio=work/'core-audio.m4a'
            run(['ffmpeg','-v','error','-y','-i',args.source,'-map','0:a:0','-t',str(core/fps),'-c:a','copy',audio])
            offset=float(source_video.get('start_time',0))-float(metadata['format'].get('start_time',0))
            video=output/'chunk.mp4'
            run(['ffmpeg','-v','error','-y','-itsoffset',str(offset),'-i',silent,'-i',audio,'-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',video])
            if audio_hash(args.source,core/fps)!=audio_hash(video):raise PipelineError('OUTPUT_INVALID','原音频压缩内容不一致')
            return video
        video=step('编码与音轨复制',encode)
        quality=step('编码后核验',lambda:verify(work,video,files,detected,args.prior,labels,core,fps,output,args.enhance_detection,args.tiled_detection,args.flipped_detection))
        report=dict(源摘要=args.sha256,迭代=args.iteration,通过=False,模型完成=True,阶段耗时秒=timings,
                    总耗时秒=time.monotonic()-started,原帧回退数=0,主体帧数=core,处理输入帧数=len(files),
                    视频=str(video),视频摘要=vision.sha256(video),核验=quality,
                    轨迹数=len(detected['tracks']),预算=budget,
                    实际补画图层=len(summaries['layers']),复用分割=args.reuse_validated_masks,联合纹理投影=detail,模型参数=dict(窗口=HLSSettings.from_env().window_frames,重叠=HLSSettings.from_env().overlap_frames,降分辨率=False,重叠上下文=args.condition_overlap),
                    输出规格=media.probe(video),目检完成=False,说明='模型结果仅作验证；候选与目检未通过不得宣称验收通过')
        write(output/'report.json',report);log(f'第1块 本轮完成 原帧回退=0 残留候选={quality["残留候选数"]} 闪烁候选={len(quality["闪烁候选"])} 清晰度最低={quality["清晰度"]["最小保留比"]}')
    except Exception as exc:
        failure=dict(通过=False,模型完成=False,失败块=1,失败阶段=phase,错误码=getattr(exc,'code',type(exc).__name__),
                     详情=getattr(exc,'details',{}),阶段耗时秒=timings,原帧回退数=0,
                     总耗时秒=time.monotonic()-started,迭代=args.iteration,说明='阶段失败，禁止原帧补齐；检查点保留，不计为通过')
        write(output/'failure.json',failure)
        log(f'失败块=1 失败阶段={phase} 错误码={failure["错误码"]} 原帧回退=禁止 通过=False')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','prior','work','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--duration',type=float,default=HLSSettings.from_env().block_seconds)
    p.add_argument('--context',type=float,default=HLSSettings.from_env().context_seconds)
    p.add_argument('--iteration',type=int,choices=[1,2,3],required=True)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--reuse-validated-masks',action='store_true')
    p.add_argument('--condition-overlap',action='store_true')
    p.add_argument('--enhance-detection',action='store_true')
    p.add_argument('--preserve-detail',action='store_true')
    p.add_argument('--tiled-detection',action='store_true')
    p.add_argument('--flipped-detection',action='store_true')
    args=p.parse_args()
    if args.duration<=0 or args.context<0 or args.duration+args.context>30:p.error('主体与上下文总输入须在30秒内')
    cv2.setNumThreads(2);execute_one(args)


if __name__=='__main__':main()
