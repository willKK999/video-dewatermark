"""只补画已冻结的逐帧图层；模型配置从历史报告读取，不重新检测或修改掩膜。"""
import argparse
from dataclasses import asdict
import hashlib,json,re,shutil,time
from pathlib import Path
import cv2
import numpy as np
from worker import media,roi_inpaint
from worker.hls_layers import VerifiedLayers
from worker.hls_settings import HLSSettings
from worker.process import PipelineError,run
from scripts.hls_chunk_check import audio_hash


def read(path):return json.loads(Path(path).read_text())
def write(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_profile(report):
    cfg=roi_inpaint.ROISettings(**report['ROI']['配置'])
    settings=HLSSettings.from_env()
    model=report['模型参数']
    if (settings.window_frames,settings.overlap_frames)!=(model['窗口'],model['重叠']):
        raise ValueError('当前窗口与历史配置不同')
    if settings.overlap_mae_limit!=35 or settings.mask_fraction_limit!=.2:
        raise ValueError('保护阈值必须保持35和20%')
    if model['降分辨率'] or report.get('联合纹理投影') is not None or not cfg.sequential_layers:
        raise ValueError('此入口要求原生逐层ROI且无联合纹理投影')
    return cfg


class RecoveryBudget:
    """沿用历史恢复参数，但同一任务只允许一次真实错误恢复。"""
    def __init__(self):self.failures={}
    def check(self,message):
        m=re.search(r'任务=(\d+) 尝试=\d+ .*通过=False 错误码=(\S+)',message)
        if m:
            task=int(m[1]);self.failures[task]=self.failures.get(task,0)+1
            if self.failures[task]>1:
                raise PipelineError('RECOVERY_EXHAUSTED','同一任务一次通用恢复后仍失败，停止')


def execute(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    work=output/'run';work.mkdir()
    frozen=Path(args.masks);baseline=read(args.baseline);cfg=frozen_profile(baseline)
    times={};started=time.monotonic();budget=RecoveryBudget()
    def log(message):
        line=time.strftime('%Y-%m-%d %H:%M:%S')+' '+message
        with (output/'progress.log').open('a') as f:f.write(line+'\n')
        print(line,flush=True);budget.check(message)
        # 已完成层之后，只保留后续计算仍需的当前层；不触碰输入、掩膜和媒体。
        m=re.search(r'逐层ROI 图层=(\d+) 完成',message)
        if m:
            current=int(m[1])-1
            for directory in work.glob('roi-layer-*'):
                if int(directory.name.rsplit('-',1)[1])<current:
                    for p in directory.glob('*.png'):p.unlink()
    def step(name,callback):
        t=time.monotonic();log('阶段开始='+name)
        try:return callback()
        finally:
            times[name]=time.monotonic()-t;write(output/'timings.json',times);log('阶段结束='+name+f' 耗时秒={times[name]:.3f}')
    try:
        settings=HLSSettings.from_env()
        def prepare():
            if digest(args.source)!=baseline['源摘要']:raise ValueError('源片与历史报告摘要不符')
            binding_file=(frozen/'frames').resolve().parent/'frames-binding.json'
            original=read(binding_file)
            for name in ['frames','masks']:(work/name).symlink_to((frozen/name).resolve(),target_is_directory=True)
            for name in ['detections.json','layers.json']:shutil.copy2(frozen/name,work/name)
            files=sorted((work/'frames').glob('*.png'))
            if len(files)!=args.frames:raise ValueError('冻结输入帧数不符')
            verifier=VerifiedLayers(work/'layers.json',work/'detections.json');binding=[]
            for i,p in enumerate(files):
                if digest(p)!=original['sha256'][p.name]:raise ValueError('冻结解码帧摘要改变')
                layers=verifier.masks(i);actual=cv2.imread(str(work/'masks'/p.name),0)
                verifier.validate(i,actual)
                if not np.array_equal(np.bitwise_or.reduce(layers),actual):raise ValueError('联合掩膜与冻结图层不严格相等')
                binding.append(dict(frame=i,source=digest(p),mask=digest(work/'masks'/p.name)))
            write(output/'input-binding.json',dict(source_sha256=digest(args.source),frames=binding,layers_sha256=digest(work/'layers.json')))
            write(output/'profile.json',dict(历史报告摘要=digest(args.baseline),ROI=asdict(cfg),窗口=settings.window_frames,重叠=settings.overlap_frames,接缝上限=settings.overlap_mae_limit,每层面积上限=settings.mask_fraction_limit,CRF=settings.video_crf,错误恢复预算=1))
            return files
        files=step('冻结输入与掩膜核验',prepare)
        summary=step('逐层ROI补画',lambda:roi_inpaint.execute(work,files,read(work/'detections.json'),log,cfg))
        def encode():
            silent=work/'core-video.mp4';audio=work/'core-audio.m4a';video=output/'第1块.mp4'
            metadata=media.probe(args.source);vs=next(s for s in metadata['streams'] if s['codec_type']=='video')
            from fractions import Fraction
            fps=float(Fraction(vs['avg_frame_rate']));duration=len(files)/fps
            run(['ffmpeg','-v','error','-y','-framerate',str(fps),'-i',work/'composite/%06d.png','-frames:v',str(len(files)),'-c:v','libx264','-preset','fast','-crf',str(settings.video_crf),'-threads','2','-pix_fmt','yuv420p',silent])
            run(['ffmpeg','-v','error','-y','-i',args.source,'-map','0:a:0','-t',str(duration),'-c:a','copy',audio])
            offset=float(vs.get('start_time',0))-float(metadata['format'].get('start_time',0))
            run(['ffmpeg','-v','error','-y','-itsoffset',str(offset),'-i',silent,'-i',audio,'-map','0:v:0','-map','1:a:0','-c','copy','-movflags','+faststart',video])
            if audio_hash(args.source,duration)!=audio_hash(video):raise ValueError('原音频压缩内容不一致')
            run(['ffmpeg','-v','error','-i',video,'-f','null','-'])
            return dict(文件=str(video),sha256=digest(video),规格=media.probe(video),音频压缩内容一致=True)
        result=step('编码与音频完整性',encode)
        write(output/'render.json',dict(模型完成=True,通过=False,尚待验收=True,帧数=len(files),ROI=summary,视频=result,阶段耗时秒=times,总耗时秒=time.monotonic()-started))
        log('780帧补画编码完成，等待独立验收；未宣称通过')
    except Exception as exc:
        write(output/'failure.json',dict(模型完成=False,通过=False,错误码=getattr(exc,'code',type(exc).__name__),说明=str(exc)[:500],阶段耗时秒=times))
        log('补画失败，停止；错误码='+getattr(exc,'code',type(exc).__name__))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','masks','baseline','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--frames',type=int,default=780)
    execute(p.parse_args())
