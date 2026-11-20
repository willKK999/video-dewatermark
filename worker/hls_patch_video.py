"""把模型补修按原映射合成到当前成品，仅修改掩膜像素，音频直接复制。"""

import subprocess
from functools import lru_cache
from pathlib import Path
import cv2,numpy as np
from .process import PipelineError
from scripts.hls_chunk_check import mux_core_audio


def blend(video, source, mapping, masks, repaired, output, mask_validator=None):
    cap=cv2.VideoCapture(str(video))
    width,height,fps=round(cap.get(3)),round(cap.get(4)),cap.get(5)
    silent=output.with_name(output.stem+'-video.mp4')
    log=output.with_suffix('.encode.log').open('wb')
    process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264rgb','-crf','0','-preset','ultrafast','-threads','4',str(silent)],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
    @lru_cache(maxsize=8)
    def repair(index):
        mask=cv2.imread(str(masks/f'{index:06d}.png'),0)
        if mask is None:raise PipelineError('OUTPUT_INVALID','补修掩膜缺帧')
        if mask_validator is not None:mask_validator.validate(index,mask)
        if not mask.any():return None
        image=cv2.imread(str(repaired/f'{index:06d}.png'))
        if image is None:raise PipelineError('OUTPUT_INVALID','模型补修结果缺帧')
        mask=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)
        weight=np.clip(cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)/3,0,1)[:,:,None]
        return cv2.resize(image,(width,height),interpolation=cv2.INTER_CUBIC),weight
    try:
        for index in mapping:
            ok,image=cap.read()
            if not ok:raise PipelineError('OUTPUT_INVALID','当前成品合成输入缺帧')
            part=repair(index)
            if part:
                changed,weight=part
                image=np.clip(np.rint(image*(1-weight)+changed*weight),0,255).astype(np.uint8)
            process.stdin.write(image.tobytes())
        if cap.read()[0]:raise PipelineError('OUTPUT_INVALID','当前成品合成输入多帧')
        process.stdin.close()
        if process.wait()!=0:raise PipelineError('OUTPUT_INVALID','局部无损合成失败')
    finally:
        cap.release();log.close()
        if process.poll() is None:
            process.terminate();process.wait()
    mux_core_audio(silent,source,output,len(mapping)/fps)
    return output
