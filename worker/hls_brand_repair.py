"""按编码后品牌证据补修独立小字，原检测、时间轴及画质参数保持不变。"""

import json,time,os
from pathlib import Path
import cv2,numpy as np
from .branding import opaque_brand_ink
from .hls_settings import HLSSettings
from .hls_model_frames import capture
from .hls_patch_video import blend
from .hls_repair import intervals
from . import vision
from .process import PipelineError
from scripts.hls_chunk_check import audio_hash


def repair_targets(evidence, scenes, count, padding):
    targets={}
    for point in evidence:
        frame=point['model_frame']
        left,right=next((a,b) for a,b in scenes if a<=frame<b)
        for index in range(max(left,frame-padding),min(right,frame+padding+1,count)):
            targets.setdefault(index,[]).append(point['box'])
    return targets


def patch(work, assets_path):
    started=time.monotonic();settings=HLSSettings.from_env()
    from .hls_v7 import load_assets
    info=json.loads((work/'prepared.json').read_text())
    assets=load_assets(assets_path,vision.sha256(work.parent/'source.mp4'))
    video=work/'native/chunk1.mp4';digest=vision.sha256(video)
    audit=json.loads((work/'audit/brand.json').read_text())
    if not audit['证据']:return
    root=work/'brand-ink-repair'
    if root.exists():
        saved=root/'report.json'
        if saved.exists() and json.loads(saved.read_text()).get('输出摘要')==digest:return
        root.rename(work/('brand-ink-repair-previous-'+str(time.time_ns())))
    root.mkdir();(root/'masks').mkdir()
    os.link(video,root/'before.mp4')
    mapping=json.loads((work/'native/native-report.json').read_text())['帧映射']
    detected=json.loads(Path(info['detected']).read_text())
    files=sorted((work/'frames').glob('*.png'))
    targets=repair_targets(audit['证据'],detected['scenes'],len(files),settings.overlap_frames)
    colors=np.load(assets['palette']);active=[];maximum=0
    for index,file in enumerate(files):
        mask=np.zeros((720,1280),np.uint8)
        if index in targets:
            image=cv2.imread(str(file))
            for box in targets[index]:mask|=opaque_brand_ink(image,box,colors)
        fraction=np.count_nonzero(mask)/mask.size;maximum=max(maximum,fraction)
        if fraction>settings.mask_fraction_limit:raise PipelineError('MASK_UNCERTAIN','独立品牌笔画单遍超过20%保护',review=True)
        if mask.any():active.append(index)
        cv2.imwrite(str(root/'masks'/file.name),mask)
    if not active:raise PipelineError('MASK_UNCERTAIN','品牌残留缺少可靠前景笔画',review=True)
    missing=capture(video,mapping,root/'frames',files)
    ranges=intervals(active,detected['scenes'],len(files),settings.model_fps)
    vision.inpaint(root,sorted((root/'frames').glob('*.png')),ranges,'propainter',True,condition_overlap=True)
    output=blend(video,Path(info['source']),mapping,root/'masks',root/'composite',root/'patched.mp4')
    if audio_hash(video)!=audio_hash(output):raise PipelineError('OUTPUT_INVALID','独立品牌补修改变了音频')
    result_digest=vision.sha256(output)
    (root/'report.json').write_text(json.dumps(dict(方法='当前成品独立不透明笔画补修-1',输入摘要=digest,输出摘要=result_digest,原始证据=audit['证据'],目标模型帧=sorted(targets),实际补修帧=active,上下文区间=ranges,最大面积比例=maximum,补齐未映射帧=missing,耗时秒=time.monotonic()-started,音频原样复制=True),ensure_ascii=False,indent=2))
    output.replace(video)
    for name in ['audit','verified.json']:
        path=work/name
        if path.exists():path.rename(work/(name+'-before-brand-ink-'+str(time.time_ns())))
