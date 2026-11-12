"""补齐移动淡标识的完整图形内部，不将灰阶填色遗漏为干净背景。"""
import json,time,os
from pathlib import Path
import cv2,numpy as np
from .hls_v7 import load_assets
from .hls_settings import HLSSettings
from .fine_masks import Supports
from .multitrack import load_prior
from .hls_graphic_repair import residual_support,transformed
from .branding import complete_play_support
from .hls_mask_pass import process
from . import vision
from .process import PipelineError


def patch(work,assets_path):
    started=time.monotonic();settings=HLSSettings.from_env()
    assets=load_assets(assets_path,vision.sha256(work.parent/'source.mp4'))
    info=json.loads((work/'prepared.json').read_text());quality=json.loads((work/'audit/quality.json').read_text())
    wanted={p['track_id'] for p in quality['moving_evidence']}
    if not wanted:return
    video=work/'native/chunk1.mp4';digest=vision.sha256(video);root=work/'moving-graphic-repair'
    if root.exists():
        report=root/'report.json'
        if report.exists() and json.loads(report.read_text()).get('输出摘要')==digest:return
        root.rename(work/('moving-graphic-repair-previous-'+str(time.time_ns())))
    root.mkdir();(root/'masks').mkdir();os.link(video,root/'before.mp4')
    detected=json.loads(Path(info['detected']).read_text());prior,signal,regions,*_=load_prior(Path(assets['prior']))
    support=Supports(Path(assets['prior']),detected,np.load(assets['alpha'])['alpha'],Path(info['graphics']))
    complete={r['id']:residual_support(prior,signal,r) for r in regions}
    files=sorted((work/'frames').glob('*.png'));mapping=json.loads((work/'native/native-report.json').read_text())['帧映射']
    active=[];shapes=[];maximum=0
    target_frames={p['frame'] for track in detected['tracks'] if track['id'] in wanted for p in track['hints']}
    for index,file in enumerate(files):
        image=cv2.imread(str(file)) if index in target_frames else None
        mask=np.zeros(prior.shape,np.uint8)
        for part in support.at(index,image) if image is not None else []:
            if part['id'] not in wanted or part['kind']!='moving':continue
            region=next(r for r in regions if r['id']==part['region'])
            mask|=transformed(complete[region['id']],region,part,image.shape[:2])
            filled,evidence=complete_play_support(image,part['point']['box']);mask|=filled
            if evidence:shapes.append(dict(frame=index,track_id=part['id'],图形=evidence))
        fraction=np.count_nonzero(mask)/mask.size;maximum=max(maximum,fraction)
        if fraction>settings.mask_fraction_limit:raise PipelineError('MASK_UNCERTAIN','移动图形单遍超过20%保护',review=True)
        if mask.any():active.append(index)
        cv2.imwrite(str(root/'masks'/file.name),mask)
    if not active:raise PipelineError('MASK_UNCERTAIN','移动残留缺少自动轨迹掩膜',review=True)
    output,ranges,missing=process(work,root,info,mapping,files,detected['scenes'],active,settings)
    result_digest=vision.sha256(output)
    (root/'report.json').write_text(json.dumps(dict(方法='移动轨迹完整播放图形与真实笔画-1',输入摘要=digest,输出摘要=result_digest,目标轨迹=sorted(wanted),原始证据=quality['moving_evidence'],自动几何=shapes,实际补修帧=active,上下文区间=ranges,最大面积比例=maximum,补齐未映射帧=missing,掩膜来源清单摘要=vision.sha256(root/'layers.json'),耗时秒=time.monotonic()-started,音频原样复制=True),ensure_ascii=False,indent=2))
    output.replace(video)
    for name in ['audit','verified.json']:
        path=work/name
        if path.exists():path.rename(work/(name+'-before-moving-graphic-'+str(time.time_ns())))
