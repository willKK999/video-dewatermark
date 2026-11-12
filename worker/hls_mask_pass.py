"""对已有自动掩膜执行同一完整补画与原规格合成流程。"""
from pathlib import Path
from .hls_model_frames import capture
from .hls_layers import write_manifest,VerifiedLayers
from .hls_repair import intervals
from .hls_patch_video import blend
from . import vision
from .process import PipelineError
from scripts.hls_chunk_check import audio_hash


def process(work,root,info,mapping,files,scenes,active,settings):
    video=work/'native/chunk1.mp4'
    write_manifest(root,Path(info['detected']),[root/'masks'],files)
    validated=VerifiedLayers(root/'layers.json',Path(info['detected']))
    missing=capture(video,mapping,root/'frames',files)
    ranges=intervals(active,scenes,len(files),settings.model_fps)
    vision.inpaint(root,sorted((root/'frames').glob('*.png')),ranges,'propainter',True,roi=False,condition_overlap=True)
    output=blend(video,Path(info['source']),mapping,root/'masks',root/'composite',root/'patched.mp4',validated)
    if audio_hash(video)!=audio_hash(output):raise PipelineError('OUTPUT_INVALID','局部补修改变了音频')
    return output,ranges,missing
