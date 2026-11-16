"""独立核验SAM3.1逐帧掩膜，并写入现有VerifiedLayers兼容清单。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from worker.hls_layers import write_manifest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',required=True);p.add_argument('--frames',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--baseline-manifest',required=True)
    p.add_argument('--baseline-prefix',required=True);p.add_argument('--baseline-local',required=True)
    args=p.parse_args();t=time.perf_counter();root=Path(args.result)
    metrics=json.loads((root/'metrics.json').read_text());source=json.loads((root/'source.json').read_text())
    if metrics['config']!=source['config']:raise ValueError('配置摘要绑定不一致')
    files=sorted(Path(args.frames).glob('*.png'));names={f.name for f in files}
    if len(files)!=metrics['input_frame_count'] or names!=set(source['frames']):raise ValueError('输入帧集合不一致')
    for f in files:
        if sha(f)!=source['frames'][f.name]:raise ValueError('输入帧内容摘要不符')
    dirs=[root/x['directory'] for x in metrics['layers']]
    write_manifest(root,root/'source.json',dirs,files)
    for row,new in zip(metrics['layers'],json.loads((root/'layers.json').read_text())['layers']):
        if row['sha256']!=new['sha256']:raise ValueError('SAM分层摘要不符')
    base_manifest=Path(args.baseline_manifest);base_data=json.loads(base_manifest.read_text())
    base_dirs=[Path(args.baseline_local)/Path(x['directory']).relative_to(args.baseline_prefix) for x in base_data['layers']]
    for row,directory in zip(base_data['layers'],base_dirs):
        for name,expected in row['sha256'].items():
            if sha(directory/name)!=expected:raise ValueError('基线分层摘要不符')
    binding={'sam_source_sha256':sha(root/'source.json'),'baseline_manifest_sha256':sha(base_manifest),
             'baseline_detection_sha256':base_data['detected_sha256'],'method':'strict-bitwise-union'}
    (root/'merged-source.json').write_text(json.dumps(binding,ensure_ascii=False,indent=2))
    merged=root/'merged';merged.mkdir(exist_ok=True)
    write_manifest(merged,root/'merged-source.json',base_dirs+dirs,files)
    removed=0
    for f in files:
        mask=cv2.imread(str(root/'masks'/f.name),0);base=cv2.imread(str(Path(args.baseline)/f.name),0)
        actual=cv2.imread(str(root/'merged-masks'/f.name),0)
        expected=np.zeros_like(mask)
        for d in dirs:expected|=cv2.imread(str(d/f.name),0)
        if not np.array_equal(expected,mask):raise ValueError('SAM联合与已验证分层不一致')
        base_union=np.zeros_like(base)
        for d in base_dirs:base_union|=cv2.imread(str(d/f.name),0)
        if not np.array_equal(base_union,base):raise ValueError('基线联合与分层不一致')
        if not np.array_equal(actual,mask|base):raise ValueError('合并掩膜不是严格并集')
        if mask.shape!=base.shape or actual.shape!=mask.shape:raise ValueError('合并形状不一致')
        removed+=int(np.count_nonzero((base>0)&(actual==0)))
    result={'frames':len(files),'sam_layers':len(dirs),'merged_layers':len(dirs)+len(base_dirs),
            'sam_source_sha256':sha(root/'source.json'),'metrics_sha256':sha(root/'metrics.json'),
            'baseline_removed_pixel_frames':removed,'all_hashes_verified':True,'all_layers_at_most_20_percent':True,
            'sam_union_equals_layers':True,'baseline_union_equals_layers':True,'merged_is_exact_union':True,
            'seconds':time.perf_counter()-t}
    (root/'独立核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
