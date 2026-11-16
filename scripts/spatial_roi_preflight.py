"""不运行模型，验证全部时空ROI覆盖及预算；写入独立预检记录。"""
import argparse,json,time
from pathlib import Path
from dataclasses import asdict
import cv2,numpy as np
from worker.roi_inpaint import ROISettings,padded_box,bbox
from worker.spatial_roi import SpatialSettings,plan,padded_area
from worker.watermark_groups import AnimationSettings,planned_windows
from worker.hls_settings import HLSSettings
from worker.v12_masks import write_json,digest


def execute(args):
    start=time.monotonic();root=args.masks;files=sorted((root/'frames').glob('*.png'));settings=HLSSettings.from_env()
    cfg=ROISettings(**json.loads(args.baseline.read_text())['ROI']['配置']);spatial=SpatialSettings(**json.loads(args.config.read_text()));animation=json.loads((root/'animation-events.json').read_text());acfg=AnimationSettings(**animation['settings']);data=json.loads((root/'detections.json').read_text());manifest=json.loads((root/'layers.json').read_text());records=[]
    for k,l in enumerate(manifest['layers']):
        masks=[]
        for p in files:
            path=Path(l['directory'])/p.name
            if digest(path)!=l['sha256'][p.name]:raise ValueError('冻结图层摘要改变')
            m=cv2.imread(str(path),0)
            if np.count_nonzero(m)/m.size>.2:raise ValueError('图层超过20%')
            masks.append(m)
        for a,b in data['scenes']:
            events=[e['interval'] for e in animation['layers'].get(str(k),[]) if a<=e['interval'][0]<b]
            for left,right in planned_windows(a,b,settings.window_frames,settings.overlap_frames,events,acfg):
                footprint=np.bitwise_or.reduce(masks[left:right])
                rows=plan(footprint,right-left+settings.overlap_frames,cfg,spatial)
                if rows:records.append(dict(layer=k,start=left,end=right,previous_whole_crop_pixels=padded_area(padded_box(bbox(footprint),footprint.shape,cfg)),parts=[dict(box=r['box'],pixels=padded_area(r['box']),budget_pixel_frames=padded_area(r['box'])*(right-left+settings.overlap_frames)) for r in rows]))
    write_json(args.output,dict(passed=True,frames=len(files),settings=asdict(spatial),windows=records,tasks=sum(len(r['parts']) for r in records),maximum_crop_pixels=max(p['pixels'] for r in records for p in r['parts']),maximum_pixel_frames=max(p['budget_pixel_frames'] for r in records for p in r['parts']),seconds=time.monotonic()-start,mask_manifest_sha256=digest(root/'layers.json')))
    print('空间ROI预检通过；冻结掩膜未重算；全部像素恰好分配一次',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['masks','baseline','config','output']:p.add_argument('--'+name,type=Path,required=True)
    execute(p.parse_args())
