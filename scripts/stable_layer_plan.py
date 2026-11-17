"""冻结掩膜逐帧并集不变，验证后将所有部件固定至同一层；超限停止。"""
import argparse,json,time,shutil
from pathlib import Path
import cv2
from worker.trusted_context import ContextSettings,stable_union_layers,exclusion_mask
from worker.hls_layers import write_manifest
from worker.v12_masks import write_json,digest


def execute(args):
    start=time.monotonic();base=args.base;out=args.output;cfg=ContextSettings(**json.loads(args.config.read_text()));files=sorted((base/'masks').glob('*.png'))
    if out.exists():raise ValueError('禁止覆盖已有稳定层计划')
    maximum_context=0.
    def frames():
        nonlocal maximum_context
        for i,p in enumerate(files):
            if i%100==0:print(f'稳定联合层面积检查 {i}/{len(files)}',flush=True)
            mask=cv2.imread(str(p),0);context=exclusion_mask(mask,cfg);fraction=cv2.countNonZero(context)/context.size;maximum_context=max(maximum_context,fraction)
            if fraction>cfg.mask_fraction_limit:raise ValueError('模型可信区排除边界后超过20%，必须重规划')
            yield mask
    proof=stable_union_layers(frames(),cfg.mask_fraction_limit);out.mkdir(parents=True);(out/'frames').symlink_to((base/'frames').resolve(),target_is_directory=True)
    (out/'masks').symlink_to((base/'masks').resolve(),target_is_directory=True);layer=out/'layers/00';layer.mkdir(parents=True)
    for p in files:(layer/p.name).hardlink_to(p)
    detected=json.loads((base/'detections.json').read_text());detected['stable_layer_settings']=json.loads(args.config.read_text());detected['bindings']['prior_layers_sha256']=digest(base/'layers.json');write_json(out/'detections.json',detected)
    if (base/'animation-events.json').exists():
        old=json.loads((base/'animation-events.json').read_text());write_json(out/'animation-events.json',dict(settings=old['settings'],events=old['events'],layers={'0':old['events']}))
    write_manifest(out,out/'detections.json',[layer],files)
    write_json(out/'stable-layer-plan.json',dict(**proof,maximum_model_exclusion_fraction=maximum_context,frames=len(files),seconds=time.monotonic()-start,prior_layers_sha256=digest(base/'layers.json'),union_pixels_changed=0,说明='全部部件完整生命周期固定在第0层，按实际逐帧并集峰值及模型安全边界同时校验20%；不按桶号拆分，不修改施工掩膜像素；超限报错停止。'))
    print('稳定联合层计划完成，最大面积比例='+str(proof['maximum_fraction']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['base','output','config']:p.add_argument('--'+n,type=Path,required=True)
    execute(p.parse_args())
