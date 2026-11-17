"""以冻结确认组的自身形状和主色收边，输出独立摘要绑定的稳定联合层。"""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from worker.edge_residue import EdgeSettings,refine_group
from worker.trusted_context import read_confirmed,ContextSettings,exclusion_mask
from worker.hls_layers import write_manifest,VerifiedLayers
from scripts.frozen_mask_inpaint import digest,write


def execute(args):
    base=args.base;out=args.output;cfg=EdgeSettings(**json.loads(args.config.read_text()))
    files=sorted((base/'masks').glob('*.png'));groups=sorted((base/'group-objects').glob('*'))
    if not files or not groups:raise ValueError('缺少冻结掩膜或确认组，禁止空收边')
    if out.exists():raise ValueError('禁止覆盖已有收边结果')
    out.mkdir(parents=True);(out/'masks').mkdir();(out/'edge-band').mkdir()
    for name in ['frames','group-objects','scene-protection','subtitle-protection']:
        if not (base/name).exists():raise ValueError('缺少保护或组输入：'+name)
        (out/name).symlink_to((base/name).resolve(),target_is_directory=True)
    verifier=VerifiedLayers(base/'layers.json',base/'detections.json');records=[];started=time.monotonic()
    for i,p in enumerate(files):
        m=read_confirmed(p);verifier.validate(i,m)
        image=cv2.imread(str(base/'frames'/p.name));protected=read_confirmed(base/'scene-protection'/p.name)|read_confirmed(base/'subtitle-protection'/p.name)
        added=np.zeros_like(m);rows=[]
        for group in groups:
            gp=group/p.name
            if not gp.exists():continue
            gm=read_confirmed(gp)
            if np.any((gm>0)&(m==0)):raise ValueError('确认组超出冻结联合掩膜')
            addition,row=refine_group(image,gm,m,protected,cfg);added|=addition;rows.append(dict(group=group.name,**row))
        new=m|added
        if np.count_nonzero(exclusion_mask(new,ContextSettings()))/m.size>.2:raise ValueError('收边后模型排除超过20%')
        cv2.imwrite(str(out/'masks'/p.name),new);cv2.imwrite(str(out/'edge-band'/p.name),added)
        records.append(dict(frame=i,输入掩膜摘要=digest(p),输出掩膜摘要=digest(out/'masks'/p.name),新增像素=int(np.count_nonzero(added)),保护交集像素=int(np.count_nonzero(added&protected)),组=rows))
        if i%30==0:print(f'确认形状收边 {i}/{len(files)} 新增像素={np.count_nonzero(added)}',flush=True)
    data=json.loads((base/'detections.json').read_text());data.setdefault('bindings',{})['edge_input_layers_sha256']=digest(base/'layers.json');data['edge_settings']=json.loads(args.config.read_text());write(out/'detections.json',data)
    if (base/'animation-events.json').exists():
        old=json.loads((base/'animation-events.json').read_text());write(out/'animation-events.json',dict(settings=old['settings'],events=old['events'],layers={'0':old['events']}))
    write_manifest(out,out/'detections.json',[out/'masks'],files)
    checked=VerifiedLayers(out/'layers.json',out/'detections.json')
    for i,p in enumerate(files):checked.validate(i,read_confirmed(out/'masks'/p.name))
    write(out/'edge-refinement.json',dict(配置=data['edge_settings'],帧=records,总耗时秒=time.monotonic()-started,新增像素=sum(r['新增像素'] for r in records),保护交集像素=sum(r['保护交集像素'] for r in records)))
    print('确认形状收边完成',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','output','config']:p.add_argument('--'+name,type=Path,required=True)
    execute(p.parse_args())
