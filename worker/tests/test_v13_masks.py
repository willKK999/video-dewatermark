from argparse import Namespace
import json
import shutil
import cv2
import numpy as np
import pytest
from worker.v13_masks import merge
from worker.watermark_groups import GroupSettings,AnimationSettings


def fixture(tmp_path):
    base=tmp_path/'base';base.mkdir();root=tmp_path/'first';root.mkdir()
    data=dict(scenes=[[0,2]],subtitles={},settings=dict(subtitle_padding=4,text_outer_padding=1),hints=[])
    (base/'detections.json').write_text(json.dumps(data));(root/'detections.json').write_text(json.dumps(data))
    (root/'group-plan.json').write_text(json.dumps(dict(groups=[],nodes={},prompts=[])))
    for folder in ['frames','masks','subtitle-independent','subtitle-ocr','layer']:(base/folder).mkdir()
    for i in range(2):
        name=f'{i:06d}.png';mask=np.zeros((32,32),np.uint8);mask[4:8,4:8]=255
        cv2.imwrite(str(base/'frames'/name),np.full((32,32,3),100,np.uint8))
        for folder in ['masks','layer']:cv2.imwrite(str(base/folder/name),mask)
        for folder in ['subtitle-independent','subtitle-ocr']:cv2.imwrite(str(base/folder/name),np.zeros_like(mask))
    (base/'layers.json').write_text(json.dumps(dict(layers=[dict(directory=str(base/'layer'))])))
    (root/'frames').symlink_to(base/'frames',target_is_directory=True)
    args=Namespace(output=root,base=base,sam=tmp_path/'sam',reuse=None)
    return args


def recover(args,tmp_path):
    root=tmp_path/'recovered';root.mkdir()
    for name in ['group-plan.json','detections.json']:shutil.copy2(args.output/name,root/name)
    (root/'frames').symlink_to(args.base/'frames',target_is_directory=True)
    return Namespace(**dict(vars(args),output=root,reuse=args.output))


def test_recovery_revalidates_complete_cached_masks(tmp_path):
    args=fixture(tmp_path);merge(args,GroupSettings(),AnimationSettings());new=recover(args,tmp_path)
    merge(new,GroupSettings(),AnimationSettings())
    rows=json.loads((new.output/'statistics.json').read_text())['frames']
    assert all(p['reused_checkpoint'] for p in rows)
    assert all((new.output/'masks'/f'{i:06d}.png').read_bytes()==(args.output/'masks'/f'{i:06d}.png').read_bytes() for i in range(2))


def test_corrupt_or_incompatible_recovery_is_rejected(tmp_path):
    args=fixture(tmp_path);merge(args,GroupSettings(),AnimationSettings());new=recover(args,tmp_path)
    mask=cv2.imread(str(args.output/'masks/000000.png'),0);mask[20,20]=255;cv2.imwrite(str(args.output/'masks/000000.png'),mask)
    with pytest.raises(ValueError,match='缓存严格并集'):merge(new,GroupSettings(),AnimationSettings())
