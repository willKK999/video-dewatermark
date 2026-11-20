"""v7策略隔离、真实像素等价与20%保护。"""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from worker import model_tracks_v7 as v7, vision
from worker.hls_layers import write_manifest
from worker.mask_equivalence import compare
from worker.process import PipelineError


def sample(root, mask):
    for folder in ['frames', 'masks', 'object-masks/1', 'templates']:
        (root/folder).mkdir(parents=True, exist_ok=True)
    file=root/'frames/000000.png'
    cv2.imwrite(str(file),np.zeros((*mask.shape,3),np.uint8))
    cv2.imwrite(str(root/'object-masks/1/000000.png'),mask)
    base=np.zeros((10,10),np.uint8);base[4:6,4:6]=255
    template=root/'templates/1.npz';np.savez(template,mask=base)
    hint=dict(frame=0,track_id=1,box=[10,10,20,20])
    d=dict(schema='yolo-model-tracks-1',tracks=[dict(id=1,template=str(template),hints=[hint])],hints=[hint])
    (root/'detections.json').write_text(json.dumps(d))
    binding=dict(source=dict(source_sha256='synthetic'),sha256={file.name:vision.sha256(file)})
    (root/'frames-binding.json').write_text(json.dumps(binding))
    return file,d,base


@pytest.mark.parametrize("mask_source", ["v7", "v8"])
def test_v7_mask_keeps_original_support_and_seven_pixel_sam_intersection(tmp_path, mask_source):
    sam=np.full((40,50),255,np.uint8)
    file,d,base=sample(tmp_path,sam)
    vision.validate_masks(tmp_path,[file],d,mask_source=mask_source)
    support=v7.place(base,[10,10,20,20],sam.shape)
    expected=support | (sam & cv2.dilate(support,np.ones((7,7),np.uint8)))
    assert np.array_equal(cv2.imread(str(tmp_path/'masks'/file.name),0),expected)
    assert not hasattr(v7.ModelTrackSettings(),'transparent_expansion')


def test_mask_comparison_rejects_missing_or_extra_pixels_and_preserves_report(tmp_path):
    roots=[tmp_path/'a',tmp_path/'b']
    for root in roots:
        file,d,_=sample(root,np.zeros((40,50),np.uint8))
        v7.validate(root,[file],d)
    output=tmp_path/'comparison.json'
    assert compare(roots[1],roots[0],output)['最低IoU']==1
    path=roots[1]/'masks/000000.png';im=cv2.imread(str(path),0);im[0,0]=255;cv2.imwrite(str(path),im)
    # 添加合法图层像素，更新摘要，证明比较拒绝的是语义差异而非损坏缓存。
    layer=roots[1]/'layer-00/masks/000000.png';cv2.imwrite(str(layer),im)
    write_manifest(roots[1],roots[1]/'detections.json',[layer.parent],[file])
    summary=roots[1]/'model-mask-summary.json';s=json.loads(summary.read_text());s['逐帧'][0]['sha256']=vision.sha256(path);summary.write_text(json.dumps(s))
    with pytest.raises(PipelineError,match='不等价'):compare(roots[1],roots[0],output)
    assert json.loads(output.read_text())['新增像素总数']==1
    binding=roots[1]/'frames-binding.json';binding.write_text('{}')
    with pytest.raises(PipelineError) as exc:compare(roots[1],roots[0],output)
    assert exc.value.code=='SOURCE_CHANGED'


def test_v7_rejects_single_target_over_twenty_percent(tmp_path):
    file,d,_=sample(tmp_path,np.full((40,50),255,np.uint8))
    np.savez(d['tracks'][0]['template'],mask=np.full((40,50),255,np.uint8))
    d['tracks'][0]['hints'][0]['box']=[0,0,50,40]
    with pytest.raises(PipelineError) as exc:v7.validate(tmp_path,[file],d)
    assert exc.value.code=='MASK_UNCERTAIN'
