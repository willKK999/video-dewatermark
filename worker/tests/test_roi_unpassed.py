from dataclasses import asdict
from pathlib import Path
import json
import cv2,numpy as np,pytest
from worker import roi_spatial_layers,roi_inpaint
from worker.spatial_roi import SpatialSettings
from worker.process import PipelineError
from worker.roi_attempt_cache import restore
from worker.tests.test_roi_inpaint import fixture


def setup(tmp_path,monkeypatch,layers=2):
    files,d,mask=fixture(tmp_path);mask[50:60,160:190]=255
    for p in files:
        for folder in ['masks','layer-00/masks']:cv2.imwrite(str(tmp_path/folder/p.name),mask)
    manifest=json.loads((tmp_path/'layers.json').read_text());manifest['layers']*=layers;(tmp_path/'layers.json').write_text(json.dumps(manifest))
    cfg=SpatialSettings(dilation_pixels=0,merge_gap_pixels=0,minimum_width=64,minimum_height=64,fixed_context_pixels=96,maximum_crop_pixels=40000,maximum_pixel_frames=1200000)
    (tmp_path/'spatial-roi.json').write_text(json.dumps(asdict(cfg)));(tmp_path/'inpaint-policy.json').write_text(json.dumps(dict(continue_unpassed=True)))
    monkeypatch.setenv('INPAINT_PYTHON','python');monkeypatch.setenv('PROPAINTER_ROOT',str(tmp_path))
    class Ref:
        def __init__(self,*args):pass
        def restore(self,original,mask,repair,**kwargs):return repair
    monkeypatch.setattr(roi_spatial_layers,'CleanReferences',Ref);monkeypatch.setattr(roi_spatial_layers,'soft_composite',lambda original,repair,mask:repair)
    return files,d,mask


def test_four_original_seam_attempts_choose_best_without_passing_and_continue_layers(tmp_path,monkeypatch):
    files,d,mask=setup(tmp_path,monkeypatch);calls=[]
    def model(args,**kwargs):
        n=len(calls);calls.append(args);delta=[20,80,70,60,75][n] if n<5 else 20
        frames=Path(args[args.index('--video')+1]);masks=Path(args[args.index('--mask')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        for p in frames.glob('*.png'):
            x=cv2.imread(str(p));m=cv2.imread(str(masks/p.name),0);x[m>0]+=delta;cv2.imwrite(str(out/p.name),x)
    monkeypatch.setattr(roi_spatial_layers,'run',model)
    result=roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True,context_ratio=.25,minimum_context=32))
    failed=result['任务'][1]
    assert [p['subvideo_length'] for p in failed['尝试']]==[36,36,36,18]
    assert [p['重叠最大差异'] for p in failed['尝试']]==[60,50,40,55]
    assert not failed['通过'] and failed['最佳尝试']==2 and failed['采用未过结果']
    assert result['未过块']==[2] and not result['全部ROI通过'] and result['覆盖核验通过']
    assert len(result['任务'])==4 and all(p['通过'] for p in result['任务'][2:])
    for p in files:
        old=cv2.imread(str(p));new=cv2.imread(str(tmp_path/'composite'/p.name));assert np.array_equal(old[mask==0],new[mask==0])
        assert np.all(new[50:60,160:190]==old[50:60,160:190]+80)
    # 失败暂存是无损缓存，后续层改写结果后仍可重建原尝试。
    image=restore(tmp_path/'roi-0002-attempt-02',tmp_path/'roi-0002-baseline',files[0].name)
    assert np.all(image[50:60,160:190]==80)
    assert (tmp_path/'roi-0002-attempt-02/seam-difference'/files[0].name).exists()


def test_no_complete_candidate_continues_other_tasks_but_refuses_source_fallback(tmp_path,monkeypatch):
    files,d,mask=setup(tmp_path,monkeypatch,layers=1);calls=[]
    def model(args,**kwargs):
        n=len(calls);calls.append(args);frames=Path(args[args.index('--video')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        if n==0:return
        for p in frames.glob('*.png'):
            x=cv2.imread(str(p));x+=20;cv2.imwrite(str(out/p.name),x)
    monkeypatch.setattr(roi_spatial_layers,'run',model)
    with pytest.raises(PipelineError) as exc:roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True,context_ratio=.25,minimum_context=32))
    assert exc.value.code=='ROI_UNFILLABLE' and len(calls)==2
    report=json.loads((tmp_path/'roi-report.json').read_text());assert report['不可填块']==[1] and report['任务'][1]['通过']
