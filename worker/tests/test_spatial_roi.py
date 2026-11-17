from dataclasses import replace,asdict
import json
from pathlib import Path
import cv2
import numpy as np
from worker.spatial_roi import SpatialSettings,plan,padded_area
from worker.roi_inpaint import ROISettings


def test_distant_regions_are_not_merged_and_masks_unchanged():
    mask=np.zeros((400,600),np.uint8);mask[20:60,20:100]=255;mask[280:330,420:540]=255;copy=mask.copy()
    jobs=plan(mask,36,ROISettings(context_ratio=.75,minimum_context=96),SpatialSettings(minimum_width=128,minimum_height=128))
    assert len(jobs)==2 and np.array_equal(copy,mask)
    assert np.array_equal(np.bitwise_or.reduce([j['ownership'] for j in jobs]),mask)


def test_large_component_splits_to_area_and_pixel_frame_budget():
    mask=np.zeros((720,1280),np.uint8);mask[200:500,300:1100]=255
    settings=SpatialSettings();jobs=plan(mask,124,ROISettings(context_ratio=.75,minimum_context=96),settings)
    assert len(jobs)>1
    for row in jobs:
        assert padded_area(row['box'])<=settings.maximum_crop_pixels
        assert padded_area(row['box'])*124<=settings.maximum_pixel_frames
    count=np.sum(np.stack([p['ownership']>0 for p in jobs]),axis=0)
    assert np.all(count[mask>0]==1) and not np.any(count[mask==0])


def test_near_regions_can_merge_only_within_budget():
    mask=np.zeros((120,180),np.uint8);mask[40:55,40:50]=255;mask[40:55,60:70]=255
    settings=SpatialSettings(dilation_pixels=1,merge_gap_pixels=15)
    assert len(plan(mask,36,ROISettings(minimum_context=4),settings))==1


def test_spatial_tasks_commit_only_owned_pixels_and_cover_all(tmp_path,monkeypatch):
    from worker.tests.test_roi_inpaint import fixture
    from worker.tests.test_roi_layers import setup
    from worker import roi_spatial_layers,roi_inpaint
    files,d,mask=fixture(tmp_path);setup(monkeypatch,tmp_path)
    mask[170:185,250:280]=255
    for p in files:
        for folder in ['masks','layer-00/masks']:cv2.imwrite(str(tmp_path/folder/p.name),mask)
    # 此样例只验证原生写回域与最终覆盖，模型返回不同颜色。
    (tmp_path/'spatial-roi.json').write_text(json.dumps(asdict(SpatialSettings(dilation_pixels=0,merge_gap_pixels=0,minimum_width=64,minimum_height=64,fixed_context_pixels=16))))
    class Ref:
        def __init__(self,*args):pass
        def restore(self,original,mask,repair,**kwargs):return repair
    monkeypatch.setattr(roi_spatial_layers,'CleanReferences',Ref)
    monkeypatch.setattr(roi_spatial_layers,'soft_composite',lambda original,repair,mask:repair)
    def model(args,**kwargs):
        frames=Path(args[args.index('--video')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        for p in frames.glob('*.png'):
            x=cv2.imread(str(p));x[:]+=30;cv2.imwrite(str(out/p.name),x)
    monkeypatch.setattr(roi_spatial_layers,'run',model)
    result=roi_inpaint.execute(tmp_path,files,d,cfg=ROISettings(sequential_layers=True))
    assert result['覆盖核验通过']
    assert len(result['任务'])==2
    for p in files:
        old=cv2.imread(str(p));new=cv2.imread(str(tmp_path/'composite'/p.name))
        assert np.array_equal(old[mask==0],new[mask==0])
        assert np.all(new[mask>0]==old[mask>0]+30)


def test_minimum_context_size_shifts_inward_at_boundary():
    mask=np.zeros((1280,720),np.uint8);mask[2:30,3:50]=255
    jobs=plan(mask,124,ROISettings(context_ratio=.75,minimum_context=96),SpatialSettings())
    assert len(jobs)==1 and jobs[0]['box']==[0,0,512,512]
    mask[:]=0;mask[-30:,-50:]=255
    jobs=plan(mask,124,ROISettings(context_ratio=.75,minimum_context=96),SpatialSettings())
    a,b,c,d=jobs[0]['box'];assert c==720 and d==1280 and c-a>=512 and d-b>=512


def test_expanded_overlap_merges_without_near_core_gap():
    mask=np.zeros((1280,720),np.uint8);mask[100:120,100:120]=255;mask[360:380,360:380]=255
    jobs=plan(mask,124,ROISettings(context_ratio=.75,minimum_context=96),SpatialSettings(merge_gap_pixels=0))
    assert len(jobs)==1
    assert padded_area(jobs[0]['box'])<=708480


def test_wide_jobs_remain_large_but_area_limited():
    mask=np.zeros((1280,720),np.uint8);mask[10:120,10:260]=255;mask[450:750,120:620]=255
    jobs=plan(mask,124,ROISettings(context_ratio=.75,minimum_context=96),SpatialSettings())
    for row in jobs:
        a,b,c,d=row['box'];assert c-a>=512 and d-b>=512
        assert padded_area(row['box'])<=708480
    assert np.array_equal(np.bitwise_or.reduce([row['ownership'] for row in jobs]),mask)
