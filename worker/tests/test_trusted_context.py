from dataclasses import asdict
from pathlib import Path
import json
import cv2,numpy as np,pytest
from worker.trusted_context import ContextSettings,verify_input,stable_union_layers
from worker.process import PipelineError


def test_stable_union_uses_same_frame_peak_and_never_migrates_layers():
    a=np.zeros((100,100),np.uint8);a[:15]=255;b=np.zeros_like(a);b[70:85]=255
    plan=stable_union_layers([a,b])
    assert plan['layers']==1 and plan['all_components_fixed_layer']==0 and plan['maximum_fraction']==.15


def test_union_over_budget_fails_instead_of_splitting_identity():
    mask=np.zeros((100,100),np.uint8);mask[:21]=255
    with pytest.raises(PipelineError,match='超过20%'):stable_union_layers([mask])
    with pytest.raises(ValueError):ContextSettings(mask_fraction_limit=.21)


def test_model_mask_must_exclude_future_and_previous_watermarks():
    expected=np.zeros((100,100),np.uint8);expected[10:20,10:20]=255;expected[40:50,40:50]=255;current=expected.copy();current[40:50]=0
    with pytest.raises(PipelineError) as exc:verify_input(expected,current,current,expected.size,ContextSettings())
    assert exc.value.code=='UNTRUSTED_CONTEXT' and exc.value.details['源水印未排除像素']==100
    checked=verify_input(expected,expected,current,expected.size,ContextSettings())
    assert checked['源水印未排除像素']==0 and checked['模型排除像素']==200


@pytest.mark.parametrize('corrupt_input',[False,True])
def test_end_to_end_model_reads_all_masks_but_only_current_domain_is_written(tmp_path,monkeypatch,corrupt_input):
    from worker.tests.test_roi_inpaint import fixture
    from worker import roi_inpaint,roi_spatial_layers
    from worker.spatial_roi import SpatialSettings
    files,d,_=fixture(tmp_path);first=np.zeros((240,320),np.uint8);first[35:50,40:70]=255;second=np.zeros_like(first);second[50:60,160:190]=255
    (tmp_path/'layer-01/masks').mkdir(parents=True)
    for p in files:
        for folder,m in [('masks',first|second),('layer-00/masks',first),('layer-01/masks',second)]:cv2.imwrite(str(tmp_path/folder/p.name),m)
    (tmp_path/'layers.json').write_text(json.dumps(dict(layers=[dict(directory=str(tmp_path/f'layer-{k:02d}/masks')) for k in range(2)])))
    (tmp_path/'trusted-context.json').write_text(json.dumps(asdict(ContextSettings())))
    (tmp_path/'spatial-roi.json').write_text(json.dumps(asdict(SpatialSettings(minimum_width=64,minimum_height=64,maximum_crop_pixels=100000,maximum_pixel_frames=10000000))))
    (tmp_path/'inpaint-policy.json').write_text(json.dumps(dict(continue_unpassed=True)))
    monkeypatch.setenv('INPAINT_PYTHON','python');monkeypatch.setenv('PROPAINTER_ROOT',str(tmp_path));calls=[]
    class Ref:
        def __init__(self,*a):pass
        def restore(self,x,m,y,**kwargs):return y
    monkeypatch.setattr(roi_spatial_layers,'CleanReferences',Ref);monkeypatch.setattr(roi_spatial_layers,'soft_composite',lambda x,y,m:y)
    def model(args,**kw):
        frames=Path(args[args.index('--video')+1]);masks=Path(args[args.index('--mask')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True);calls.append(1)
        for p in frames.glob('*.png'):
            m=cv2.imread(str(masks/p.name),0);assert np.all(m[first|second>0]==255)
            x=cv2.imread(str(p))
            if len(calls)==2:assert np.all(x[first>0]==130) and np.all(x[second>0]<130)
            x[m>0]=130;cv2.imwrite(str(out/p.name),x)
    monkeypatch.setattr(roi_spatial_layers,'run',model)
    if corrupt_input:
        atomic=roi_inpaint.atomic_png
        def corrupt(path,value):
            if path.parent.name=='masks' and path.parent.parent.name.startswith('roi-'):value=np.zeros_like(value)
            atomic(path,value)
        monkeypatch.setattr(roi_inpaint,'atomic_png',corrupt)
        with pytest.raises(PipelineError,match='禁止推理'):
            roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True,minimum_context=400))
        audit=json.loads((tmp_path/'层输入自检.json').read_text())
        assert not calls and audit['源水印未排除像素']>0 and not audit['全部已检查尝试通过']
        return
    result=roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True,minimum_context=400))
    assert result['全部ROI通过'] and len(calls)==2
    audit=json.loads((tmp_path/'层输入自检.json').read_text());assert audit['源水印未排除像素']==0 and audit['全部已检查尝试通过']
    assert len(list((tmp_path/'model-samples').glob('**/*-raw-model.png')))==6
    for p in files:
        source=cv2.imread(str(p));final=cv2.imread(str(tmp_path/'composite'/p.name));assert np.all(final[first|second>0]==130);assert np.array_equal(source[(first|second)==0],final[(first|second)==0])


def test_reference_safety_boundary_is_model_only_and_must_be_excluded():
    from worker.trusted_context import exclusion_mask
    confirmed=np.zeros((100,100),np.uint8);confirmed[40:50,40:60]=255;before=confirmed.copy()
    expected=exclusion_mask(confirmed,ContextSettings())
    assert np.array_equal(confirmed,before) and expected[38,45]==255 and confirmed[38,45]==0
    with pytest.raises(PipelineError) as exc:verify_input(expected,confirmed,confirmed,confirmed.size,ContextSettings())
    assert exc.value.code=='UNTRUSTED_CONTEXT'
    assert verify_input(expected,expected,confirmed,confirmed.size,ContextSettings())['源水印未排除像素']==0


def test_empty_input_cannot_create_a_successful_stable_plan():
    with pytest.raises(PipelineError,match='输入为空'):stable_union_layers([])
