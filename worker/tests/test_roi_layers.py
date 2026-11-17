"""通用逐层ROI的输入顺序、跨层重叠、时间窗及失败保护。"""
import json
from pathlib import Path
import cv2,numpy as np,pytest
from worker import roi_spatial_layers as roi_layers,roi_inpaint
from worker.process import PipelineError
from worker.tests.test_roi_inpaint import fixture


class References:
    def __init__(self,*args):pass
    def restore(self,original,mask,repair,**kwargs):return repair


def setup(monkeypatch,tmp_path):
    monkeypatch.setenv('INPAINT_PYTHON','python');monkeypatch.setenv('PROPAINTER_ROOT',str(tmp_path))
    monkeypatch.setattr(roi_layers,'CleanReferences',References)
    monkeypatch.setattr(roi_layers,'soft_composite',lambda original,repair,mask:repair)


def test_layer_overlap_is_repainted_from_previous_layer(tmp_path,monkeypatch):
    files,d,mask=fixture(tmp_path);setup(monkeypatch,tmp_path)
    # 同一掩膜分两遍，验证全局覆盖记录不会取消第二层。
    manifest=json.loads((tmp_path/'layers.json').read_text());manifest['layers']*=2
    (tmp_path/'layers.json').write_text(json.dumps(manifest));inputs=[]
    def model(args,**kw):
        frame_dir=Path(args[args.index('--video')+1]);mask_dir=Path(args[args.index('--mask')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        assert args[args.index('--subvideo_length')+1]=='36'
        assert args[args.index('--neighbor_length')+1]=='6'
        assert args[args.index('--ref_stride')+1]=='20' and '--fp16' in args
        for p in sorted(frame_dir.glob('*.png')):
            image=cv2.imread(str(p));m=cv2.imread(str(mask_dir/p.name),0)
            inputs.append(float(image[m>0].max()));image[m>0]+=40;cv2.imwrite(str(out/p.name),image)
    monkeypatch.setattr(roi_layers,'run',model)
    result=roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True,context_ratio=.75,minimum_context=96))
    assert result['实际图层数']==2 and result['覆盖核验通过']
    assert inputs[:6]==list(range(20,26)) and inputs[6:]==list(range(60,66))
    for p in files:
        original=cv2.imread(str(p));final=cv2.imread(str(tmp_path/'composite'/p.name))
        assert np.array_equal(final[mask==0],original[mask==0])
        assert np.all(final[mask>0]==original[mask>0]+80)


def test_overlap_windows_read_frozen_previous_layer_and_mix(tmp_path,monkeypatch):
    files,d,mask=fixture(tmp_path);setup(monkeypatch,tmp_path)
    monkeypatch.setenv('HLS_WINDOW_FRAMES','5');monkeypatch.setenv('HLS_OVERLAP_FRAMES','2')
    calls=[]
    def model(args,**kw):
        frames=Path(args[args.index('--video')+1]);masks=Path(args[args.index('--mask')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        call=[]
        for p in sorted(frames.glob('*.png')):
            im=cv2.imread(str(p));m=cv2.imread(str(masks/p.name),0);call.append(float(im[m>0].mean()))
            im[m>0]+=20+len(calls)*3;cv2.imwrite(str(out/p.name),im)
        calls.append(call)
    monkeypatch.setattr(roi_layers,'run',model)
    result=roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True))
    assert calls==[[20,21,22,23,24],[23,24,25]]
    assert result['任务'][1]['重叠最大差异']==3
    assert np.all(cv2.imread(str(tmp_path/'composite/000003.png'))[mask>0]==44)


def test_large_context_does_not_increase_mask_or_bypass_area_limit(tmp_path,monkeypatch):
    files,d,mask=fixture(tmp_path);setup(monkeypatch,tmp_path)
    for p in files:cv2.imwrite(str(tmp_path/'layer-00/masks'/p.name),np.full(mask.shape,255,np.uint8))
    monkeypatch.setattr(roi_layers,'run',lambda *a,**kw:pytest.fail('面积超限必须在推理前拒绝'))
    with pytest.raises(PipelineError,match='20%'):
        roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True,context_ratio=1.))


def test_layer_model_missing_frames_keeps_failure_checkpoint(tmp_path,monkeypatch):
    files,d,_=fixture(tmp_path);setup(monkeypatch,tmp_path)
    def missing(args,**kw):(Path(args[args.index('--output')+1])/'frames/frames').mkdir(parents=True)
    monkeypatch.setattr(roi_layers,'run',missing)
    with pytest.raises(PipelineError) as exc:roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True))
    assert exc.value.code=='ROI_INCOMPLETE'
    assert json.loads((tmp_path/'roi-report.json').read_text())['任务'][0]['错误码']=='ROI_INCOMPLETE'
    assert (tmp_path/'roi-0001-attempt-00/frames').exists() and not (tmp_path/'composite').exists()


@pytest.mark.parametrize('recover',[True,False])
def test_seam_retry_is_bounded_and_failed_attempt_never_commits(tmp_path,monkeypatch,recover):
    files,d,mask=fixture(tmp_path);setup(monkeypatch,tmp_path)
    monkeypatch.setenv('HLS_WINDOW_FRAMES','5');monkeypatch.setenv('HLS_OVERLAP_FRAMES','2')
    calls=[];snapshots=[];logs=[]
    def model(args,**kw):
        frames=Path(args[args.index('--video')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        if calls:
            state=[cv2.imread(str(tmp_path/'roi-layer-00'/p.name)).copy() for p in files]
            if snapshots:
                assert all(np.array_equal(a,b) for a,b in zip(state,snapshots[0]))
            snapshots.append(state)
        call=len(calls);calls.append(args)
        for p in sorted(frames.glob('*.png')):
            image=cv2.imread(str(p));image[cv2.imread(str(Path(args[args.index('--mask')+1])/p.name),0)>0]+=20 if call==0 or (recover and call==2) else 80
            cv2.imwrite(str(out/p.name),image)
    monkeypatch.setattr(roi_layers,'run',model)
    if recover:
        result=roi_inpaint.execute(tmp_path,files,d,log=logs.append,cfg=roi_inpaint.ROISettings(sequential_layers=True))
        assert result['任务'][1]['通过'] and len(calls)==3
    else:
        with pytest.raises(PipelineError) as exc:
            roi_inpaint.execute(tmp_path,files,d,log=logs.append,cfg=roi_inpaint.ROISettings(sequential_layers=True))
        assert exc.value.code=='SEAM_UNCERTAIN' and len(calls)==5
        assert not (tmp_path/'composite').exists()
    report=json.loads((tmp_path/'roi-report.json').read_text());trials=report['任务'][1]['尝试']
    assert trials[0]['重叠最大差异']==60 and not trials[0]['通过']
    assert trials[-1]['通过']==recover
    assert all('耗时秒' in t and 'subvideo_length' in t for t in trials)
    assert any('接缝差异=60.0' in line and '通过=False' in line for line in logs)


def test_seam_variants_stay_inside_scene_and_limit_retries():
    from worker.hls_settings import HLSSettings
    cfg=roi_inpaint.ROISettings()
    variants=roi_layers.seam_variants(40,76,20,80,HLSSettings(),cfg)
    assert variants==[(40,76,36),(32,76,36),(24,80,36),(32,80,18)]
    with pytest.raises(ValueError):
        roi_layers.seam_variants(40,76,20,80,HLSSettings(),roi_inpaint.ROISettings(seam_retries=4))


def test_gpu_oom_only_retries_once_and_keeps_checkpoint(tmp_path,monkeypatch):
    files,d,_=fixture(tmp_path);setup(monkeypatch,tmp_path);calls=[]
    def oom(*args,**kwargs):
        calls.append(kwargs['timeout']);raise PipelineError('GPU_OOM','合成显存不足')
    monkeypatch.setattr(roi_layers,'run',oom)
    with pytest.raises(PipelineError) as exc:
        roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True))
    assert exc.value.code=='GPU_OOM' and len(calls)==2
    report=json.loads((tmp_path/'roi-report.json').read_text())
    assert len(report['任务'][0]['尝试'])==2 and not report['任务'][0]['通过']


def test_clean_reference_rejects_other_watermark_layers(tmp_path,monkeypatch):
    files,d,mask=fixture(tmp_path);setup(monkeypatch,tmp_path)
    joint=mask.copy();joint[:4,:4]=255
    for p in files:cv2.imwrite(str(tmp_path/'masks'/p.name),joint)
    captured=[]
    class CheckReferences:
        def __init__(self,images,masks):captured.extend(masks)
        def restore(self,original,mask,repair,**kwargs):return repair
    monkeypatch.setattr(roi_layers,'CleanReferences',CheckReferences)
    def model(args,**kw):
        frames=Path(args[args.index('--video')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        for p in frames.glob('*.png'):cv2.imwrite(str(out/p.name),cv2.imread(str(p)))
    monkeypatch.setattr(roi_layers,'run',model)
    # 本测试只观察参考禁区；全局覆盖会因为故意没有施工的另一层而拒绝。
    with pytest.raises(PipelineError) as exc:roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True,minimum_context=96))
    assert exc.value.code=='ROI_INCOMPLETE'
    assert captured and all(m[:4,:4].all() for m in captured)


def test_animation_plan_changes_outer_window_only(tmp_path,monkeypatch):
    from dataclasses import asdict
    from worker.watermark_groups import AnimationSettings
    files,d,_=fixture(tmp_path);setup(monkeypatch,tmp_path)
    monkeypatch.setenv('HLS_WINDOW_FRAMES','5');monkeypatch.setenv('HLS_OVERLAP_FRAMES','2')
    (tmp_path/'animation-events.json').write_text(json.dumps(dict(settings=asdict(AnimationSettings()),layers={'0':[dict(interval=[0,6])]})))
    calls=[]
    def model(args,**kwargs):
        frames=Path(args[args.index('--video')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        assert args[args.index('--subvideo_length')+1]=='5'
        calls.append(len(list(frames.glob('*.png'))))
        for p in frames.glob('*.png'):cv2.imwrite(str(out/p.name),cv2.imread(str(p)))
    monkeypatch.setattr(roi_layers,'run',model)
    result=roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True))
    assert calls==[6] and result['覆盖核验通过']
