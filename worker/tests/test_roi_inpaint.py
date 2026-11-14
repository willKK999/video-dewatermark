"""ROI必须保留原生几何、掩膜外像素与原片缓存；失败禁止用源帧补齐。"""
from pathlib import Path
import json,hashlib
import cv2,numpy as np,pytest
from worker import roi_inpaint as roi
from worker.model_tracks import adaptive_mask
from worker.process import PipelineError


def fixture(tmp_path):
    for name in ['frames','masks','layer-00/masks','validated-objects/1']:(tmp_path/name).mkdir(parents=True)
    files=[];mask=np.zeros((240,320),np.uint8);mask[35:50,40:70]=255
    for i in range(6):
        p=tmp_path/'frames'/f'{i:06d}.png';image=np.full((240,320,3),20+i,np.uint8);cv2.imwrite(str(p),image);files.append(p)
        for folder in ['masks','layer-00/masks','validated-objects/1']:cv2.imwrite(str(tmp_path/folder/p.name),mask)
    (tmp_path/'layers.json').write_text(json.dumps(dict(layers=[dict(directory=str(tmp_path/'layer-00/masks'))])))
    d=dict(scenes=[[0,6]],tracks=[dict(id=1,hints=[dict(frame=i) for i in range(6)])])
    return files,d,mask


def fake_model(args,*a,**kw):
    frames=Path(args[args.index('--video')+1]);masks=Path(args[args.index('--mask')+1]);dest=Path(args[args.index('--output')+1])/'frames/frames';dest.mkdir(parents=True)
    assert '--fp16' in args
    for p in sorted(frames.glob('*.png')):
        im=cv2.imread(str(p));mask=cv2.imread(str(masks/p.name),0);im[mask>0]=130;cv2.imwrite(str(dest/p.name),im)


def test_roi_paste_keeps_geometry_background_and_source_files(tmp_path,monkeypatch):
    files,d,mask=fixture(tmp_path);before=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    monkeypatch.setenv('INPAINT_PYTHON','python');monkeypatch.setenv('PROPAINTER_ROOT',str(tmp_path));monkeypatch.delenv('PROPAINTER_FP16',raising=False)
    monkeypatch.setattr(roi,'run',fake_model)
    cfg=roi.ROISettings(maximum_frames=5,minimum_frames=3,minimum_context=16,overlap_frames=1)
    result=roi.execute(tmp_path,files,d,cfg=cfg)
    assert result['覆盖核验通过'] and result['原帧回退数']==0
    assert result['模型像素帧']<result['原整帧六层像素帧']/10
    for p,h in zip(files,before):
        a=cv2.imread(str(p));b=cv2.imread(str(tmp_path/'composite'/p.name))
        assert a.shape==b.shape and np.array_equal(a[mask==0],b[mask==0])
        assert np.any(a[mask>0]!=b[mask>0]) and hashlib.sha256(p.read_bytes()).hexdigest()==h


def test_incomplete_roi_output_reports_failure_instead_of_original_fallback(tmp_path,monkeypatch):
    files,d,mask=fixture(tmp_path)
    monkeypatch.setenv('INPAINT_PYTHON','python');monkeypatch.setenv('PROPAINTER_ROOT',str(tmp_path))
    def missing(args,*a,**kw):fake_model(args);next((Path(args[args.index('--output')+1])/'frames/frames').glob('*.png')).unlink()
    monkeypatch.setattr(roi,'run',missing)
    with pytest.raises(PipelineError) as caught:roi.execute(tmp_path,files,d,cfg=roi.ROISettings(maximum_frames=6,minimum_frames=3,minimum_context=16,overlap_frames=1))
    assert caught.value.code=='ROI_INCOMPLETE'
    with pytest.raises(PipelineError):roi.assert_coverage(tmp_path,files)


def test_roi_padding_never_rescales_native_pixels():
    image=np.arange(31*47*3,dtype=np.uint16).reshape(31,47,3)
    out=roi.crop(image,[3,4,46,30])
    assert out.shape[0]%8==0 and out.shape[1]%8==0
    assert np.array_equal(out[:26,:43],image[4:30,3:46])


def test_transparent_expansion_exceeds_opaque_without_moving_coordinates():
    mask=np.zeros((720,1280),np.uint8);mask[300:310,500:520]=255;base=np.full((10,20),255,np.uint8)
    semi,policy=adaptive_mask(mask,base,np.full(base.shape,.2,np.float32),[500,300,520,310],.9,mask.shape)
    opaque,_=adaptive_mask(mask,base,np.full(base.shape,.9,np.float32),[500,300,520,310],.9,mask.shape)
    assert policy['半透明外扩像素']>=12 and policy['不透明外扩像素']==3
    assert np.count_nonzero(semi)>np.count_nonzero(opaque) and np.all(semi[mask>0]==255)
