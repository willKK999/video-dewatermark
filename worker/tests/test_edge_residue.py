"""主色、形状和保护区的合成回归；不使用影片或坐标规则。"""
from dataclasses import asdict
import json
from pathlib import Path
import cv2,numpy as np,pytest
from worker.edge_residue import EdgeSettings,refine_group,composite,measure,highlight_guard
from worker.trusted_context import ContextSettings,verify_from_confirmed,overlap_weight
from worker.process import PipelineError


def sample():
    x=np.full((96,120,3),30,np.uint8);m=np.zeros(x.shape[:2],np.uint8)
    cv2.circle(m,(40,40),14,255,-1)
    colored=cv2.dilate(m,np.ones((3,3),np.uint8))>0;x[colored]=(220,160,30)
    return x,m


def test_own_shape_never_imports_distant_same_color_or_highlight_edges():
    x,m=sample();x[:,90:94]=(220,160,30);x[24:55,54:59]=255
    guard=np.zeros_like(m);guard[39:44,20:28]=255
    added,row=refine_group(x,m,m,guard,EdgeSettings())
    assert added.any() and not added[:,90:].any() and not np.any(added&guard)
    assert not np.any((added>0)&highlight_guard(x)) and row['保护交集像素']==0
    # 白衣、栏杆、公告纸的低饱和亮边，即使邻接确认组也不能被收边盖住。
    for shape in ['cloth','rail','paper']:
        y=x.copy()
        if shape=='cloth':cv2.ellipse(y,(55,40),(5,18),0,0,360,(245,245,245),-1)
        elif shape=='rail':cv2.line(y,(53,16),(53,65),(255,255,255),2)
        else:cv2.rectangle(y,(52,25),(70,53),(230,230,230),-1)
        a,_=refine_group(y,m,m,guard,EdgeSettings());assert not np.any((a>0)&(y.min(axis=2)>180))


def test_confirmed_edge_does_not_mix_source_watermark_and_reports_both_domains():
    x,m=sample();repair=np.full_like(x,30);a,_=refine_group(x,m,m,np.zeros_like(m),EdgeSettings())
    output=composite(x,repair,m|a)
    assert np.array_equal(output[(m|a)==0],x[(m|a)==0])
    assert measure(x,output,m,a,EdgeSettings())['通过']
    assert not measure(x,x,m,a,EdgeSettings())['通过']
    with pytest.raises(ValueError):EdgeSettings(maximum_residue_pixels=1)


def test_independent_expected_reloads_union_and_checks_context_crop_padding(tmp_path):
    _,m=sample();p=tmp_path/'union.png';cv2.imwrite(str(p),m)
    from worker.roi_inpaint import crop
    actual=crop(m,[7,8,100,90],True)
    with pytest.raises(PipelineError) as exc:verify_from_confirmed(p,[7,8,100,90],actual,actual,m.size,ContextSettings())
    assert exc.value.code=='UNTRUSTED_CONTEXT'
    with pytest.raises(PipelineError) as exc:verify_from_confirmed(tmp_path/'missing.png',[0,0,8,8],actual,actual,m.size,ContextSettings())
    assert exc.value.code=='MASK_INVALID'


def test_nonprefix_overlap_weights_are_convex():
    weights=[overlap_weight(j,10,20,{15,16,17}) for j in [15,16,17]]
    assert weights==[.25,.5,.75] and overlap_weight(19,10,20,{15,16,17})==1


def test_comparison_mask_shape_is_three_channel_video_not_four_dimensions(tmp_path):
    from scripts.compare_blue_residue import mask_shape
    p=tmp_path/'mask.png';cv2.imwrite(str(p),np.zeros((10,20),np.uint8))
    assert mask_shape([p])+(3,)==(10,20,3)
    with pytest.raises(ValueError):mask_shape([])


@pytest.mark.parametrize('path',['objects','layers','spatial','hls'])
def test_unsafe_legacy_or_missing_protection_fails_before_model(tmp_path,path):
    from worker import roi_inpaint,vision
    from worker.tests.test_roi_inpaint import fixture
    files,d,_=fixture(tmp_path)
    if path=='hls':
        with pytest.raises(PipelineError) as exc:vision.inpaint(tmp_path,files,d['scenes'],'propainter')
    else:
        if path=='layers':(tmp_path/'spatial-roi.json').unlink()
        if path=='spatial':(tmp_path/'trusted-context.json').unlink()
        with pytest.raises(PipelineError) as exc:roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=path!='objects'))
    assert exc.value.code=='UNTRUSTED_CONTEXT' and not (tmp_path/'composite').exists()


def test_flow_source_points_and_error_ring_exclude_other_layers(monkeypatch):
    from worker.restoration import CleanReferences,context_mask
    x,m=sample();confirmed=m.copy();confirmed[40:48,56:67]=255
    refs=CleanReferences([x]*3,[confirmed]*3);captured=[]
    def points(gray,*args,**kw):captured.append(kw['mask']);return None
    monkeypatch.setattr(cv2,'goodFeaturesToTrack',points)
    refs.restore(x,m,x,confirmed=confirmed)
    assert not captured[0][context_mask(confirmed,5)>0].any()


def test_wrong_exclusion_builder_cannot_fool_independent_audit(tmp_path,monkeypatch):
    from worker import roi_inpaint,roi_spatial_layers,trusted_context
    from worker.tests.test_roi_inpaint import fixture
    files,d,_=fixture(tmp_path)
    (tmp_path/'inpaint-policy.json').write_text(json.dumps(dict(continue_unpassed=True)))
    monkeypatch.setattr(trusted_context,'exclusion_mask',lambda m,cfg:np.zeros_like(m))
    monkeypatch.setattr(roi_spatial_layers,'run',lambda *a,**kw:pytest.fail('自检失败不得推理'))
    with pytest.raises(PipelineError) as exc:roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True))
    assert exc.value.code=='UNTRUSTED_CONTEXT'
    report=json.loads((tmp_path/'roi-report.json').read_text());assert len(report['任务'])==1


def test_residue_retries_all_v6_variants_and_remains_unpassed(tmp_path,monkeypatch):
    from worker import roi_inpaint,roi_spatial_layers
    from worker.tests.test_roi_inpaint import fixture
    files,d,mask=fixture(tmp_path)
    for p in files:
        x=cv2.imread(str(p));x[mask>0]=(220,160,30);cv2.imwrite(str(p),x)
    (tmp_path/'inpaint-policy.json').write_text(json.dumps(dict(continue_unpassed=True)))
    monkeypatch.setenv('INPAINT_PYTHON','python');monkeypatch.setenv('PROPAINTER_ROOT',str(tmp_path));calls=[]
    def model(args,**kw):
        calls.append(1);frames=Path(args[args.index('--video')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        for p in frames.glob('*.png'):cv2.imwrite(str(out/p.name),cv2.imread(str(p)))
    monkeypatch.setattr(roi_spatial_layers,'run',model)
    result=roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True,seam_retries=3))
    assert len(calls)==4 and result['未过块']==[1] and not result['全部ROI通过']
    assert all(t['错误码']=='RESIDUE_UNCERTAIN' for t in result['任务'][0]['尝试'])
    assert result['任务'][0]['采用未过结果'] and result['原帧回退数']==0


def test_stable_plan_retains_groups_and_protection_with_cross_device_copy(tmp_path,monkeypatch):
    from scripts.stable_layer_plan import execute
    from worker.hls_layers import write_manifest,VerifiedLayers
    from types import SimpleNamespace
    import errno
    base=tmp_path/'base';base.mkdir();(base/'frames').mkdir();(base/'masks').mkdir()
    for name in ['group-objects','scene-protection','subtitle-protection']:(base/name).mkdir()
    mask=np.zeros((80,80),np.uint8);mask[30:35,30:35]=255;p=base/'masks/000000.png';cv2.imwrite(str(p),mask)
    (base/'detections.json').write_text('{"hints":[]}');write_manifest(base,base/'detections.json',[base/'masks'],[p])
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(asdict(ContextSettings())))
    def exdev(*args):raise OSError(errno.EXDEV,'跨文件系统')
    monkeypatch.setattr(Path,'hardlink_to',exdev)
    out=tmp_path/'out';execute(SimpleNamespace(base=base,output=out,config=cfg))
    assert all((out/n).is_dir() for n in ['group-objects','scene-protection','subtitle-protection'])
    assert VerifiedLayers(out/'layers.json',out/'detections.json').validate(0,mask) is True


def test_background_color_under_gray_group_is_not_a_watermark_template():
    image=np.full((96,120,3),(25,95,140),np.uint8)
    m=np.zeros(image.shape[:2],np.uint8);m[25:60,30:70]=255
    # 无彩色半透明组覆盖着大面积棕色背景；不得用背景颜色长出整圈。
    image[30:55,35:65]=(90,115,140)
    added,row=refine_group(image,m,m,np.zeros_like(m),EdgeSettings())
    assert row['hue'] is None and not added.any()


def test_color_comparison_decodes_full_frames_and_rejects_empty_group_cache(tmp_path,monkeypatch):
    from scripts import compare_blue_residue as compare
    from types import SimpleNamespace
    import io
    base=tmp_path/'base';(base/'masks').mkdir(parents=True);(base/'group-objects/g').mkdir(parents=True)
    mask=np.zeros((40,48),np.uint8);mask[15:20,15:20]=255
    image=np.full((40,48,3),30,np.uint8);image[mask>0]=(230,150,30)
    for i in range(2):
        name=f'{i:06d}.png';cv2.imwrite(str(base/'masks'/name),mask);cv2.imwrite(str(base/'group-objects/g'/name),mask)
    class Decoder:
        def __init__(self,*a,**kw):self.stdout=io.BytesIO(image.tobytes()*2)
        def wait(self,**kw):return 0
        def poll(self):return 0
    monkeypatch.setattr(compare.subprocess,'Popen',Decoder)
    args=SimpleNamespace(masks=base,output=tmp_path/'out',source='source',reference='reference',video='video',fps=24,times='',quality=[],reference_label='before',video_label='after')
    compare.execute(args)
    result=json.loads((args.output/'蓝色残留统计.json').read_text())
    assert len(result['frames'])==2 and result['summary']['reference']['blue_inside']==50
    for p in (base/'group-objects/g').glob('*.png'):p.unlink()
    with pytest.raises(ValueError,match='缺少确认组'):compare.execute(args)
