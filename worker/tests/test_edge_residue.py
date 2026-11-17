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
    assert measure(x,output,m|a,a,EdgeSettings())['通过']
    source_check=measure(x,x,m|a,a,EdgeSettings())
    assert not source_check['通过'] and source_check['判定']=='RESIDUE_FOUND'
    assert source_check['遮罩内确认残留']>0 and source_check['收边带确认残留']>0
    with pytest.raises(PipelineError) as exc:measure(x,output,m,a,EdgeSettings())
    assert exc.value.code=='RESIDUE_INPUT_INVALID'
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


@pytest.mark.parametrize('protect_ring,expected_code',[(False,'RESIDUE_FOUND'),(True,'RESIDUE_UNCERTAIN')])
def test_residue_errors_do_not_trigger_seam_retries_and_remain_unpassed(tmp_path,monkeypatch,protect_ring,expected_code):
    from worker import roi_inpaint,roi_spatial_layers
    from worker.tests.test_roi_inpaint import fixture
    files,d,mask=fixture(tmp_path)
    for p in files:
        x=cv2.imread(str(p));x[mask>0]=(220,160,30);cv2.imwrite(str(p),x)
    if protect_ring:
        (tmp_path/'scene-protection').mkdir()
        guard=np.where(mask>0,0,255).astype(np.uint8)
        for p in files:cv2.imwrite(str(tmp_path/'scene-protection'/p.name),guard)
    (tmp_path/'inpaint-policy.json').write_text(json.dumps(dict(continue_unpassed=True)))
    monkeypatch.setenv('INPAINT_PYTHON','python');monkeypatch.setenv('PROPAINTER_ROOT',str(tmp_path));calls=[]
    def model(args,**kw):
        calls.append(1);frames=Path(args[args.index('--video')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        for p in frames.glob('*.png'):cv2.imwrite(str(out/p.name),cv2.imread(str(p)))
    monkeypatch.setattr(roi_spatial_layers,'run',model)
    result=roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True,seam_retries=3))
    assert len(calls)==1 and result['未过块']==[1] and not result['全部ROI通过']
    assert [t['错误码'] for t in result['任务'][0]['尝试']]==[expected_code]
    assert result['任务'][0]['残留判定']==expected_code
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


def test_later_layer_residue_template_comes_from_frozen_source_not_prior_repair(tmp_path,monkeypatch):
    from worker import roi_inpaint,roi_spatial_layers
    from worker.tests.test_roi_inpaint import fixture
    files,d,mask=fixture(tmp_path)
    for p in files:
        x=cv2.imread(str(p));x[mask>0]=(220,160,30);cv2.imwrite(str(p),x)
    manifest=json.loads((tmp_path/'layers.json').read_text());manifest['layers']*=2
    (tmp_path/'layers.json').write_text(json.dumps(manifest));(tmp_path/'inpaint-policy.json').write_text(json.dumps(dict(continue_unpassed=True)))
    monkeypatch.setenv('INPAINT_PYTHON','python');monkeypatch.setenv('PROPAINTER_ROOT',str(tmp_path));calls=[]
    class References:
        def __init__(self,*args):pass
        def restore(self,x,m,y,**kwargs):return y
    monkeypatch.setattr(roi_spatial_layers,'CleanReferences',References)
    def model(args,**kw):
        frames=Path(args[args.index('--video')+1]);md=Path(args[args.index('--mask')+1]);dest=Path(args[args.index('--output')+1])/'frames/frames';dest.mkdir(parents=True)
        value=(50,50,50) if not calls else (220,160,30);calls.append(1)
        for p in frames.glob('*.png'):
            x=cv2.imread(str(p));m=cv2.imread(str(md/p.name),0);x[m>0]=value;cv2.imwrite(str(dest/p.name),x)
    monkeypatch.setattr(roi_spatial_layers,'run',model)
    result=roi_inpaint.execute(tmp_path,files,d,cfg=roi_inpaint.ROISettings(sequential_layers=True))
    assert result['任务'][0]['通过'] and not result['任务'][1]['通过']
    assert result['任务'][1]['尝试'][0]['错误码']=='RESIDUE_FOUND'
    assert result['任务'][1]['尝试'][0]['主色残留'][0]['遮罩内确认残留']>0
    assert result['任务'][1]['尝试'][0]['主色残留'][0]['遮罩内残留']>0


def _hsv_bgr(h,s,v,shape=(96,120)):
    hsv=np.zeros((shape[0],shape[1],3),np.uint8)
    hsv[...,0]=h;hsv[...,1]=s;hsv[...,2]=v
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


def test_color_template_flat_plateau_center():
    from worker.edge_residue import color_template,color_peaks
    cfg=EdgeSettings()
    mask=np.zeros((96,120),np.uint8);mask[20:40,20:40]=255
    for hue in [100,2,178]:
        img=_hsv_bgr(hue,200,200)
        assert color_template(img,mask,cfg)==hue
    # 大面积橙色背景 + 小面积蓝色水印，多峰仍能把蓝色水印峰拆出。
    mixed=_hsv_bgr(15,200,200)
    mixed[20:28,20:28]=_hsv_bgr(100,200,200,(8,8))
    peaks=color_peaks(mixed,mask,cfg)
    assert peaks[0]==15 and any(abs(p-100)<=2 for p in peaks[1:])


def test_measure_temporal_proof_matches_hue_without_exact_rgb_equality():
    cfg=EdgeSettings()
    source=np.full((96,120,3),40,np.uint8)
    mask=np.zeros((96,120),np.uint8);mask[30:50,40:60]=255
    band=np.zeros_like(mask)
    source[mask>0]=_hsv_bgr(100,200,200)[mask>0]
    painted=np.full((96,120,3),40,np.uint8)
    painted[mask>0]=_hsv_bgr(102,150,180)[mask>0]
    bg=np.full((96,120,3),40,np.uint8)
    bg[mask>0]=_hsv_bgr(99,180,220)[mask>0]
    assert not np.array_equal(painted[mask>0],bg[mask>0])
    evidence=dict(background=bg,confidence=(mask>0).astype(np.uint8),counts=np.where(mask>0,3,0).astype(np.uint8))
    ok=measure(source,painted,mask,band,cfg,reference_evidence=evidence)
    assert ok['通过'] and ok['判定']=='通过' and ok['时域证实背景本色']==400 and ok['确认残留']==0
    weak=dict(background=bg,confidence=(mask>0).astype(np.uint8),counts=np.where(mask>0,2,0).astype(np.uint8))
    refuted=measure(source,painted,mask,band,cfg,reference_evidence=weak)
    assert not refuted['通过'] and refuted['判定']=='RESIDUE_FOUND'


def test_measure_outer_ring_proves_background_color():
    cfg=EdgeSettings()
    blue=_hsv_bgr(100,200,200)
    mask=np.zeros((96,120),np.uint8);mask[30:50,40:60]=255
    band=np.zeros_like(mask)
    # 源图组内全蓝，背景仅上方 y<34 为蓝色天空、其余为灰：整环蓝密度<50%故蓝色仍为区别色，
    # 而补画仅贴近上缘的候选区域在局部外环窗口内有>=50%蓝色可被外环证实。
    source=np.full((96,120,3),40,np.uint8)
    source[:34,:]=blue[:34,:]
    source[mask>0]=blue[mask>0]
    painted=np.full((96,120,3),40,np.uint8)
    painted[30:32,42:58]=blue[30:32,42:58]
    ok=measure(source,painted,mask,band,cfg)
    assert ok['通过'] and ok['判定']=='通过' and ok['外环证实背景本色']==32 and ok['背景本色证实']==32 and ok['确认残留']==0
    guard=np.where(mask>0,0,255).astype(np.uint8)
    blocked=measure(source,painted,mask,band,cfg,protected=guard)
    assert not blocked['通过'] and blocked['判定']=='RESIDUE_UNCERTAIN' and blocked['外环证实背景本色']==0


def test_measure_adjacent_halo_cannot_prove_background():
    cfg=EdgeSettings()
    source=np.full((96,120,3),40,np.uint8)
    mask=np.zeros((96,120),np.uint8);mask[30:50,40:60]=255
    halo=np.zeros_like(mask);halo[28:52,38:62]=255
    blue=_hsv_bgr(100,200,200)
    source[halo>0]=blue[halo>0]
    painted=source.copy()
    res=measure(source,painted,mask,np.zeros_like(mask),cfg)
    assert not res['通过'] and res['判定']=='RESIDUE_FOUND' and res['外环证实背景本色']==0 and res['确认残留']==400
    # 相邻已确认水印并集若占满外环区域，传入 confirmed 后不得被当作外环背景证据。
    confirmed=mask.copy();confirmed[30:50,60:72]=255
    source_adj=np.full((96,120,3),40,np.uint8)
    source_adj[confirmed>0]=blue[confirmed>0]
    res_adj=measure(source_adj,source_adj.copy(),mask,np.zeros_like(mask),cfg,confirmed=confirmed)
    assert not res_adj['通过'] and res_adj['判定']=='RESIDUE_FOUND' and res_adj['外环证实背景本色']==0


def test_measure_real_residue_returns_residue_found():
    cfg=EdgeSettings()
    source=np.full((96,120,3),40,np.uint8)
    core=np.zeros((96,120),np.uint8);core[30:50,40:60]=255
    band=np.zeros_like(core);band[29:30,40:60]=255
    mask=core|band
    source[mask>0]=_hsv_bgr(100,200,200)[mask>0]
    painted=np.full((96,120,3),40,np.uint8)
    painted[mask>0]=_hsv_bgr(100,75,85)[mask>0]
    res=measure(source,painted,mask,band,cfg)
    assert not res['通过'] and res['判定']=='RESIDUE_FOUND'
    assert res['遮罩内确认残留']==400 and res['收边带确认残留']==20


def test_measure_untemplated_and_unverifiable_return_uncertain():
    cfg=EdgeSettings()
    # 组内彩色样本 >=10% 但 <32 像素，无有效主色模板 -> 待目检。
    source=np.full((96,120,3),40,np.uint8)
    mask=np.zeros((96,120),np.uint8);mask[30:40,40:50]=255  # 100 px
    source[32:36,42:47]=_hsv_bgr(100,200,200,(4,5))  # 20 px (>=10%, <32)
    untemplated=measure(source,np.full_like(source,40),mask,np.zeros_like(mask),cfg)
    assert not untemplated['通过'] and untemplated['判定']=='RESIDUE_UNCERTAIN' and untemplated['无模板组数']==1
    # 有合法主色且补画命中候选，但无时域证据且外环被保护区遮挡 -> 待目检。
    mask2=np.zeros((96,120),np.uint8);mask2[30:50,40:60]=255
    src2=np.full((96,120,3),40,np.uint8);src2[mask2>0]=_hsv_bgr(100,200,200)[mask2>0]
    guard=np.where(mask2>0,0,255).astype(np.uint8)
    pending=measure(src2,src2.copy(),mask2,np.zeros_like(mask2),cfg,protected=guard)
    assert not pending['通过'] and pending['判定']=='RESIDUE_UNCERTAIN' and pending['待目检']==400


@pytest.mark.parametrize('kwargs',[
    dict(template_peaks=2),
    dict(template_peaks=7),
    dict(background_ring_gap=2),
    dict(background_ring_gap=7),
    dict(background_ring_window=7),
    dict(background_ring_window=17),
    dict(background_ring_minimum_pixels=31),
    dict(background_ring_minimum_pixels=2000),
    dict(background_proof_fraction=.49),
    dict(background_refute_fraction=.5),
    dict(background_refute_fraction=.3),
    dict(background_refute_fraction=-.01),
    dict(maximum_residue_pixels=1),
])
def test_edge_settings_reject_relaxed_parameters(kwargs):
    with pytest.raises(ValueError):
        EdgeSettings(**kwargs)


def test_roi_resume_recheck_aligns_with_measure_confirmed_and_protected(tmp_path):
    from worker.roi_resume import prepare,recheck
    from worker.hls_settings import HLSSettings
    from worker.spatial_roi import SpatialSettings
    from worker.roi_inpaint import ROISettings
    old_parent=tmp_path/'old_wrap';work_parent=tmp_path/'work_wrap'
    old=old_parent/'run';work=work_parent/'run'
    for sub in ['frames','masks','edge-band','scene-protection','roi-layer-00','roi-coverage']:
        (old/sub).mkdir(parents=True)
    for sub in ['frames','masks','edge-band','scene-protection']:
        (work/sub).mkdir(parents=True)
    h,w=96,120;mask=np.zeros((h,w),np.uint8);mask[30:50,40:60]=255
    band=np.zeros((h,w),np.uint8);band[30:32,40:60]=255
    blue=_hsv_bgr(100,200,200,(h,w))
    files=[]
    for i in range(4):
        name=f'{i:06d}.png'
        src=np.full((h,w,3),40,np.uint8);src[mask>0]=blue[mask>0]
        cv2.imwrite(str(work/'frames'/name),src);files.append(work/'frames'/name)
        cv2.imwrite(str(work/'masks'/name),mask);cv2.imwrite(str(work/'edge-band'/name),band)
        cv2.imwrite(str(old/'roi-layer-00'/name),src)  # 仍留有源蓝色残留
        cv2.imwrite(str(old/'roi-coverage'/name),mask)
    box=[0,0,w,h]
    shared_meta={'layers':[{'directory':str(work/'masks')}]}
    for root in [old,work]:
        (root/'layers.json').write_text(json.dumps(shared_meta))
        (root/'detections.json').write_text(json.dumps(dict(scenes=[[0,4]])))
        (root/'spatial-roi.json').write_text(json.dumps(asdict(SpatialSettings(minimum_width=64,minimum_height=64))))
        (root/'trusted-context.json').write_text(json.dumps(asdict(ContextSettings())))
    old_row=dict(任务=1,图层=0,开始帧=0,结束帧=4,ROI=box,施工面积像素=int(np.count_nonzero(mask)),通过=False,采用未过结果=True,最佳尝试=0,耗时秒=1.0,
                 尝试=[dict(尝试=0,重叠最大差异=0.0,通过=False)])
    (old/'roi-report.json').write_text(json.dumps(dict(任务=[old_row])))
    cfg=ROISettings(sequential_layers=True);settings=HLSSettings();spatial=SpatialSettings(minimum_width=64,minimum_height=64)
    profile=dict(ROI=asdict(cfg),窗口=settings.window_frames,重叠=settings.overlap_frames,接缝上限=settings.overlap_mae_limit,每层面积上限=settings.mask_fraction_limit)
    for parent in [old_parent,work_parent]:
        (parent/'profile.json').write_text(json.dumps(profile))
        (parent/'input-binding.json').write_text(json.dumps(dict(frames=4)))
    state=prepare(work,old,files,cfg,settings,spatial,shared_meta['layers'],lambda *_:None)
    row=dict(任务=1,图层=0,开始帧=0,结束帧=4,ROI=box,施工面积像素=int(np.count_nonzero(mask)),通过=False)
    recheck(state,row,mask,work,files,np.array([0,1,2,3]),box,0,4,EdgeSettings(),settings,lambda *_:None)
    assert not row['通过'] and row['错误码']=='RESIDUE_FOUND' and row['残留判定']=='RESIDUE_FOUND'
    # 若保护区覆盖外环且无时域参考，则应从 RESIDUE_FOUND 变成 RESIDUE_UNCERTAIN（待目检）。
    guard=np.where(mask>0,0,255).astype(np.uint8)
    for p in files:cv2.imwrite(str(work/'scene-protection'/p.name),guard)
    row_guard=dict(任务=1,图层=0,开始帧=0,结束帧=4,ROI=box,施工面积像素=int(np.count_nonzero(mask)),通过=False)
    recheck(state,row_guard,mask,work,files,np.array([0,1,2,3]),box,0,4,EdgeSettings(),settings,lambda *_:None)
    assert not row_guard['通过'] and row_guard['错误码']=='RESIDUE_UNCERTAIN' and row_guard['残留判定']=='RESIDUE_UNCERTAIN'


def test_temporal_refute_overrides_ring_proof():
    # 区别色候选所在局部外环同色但可信时域参考显示该处背景不是主色：不得用外环证实抵消，必须判确认残留。
    blue=_hsv_bgr(100,200,200)
    m=np.zeros((96,120),np.uint8);m[30:50,40:60]=255
    band=np.zeros_like(m)
    source=np.full((96,120,3),40,np.uint8)
    source[:34,:]=blue[:34,:]
    source[m>0]=blue[m>0]
    painted=np.full((96,120,3),40,np.uint8)
    painted[30:32,42:58]=blue[30:32,42:58]
    evidence=dict(background=np.full_like(source,30),confidence=np.full(m.shape,255,np.uint8),counts=np.full(m.shape,3,np.uint8))
    result=measure(source,painted,m,band,EdgeSettings(),reference_evidence=evidence)
    assert result['判定']=='RESIDUE_FOUND' and not result['通过'] and result['外环证实但时域反证']==32
    # 无时域证据时，同样的局部外环同色可证实背景本色。
    ok=measure(source,painted,m,band,EdgeSettings())
    assert ok['通过'] and ok['外环证实背景本色']==32


def test_same_color_background_peak_produces_no_candidates_and_counts_separately():
    cfg=EdgeSettings()
    skin=_hsv_bgr(10,180,180)
    blue=_hsv_bgr(100,200,200)
    mask=np.zeros((96,120),np.uint8);mask[30:50,40:60]=255  # 400 px
    band=np.zeros_like(mask)
    # 1) 整组与外环同色（肤色/棕色）：不产生候选也不判残留，单列为同色不适用。
    res=measure(skin,skin.copy(),mask,band,cfg)
    assert res['通过'] and res['判定']=='通过'
    assert res['颜色候选']==0 and res['区别色候选数']==0 and res['确认残留']==0 and res['待目检']==0
    assert res['同色不适用组数']==1 and res['同色不适用施工像素']==400
    g=res['组'][0]
    assert g['状态']=='颜色复检不适用：水印源色与周围背景同色'
    assert g['区别色列表']==[] and g['非区别色列表']==[10]
    assert g['主色模板对比'][0]['是否区别色'] is False and '非区别色' in g['主色模板对比'][0]['原因']

    # 2) 组内高密度背景棕色（~0.75）+ 外环高密度棕色（~0.75）+ 局部蓝色区别色水印（~0.25，外环0.0）：
    #    棕色峰必须被跳过，仅蓝峰作为区别色参与候选判定。
    mixed_src=skin.copy()
    mixed_src[30:35,40:60]=blue[30:35,40:60]  # 100/400 = 0.25 >= 0.10 且 >= 32 px
    painted_clean=skin.copy()  # 蓝色被补净成肤色，仅留背景棕色
    clean_res=measure(mixed_src,painted_clean,mask,band,cfg)
    assert clean_res['通过'] and clean_res['判定']=='通过'
    assert clean_res['颜色候选']==0 and clean_res['区别色候选数']==0 and clean_res['同色不适用组数']==0
    mg=clean_res['组'][0]
    assert mg['区别色列表']==[100] and mg['非区别色列表']==[10]
    by_hue={item['色相']:item for item in mg['主色模板对比']}
    assert by_hue[10]['是否区别色'] is False and '非区别色' in by_hue[10]['原因']
    assert by_hue[100]['是否区别色'] is True and by_hue[100]['原因']=='水印区别色'
    # 若蓝色未补干净，仅蓝色像素进入区别色候选并判 RESIDUE_FOUND，棕色背景像素不计入。
    dirty_res=measure(mixed_src,mixed_src.copy(),mask,band,cfg)
    assert not dirty_res['通过'] and dirty_res['判定']=='RESIDUE_FOUND'
    assert dirty_res['颜色候选']==100 and dirty_res['区别色候选数']==100 and dirty_res['确认残留']==100


def test_distinguishing_color_residue_found_and_clean_pass():
    cfg=EdgeSettings()
    blue=_hsv_bgr(100,200,200)
    mask=np.zeros((96,120),np.uint8);mask[30:50,40:60]=255
    band=np.zeros_like(mask);band[29:30,40:60]=255
    work=mask|band  # 420 px
    source=np.full((96,120,3),40,np.uint8)
    source[work>0]=blue[work>0]
    dirty=measure(source,source.copy(),work,band,cfg)
    assert not dirty['通过'] and dirty['判定']=='RESIDUE_FOUND'
    assert dirty['颜色候选']==420 and dirty['区别色候选数']==420 and dirty['确认残留']==420
    assert dirty['同色不适用组数']==0 and dirty['同色不适用施工像素']==0
    assert dirty['组'][0]['区别色列表']==[100] and dirty['组'][0]['非区别色列表']==[]
    clean=measure(source,np.full_like(source,40),work,band,cfg)
    assert clean['通过'] and clean['判定']=='通过'
    assert clean['颜色候选']==0 and clean['区别色候选数']==0 and clean['确认残留']==0


def test_insufficient_outer_ring_for_non_distinguishing_peak_returns_uncertain():
    cfg=EdgeSettings()
    skin=_hsv_bgr(10,180,180)
    mask=np.zeros((96,120),np.uint8);mask[30:50,40:60]=255  # 400 px
    band=np.zeros_like(mask)
    # 1) 外环完全被保护区遮住（外环像素=0）
    guard_all=np.where(mask>0,0,255).astype(np.uint8)
    empty_ring=measure(skin,skin.copy(),mask,band,cfg,protected=guard_all)
    assert not empty_ring['通过'] and empty_ring['判定']=='RESIDUE_UNCERTAIN'
    assert empty_ring['同色不适用组数']==0 and empty_ring['待目检']==400
    assert empty_ring['组'][0]['状态']=='待目检：外环不足无法证明主色区别度'
    assert empty_ring['组'][0]['主色模板对比'][0]['是否区别色'] is False

    # 2) 外环大部分被遮挡，仅留少量同色外环像素（0 < 外环像素 < background_ring_minimum_pixels=32）：
    #    不得判为“同色不适用（通过）”，必须判待目检（不通过）。
    guard_partial=np.where(mask>0,0,255).astype(np.uint8)
    guard_partial[23:27,45:49]=0  # 距 mask 距离在 [3, 7]，仅 16 px < 32
    partial_ring=measure(skin,skin.copy(),mask,band,cfg,protected=guard_partial)
    assert 0<partial_ring['组'][0]['外环像素']<cfg.background_ring_minimum_pixels
    assert not partial_ring['通过'] and partial_ring['判定']=='RESIDUE_UNCERTAIN'
    assert partial_ring['同色不适用组数']==0 and partial_ring['同色不适用施工像素']==0
    assert partial_ring['待目检']==400
    assert partial_ring['组'][0]['状态']=='待目检：外环不足无法证明主色区别度'


def test_same_color_group_counted_independently_with_other_groups_and_summary():
    from worker.roi_spatial_layers import residue_summary,residue_error
    cfg=EdgeSettings()
    skin=_hsv_bgr(10,180,180)
    blue=_hsv_bgr(100,200,200)
    m1=np.zeros((96,120),np.uint8);m1[15:35,15:35]=255  # 400 px 同色背景组
    m2=np.zeros((96,120),np.uint8);m2[55:70,70:85]=255  # 225 px 区别色蓝水印组
    mask=m1|m2;band=np.zeros_like(mask)
    source=skin.copy();source[m2>0]=blue[m2>0]
    painted_ok=skin.copy()  # 蓝水印清干净
    ok=measure(source,painted_ok,mask,band,cfg)
    assert ok['通过'] and ok['判定']=='通过' and ok['组数']==2
    assert ok['同色不适用组数']==1 and ok['同色不适用施工像素']==400
    assert ok['颜色候选']==0 and ok['区别色候选数']==0

    painted_bad=source.copy()  # 蓝水印未清
    bad=measure(source,painted_bad,mask,band,cfg)
    assert not bad['通过'] and bad['判定']=='RESIDUE_FOUND' and bad['组数']==2
    assert bad['同色不适用组数']==1 and bad['同色不适用施工像素']==400
    assert bad['颜色候选']==225 and bad['区别色候选数']==225 and bad['确认残留']==225

    summary_ok=residue_summary([dict(frame=0,**ok),dict(frame=1,**ok)])
    assert summary_ok['判定']=='通过' and summary_ok['同色不适用组数']==2 and summary_ok['同色不适用施工像素']==800
    assert summary_ok['区别色候选数']==0 and residue_error([dict(frame=0,**ok)]) is None
    summary_mix=residue_summary([dict(frame=0,**ok),dict(frame=1,**bad)])
    assert summary_mix['判定']=='RESIDUE_FOUND' and summary_mix['同色不适用组数']==2 and summary_mix['同色不适用施工像素']==800
    assert summary_mix['区别色候选数']==225 and summary_mix['确认残留']==225
