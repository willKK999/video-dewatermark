"""逐帧施工约束与互斥分层，不能靠减少掩膜面积绕过水印覆盖。"""
import numpy as np,pytest
from worker.model_tracks import temporal_bins,disjoint_masks,prototype_score
from worker.process import PipelineError
from worker.multitrack import highpass


def test_nonconcurrent_fragments_share_layer_without_adding_maxima():
    areas={k:{k:.15,k+10:.15} for k in range(6)}
    assert len(temporal_bins(areas))==1
    assert len(temporal_bins({1:{0:.15},2:{0:.15}}))==2
    with pytest.raises(PipelineError):temporal_bins({1:{0:.201}})


def test_layer_pixels_are_repaired_once_and_union_is_identical():
    a=np.zeros((30,40),np.uint8);a[2:15,3:20]=255
    b=np.zeros_like(a);b[5:20,12:30]=255
    x,y=disjoint_masks([a,b])
    assert not np.any(x&y) and np.array_equal(x|y,a|b)
    assert np.array_equal(x,a)


def test_full_graphic_confirms_weak_text_anchor_without_searching_frame():
    rng=np.random.default_rng(34);image=rng.integers(0,255,(90,140,3),dtype=np.uint8)
    support=np.full((60,90),255,np.uint8)
    proto=dict(anchor=[50,45,90,55],box=[20,10,110,70],
               examples=[np.zeros((10,40),np.float32)],mask=support,
               whole_feature=highpass(image[10:70,20:110]))
    assert prototype_score(image,dict(box=[50,45,90,55]),proto)>.99
    assert prototype_score(image,dict(box=[70,65,110,75]),proto)<.55


def test_confirmed_multiframe_logo_can_identify_text_without_treating_all_text_as_watermarks(tmp_path):
    import cv2
    from worker.model_tracks import learn_logo_prototypes,ModelTrackSettings,candidates_for_frame
    files=[];rows=[]
    for i in range(20):
        image=np.full((120,220,3),40+i,np.uint8)
        cv2.putText(image,'AB',(60+i,65),cv2.FONT_HERSHEY_SIMPLEX,.7,(115+i,115+i,115+i),2)
        p=tmp_path/f'{i:06d}.png';cv2.imwrite(str(p),image);files.append(p)
        rows.append(dict(frame=i,boxes=[dict(box=[55+i,44,92+i,70],confidence=.9,class_id=1)]))
    cfg=ModelTrackSettings();labels={1:'float_logo',2:'float_text'}
    proto=learn_logo_prototypes(files,rows,labels,[],cfg,30)
    assert proto and all(p['samples']>=cfg.minimum_template_frames for p in proto)
    image=cv2.imread(str(files[-1]));raw=dict(rows[-1]['boxes'][0],class_id=2)
    result=candidates_for_frame(image,dict(boxes=[raw]),proto,labels,cfg)
    assert result and result[0]['kind']=='prototype'
    assert candidates_for_frame(image,dict(boxes=[]),proto,labels,cfg)==[]


def test_opaque_pixels_never_recover_source_watermark_through_detail_projection():
    from worker.model_detail import fields
    from worker.matte import projected_texture_composite
    rng=np.random.default_rng(8);source=rng.integers(0,255,(80,100,3),dtype=np.uint8)
    base=np.zeros_like(source);mask=np.zeros((80,100),np.uint8);mask[20:60,30:70]=255
    template=dict(mask=np.full((40,40),255,np.uint8),feature=np.ones((40,40),np.float32),alpha=np.full((40,40),.9,np.float32))
    glyph,guard=fields([dict(track_id=1,box=[30,20,70,60])],{1:template},mask.shape)
    result=projected_texture_composite(source,base,mask,glyph,guard,detail_sigma=3)
    assert np.array_equal(result[25:55,35:65],base[25:55,35:65])
    assert np.array_equal(result[mask==0],source[mask==0])


def test_overlapping_detection_views_cover_all_pixels_without_changing_source():
    from scripts.hls_candidates import detection_views
    image=np.zeros((120,80,3),np.uint8);image[40:60,30:50]=127
    original=image.copy();seen=np.zeros(image.shape[:2],np.uint8)
    views=detection_views(image,enhance=True,tiles=True)
    assert len(views)==6
    for crop,x,y,flipped in views[2:]:seen[y:y+crop.shape[0],x:x+crop.shape[1]]=1
    assert seen.all() and np.array_equal(image,original)


def test_stable_colored_opaque_brand_cannot_be_misclassified_as_weak_white_overlay():
    from worker.model_tracks import learn_opacity
    mask=np.zeros((50,70),np.uint8);mask[10:40,20:50]=255;images=[]
    for i in range(20):
        image=np.full((50,70,3),30+i*6,np.uint8);image[mask>0]=[240,170,10];images.append(image)
    alpha=learn_opacity(images,mask)
    assert np.all(alpha[mask>0]>.45)


def test_unknown_single_scene_graphic_is_not_promoted_to_overlay_template(tmp_path):
    import cv2
    from worker.model_tracks import learn_logo_prototypes,ModelTrackSettings
    files=[];rows=[]
    for i in range(20):
        im=np.full((120,220,3),50+i,np.uint8)
        cv2.putText(im,'AB',(60+i,65),cv2.FONT_HERSHEY_SIMPLEX,.7,(160,160,160),2)
        p=tmp_path/f'{i:06d}.png';cv2.imwrite(str(p),im);files.append(p)
        rows.append(dict(frame=i,boxes=[dict(box=[55+i,44,92+i,70],confidence=.9,class_id=1)]))
    args=(files,rows,{1:'float_logo'},[],ModelTrackSettings(),30)
    assert learn_logo_prototypes(*args,scenes=[[0,20]])==[]
    assert learn_logo_prototypes(*args,scenes=[[0,10],[10,20]])


def test_different_graphics_never_swap_tracks_but_anchor_variants_share_identity():
    from worker.model_tracks import associate,ModelTrackSettings
    base=dict(box=[20,20,50,50],confidence=.9,kind='prototype')
    rows=[[dict(base,prototype=0,family=7)],[dict(base,prototype=1,family=8)],
          [dict(base,prototype=2,family=7)]]
    tracks=associate(rows,[[0,3]],30,ModelTrackSettings())
    assert len(tracks)==2
    a=next(t for t in tracks if t['family']==7)
    assert {p['prototype'] for p in a['hints'] if not p.get('interpolated')}=={0,2}


def test_alternate_text_anchor_keeps_confirmed_complete_graphic_geometry(tmp_path,monkeypatch):
    import cv2
    from worker import model_tracks as model
    files=[];rows=[]
    for i in range(20):
        im=np.full((120,220,3),40+i,np.uint8)
        cv2.circle(im,(38,43),20,(230,150,5),-1)
        cv2.putText(im,'AB',(75,42),cv2.FONT_HERSHEY_SIMPLEX,.8,(240,240,240),2)
        cv2.putText(im,'abc',(76,68),cv2.FONT_HERSHEY_SIMPLEX,.5,(240,240,240),1)
        p=tmp_path/f'{i:06d}.png';cv2.imwrite(str(p),im);files.append(p)
        rows.append(dict(frame=i,boxes=[dict(box=[73,20,115,46],confidence=.99,class_id=2),
                                        dict(box=[74,55,110,72],confidence=.9,class_id=2)]))
    im=cv2.imread(str(files[0]));hp=highpass(im);mask=np.zeros(hp.shape,np.uint8);mask[10:85,10:145]=255
    monkeypatch.setattr(model,'load_prior',lambda p:(hp,np.abs(hp),[dict(id=0,box=[10,10,145,85],mask=mask,confidence=.95)],{},{}))
    prototypes=model.learn_text_prototypes(files,rows,None,{2:'float_text'},model.ModelTrackSettings(),30,[[0,10],[10,20]])
    result=model.candidates_for_frame(im,rows[0],prototypes,{2:'float_text'},model.ModelTrackSettings())
    assert result and result[0]['box']==[10,10,145,85]
    assert result[0]['box'][0]<38<result[0]['box'][2]


def test_flipped_detector_coordinates_map_back_without_mirroring_source():
    from scripts.hls_candidates import detection_views,mapped_box
    image=np.zeros((80,120,3),np.uint8);image[10:20,3:9]=255
    views=detection_views(image,flips=True)
    assert mapped_box([111,10,117,20],views[-1])==[3,10,9,20]
    assert np.all(image[10:20,3:9]==255)
