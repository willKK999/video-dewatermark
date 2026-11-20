"""逐帧模型约束排除未印证模板峰，通用形状缓存不固定到画面坐标。"""
import numpy as np
from worker.model_tracks import ModelTrackSettings,candidates_for_frame,associate,place,feature_score
from worker.multitrack import highpass
from worker.sam_schedule import groups_for_scene


def test_template_cannot_create_tracks_without_frame_yolo_confirmation():
    im=np.zeros((120,160,3),np.uint8)
    im[20:40,30:70]=255
    proto=dict(id=0,anchor=[30,20,70,40],box=[20,10,80,50],examples=[highpass(im[20:40,30:70])])
    assert candidates_for_frame(im,{'boxes':[]},[proto],{2:'float_text'},ModelTrackSettings())==[]


def test_duplicate_model_parts_do_not_expand_into_independent_tracks():
    cfg=ModelTrackSettings()
    p=dict(box=[30,20,70,40],confidence=.9,class_id=1)
    result=candidates_for_frame(np.zeros((120,160,3),np.uint8),{'boxes':[p,p]},[],{1:'float_logo'},cfg)
    assert len(result)==1
    rows=[[dict(result[0],box=[p+shift for p,shift in zip(result[0]['box'],[i*.1,0,i*.1,0])])] for i in range(100)]
    tracks=associate(rows,[[0,100]],30,cfg)
    d=dict(multitrack=True,hints=[p for t in tracks for p in t['hints']])
    assert len(tracks)==1 and len(groups_for_scene(d,0,100))==1


def test_plain_subtitle_without_brand_visual_prototype_is_not_watermark():
    rng=np.random.default_rng(12)
    image=rng.integers(0,255,(100,200,3),dtype=np.uint8)
    proto=dict(id=0,anchor=[0,0,60,12],box=[0,0,90,40],examples=[rng.normal(size=(12,60)).astype(np.float32)])
    result=candidates_for_frame(image,{'boxes':[dict(box=[50,70,130,85],confidence=.95,class_id=2)]},[proto],{2:'float_text'},ModelTrackSettings())
    assert result==[]


def test_objects_are_not_propagated_before_first_detection_or_across_cuts():
    point=dict(box=[10,10,20,20],confidence=.9,kind='logo',prototype=-1)
    rows=[[] for i in range(20)]
    for i in range(5,10):rows[i]=[point]
    for i in range(10,20):rows[i]=[point]
    tracks=associate(rows,[[0,10],[10,20]],30,ModelTrackSettings())
    assert len(tracks)==2
    assert min(p['frame'] for p in tracks[0]['hints'])==5
    assert min(p['frame'] for p in tracks[1]['hints'])==10


def test_shape_at_frame_edge_is_cropped_not_squeezed():
    mask=np.zeros((10,20),np.uint8);mask[:,10:]=255
    result=place(mask,[-10,0,10,10],(20,20))
    assert np.all(result[:10,:10]==255) and np.count_nonzero(result)==100


def test_model_category_jitter_does_not_lose_confirmed_brand_graphic():
    rng=np.random.default_rng(13)
    im=rng.integers(0,255,(100,160,3),dtype=np.uint8)
    feature=highpass(im[20:32,40:100])
    proto=dict(id=0,anchor=[40,20,100,32],box=[20,0,120,45],examples=[feature])
    cfg=ModelTrackSettings()
    result=candidates_for_frame(im,{'boxes':[dict(box=[40,20,100,32],confidence=.9,class_id=1)]},[proto],{1:'float_logo'},cfg)
    assert len(result)==1 and result[0]['kind']=='prototype'
    assert result[0]['box']==[20,0,120,45]
