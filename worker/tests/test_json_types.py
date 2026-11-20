"""NumPy整数、浮点、布尔和多维数组必须完整写出，不丢弃原始候选。"""
import json
import numpy as np
from worker.json_types import json_native
from worker.model_tracks import save_prototypes,load_prototypes


def test_nested_numpy_values_keep_numeric_values_and_array_dimensions():
    data={np.int32(7):dict(帧=np.int64(31),分数=np.float32(.25),通过=np.bool_(False),
                          框=np.array([[1,2],[3,4]],dtype=np.int32),候选=(np.int32(9),np.float64(.5)))}
    text=json.dumps(json_native(data),ensure_ascii=False)
    assert json.loads(text)=={'7':dict(帧=31,分数=.25,通过=False,框=[[1,2],[3,4]],候选=[9,.5])}


def test_prototype_checkpoint_accepts_numpy_geometry_and_preserves_templates(tmp_path):
    feature=np.arange(12,dtype=np.float32).reshape(3,4)
    p=dict(id=np.int32(1),family=np.int32(2),anchor=[np.int32(v) for v in [2,3,6,6]],box=[np.int32(v) for v in [1,2,8,8]],
           mask=np.full((3,4),255,np.uint8),alpha=np.full((3,4),.2,np.float32),feature=feature,whole_feature=feature,examples=[feature])
    path=tmp_path/'prototypes.npz';save_prototypes(path,[p]);loaded=load_prototypes(path)[0]
    assert loaded['anchor']==[2,3,6,6] and loaded['id']==1
    assert np.array_equal(loaded['feature'],feature) and np.array_equal(loaded['mask'],p['mask'])
