"""比较源片和输出时必须对称地约束自动品牌范围，不能靠统计口径制造下降。"""
from scripts import hls_single_check as single


def test_same_image_observations_have_identical_source_and_output_scope(monkeypatch):
    observed=[dict(kind='logo',model_box=[0,0,10,10]),
              dict(kind='logo',model_box=[30,30,40,40]),
              dict(kind='prototype',model_box=[50,50,60,60])]
    monkeypatch.setattr(single,'candidates_for_frame',lambda *a:observed)
    source=single.tracked_brand_candidates(None,{},[],{},[[0,0,10,10]])
    output=single.tracked_brand_candidates(None,{},[],{},[[0,0,10,10]])
    assert source==output==[observed[0],observed[2]]
    assert len(observed)==3  # 原始检测不得被筛选操作改写。
