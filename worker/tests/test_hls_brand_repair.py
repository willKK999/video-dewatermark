"""补修时间只来自当前自动候选，不跨越镜头。"""
from worker.hls_brand_repair import repair_targets


def test_disconnected_footer_repair_padding_cannot_cross_scene_cut():
    evidence=[dict(model_frame=12,box=[20,30,80,60])]
    targets=repair_targets(evidence,[[0,10],[10,20]],20,8)
    assert set(targets)==set(range(10,20))
    assert all(boxes==[[20,30,80,60]] for boxes in targets.values())


def test_independent_brands_in_same_frame_keep_both_masks():
    evidence=[dict(model_frame=12,box=[10,10,40,30]),dict(model_frame=12,box=[70,60,90,80])]
    targets=repair_targets(evidence,[[0,20]],20,1)
    assert len(targets[12])==2
