"""形态变化重新初始化；多个同帧副本不能合并到同一个对象。"""

from worker.graphic_tracks import associate


def point(box, kind=1):
    return dict(box=box, class_id=kind, confidence=0.9)


def test_scale_change_trims_old_context_and_initializes_new_target():
    rows = [
        dict(frame=0, boxes=[point([0, 0, 100, 80])]),
        dict(frame=1, boxes=[point([0, 0, 100, 80])]),
        dict(frame=2, boxes=[point([0, 0, 30, 20])]),
        dict(frame=3, boxes=[point([0, 0, 30, 20])]),
    ]
    tracks = associate(rows, [[0, 10]])
    assert [p["scenes"] for p in tracks] == [[[0, 2]], [[2, 10]]]
    assert [[p["frame"] for p in t["hints"]] for t in tracks] == [[0, 1], [2, 3]]


def test_nested_new_object_does_not_close_observed_old_object():
    large, small = point([0, 0, 100, 80]), point([0, 0, 30, 20])
    tracks = associate(
        [dict(frame=0, boxes=[large]), dict(frame=1, boxes=[small, large])], [[0, 10]]
    )
    assert len(tracks) == 2 and all(p["scenes"][0][1] == 10 for p in tracks)
    assert all(len({p["frame"] for p in t["hints"]}) == len(t["hints"]) for t in tracks)


def test_source_cut_and_mid_scene_birth_keep_separate_initializers():
    p = point([10, 10, 40, 40])
    tracks = associate(
        [dict(frame=i, boxes=[p] if i in [1, 2, 5] else []) for i in range(8)],
        [[0, 4], [4, 8]],
    )
    assert [t["scenes"] for t in tracks] == [[[1, 4]], [[5, 8]]]


def test_reciprocal_extent_change_uses_same_rule_for_matching_and_closing():
    large, small = point([0, 0, 400, 100]), point([0, 0, 185, 94])
    tracks = associate(
        [dict(frame=0, boxes=[large]), dict(frame=1, boxes=[small])], [[0, 10]]
    )
    assert [t["scenes"] for t in tracks] == [[[0, 1]], [[1, 10]]]
