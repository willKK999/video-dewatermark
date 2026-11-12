"""晚出现目标和碎镜头不能在无提示时传播，也不能遗失对象。"""

from worker.sam_schedule import groups_for_scene, sampled_hints


def test_late_object_gets_own_lifetime_and_initial_prompt():
    hints = [dict(frame=i, track_id=1, box=[1, 1, 9, 9]) for i in range(100)]
    hints += [dict(frame=i, track_id=6, box=[20, 20, 30, 30]) for i in range(41, 65)]
    groups = groups_for_scene(dict(multitrack=True, hints=hints), 0, 100)
    assert [(a, b) for a, b, h in groups] == [(0, 100), (41, 65)]
    for a, b, h in groups:
        prompts = sampled_hints(h, a, b)
        for identity in {p["track_id"] for p in prompts}:
            assert min(p["frame"] for p in prompts if p["track_id"] == identity) == a


def test_multiple_objects_with_same_lifetime_share_state_and_all_remain():
    hints = [dict(frame=i, track_id=t) for i in range(4) for t in (1, 2, 9)]
    groups = groups_for_scene(dict(multitrack=True, hints=hints), 1, 3)
    assert len(groups) == 1
    a, b, points = groups[0]
    assert (a, b) == (1, 3) and {p["track_id"] for p in points} == {1, 2, 9}
    assert len(sampled_hints(points, a, b)) == 6


def test_single_frame_shot_and_absent_tracks_are_separate():
    d = dict(
        multitrack=True, hints=[dict(frame=2, track_id=1), dict(frame=9, track_id=2)]
    )
    assert [(a, b) for a, b, h in groups_for_scene(d, 2, 3)] == [(2, 3)]
    assert groups_for_scene(d, 4, 8) == []


def test_generated_mask_prompt_checks_path_shape_and_values(tmp_path):
    import cv2, numpy as np, pytest
    from worker.sam_schedule import load_mask_prompt

    (tmp_path / "prompts").mkdir()
    mask = np.zeros((20, 30), np.uint8)
    mask[4:8, 6:12] = 255
    cv2.imwrite(str(tmp_path / "prompts/seed.png"), mask)
    assert load_mask_prompt(tmp_path, "prompts/seed.png", (20, 30)).sum() == 24
    for path in ["../key.bin", "/tmp/mask.png", "key.bin"]:
        with pytest.raises(ValueError):
            load_mask_prompt(tmp_path, path, (20, 30))
    with pytest.raises(ValueError):
        load_mask_prompt(tmp_path, "prompts/seed.png", (10, 10))


def test_mask_deformation_adds_prompt_when_box_does_not_move():
    import numpy as np
    from worker.sam_schedule import adaptive_mask_hints

    hints = [dict(frame=i, box=[0, 0, 20, 20]) for i in range(6)]
    masks = []
    for i in range(6):
        mask = np.zeros((20, 20), np.uint8)
        mask[2:10, 2:10] = 255
        if i >= 2:
            mask[12:18, 12:18] = 255
        masks.append(mask)
    selected = list(adaptive_mask_hints(hints, 0, 6, lambda p: masks[p["frame"]]))
    assert [p["frame"] for p, mask in selected] == [0, 2, 5]
    assert [
        p["frame"]
        for p, mask in adaptive_mask_hints(
            hints, 0, 6, lambda p: masks[p["frame"]], fast=False
        )
    ] == list(range(6))


def test_weak_pixel_observations_remain_covered_by_reliable_prompts():
    import numpy as np
    from worker.sam_schedule import adaptive_mask_hints

    hints = [dict(frame=i, box=[0, 0, 20, 20]) for i in range(5)]
    mask = np.zeros((20, 20), np.uint8)
    mask[3:12, 3:12] = 255
    selected = list(
        adaptive_mask_hints(
            hints, 0, 5, lambda p: mask if p["frame"] in (1, 3) else None
        )
    )
    assert [p["frame"] for p, _ in selected] == [1, 3]
    assert [p["frame"] for p in hints] == list(range(5))
    assert list(adaptive_mask_hints(hints, 0, 5, lambda p: None)) == []
