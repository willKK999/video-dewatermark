"""同帧独立目标、目标中途出现及镜头切换必须分别跟踪。"""

from worker.colour_tracks import groups
from worker.hls_v7 import link


def test_multiple_late_targets_and_scene_change():
    rows = [dict(frame=i, color_matches=[]) for i in range(12)]
    for i in range(2, 12):
        rows[i]["color_matches"].append(dict(box=[10, 10, 70, 70], confidence=0.8))
    for i in range(4, 12):
        rows[i]["color_matches"].append(dict(box=[500, 400, 560, 460], confidence=0.9))
    tracks = groups(rows, [(0, 8), (8, 12)])
    assert len(tracks) == 4
    assert sorted(len(t["points"]) for t in tracks) == [4, 4, 4, 6]
    assert sorted(t["points"][0]["frame"] for t in tracks) == [2, 4, 8, 8]


def test_retry_can_replace_dangling_stage_link(tmp_path):
    path = tmp_path / "frames"
    path.symlink_to(tmp_path / "missing", target_is_directory=True)
    target = tmp_path / "new"
    target.mkdir()
    link(path, target)
    assert path.resolve() == target
    link(path, target)
    assert path.resolve() == target
