"""多轨迹的接缝与抽样必须按同一时间戳和全部目标取证。"""

import cv2, numpy as np
from worker.hls import seam_metrics
from scripts.hls_verify import boxes_at


def test_reference_boxes_do_not_collapse_independent_tracks():
    refs = dict(
        multitrack=True,
        tracks=[
            dict(hints=[dict(frame=3, box=[1, 2, 9, 8])]),
            dict(hints=[dict(frame=3, box=[30, 20, 50, 40])]),
        ],
    )
    assert boxes_at(refs, 3) == [[1, 2, 9, 8], [30, 20, 50, 40]]


def test_seam_checks_each_track_and_resamples_source_fps(tmp_path, monkeypatch):
    from worker import hls

    calls = []
    frames = tmp_path / "frames"
    frames.mkdir()
    image = np.full((60, 100, 3), 80, np.uint8)
    for i in [29, 30]:
        cv2.imwrite(str(frames / f"{i:08d}.png"), image)

    def run(args, check):
        calls.append(args)
        cv2.imwrite(str(tmp_path / "seam-source-1/1.png"), image)
        cv2.imwrite(str(tmp_path / "seam-source-1/2.png"), image)

    monkeypatch.setattr(hls, "run", run)
    refs = [
        dict(frame=30, box=[1, 2, 9, 8], track_id=1),
        dict(frame=30, box=[60, 40, 80, 55], track_id=2),
    ]
    records = seam_metrics(
        "source.mp4",
        frames,
        [dict(start=0), dict(start=30)],
        30,
        tmp_path,
        lambda: None,
        refs,
    )
    assert len(records[0]["tracks"]) == 2
    assert str(calls[0][calls[0].index("-vf") + 1]).startswith("fps=30,")
    assert all(p["result_mae"] == 0 for p in records[0]["tracks"])
