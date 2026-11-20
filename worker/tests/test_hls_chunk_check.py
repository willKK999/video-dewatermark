"""单块导出必须去掉上下文，不能把后续目标或镜头写进结果引用。"""

from scripts.hls_chunk_check import cut_references


def test_core_references_drop_future_shots_and_objects():
    d = dict(
        multitrack=True,
        scenes=[[0, 4], [4, 8], [8, 10]],
        tracks=[
            dict(id=1, hints=[dict(frame=i, track_id=1) for i in range(10)]),
            dict(id=2, hints=[dict(frame=7, track_id=2)]),
        ],
    )
    result = cut_references(d, 0, 5)
    assert result["scenes"] == [[0, 4], [4, 5]]
    assert [t["id"] for t in result["tracks"]] == [1]
    assert [p["frame"] for p in result["hints"]] == list(range(5))
    assert len(d["tracks"][0]["hints"]) == 10


def test_context_offset_is_rebased_for_every_track():
    d = dict(
        scenes=[[0, 9]],
        tracks=[
            dict(id=t, hints=[dict(frame=i, track_id=t) for i in range(9)])
            for t in [1, 7]
        ],
    )
    result = cut_references(d, 2, 5)
    assert result["scenes"] == [[0, 3]]
    assert {p["frame"] for p in result["hints"]} == {0, 1, 2}
    assert {p["track_id"] for p in result["hints"]} == {1, 7}


def test_audio_mux_retains_last_aac_packet_at_video_boundary(tmp_path):
    from scripts.hls_chunk_check import mux_core_audio, audio_hash
    from worker.process import run

    audio = tmp_path / "input.aac"
    video = tmp_path / "video.mp4"
    output = tmp_path / "output.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-t",
            "2",
            "-c:a",
            "aac",
            "-f",
            "adts",
            audio,
        ]
    )
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=size=64x64:rate=30",
            "-frames:v",
            "30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video,
        ]
    )
    mux_core_audio(video, audio, output, 1)
    assert audio_hash(audio, 1) == audio_hash(output)


def test_recheck_does_not_schedule_models_or_next_chunk(tmp_path, monkeypatch):
    import json, sys
    from scripts import hls_chunk_check as check

    (tmp_path / "report.json").write_text(
        json.dumps(
            {"帧率": 30, "阶段耗时毫秒": {}, "自动复修": [], "输入提取耗时秒": 0}
        )
    )
    calls = []
    monkeypatch.setattr(
        check,
        "run_one",
        lambda *a: (_ for _ in ()).throw(AssertionError("不能重新调度模型")),
    )
    monkeypatch.setattr(check, "run", lambda *a: calls.append("提取视频流"))
    monkeypatch.setattr(check, "mux_core_audio", lambda *a: calls.append("复制音轨"))
    monkeypatch.setattr(check, "verify_one", lambda *a: calls.append("完整复检"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hls_chunk_check",
            "--source",
            "source.mp4",
            "--prior",
            "prior.npz",
            "--output",
            str(tmp_path),
            "--resume-root",
            str(tmp_path / "work"),
            "--recheck",
        ],
    )
    check.main()
    assert calls == ["提取视频流", "复制音轨", "完整复检"]
