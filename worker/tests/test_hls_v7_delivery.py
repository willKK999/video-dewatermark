"""原规格有界重叠合并与原时间轴封装，必须保留每一帧及压缩音轨。"""

from pathlib import Path
import json, time
import cv2
from worker.process import run
from worker.media import probe
from worker.hls_v7 import join_videos
from worker.hls_package import package


def test_native_cadence_overlap_and_hls_audio_copy(tmp_path):
    source = tmp_path / "source.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=60:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            source,
        ]
    )
    plans = [
        dict(start=0, end=60, left=0, right=72),
        dict(start=60, end=120, left=48, right=120),
    ]
    works = []
    for i, p in enumerate(plans):
        work = tmp_path / f"block-{i}"
        (work / "native").mkdir(parents=True)
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(p["left"] / 60),
                "-i",
                source,
                "-frames:v",
                str(p["right"] - p["left"]),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "0",
                work / "native/chunk1.mp4",
            ]
        )
        works.append(work)
    output = tmp_path / "result.mp4"
    join_videos(works, plans, output, source, [0.5, 1], lambda: None)
    v = next(
        s for s in probe(output, count=True)["streams"] if s["codec_type"] == "video"
    )
    assert v["avg_frame_rate"] == "60/1"
    assert v["nb_read_frames"] == "120"
    # 重叠混合必须取同一时间戳，不能造成前进/后退或把两份上下文都拼入成品。
    a = cv2.VideoCapture(str(source))
    b = cv2.VideoCapture(str(output))
    for n in range(120):
        ok, x = a.read()
        good, y = b.read()
        assert ok and good
        assert abs(x.astype(float) - y).mean() < 4
    a.release()
    b.release()
    segments = [(0.5, "a.ts"), (0.5, "b.ts"), (1.0, "c.ts")]
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-TARGETDURATION:1",
        "#EXT-X-MEDIA-SEQUENCE:17",
    ]
    for seconds, name in segments:
        lines.extend([f"#EXTINF:{seconds},", name])
    lines.append("#EXT-X-ENDLIST")
    dest, meta, _ = package(
        source,
        output,
        tmp_path,
        segments,
        lines,
        3.0,
        plans,
        60,
        60,
        [dict(index=i + 1, **p) for i, p in enumerate(plans)],
        "digest",
        "config",
        time.monotonic(),
        [],
        lambda: None,
    )
    assert meta["audio_copy_verified"] is True
    assert meta["fps"] == 60
    playlist = (tmp_path / "hls/index.m3u8").read_text()
    assert "#EXT-X-MEDIA-SEQUENCE:17" in playlist
    assert "#EXT-X-TARGETDURATION:1" in playlist
    report = json.loads((tmp_path / "metrics-hls.json").read_text())
    assert report["audio_timeline_max_error_ms"] <= 1
    assert len(report["segment_timing"]) == 3
    assert (
        abs(
            float(
                next(s for s in probe(dest)["streams"] if s["codec_type"] == "video")[
                    "duration"
                ]
            )
            - 2
        )
        < 0.001
    )
