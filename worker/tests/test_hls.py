"""HLS 解析、时间轴、重叠融合及封装集成验证。"""

from pathlib import Path
import json
import cv2
import numpy as np
import pytest
from worker.hls import parse, chunks, resolve, merge_frames, Reader, snapshot
from worker.process import PipelineError


def test_master_and_relative_paths():
    _, variants, _ = parse(
        "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100\nlow/index.m3u8\n#EXT-X-STREAM-INF:BANDWIDTH=200\nhigh/index.m3u8"
    )
    assert variants == [(100, "low/index.m3u8"), (200, "high/index.m3u8")]
    assert resolve("tenant/video/index.m3u8", "../a.ts") == "tenant/a.ts"
    assert (
        resolve("tenant/video/index.m3u8", "/tenant/a%20b.ts?placeholder=x")
        == "tenant/a b.ts"
    )
    assert (
        resolve("https://cdn.example/a/index.m3u8", "../b.ts")
        == "https://cdn.example/b.ts"
    )


@pytest.mark.parametrize(
    "tag",
    [
        '#EXT-X-KEY:METHOD=AES-128,URI="secret"',
        '#EXT-X-SESSION-KEY:METHOD=AES-128,URI="secret"',
        '#EXT-X-MAP:URI="init.mp4"',
        "#EXT-X-BYTERANGE:12",
        "#EXT-X-DISCONTINUITY",
        "#EXT-X-GAP",
    ],
)
def test_unsupported_features_are_explicit(tag):
    with pytest.raises(PipelineError, match="暂不支持"):
        parse("#EXTM3U\n" + tag + "\n#EXTINF:6,\na.ts\n#EXT-X-ENDLIST")


@pytest.mark.parametrize("duration", ["nan", "inf", "-1", "0"])
def test_invalid_duration(duration):
    with pytest.raises(PipelineError):
        parse(f"#EXTM3U\n#EXTINF:{duration},\na.ts\n#EXT-X-ENDLIST")


def test_vod_required():
    with pytest.raises(PipelineError):
        parse("#EXTM3U\n#EXTINF:6,\na.ts")


@pytest.mark.parametrize(
    "durations", [[6] * 41 + [4.233333], [3.7] * 25, [40, 3], [1.001] * 200]
)
@pytest.mark.parametrize("fps", [25, 30, 30000 / 1001])
def test_chunk_bounds_and_coverage(durations, fps):
    plans = chunks(durations, fps)
    assert plans[0]["start"] == 0
    assert plans[-1]["end"] == round(sum(durations) * fps)
    for i, p in enumerate(plans):
        assert 0 <= p["left"] <= p["start"] < p["end"] <= p["right"]
        assert (p["right"] - p["left"]) / fps <= 30
        if i:
            assert plans[i - 1]["end"] == p["start"]


def test_overlap_blends_same_timestamp(tmp_path):
    joined = tmp_path / "joined"
    joined.mkdir()
    incoming = tmp_path / "new"
    incoming.mkdir()
    for i in range(8):
        cv2.imwrite(
            str(joined / f"{i:08d}.png"), np.full((4, 4, 3), 10, dtype=np.uint8)
        )
    for i in range(6):
        cv2.imwrite(
            str(incoming / f"{i:06d}.png"), np.full((4, 4, 3), 110, dtype=np.uint8)
        )
    merge_frames(joined, incoming, dict(left=4, right=10), 8, 30, lambda: None)
    values = [int(cv2.imread(str(joined / f"{i:08d}.png"))[0, 0, 0]) for i in range(10)]
    assert values == [10] * 4 + [30, 50, 70, 90, 110, 110]


def test_reader_cancellation_is_not_retried():
    from worker.process import LeaseLost

    def cancel():
        raise LeaseLost()

    with pytest.raises(LeaseLost):
        Reader(None, "bucket", cancel).read("a.ts", 10)


def test_snapshot_selects_requested_bandwidth(tmp_path):
    class Fake:
        def read(self, key, limit):
            if key == "master.m3u8":
                return b"#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=100\nlow/index.m3u8\n#EXT-X-STREAM-INF:BANDWIDTH=200\nhigh/index.m3u8"
            if key == "low/index.m3u8":
                return b"#EXTM3U\n#EXT-X-TARGETDURATION:6\n#EXT-X-MEDIA-SEQUENCE:17\n#EXTINF:6,\n../a.ts\n#EXT-X-ENDLIST"
            assert key == "a.ts"
            return b"ts"

    playlist, lines, segments, digest = snapshot("master.m3u8", Fake(), tmp_path, 150)
    assert (playlist.parent / "000000.ts").read_bytes() == b"ts"
    assert "#EXT-X-MEDIA-SEQUENCE:17" in playlist.read_text()
    assert len(digest) == 64


@pytest.mark.parametrize("input_fps", [None, 60])
def test_real_hls_pipeline_retry_timestamps_and_audio(tmp_path, monkeypatch, input_fps):
    """用真实 FFmpeg 和完整开发流水线覆盖跨块、检查点及音频包保持。"""
    import io
    from botocore.exceptions import ClientError
    from worker.hls import execute_hls
    from worker.process import run
    from scripts.fixtures import generate
    from worker.media import probe

    monkeypatch.delenv("DETECTOR", raising=False)
    monkeypatch.delenv("TEMPLATE_PATH", raising=False)
    source = generate(tmp_path)
    hls = tmp_path / "input-hls"
    hls.mkdir()
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            source,
            *(["-r", str(input_fps)] if input_fps else []),
            "-t",
            "31",
            "-c:v",
            "libx264",
            "-force_key_frames",
            "expr:gte(t,n_forced*6)",
            "-sc_threshold",
            "0",
            "-af",
            r"asetpts=PTS+0.008/TB*eq(N\,45056)",
            "-c:a",
            "aac",
            "-hls_segment_options",
            "pes_payload_size=0",
            "-f",
            "hls",
            "-hls_time",
            "6",
            "-hls_list_size",
            "0",
            hls / "index.m3u8",
        ]
    )

    class MemoryStore:
        def __init__(self):
            self.objects = {p.name: p.read_bytes() for p in hls.iterdir()}

        def get_object(self, Bucket, Key):
            if Key not in self.objects:
                raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
            return {"Body": io.BytesIO(self.objects[Key])}

        def download_file(self, bucket, key, path, **kwargs):
            Path(path).write_bytes(self.get_object(bucket, key)["Body"].read())

        def upload_file(self, path, bucket, key, **kwargs):
            self.objects[key] = Path(path).read_bytes()

        def put_object(self, Bucket, Key, Body, **kwargs):
            self.objects[Key] = Body

    s3 = MemoryStore()
    options = {
        "max_short_edge": 720,
        "prompts": [
            {"frame": 0, "box": [20, 30, 62, 52]},
            {"frame": 11, "box": [53, 30, 95, 52]},
        ],
    }
    import worker.hls as hls_module

    original_execute = hls_module.execute
    calls = []

    def fail_second_block(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise PipelineError("STORAGE_UNAVAILABLE", "模拟块失败", retryable=True)
        return original_execute(*args, **kwargs)

    monkeypatch.setattr(hls_module, "execute", fail_second_block)
    reports = []
    for attempt in range(3):
        root = tmp_path / f"attempt-{attempt}"
        root.mkdir()
        if attempt == 0:
            with pytest.raises(PipelineError, match="模拟块失败"):
                execute_hls(
                    "index.m3u8",
                    root,
                    options,
                    "telea-dev",
                    False,
                    lambda _: None,
                    lambda: None,
                    s3,
                    "bucket",
                    "cache",
                    lambda *args: None,
                )
            continue
        result, meta, _ = execute_hls(
            "index.m3u8",
            root,
            options,
            "telea-dev",
            False,
            lambda _: None,
            lambda: None,
            s3,
            "bucket",
            "cache",
            lambda *args: None,
        )
        report = json.loads((root / "metrics-hls.json").read_text())
        reports.append(report)
        assert meta["chunk_count"] == 2
        if input_fps:
            assert report["source_fps"] == input_fps
            assert meta["fps"] == 30
        assert abs(float(probe(result)["format"]["duration"]) - 31) < 0.1
        assert len(report["segment_timing"]) == 6
        assert report["source_start_pts"] == pytest.approx(
            float(
                next(
                    s
                    for s in probe(hls / "index0.ts")["streams"]
                    if s["codec_type"] == "video"
                )["start_time"]
            )
        )
        assert [p["reused"] for p in report["chunks"]] == (
            [True, False] if attempt == 1 else [True, True]
        )

        # ADTS 中的压缩音频负载在源、MP4 成品及新 HLS 中保持相同。
        def audio(path):
            return run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    path,
                    "-map",
                    "0:a:0",
                    "-c",
                    "copy",
                    "-f",
                    "adts",
                    "-",
                ]
            )

        assert audio(hls / "index.m3u8") == audio(root / "source.mp4")
        assert audio(root / "source.mp4") == audio(result)
        assert audio(result) == audio(root / "hls" / "index.m3u8")
    assert reports[0]["seams"] == reports[1]["seams"]
    assert len(calls) == 3


def test_reader_rejects_cross_tenant_key():
    with pytest.raises(PipelineError, match="对象键无效"):
        Reader(None, "bucket", lambda: None, key_prefix="kelin/").read(
            "other/video.ts", 10
        )

    with pytest.raises(PipelineError, match="对象键无效"):
        Reader(None, "bucket", lambda: None, key_prefix="kelin/").read(
            "kelin/../other/video.ts", 10
        )


def test_url_rejects_private_addresses(monkeypatch):
    from worker.hls import public_url
    import socket

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 80))],
    )
    with pytest.raises(PipelineError, match="内网"):
        public_url("http://example.test/video.ts")


def test_reader_preserves_lease_loss_during_download():
    from worker.process import LeaseLost

    class Stream:
        closed = False

        def read(self, size):
            raise LeaseLost()

        def close(self):
            self.closed = True

    stream = Stream()

    class Store:
        def get_object(self, **kwargs):
            return {"Body": stream}

    with pytest.raises(LeaseLost):
        Reader(Store(), "bucket", lambda: None).read("video.ts", 100)
    assert stream.closed


def test_private_r2_url_uses_object_credentials_without_query(monkeypatch):
    import io

    monkeypatch.setenv("S3_ENDPOINT", "account.r2.cloudflarestorage.com")

    class Store:
        def get_object(self, Bucket, Key):
            assert Bucket == "bucket" and Key == "kelin/video/a.ts"
            return {"Body": io.BytesIO(b"fragment")}

    reader = Reader(Store(), "bucket", lambda: None, key_prefix="kelin/")
    assert (
        reader.read(
            "https://account.r2.cloudflarestorage.com/bucket/kelin/video/a.ts?placeholder=value",
            100,
        )
        == b"fragment"
    )
    with pytest.raises(PipelineError):
        reader.read("https://account.r2.cloudflarestorage.com/bucket/other/a.ts", 100)


def test_fractional_segment_alignment_keeps_four_segments():
    plans = chunks([6.006] * 10, 30000 / 1001)
    assert plans[0]["end"] == 720
    assert all((p["right"] - p["left"]) / (30000 / 1001) <= 30 for p in plans)


def test_seam_comparison_selects_exact_frame_indices(tmp_path):
    from scripts.fixtures import generate
    from worker.hls import seam_metrics

    source = generate(tmp_path)
    output = tmp_path / "audit"
    output.mkdir()
    plans = [
        dict(start=0, end=6, left=0, right=8),
        dict(start=6, end=12, left=4, right=12),
    ]
    rows = seam_metrics(
        source, tmp_path / "unused", plans, 12, output, lambda: None, result_path=source
    )
    assert rows[0]["repair_delta_mae"] == 0
    capture = cv2.VideoCapture(str(source))
    expected = []
    for i in range(7):
        ok, frame = capture.read()
        assert ok
        if i in (5, 6):
            expected.append(frame)
    capture.release()
    for i, image in enumerate(expected, 1):
        actual = cv2.imread(str(output / "seam-source-1" / f"{i}.png"))
        assert np.array_equal(actual, image)


def test_checkpoint_changes_with_detector_threshold(monkeypatch):
    from worker.hls import checkpoint_hash

    first = checkpoint_hash("source", {}, "propainter", False, "execution")
    monkeypatch.setenv("YOLO_CONFIDENCE", ".47")
    assert checkpoint_hash("source", {}, "propainter", False, "execution") != first


@pytest.mark.parametrize("hls", [False, True])
def test_hls_quality_counts_neighboring_residual_icon(tmp_path, monkeypatch, hls):
    from worker import vision

    monkeypatch.setenv("DETECTOR", "yolo")
    monkeypatch.setenv("DETECT_PYTHON", "detector-python")
    monkeypatch.setenv("YOLO_WEIGHTS", "dedicated.pt")
    monkeypatch.delenv("RESIDUAL_CONFIDENCE", raising=False)
    monkeypatch.delenv("HLS_RESIDUAL_CONFIDENCE", raising=False)
    detections = dict(
        hls=hls, scenes=[[0, 3]], hints=[dict(frame=0, box=[100, 200, 180, 230])]
    )
    (tmp_path / "detections.json").write_text(json.dumps(detections))

    def detect(args, check):
        confidence = float(args[args.index("--confidence") + 1])
        assert confidence == (0.15 if hls else 0.35)
        assert ("--reference-margin" in args) == hls
        Path(args[args.index("--output") + 1]).write_text(
            json.dumps([dict(frame=1, box=[58, 195, 99, 235], confidence=0.3)])
        )

    monkeypatch.setattr(vision, "run", detect)
    vision.residual_quality(
        tmp_path, [tmp_path / f"{i:06d}.png" for i in range(3)], detections
    )
    quality = json.loads((tmp_path / "quality.json").read_text())
    assert quality["residual_count"] == (1 if hls else 0)


@pytest.mark.parametrize("uncovered", [4, 10])
def test_hls_does_not_publish_single_track_when_other_logos_persist(
    tmp_path, monkeypatch, uncovered
):
    import cv2, numpy as np
    from worker import vision

    monkeypatch.setenv("DETECTOR", "yolo")
    monkeypatch.setenv("DETECT_PYTHON", "detector")
    monkeypatch.setenv("YOLO_WEIGHTS", "dedicated.pt")
    folder = tmp_path / "frames"
    folder.mkdir()
    files = []
    for i in range(20):
        path = folder / f"{i:06d}.png"
        cv2.imwrite(str(path), np.zeros((10, 10, 3), np.uint8))
        files.append(path)

    def detect(args, check):
        (tmp_path / "detector-output.json").write_text(
            json.dumps(
                [dict(frame=i, box=[1, 1, 4, 4], confidence=0.9) for i in range(20)]
            )
        )
        (tmp_path / "metrics.detector-output.json").write_text(
            json.dumps(dict(uncovered_watermark_frames=uncovered))
        )

    monkeypatch.setattr(vision, "run", detect)
    if uncovered > 5:
        with pytest.raises(PipelineError, match="多个独立水印"):
            vision.detect(files, [(0, 20)], dict(hls=True), tmp_path)
    else:
        result = vision.detect(files, [(0, 20)], dict(hls=True), tmp_path)
        assert len(result["hints"]) == 20


def test_short_tail_is_merged_without_exceeding_context_limit():
    plans = chunks([5] * 50 + [0.133333], 30)
    assert len(plans) == 10 and plans[-1]["end"] == 7504
    assert all(p["right"] - p["left"] <= 900 for p in plans)
    assert plans[-1]["start"] == 6750


def test_failed_block_preserves_phase_and_total_timing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import worker.hls as hls

    work = tmp_path / "chunk-0000"
    work.mkdir()
    (work / "phase_ms.json").write_text(json.dumps(dict(probe=20, detect=40)))
    monkeypatch.setattr(hls, "time", SimpleNamespace(monotonic=lambda: 100))
    hls.record_failed_chunk(tmp_path, work, 1, 90, 80)
    data = json.loads((tmp_path / "phase_ms.json").read_text())
    assert data == dict(
        hls_chunk_01_probe=20,
        hls_chunk_01_detect=40,
        hls_chunk_01_total=10000,
        hls_total_to_failure=20000,
    )
