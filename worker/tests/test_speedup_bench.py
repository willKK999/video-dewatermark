"""基准报告必须区分质量检查与缺失证据，缓存必须识别内容变化。"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scripts import hls_speedup_bench as bench
from worker.speedup_cache import StageCache, fingerprint
from worker.media import normalize_direct, encode, probe


def make_video(path, rate=60, seconds=0.2, audio=False):
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=128x96:rate={rate}:duration={seconds}",
    ]
    if audio:
        args += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:a",
            "aac",
        ]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True, capture_output=True)


def test_real_metrics_align_sixty_to_thirty_frames_and_report_crops(tmp_path):
    sixty = tmp_path / "sixty.mp4"
    thirty = tmp_path / "thirty.mp4"
    make_video(sixty)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(sixty),
            "-vf",
            "select=not(mod(n\\,2)),setpts=N/(30*TB)",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            str(thirty),
        ],
        check=True,
        capture_output=True,
    )
    report = bench.compare(sixty, thirty, tmp_path / "metrics", 0.2, [(0, 0, 64, 48)])
    assert report["全画面"]["帧数"] == 6
    assert report["全画面"]["SSIM均值"] == 1
    assert report["全画面"]["均方误差日志为零"]
    assert report["0,0,64,48"]["帧数"] == 6


def test_clarity_is_record_only_and_lists_every_low_sample():
    samples = [
        dict(measurable=True, retention_ratio=0.62, passed=False),
        dict(measurable=True, retention_ratio=0.75, passed=False),
        dict(measurable=False, retention_ratio=0.1),
    ]
    summary = bench.clarity_summary(samples)
    assert summary["最低值"] == 0.62
    assert summary["全部样本"] == samples
    assert summary["低于零点七的样本"] == samples[:1]
    assert summary["仅记录不作绝对门槛"]


def test_checkpoint_reuses_only_unchanged_outputs_and_handles_failure(tmp_path):
    source = tmp_path / "source.txt"
    target = tmp_path / "out.txt"
    source.write_text("原始输入")
    cache = StageCache(tmp_path / "stages")
    calls = []

    def operation():
        calls.append(1)
        target.write_text("已完成")

    identity = fingerprint([source], {"参数": 1})
    cache.run("detect", identity, operation, lambda: [target])
    cache.run("detect", identity, operation, lambda: [target])
    assert len(calls) == 1 and cache.timings["detect"]["reused"]
    target.write_text("被改动")
    cache.run("detect", identity, operation, lambda: [target])
    assert len(calls) == 2
    source.write_text("不同输入")
    assert fingerprint([source], {"参数": 1}) != identity

    def failure():
        target.write_text("未完成")
        raise RuntimeError("中断")

    with pytest.raises(RuntimeError):
        cache.run("sam", identity, failure, lambda: [target])
    assert not (tmp_path / "stages/sam.json").exists()
    assert not cache.timings["sam"]["reused"]


def test_direct_decode_and_audio_copy_without_intermediate_video(tmp_path):
    from scripts.hls_chunk_check import audio_hash

    source = tmp_path / "source.mp4"
    make_video(source, seconds=1, audio=True)
    work = tmp_path / "work"
    work.mkdir()
    info = normalize_direct(source, work)
    assert info["frame_count"] == 30
    assert not (work / "normalized.mp4").exists()
    (work / "composite").symlink_to(work / "frames", target_is_directory=True)
    result = encode(work, info)
    assert audio_hash(source) == audio_hash(result)
    assert (
        next(s for s in probe(result)["streams"] if s["codec_type"] == "video")["width"]
        == 128
    )


def test_help_never_initializes_a_device():
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "scripts.hls_speedup_bench", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "从原片重建九项输入" in result.stdout and "--baseline" in result.stdout


def test_assessment_does_not_turn_missing_residual_and_visual_checks_into_pass(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.mp4"
    make_video(source)
    args = SimpleNamespace(
        source=source,
        candidate=source,
        baseline=source,
        output=tmp_path / "output",
        native_report=None,
        detected=None,
        prior=None,
        alpha=None,
        graphic=None,
        duration=0.2,
        crop=[],
        make_comparisons=False,
    )
    args.output.mkdir()
    result = bench.assess(args, StageCache(args.output / "stages"))
    assert result["通过"] is None and result["目检"] == "未进行"
    assert result["原始残留检查"] is None
    assert not result["规格通过"]


def test_failed_assessment_keeps_earlier_evidence_and_writes_report(
    tmp_path, monkeypatch
):
    import sys

    def fail(args, phases):
        args._partial_report = {"原始残留候选": [{"帧": 7}]}
        raise RuntimeError("不应打印的配置内容")

    monkeypatch.setattr(bench, "assess", fail)
    monkeypatch.setattr(
        sys, "argv", ["基准", "--candidate", "不存在.mp4", "--output", str(tmp_path)]
    )
    with pytest.raises(SystemExit):
        bench.main()
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["原始残留候选"] == [{"帧": 7}]
    assert saved["通过"] is False
    assert "不应打印" not in (tmp_path / "report.json").read_text()


def test_upstream_token_changes_when_regeneration_changes_content(tmp_path):
    path = tmp_path / "out.txt"
    cache = StageCache(tmp_path / "stages")
    cache.run("detect", "输入", lambda: path.write_text("第一版"), lambda: [path])
    before = cache.tokens["detect"]
    path.write_text("损坏")
    cache.run("detect", "输入", lambda: path.write_text("第二版"), lambda: [path])
    assert before != cache.tokens["detect"]


def test_pipeline_speedup_option_uses_lossless_decode_and_copies_audio(
    tmp_path, monkeypatch
):
    from worker.pipeline import execute
    from scripts.hls_chunk_check import audio_hash

    monkeypatch.delenv("DETECTOR", raising=False)
    source = tmp_path / "source.mp4"
    make_video(source, seconds=1, audio=True)
    root = tmp_path / "pipeline"
    root.mkdir()
    result, info, phases = execute(
        source,
        root,
        dict(
            speedup_v2=True,
            prompts=[
                dict(frame=0, box=[20, 20, 35, 35]),
                dict(frame=29, box=[20, 20, 35, 35]),
            ],
        ),
        "telea-dev",
    )
    assert info["fps"] == 30 and not (root / "normalized.mp4").exists()
    assert audio_hash(source) == audio_hash(result)
    assert {"detect", "mask", "inpaint", "encode"} <= phases.keys()


def test_benchmark_resume_never_repeats_models_for_an_encoder_change(
    tmp_path, monkeypatch
):
    from worker import native_assets, native_chunk

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "result"
    output.mkdir()
    args = SimpleNamespace(
        source=source,
        output=output,
        duration=0.1,
        context=0.1,
        prior=None,
        alpha=None,
        template=None,
        graphic=None,
        projection_calibration=None,
        backend="numpy",
        encoder="libx264",
        path="speedup-v2",
    )
    assets = {}
    for name in ("prior", "alpha", "template", "palette"):
        path = tmp_path / (name + ".npz")
        path.write_bytes(b"auto")
        assets[name] = path
    monkeypatch.setenv("DETECTOR", "yolo")
    monkeypatch.setenv("MODEL_MANIFEST", str(tmp_path / "models.json"))
    monkeypatch.setenv("MODEL_LOCK_HASH", "固定模型测试指纹")
    monkeypatch.setattr(bench.vision, "verify_models", lambda path: "固定模型测试指纹")
    monkeypatch.setattr(
        bench.media,
        "probe",
        lambda *a, **kw: {
            "streams": [
                dict(codec_type="video", width=1920, height=1080, avg_frame_rate="60/1")
            ],
            "format": {"duration": "1"},
        },
    )
    monkeypatch.setattr(native_assets, "prepare_assets", lambda *a, **kw: assets)
    calls = []

    def shared(source, work, plan, assets, config, source_hash):
        # 验证基准只调用共用执行器，编码器没有进入模型配置身份。
        calls.append((plan, config["mode"]))
        (work / "rendered").mkdir(parents=True, exist_ok=True)
        report = dict(帧映射=[0] * 12, 原片帧索引=list(range(12)), 清晰度统计=[])
        (work / "rendered/native-report.json").write_text(json.dumps(report))
        return dict(native=report, stages={"detect": dict(seconds=0, reused=True)})

    monkeypatch.setattr(native_chunk, "process_chunk", shared)

    def encode(command, **kw):
        Path(command[-1]).write_bytes(b"encoded")

    monkeypatch.setattr(bench, "run", encode)
    from scripts import hls_chunk_check

    monkeypatch.setattr(
        hls_chunk_check,
        "mux_core_audio",
        lambda a, b, out, d: out.write_bytes(a.read_bytes()),
    )
    phases = StageCache(output / "stages")
    bench.prepare_run(args, phases)
    args.alpha = args.template = args.graphic = args.projection_calibration = None
    args.encoder = "h264_nvenc"
    bench.prepare_run(args, phases)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert args._native_run["plan"]["end"] == 6
    assert json.loads(args.native_report.read_text())["帧数"] == 6


def test_output_cannot_overwrite_baseline_or_running_directories(tmp_path):
    args = SimpleNamespace(
        output=tmp_path,
        source=tmp_path / "source.mp4",
        baseline=Path("/只读/七版.mp4"),
        run=False,
    )
    with pytest.raises(ValueError, match="覆盖"):
        bench.validate_output(args)
    args.output = Path("/workspace/video-dewatermark/bench")
    with pytest.raises(ValueError, match="正在使用"):
        bench.validate_output(args)


def test_encoded_clarity_below_point_seven_is_recorded_without_failure(
    tmp_path, monkeypatch
):
    from scripts import hls_detail_verify as detail

    source = tmp_path / "source.mp4"
    make_video(source)
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"帧映射": [i // 2 for i in range(12)]}))
    detected = tmp_path / "detected.json"
    detected.write_text(json.dumps({"scenes": [[0, 6]], "tracks": []}))
    alpha = tmp_path / "alpha.npz"
    np.savez(alpha, alpha=np.zeros((96, 128), np.float32))
    mask = np.zeros((96, 128), np.uint8)
    mask[20:40, 30:70] = 255
    supports = SimpleNamespace(
        median=mask, at=lambda *a: [dict(mask=mask, kind="fixed", opaque=False, id=1)]
    )
    monkeypatch.setattr(detail, "Supports", lambda *a: supports)
    monkeypatch.setattr(
        detail,
        "temporal_detail",
        lambda *a: dict(measurable=True, retention_ratio=0.62, passed=False),
    )
    output = tmp_path / "detail.json"
    detail.verify(
        source, source, mapping, detected, tmp_path / "prior.npz", alpha, None, output
    )
    report = json.loads(output.read_text())
    assert report["通过"] is None and report["清晰度仅记录"]
    assert report["最小保留比"] == 0.62
    assert len(report["低于零点七的样本"]) == len(report["全部样本"]) == 11


def test_worst_crops_are_selected_by_metrics_and_keep_three_sources(tmp_path):
    video = tmp_path / "source.mp4"
    make_video(video, seconds=1)
    metrics = tmp_path / "metrics"
    (metrics / "0").mkdir(parents=True)
    (metrics / "1").mkdir()
    (metrics / "regions.json").write_text(json.dumps([None, [0, 0, 64, 48]]))
    for i in range(2):
        (metrics / str(i) / "ssim.log").write_text(
            "n:1 All:0.99\nn:2 All:0.85\nn:31 All:0.90\n"
        )
    report = bench.worst_comparisons(
        video, video, video, metrics, tmp_path / "worst", limit=3
    )
    assert len(report["样本"]) == 3
    assert report["样本"][0]["SSIM"] == 0.85
    import cv2

    for sample in report["样本"]:
        image = cv2.imread(sample["图片"])
        assert image.shape[1] == (sample["区域"][2] if sample["区域"] else 128) * 3


def test_temporal_check_reports_repair_flash_without_counting_source_cut(
    tmp_path, monkeypatch
):
    from scripts import hls_speedup_quality as quality
    import cv2

    frames = [np.full((72, 128, 3), 60 if i < 6 else 180, np.uint8) for i in range(12)]
    repaired = [frame.copy() for frame in frames]
    repaired[4][:] = 255

    class Capture:
        def __init__(self, path):
            self.frames = iter(frames if path == "源片" else repaired)

        def get(self, n):
            return 60

        def read(self):
            image = next(self.frames, None)
            return image is not None, image

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    detected = tmp_path / "detected.json"
    detected.write_text(
        json.dumps(
            dict(
                scenes=[[0, 3], [3, 6]],
                hints=[
                    dict(frame=i, track_id=1, box=[0, 0, 1280, 720]) for i in range(6)
                ],
            )
        )
    )
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"帧映射": [i // 2 for i in range(12)]}))
    report = quality.temporal_candidates(
        "源片", "成品", detected, mapping, 0.2, tmp_path / "temporal.json"
    )
    assert {r["frame"] for r in report["异常候选"]} == {4, 5}
    assert any(r["frame"] == 6 and r["source_scene_cut"] for r in report["全体样本"])


def test_failure_report_keeps_completed_and_partial_stage_times(tmp_path):
    root = tmp_path / "work/stages"
    root.mkdir(parents=True)
    (root / "timings.json").write_text(
        json.dumps(
            dict(
                detect=dict(seconds=34, reused=False),
                inpaint=dict(seconds=17, reused=False),
            )
        )
    )
    (root / "detect.json").write_text("{}")
    values = bench.recover_timings(tmp_path, {})
    assert values["detect"] == dict(seconds=34, reused=False, completed=True)
    assert values["inpaint"] == dict(seconds=17, reused=False, completed=False)


def test_resumed_processing_never_reports_cold_speed_factors(tmp_path, monkeypatch):
    import sys

    source, baseline = tmp_path / "source.mp4", tmp_path / "baseline.mp4"
    source.write_bytes(b"source")
    baseline.write_bytes(b"baseline")
    output = tmp_path / "output"

    def prepare(args, phases):
        phases.timings["sam"] = dict(seconds=0, reused=True, original_seconds=356)

    monkeypatch.setattr(bench, "prepare_run", prepare)
    monkeypatch.setattr(bench, "assess", lambda *a: dict(通过=None))
    for name in [
        "PROPAINTER_ROI",
        "PROPAINTER_ROI_MARGIN",
        "SAM_SPEEDUP",
        "SAM_STATE_ON_GPU",
        "SAM_PREPARE_MAX_EDGE",
        "SAM_HINT_STRIDE",
    ]:
        monkeypatch.setenv(name, "未启用")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "基准",
            "--run",
            "--source",
            str(source),
            "--baseline",
            str(baseline),
            "--output",
            str(output),
        ],
    )
    bench.main()
    report = json.loads((output / "report.json").read_text())
    assert report["复用检查点"] and not report["可用于冷启动提速结论"]
    assert report["三倍提速目标达成"] is None
    assert report["相对分项基线倍数"] is None and report["相对整集首块倍数"] is None


def test_cold_start_rejects_all_reused_material_and_nonempty_directory(
    tmp_path, monkeypatch
):
    args = bench.parser().parse_args(
        ["--run", "--cold-start", "--output", str(tmp_path / "fresh")]
    )
    monkeypatch.delenv("HLS_V7_ASSETS", raising=False)
    bench.validate_output(args)
    args.prior = tmp_path / "old-prior.npz"
    with pytest.raises(ValueError, match="禁止复用"):
        bench.validate_output(args)
    args.prior = None
    monkeypatch.setenv("HLS_V7_ASSETS", "旧素材清单")
    with pytest.raises(ValueError, match="禁止复用"):
        bench.validate_output(args)
    monkeypatch.delenv("HLS_V7_ASSETS")
    args.v7_inputs = tmp_path / "冻结素材.json"
    with pytest.raises(ValueError, match="禁止复用"):
        bench.validate_output(args)
    args.v7_inputs = None
    args.comparison_assets = tmp_path / "仅供对照.json"
    bench.validate_output(args)
    args.output.mkdir()
    (args.output / "report.json").write_text("{}")
    with pytest.raises(ValueError, match="全新空"):
        bench.validate_output(args)


def test_three_column_comparison_is_720p_and_under_size_limit(tmp_path):
    import cv2

    videos = []
    for name, color in [
        ("source", "red"),
        ("baseline", "green"),
        ("candidate", "blue"),
    ]:
        path = tmp_path / (name + ".mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color={color}:size=128x96:rate=30:duration=0.2",
                "-c:v",
                "libx264",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        videos.append(path)
    report = bench.comparisons(
        videos[0], videos[2], videos[1], tmp_path / "compare", 0.2, []
    )
    cap = cv2.VideoCapture(report["并排视频"])
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (720, 1280)
    assert frame[360, 200, 2] > 200
    assert frame[360, 600, 1] > 100
    assert frame[360, 1000, 0] > 200
    assert report["视频字节数"] < 50_000_000
    assert report["视频列顺序"] == "原片、七版、提速成品"


def test_canonical_native_core_uses_stream_copy_without_changing_pixels_or_audio(
    tmp_path, monkeypatch
):
    import cv2
    from worker import native_assets, native_chunk, hls_native
    from scripts.hls_chunk_check import audio_hash

    source = tmp_path / "source.mp4"
    make_video(source, seconds=0.4, audio=True)
    args = bench.parser().parse_args(
        [
            "--run",
            "--cold-start",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "out"),
            "--duration",
            "0.2",
            "--context",
            "0.2",
        ]
    )
    monkeypatch.setenv("DETECTOR", "yolo")
    monkeypatch.setenv("MODEL_MANIFEST", "fixed-model")
    monkeypatch.setattr(bench.vision, "verify_models", lambda p: "fixed")
    monkeypatch.setattr(hls_native, "validate_source", lambda info: None)
    inputs = {}
    for name in ("prior", "alpha", "template", "palette"):
        p = tmp_path / (name + ".npz")
        p.write_bytes(name.encode())
        inputs[name] = p
    monkeypatch.setattr(native_assets, "prepare_assets", lambda *a, **kw: inputs)

    def shared(source, work, plan, assets, config, source_hash):
        assert not config["materialize_frames"]
        (work / "rendered").mkdir(parents=True)
        report = dict(
            帧映射=list(range(24)), 原片帧索引=list(range(24)), 帧数=24, 清晰度统计=[]
        )
        (work / "rendered/native-report.json").write_text(json.dumps(report))
        return dict(native=report, stages={}, native_video=str(source))

    monkeypatch.setattr(native_chunk, "process_chunk", shared)
    phases = StageCache(args.output / "stages")
    bench.prepare_run(args, phases)
    info = probe(args.candidate, count=True)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(video["nb_read_frames"]) == 12
    assert audio_hash(source, 0.2) == audio_hash(args.candidate)
    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(args.candidate))
    try:
        for i in range(12):
            ok, x = a.read()
            good, y = b.read()
            assert ok and good and np.array_equal(x, y)
        assert not b.read()[0]
    finally:
        a.release()
        b.release()
