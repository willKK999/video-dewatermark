"""整集原生交付必须在不运行真实模型的情况下验证分块、缓存、回退及交付合同。"""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import cv2
import numpy as np
import pytest
from botocore.exceptions import ClientError
from worker import hls_native, native_chunk
from worker.native_checkpoint import BundleStore
from worker.process import PipelineError, run
from worker.media import probe


class MemoryStore:
    def __init__(self, objects=None):
        self.objects = objects or {}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def upload_file(self, path, bucket, key, **kwargs):
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, bucket, key, path, **kwargs):
        Path(path).write_bytes(self.get_object(bucket, key)["Body"].read())

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body


def test_default_and_explicit_rollback_ignore_fast_environment(monkeypatch):
    for key in ("HLS_NATIVE_PATH", "HLS_COMPOSITE_BACKEND", "HLS_VIDEO_ENCODER"):
        monkeypatch.delenv(key, raising=False)
    assert hls_native.configuration({}) == dict(
        mode="speedup-v2", fast=True, backend="torch-cuda", encoder="libx264"
    )
    monkeypatch.setenv("SAM_SPEEDUP", "true")
    monkeypatch.setenv("PROPAINTER_ROI", "true")
    monkeypatch.setenv("HLS_VIDEO_ENCODER", "h264_nvenc")
    assert hls_native.configuration({"hls_native_path": "v7"}) == dict(
        mode="v7", fast=False, backend="numpy", encoder="libx264"
    )
    monkeypatch.setenv("HLS_NATIVE_PATH", "v7")
    assert not hls_native.configuration({})["fast"]
    assert hls_native.configuration({"hls_native_path": "speedup-v2"})["fast"]


@pytest.mark.parametrize(
    "options", [{"hls_native_path": "invalid"}, {"hls_encoder": "invalid"}]
)
def test_invalid_delivery_configuration_fails_without_fallback(options):
    with pytest.raises((ValueError, PipelineError)):
        hls_native.configuration(options)


def test_native_source_rejects_fake_upscale_and_rate_conversion():
    info = {
        "streams": [
            dict(codec_type="video", width=1920, height=1080, avg_frame_rate="60/1")
        ]
    }
    hls_native.validate_source(info)
    for field, value in [("width", 1280), ("avg_frame_rate", "30/1")]:
        video = dict(info["streams"][0], **{field: value})
        with pytest.raises(PipelineError, match="禁止放大"):
            hls_native.validate_source({"streams": [video]})


def test_checkpoint_is_portable_validated_and_committed_after_upload(tmp_path):
    store = MemoryStore()
    cache = BundleStore(store, "bucket", "safe/native")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "masks").mkdir()
    (a / "chunk.json").write_text('{"块":1}')
    cv2.imwrite(str(a / "masks/000000.png"), np.zeros((20, 30), np.uint8))
    cache.save("chunk-0000", a, [a / "chunk.json", a / "masks"], "输入指纹")
    assert cache.restore("chunk-0000", b, "输入指纹")
    assert (b / "chunk.json").read_bytes() == (a / "chunk.json").read_bytes()
    assert not cache.restore("chunk-0000", b, "不同指纹")
    marker = json.loads(store.objects["safe/native/chunk-0000.json"])
    key = "safe/native/chunk-0000-" + marker["sha256"] + ".zip"
    store.objects[key] += b"corrupt"
    assert not cache.restore("chunk-0000", b, "输入指纹")


def test_checkpoint_rejects_traversal_before_extracting(tmp_path):
    store = MemoryStore()
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as out:
        payload = b"xx"
        out.writestr("../escaped.json", payload)
        out.writestr(
            "bundle.json",
            json.dumps(
                dict(
                    identity="输入",
                    files={"../escaped.json": hashlib.sha256(payload).hexdigest()},
                )
            ),
        )
    digest = hashlib.sha256(data.getvalue()).hexdigest()
    store.objects["p/c.json"] = json.dumps(
        dict(identity="输入", sha256=digest)
    ).encode()
    store.objects[f"p/c-{digest}.zip"] = data.getvalue()
    root = tmp_path / "safe"
    root.mkdir()
    with pytest.raises(PipelineError, match="路径"):
        BundleStore(store, "bucket", "p").restore("c", root, "输入")
    assert not (tmp_path / "escaped.json").exists()


def test_encoder_does_not_invalidate_native_block_cache(monkeypatch):
    c = hls_native.configuration({})
    first = hls_native.identity("源片", dict(hls=True), c, "固定环境")
    monkeypatch.setenv("HLS_VIDEO_ENCODER", "h264_nvenc")
    c2 = hls_native.configuration({"hls_encoder": "h264_nvenc"})
    assert (
        hls_native.identity(
            "源片", dict(hls=True, hls_encoder="h264_nvenc"), c2, "固定环境"
        )
        == first
    )
    rollback = hls_native.configuration({"hls_native_path": "v7"})
    assert hls_native.identity("源片", dict(hls=True), rollback, "固定环境") != first


def test_native_overlap_has_same_hard_limit_and_never_blends_uncertain_frame(tmp_path):
    joined, incoming = tmp_path / "joined", tmp_path / "incoming"
    joined.mkdir()
    incoming.mkdir()
    cv2.imwrite(str(joined / "00000000.png"), np.full((20, 30, 3), 50, np.uint8))
    cv2.imwrite(str(incoming / "000000.png"), np.full((20, 30, 3), 100, np.uint8))
    with pytest.raises(PipelineError, match="三十五"):
        hls_native.overlap_metrics(
            joined, incoming, dict(left=0, right=1), 1, lambda: None
        )


@pytest.mark.parametrize("fast", [True, False])
def test_explicit_sam_mode_overrides_environment(tmp_path, monkeypatch, fast):
    from worker import vision

    monkeypatch.setenv("SAM_SPEEDUP", "true")
    monkeypatch.setenv("TRACK_PYTHON", "模型替身")
    calls = []
    monkeypatch.setattr(vision, "run", lambda args, *a, **kw: calls.append(args))
    monkeypatch.setattr(vision, "validate_masks", lambda *a: [])
    vision.masks(tmp_path, [], {}, "propainter", speedup=fast)
    assert "--speedup" not in calls[0] and "--no-speedup" not in calls[0]


def test_explicit_roi_false_does_not_use_fast_environment(tmp_path, monkeypatch):
    from worker import vision, roi_inpaint

    monkeypatch.setenv("PROPAINTER_ROI", "true")
    monkeypatch.setattr(
        roi_inpaint, "inpaint_roi", lambda *a: pytest.fail("回退路径错误使用局部补画")
    )
    # 到旧路径后首先读取掩膜；用专门异常验证分派，不启动模型。
    monkeypatch.setattr(
        vision, "read", lambda *a: (_ for _ in ()).throw(ValueError("七版路径"))
    )
    with pytest.raises(ValueError, match="七版路径"):
        vision.inpaint(
            tmp_path, [tmp_path / "0.png"], [[0, 1]], "propainter", roi=False
        )


def test_native_render_frames_only_preserves_source_offset_without_encoding(
    tmp_path, monkeypatch
):
    from scripts import hls_native_chunk as renderer

    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 60, (96, 64))
    for i in range(20):
        writer.write(np.full((64, 96, 3), i * 10, np.uint8))
    writer.release()
    coarse = tmp_path / "work"
    for folder in ("frames", "composite"):
        (coarse / folder).mkdir(parents=True)
    for i in range(8):
        im = np.full((64, 96, 3), (4 + i * 2) * 10, np.uint8)
        for folder in ("frames", "composite"):
            cv2.imwrite(str(coarse / folder / f"{i:06d}.png"), im)
    alpha, template = tmp_path / "alpha.npz", tmp_path / "template.npz"
    np.savez(alpha, alpha=np.zeros((64, 96), np.float32))
    np.savez(template, template=np.zeros((64, 96), np.float32))
    detected = tmp_path / "detected.json"
    detected.write_text('{"scenes":[[0,8]]}')
    monkeypatch.setattr(
        renderer,
        "Supports",
        lambda *a: SimpleNamespace(median=np.zeros((64, 96)), at=lambda *a: []),
    )
    dest = tmp_path / "render"
    renderer.render(
        source,
        coarse,
        detected,
        tmp_path / "prior.npz",
        alpha,
        template,
        None,
        dest,
        duration=0.1,
        output_fps=60,
        frames_only=True,
        source_start_frame=4,
    )
    frames = sorted((dest / "frames").glob("*.png"))
    assert len(frames) == 6 and not (dest / "chunk1.mp4").exists()
    for n, path in enumerate(frames):
        assert cv2.imread(str(path)).mean() == pytest.approx((4 + n) * 10, abs=2)
    saved = json.loads((dest / "native-report.json").read_text())
    assert (
        saved["原片起始帧"] == 4
        and saved["无损帧交付"]
        and saved["音频内容一致"] is None
    )


def test_odd_native_boundaries_decode_with_local_sampling_phase(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 60, (96, 64))
    for i in range(20):
        writer.write(np.full((64, 96, 3), i * 10, np.uint8))
    writer.release()
    work = tmp_path / "work"
    work.mkdir()
    files = native_chunk.decode_chunk(source, work, dict(left=3, right=14))
    assert 5 <= len(files) <= 6
    for i, path in enumerate(files):
        assert cv2.imread(str(path)).mean() == pytest.approx((3 + i * 2) * 10, abs=2)


@pytest.mark.parametrize("residual", [0, 1])
@pytest.mark.parametrize("enforce", [False, True])
def test_encoded_audit_keeps_raw_evidence_and_does_not_gate_clarity(
    tmp_path, monkeypatch, residual, enforce
):
    from worker.tests.test_speedup_bench import make_video

    source = tmp_path / "source.mp4"
    make_video(source)
    work = tmp_path / "work"
    (work / "frames").mkdir(parents=True)
    (work / "graphic").mkdir()
    (work / "rendered").mkdir()
    (work / "graphic/brands.json").write_text("[]")
    (work / "rendered/native-report.json").write_text(
        json.dumps({"帧映射": [i // 2 for i in range(12)]})
    )
    hints = [dict(frame=i, track_id=1, box=[20, 20, 40, 30]) for i in range(6)]
    (work / "detections.json").write_text(
        json.dumps(
            dict(
                tracks=[dict(id=1, kind="fixed", hints=hints)],
                hints=hints,
                scenes=[[0, 6]],
                prior="旧位置",
            )
        )
    )
    mask = np.zeros((96, 128), np.uint8)
    mask[20:40, 30:70] = 255
    for i in range(6):
        cv2.imwrite(
            str(work / "frames" / f"{i:06d}.png"), np.full((96, 128, 3), 80, np.uint8)
        )
    alpha = tmp_path / "alpha.npz"
    np.savez(alpha, alpha=np.zeros_like(mask))
    monkeypatch.setattr(
        native_chunk,
        "Supports",
        lambda *a: SimpleNamespace(
            at=lambda *a: [dict(mask=mask, opaque=False, kind="fixed", id=1)]
        ),
    )
    from worker import matte

    monkeypatch.setattr(
        matte,
        "temporal_detail",
        lambda *a: dict(measurable=True, retention_ratio=0.62, passed=False),
    )

    def quality(root, files, refs, check):
        assert len(files) == 12 and refs["tracks"][0]["hints"][-1]["frame"] == 11
        (root / "quality.json").write_text(
            json.dumps(
                dict(
                    residual_count=residual,
                    needs_review=bool(residual),
                    moving_evidence=[{"帧": 7}] if residual else [],
                )
            )
        )
        if residual:
            raise PipelineError("RESIDUAL_UNCERTAIN", "残留必须阻止交付", review=True)

    from worker import native_assets

    monkeypatch.setattr(
        native_assets, "residual_candidates", lambda *a, **kw: dict(残留候选=[])
    )
    monkeypatch.setattr(native_chunk.vision, "residual_quality", quality)
    plan = dict(left=0, right=12, start=0, end=12)
    if residual and enforce:
        with pytest.raises(PipelineError, match="阻止交付"):
            native_chunk.audit_chunk(
                source,
                source,
                work,
                plan,
                dict(prior=tmp_path / "prior.npz", alpha=alpha),
            )
    else:
        saved = native_chunk.audit_chunk(
            source,
            source,
            work,
            plan,
            dict(prior=tmp_path / "prior.npz", alpha=alpha),
            enforce=enforce,
        )
        assert saved["通过"] is None and len(saved["低于零点七的样本"]) == 11
        assert saved["硬门槛通过"] == (residual == 0)
    saved = json.loads((work / "audit/native-quality.json").read_text())
    assert saved["原始残留检查"]["residual_count"] == residual


def test_no_translucent_fixed_mark_does_not_require_fake_calibration(
    tmp_path, monkeypatch
):
    from scripts import hls_fit_projection

    work = tmp_path / "work"
    (work / "frames").mkdir(parents=True)
    cv2.imwrite(str(work / "frames/000000.png"), np.zeros((32, 48, 3), np.uint8))
    detected = tmp_path / "detections.json"
    detected.write_text("{}")
    alpha = tmp_path / "alpha.npz"
    np.savez(alpha, alpha=np.zeros((32, 48), np.float32))
    template = tmp_path / "template.npz"
    np.savez(template, template=np.zeros((32, 48), np.float32))
    monkeypatch.setattr(
        hls_fit_projection, "Supports", lambda *a: SimpleNamespace(at=lambda *a: [])
    )
    out = tmp_path / "projection.json"
    hls_fit_projection.fit(
        work, tmp_path, detected, tmp_path / "prior.npz", alpha, template, None, out
    )
    assert json.loads(out.read_text()) == {}
    assert json.loads(out.with_suffix(".samples.json").read_text())["样本数"] == 0


def test_real_native_hls_delivery_retries_only_failed_block_and_preserves_audio(
    tmp_path, monkeypatch
):
    """只替换模型和素材发现，真实执行两块、无损检查点、最终编码、分片及音频时序核验。"""
    from worker import hls
    from scripts.hls_chunk_check import audio_hash

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
            "color=c=gray:size=1920x1080:rate=60:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "10",
            "-c:a",
            "aac",
            source,
        ]
    )
    input_hls = tmp_path / "input"
    input_hls.mkdir()
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            source,
            "-c:v",
            "libx264",
            "-force_key_frames",
            "0,0.4,0.8",
            "-sc_threshold",
            "0",
            "-c:a",
            "copy",
            "-hls_segment_options",
            "pes_payload_size=0",
            "-f",
            "hls",
            "-hls_time",
            "0.4",
            "-hls_list_size",
            "0",
            input_hls / "index.m3u8",
        ]
    )
    store = MemoryStore({p.name: p.read_bytes() for p in input_hls.iterdir()})
    monkeypatch.setenv("DETECTOR", "yolo")
    monkeypatch.setenv("HLS_COMPOSITE_BACKEND", "numpy")
    monkeypatch.delenv("HLS_NATIVE_PATH", raising=False)
    monkeypatch.delenv("HLS_VIDEO_ENCODER", raising=False)
    asset_calls = []

    def assets(source, root, *args, **kwargs):
        asset_calls.append(1)
        root.mkdir(exist_ok=True)
        result = hls_native.assets_from(root)
        for p in result.values():
            p.write_bytes(b"fixed-auto-asset")
        return result

    monkeypatch.setattr(hls_native, "prepare_assets", assets)
    from worker import multitrack

    monkeypatch.setattr(
        multitrack,
        "combine_tracks",
        lambda d, p: dict(
            d,
            multitrack=True,
            tracks=[dict(id=1, kind="fixed", hints=d["hints"])],
            prior=str(p),
        ),
    )
    # 两块都携带原片同一秒上下文，主体边界在原分片零点四秒。
    monkeypatch.setattr(
        hls,
        "chunks",
        lambda durations, fps: [
            dict(left=0, right=60, start=0, end=24),
            dict(left=0, right=60, start=24, end=60),
        ],
    )
    calls = []

    def chunk(source, work, plan, assets, config, *args):
        calls.append(plan["start"])
        if len(calls) == 2:
            raise PipelineError("STORAGE_UNAVAILABLE", "模拟第二块失败", retryable=True)
        assert config["mode"] == "speedup-v2"
        (work / "rendered/frames").mkdir(parents=True)
        cap = cv2.VideoCapture(str(source))
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            cv2.imwrite(str(work / "rendered/frames" / f"{count:06d}.png"), image)
            count += 1
        cap.release()
        assert count == 60
        hints = [
            dict(frame=i, track_id=1, box=[50, 50, 100, 80], confidence=0.9)
            for i in range(30)
        ]
        (work / "detections.json").write_text(
            json.dumps(
                dict(
                    hints=hints,
                    tracks=[dict(id=1, kind="fixed", hints=hints)],
                    scenes=[[0, 30]],
                    prior=str(assets["prior"]),
                )
            )
        )
        (work / "graphic").mkdir()
        (work / "graphic/brands.json").write_text("[]")
        (work / "masks").mkdir()
        (work / "performance.json").write_text("{}")
        (work / "projection.json").write_text("{}")
        native = dict(
            帧映射=[i // 2 for i in range(60)],
            清晰度最小保留比=None,
            清晰度统计=[],
            原片起始帧=0,
        )
        (work / "rendered/native-report.json").write_text(json.dumps(native))
        summary = dict(
            native=native, stages={"detect": dict(seconds=0.01, reused=False)}
        )
        (work / "chunk.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(hls_native, "process_chunk", chunk)
    audits = []

    def audit(source, output, work, plan, assets, check):
        audits.append(plan["start"])
        assert output.exists()
        return dict(
            清晰度统计=[],
            原始残留检查=dict(residual_count=0, needs_review=False),
            品牌残留候选=[],
            闪烁指标=[],
        )

    monkeypatch.setattr(hls_native, "audit_chunk", audit)
    first = tmp_path / "attempt-1"
    first.mkdir()
    args = [
        dict(hls=True),
        "propainter",
        False,
        lambda _: None,
        lambda: None,
        store,
        "bucket",
        "cache",
        lambda *a: None,
    ]
    with pytest.raises(PipelineError, match="第二块失败"):
        hls.execute_hls("index.m3u8", first, *args)
    second = tmp_path / "attempt-2"
    second.mkdir()
    output, meta, _ = hls.execute_hls("index.m3u8", second, *args)
    assert calls == [0, 24, 24] and len(asset_calls) == 2 and audits == [0, 24]
    metrics = json.loads((second / "metrics-hls.json").read_text())
    assert [c["reused"] for c in metrics["chunks"]] == [True, False]
    assert meta["fps"] == 60 and (meta["width"], meta["height"]) == (1920, 1080)
    v = next(
        s for s in probe(output, count=True)["streams"] if s["codec_type"] == "video"
    )
    assert int(v["nb_read_frames"]) == 60
    assert (
        audio_hash(second / "source.mp4")
        == audio_hash(output)
        == audio_hash(second / "hls/index.m3u8")
        == audio_hash(second / "merged-from-hls.mp4")
    )
    report = json.loads((second / "report.json").read_text())
    assert (
        report["通过"] is None
        and report["目检"] == "未进行"
        and report["无损合并音频压缩数据一致"]
    )
    assert output.read_bytes() == (second / "merged-from-hls.mp4").read_bytes()
    assert (second / "detections.json").exists()
    assert (second / "multitrack-prior.npz").exists()
    assert (second / "comparisons/source-candidate.mp4").exists()
    assert (
        len(metrics["segment_timing"]) == 3
        and metrics["audio_timeline_max_error_ms"] <= 1
    )
    assert "块号=2" in (second / "progress.log").read_text()


def test_all_candidate_mode_keeps_each_raw_box_and_original_threshold(
    tmp_path, monkeypatch
):
    import sys, types
    from worker import yolo_stage

    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(2):
        cv2.imwrite(str(frames / f"{i:06d}.png"), np.zeros((20, 30, 3), np.uint8))
    weights = tmp_path / "test.pt"
    weights.write_bytes(b"model-stub")
    indices = tmp_path / "indices.json"
    indices.write_text("[0,1]")
    out = tmp_path / "raw.json"

    class Coordinates:
        def cpu(self):
            return self

        def tolist(self):
            return [1, 2, 10, 12]

    calls = []

    class Model:
        names = {0: "corner_logo", 1: "float_logo", 2: "float_text"}

        def predict(self, file, **kwargs):
            calls.append(kwargs)
            return [
                SimpleNamespace(
                    boxes=[
                        SimpleNamespace(cls=[i], conf=[0.8], xyxy=[Coordinates()])
                        for i in range(3)
                    ]
                )
            ]

    module = types.ModuleType("ultralytics")
    module.YOLO = lambda path: Model()
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setenv("YOLO_CONFIDENCE", "0.35")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "检测",
            "--frames",
            str(frames),
            "--output",
            str(out),
            "--weights",
            str(weights),
            "--indices",
            str(indices),
            "--all-candidates",
        ],
    )
    yolo_stage.main()
    rows = json.loads(out.read_text())
    assert [p["frame"] for p in rows] == [0, 1]
    assert all(len(p["boxes"]) == 3 for p in rows)
    assert all(p["conf"] == 0.35 for p in calls)


def test_asset_preparation_delegates_to_original_recipe_without_old_inputs(
    tmp_path, monkeypatch
):
    from worker import native_assets, v7_rebuild

    calls = []
    monkeypatch.delenv("HLS_V7_ASSETS", raising=False)
    monkeypatch.setattr(
        v7_rebuild, "rebuild", lambda *a: calls.append(a) or {"本次": True}
    )
    result = native_assets.prepare_assets(
        tmp_path / "source.mp4", tmp_path / "assets", "原片摘要"
    )
    assert result == {"本次": True} and len(calls) == 1
    assert calls[0][:3] == (tmp_path / "source.mp4", tmp_path / "assets", "原片摘要")


def test_raw_residual_keeps_subtitles_and_uses_existing_hls_threshold(
    tmp_path, monkeypatch
):
    from worker import native_assets

    audit = tmp_path / "audit"
    (audit / "composite").mkdir(parents=True)
    image = np.full((20, 30, 3), 80, np.uint8)
    image[:10, :10] = 200
    cv2.imwrite(str(audit / "composite/000000.png"), image)
    raw = [
        dict(
            frame=0,
            boxes=[
                dict(class_id=0, confidence=0.2, box=[0, 10, 8, 18]),
                dict(class_id=2, confidence=0.3, box=[0, 0, 10, 10]),
                dict(class_id=2, confidence=0.4, box=[12, 0, 22, 10]),
            ],
        )
    ]

    def detector(args, *a, **kw):
        assert args[args.index("--confidence") + 1] == "0.15"
        assert "--all-candidates" in args
        Path(args[-1]).write_text(json.dumps(raw))

    monkeypatch.setattr(native_assets, "run", detector)
    monkeypatch.setenv("DETECT_PYTHON", "替身")
    monkeypatch.setenv("YOLO_WEIGHTS", "替身.pt")
    monkeypatch.setenv("RESIDUAL_CONFIDENCE", "0.35")
    monkeypatch.setenv("HLS_RESIDUAL_CONFIDENCE", "0.15")
    monkeypatch.setattr(
        native_assets,
        "load_prior",
        lambda *a: (
            np.zeros((20, 30), np.float32),
            None,
            [dict(box=[0, 0, 10, 10])],
            None,
            None,
        ),
    )
    from worker import branding

    monkeypatch.setattr(
        branding,
        "recognize",
        lambda image, psm=7: "examplebrand.com"
        if psm == 11 or image.mean() > 100
        else "普通字幕",
    )
    report = native_assets.residual_candidates(
        audit, tmp_path, dict(prior=tmp_path / "prior.npz")
    )
    assert len(report["残留候选"]) == 1
    assert report["原始候选总数"] == 3 and report["需要复核"]
    assert (
        len(report["文字复核依据"]) == 2
        and "字幕" in report["文字复核依据"][0]["basis"]
    )
    assert json.loads((audit / "raw-residual-candidates.json").read_text()) == raw


def test_empty_palette_keeps_main_v7_failure_behavior():
    from worker.branding import palette_support

    with pytest.raises(cv2.error):
        palette_support(
            np.full((20, 30, 3), 255, np.uint8), np.empty((0, 3), np.float32)
        )


def test_failed_upload_keeps_previous_checkpoint_marker(tmp_path, monkeypatch):
    store = MemoryStore()
    cache = BundleStore(store, "bucket", "safe")
    source = tmp_path / "chunk.json"
    source.write_text("第一版")
    cache.save("chunk", tmp_path, [source], "输入")
    before = store.objects["safe/chunk.json"]
    source.write_text("第二版")

    def failed(*a, **kw):
        raise PipelineError("STORAGE_UNAVAILABLE", "模拟上传中断", retryable=True)

    monkeypatch.setattr(store, "upload_file", failed)
    with pytest.raises(PipelineError, match="上传中断"):
        cache.save("chunk", tmp_path, [source], "输入")
    assert store.objects["safe/chunk.json"] == before


def test_nvenc_forces_independent_idr_at_hls_boundaries():
    from worker.speedup import encoder_args

    args = encoder_args("h264_nvenc", 10)
    assert args[args.index("-forced-idr") + 1] == "1"
    assert args[args.index("-no-scenecut") + 1] == "1"


def test_native_export_includes_report_progress_comparison_and_raw_evidence(tmp_path):
    from scripts.hls_export import export_native_artifacts

    prefix = "task/native"
    report = dict(
        通过=None,
        目检="未进行",
        对比产物={
            "并排视频": "comparisons/source-candidate.mp4",
            "图片": ["comparisons/5-左上.png"],
        },
        原始残留检查=[{"原始证据目录": "native-chunk-0000/audit/"}],
    )
    objects = {
        prefix + "/report.json": json.dumps(report).encode(),
        prefix + "/progress.log": "块号=1".encode(),
        prefix + "/merged-from-hls.mp4": b"video",
        prefix + "/comparisons/source-candidate.mp4": b"video",
        prefix + "/comparisons/5-左上.png": b"image",
    }
    for name in (
        "quality.json",
        "quality-detections.json",
        "native-quality.json",
        "raw-residual-candidates.json",
        "raw-residual-review.json",
    ):
        objects[prefix + "/native-chunk-0000/audit/" + name] = b"{}"
    result = export_native_artifacts(MemoryStore(objects), "bucket", prefix, tmp_path)
    assert not result["缺失核验证据"] and len(result["已导出"]) == 9
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert (tmp_path / "progress.log").read_text() == "块号=1"
    assert (tmp_path / "native-chunk-0000/audit/raw-residual-candidates.json").exists()


def test_native_export_rejects_report_path_escape(tmp_path):
    from scripts.hls_export import export_native_artifacts

    report = dict(对比产物={"并排视频": "../outside.mp4"})
    store = MemoryStore({"p/report.json": json.dumps(report).encode()})
    with pytest.raises(PipelineError, match="路径无效"):
        export_native_artifacts(store, "bucket", "p", tmp_path)
    assert not (tmp_path.parent / "outside.mp4").exists()


@pytest.mark.parametrize("kind", ["prior", "profile", "environment"])
def test_frozen_material_is_never_accepted_as_processing_input(
    tmp_path, monkeypatch, kind
):
    from worker.native_assets import prepare_assets

    monkeypatch.delenv("HLS_V7_ASSETS", raising=False)
    kwargs = {}
    if kind == "environment":
        monkeypatch.setenv("HLS_V7_ASSETS", str(tmp_path / "must-not-read.json"))
    else:
        kwargs[kind] = tmp_path / "must-not-read.json"
    with pytest.raises(PipelineError, match="禁止"):
        prepare_assets(None, tmp_path / "assets", "原片摘要", **kwargs)
    assert not (tmp_path / "assets").exists()
