"""提速路径须保持覆盖、原像素、跟踪边界和七版纹理数学行为。"""

import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from worker.roi import plan_rois, paste_roi
from worker.speedup import bounded_map, encoder_args
from worker.sam_schedule import sampled_hints, groups_for_scene


def test_roi_covers_every_moving_pixel_and_keeps_distant_marks_separate():
    masks = []
    for i in range(12):
        m = np.zeros((1080, 1920), np.uint8)
        m[50:80, 50 + i * 5 : 100 + i * 5] = 255
        m[950:980, 1750:1800] = 255
        masks.append(m)
    boxes = plan_rois(masks)
    assert len(boxes) == 2
    support = np.zeros_like(masks[0])
    for x, y, a, b in boxes:
        assert (a - x) % 8 == (b - y) % 8 == 0
        support[y:b, x:a] = 255
    assert all(not np.any((m > 0) & (support == 0)) for m in masks)
    assert np.count_nonzero(support) < support.size / 3


def test_roi_transitive_merge_edges_empty_and_full_fallback():
    m = np.zeros((256, 600), np.uint8)
    m[:10, :10] = 255
    m[5:10, 170:180] = 255
    m[5:10, 330:340] = 255
    assert len(plan_rois([m])) == 1
    assert plan_rois([np.zeros_like(m)]) == []
    assert plan_rois([np.full_like(m, 255)]) == [(0, 0, 600, 256)]
    with pytest.raises(ValueError):
        plan_rois([m, m[:5]])


def test_paste_does_not_touch_any_pixel_outside_original_mask():
    source = np.full((200, 300, 3), 50, np.uint8)
    m = np.zeros((200, 300), np.uint8)
    m[30:60, 80:120] = 255
    out = source.copy()
    for box in plan_rois([m]):
        x, y, a, b = box
        paste_roi(out, np.full((b - y, a - x, 3), 140, np.uint8), m, box)
    assert np.array_equal(out[m == 0], source[m == 0])
    assert np.all(out[m > 0] == 140)


def test_prefetch_is_bounded_ordered_and_propagates_failures():
    read = []

    def prepare(i):
        read.append(i)
        if i == 4:
            raise ValueError("准备失败")
        return i * 3

    generator = bounded_map(prepare, range(100), depth=2)
    assert next(generator) == 0
    assert len(read) <= 3
    assert [next(generator) for _ in range(3)] == [3, 6, 9]
    with pytest.raises(ValueError, match="准备失败"):
        next(generator)
    assert len(read) <= 7


def test_encoding_policy_copies_audio_elsewhere_and_does_not_cap_bitrate():
    cpu = encoder_args("libx264", 10)
    gpu = encoder_args("h264_nvenc", 10)
    assert cpu[-2:] == ["-crf", "10"]
    assert "p7" in gpu and "constqp" in gpu
    assert "-b:v" not in gpu and "-c:a" not in gpu
    with pytest.raises(ValueError):
        encoder_args("unknown")


def test_v7_keys_keep_original_stride_and_lifetimes():
    hints = [dict(frame=i, track_id=1, box=[10, 10, 30, 30]) for i in range(100)]
    hints[23]["box"] = [80, 80, 100, 100]
    hints += [dict(frame=i, track_id=2, box=[10, 10, 30, 30]) for i in range(41, 65)]
    for a, b, points in groups_for_scene(dict(multitrack=True, hints=hints), 0, 70):
        actual = sampled_hints(points, a, b)
        assert actual == [
            p for p in points if (p["frame"] - a) % 30 == 0 or p["frame"] in (a, b - 1)
        ]
        if a == 0:
            assert 23 not in [p["frame"] for p in actual]


@pytest.mark.parametrize("sigma", [1.2, 3.0])
@pytest.mark.parametrize("calibrated", [False, True])
def test_torch_projection_matches_v7_and_retains_unmasked_pixels(sigma, calibrated):
    torch = pytest.importorskip("torch")
    from worker.torch_composite import TorchCompositor
    from worker.matte import projected_texture_composite, solve_projection

    torch.set_num_threads(2)
    rng = np.random.default_rng(102)
    background = rng.integers(80, 150, (75, 113, 3), dtype=np.uint8)
    glyphs = []
    for word, origin in [("WM", (2, 40)), ("AB", (42, 61))]:
        g = np.zeros((75, 113), np.float32)
        cv2.putText(g, word, origin, 0, 1.0, 1, 2)
        glyphs.append(g - cv2.GaussianBlur(g, (0, 0), sigma))
    source = np.clip(
        background.astype(float)
        + glyphs[0][..., None] * 20
        + glyphs[1][..., None] * 15,
        0,
        255,
    ).astype(np.uint8)
    mask = np.zeros((75, 113), np.uint8)
    mask[8:74, 5:110] = 255
    base = cv2.GaussianBlur(background, (0, 0), 3)
    calibration = None
    if calibrated:
        acc = {}
        projected_texture_composite(
            source, base, mask, glyphs[:1], accumulator=acc, detail_sigma=sigma
        )
        calibration = solve_projection(acc)
    exclusion = np.zeros_like(mask)
    exclusion[30:33, 30:33] = 255
    expected = projected_texture_composite(
        source,
        base,
        mask,
        glyphs,
        detail_exclusion=exclusion,
        calibration=calibration,
        detail_sigma=sigma,
    )
    actual = TorchCompositor("cpu", tile_batch=7).project(
        source,
        base,
        mask,
        glyphs,
        detail_exclusion=exclusion,
        calibration=calibration,
        detail_sigma=sigma,
    )
    error = np.abs(actual.astype(float) - expected)
    assert error.max() <= 1
    assert error.mean() < 0.01
    assert np.array_equal(actual[mask == 0], source[mask == 0])


def test_torch_soft_and_empty_projection_match_v7():
    pytest.importorskip("torch")
    from worker.torch_composite import TorchCompositor
    from worker.restoration import soft_composite

    rng = np.random.default_rng(5)
    x = rng.integers(0, 256, (60, 100, 3), dtype=np.uint8)
    y = np.roll(x, 2, axis=1)
    m = np.zeros((60, 100), np.uint8)
    m[10:50, 20:80] = 255
    engine = TorchCompositor("cpu")
    assert np.array_equal(engine.soft(x, y, m), soft_composite(x, y, m))
    assert np.array_equal(engine.project(x, y, m * 0, m.astype(np.float32)), x)


def test_roi_runner_caches_model_work_and_invalidates_changed_masks(
    tmp_path, monkeypatch
):
    from worker import roi_inpaint
    from worker.vision import inpaint

    for name in ("frames", "masks"):
        (tmp_path / name).mkdir()
    files = []
    image = np.full((256, 512, 3), 80, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[50:55, 60:70] = 255
    for i in range(3):
        path = tmp_path / "frames" / f"{i:06d}.png"
        cv2.imwrite(str(path), image)
        cv2.imwrite(str(tmp_path / "masks" / path.name), mask)
        files.append(path)
    monkeypatch.setenv("PROPAINTER_ROI", "true")
    monkeypatch.setenv("INPAINT_PYTHON", "python")
    monkeypatch.setenv("PROPAINTER_ROOT", str(tmp_path / "model"))
    monkeypatch.setattr(
        roi_inpaint,
        "CleanReferences",
        lambda *a: SimpleNamespace(restore=lambda x, m, y: y),
    )
    calls = []

    def model(args, *a, **kw):
        source = Path(args[args.index("--video") + 1])
        target = Path(args[args.index("--output") + 1]) / "frames/frames"
        target.mkdir(parents=True)
        for path in source.glob("*.png"):
            frame = cv2.imread(str(path))
            calls.append(frame.shape)
            assert frame.shape[1] < image.shape[1]
            cv2.imwrite(str(target / path.name), frame)

    monkeypatch.setattr(roi_inpaint, "run", model)
    inpaint(tmp_path, files, [[0, 3]], "propainter")
    assert len(calls) == 3
    inpaint(tmp_path, files, [[0, 3]], "propainter")
    assert len(calls) == 3
    mask[70:74, 60:70] = 255
    cv2.imwrite(str(tmp_path / "masks" / files[0].name), mask)
    inpaint(tmp_path, files, [[0, 3]], "propainter")
    assert len(calls) == 6
    for path in (tmp_path / "composite").glob("*.png"):
        assert cv2.imread(str(path)).shape == image.shape


@pytest.mark.parametrize("gpu_state", [False, True])
def test_sam_runner_ignores_experimental_options_and_preserves_v7_calls(
    tmp_path, monkeypatch, gpu_state
):
    import importlib.util
    import sys
    import types

    torch = pytest.importorskip("torch")
    from worker import sam_schedule, workflow

    for folder in ("frames", "masks", "prompts"):
        (tmp_path / folder).mkdir()
    image = np.full((144, 256, 3), 90, np.uint8)
    for i in range(4):
        cv2.imwrite(str(tmp_path / "frames" / f"{i:06d}.png"), image)
    prompt = np.zeros((144, 256), np.uint8)
    prompt[30:50, 40:60] = 255
    cv2.imwrite(str(tmp_path / "prompts/seed.png"), prompt)
    hints = [dict(frame=i, track_id=1, box=[40, 30, 60, 50]) for i in range(4)]
    hints.append(dict(frame=1, track_id=2, mask_prompt="prompts/seed.png"))
    (tmp_path / "detections.json").write_text(
        json.dumps(dict(scenes=[[0, 3], [3, 4]], hints=hints, multitrack=True))
    )
    states, directions, masks = [], [], []

    class Predictor:
        image_size = 1024

        def init_state(self, video_path, **kwargs):
            assert kwargs["offload_state_to_cpu"] is True
            paths = sorted(Path(video_path).glob("*.jpg"))
            assert cv2.imread(str(paths[0])).shape[:2] == (144, 256)
            state = dict(count=len(paths), objects=set())
            states.append(state)
            return state

        def add_new_mask(self, state, frame_idx, obj_id, mask):
            assert mask.shape == (144, 256)
            masks.append(mask)
            state["objects"].add(obj_id)

        def add_new_points_or_box(self, state, frame_idx, obj_id, box):
            assert 0 <= box[0] < box[2] <= 256 and 0 <= box[1] < box[3] <= 144
            state["objects"].add(obj_id)

        def propagate_in_video(self, state, start_frame_idx, reverse):
            directions.append(reverse)
            for i in range(state["count"]):
                ids = sorted(state["objects"])
                yield i, ids, torch.ones((len(ids), 1, 144, 256))

        def reset_state(self, state):
            pass

    stub = types.ModuleType("sam2.build_sam")
    stub.build_sam2_video_predictor = lambda *a, **kw: Predictor()
    monkeypatch.setitem(sys.modules, "sam2.build_sam", stub)
    monkeypatch.setitem(sys.modules, "sam_schedule", sam_schedule)
    monkeypatch.setitem(sys.modules, "workflow", workflow)
    for name in (
        "reset_peak_memory_stats",
        "synchronize",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    for key, value in dict(
        SAM_SPEEDUP="true",
        SAM_STATE_ON_GPU="true" if gpu_state else "false",
        SAM_PREPARE_MAX_EDGE="128",
        SAM_CONFIG="测试配置",
        SAM_CHECKPOINT="测试模型",
    ).items():
        monkeypatch.setenv(key, value)
    writes = []
    original_write = cv2.imwrite

    def write(path, *args, **kwargs):
        if path.endswith(".jpg"):
            writes.append(path)
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(cv2, "imwrite", write)
    monkeypatch.setattr(sys, "argv", ["分割", "--root", str(tmp_path)])
    spec = importlib.util.spec_from_file_location(
        "测试分割", Path(__file__).parents[1] / "sam_stage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    assert len(states) == 3 and len(writes) == 5
    assert len(masks) == 1 and directions == [False, True] * 3
    assert {p.name for p in (tmp_path / "object-masks/2").glob("*.png")} == {
        "000001.png"
    }
    for path in (tmp_path / "masks").glob("*.png"):
        assert cv2.imread(str(path), 0).shape == (144, 256)


@pytest.mark.parametrize("canonical", [False, True])
def test_native_fused_renderer_executes_both_v7_passes_and_retains_sixty_source_frames(
    tmp_path, monkeypatch, canonical
):
    torch = pytest.importorskip("torch")

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    from scripts import hls_native_chunk as native
    from worker.matte import projected_texture_composite
    from worker.tests.test_speedup_bench import make_video

    source = tmp_path / "source.mp4"
    make_video(source)
    coarse = tmp_path / "coarse"
    for folder in ("frames", "composite"):
        (coarse / folder).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    originals = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        originals.append(frame)
    cap.release()
    for i in range(8):
        frame = originals[min(i * 2, len(originals) - 1)]
        cv2.imwrite(str(coarse / "frames" / f"{i:06d}.png"), frame)
        cv2.imwrite(
            str(coarse / "composite" / f"{i:06d}.png"),
            cv2.GaussianBlur(frame, (0, 0), 3),
        )
    mask = np.zeros(originals[0].shape[:2], np.uint8)
    mask[20:40, 30:70] = 255
    glyph = np.zeros(mask.shape, np.float32)
    cv2.putText(glyph, "A", (32, 38), 0, 0.5, 1, 1)
    template = tmp_path / "template.npz"
    alpha = tmp_path / "alpha.npz"
    np.savez(template, template=glyph)
    np.savez(alpha, alpha=glyph)
    detected = tmp_path / "detected.json"
    detected.write_text(json.dumps(dict(scenes=[[0, 8]])))
    coefficients = tmp_path / "coefficients.json"
    coefficients.write_text("{}")
    supports = SimpleNamespace(
        median=mask,
        at=lambda *args: [dict(mask=mask, opaque=False, kind="fixed", id=1)],
    )
    monkeypatch.setattr(native, "Supports", lambda *a: supports)
    captured = []

    class Sink:
        def write(self, value):
            captured.append(
                np.frombuffer(value, np.uint8).reshape(originals[0].shape).copy()
            )

        def close(self):
            pass

    class Encoder:
        stdin = Sink()

        def wait(self):
            return 0

        def poll(self):
            return 0

    original_popen = native.subprocess.Popen
    monkeypatch.setattr(
        native.subprocess,
        "Popen",
        lambda args, **kw: Encoder()
        if "rawvideo" in args
        else original_popen(args, **kw),
    )
    monkeypatch.setattr(native, "mux_core_audio", lambda *a: None)
    monkeypatch.setattr(native, "audio_hash", lambda *a: "测试摘要")
    output = tmp_path / "rendered"
    native.render(
        source,
        coarse,
        detected,
        tmp_path / "prior.npz",
        alpha,
        template,
        None,
        output,
        duration=0.2,
        projection_calibration=coefficients,
        output_fps=60,
        backend="torch-cpu",
        fused_detail=not canonical,
        final_recompose=canonical,
        detail_sigma=3 if canonical else 1.2,
        video_crf=10,
    )
    report = json.loads((output / "native-report.json").read_text())
    assert len(captured) == 12 and report["视频规格"] == [128, 96, 60]
    assert report["原片帧索引"] == list(range(12))
    for n, frame in enumerate(captured):
        base = cv2.imread(str(coarse / "composite" / f"{report['帧映射'][n]:06d}.png"))
        expected = originals[n].copy()
        first = projected_texture_composite(
            originals[n],
            base,
            mask,
            [glyph],
            calibration={},
            detail_sigma=3 if canonical else 1.2,
        )
        expected[mask > 0] = first[mask > 0]
        second = projected_texture_composite(
            originals[n], expected, mask, [glyph], calibration={}, detail_sigma=3
        )
        expected[mask > 0] = second[mask > 0]
        assert np.abs(frame.astype(float) - expected).max() <= 1
        assert np.array_equal(frame[mask == 0], originals[n][mask == 0])


@pytest.mark.parametrize("count", [1, 37])
def test_roi_padding_and_overlap_keep_geometry_and_unmasked_pixels(
    tmp_path, monkeypatch, count
):
    from worker import roi_inpaint

    image = np.full((250, 501, 3), 80, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[-12:, -20:] = 255
    for name in ("frames", "masks"):
        (tmp_path / name).mkdir()
    files = []
    for i in range(count):
        file = tmp_path / "frames" / f"{i:06d}.png"
        cv2.imwrite(str(file), image)
        cv2.imwrite(str(tmp_path / "masks" / file.name), mask)
        files.append(file)
    monkeypatch.setenv("INPAINT_PYTHON", "python")
    monkeypatch.setenv("PROPAINTER_ROOT", str(tmp_path / "model"))
    monkeypatch.setattr(
        roi_inpaint,
        "CleanReferences",
        lambda *a: SimpleNamespace(restore=lambda x, m, y: y),
    )
    calls = []

    def model(args, *a, **kw):
        frames = Path(args[args.index("--video") + 1])
        paths = sorted(frames.glob("*.png"))
        calls.append(len(paths))
        target = Path(args[args.index("--output") + 1]) / "frames/frames"
        target.mkdir(parents=True)
        for path in paths:
            crop = cv2.imread(str(path))
            assert crop.shape[0] % 8 == crop.shape[1] % 8 == 0
            cv2.imwrite(str(target / path.name), crop + 20)

    monkeypatch.setattr(roi_inpaint, "run", model)
    metrics = roi_inpaint.inpaint_roi(tmp_path, files, [[0, count]])
    assert calls == ([2] if count == 1 else [36, 9])
    for file in files:
        out = cv2.imread(str(tmp_path / "composite" / file.name))
        assert out.shape == image.shape
        assert np.array_equal(out[mask == 0], image[mask == 0])
    if count == 37:
        assert metrics["overlap_samples"] == 8 and metrics["overlap_max_mae"] == 0


def test_native_references_cache_both_flow_directions_and_stop_at_shot(tmp_path):
    from worker.texture_refs import NativeReferences

    for folder in ("composite", "masks", "source"):
        (tmp_path / folder).mkdir()
    rng = np.random.default_rng(9)
    image = rng.integers(95, 115, (60, 100, 3), dtype=np.uint8)
    for i in range(11):
        mask = np.zeros((60, 100), np.uint8)
        if i in (5, 10):
            mask[20:40, 30:70] = 255
        for folder in ("source", "composite"):
            cv2.imwrite(str(tmp_path / folder / f"{i:06d}.png"), image)
        cv2.imwrite(str(tmp_path / "masks" / f"{i:06d}.png"), mask)
    refs = NativeReferences(tmp_path, tmp_path / "source", [[0, 11], [11, 22]])
    calls = []

    class Flow:
        def calc(self, a, b, c):
            calls.append(1)
            return np.zeros((*a.shape, 2), np.float32)

    refs.flow = Flow()
    refs.restore(5)
    assert len(calls) == 4
    refs.restore(10)
    assert len(calls) == 6


def test_checkpoint_hash_never_reads_a_key_or_runtime_configuration(tmp_path):
    from worker.roi_inpaint import digest

    for name in ("key.bin", "runtime.env"):
        with pytest.raises(ValueError, match="禁止读取"):
            digest(tmp_path / name)


def test_parallel_prefetch_preserves_order_and_bound():
    from threading import Lock
    from time import sleep

    lock = Lock()
    active = peak = 0

    def prepare(n):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        sleep(0.004 * (4 - n % 4))
        with lock:
            active -= 1
        return n

    assert list(bounded_map(prepare, range(12), depth=4, workers=4)) == list(range(12))
    assert 1 < peak <= 4
    with pytest.raises(ValueError, match="线程数"):
        list(bounded_map(prepare, [], depth=2, workers=3))


def test_indexed_nearest_hint_keeps_unsorted_ties_and_first_duplicate():
    from worker.vision import nearest_hints

    hints = [
        dict(frame=8, confidence=0.8),
        dict(frame=2, confidence=0.2),
        dict(frame=8, confidence=0.9),
    ]
    indexed = nearest_hints(hints, 0, 11)
    for i in range(11):
        assert indexed[i] is min(hints, key=lambda p: abs(p["frame"] - i))
