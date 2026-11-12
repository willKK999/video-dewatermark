"""半透明图形的平坦内部不能漏出，低频修复不得抹掉已恢复的高频。"""

import cv2, numpy as np
from worker.low_frequency import (
    band,
    fixed_scores,
    complete_play_graphics,
    repair_low_band,
)
from worker.multitrack import highpass


def test_low_band_detects_smooth_fill_that_highpass_amplitude_misses():
    shape = (160, 260)
    logo = np.zeros(shape, np.float32)
    cv2.circle(logo, (55, 80), 37, 12, -1)
    cv2.fillPoly(logo, [np.array([[46, 61], [46, 99], [77, 80]], np.int32)], 0)
    source = logo + 50
    result = 50 + cv2.GaussianBlur(logo, (0, 0), 12) * 0.8
    t = band(source)
    u = band(result)
    regions = [dict(id=1, box=[10, 30, 105, 130])]
    records = fixed_scores(t, u, regions)
    assert records[0]["residual"]
    from worker.multitrack import component_scores

    old = component_scores(
        highpass(source)[30:130, 10:105],
        highpass(result)[30:130, 10:105],
        np.ones((100, 95), np.uint8),
    )
    assert max(abs(p["relative_amplitude"]) for p in old) < 0.15
    assert not fixed_scores(t, np.zeros_like(u), regions)[0]["residual"]


def test_complete_round_graphic_keeps_separate_text_sparse():
    image = np.full((150, 420), 128, np.uint8)
    cv2.circle(image, (65, 75), 40, 151, -1)
    cv2.fillPoly(image, [np.array([[53, 55], [53, 95], [88, 75]], np.int32)], 128)
    cv2.putText(image, "WM ABC", (125, 88), 0, 1.3, 151, 2)
    hp = highpass(image)
    signal = (np.abs(hp) > 0.5).astype(np.uint8) * 255
    region = dict(box=[0, 0, 420, 150], mask=np.full(image.shape, 255, np.uint8))
    support = complete_play_graphics(hp, signal, region)
    assert support[75, 65] == 255
    assert not support[:, 125:].any()


def test_low_band_repair_preserves_every_pixel_outside_mask_and_fine_texture():
    rng = np.random.default_rng(4)
    clean = rng.integers(60, 150, (160, 200, 3), np.uint8)
    mask = np.zeros(clean.shape[:2], np.uint8)
    cv2.circle(mask, (85, 80), 35, 255, -1)
    glow = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 5) * 18
    old = np.clip(clean.astype(float) + glow[:, :, None], 0, 255).astype(np.uint8)
    # 基底差别只有待去除的平坦叠加，原生纹理必须被保留。
    old_base = np.full_like(clean, 100)
    old_base = np.clip(old_base.astype(float) + glow[:, :, None], 0, 255).astype(
        np.uint8
    )
    new_base = np.full_like(clean, 100)
    result = repair_low_band(old, old_base, new_base, mask)
    assert np.array_equal(result[mask == 0], old[mask == 0])
    core = cv2.erode(mask, np.ones((21, 21), np.uint8)) > 0
    assert np.abs(result.astype(float) - clean)[core].mean() < 1
    assert np.abs(highpass(result) - highpass(clean))[core].mean() < 0.5


def test_rgb_lossless_intermediate_keeps_all_existing_pixels(tmp_path):
    import subprocess

    rng = np.random.default_rng(19)
    frames = rng.integers(0, 256, (4, 64, 96, 3), np.uint8)
    path = tmp_path / "lossless.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "96x64",
            "-r",
            "60",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
        timeout=20,
    )
    cap = cv2.VideoCapture(str(path))
    assert cap.get(5) == 60
    for expected in frames:
        ok, actual = cap.read()
        assert ok
        assert np.array_equal(actual, expected)
    assert not cap.read()[0]
    cap.release()


def test_text_without_circular_icon_has_reliable_residual_support():
    from worker.hls_graphic_repair import residual_support

    image = np.full((100, 240), 128, np.uint8)
    cv2.putText(image, "Sample", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, 220, 2)
    feature = image.astype(np.float32) - 128
    signal = (feature > 0).astype(np.uint8) * 255
    region = dict(id=1, box=[10, 20, 210, 80], mask=signal.copy())
    support = residual_support(feature, signal, region)
    assert np.all(support[signal > 0] > 0)
    assert not support[90, 230] and np.count_nonzero(support) < support.size * 0.2


def test_low_band_retry_uses_current_baseline_without_reapplying_old_delta():
    mask = np.full((100, 100), 255, np.uint8)
    source = np.full((100, 100, 3), 120, np.uint8)
    source[30:70, 30:70] = 160
    clean = np.full_like(source, 120)
    once = repair_low_band(source, source, clean, mask)
    # 已清理的当前输入和模型输出一致时，不可再次扣除最初源片的水印差分。
    again = repair_low_band(once, clean, clean, mask)
    assert np.array_equal(again, once)
    doubled = repair_low_band(once, source, clean, mask)
    assert (
        np.abs(doubled.astype(float) - clean).mean()
        > np.abs(again.astype(float) - clean).mean()
    )


def test_low_verification_samples_only_requested_core(tmp_path, monkeypatch):
    from worker import hls_graphic_repair as repair
    from worker.tests.test_speedup_bench import make_video
    from worker.process import PipelineError
    import pytest

    video = tmp_path / "source.mp4"
    make_video(video, seconds=1)
    calls = []
    monkeypatch.setattr(repair, "fixed_scores", lambda t, c, r: calls.append(c) or [])
    result = repair.verify_low(
        video, np.zeros((96, 128)), [], start_frame=12, frame_count=24
    )
    assert result["样本数"] == 1 and len(calls) == 1
    with pytest.raises(PipelineError, match="范围"):
        repair.verify_low(video, np.zeros((96, 128)), [], start_frame=59, frame_count=2)
