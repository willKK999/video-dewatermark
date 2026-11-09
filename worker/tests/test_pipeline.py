import json
import time
import cv2
import numpy as np
import pytest
from worker.media import plan, validate_output
from worker.pipeline import execute
from worker.process import PipelineError, LeaseLost, run
from worker.vision import windows, detect


from scripts.fixtures import generate


@pytest.mark.parametrize("portrait,audio", [(False, True), (True, False)])
def test_real_video_pipeline(tmp_path, portrait, audio):
    source = generate(tmp_path, portrait, audio)
    root = tmp_path / "work"
    root.mkdir()
    stages = []
    dest, info, timings = execute(
        source,
        root,
        {
            "max_short_edge": 720,
            "prompts": [
                {"frame": 0, "box": [20, 30, 62, 52]},
                {"frame": 11, "box": [53, 30, 95, 52]},
            ],
        },
        "telea-dev",
        stage=stages.append,
    )
    assert stages == ["probe", "detect", "mask", "inpaint", "encode"]
    assert dest.stat().st_size > 1000 and info["frame_count"] == 12
    assert (info["height"] > info["width"]) == portrait
    for original in (root / "frames").glob("*.png"):
        raw = cv2.imread(str(original))
        result = cv2.imread(str(root / "composite" / original.name))
        mask = cv2.imread(str(root / "masks" / original.name), 0)
        assert np.array_equal(raw[mask == 0], result[mask == 0])
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["frames"]) == 12 and manifest["frames"][0]["frame_index"] == 0
    assert manifest["media"]["timeline"]["normalized_start_ms"] == 0
    validate_output(dest, dict(info, has_audio=audio))


def test_media_limits():
    data = {
        "format": {"format_name": "mov,mp4", "duration": "30"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 3840,
                "height": 2160,
                "avg_frame_rate": "60/1",
            }
        ],
    }
    assert (
        plan(data)["width"] == 1280
        and plan(data)["height"] == 720
        and plan(data)["fps"] == 30
    )
    data["streams"][0]["width"] = 1080
    data["streams"][0]["height"] = 1920
    assert plan(data)["width"] == 720 and plan(data)["height"] == 1280
    data["streams"][0]["color_transfer"] = "smpte2084"
    with pytest.raises(PipelineError, match="SDR"):
        plan(data)
    del data["streams"][0]["color_transfer"]
    data["format"]["duration"] = "31"
    with pytest.raises(PipelineError):
        plan(data)


def test_windows_complete_and_bounded():
    result = list(windows(0, 900))
    assert max(e - s for s, e in result) <= 60
    assert set(i for s, e in result for i in range(s, e)) == set(range(900))
    assert all(result[i][1] - result[i + 1][0] == 12 for i in range(len(result) - 1))


def test_missing_detection_review(tmp_path, monkeypatch):
    monkeypatch.delenv("TEMPLATE_PATH", raising=False)
    monkeypatch.delenv("DETECTOR", raising=False)
    image = tmp_path / "000000.png"
    cv2.imwrite(str(image), np.zeros((100, 100, 3), np.uint8))
    with pytest.raises(PipelineError) as exc:
        detect([image], [(0, 1)], {}, tmp_path)
    assert exc.value.review


def test_cancel_interrupts_process_group(tmp_path):
    marker = tmp_path / "late"
    start = time.monotonic()

    def check():
        if time.monotonic() - start > 0.4:
            raise LeaseLost()

    with pytest.raises(LeaseLost):
        run(["sh", "-c", 'sleep 2; touch "$1"', "sh", marker], check)
    time.sleep(2)
    assert not marker.exists()


def test_template_dense_retry_finds_between_keyframes(tmp_path, monkeypatch):
    template = np.random.default_rng(7).integers(0, 255, (12, 20), dtype=np.uint8)
    path = tmp_path / "template.png"
    cv2.imwrite(str(path), template)
    monkeypatch.setenv("TEMPLATE_PATH", str(path))
    monkeypatch.delenv("DETECTOR", raising=False)
    files = []
    for i in range(6):
        image = np.zeros((100, 100), np.uint8)
        if i == 3:
            image[20:32, 30:50] = template
        f = tmp_path / f"{i:06d}.png"
        cv2.imwrite(str(f), image)
        files.append(f)
    result = detect(files, [(0, 6)], {}, tmp_path)
    assert result["hints"][0]["frame"] == 3 and result["dense"]


def test_output_corruption_classified(tmp_path):
    dest = tmp_path / "broken.mp4"
    dest.write_bytes(b"not a movie")
    with pytest.raises(PipelineError) as exc:
        validate_output(dest, {})
    assert exc.value.code == "OUTPUT_INVALID"


def test_mask_quality_detects_degradation():
    from scripts.quality import score

    clean = np.full((30, 30, 3), 120, np.uint8)
    mask = np.full((30, 30), 255, np.uint8)
    same = score(clean, clean, mask)
    assert same["identical"] and same["ssim"] == pytest.approx(1)
    altered = clean.copy()
    altered[10:20, 10:20] = 0
    changed = score(clean, altered, mask)
    assert (
        not changed["identical"] and changed["ssim"] < 0.9 and changed["psnr_db"] < 30
    )


def test_yolo_ignores_template(tmp_path, monkeypatch):
    import worker.vision as vision

    monkeypatch.setenv("DETECTOR", "yolo")
    monkeypatch.setenv("TEMPLATE_PATH", str(tmp_path / "missing-template.png"))
    monkeypatch.setenv("DETECT_PYTHON", "python")
    monkeypatch.setenv("YOLO_WEIGHTS", "dedicated.pt")
    image = tmp_path / "000000.png"
    cv2.imwrite(str(image), np.zeros((100, 100, 3), dtype=np.uint8))

    def detector_run(command, check):
        (tmp_path / "detector-output.json").write_text(json.dumps([
            dict(frame=0, box=[10, 10, 30, 30], confidence=.6)
        ]))

    monkeypatch.setattr(vision, "run", detector_run)
    assert detect([image], [(0, 1)], {}, tmp_path)["hints"][0]["confidence"] == .6
