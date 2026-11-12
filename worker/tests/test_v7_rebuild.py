"""原输入重建只连接原方法；冻结对照不得回流到处理路径。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from worker import v7_rebuild as rebuild
from worker.process import PipelineError
from worker.roi_inpaint import digest
from scripts.hls_v7_inputs_compare import compare, rebuilt_paths


def setup_rebuild(tmp_path, monkeypatch, seed_exists=True):
    from scripts import hls_template, hls_align_reference, hls_fit_projection

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    calls = []
    image = np.full((30, 50, 3), 100, np.uint8)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[3:7, 5:15] = 255
    detected = dict(scenes=[[0, 5]], tracks=[], hints=[])

    def prior(source, root, check):
        calls.append("prior")
        p = root / "multitrack-prior.npz"
        p.write_bytes(b"prior")
        return p

    def normalize(source, root, *args):
        (root / "frames").mkdir()
        for i in range(5):
            cv2.imwrite(str(root / "frames" / f"{i:06d}.png"), image)

    def tracks(files, scenes, prior, root, check):
        (root / "detections.json").write_text(json.dumps(detected))
        return detected

    def calibrate(samples, prior, root):
        calls.append("calibration")
        np.savez(root / "alpha.npz", alpha=mask.astype(float))
        np.save(root / "palette.npy", np.zeros((1, 3), np.float32))

    def seed(files, detected, prior, rows, palette, root):
        if not seed_exists:
            return None
        (root / "prompts").mkdir()
        cv2.imwrite(str(root / "prompts/seed.png"), mask)
        result = dict(hints=[dict(frame=2)], scenes=[[0, 5]])
        (root / "detections.json").write_text(json.dumps(result))
        return result

    def run(args, *a, **kw):
        args = list(map(str, args))
        calls.append(args)
        if args[-1].endswith(".jpg"):
            for i in range(20):
                cv2.imwrite(str(Path(args[-1]).parent / f"{i:06d}.jpg"), image)
        elif "scripts.hls_candidates" in args:
            Path(args[-1]).write_text("[]")
        elif any(arg.endswith("sam_stage.py") for arg in args):
            root = Path(args[-1])
            for i in range(5):
                cv2.imwrite(str(root / "masks" / f"{i:06d}.png"), mask)
        elif "scripts.hls_brand_variants" in args:
            root = Path(args[-1])
            (root / "masks").mkdir(parents=True)
            cv2.imwrite(str(root / "seed-image.png"), image)
            cv2.imwrite(str(root / "seed-mask.png"), mask)
            for i in range(5):
                cv2.imwrite(str(root / "masks" / f"{i:06d}.png"), mask)

    def inpaint(root, files, scenes, engine, low_memory, check, **kwargs):
        assert kwargs == dict(roi=False)
        calls.append("inpaint")
        (root / "composite").mkdir()
        for file in files:
            cv2.imwrite(str(root / "composite" / file.name), image)

    class Supports:
        def __init__(self, *args):
            pass

        def at(self, index, image):
            return [dict(mask=mask)]

    monkeypatch.setattr(rebuild.multitrack, "prepare", prior)
    monkeypatch.setattr(rebuild.multitrack, "make_tracks", tracks)
    monkeypatch.setattr(
        rebuild.media,
        "probe",
        lambda *a: dict(streams=[dict(codec_type="video", avg_frame_rate="60/1")]),
    )
    monkeypatch.setattr(rebuild.media, "normalize", normalize)
    monkeypatch.setattr(rebuild.vision, "shots", lambda *a: [[0, 5]])
    monkeypatch.setattr(rebuild.vision, "inpaint", inpaint)
    monkeypatch.setattr(rebuild, "Supports", Supports)
    monkeypatch.setattr(rebuild, "calibrate", calibrate)
    monkeypatch.setattr(
        rebuild,
        "low_band_template",
        lambda samples, prior, root, check: np.savez(
            root / "low-band.npz", template=mask
        ),
    )
    monkeypatch.setattr(rebuild, "discover", seed)
    monkeypatch.setattr(rebuild, "palette_support", lambda *a: np.zeros_like(mask))
    monkeypatch.setattr(rebuild, "run", run)
    monkeypatch.setattr(
        hls_template,
        "build",
        lambda files, output, sigma: np.savez(
            output, template=mask, detail_sigma=sigma
        ),
    )
    monkeypatch.setattr(
        hls_align_reference, "align", lambda *a, **kw: calls.append("reference")
    )
    monkeypatch.setattr(
        hls_fit_projection,
        "fit",
        lambda *a: (calls.append("projection"), a[-2].write_text("{}")),
    )
    for name in ("DETECT_PYTHON", "TRACK_PYTHON", "INPAINT_PYTHON"):
        monkeypatch.setenv(name, "模型解释器")
    return source, calls


def test_all_nine_inputs_are_built_in_order_without_frozen_material(
    tmp_path, monkeypatch
):
    source, calls = setup_rebuild(tmp_path, monkeypatch)
    root = tmp_path / "new"
    assets = rebuild.rebuild(source, root, digest(source))
    assert set(assets) == set(rebuild.REQUIRED_RUN_ASSETS)
    assert all(path.is_file() and path.is_relative_to(root) for path in assets.values())
    assert (
        calls.index("calibration")
        < calls.index("inpaint")
        < calls.index("reference")
        < calls.index("projection")
    )
    extraction = next(
        c for c in calls if isinstance(c, list) and c[-1].endswith(".jpg")
    )
    assert extraction[extraction.index("-vf") + 1] == "select=not(mod(n\\,60))"
    assert json.loads((root / "rebuilt-assets.json").read_text())["本次从原片重建"]
    times = json.loads((root / "stages/timings.json").read_text())
    assert all(not v["reused"] and v["completed"] for v in times.values())
    with pytest.raises(PipelineError, match="全新空目录"):
        rebuild.rebuild(source, root, digest(source))


def test_missing_original_seed_stops_and_records_partial_time(tmp_path, monkeypatch):
    source, calls = setup_rebuild(tmp_path, monkeypatch, seed_exists=False)
    root = tmp_path / "new"
    with pytest.raises(PipelineError, match="禁止改用其他提示"):
        rebuild.rebuild(source, root, digest(source))
    times = json.loads((root / "stages/timings.json").read_text())
    assert not times["color_seed"]["completed"]
    assert "projection" not in calls and "inpaint" not in calls


def test_rebuilt_fine_masks_keep_original_area_protection(tmp_path, monkeypatch):
    image = np.full((20, 30, 3), 100, np.uint8)
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), image)
    alpha = tmp_path / "alpha.npz"
    np.savez(alpha, alpha=np.zeros((20, 30)))

    class Supports:
        def __init__(self, *a):
            pass

        def at(self, *a):
            return [dict(mask=np.full((20, 30), 255, np.uint8))]

    monkeypatch.setattr(rebuild, "Supports", Supports)
    with pytest.raises(PipelineError, match="20%"):
        rebuild.fine_masks(tmp_path / "fine", [p], {}, p, alpha, p, lambda: None)


def test_input_comparison_keeps_missing_items_and_never_replaces_results(tmp_path):
    root = tmp_path / "rebuilt"
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    data = dict(method="v7", source_sha256="原片", sha256={})
    current = rebuilt_paths(root)
    for key in rebuild.REQUIRED_ASSETS:
        reference = frozen / current[key].name
        if reference.suffix == ".npz":
            np.savez(reference, values=np.array([1, 2], np.float32))
        elif reference.suffix == ".npy":
            np.save(reference, np.ones((2, 3), np.float32))
        elif reference.suffix == ".png":
            cv2.imwrite(str(reference), np.zeros((10, 20), np.uint8))
        else:
            reference.write_text('{"coefficient": [1, 2]}')
        data[key] = str(reference)
        data["sha256"][key] = digest(reference)
    manifest = frozen / "assets.json"
    manifest.write_text(json.dumps(data))
    current["alpha"].parent.mkdir(parents=True)
    np.savez(current["alpha"], values=np.array([2, 4], np.float32))
    before = {p: digest(p) for p in frozen.iterdir()}
    rebuilt_before = digest(current["alpha"])
    result = compare(root, manifest, "原片")
    assert not result["完整重建"] and not result["冻结结果参与处理"]
    assert len(result["项目"]) == 9
    alpha = next(r for r in result["项目"] if r["项目"] == "alpha")
    assert alpha["数组差异"]["values"]["最大绝对误差"] == 2
    assert all(
        r["状态"] == "本次尚未重建" for r in result["项目"] if r["项目"] != "alpha"
    )
    assert digest(current["alpha"]) == rebuilt_before
    assert {p: digest(p) for p in frozen.iterdir()} == before
    with pytest.raises(ValueError, match="原片摘要"):
        compare(root, manifest, "另一个原片")


def test_calibration_preserves_native_pixels_visibility_and_palette_sampling(
    tmp_path, monkeypatch
):
    samples = tmp_path / "samples"
    samples.mkdir()
    native = np.full((40, 80, 3), 120, np.uint8)
    for i in range(20):
        cv2.imwrite(str(samples / f"{i:06d}.jpg"), native)
    median = np.ones((20, 40), np.float32) * 3
    signal = np.zeros((20, 40), np.uint8)
    signal[3:7, 5:15] = 255
    region = dict(id=1, box=[5, 3, 15, 7], confidence=0.9, mask=signal)
    monkeypatch.setattr(
        rebuild.multitrack, "load_prior", lambda _: (median, signal, [region], [], {})
    )
    monkeypatch.setattr(rebuild.multitrack, "highpass", lambda image: median.copy())
    captured = {}

    def overlay(images, mask):
        images = list(images)
        captured["alpha"] = (len(images), images[0].shape, mask.shape)
        return np.zeros(mask.shape, np.float32), np.zeros((*mask.shape, 3), np.float32)

    def palette(images, mask):
        captured["palette"] = (len(images), images[0].shape, mask.copy())
        return np.ones((8, 3), np.float32)

    monkeypatch.setattr(rebuild, "fit_overlay", overlay)
    monkeypatch.setattr(rebuild, "learn_palette", palette)
    rebuild.calibrate(samples, tmp_path / "prior.npz", tmp_path)
    assert captured["alpha"] == (20, (40, 80, 3), (40, 80))
    assert captured["palette"][:2] == (4, (4, 10, 3))
    assert (captured["palette"][2] == 255).all()
    report = json.loads((tmp_path / "calibration-samples.json").read_text())
    assert report["颜色样本"] == [0, 5, 10, 15]
    assert len(report["透明度样本"]) == 20


@pytest.mark.parametrize("difference, expected", [(17, True), (18, False)])
def test_original_remainder_threshold_and_mask_union(
    tmp_path, monkeypatch, difference, expected
):
    files = []
    coarse, graphic = tmp_path / "coarse", tmp_path / "graphic"
    for directory in (coarse / "composite", coarse / "masks", graphic / "masks"):
        directory.mkdir(parents=True)
    image = np.full((20, 30, 3), 100, np.uint8)
    zero = np.zeros((20, 30), np.uint8)
    sam = zero.copy()
    sam[3:6, 5:8] = 255
    ink = zero.copy()
    ink[5, 8] = 255
    for i in range(3):
        p = tmp_path / f"{i:06d}.png"
        cv2.imwrite(str(p), image)
        files.append(p)
        cv2.imwrite(str(coarse / "composite" / p.name), image + difference)
        cv2.imwrite(str(coarse / "masks" / p.name), zero)
        cv2.imwrite(str(graphic / "masks" / p.name), sam if i == 1 else zero)
    (graphic / "detections.json").write_text(json.dumps(dict(scenes=[[1, 2]])))
    monkeypatch.setattr(rebuild, "palette_support", lambda *a: ink.copy())
    calls = []

    def inpaint(root, frames, scenes, engine, low_memory, check, **kwargs):
        assert low_memory and kwargs == dict(roi=False)
        calls.append(scenes)
        (root / "composite").mkdir()
        for p in frames:
            cv2.imwrite(str(root / "composite" / p.name), image)

    monkeypatch.setattr(rebuild.vision, "inpaint", inpaint)
    rebuild.remainder(
        coarse, graphic, np.zeros((1, 3)), tmp_path / "remainder", files, lambda: None
    )
    assert bool(calls) == expected
    assert bool(cv2.imread(str(coarse / "masks/000001.png"), 0).any()) == expected
    if expected:
        assert calls == [[(0, 3)]]
        assert cv2.imread(str(graphic / "masks/000001.png"), 0)[5, 8] == 255


def test_original_low_band_uses_five_group_medians_at_prior_resolution(
    tmp_path, monkeypatch
):
    from worker import low_frequency

    samples = tmp_path / "samples"
    samples.mkdir()
    values = [0, 0, 255, 255] + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3
    for i in range(len(values)):
        (samples / f"{i:06d}.jpg").touch()
    monkeypatch.setattr(
        rebuild.multitrack,
        "load_prior",
        lambda _: (np.zeros((4, 8)), None, None, None, None),
    )
    monkeypatch.setattr(
        rebuild.cv2,
        "imread",
        lambda name: np.full((8, 16, 3), values[int(Path(name).stem)], np.uint8),
    )
    monkeypatch.setattr(
        low_frequency, "band", lambda image: image[:, :, 0].astype(np.float32)
    )
    rebuild.low_band_template(samples, tmp_path / "prior", tmp_path)
    template = np.load(tmp_path / "low-band.npz")["template"]
    assert template.shape == (4, 8) and (template == 3).all()
    assert float(np.median(values)) == 2.5


def test_comparison_includes_original_additional_low_band_without_importing_it(
    tmp_path,
):
    root = tmp_path / "rebuilt"
    reference = tmp_path / "low-band.npz"
    np.savez(reference, template=np.ones((4, 8), np.float32))
    manifest = tmp_path / "frozen.json"
    manifest.write_text(
        json.dumps(
            dict(
                method="v7",
                source_sha256="原片",
                low_band_template=str(reference),
                low_band_sha256=digest(reference),
            )
        )
    )
    report = compare(root, manifest, "原片")
    row = next(r for r in report["项目"] if r["项目"] == "low_band_template")
    assert row["状态"] == "本次尚未重建" and report["项目数"] == 10
    assert not report["完整重建"] and not root.exists()
