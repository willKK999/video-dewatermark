"""以主分支冻结函数及独立像素输出验证七版判定没有被另写。"""

import ast
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import pytest
from worker import hls_v7, v7_equivalent, native_assets, colour_tracks
from scripts.hls_mask_equivalence import compare

FROZEN = Path(__file__).parent / "fixtures/v7_0e4dc7b"


def function_tree(source, name):
    tree = ast.parse(source)
    return ast.dump(
        next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    )


def test_sam_and_hint_rules_are_identical_to_main_branch_snapshot():
    for filename, functions in [
        ("sam_stage.py", ["main"]),
        (
            "branding.py",
            ["palette_support", "grow_brand_components", "opaque_brand_ink"],
        ),
        ("sam_schedule.py", ["sampled_hints", "groups_for_scene", "load_mask_prompt"]),
    ]:
        reference = (FROZEN / (filename.removesuffix(".py") + ".txt")).read_text()
        actual = (Path(__file__).parents[1] / filename).read_text()
        for name in functions:
            assert function_tree(actual, name) == function_tree(reference, name)


def test_graphics_delegate_to_v7_without_new_detection_or_prompt_rules(
    tmp_path, monkeypatch
):
    (tmp_path / "frames").mkdir()
    calls = []
    assets = {"color_seed_image": tmp_path / "fixed.png"}
    detections = {"scenes": [[0, 2]], "hints": []}

    def graphics(files, supplied, work, check):
        calls.append(("模板", supplied))
        (work / "graphics").mkdir()

    def complete(work, supplied, detected, check):
        calls.append(("原分割", detected))

    monkeypatch.setattr(hls_v7, "graphic_masks", graphics)
    monkeypatch.setattr(colour_tracks, "complete", complete)
    native_assets.prepare_graphics(tmp_path, detections, assets, True)
    assert calls == [
        ("模板", {"color_seed_image": str(assets["color_seed_image"])}),
        ("原分割", detections),
    ]
    assert (tmp_path / "graphic").resolve() == tmp_path / "graphics"


def test_real_graphic_masks_match_frozen_main_function_per_frame(tmp_path):
    from worker import hls_v7

    root = tmp_path / "inputs"
    root.mkdir()
    random = np.random.default_rng(9)
    image = random.integers(20, 90, (96, 128, 3), dtype=np.uint8)
    mask = np.zeros((96, 128), np.uint8)
    cv2.putText(mask, "A7", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    image[mask > 0] = 255
    cv2.imwrite(str(root / "image.png"), image)
    cv2.imwrite(str(root / "mask.png"), mask)
    assets = {
        key: str(root / ("image.png" if key.endswith("image") else "mask.png"))
        for key in (
            "color_seed_image",
            "color_seed_mask",
            "variant_seed_image",
            "variant_seed_mask",
        )
    }
    actual, baseline = tmp_path / "actual", tmp_path / "baseline"
    for work in (actual, baseline):
        (work / "frames").mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(
                str(work / "frames" / f"{i:06d}.png"),
                cv2.flip(image, 1) if i == 1 else image,
            )
    namespace = dict(
        __package__="worker",
        cv2=cv2,
        np=np,
        write_json=hls_v7.write_json,
        shutil=shutil,
    )
    exec(
        compile((FROZEN / "hls_v7.txt").read_text(), "冻结七版图形函数", "exec"),
        namespace,
    )
    namespace["graphic_masks"](
        sorted((baseline / "frames").glob("*.png")), assets, baseline, lambda: None
    )
    hls_v7.graphic_masks(
        sorted((actual / "frames").glob("*.png")), assets, actual, lambda: None
    )
    report = compare(actual, baseline, tmp_path / "equivalence.json")
    assert report["通过"] and report["最低交并比"] == 1
    assert len(report["全部样本"]) == 9


def test_equivalence_reports_every_difference_missing_frame_and_changed_input(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        (root / "frames").mkdir(parents=True)
        (root / "graphics/masks").mkdir(parents=True)
        cv2.imwrite(str(root / "frames/000000.png"), np.zeros((20, 30, 3), np.uint8))
        cv2.imwrite(
            str(root / "graphics/masks/000000.png"), np.zeros((20, 30), np.uint8)
        )
    first = compare(a, b, tmp_path / "same.json", names=("graphics",))
    assert first["通过"] and first["最低交并比"] == 1
    mask = np.zeros((20, 30), np.uint8)
    mask[2:5, 3:8] = 255
    cv2.imwrite(str(a / "graphics/masks/000000.png"), mask)
    cv2.imwrite(str(a / "graphics/masks/000001.png"), mask)
    cv2.imwrite(str(a / "frames/000000.png"), np.ones((20, 30, 3), np.uint8))
    report = compare(a, b, tmp_path / "different.json", names=("graphics",))
    assert not report["通过"] and not report["同一输入"]
    assert len(report["掩膜差异"]) == 2 and report["全部样本"][0]["交并比"] == 0
    with pytest.raises(ValueError, match="自身比较"):
        compare(a, a, tmp_path / "invalid.json")


def test_canonical_chunk_calls_original_prepare_gpu_and_native_stages(
    tmp_path, monkeypatch
):
    assets = {}
    for key in v7_equivalent.REQUIRED_ASSETS:
        path = tmp_path / (key + ".npz")
        path.write_bytes(key.encode())
        assets[key] = path
    work = tmp_path / "work"
    plan = dict(left=2, right=14, start=4, end=12)
    calls = []

    def prepare(source, root, converted, supplied, check, timings=None):
        calls.append(("准备", converted))
        assert supplied["prior"] == str(assets["prior"])
        (root / "graphics").mkdir()
        return {"seconds": 0.2}

    def gpu(root, supplied, info, engine, stage, check, timings=None):
        calls.append(("原图形分割分层补画", engine))
        return {"stages": []}

    def native(root, asset_path, backend="numpy"):
        calls.append(("原生合成", backend))
        (root / "native").mkdir()
        report = dict(原片帧索引=list(range(12)), 帧映射=[0] * 12, 帧数=12)
        (root / "native/native-report.json").write_text(json.dumps(report))

    monkeypatch.setattr(hls_v7, "prepare_block", prepare)
    monkeypatch.setattr(hls_v7, "gpu_block", gpu)
    monkeypatch.setattr(hls_v7, "native_block", native)
    report = v7_equivalent.run_chunk(
        tmp_path / "source.mp4",
        work,
        plan,
        assets,
        dict(mode="speedup-v2", backend="torch-cuda", materialize_frames=False),
        "冻结原片",
    )
    assert calls == [
        ("准备", {k: v / 2 for k, v in plan.items()}),
        ("原图形分割分层补画", "propainter"),
        ("原生合成", "torch-cuda"),
    ]
    assert report["native"]["原片帧索引"] == list(range(2, 14))
    proof = json.loads((work / "decision-provenance.json").read_text())
    assert not proof["局部补画"] and not proof["提示重采样"]


def test_canonical_checkpoint_restores_exact_model_and_native_pixels(tmp_path):
    import subprocess
    from worker.hls_native import cache_chunk, restore_chunk
    from worker.native_checkpoint import BundleStore
    from worker.tests.test_native_delivery import MemoryStore

    a, b = tmp_path / "first", tmp_path / "restored"
    for root in (a, b):
        root.mkdir()
    source = a / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=60",
            "-frames:v",
            "4",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    (a / "frames").mkdir()
    (a / "rendered/frames").mkdir(parents=True)
    (a / "native").mkdir()
    cap = cv2.VideoCapture(str(source))
    original = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        original.append(frame)
    cap.release()
    for i, frame in enumerate(original):
        cv2.imwrite(str(a / "rendered/frames" / f"{i:06d}.png"), frame)
    model = []
    for i in range(2):
        frame = cv2.resize(original[i * 2], (1280, 720))
        model.append(frame)
        cv2.imwrite(str(a / "frames" / f"{i:06d}.png"), frame)
    assets = {"prior": tmp_path / "prior.npz"}
    assets["prior"].write_bytes(b"input")
    (a / "detections.json").write_text(
        json.dumps({"prior": str(assets["prior"]), "hints": []})
    )
    (a / "v7-assets.json").write_text("{}")
    (a / "chunk.json").write_text("{}")
    (a / "performance.json").write_text("{}")
    report = {"帧数": 4}
    for name in ("native/native-report.json", "rendered/native-report.json"):
        (a / name).write_text(json.dumps(report))
    (a / "prepared.json").write_text(
        json.dumps(
            dict(
                detected=str(a / "detections.json"),
                source=str(source),
                coarse=str(a / "coarse"),
                graphics=str(a / "graphics"),
                color=str(a / "color"),
                variant=str(a / "variant"),
            )
        )
    )
    for name in ("color", "variant", "graphics"):
        (a / name / "masks").mkdir(parents=True)
        for i in range(2):
            mask = np.zeros((720, 1280), np.uint8)
            mask[20:25, 30 + i : 40 + i] = 255
            cv2.imwrite(str(a / name / "masks" / f"{i:06d}.png"), mask)
    from worker.hls_layers import write_manifest, VerifiedLayers

    (a / "fine/masks").mkdir(parents=True)
    for i in range(2):
        shutil.copyfile(
            a / "color/masks" / f"{i:06d}.png", a / "fine/masks" / f"{i:06d}.png"
        )
    (a / "coarse").mkdir()
    write_manifest(
        a / "coarse",
        a / "detections.json",
        [a / "fine/masks"],
        sorted((a / "frames").glob("*.png")),
    )
    store = BundleStore(MemoryStore(), "bucket", "isolated/checkpoint")
    cache_chunk(a, store, "chunk", "identity", lambda: None)
    assert restore_chunk(
        b,
        store,
        "chunk",
        "identity",
        source,
        dict(left=0, right=4, start=0, end=4),
        assets,
        lambda: None,
    )
    for i, expected in enumerate(model):
        assert np.array_equal(cv2.imread(str(b / "frames" / f"{i:06d}.png")), expected)
    for i, expected in enumerate(original):
        assert np.array_equal(
            cv2.imread(str(b / "rendered/frames" / f"{i:06d}.png")), expected
        )
    prepared = json.loads((b / "prepared.json").read_text())
    assert Path(prepared["detected"]).parent == b
    assert Path(prepared["source"]) == b / "source-input.mkv"
    assert (b / "graphic").resolve() == b / "graphics"
    proof = VerifiedLayers(b / "coarse/layers.json", b / "detections.json")
    assert proof.validate(0, cv2.imread(str(b / "fine/masks/000000.png"), 0))


def test_fixed_input_copying_bypass_has_been_removed():
    assert not hasattr(v7_equivalent, "fixed_inputs")


def test_independent_reference_verifies_commit_and_every_code_digest(tmp_path):
    import hashlib
    from scripts.hls_v7_mask_reference import verify_reference, REFERENCE_COMMIT

    files = [
        "worker/hls_v7.py",
        "worker/colour_tracks.py",
        "worker/brand_templates.py",
        "worker/sam_stage.py",
        "worker/sam_schedule.py",
        "worker/workflow.py",
    ]
    metadata = dict(commit=REFERENCE_COMMIT, sha256={})
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("# 冻结源码\n")
        metadata["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "reference-code.json").write_text(json.dumps(metadata))
    assert verify_reference(tmp_path)["commit"] == REFERENCE_COMMIT
    (tmp_path / files[0]).write_text("# 已改变\n")
    with pytest.raises(ValueError, match="摘要不一致"):
        verify_reference(tmp_path)


def test_canonical_runner_refuses_old_stage_files_instead_of_claiming_cold_start(
    tmp_path,
):
    from worker.process import PipelineError

    work = tmp_path / "old"
    work.mkdir()
    (work / "prepared.json").write_text("{}")
    assets = {name: tmp_path / name for name in v7_equivalent.REQUIRED_ASSETS}
    with pytest.raises(PipelineError, match="全新空块目录"):
        v7_equivalent.run_chunk(
            tmp_path / "source.mp4",
            work,
            {},
            assets,
            dict(mode="speedup-v2", backend="torch-cuda"),
            "source",
        )


def test_model_launcher_keeps_existing_packages_first_and_disables_all_bytecode_writes(
    tmp_path, monkeypatch
):
    import sys, os
    from scripts import hls_speedup_launch as launch

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["启动器"])
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    before = list(sys.path)
    calls = []
    monkeypatch.setattr(
        launch.runpy,
        "run_module",
        lambda name, run_name: calls.append((name, run_name, list(sys.argv))),
    )
    launch.main(["--extra-python-path", str(tmp_path), "--run", "--output", "独立输出"])
    assert sys.path[:-1] == before and sys.path[-1] == str(tmp_path)
    assert sys.dont_write_bytecode and os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert calls == [
        (
            "scripts.hls_speedup_bench",
            "__main__",
            ["scripts.hls_speedup_bench", "--run", "--output", "独立输出"],
        )
    ]
