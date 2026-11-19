"""逐层ROI多进程并行与缓存优化路径的逐位等价性单元测试（合成数据）。"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from worker import roi_inpaint, roi_spatial_layers as rsl
from worker.edge_residue import EdgeSettings, measure
from worker.hls_layers import write_manifest
from worker.process import PipelineError
from worker.restoration import CleanReferences, restore_orthogonal_luma_detail
from worker.spatial_roi import SpatialSettings
from worker.trusted_context import ContextSettings


def _make_textured_frame(idx: int, h: int = 160, w: int = 224) -> np.ndarray:
    rng = np.random.default_rng(101)
    canvas = rng.integers(40, 150, size=(h + 80, w + 120, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0 : h + 80, 0 : w + 120]
    pat = ((np.sin(xx / 6.0) + np.cos(yy / 8.0)) * 24.0).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + pat[..., None], 20, 210).astype(np.uint8)
    # 局部区域加不同色彩纹理以锻炼局部供体匹配
    canvas[45:95, 70:130] = np.clip(canvas[45:95, 70:130].astype(np.int16) + np.array([30, -20, 10]), 0, 255).astype(np.uint8)
    ox = (idx * 3) % 80
    oy = (idx * 2) % 60
    return canvas[oy : oy + h, ox : ox + w].copy()


def _build_synthetic_workspace(root: Path) -> tuple[list[Path], dict]:
    h, w = 160, 224
    n_frames = 14
    for sub in ('frames', 'masks', 'layer-00/masks', 'edge-band', 'scene-protection', 'subtitle-protection'):
        (root / sub).mkdir(parents=True)

    spatial = SpatialSettings(
        dilation_pixels=2,
        merge_gap_pixels=12,
        minimum_width=64,
        minimum_height=64,
        fixed_context_pixels=24,
        maximum_crop_pixels=h * w,
        maximum_pixel_frames=h * w * 20,
    )
    trusted = ContextSettings(
        model_sample_count=2,
        retain_stage_sample_count=1,
        boundary_exclusion_pixels=3,
        dense_reference_frames=20,  # > n_frames，触发 _dense_frame_evidence
        detail_preserve_mode='orthogonal_luma',
    )
    edge_cfg = EdgeSettings(
        shape_radius=1,
        residue_repair_rounds=1,
        residue_repair_dilate_radius=4,
    )
    (root / 'spatial-roi.json').write_text(json.dumps(asdict(spatial)), encoding='utf-8')
    (root / 'trusted-context.json').write_text(json.dumps(asdict(trusted)), encoding='utf-8')
    (root / 'edge-policy.json').write_text(json.dumps(asdict(edge_cfg)), encoding='utf-8')
    (root / 'inpaint-policy.json').write_text(json.dumps(dict(continue_unpassed=True)), encoding='utf-8')

    blue = np.array([225, 65, 25], dtype=np.uint8)
    orange = np.array([25, 130, 230], dtype=np.uint8)
    files: list[Path] = []
    hints = []
    for i in range(n_frames):
        name = f'{i:06d}.png'
        im = _make_textured_frame(i, h, w)
        # 两组彩色水印 + 边缘收边带；第 5 帧移动以便前后帧有局部干净供体
        dx = 0 if i < 6 else 40
        m = np.zeros((h, w), dtype=np.uint8)
        m[36:56, 36 + dx : 64 + dx] = 255
        m[88:106, 128:156] = 255
        band = np.zeros((h, w), dtype=np.uint8)
        band[35:36, 36 + dx : 64 + dx] = 255
        m |= band
        im[35:56, 36 + dx : 64 + dx] = blue
        im[88:106, 128:156] = orange
        fp = root / 'frames' / name
        cv2.imwrite(str(fp), im)
        files.append(fp)
        cv2.imwrite(str(root / 'masks' / name), m)
        cv2.imwrite(str(root / 'layer-00/masks' / name), m)
        cv2.imwrite(str(root / 'edge-band' / name), band)
        sprot = np.zeros((h, w), dtype=np.uint8)
        sprot[120:150, 10:40] = 255
        cv2.imwrite(str(root / 'scene-protection' / name), sprot)
        cv2.imwrite(str(root / 'subtitle-protection' / name), np.zeros((h, w), dtype=np.uint8))
        hints.append({'frame': i, 'box': [36.0 + dx, 35.0, 64.0 + dx, 56.0]})

    det = {'scenes': [[0, n_frames]], 'tracks': [{'id': 1, 'scene': 0, 'hints': hints}]}
    (root / 'detections.json').write_text(json.dumps(det), encoding='utf-8')
    write_manifest(root, root / 'detections.json', [root / 'layer-00/masks'], files)
    return files, det


def _fake_propainter_factory():
    def fake_model(args, **kwargs):
        frames_dir = Path(args[args.index('--video') + 1])
        masks_dir = Path(args[args.index('--mask') + 1])
        out_dir = Path(args[args.index('--output') + 1]) / 'frames/frames'
        out_dir.mkdir(parents=True, exist_ok=True)
        is_repair = 'residue-repair' in str(frames_dir)
        for idx, p in enumerate(sorted(frames_dir.glob('*.png'))):
            im = cv2.imread(str(p))
            m = cv2.imread(str(masks_dir / p.name), 0) > 0
            # 初次补画故意在前几帧保留少量蓝色残余触发有界补修轮；补修轮清除干净
            rep = cv2.GaussianBlur(im, (15, 15), 4.0)
            out = im.copy()
            out[m] = rep[m]
            if not is_repair and idx in (1, 2):
                # 在局部区域留下一块蓝色残留以触发 RESIDUE_FOUND 与补修
                sub_m = np.zeros_like(m)
                sub_m[38:50, 38:56] = True
                hit = m & sub_m
                out[hit] = np.array([225, 65, 25], dtype=np.uint8)
            elif is_repair:
                out[m] = np.array([95, 105, 100], dtype=np.uint8)
            cv2.imwrite(str(out_dir / p.name), out)

    return fake_model


def _strip_timing(obj, root: Path):
    excluded = {'总耗时秒', '耗时秒', '超时秒', '复检耗时秒', '历史补画耗时秒'}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in excluded:
                continue
            if k in {'暂存路径', '基线路径'} and isinstance(v, str):
                out[k] = str(Path(v).relative_to(root))
            else:
                out[k] = _strip_timing(v, root)
        return out
    if isinstance(obj, list):
        return [_strip_timing(v, root) for v in obj]
    return obj


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize("prefetch", ["0", "1"])
def test_parallel_and_serial_execute_are_bitwise_identical(tmp_path: Path, monkeypatch, prefetch):
    """验证并行（多进程+SharedMemory）与串行（ROI_CPU_WORKERS=1）在合成数据上逐帧逐像素一致。"""
    monkeypatch.setenv('INPAINT_PYTHON', 'python3')
    monkeypatch.setenv('PROPAINTER_ROOT', str(tmp_path))
    monkeypatch.setenv('HLS_WINDOW_FRAMES', '8')
    monkeypatch.setenv('HLS_OVERLAP_FRAMES', '3')
    monkeypatch.setattr(rsl, 'run', _fake_propainter_factory())

    cfg = roi_inpaint.ROISettings(sequential_layers=True, fp16=False, seam_retries=1)

    serial_dir = tmp_path / 'serial'
    files_s, det_s = _build_synthetic_workspace(serial_dir)
    monkeypatch.setenv('ROI_CPU_WORKERS', '1')
    rep_s = roi_inpaint.execute(serial_dir, files_s, det_s, cfg=cfg)

    parallel_dir = tmp_path / 'parallel'
    files_p, det_p = _build_synthetic_workspace(parallel_dir)
    monkeypatch.setenv('ROI_CPU_WORKERS', '3')
    monkeypatch.setenv('ROI_MODEL_PREFETCH', prefetch)
    logs = []
    rep_p = roi_inpaint.execute(parallel_dir, files_p, det_p, log=logs.append, cfg=cfg)
    assert any('开始预取首轮模型' in line for line in logs) == (prefetch == '1')

    # 1. roi-report.json 除耗时外完全一致
    assert _strip_timing(rep_s, serial_dir) == _strip_timing(rep_p, parallel_dir)
    assert len(rep_s['任务']) >= 2
    assert any('确认残留补修' in t for task in rep_s['任务'] for t in task['尝试'])

    # 2. 层输入自检.json 完全一致
    assert json.loads((serial_dir / '层输入自检.json').read_text()) == json.loads((parallel_dir / '层输入自检.json').read_text())

    # 3. staged-delta/*.npz 逐位相等
    s_deltas = sorted(serial_dir.glob('roi-*-attempt-*/staged-delta/*.npz'))
    p_deltas = sorted(parallel_dir.glob('roi-*-attempt-*/staged-delta/*.npz'))
    assert [x.relative_to(serial_dir) for x in s_deltas] == [x.relative_to(parallel_dir) for x in p_deltas]
    assert len(s_deltas) > 0
    for sp, pp in zip(s_deltas, p_deltas):
        with np.load(sp, allow_pickle=False) as sa, np.load(pp, allow_pickle=False) as pa:
            assert sa.files == pa.files
            for k in sa.files:
                assert np.array_equal(sa[k], pa[k])

    # 4. roi-layer-00、roi-coverage、seam-difference、seam-shared 逐位相等
    for sub in ('roi-layer-00', 'roi-coverage'):
        s_files = sorted((serial_dir / sub).glob('*.png'))
        p_files = sorted((parallel_dir / sub).glob('*.png'))
        assert [x.name for x in s_files] == [x.name for x in p_files]
        for sf, pf in zip(s_files, p_files):
            assert _sha(sf) == _sha(pf), f'{sub}/{sf.name} 不一致'

    for sub in ('seam-difference', 'seam-shared'):
        s_files = sorted(serial_dir.glob(f'roi-*-attempt-*/{sub}/*.png'))
        p_files = sorted(parallel_dir.glob(f'roi-*-attempt-*/{sub}/*.png'))
        assert [x.relative_to(serial_dir) for x in s_files] == [x.relative_to(parallel_dir) for x in p_files]
        for sf, pf in zip(s_files, p_files):
            assert _sha(sf) == _sha(pf), f'{sf} 不一致'


@pytest.mark.parametrize('prefetch', ['0', '1'])
def test_model_oom_retries_identical_inputs_and_preserves_other_tasks(tmp_path, monkeypatch, prefetch):
    monkeypatch.setenv('INPAINT_PYTHON', 'python3')
    monkeypatch.setenv('PROPAINTER_ROOT', str(tmp_path))
    monkeypatch.setenv('ROI_CPU_WORKERS', '2')
    monkeypatch.setenv('ROI_MODEL_PREFETCH', prefetch)
    monkeypatch.setenv('HLS_WINDOW_FRAMES', '8')
    monkeypatch.setenv('HLS_OVERLAP_FRAMES', '3')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'roundup_power2_divisions:[256:1,>:2],expandable_segments:False')
    monkeypatch.delenv('PYTORCH_ALLOC_CONF', raising=False)
    cfg = roi_inpaint.ROISettings(sequential_layers=True, fp16=False, seam_retries=0)
    fake = _fake_propainter_factory()
    target = 'roi-0002-attempt-00'

    def execute(name, fail_count):
        root = tmp_path / name
        files, det = _build_synthetic_workspace(root)
        calls = {}
        seen = Counter()

        def model(args, **kwargs):
            task = Path(args[args.index('--video') + 1]).parent
            key = str(task.relative_to(root))
            seen[key] += 1
            inputs = [p.read_bytes() for sub in ('frames', 'masks') for p in sorted((task / sub).glob('*.png'))]
            command = [str(arg).replace(str(root), '<工作目录>') for arg in args]
            if seen[key] == 1:
                calls[key] = command, inputs
            else:
                assert key == target and calls[key] == (command, inputs)
                assert kwargs['env']['PYTORCH_CUDA_ALLOC_CONF'] == 'roundup_power2_divisions:[256:1,>:2],expandable_segments:True'
                assert not (task / 'out').exists()
                assert not (task / 'metrics.json').exists()
            if key == target and seen[key] <= fail_count:
                (task / 'out').mkdir()
                (task / 'out' / 'partial').touch()
                (task / 'metrics.json').touch()
                raise PipelineError('GPU_OOM', '合成显存不足')
            return fake(args, **kwargs)

        monkeypatch.setattr(rsl, 'run', model)
        if fail_count == 2:
            with pytest.raises(PipelineError) as error:
                roi_inpaint.execute(root, files, det, cfg=cfg)
            assert error.value.code == 'ROI_UNFILLABLE'
        else:
            roi_inpaint.execute(root, files, det, cfg=cfg)
        report = json.loads((root / 'roi-report.json').read_text())
        return root, calls, seen, report

    base, inputs, counts, _ = execute('正常', 0)
    retried, retry_inputs, retry_counts, recovered = execute('恢复', 1)
    assert retry_inputs == inputs
    assert retry_counts == counts + Counter({target: 1})
    assert recovered['任务'][1]['尝试'][0]['显存降档']
    assert 'GPU_OOM' not in [row.get('错误码') for row in recovered['任务']]
    for folder in ('composite', 'roi-coverage'):
        for p in (base / folder).glob('*.png'):
            assert p.read_bytes() == (retried / folder / p.name).read_bytes()

    exhausted, _, failed_counts, failed = execute('耗尽', 2)
    assert failed_counts[target] == 2
    assert failed_counts['roi-0001-attempt-00'] == 1
    assert failed_counts['roi-0003-attempt-00'] == 1
    assert failed['不可填块'] == [2]
    assert failed['任务'][1]['错误码'] == 'GPU_OOM'
    assert failed['任务'][1]['采用未过结果'] is False
    assert not (exhausted / 'composite').exists()


def test_clean_references_and_ortho_luma_cache_equivalence():
    """验证预计算 trusted/grays、donor evaluator 缓存与正交亮度细节局部化完全逐位等价。"""
    frames = [_make_textured_frame(i, 128, 160) for i in range(6)]
    masks = []
    for i in range(6):
        m = np.zeros((128, 160), dtype=np.uint8)
        m[20:48, 30:70] = 255
        masks.append(m)

    tgt = frames[0].copy()
    tgt[20:48, 30:70] = np.array([220, 60, 30], dtype=np.uint8)
    rep = cv2.GaussianBlur(tgt, (9, 9), 2.5)

    cr_std = CleanReferences(frames[1:], masks[1:])
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    pre_tr = [255 - cv2.dilate(m, ker) for m in masks[1:]]
    pre_gr = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in frames[1:]]
    cr_pre = CleanReferences(frames[1:], [None] * 5, trusted=pre_tr, grays=pre_gr)

    out_std = cr_std.restore(tgt, masks[0], rep, confirmed=masks[0])
    out_pre = cr_pre.restore(tgt, masks[0], rep, confirmed=masks[0])
    assert np.array_equal(out_std, out_pre)
    for k in ('background', 'confidence', 'counts'):
        assert np.array_equal(cr_std.last_evidence[k], cr_pre.last_evidence[k])

    cfg = EdgeSettings()
    wm_hp = np.zeros((128, 160), dtype=np.float32)
    wm_hp[25:40, 35:60] = 40.0
    o1 = restore_orthogonal_luma_detail(tgt, out_std, masks[0], wm_hp, cfg)
    o2 = restore_orthogonal_luma_detail(tgt, out_pre, masks[0], wm_hp, cfg)
    assert np.array_equal(o1, o2)

    m1 = measure(tgt, o1, masks[0], np.zeros_like(masks[0]), cfg, reference_evidence=cr_std.last_evidence, confirmed=masks[0])
    m2 = measure(tgt, o2, masks[0], np.zeros_like(masks[0]), cfg, reference_evidence=cr_pre.last_evidence, confirmed=masks[0])
    assert m1 == m2


def test_cpu_workers_env_control(monkeypatch):
    monkeypatch.setenv('ROI_CPU_WORKERS', '1')
    assert rsl._cpu_workers() == 1
    monkeypatch.setenv('ROI_CPU_WORKERS', '4')
    assert rsl._cpu_workers() == 4
    monkeypatch.delenv('ROI_CPU_WORKERS', raising=False)
    assert 1 <= rsl._cpu_workers() <= 12
