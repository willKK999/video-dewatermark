"""v17b 离线复验脚本的合成回归：验证增量重建、多进程复检、旧口径对照且不写任何图片。"""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from scripts.v17b_offline_recheck import PANEL_LABEL_HEIGHT, build_parser, execute, parse_tasks
from worker.hls_settings import HLSSettings
from worker.roi_inpaint import ROISettings
from worker.roi_spatial_layers import seam_variants
from worker.spatial_roi import SpatialSettings, plan


def _hsv_bgr(h, s, v, shape=(96, 120)):
    hsv = np.zeros((shape[0], shape[1], 3), np.uint8)
    hsv[..., 0] = h
    hsv[..., 1] = s
    hsv[..., 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_parse_tasks_handles_ranges_lists_and_rejects_invalid():
    assert parse_tasks('1-8') == [1, 2, 3, 4, 5, 6, 7, 8]
    assert parse_tasks('1,3-5,3') == [1, 3, 4, 5]
    assert parse_tasks('all', available=[4, 2, 1]) == [1, 2, 4]
    for bad in ['', '0', '5-2', 'abc', '1-']:
        with pytest.raises(ValueError):
            parse_tasks(bad)


def test_offline_recheck_synthetic_run_with_multiprocessing(tmp_path, capsys):
    run = tmp_path / 'run'
    for sub in ['frames', 'masks', 'edge-band', 'scene-protection', 'roi-0001-baseline', 'roi-0002-baseline']:
        (run / sub).mkdir(parents=True)
    h, w = 96, 120
    core = np.zeros((h, w), np.uint8)
    core[30:50, 40:60] = 255  # 400 px
    band = np.zeros((h, w), np.uint8)
    band[29:30, 40:60] = 255  # 20 px
    mask = core | band  # 420 px
    indices = np.flatnonzero(mask.ravel()).astype(np.int32)
    blue = _hsv_bgr(100, 200, 200, (h, w))
    gray = np.full((h, w, 3), 40, np.uint8)

    for i in range(6):
        name = f'{i:06d}.png'
        cv2.imwrite(str(run / 'masks' / name), mask)
        cv2.imwrite(str(run / 'edge-band' / name), band)
        cv2.imwrite(str(run / 'scene-protection' / name), np.zeros((h, w), np.uint8))
        if i < 3:
            src = gray.copy()
            src[mask > 0] = blue[mask > 0]
            cv2.imwrite(str(run / 'frames' / name), src)
            cv2.imwrite(str(run / 'roi-0001-baseline' / name), src)
        else:
            cv2.imwrite(str(run / 'frames' / name), blue)
            cv2.imwrite(str(run / 'roi-0002-baseline' / name), blue)

    for k, painted_img in [(0, blue), (1, gray)]:
        delta = run / f'roi-0001-attempt-{k:02d}' / 'staged-delta'
        delta.mkdir(parents=True)
        pixels = painted_img.reshape(-1, 3)[indices]
        for i in range(3):
            np.savez_compressed(delta / f'{i:06d}.npz', indices=indices, pixels=pixels)

    delta2 = run / 'roi-0002-attempt-00' / 'staged-delta'
    delta2.mkdir(parents=True)
    pixels2 = blue.reshape(-1, 3)[indices]
    for i in range(3, 6):
        np.savez_compressed(delta2 / f'{i:06d}.npz', indices=indices, pixels=pixels2)

    (run / 'detections.json').write_text(json.dumps(dict(scenes=[[0, 6]])))
    report = dict(
        任务=[
            dict(任务=1, 图层=0, 开始帧=0, 结束帧=3, ROI=[0, 0, w, h], 最佳尝试=1, 尝试=[dict(尝试=0, 重叠最大差异=0.0, 错误码='RESIDUE_FOUND'), dict(尝试=1, 重叠最大差异=0.0)]),
            dict(任务=2, 图层=0, 开始帧=3, 结束帧=6, ROI=[0, 0, w, h], 最佳尝试=0, 尝试=[dict(尝试=0, 重叠最大差异=0.0, 错误码='RESIDUE_UNCERTAIN')]),
        ]
    )
    (run / 'roi-report.json').write_text(json.dumps(report, ensure_ascii=False))

    out_dir = tmp_path / 'out'
    out_json = out_dir / 'recheck.json'
    result = execute(SimpleNamespace(
        run=run,
        tasks='1-2',
        config=Path('config/v17-edge-residue.json'),
        output=out_json,
        workers=2,
    ))

    assert out_json.is_file()
    # 不得写出任何图片文件
    assert not list(out_dir.rglob('*.png')) and not list(out_dir.rglob('*.jpg')) and not list(out_dir.rglob('*.jpeg'))

    loaded = json.loads(out_json.read_text())
    assert loaded['tasks'] == [1, 2]
    assert loaded['判定计数'] == {'通过': 2, 'RESIDUE_FOUND': 1, 'RESIDUE_UNCERTAIN': 0, '跳过': 0}
    assert loaded['帧判定计数'] == {'通过': 6, 'RESIDUE_FOUND': 3, 'RESIDUE_UNCERTAIN': 0}

    t1_a0, t1_a1 = loaded['任务'][0]['尝试']
    assert t1_a0['有效施工帧数'] == 3 and t1_a0['判定'] == 'RESIDUE_FOUND'
    assert t1_a0['颜色候选'] == 1260 and t1_a0['区别色候选数'] == 1260
    assert t1_a0['遮罩内确认残留'] == 1200 and t1_a0['收边带确认残留'] == 60
    assert t1_a0['确认残留'] == 1260 and t1_a0['待目检'] == 0 and t1_a0['单帧最大未解释'] == 420
    assert t1_a0['同色不适用组数'] == 0 and t1_a0['同色不适用施工像素'] == 0
    assert t1_a0['旧口径蓝色候选'] == 1260

    assert t1_a1['有效施工帧数'] == 3 and t1_a1['判定'] == '通过'
    assert t1_a1['颜色候选'] == 0 and t1_a1['区别色候选数'] == 0 and t1_a1['确认残留'] == 0 and t1_a1['旧口径蓝色候选'] == 0

    t2_a0 = loaded['任务'][1]['尝试'][0]
    assert t2_a0['有效施工帧数'] == 3 and t2_a0['判定'] == '通过'
    assert t2_a0['颜色候选'] == 0 and t2_a0['区别色候选数'] == 0 and t2_a0['外环证实背景本色'] == 0
    assert t2_a0['同色不适用组数'] == 3 and t2_a0['同色不适用施工像素'] == 1260
    assert t2_a0['确认残留'] == 0 and t2_a0['待目检'] == 0 and t2_a0['旧口径蓝色候选'] == 1260

    assert result['汇总']['旧口径蓝色候选'] == 2520
    assert result['汇总']['区别色候选数'] == 1260
    assert result['汇总']['同色不适用组数'] == 3
    assert result['汇总']['同色不适用施工像素'] == 1260
    stdout = capsys.readouterr().out
    assert '区别色候选数=1260' in stdout
    assert '同色不适用组数=3' in stdout
    assert '同色不适用施工像素=1260' in stdout


def test_offline_recheck_attempt_mode_best_and_panels_top_dedup(tmp_path):
    run = tmp_path / 'run'
    for sub in ['frames', 'masks', 'edge-band', 'scene-protection', 'roi-0001-baseline']:
        (run / sub).mkdir(parents=True)
    h, w = 160, 220
    blue = _hsv_bgr(100, 200, 200, (h, w))
    skin = _hsv_bgr(10, 180, 180, (h, w))
    gray = np.full((h, w, 3), 40, np.uint8)

    # 组A（左上）：区别色蓝色；组B（右下）：外环被保护区遮住产生待目检
    mask_a = np.zeros((h, w), np.uint8)
    mask_a[15:35, 15:35] = 255  # 400 px
    mask_b = np.zeros((h, w), np.uint8)
    mask_b[100:120, 150:170] = 255  # 400 px
    mask = mask_a | mask_b
    band = np.zeros((h, w), np.uint8)
    guard = np.zeros((h, w), np.uint8)
    guard[90:130, 140:180] = 255
    guard[100:120, 150:170] = 0

    src = gray.copy()
    src[mask_a > 0] = blue[mask_a > 0]
    src[mask_b > 0] = skin[mask_b > 0]

    indices = np.flatnonzero(mask.ravel()).astype(np.int32)
    delta0 = run / 'roi-0001-attempt-00' / 'staged-delta'
    delta1 = run / 'roi-0001-attempt-01' / 'staged-delta'
    delta0.mkdir(parents=True)
    delta1.mkdir(parents=True)

    # 5帧设定（attempt 1 作为最佳尝试）：
    # 帧0: 无残留无待目检 (0, 0) -> 像素数为0不取
    # 帧1: 确认残留=400, 待目检=400 (B无外环且A全蓝)
    # 帧2: 确认残留=100, 待目检=400
    # 帧3: 确认残留=200, 待目检=0 (去掉B区guard) -> 确认残留第2名
    # 帧4: 确认残留=0, 待目检=400 -> 与帧1、帧2共同构成待目检前2名
    painted_spec = [
        (0, 0, False),
        (20, 400, True),
        (5, 400, True),
        (10, 0, False),
        (0, 400, True),
    ]
    for i, (rows_a, _, protect_b) in enumerate(painted_spec):
        name = f'{i:06d}.png'
        cv2.imwrite(str(run / 'frames' / name), src)
        cv2.imwrite(str(run / 'roi-0001-baseline' / name), src)
        cv2.imwrite(str(run / 'masks' / name), mask)
        cv2.imwrite(str(run / 'edge-band' / name), band)
        g_frame = guard.copy()
        if not protect_b:
            # 帧0和帧3在B外环也无保护区，B变成同色不适用或无彩色（周围40灰）且与外环40灰比？
            # 让B周围也是skin以便不产生待目检（同色不适用）
            src_i = src.copy()
            src_i[90:130, 140:180] = skin[90:130, 140:180]
            cv2.imwrite(str(run / 'frames' / name), src_i)
            cv2.imwrite(str(run / 'roi-0001-baseline' / name), src_i)
            g_frame[:] = 0
        cv2.imwrite(str(run / 'scene-protection' / name), g_frame)

        # attempt 0：全留蓝（仅用于测试 attempt-mode best 会跳过 attempt 0）
        np.savez_compressed(delta0 / f'{i:06d}.npz', indices=indices, pixels=src.reshape(-1, 3)[indices])

        # attempt 1：按设计控制确认残留与待目检数量
        painted = gray.copy()
        if rows_a > 0:
            painted[15:15 + rows_a, 15:35] = blue[15:15 + rows_a, 15:35]
        if not protect_b:
            painted[mask_b > 0] = skin[mask_b > 0]
        np.savez_compressed(delta1 / f'{i:06d}.npz', indices=indices, pixels=painted.reshape(-1, 3)[indices])

    (run / 'detections.json').write_text(json.dumps(dict(scenes=[[0, 5]])))
    report = dict(
        任务=[
            dict(
                任务=1,
                图层=0,
                开始帧=0,
                结束帧=5,
                ROI=[0, 0, w, h],
                最佳尝试=1,
                尝试=[
                    dict(尝试=0, 重叠最大差异=0.0, 错误码='RESIDUE_FOUND'),
                    dict(尝试=1, 重叠最大差异=0.0, 错误码='RESIDUE_FOUND'),
                ],
            )
        ]
    )
    (run / 'roi-report.json').write_text(json.dumps(report, ensure_ascii=False))

    out_json = tmp_path / 'out' / 'best.json'
    panels_dir = tmp_path / 'panels'
    args = build_parser().parse_args([
        '--run', str(run),
        '--tasks', '1',
        '--config', 'config/v17-edge-residue.json',
        '--output', str(out_json),
        '--workers', '1',
        '--attempt-mode', 'best',
        '--panels-dir', str(panels_dir),
        '--panel-top', '2',
    ])
    result = execute(args)

    assert result['attempt_mode'] == 'best'
    assert result['已复验尝试数'] == 1
    assert len(result['任务'][0]['尝试']) == 1
    assert result['任务'][0]['尝试'][0]['尝试'] == 1
    # 确认残留前2名: 帧1(400), 帧3(200); 待目检前2名: 帧1(400), 帧2(400) [帧4同为400但帧号靠后]; 去重共3帧: {1, 2, 3}
    # 帧0 (0,0) 必不取
    manifest = json.loads((panels_dir / 'panels.json').read_text())
    assert manifest['总数'] == 3
    got_frames = [item['帧'] for item in manifest['panels']]
    assert got_frames == [1, 2, 3]
    expected_counts = {1: (400, 400), 2: (100, 400), 3: (200, 0)}
    for item in manifest['panels']:
        r_px, p_px = expected_counts[item['帧']]
        assert item['任务'] == 1
        assert item['确认残留像素'] == r_px and item['待目检像素'] == p_px
        assert item['文件名'] == f"task01-frame{item['帧']:06d}-residue{r_px}-pending{p_px}.png"
        assert item['宽'] == 4 * w and item['高'] == h + PANEL_LABEL_HEIGHT
        img = cv2.imread(str(panels_dir / item['文件名']))
        assert img is not None and img.shape == (h + PANEL_LABEL_HEIGHT, 4 * w, 3)
        # 确认第3栏把确认残留涂纯红(0,0,255)，第4栏把待目检涂纯黄(0,255,255)
        if r_px > 0:
            assert tuple(img[PANEL_LABEL_HEIGHT + 15, 2 * w + 15]) == (0, 0, 255)
        if p_px > 0:
            assert tuple(img[PANEL_LABEL_HEIGHT + 100, 3 * w + 150]) == (0, 255, 255)


def test_offline_recheck_reconstructs_passed_task_from_roi_layer_and_errors_when_incomplete(tmp_path):
    run = tmp_path / 'run'
    for sub in ['frames', 'masks', 'layer-00', 'edge-band', 'roi-layer-00', 'roi-coverage']:
        (run / sub).mkdir(parents=True)
    h, w = 160, 220
    blue = _hsv_bgr(100, 200, 200, (h, w))
    gray = np.full((h, w, 3), 40, np.uint8)

    # 同帧左右两个独立连通域，保证相邻任务块会落入彼此 padded ROI 内，测试 masks&ownership 严格分离。
    m1 = np.zeros((h, w), np.uint8)
    m1[20:40, 20:40] = 255  # 400 px
    m2 = np.zeros((h, w), np.uint8)
    m2[20:40, 110:130] = 255  # 400 px
    union = m1 | m2

    src = gray.copy()
    src[union > 0] = blue[union > 0]
    # 任务1（左）清干净；任务2（右）在 roi-layer-00 中故意留源蓝色，检验按 ownership 只报任务2残留而不误伤任务1
    layer_out = gray.copy()
    layer_out[m2 > 0] = blue[m2 > 0]

    for i in range(4):
        name = f'{i:06d}.png'
        cv2.imwrite(str(run / 'frames' / name), src)
        cv2.imwrite(str(run / 'masks' / name), union)
        cv2.imwrite(str(run / 'layer-00' / name), union)
        cv2.imwrite(str(run / 'edge-band' / name), np.zeros((h, w), np.uint8))
        cv2.imwrite(str(run / 'roi-layer-00' / name), layer_out)
        cv2.imwrite(str(run / 'roi-coverage' / name), union)

    spatial = SpatialSettings(dilation_pixels=0, merge_gap_pixels=0, minimum_width=64, minimum_height=64, fixed_context_pixels=32, maximum_crop_pixels=40000, maximum_pixel_frames=2000000)
    roi_cfg = ROISettings(sequential_layers=True, maximum_frames=4, minimum_frames=2, minimum_context=32, overlap_frames=1)
    hls = HLSSettings(window_frames=4, overlap_frames=1)
    (run / 'spatial-roi.json').write_text(json.dumps(asdict(spatial)))
    (run / 'layers.json').write_text(json.dumps(dict(layers=[dict(directory=str(run / 'layer-00'))])))
    (run / 'detections.json').write_text(json.dumps(dict(scenes=[[0, 4]])))

    budget = max(r - l for l, r, _ in seam_variants(0, 4, 0, 4, hls, roi_cfg))
    jobs = plan(union, budget, roi_cfg, spatial)
    assert len(jobs) == 2

    tasks = []
    for tid, job in enumerate(jobs, 1):
        tasks.append(dict(
            任务=tid,
            图层=0,
            开始帧=0,
            结束帧=4,
            ROI=job['box'],
            施工面积像素=int(np.count_nonzero(job['ownership'])),
            通过=True,
            尝试=[dict(尝试=0, 重叠最大差异=0.0, 错误码='SEAM_UNCERTAIN'), dict(尝试=1, 重叠最大差异=0.0, 通过=True)],
        ))
    report = dict(
        空间配置=asdict(spatial),
        配置=asdict(roi_cfg),
        窗口=4,
        重叠=1,
        任务=tasks,
    )
    (run / 'roi-report.json').write_text(json.dumps(report, ensure_ascii=False))

    out_json = tmp_path / 'out' / 'passed_best.json'
    panels_dir = tmp_path / 'panels_passed'
    res = execute(SimpleNamespace(
        run=run,
        tasks='1-2',
        config=Path('config/v17-edge-residue.json'),
        output=out_json,
        workers=2,
        attempt_mode='best',
        panels_dir=panels_dir,
        panel_top=2,
    ))
    t1_row = res['任务'][0]['尝试'][0]
    t2_row = res['任务'][1]['尝试'][0]
    assert t1_row['重建来源'] == 'roi-layer-00' and t1_row['判定'] == '通过' and t1_row['确认残留'] == 0
    assert t2_row['重建来源'] == 'roi-layer-00' and t2_row['判定'] == 'RESIDUE_FOUND' and t2_row['确认残留'] == 1600
    manifest = json.loads((panels_dir / 'panels.json').read_text())
    assert manifest['总数'] == 2 and all(item['任务'] == 2 for item in manifest['panels'])

    # 覆盖不完整时必须报错，不能静默跳过
    cv2.imwrite(str(run / 'roi-coverage' / '000001.png'), m1)
    with pytest.raises(ValueError, match='roi-coverage 未覆盖全部施工像素'):
        execute(SimpleNamespace(
            run=run,
            tasks='2',
            config=Path('config/v17-edge-residue.json'),
            output=out_json,
            workers=1,
            attempt_mode='best',
        ))

    # 缺少 roi-layer-00 时必须报错
    (run / 'roi-layer-00' / '000001.png').unlink()
    with pytest.raises(ValueError, match='源帧或图层输出无效'):
        execute(SimpleNamespace(
            run=run,
            tasks='1',
            config=Path('config/v17-edge-residue.json'),
            output=out_json,
            workers=1,
            attempt_mode='best',
        ))

    # attempt-mode best 选中未通过且无 staged-delta 的尝试时必须报错
    report['任务'][0]['通过'] = False
    report['任务'][0]['最佳尝试'] = 0
    report['任务'][0]['尝试'] = [dict(尝试=0, 重叠最大差异=0.0, 错误码='GPU_OOM')]
    (run / 'roi-report.json').write_text(json.dumps(report, ensure_ascii=False))
    with pytest.raises(ValueError, match='缺少 staged-delta'):
        execute(SimpleNamespace(
            run=run,
            tasks='1',
            config=Path('config/v17-edge-residue.json'),
            output=out_json,
            workers=1,
            attempt_mode='best',
        ))

