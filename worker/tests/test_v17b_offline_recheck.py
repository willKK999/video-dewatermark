"""v17b 离线复验脚本的合成回归：验证增量重建、多进程复检、旧口径对照且不写任何图片。"""
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from scripts.v17b_offline_recheck import execute, parse_tasks


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


def test_offline_recheck_synthetic_run_with_multiprocessing(tmp_path):
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
    assert t1_a0['颜色候选'] == 1260 and t1_a0['遮罩内确认残留'] == 1200 and t1_a0['收边带确认残留'] == 60
    assert t1_a0['确认残留'] == 1260 and t1_a0['待目检'] == 0 and t1_a0['单帧最大未解释'] == 420
    assert t1_a0['旧口径蓝色候选'] == 1260

    assert t1_a1['有效施工帧数'] == 3 and t1_a1['判定'] == '通过'
    assert t1_a1['颜色候选'] == 0 and t1_a1['确认残留'] == 0 and t1_a1['旧口径蓝色候选'] == 0

    t2_a0 = loaded['任务'][1]['尝试'][0]
    assert t2_a0['有效施工帧数'] == 3 and t2_a0['判定'] == '通过'
    assert t2_a0['颜色候选'] == 1260 and t2_a0['外环证实背景本色'] == 1260
    assert t2_a0['确认残留'] == 0 and t2_a0['待目检'] == 0 and t2_a0['旧口径蓝色候选'] == 1260
    assert result['汇总']['旧口径蓝色候选'] == 2520
