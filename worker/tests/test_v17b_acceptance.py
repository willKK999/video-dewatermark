"""v17b 只读验收脚本的合成回归：验证三版同口径闪烁、清晰度、原生保护汇总及只写 JSON。"""
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
import pytest
from scripts.v17b_acceptance import build_parser, execute
from worker.edge_residue import highlight_guard


def _encode_lossless_mp4(path, frames, fps=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    proc = subprocess.run(
        [
            'ffmpeg', '-y', '-v', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}',
            '-r', str(fps), '-i', '-',
            '-c:v', 'libx264rgb', '-preset', 'ultrafast', '-crf', '0',
            str(path),
        ],
        input=b''.join(f.tobytes() for f in frames),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode('utf-8', 'replace'))


def _textured_frame(h, w, shift):
    yy, xx = np.mgrid[0:h, 0:w]
    texture = 120 + 60 * np.sin((xx + shift * 3) * 0.45) * np.cos((yy + shift * 2) * 0.45)
    img = np.repeat(np.clip(np.rint(texture), 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    # 放入低饱和高亮白条边（激发 highlight_guard）
    img[10:30, 10:30] = (240, 240, 240)
    return img


def test_v17b_acceptance_three_versions_and_protection_metrics(tmp_path):
    base = tmp_path / 'data'
    run = base / 'v17b' / 'run'
    masks_dir = base / 'v13' / 'masks-run-strokes'
    edge_band_dir = base / 'v17' / 'masks-run' / 'edge-band'
    for d in [
        run / 'frames',
        run / 'masks',
        run / 'composite',
        masks_dir / 'masks',
        masks_dir / 'scene-protection',
        masks_dir / 'subtitle-protection',
        edge_band_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    h, w, n = 128, 160, 6
    oldmask = np.zeros((h, w), np.uint8)
    oldmask[50:90, 60:110] = 255
    newmask = oldmask.copy()
    newmask[50:52, 60:65] = 0  # 每帧故意丢失 2x5=10 像素

    band = np.zeros((h, w), np.uint8)
    band[9:12, 9:12] = 255  # 与 highlight_guard 在白块边缘重叠
    band[100:104, 100:104] = 255  # 16 px 与 scene/subtitle protection 重叠

    scene_guard = np.zeros((h, w), np.uint8)
    scene_guard[100:102, 100:104] = 255  # 8 px
    sub_guard = np.zeros((h, w), np.uint8)
    sub_guard[102:104, 100:104] = 255  # 8 px（合计 16 px）

    sources = [_textured_frame(h, w, i) for i in range(n)]
    # 第 3 帧模拟镜头切换（整体大幅变化）
    sources[3] = np.clip(sources[3].astype(np.int16) + 50, 0, 255).astype(np.uint8)

    v13e_frames = [f.copy() for f in sources]
    v16_frames = [f.copy() for f in sources]
    v17b_frames = [f.copy() for f in sources]

    # v13e 第 2 帧在掩膜区内制造大幅亮度跳变 -> 触发 1 个闪烁候选（帧3在cuts里会被排除）
    v13e_frames[2][oldmask > 0] = 240
    # v16 掩膜区做高斯模糊 -> 清晰度下降 (< 0.7)
    for i in range(n):
        blurred = cv2.GaussianBlur(sources[i], (9, 9), 3.0)
        v16_frames[i][oldmask > 0] = blurred[oldmask > 0]

    expected_hl = 0
    for i, src in enumerate(sources):
        name = f'{i:06d}.png'
        cv2.imwrite(str(run / 'frames' / name), src)
        cv2.imwrite(str(run / 'masks' / name), newmask)
        cv2.imwrite(str(masks_dir / 'masks' / name), oldmask)
        cv2.imwrite(str(masks_dir / 'scene-protection' / name), scene_guard)
        cv2.imwrite(str(masks_dir / 'subtitle-protection' / name), sub_guard)
        cv2.imwrite(str(edge_band_dir / name), band)

        comp = src.copy()
        # 在确认域外 (0, 0) 修改 1 像素（每帧1px，共6px）
        comp[0, 0] = (0, 0, 0) if tuple(src[0, 0]) != (0, 0, 0) else (1, 1, 1)
        cv2.imwrite(str(run / 'composite' / name), comp)
        expected_hl += int(np.count_nonzero((band > 0) & highlight_guard(src)))

    (run / 'detections.json').write_text(json.dumps(dict(scenes=[[0, 3], [3, 6]])))

    v13e_mp4 = base / 'v13' / 'inpaint-v6-retries' / '第1块.mp4'
    v16_mp4 = base / 'v16' / 'inpaint-v6-retries-context' / '第1块.mp4'
    v17b_mp4 = base / 'v17b' / '第1块.mp4'
    _encode_lossless_mp4(v13e_mp4, v13e_frames, fps=5)
    _encode_lossless_mp4(v16_mp4, v16_frames, fps=5)
    _encode_lossless_mp4(v17b_mp4, v17b_frames, fps=5)

    out_dir = tmp_path / 'out'
    out_json = out_dir / 'acceptance.json'
    args = build_parser().parse_args([
        '--base', str(base),
        '--run', str(run),
        '--video', str(v17b_mp4),
        '--output', str(out_json),
    ])
    result = execute(args)

    assert out_json.is_file()
    # 只写 JSON，绝不写图片
    assert not list(out_dir.rglob('*.png')) and not list(out_dir.rglob('*.jpg')) and not list(out_dir.rglob('*.jpeg'))

    loaded = json.loads(out_json.read_text())
    assert loaded['帧数'] == n and loaded['源片镜头切换'] == [0, 3]
    assert set(loaded['版本'].keys()) == {'v13e', 'v16', 'v17b'}
    assert loaded['版本']['v13e']['闪烁候选数'] == 1
    assert loaded['版本']['v13e']['闪烁候选'][0]['frame'] == 2
    assert loaded['版本']['v17b']['闪烁候选数'] == 0
    assert loaded['版本']['v17b']['清晰度']['最小保留比'] > 0.9
    assert loaded['版本']['v16']['清晰度']['低于0.7样本数'] >= 1

    prot = loaded['原生保护汇总']
    assert prot['新增像素'] == 25 * n
    assert prot['场景字幕保护交集'] == 16 * n
    assert prot['高亮边保护交集'] == expected_hl and expected_hl > 0
    assert prot['未编码确认域外修改像素'] == n
    assert prot['原确认域丢失像素'] == 10 * n
    assert result['原生保护汇总'] == prot

    # 不传 --base 时显式传齐全部路径应得到完全相同的汇总结果
    explicit_json = out_dir / 'acceptance_explicit.json'
    explicit_args = build_parser().parse_args([
        '--run', str(run),
        '--video', str(v17b_mp4),
        '--v13e', str(v13e_mp4),
        '--v16', str(v16_mp4),
        '--masks', str(masks_dir),
        '--edge-band', str(edge_band_dir),
        '--output', str(explicit_json),
    ])
    explicit_result = execute(explicit_args)
    assert explicit_result['原生保护汇总'] == prot
    assert {k: v['闪烁候选数'] for k, v in explicit_result['版本'].items()} == {'v13e': 1, 'v16': 0, 'v17b': 0}


def test_v17b_acceptance_errors_on_missing_inputs(tmp_path):
    args = build_parser().parse_args([
        '--run', str(tmp_path / 'run'),
        '--video', str(tmp_path / 'v17b.mp4'),
        '--output', str(tmp_path / 'out.json'),
    ])
    with pytest.raises(ValueError, match='缺少输入路径'):
        execute(args)
