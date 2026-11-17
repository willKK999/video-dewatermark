"""v17c 通用修复单元测试（合成数据：半透明淡字、同色孔洞、局部背景举证、清晰度与有界补修）。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.v17c_faint_diagnose import diagnose_faint
from scripts.v17c_refine_masks import refine_v17c_masks
from worker.edge_residue import (
    EdgeSettings,
    complete_group_color_holes,
    measure,
    refine_group,
)
from worker.faint_watermark import (
    FaintSettings,
    build_faint_frame_addition,
    detect_translucent_candidates,
    reclassify_watermark_subtitles,
)
from worker.hls_layers import write_manifest
from worker.process import PipelineError
from worker.restoration import CleanReferences, restore_orthogonal_luma_detail
from worker.roi_spatial_layers import _repair_confirmed_residue_pass
from worker.trusted_context import ContextSettings


def _textured_moving_bg(frame_idx: int, h: int = 240, w: int = 320, shift_x: int = 9, shift_y: int = 5) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = rng.integers(50, 140, size=(h + 240, w + 320, 3), dtype=np.uint8)
    # 铺设周期大块纹理保证平滑区域同时存在边缘与纹理运动
    yy, xx = np.mgrid[0 : h + 240, 0 : w + 320]
    pattern = ((np.sin(xx / 7.0) + np.cos(yy / 9.0)) * 22.0).astype(np.int16)
    canvas = np.clip(base.astype(np.int16) + pattern[..., None], 25, 190).astype(np.uint8)
    ox = (frame_idx * shift_x) % 240
    oy = (frame_idx * shift_y) % 200
    return canvas[oy : oy + h, ox : ox + w].copy()


def _paint_translucent_track_frame(frame_idx: int, total_h: int = 240, total_w: int = 320) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    img = _textured_moving_bg(frame_idx, total_h, total_w)
    # 随帧漂移的核心框与右下方半透明未遮蔽后缀
    x0 = 40 + frame_idx * 3
    y0 = 40 + frame_idx * 2
    core_w, core_h = 80, 28
    # 核心条纹（已有确认掩膜）
    img[y0 + 6 : y0 + 22, x0 + 8 : x0 + 70] = np.clip(img[y0 + 6 : y0 + 22, x0 + 8 : x0 + 70].astype(np.int16) + 85, 0, 255).astype(np.uint8)
    conf = np.zeros((total_h, total_w), dtype=np.uint8)
    conf[y0 + 4 : y0 + 24, x0 + 6 : x0 + 72] = 255

    # 右上与下方未入掩膜的半透明淡字笔画（3-4px 宽文字笔画级高对比边缘，贴近实际字幕/商标）
    faint_gt = np.zeros((total_h, total_w), dtype=bool)
    faint_gt[y0 + 6 : y0 + 18, x0 + 86 : x0 + 90] = True
    faint_gt[y0 + 6 : y0 + 10, x0 + 84 : x0 + 96] = True
    faint_gt[y0 + 34 : y0 + 38, x0 + 20 : x0 + 56] = True
    faint_gt[y0 + 34 : y0 + 44, x0 + 34 : x0 + 38] = True
    img[faint_gt] = np.clip(img[faint_gt].astype(np.int16) + 34, 0, 255).astype(np.uint8)
    return img, conf, faint_gt, [float(x0), float(y0), float(x0 + core_w), float(y0 + core_h)]


def test_translucent_track_and_cross_transfer_detection():
    """验证高运动轨迹直接检出及跨镜头低运动轨迹模板迁移。"""
    n_s0, n_s1 = 16, 14
    frames: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    faint_gts: list[np.ndarray] = []
    hints_s0, hints_s1 = [], []

    for i in range(n_s0):
        im, msk, gt, bx = _paint_translucent_track_frame(i)
        frames.append(im)
        confs.append(msk)
        faint_gts.append(gt)
        hints_s0.append({'frame': i, 'box': bx})

    # 第二镜头背景几乎静止（平滑室内背景），核心条纹与后缀位置结构与第一镜头一致
    yy_s, xx_s = np.mgrid[0:240, 0:320]
    bg_static = np.clip(95 + (14 * np.sin(xx_s / 18.0) + 10 * np.cos(yy_s / 16.0))[..., None], 40, 160).astype(np.uint8).repeat(3, axis=2)
    for k in range(n_s1):
        f = n_s0 + k
        im = bg_static.copy()
        x0, y0 = 60, 120
        im[y0 + 6 : y0 + 22, x0 + 8 : x0 + 70] = np.clip(im[y0 + 6 : y0 + 22, x0 + 8 : x0 + 70].astype(np.int16) + 85, 0, 255).astype(np.uint8)
        msk = np.zeros(im.shape[:2], dtype=np.uint8)
        msk[y0 + 4 : y0 + 24, x0 + 6 : x0 + 72] = 255
        faint_gt = np.zeros(im.shape[:2], dtype=bool)
        faint_gt[y0 + 6 : y0 + 18, x0 + 86 : x0 + 90] = True
        faint_gt[y0 + 6 : y0 + 10, x0 + 84 : x0 + 96] = True
        faint_gt[y0 + 34 : y0 + 38, x0 + 20 : x0 + 56] = True
        faint_gt[y0 + 34 : y0 + 44, x0 + 34 : x0 + 38] = True
        im[faint_gt] = np.clip(im[faint_gt].astype(np.int16) + 34, 0, 255).astype(np.uint8)
        frames.append(im)
        confs.append(msk)
        faint_gts.append(faint_gt)
        hints_s1.append({'frame': f, 'box': [float(x0), float(y0), float(x0 + 80), float(y0 + 28)]})

    detections = {
        'scenes': [[0, n_s0], [n_s0, n_s0 + n_s1]],
        'tracks': [
            {'id': 1, 'scene': 0, 'hints': hints_s0},
            {'id': 2, 'scene': 1, 'hints': hints_s1},
        ],
    }
    res = detect_translucent_candidates(
        total_frames=len(frames),
        detections=detections,
        load_bgr=lambda f: frames[f],
        load_confirmed=lambda f: confs[f],
        cfg=FaintSettings(sample_frames=12, min_track_frames=8, min_unmasked_samples=4),
        enable_static_scene=False,
    )
    # 第0镜头直接召回核心区外的淡字，第1镜头经跨镜头互相关迁移召回
    for idx in (4, n_s0 + 3):
        pred = res['per_frame_masks'][idx]
        gt = faint_gts[idx]
        recall = float(np.count_nonzero(pred & gt)) / float(np.count_nonzero(gt))
        outside_false = int(np.count_nonzero(pred & (~gt) & (confs[idx] == 0)))
        assert recall >= 0.80, f'frame={idx} recall={recall}'
        assert outside_false <= 15


def test_reclassify_ocr_subtitles_keeps_dialogue_and_respects_budget():
    """孤立OCR水印碎片释放后不得伤及真实对话字幕与20%面积约束。"""
    h, w = 240, 320
    img = np.full((h, w, 3), 45, dtype=np.uint8)
    # 底部真实高对比横排对话字幕行（detect_text_rows 应保护）
    cv2.putText(img, 'HELLO WORLD TEST LINE', (30, 212), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (250, 250, 250), 2)

    sub_prot = np.zeros((h, w), dtype=np.uint8)
    # 真实对话区域保护 + 右侧非对话孤立OCR框
    sub_prot[188:225, 25:295] = 255
    sub_prot[60:78, 200:222] = 255

    translucent = np.zeros((h, w), dtype=bool)
    translucent[62:76, 202:218] = True  # 独立检测到的半透明笔画

    ocr_items = [
        {'text': 'HELLO WORLD TEST LINE', 'box': [25, 188, 295, 225]},
        {'text': 'AI', 'box': [200, 60, 222, 78]},
    ]
    upd_sub, audit, row_mask = reclassify_watermark_subtitles(
        0, img, sub_prot, ocr_items, translucent, cfg=FaintSettings()
    )
    # 对话框保持受保护，AI 碎片被精确重分类
    assert len(audit) == 1 and audit[0]['text'] == 'AI'
    assert np.all(upd_sub[row_mask] == 255)
    assert np.count_nonzero(upd_sub[60:78, 200:222]) == 0

    # 验证 build_faint_frame_addition 遵守20%限制且不侵入场景保护
    big_conf = np.zeros((h, w), dtype=np.uint8)
    big_conf[:47, :] = 255  # 约19.5%，接近20%
    scene_prot = np.zeros((h, w), dtype=np.uint8)
    scene_prot[65:70, 205:210] = 255
    added, info = build_faint_frame_addition(
        img, translucent, big_conf, scene_prot | upd_sub, max_layer_fraction=0.2
    )
    assert np.count_nonzero(added & scene_prot) == 0


def test_complete_group_color_holes_and_guards():
    """验证邻帧重合证据补全同色掉块角，无邻帧支撑或保护区像素绝不补入。"""
    h, w = 160, 160
    cfg = EdgeSettings(shape_radius=1, require_ring_for_refinement=True)
    blue = np.array([220, 60, 30], dtype=np.uint8)
    img = np.full((h, w, 3), 120, dtype=np.uint8)

    # 图标主体 + 左下角同色脱落区域
    full_icon = np.zeros((h, w), dtype=bool)
    full_icon[40:80, 60:110] = True
    dropped = np.zeros((h, w), dtype=bool)
    dropped[80:95, 50:66] = True  # 掉块240像素
    img[full_icon | dropped] = blue

    group_cur = (full_icon.astype(np.uint8) * 255)
    conf_cur = group_cur.copy()
    prot = np.zeros((h, w), dtype=np.uint8)
    # 邻居帧包含掉块区域（且整体稍微平移验证跟踪）
    nbrs = []
    for dx in (-2, -1, 1, 2):
        nm = np.zeros((h, w), dtype=np.uint8)
        nm[40:95, 50 + dx : 110 + dx] = 255
        nbrs.append(nm)

    added, info = complete_group_color_holes(img, group_cur, conf_cur, prot, nbrs, cfg)
    assert info['新增像素'] >= 200
    assert np.all(added[dropped] == 255)

    # 若放入保护区或无足够邻帧支撑，均不得补入
    prot_block = prot.copy()
    prot_block[dropped] = 255
    added_p, _ = complete_group_color_holes(img, group_cur, conf_cur, prot_block, nbrs, cfg)
    assert int(np.count_nonzero(added_p)) == 0
    added_no_nbr, _ = complete_group_color_holes(img, group_cur, conf_cur, prot, [group_cur] * 3, cfg)
    assert int(np.count_nonzero(added_no_nbr)) == 0


def test_local_clean_references_distinguishes_background_vs_true_residue_and_padding():
    """验证局部背景举证仅清除有证据的同色背景，真残留与无证据候选绝不误判通过。"""
    h, w = 220, 260
    cfg = EdgeSettings(shape_radius=1)
    # 构造带丰富清晰角点的高饱和紫色海报背景，同时底部放入干扰反方向运动前景（破坏全局仿射）
    bg_full = np.full((h + 60, w + 80, 3), 90, dtype=np.uint8)
    for gy in range(8, h + 52, 18):
        for gx in range(8, w + 72, 18):
            cv2.rectangle(bg_full, (gx, gy), (gx + 7, gy + 7), (150, 145, 140), -1)
    purple = np.array([210, 60, 165], dtype=np.uint8)  # HSV 约 H~138, S~180
    bg_full[70:115, 110:155] = purple

    frames, ref_masks = [], []
    for k in range(10):
        dx = k * 3
        frm = bg_full[20 : 20 + h, 10 + dx : 10 + dx + w].copy()
        # 前景区块反方向运动破坏全局仿射，仅保留海报侧局部运动可用
        frm[150:210, 10 : 70] = (k * 20) % 255
        rm = np.zeros((h, w), dtype=np.uint8)
        # 在第0帧海报紫色块被遮罩，之后3..9帧水印移走、紫色块裸露！
        if k == 0:
            rm[50:95, 100:145] = 255
        else:
            rm[10:40, 200:235] = 255
        frames.append(frm)
        ref_masks.append(rm)

    src0, msk0 = frames[0], ref_masks[0]
    bnd0 = np.zeros_like(msk0)
    cr = CleanReferences(frames[1:], ref_masks[1:])

    # 情况 A：修复结果干净还原海报紫色图案（且加轻微色相边界漂移测试）-> 局部时域证实通过
    painted_ok = src0.copy()
    ev_ok = cr.reference_evidence(src0, msk0, confirmed=msk0)
    res_ok = measure(src0, painted_ok, msk0, bnd0, cfg, reference_evidence=ev_ok, confirmed=msk0)
    assert res_ok['通过'] is True and res_ok['时域证实背景本色'] > 0

    # 情况 B：在无紫色的灰墙区域 [10..40, 200..235] 的修复输出意外残余高饱和紫色/蓝色 -> 时域反证保持 RESIDUE_FOUND
    src_wall = frames[1].copy()
    msk_wall = np.zeros((h, w), dtype=np.uint8)
    msk_wall[50:80, 170:200] = 255
    src_wall[50:80, 170:200] = purple
    painted_bad = src_wall.copy()
    cr_wall = CleanReferences([f for i, f in enumerate(frames) if i != 1], [np.zeros((h, w), dtype=np.uint8)] * 9)
    ev_bad = cr_wall.reference_evidence(src_wall, msk_wall, confirmed=msk_wall)
    res_bad = measure(src_wall, painted_bad, msk_wall, bnd0, cfg, reference_evidence=ev_bad, confirmed=msk_wall)
    assert res_bad['判定'] == 'RESIDUE_FOUND' and res_bad['通过'] is False

    # 情况 C：所有参考帧都在同一位置被遮蔽（无参考证据）且无外环支持 -> 绝不通过
    cr_none = CleanReferences(frames[1:], [msk0.copy()] * 9)
    ev_none = cr_none.reference_evidence(src0, msk0, confirmed=msk0)
    res_none = measure(src0, painted_ok, msk0, bnd0, cfg, reference_evidence=ev_none, confirmed=msk0)
    assert res_none['通过'] is False

    # 情况 D：验证模8填充差异（source/painted小于reference形状时）不越界崩溃
    ev_pad = cr.reference_evidence(src0, msk0, confirmed=msk0)
    cropped_h, cropped_w = h - 3, w - 5
    res_pad = measure(
        src0[:cropped_h, :cropped_w],
        painted_ok[:cropped_h, :cropped_w],
        msk0[:cropped_h, :cropped_w],
        bnd0[:cropped_h, :cropped_w],
        cfg,
        reference_evidence={
            k: (v[:cropped_h, :cropped_w] if hasattr(v, 'shape') else v) for k, v in ev_pad.items()
        },
        confirmed=msk0[:cropped_h, :cropped_w],
    )
    assert res_pad['通过'] is True


def test_restore_orthogonal_luma_detail_preserves_bg_blocks_strokes():
    """验证正交亮度细节恢复在无色安全区提升纹理保留，同时绝不回注水印笔画或新色相。"""
    h, w = 128, 128
    yy, xx = np.mgrid[0:h, 0:w]
    bg_luma = (110 + 18 * np.sin(xx / 2.2) * np.cos(yy / 2.2)).astype(np.float32)
    src = np.repeat(np.clip(bg_luma, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
    # 中心条带叠加清晰白字水印笔画
    src[56:72, 48:80] = 240
    msk = np.zeros((h, w), dtype=np.uint8)
    msk[20:108, 20:108] = 255
    # 模拟 ProPainter 把局部高对比细节平滑掉
    repaired = cv2.GaussianBlur(np.repeat(np.clip(bg_luma, 0, 255).astype(np.uint8)[..., None], 3, axis=2), (7, 7), 2.5)
    wm_hp = np.zeros((h, w), dtype=np.float32)
    wm_hp[56:72, 48:80] = 80.0

    out = restore_orthogonal_luma_detail(src, repaired, msk, wm_hp, EdgeSettings())
    # 安全外围区域细节应更接近源图高频纹理，而水印核心区域不应出现240高亮回注
    safe_zone = np.zeros((h, w), dtype=bool)
    safe_zone[24:44, 24:104] = True
    mae_rep = float(np.abs(repaired[safe_zone].astype(float) - src[safe_zone].astype(float)).mean())
    mae_out = float(np.abs(out[safe_zone].astype(float) - src[safe_zone].astype(float)).mean())
    assert mae_out < mae_rep
    assert float(out[58:70, 50:78].mean()) < 145.0


def test_repair_confirmed_residue_pass_bounded(tmp_path: Path, monkeypatch):
    """验证确认残留有界一轮补修：仅对确认残留膨胀区域补画并重检，若仍残留或接缝超限如实失败。"""
    from types import SimpleNamespace
    from worker import roi_spatial_layers as rsl
    from worker.hls_settings import HLSSettings

    work = tmp_path / 'work'
    task = work / 'roi-0001-attempt-00'
    staged = task / 'staged'
    staged.mkdir(parents=True)
    (work / 'masks').mkdir(parents=True)
    (work / 'edge-band').mkdir(parents=True)
    h, w = 64, 64
    blue = np.array([220, 40, 20], dtype=np.uint8)
    grey = np.full((h, w, 3), 128, dtype=np.uint8)

    files = []
    masks = []
    current = []
    for j in range(3):
        p = tmp_path / f'{j:06d}.png'
        src_j = grey.copy()
        src_j[16:48, 16:48] = blue
        cv2.imwrite(str(p), src_j)
        files.append(p)
        current.append(p)

        m = np.zeros((h, w), dtype=np.uint8)
        m[16:48, 16:48] = 255
        masks.append(m)
        cv2.imwrite(str(work / 'masks' / p.name), m)
        cv2.imwrite(str(work / 'edge-band' / p.name), np.zeros((h, w), dtype=np.uint8))

        st_img = grey.copy()
        if j == 1:
            st_img[24:38, 24:38] = blue  # 确认残留核
        cv2.imwrite(str(staged / p.name), st_img)

    edge_cfg = EdgeSettings(shape_radius=1, residue_repair_rounds=1, residue_repair_dilate_radius=4)
    ownership = np.ones((h, w), dtype=np.uint8) * 255
    box = (0, 0, w, h)
    scores_before = []
    for j in range(3):
        st_j = cv2.imread(str(staged / files[j].name))
        sc = measure(cv2.imread(str(files[j])), st_j, masks[j], np.zeros((h, w), dtype=np.uint8), edge_cfg, confirmed=masks[j])
        scores_before.append(dict(frame=j, **sc))
    assert any(s['判定'] == 'RESIDUE_FOUND' for s in scores_before)

    monkeypatch.setenv('INPAINT_PYTHON', 'python3')
    monkeypatch.setenv('PROPAINTER_ROOT', str(tmp_path))

    def fake_run(args, **kwargs):
        out_dir = Path(args[args.index('--output') + 1]) / 'frames/frames'
        out_dir.mkdir(parents=True, exist_ok=True)
        for q in range(3):
            cv2.imwrite(str(out_dir / f'{q:06d}.png'), grey)

    monkeypatch.setattr(rsl, 'run', fake_run)
    logs = []
    settings = HLSSettings.from_env()
    cfg = SimpleNamespace(fp16=False)
    new_scores, new_seam, info = _repair_confirmed_residue_pass(
        task, work, files, current, tmp_path, staged, tmp_path, set(), {},
        masks, ownership, box, 0, 3, 0, 3, 36, (h, w), h * w,
        edge_cfg, settings, cfg, ContextSettings(), {}, scores_before, 0.0, [],
        np.array([0, 1, 2]), {}, logs.append, measure,
        lambda p: cv2.imread(str(p), 0), lambda m, c: m, lambda *a: 1.0,
    )
    assert info is not None and info['采用补修'] is True
    assert info['补修前残留'] > 0 and info['补修后残留'] == 0
    assert rsl.residue_error(new_scores) is None


def test_end_to_end_refine_and_diagnose_synthetic(tmp_path: Path):
    """端到端合成测试 refine_v17c_masks 与 diagnose_faint 输出与不变量。"""
    base = tmp_path / 'base'
    frames_dir = tmp_path / 'frames'
    out_dir = tmp_path / 'out'
    for d in ['masks', 'group-objects/group-text', 'scene-protection', 'subtitle-protection']:
        (base / d).mkdir(parents=True)
    frames_dir.mkdir()

    n = 14
    hints = []
    subs = {}
    for i in range(n):
        im, msk, _, bx = _paint_translucent_track_frame(i)
        fname = f'{i:06d}.png'
        cv2.imwrite(str(frames_dir / fname), im)
        cv2.imwrite(str(base / 'masks' / fname), msk)
        cv2.imwrite(str(base / 'group-objects/group-text' / fname), msk)
        cv2.imwrite(str(base / 'scene-protection' / fname), np.zeros(msk.shape, dtype=np.uint8))
        subp = np.zeros(msk.shape, dtype=np.uint8)
        sub_bx = [int(bx[0] + 82), int(bx[1] + 2), int(bx[0] + 100), int(bx[1] + 20)]
        subp[sub_bx[1] : sub_bx[3], sub_bx[0] : sub_bx[2]] = 255
        cv2.imwrite(str(base / 'subtitle-protection' / fname), subp)
        subs[str(i)] = [{'text': 'AI', 'box': sub_bx}]
        hints.append({'frame': i, 'box': bx})

    det = {'scenes': [[0, n]], 'tracks': [{'id': 1, 'scene': 0, 'hints': hints}], 'subtitles': subs, 'hints': {}}
    (base / 'detections.json').write_text(json.dumps(det), encoding='utf-8')
    events = [{'group': 0, 'scene': 0, 'interval': [0, 6]}, {'group': 1, 'scene': 0, 'interval': [4, 12]}]
    (base / 'animation-events.json').write_text(
        json.dumps({'settings': {}, 'events': events, 'layers': {'0': [events[0]], '1': [events[1]]}}),
        encoding='utf-8',
    )
    write_manifest(base, base / 'detections.json', [base / 'masks'], [base / 'masks' / f'{i:06d}.png' for i in range(n)])

    f_cfg = FaintSettings(sample_frames=10, min_track_frames=8, min_unmasked_samples=4)
    e_cfg = EdgeSettings(shape_radius=1, require_ring_for_refinement=True)
    diag = diagnose_faint(base, frames_dir, probe_frames=[6], cfg=f_cfg)
    assert diag['存在未覆盖半透明候选帧数'] > 0

    prov = refine_v17c_masks(base, out_dir, edge_cfg=e_cfg, faint_cfg=f_cfg, frames_dir=frames_dir)
    assert prov['新增淡字像素'] > 0
    assert prov['重分类OCR字幕条目数'] > 0
    assert prov['全部帧满足20%保护'] is True
    assert (out_dir / 'group-objects/group-faint/000006.png').exists()
    anim_out = json.loads((out_dir / 'animation-events.json').read_text(encoding='utf-8'))
    assert anim_out['layers'] == {'0': events}
