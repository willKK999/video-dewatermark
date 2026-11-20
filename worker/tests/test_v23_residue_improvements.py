"""v23 通用降残留改进单元测试（合成数据：时域掩膜补全、低饱和外晕、孤立噪点清理、干净参考帧筛选与二次补修）。"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from worker.edge_residue import EdgeSettings, measure
from worker.hls_settings import HLSSettings
from worker.mask_continuity import (
    ContinuitySettings,
    expand_adaptive_halo,
    prune_isolated_specks,
    repair_group_temporal_dropouts,
)
from worker.restoration import restore_orthogonal_luma_detail
from worker.roi_spatial_layers import (
    _pack_bool,
    _repair_confirmed_residue_pass,
    _window_refs,
    residue_error,
)
from worker.trusted_context import ContextSettings


def _moving_bg(idx: int, h: int = 160, w: int = 200) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    base = 70 + 18 * np.sin((xx + idx * 5) / 8.0) * np.cos((yy + idx * 3) / 9.0)
    return np.repeat(np.clip(base, 20, 180).astype(np.uint8)[..., None], 3, axis=2)


def test_repair_group_temporal_dropouts_recovers_false_ocr_and_skips_vanished():
    """同镜头短时掩膜缺失（含非对话OCR误保护）应按源图外观恢复；水印真正消失帧不得误加。"""
    n = 10
    h, w = 160, 200
    frames = [_moving_bg(i, h, w) for i in range(n)]
    groups: dict[int, np.ndarray] = {}
    confs: dict[int, np.ndarray] = {}
    sub_prots: dict[int, np.ndarray] = {}
    scene_prots = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n)}

    for i in range(n):
        m = np.zeros((h, w), dtype=np.uint8)
        sp = np.zeros((h, w), dtype=np.uint8)
        if i < 8:
            # 顶部蓝色水印文字块（远离下部字幕区）
            frames[i][20:48, 30:110] = (220, 160, 25)
            if i not in (4, 5):
                m[20:48, 30:110] = 255
            else:
                # 模拟第 4-5 帧被非对话 OCR 误保护抹除
                sp[18:50, 28:112] = 255
        groups[i] = m.copy()
        confs[i] = m.copy()
        sub_prots[i] = sp

    rep = repair_group_temporal_dropouts(
        total_frames=n,
        scenes=[[0, n]],
        groups=['group-00'],
        load_bgr=lambda i: frames[i],
        load_group=lambda g, i: groups[i],
        save_group=lambda g, i, arr: groups.__setitem__(i, arr),
        load_confirmed=lambda i: confs[i],
        save_confirmed=lambda i, arr: confs.__setitem__(i, arr),
        load_scene_prot=lambda i: scene_prots[i],
        load_sub_prot=lambda i: sub_prots[i],
        save_sub_prot=lambda i, arr: sub_prots.__setitem__(i, arr),
        edge_cfg=EdgeSettings(shape_radius=1),
        ctx_cfg=ContextSettings(),
        cfg=ContinuitySettings(),
    )
    assert rep['新增像素'] > 0
    assert rep['释放字幕保护像素'] > 0
    assert np.count_nonzero(groups[4][20:48, 30:110]) >= 2000
    assert np.count_nonzero(groups[5][20:48, 30:110]) >= 2000
    assert np.count_nonzero(sub_prots[4][20:48, 30:110]) == 0
    # 水印真正消失的 8-9 帧不得误加
    assert np.count_nonzero(confs[8]) == 0
    assert np.count_nonzero(confs[9]) == 0


def test_repair_group_temporal_dropouts_respects_scene_protection_and_dialogue():
    """真实场景实物保护与底部对话字幕行在时域补全中必须绝对保留。"""
    n = 8
    h, w = 240, 320
    frames = [_moving_bg(i, h, w) for i in range(n)]
    groups = {}
    confs = {}
    sub_prots = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n)}
    scene_prots = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n)}
    for i in range(n):
        frames[i][180:200, 60:260] = (248, 248, 248)
        cv2.putText(frames[i], 'DIALOGUE LINE', (65, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        m = np.zeros((h, w), dtype=np.uint8)
        if i != 3:
            m[180:200, 60:260] = 255
        else:
            scene_prots[i][180:200, 60:140] = 255
            sub_prots[i][180:200, 140:260] = 255
        groups[i] = m.copy()
        confs[i] = m.copy()

    repair_group_temporal_dropouts(
        total_frames=n,
        scenes=[[0, n]],
        groups=['group-00'],
        load_bgr=lambda i: frames[i],
        load_group=lambda g, i: groups[i],
        save_group=lambda g, i, arr: groups.__setitem__(i, arr),
        load_confirmed=lambda i: confs[i],
        save_confirmed=lambda i, arr: confs.__setitem__(i, arr),
        load_scene_prot=lambda i: scene_prots[i],
        load_sub_prot=lambda i: sub_prots[i],
        save_sub_prot=lambda i, arr: sub_prots.__setitem__(i, arr),
        edge_cfg=EdgeSettings(shape_radius=1),
        ctx_cfg=ContextSettings(),
        cfg=ContinuitySettings(),
    )
    assert not np.any(confs[3] & scene_prots[3])
    assert not np.any(confs[3] & sub_prots[3])


def test_expand_adaptive_halo_captures_low_sat_fringe_only():
    """自适应外晕扩展应覆盖1-2px低饱和抗锯齿色晕，并避开高亮边缘与保护区。"""
    img = np.full((96, 120, 3), 40, dtype=np.uint8)
    group = np.zeros((96, 120), dtype=np.uint8)
    cv2.circle(group, (45, 45), 14, 255, -1)
    img[group > 0] = (220, 160, 25)
    # 外侧2px抗锯齿低饱和青色晕（HSV 饱和度约 40 < 60）
    ring2 = (cv2.dilate(group, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0) & (group == 0)
    img[ring2] = (145, 130, 115)
    # 旁边高亮白色衣物块应受 highlight_guard 保护
    img[35:55, 61:72] = 245

    prot = np.zeros_like(group)
    added, info = expand_adaptive_halo(img, group, group, prot, EdgeSettings(shape_radius=1), ContinuitySettings())
    assert info['新增像素'] > 0
    assert np.any(added & ring2)
    assert not np.any((added > 0) & (img.min(axis=2) > 200))


def test_prune_isolated_specks_removes_single_frame_noise_only():
    """仅删除单帧孤立微小噪点（<12px 且前后帧邻域为空），保留连续帧小笔画。"""
    h, w = 64, 64
    confs = [np.zeros((h, w), dtype=np.uint8) for _ in range(4)]
    edges = [np.zeros((h, w), dtype=np.uint8) for _ in range(4)]
    grps = [np.zeros((h, w), dtype=np.uint8) for _ in range(4)]
    # 真实跨帧小笔画（4px，出现在第1、2、3帧）
    for i in (1, 2, 3):
        confs[i][20:22, 20:22] = 255
        grps[i][20:22, 20:22] = 255
    # 单帧孤立1px噪点（仅出现在第2帧）
    confs[2][50, 50] = 255
    edges[2][50, 50] = 255
    grps[2][50, 50] = 255

    rep = prune_isolated_specks(
        total_frames=4,
        scenes=[[0, 4]],
        groups=['group-00'],
        load_confirmed=lambda i: confs[i],
        save_confirmed=lambda i, arr: confs.__setitem__(i, arr),
        load_edge=lambda i: edges[i],
        save_edge=lambda i, arr: edges.__setitem__(i, arr),
        load_group=lambda g, i: grps[i],
        save_group=lambda g, i, arr: grps.__setitem__(i, arr),
        cfg=ContinuitySettings(speck_min_area=12),
    )
    assert rep['清理像素'] == 1
    assert confs[2][50, 50] == 0 and edges[2][50, 50] == 0 and grps[2][50, 50] == 0
    assert np.all(confs[2][20:22, 20:22] == 255)


def test_window_refs_prefers_clean_roi_frames():
    """参考帧挑选应优先纳入目标ROI几乎无水印的帧。"""
    base_refs = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    # 0..59 被水印遮挡，60..99 干净
    fracs = {j: (0.45 if j < 60 else 0.0) for j in range(100)}
    picked = _window_refs(0, 100, 10, 40, base_refs, 24, roi_mask_fractions=fracs)
    clean_count = sum(1 for f in picked if fracs[int(f)] == 0.0)
    assert len(picked) <= 24
    assert clean_count >= 12


def test_repair_confirmed_residue_pass_masks_connected_pending_and_two_rounds(tmp_path: Path, monkeypatch):
    """确认残留补修应连带遮蔽同连通域内待目检色块并支持两轮补修。"""
    work = tmp_path / 'work'
    task = work / 'roi-0001-attempt-00'
    staged = task / 'staged'
    staged.mkdir(parents=True)
    (work / 'masks').mkdir(parents=True)
    (work / 'edge-band').mkdir(parents=True)
    h, w = 96, 96
    blue = np.array([220, 40, 20], dtype=np.uint8)
    grey = np.full((h, w, 3), 128, dtype=np.uint8)

    files, masks, current = [], [], []
    for j in range(3):
        p = tmp_path / f'{j:06d}.png'
        src_j = grey.copy()
        src_j[30:66, 30:66] = blue
        cv2.imwrite(str(p), src_j)
        files.append(p)
        current.append(p)
        m = np.zeros((h, w), dtype=np.uint8)
        m[30:66, 30:66] = 255
        masks.append(m)
        cv2.imwrite(str(work / 'masks' / p.name), m)
        cv2.imwrite(str(work / 'edge-band' / p.name), np.zeros((h, w), dtype=np.uint8))
        st = grey.copy()
        if j == 1:
            st[36:58, 36:58] = blue
        cv2.imwrite(str(staged / p.name), st)

    edge_cfg = EdgeSettings(shape_radius=1, residue_repair_rounds=2, residue_repair_dilate_radius=4)
    ownership = np.ones((h, w), dtype=np.uint8) * 255
    box = (0, 0, w, h)
    scores_before = []
    for j in range(3):
        st_j = cv2.imread(str(staged / files[j].name))
        sc = measure(cv2.imread(str(files[j])), st_j, masks[j], np.zeros((h, w), dtype=np.uint8), edge_cfg, confirmed=masks[j], return_masks=True)
        rm = sc.pop('_masks')
        entry = dict(frame=j, **sc)
        if sc['判定'] == 'RESIDUE_FOUND':
            entry['_repair_residue'] = _pack_bool(rm['确认残留'])
        if np.any(rm['待目检']):
            entry['_repair_pending'] = _pack_bool(rm['待目检'])
        scores_before.append(entry)

    monkeypatch.setenv('INPAINT_PYTHON', 'python3')
    monkeypatch.setenv('PROPAINTER_ROOT', str(tmp_path))
    rounds_called = []

    def fake_run(args, **kwargs):
        rounds_called.append(1)
        out_dir = Path(args[args.index('--output') + 1]) / 'frames/frames'
        out_dir.mkdir(parents=True, exist_ok=True)
        for q in range(3):
            out_img = grey.copy()
            # 第一轮故意留下小块残留，验证第二轮继续清理
            if len(rounds_called) == 1 and q == 1:
                out_img[32:38, 32:44] = blue
            cv2.imwrite(str(out_dir / f'{q:06d}.png'), out_img)

    from worker import roi_spatial_layers as rsl
    monkeypatch.setattr(rsl, 'run', fake_run)
    new_scores, _, info = _repair_confirmed_residue_pass(
        task, work, files, current, tmp_path, staged, tmp_path, set(), {},
        masks, ownership, box, 0, 3, 0, 3, 36, (h, w), h * w,
        edge_cfg, HLSSettings.from_env(), SimpleNamespace(fp16=False), ContextSettings(), {}, scores_before, 0.0, [],
        np.array([0, 1, 2]), {}, lambda *a: None, measure,
        lambda p: cv2.imread(str(p), 0), lambda m, c: m, lambda *a: 1.0,
    )
    assert len(rounds_called) == 2
    assert info is not None and info['采用补修'] is True and info['补修轮数'] == 2
    assert info['补修后残留'] == 0
    assert residue_error(new_scores) is None


def test_restore_orthogonal_luma_detail_blocks_coherent_faint_watermark():
    """正交亮度投影不得将无色低对比度半透明水印鬼影重新注入模型输出。"""
    h, w = 128, 128
    bg = np.full((h, w, 3), 90, dtype=np.uint8)
    src = bg.copy()
    # 低对比度（+12灰度）一致半透明淡字条带，低于16且无动态高通掩蔽
    src[52:68, 40:88] = 102
    msk = np.zeros((h, w), dtype=np.uint8)
    msk[20:108, 20:108] = 255
    repaired = bg.copy()
    wm_hp = np.zeros((h, w), dtype=np.float32)

    out = restore_orthogonal_luma_detail(src, repaired, msk, wm_hp, EdgeSettings())
    assert float(out[54:66, 42:86].mean()) < 93.0


def test_repair_group_temporal_dropouts_recovers_partial_multiframe_gap():
    """同组仅部分区域连续多帧漏检（未漏部分使双侧交集非空）仍应从 >=min_boundary_votes 恢复，且不误加已消失子块。"""
    n = 10
    h, w = 160, 220
    frames = [_moving_bg(i, h, w) for i in range(n)]
    groups: dict[int, np.ndarray] = {}
    confs: dict[int, np.ndarray] = {}
    sub_prots = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n)}
    scene_prots = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n)}

    for i in range(n):
        m = np.zeros((h, w), dtype=np.uint8)
        # 左侧主块全程存在且被掩膜覆盖
        frames[i][20:48, 20:80] = (220, 160, 25)
        m[20:48, 20:80] = 255
        # 右侧子块在前8帧存在，但第2-4帧连续漏检（第8-9帧真正消失）
        if i < 8:
            frames[i][20:48, 95:165] = (220, 160, 25)
            if i not in (2, 3, 4):
                m[20:48, 95:165] = 255
        groups[i] = m.copy()
        confs[i] = m.copy()

    rep = repair_group_temporal_dropouts(
        total_frames=n,
        scenes=[[0, n]],
        groups=['group-00'],
        load_bgr=lambda i: frames[i],
        load_group=lambda g, i: groups[i],
        save_group=lambda g, i, arr: groups.__setitem__(i, arr),
        load_confirmed=lambda i: confs[i],
        save_confirmed=lambda i, arr: confs.__setitem__(i, arr),
        load_scene_prot=lambda i: scene_prots[i],
        load_sub_prot=lambda i: sub_prots[i],
        save_sub_prot=lambda i, arr: sub_prots.__setitem__(i, arr),
        edge_cfg=EdgeSettings(shape_radius=1),
        ctx_cfg=ContextSettings(),
        cfg=ContinuitySettings(),
    )
    assert rep['新增像素'] > 0
    for i in (2, 3, 4):
        assert np.count_nonzero(confs[i][20:48, 95:165]) >= 1500
    # 真正消失的第8-9帧右侧子块不得被误恢复
    assert np.count_nonzero(confs[8][20:48, 95:165]) == 0
    assert np.count_nonzero(confs[9][20:48, 95:165]) == 0


def test_repair_confirmed_residue_pass_does_not_reblend_dirty_completed(tmp_path: Path, monkeypatch):
    """前序未过任务遗留确认残留的已完成帧（dirty_frames）不得在补修回写时重新混入旧残留。"""
    work = tmp_path / 'work'
    task = work / 'roi-0001-attempt-00'
    staged = task / 'staged'
    dest = work / 'dest'
    staged.mkdir(parents=True)
    dest.mkdir(parents=True)
    (work / 'masks').mkdir(parents=True)
    (work / 'edge-band').mkdir(parents=True)
    h, w = 96, 96
    blue = np.array([220, 40, 20], dtype=np.uint8)
    grey = np.full((h, w, 3), 128, dtype=np.uint8)

    files, masks, current = [], [], []
    layer_done = {}
    for j in range(3):
        p = tmp_path / f'{j:06d}.png'
        src_j = grey.copy()
        src_j[30:66, 30:66] = blue
        cv2.imwrite(str(p), src_j)
        files.append(p)
        current.append(p)
        m = np.zeros((h, w), dtype=np.uint8)
        m[30:66, 30:66] = 255
        masks.append(m)
        layer_done[j] = m.copy()
        cv2.imwrite(str(work / 'masks' / p.name), m)
        cv2.imwrite(str(work / 'edge-band' / p.name), np.zeros((h, w), dtype=np.uint8))
        # dest (base_old) 在帧0有旧确认残留
        old_j = grey.copy()
        if j == 0:
            old_j[40:54, 40:54] = blue
        cv2.imwrite(str(dest / p.name), old_j)
        cv2.imwrite(str(staged / p.name), old_j)

    edge_cfg = EdgeSettings(shape_radius=1, residue_repair_rounds=1, residue_repair_dilate_radius=4)
    ownership = np.ones((h, w), dtype=np.uint8) * 255
    scores_before = []
    for j in range(3):
        st_j = cv2.imread(str(staged / files[j].name))
        sc = measure(cv2.imread(str(files[j])), st_j, masks[j], np.zeros((h, w), dtype=np.uint8), edge_cfg, confirmed=masks[j], return_masks=True)
        rm = sc.pop('_masks')
        entry = dict(frame=j, **sc)
        if sc['判定'] == 'RESIDUE_FOUND':
            entry['_repair_residue'] = _pack_bool(rm['确认残留'])
        scores_before.append(entry)

    monkeypatch.setenv('INPAINT_PYTHON', 'python3')
    monkeypatch.setenv('PROPAINTER_ROOT', str(tmp_path))

    def fake_run(args, **kwargs):
        out_dir = Path(args[args.index('--output') + 1]) / 'frames/frames'
        out_dir.mkdir(parents=True, exist_ok=True)
        for q in range(3):
            cv2.imwrite(str(out_dir / f'{q:06d}.png'), grey.copy())

    from worker import roi_spatial_layers as rsl
    monkeypatch.setattr(rsl, 'run', fake_run)
    # 若不传 dirty_frames，completed={0,1,2} 对帧0权重仅为 1/4，会把 75% 的旧蓝色残留混回，导致补修失败
    new_scores, _, info = _repair_confirmed_residue_pass(
        task, work, files, current, dest, staged, tmp_path, {0, 1, 2}, layer_done,
        masks, ownership, (0, 0, w, h), 0, 3, 0, 3, 36, (h, w), h * w,
        edge_cfg, HLSSettings.from_env(), SimpleNamespace(fp16=False), ContextSettings(), {}, scores_before, 0.0, [],
        np.array([0, 1, 2]), {}, lambda *a: None, measure,
        lambda p: cv2.imread(str(p), 0), lambda m, c: m, lambda j, s, e, comp: 0.25,
        dirty_frames={0},
    )
    assert info is not None and info['采用补修'] is True
    assert info['补修后残留'] == 0
    assert residue_error(new_scores) is None


def test_repair_group_temporal_dropouts_does_not_leak_before_appearance_on_textured_bg():
    """在静止/缓动强纹理背景上，中途才出现的新水印不得向出现前的干净帧倒灌。"""
    n = 10
    h, w = 180, 220
    yy, xx = np.mgrid[0:h, 0:w]
    tex = 175 + 22 * np.sin(xx / 4.0) * np.cos(yy / 5.0)
    base_bg = np.repeat(np.clip(tex, 40, 230).astype(np.uint8)[..., None], 3, axis=2)
    frames = [base_bg.copy() for _ in range(n)]
    groups: dict[int, np.ndarray] = {}
    confs: dict[int, np.ndarray] = {}
    sub_prots = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n)}
    scene_prots = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n)}

    for i in range(n):
        m = np.zeros((h, w), dtype=np.uint8)
        # 左上角小标全程存在（使该组在干净帧上也有部分像素）
        frames[i][10:25, 10:35] = (210, 140, 30)
        m[10:25, 10:35] = 255
        # 中下部半透明灰水印从第5帧才开始出现
        if i >= 5:
            cv2.putText(frames[i], 'LOGO', (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 3)
            m[80:140, 50:170] = 255
        groups[i] = m.copy()
        confs[i] = m.copy()

    repair_group_temporal_dropouts(
        total_frames=n,
        scenes=[[0, n]],
        groups=['group-00'],
        load_bgr=lambda i: frames[i],
        load_group=lambda g, i: groups[i],
        save_group=lambda g, i, arr: groups.__setitem__(i, arr),
        load_confirmed=lambda i: confs[i],
        save_confirmed=lambda i, arr: confs.__setitem__(i, arr),
        load_scene_prot=lambda i: scene_prots[i],
        load_sub_prot=lambda i: sub_prots[i],
        save_sub_prot=lambda i, arr: sub_prots.__setitem__(i, arr),
        edge_cfg=EdgeSettings(shape_radius=1),
        ctx_cfg=ContextSettings(),
        cfg=ContinuitySettings(),
    )
    for i in range(5):
        assert np.count_nonzero(confs[i][80:140, 50:170]) == 0


def test_repair_group_temporal_dropouts_does_not_smear_moving_watermark():
    """正常移动中的水印完整帧不得把前后邻帧位移轨迹拖影进当前掩膜；单帧漏检应先对齐再恢复。"""
    n = 10
    h, w = 160, 220
    frames = [_moving_bg(i, h, w) for i in range(n)]
    groups: dict[int, np.ndarray] = {}
    confs: dict[int, np.ndarray] = {}
    sub_prots = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n)}
    scene_prots = {i: np.zeros((h, w), dtype=np.uint8) for i in range(n)}

    for i in range(n):
        x0 = 20 + i * 6
        frames[i][50:80, x0:x0 + 40] = (220, 150, 25)
        m = np.zeros((h, w), dtype=np.uint8)
        if i != 5:
            m[50:80, x0:x0 + 40] = 255
        groups[i] = m.copy()
        confs[i] = m.copy()

    repair_group_temporal_dropouts(
        total_frames=n,
        scenes=[[0, n]],
        groups=['group-00'],
        load_bgr=lambda i: frames[i],
        load_group=lambda g, i: groups[i],
        save_group=lambda g, i, arr: groups.__setitem__(i, arr),
        load_confirmed=lambda i: confs[i],
        save_confirmed=lambda i, arr: confs.__setitem__(i, arr),
        load_scene_prot=lambda i: scene_prots[i],
        load_sub_prot=lambda i: sub_prots[i],
        save_sub_prot=lambda i, arr: sub_prots.__setitem__(i, arr),
        edge_cfg=EdgeSettings(shape_radius=1),
        ctx_cfg=ContextSettings(),
        cfg=ContinuitySettings(),
    )
    # 第4帧原本完整，不得向前后位移拖影扩出多余像素
    assert np.count_nonzero(confs[4]) == 30 * 40
    # 第5帧漏检，应精准对齐到 x0=50..90 处恢复且不超出太多
    x5 = 20 + 5 * 6
    assert np.count_nonzero(confs[5][50:80, x5:x5 + 40]) >= 900
    assert np.count_nonzero(confs[5]) <= 30 * 40 + 100
