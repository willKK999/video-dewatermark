"""通用掩膜时域连续性修复、低饱和抗锯齿外晕扩展与单帧孤立微噪点清理。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .edge_residue import (
    EdgeSettings,
    _hue_distance,
    _is_distinguishing_color,
    color_peaks,
    highlight_guard,
)
from .faint_watermark import dialogue_row_mask, signed_highpass
from .process import PipelineError
from .trusted_context import ContextSettings, exclusion_mask
from .v12_masks import Settings as V12Settings


@dataclass(frozen=True)
class ContinuitySettings:
    """通用时域掩膜补全与自适应收边配置。"""

    temporal_radius: int = 6
    min_neighbor_votes: int = 2
    min_boundary_votes: int = 3
    min_candidate_pixels: int = 16
    min_ncc: float = 0.30
    min_ocr_release_ncc: float = 0.40
    min_color_match_fraction: float = 0.15
    min_hp_match_fraction: float = 0.15
    min_direct_color_fraction: float = 0.35
    min_direct_color_pixels: int = 40
    halo_radius: int = 2
    halo_min_saturation: int = 28
    halo_min_value: int = 40
    halo_hue_padding: int = 3
    halo_max_outer_fraction: float = 0.25
    speck_min_area: int = 12
    speck_neighbor_radius: int = 4

    def __post_init__(self) -> None:
        if self.temporal_radius < 1 or self.min_neighbor_votes < 2:
            raise ValueError('时域连续性邻域参数无效')
        if not 0 <= self.halo_radius <= 3:
            raise ValueError('自适应外晕半径必须在[0, 3]之间')
        if self.speck_min_area < 1:
            raise ValueError('孤立微噪点面积下限必须为正')


def _distinguishing_peaks(
    image: np.ndarray,
    group_bool: np.ndarray,
    protected_bool: np.ndarray,
    edge_cfg: EdgeSettings,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """提取确认组相对干净外环的显著区别色峰。"""
    inner = group_bool & (~protected_bool)
    if int(np.count_nonzero(inner)) < edge_cfg.minimum_template_pixels:
        return [], np.zeros(group_bool.shape, dtype=bool), np.zeros((0,), dtype=bool)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    peaks = color_peaks(image, inner, edge_cfg, hsv=hsv)
    if not peaks:
        return [], np.zeros(group_bool.shape, dtype=bool), np.zeros((0,), dtype=bool)
    rad = edge_cfg.template_reference_radius
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    outer = (cv2.dilate(group_bool.astype(np.uint8), ker) > 0) & (~group_bool) & (~protected_bool)
    outer_n = int(np.count_nonzero(outer))
    if outer_n <= 0:
        return [], np.zeros(group_bool.shape, dtype=bool), outer
    dist: list[int] = []
    for pk in peaks:
        cm = (
            (_hue_distance(hsv[:, :, 0], int(pk)) <= edge_cfg.hue_radius)
            & (hsv[:, :, 1] >= edge_cfg.minimum_saturation)
            & (hsv[:, :, 2] >= edge_cfg.minimum_value)
        )
        inside = float(cm[inner].mean())
        outside = float(cm[outer].mean())
        if _is_distinguishing_color(inside, outside, outer_n, edge_cfg):
            dist.append(int(pk))
    return dist, hsv, outer


def _relaxed_hue_match(
    hsv: np.ndarray,
    peaks: list[int],
    hue_radius: int,
    min_sat: int,
    min_val: int,
) -> np.ndarray:
    """在抗锯齿低饱和阈值下匹配指定色相峰集合。"""
    if not peaks:
        return np.zeros(hsv.shape[:2], dtype=bool)
    sat_ok = (hsv[:, :, 1] >= min_sat) & (hsv[:, :, 2] >= min_val)
    out = np.zeros(hsv.shape[:2], dtype=bool)
    h_plane = hsv[:, :, 0]
    for pk in peaks:
        out |= (_hue_distance(h_plane, int(pk)) <= hue_radius) & sat_ok
    return out


def _local_highpass_ncc(
    hp_cur: np.ndarray,
    hp_ref: np.ndarray,
    roi_mask: np.ndarray,
    search_pad: int = 6,
) -> tuple[float, int, int]:
    """在局部外接框内用高斯高通做有界平移归一化互相关，验证结构一致性。"""
    ys, xs = np.where(roi_mask)
    if not len(xs):
        return -1.0, 0, 0
    h, w = hp_cur.shape[:2]
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    if x1 - x0 < 12 or y1 - y0 < 12:
        return -1.0, 0, 0
    tmpl = hp_ref[y0:y1, x0:x1].astype(np.float32)
    if float(np.std(tmpl)) < 1e-3:
        return -1.0, 0, 0
    sx0, sy0 = max(0, x0 - search_pad), max(0, y0 - search_pad)
    sx1, sy1 = min(w, x1 + search_pad), min(h, y1 + search_pad)
    patch = hp_cur[sy0:sy1, sx0:sx1].astype(np.float32)
    if patch.shape[0] < tmpl.shape[0] or patch.shape[1] < tmpl.shape[1]:
        return -1.0, 0, 0
    res = cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if not np.isfinite(max_val):
        return -1.0, 0, 0
    dx = int(sx0 + max_loc[0] - x0)
    dy = int(sy0 + max_loc[1] - y0)
    return float(max_val), dx, dy


def _masked_cand_presence(
    hp_cur: np.ndarray,
    hp_ref: np.ndarray,
    cand_mask: np.ndarray,
    ref_mask: np.ndarray,
    dx: int = 0,
    dy: int = 0,
) -> tuple[float, float, float]:
    """仅在候选掩膜像素及其外侧干净环上计算高通相关与能量比，区分真实漏检与尚未出现/已消失水印。"""
    h, w = hp_cur.shape[:2]
    ys, xs = np.where(cand_mask)
    if len(xs) < 16:
        return -1.0, 0.0, 1.0
    ry = ys - int(dy)
    rx = xs - int(dx)
    ok = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
    if int(np.count_nonzero(ok)) < 16:
        return -1.0, 0.0, 1.0
    c_vals = hp_cur[ys[ok], xs[ok]].astype(np.float32)
    r_vals = hp_ref[ry[ok], rx[ok]].astype(np.float32)
    std_c = float(np.std(c_vals))
    std_r = float(np.std(r_vals))
    if std_r < 1e-3:
        return -1.0, 0.0, 1.0
    vc = c_vals - float(np.mean(c_vals))
    vr = r_vals - float(np.mean(r_vals))
    masked_ncc = float(np.sum(vc * vr) / (float(np.sqrt(np.sum(vc * vc) * np.sum(vr * vr))) + 1e-6))
    std_ratio = std_c / (std_r + 1e-6)

    ring_u8 = cv2.dilate(cand_mask.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    ring = ring_u8 & (~ref_mask) & (~cand_mask)
    ry_r, rx_r = np.where(ring)
    ry_r2 = ry_r - int(dy)
    rx_r2 = rx_r - int(dx)
    ok_r = (ry_r2 >= 0) & (ry_r2 < h) & (rx_r2 >= 0) & (rx_r2 < w)
    ring_ncc = 0.0
    if int(np.count_nonzero(ok_r)) >= 64:
        rc = hp_cur[ry_r[ok_r], rx_r[ok_r]].astype(np.float32)
        rr = hp_ref[ry_r2[ok_r], rx_r2[ok_r]].astype(np.float32)
        rc0 = rc - float(np.mean(rc))
        rr0 = rr - float(np.mean(rr))
        denom_r = float(np.sqrt(np.sum(rc0 * rc0) * np.sum(rr0 * rr0)))
        if denom_r > 1e-3:
            ring_ncc = float(np.sum(rc0 * rr0) / denom_r)
    return masked_ncc, ring_ncc, std_ratio


def repair_group_temporal_dropouts(
    total_frames: int,
    scenes: list[list[int]] | list[tuple[int, int]],
    groups: list[str],
    load_bgr: Callable[[int], np.ndarray],
    load_group: Callable[[str, int], np.ndarray],
    save_group: Callable[[str, int, np.ndarray], None],
    load_confirmed: Callable[[int], np.ndarray],
    save_confirmed: Callable[[int, np.ndarray], None],
    load_scene_prot: Callable[[int], np.ndarray],
    load_sub_prot: Callable[[int], np.ndarray],
    save_sub_prot: Callable[[int, np.ndarray], None],
    edge_cfg: EdgeSettings | None = None,
    ctx_cfg: ContextSettings | None = None,
    cfg: ContinuitySettings | None = None,
    v12_cfg: V12Settings | None = None,
) -> dict:
    """按同镜头邻帧掩膜共识与源图外观/结构验证，修复单帧或短窗口掩膜漏检与OCR误删。"""
    e_cfg = edge_cfg or EdgeSettings()
    c_cfg = ctx_cfg or ContextSettings()
    cont = cfg or ContinuitySettings()
    sub_cfg = v12_cfg or V12Settings()

    if not groups or total_frames <= 1:
        return dict(新增像素=0, 释放字幕保护像素=0, 修复组帧记录数=0)

    sample = load_bgr(0)
    shape = sample.shape[:2]
    pixels = shape[0] * shape[1]
    hp_cache: dict[int, np.ndarray] = {}
    bgr_cache: dict[int, np.ndarray] = {}
    dlg_cache: dict[int, np.ndarray] = {}

    def get_bgr(idx: int) -> np.ndarray:
        if idx not in bgr_cache:
            if len(bgr_cache) >= 48:
                bgr_cache.pop(next(iter(bgr_cache)))
            bgr_cache[idx] = load_bgr(idx)
        return bgr_cache[idx]

    def get_hp(idx: int) -> np.ndarray:
        if idx not in hp_cache:
            if len(hp_cache) >= 48:
                hp_cache.pop(next(iter(hp_cache)))
            gray = cv2.cvtColor(get_bgr(idx), cv2.COLOR_BGR2GRAY)
            hp_cache[idx] = signed_highpass(gray, 4.0)
        return hp_cache[idx]

    def get_dialogue(idx: int) -> np.ndarray:
        if idx not in dlg_cache:
            if len(dlg_cache) >= 48:
                dlg_cache.pop(next(iter(dlg_cache)))
            dlg_cache[idx] = dialogue_row_mask(get_bgr(idx), sub_cfg)
        return dlg_cache[idx]

    added_total = 0
    released_sub_total = 0
    records: list[dict] = []
    ker5 = np.ones((5, 5), dtype=np.uint8)
    ker3 = np.ones((3, 3), dtype=np.uint8)

    for sc in (scenes or [[0, total_frames]]):
        s0, s1 = max(0, int(sc[0])), min(total_frames, int(sc[1]))
        if s1 - s0 < 2:
            continue
        for g in groups:
            if g == 'group-faint':
                continue
            # 滑动加载本组在本镜头内的掩膜
            g_masks: dict[int, np.ndarray] = {}
            for idx in range(s0, s1):
                gm = load_group(g, idx) > 0
                if np.any(gm):
                    g_masks[idx] = gm
            if len(g_masks) < cont.min_neighbor_votes:
                continue

            raw_masks = dict(g_masks)
            for _pass in range(2):
                orig_masks = dict(g_masks)
                pass_added = 0
                for idx in range(s0, s1):
                    cur_gm = orig_masks.get(idx)
                    cur_area = int(np.count_nonzero(cur_gm)) if cur_gm is not None else 0
                    prev_nbrs = [j for j in range(max(s0, idx - cont.temporal_radius), idx) if j in orig_masks]
                    next_nbrs = [j for j in range(idx + 1, min(s1, idx + cont.temporal_radius + 1)) if j in orig_masks]
                    all_nbrs = prev_nbrs + next_nbrs
                    if len(all_nbrs) < cont.min_neighbor_votes:
                        continue
                    nbr_areas = [int(np.count_nonzero(orig_masks[j])) for j in all_nbrs]
                    med_area = float(np.median(nbr_areas))
                    if cur_area >= 0.85 * med_area:
                        continue

                    # 检查邻帧同组是否处于移动状态：若静止则直接投票；若移动则先按高通NCC对齐近邻再投票，避免拖影扩散。
                    full_nbrs = [j for j in all_nbrs if int(np.count_nonzero(orig_masks[j])) >= 0.75 * med_area] or all_nbrs
                    is_moving = False
                    if len(full_nbrs) >= 2:
                        m_first = orig_masks[full_nbrs[0]]
                        m_last = orig_masks[full_nbrs[-1]]
                        inter_n = int(np.count_nonzero(m_first & m_last))
                        union_n = int(np.count_nonzero(m_first | m_last))
                        if union_n > 0 and (inter_n / float(union_n)) < 0.80:
                            is_moving = True

                    hp_cur = get_hp(idx) if is_moving else None
                    if is_moving:
                        near_nbrs = sorted(full_nbrs, key=lambda j: abs(j - idx))[:4]
                        aligned_prev: list[np.ndarray] = []
                        aligned_next: list[np.ndarray] = []
                        votes = np.zeros(shape, dtype=np.uint8)
                        for j in near_nbrs:
                            ncc_j, dx_j, dy_j = _local_highpass_ncc(hp_cur, get_hp(j), orig_masks[j])
                            if ncc_j < cont.min_ncc:
                                continue
                            if abs(dx_j) > 0 or abs(dy_j) > 0:
                                mat_j = np.array([[1.0, 0.0, float(dx_j)], [0.0, 1.0, float(dy_j)]], dtype=np.float32)
                                wm_j = (
                                    cv2.warpAffine(
                                        orig_masks[j].astype(np.uint8), mat_j, (shape[1], shape[0]), flags=cv2.INTER_NEAREST
                                    )
                                    > 0
                                )
                            else:
                                wm_j = orig_masks[j]
                            votes += wm_j.astype(np.uint8)
                            if j < idx:
                                aligned_prev.append(wm_j)
                            else:
                                aligned_next.append(wm_j)
                        has_both = bool(aligned_prev) and bool(aligned_next)
                        both_support = np.zeros(shape, dtype=bool)
                        if has_both:
                            p_any = np.bitwise_or.reduce(aligned_prev)
                            n_any = np.bitwise_or.reduce(aligned_next)
                            both_support = (votes >= cont.min_neighbor_votes) & p_any & n_any
                            consensus = both_support | (votes >= cont.min_boundary_votes)
                        else:
                            consensus = votes >= cont.min_boundary_votes
                    else:
                        # 统计邻帧投票；同时支持双侧邻帧共识与同镜头 >= min_boundary_votes 稳定共识（解决部分区域连续多帧漏检）。
                        votes = np.zeros(shape, dtype=np.uint8)
                        for j in all_nbrs:
                            votes += orig_masks[j].astype(np.uint8)
                        has_both = bool(prev_nbrs) and bool(next_nbrs)
                        both_support = np.zeros(shape, dtype=bool)
                        if has_both:
                            prev_any = np.zeros(shape, dtype=bool)
                            for j in prev_nbrs:
                                prev_any |= orig_masks[j]
                            next_any = np.zeros(shape, dtype=bool)
                            for j in next_nbrs:
                                next_any |= orig_masks[j]
                            both_support = (votes >= cont.min_neighbor_votes) & prev_any & next_any
                            consensus = both_support | (votes >= cont.min_boundary_votes)
                        else:
                            consensus = votes >= cont.min_boundary_votes
                    if not np.any(consensus):
                        continue

                    cur_bool = cur_gm if cur_gm is not None else np.zeros(shape, dtype=bool)
                    scene_prot = load_scene_prot(idx) > 0
                    sub_prot = load_sub_prot(idx) > 0
                    dlg_mask = get_dialogue(idx)
                    raw_missing = consensus & (~cur_bool) & (~scene_prot) & (~dlg_mask)
                    if int(np.count_nonzero(raw_missing)) < cont.min_candidate_pixels:
                        continue

                    # 参考帧必须来自原始已检测帧（避免用上一轮刚补出来的帧充当参考导致空白背景链式传播）
                    raw_cands = [j for j in all_nbrs if j in raw_masks]
                    if not raw_cands:
                        continue
                    cons_overlaps = {j: int(np.count_nonzero(raw_masks[j] & consensus)) for j in raw_cands}
                    max_ov = max(cons_overlaps.values())
                    if max_ov < cont.min_candidate_pixels:
                        continue
                    ref_idx = min(
                        [j for j in raw_cands if cons_overlaps[j] >= 0.75 * max_ov] or raw_cands,
                        key=lambda j: (abs(j - idx), -cons_overlaps[j]),
                    )
                    ref_gm = raw_masks[ref_idx]
                    ref_bgr = get_bgr(ref_idx)
                    ref_prot = (load_scene_prot(ref_idx) > 0) | (load_sub_prot(ref_idx) > 0)
                    dist_peaks, _, _ = _distinguishing_peaks(ref_bgr, ref_gm, ref_prot, e_cfg)

                    # 在源帧上核验结构 NCC 与颜色/高通证据
                    cur_bgr = get_bgr(idx)
                    hp_cur = get_hp(idx)
                    hp_ref = get_hp(ref_idx)
                    ncc, dx, dy = _local_highpass_ncc(hp_cur, hp_ref, consensus)

                    cand = raw_missing
                    if abs(dx) > 0 or abs(dy) > 0:
                        mat = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
                        shifted_cons = (
                            cv2.warpAffine(consensus.astype(np.uint8), mat, (shape[1], shape[0]), flags=cv2.INTER_NEAREST)
                            > 0
                        )
                        shifted_missing = shifted_cons & (~cur_bool) & (~scene_prot) & (~dlg_mask)
                        if int(np.count_nonzero(shifted_missing)) >= cont.min_candidate_pixels:
                            cand = shifted_missing

                    # 仅在候选缺失像素上核验局部高通相关与外环比较，防止单侧新出现水印在纹理背景上误扩散
                    masked_ncc, ring_ncc, std_ratio = _masked_cand_presence(hp_cur, hp_ref, cand, ref_gm, dx, dy)
                    eff_ncc = masked_ncc if masked_ncc >= 0.0 else ncc

                    # 只保留受当前帧确认与/或非对话OCR释放约束的候选像素
                    overlaps_sub = bool(np.any(cand & sub_prot))
                    if overlaps_sub:
                        # 仅在参考帧未被实物保护且当前帧结构一致度高时才释放非对话OCR保护
                        ref_scene_prot = load_scene_prot(ref_idx) > 0
                        if np.mean(ref_scene_prot[ref_gm]) > 0.2 or eff_ncc < cont.min_ocr_release_ncc:
                            cand = cand & (~sub_prot)
                    if int(np.count_nonzero(cand)) < cont.min_candidate_pixels:
                        continue

                    cur_hsv = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2HSV)
                    color_hit = _relaxed_hue_match(
                        cur_hsv,
                        dist_peaks,
                        e_cfg.hue_radius + 2,
                        cont.halo_min_saturation,
                        cont.halo_min_value,
                    )
                    color_dil = cv2.dilate(color_hit.astype(np.uint8), ker5) > 0
                    cand_px = int(np.count_nonzero(cand))
                    color_match_px = int(np.count_nonzero(cand & color_dil))
                    color_frac = color_match_px / float(cand_px)
                    hp_match = (np.abs(hp_cur) >= 4.0) & (hp_cur * hp_ref > 0)
                    hp_dil = cv2.dilate(hp_match.astype(np.uint8), ker3) > 0
                    hp_frac = float(np.count_nonzero(cand & hp_dil)) / float(cand_px)

                    cand_has_both = bool(np.any(cand & both_support))
                    min_req_ncc = cont.min_ncc if cand_has_both else max(cont.min_ncc, 0.45)
                    bg_only_trap = ring_ncc >= 0.55 and masked_ncc < ring_ncc - 0.18 and std_ratio < 0.75
                    color_ok = (not dist_peaks) or (cand_px < 200) or (color_frac >= 0.05)

                    verified = (
                        (
                            not bg_only_trap
                            and color_ok
                            and eff_ncc >= min_req_ncc
                            and (color_frac >= cont.min_color_match_fraction or hp_frac >= cont.min_hp_match_fraction)
                        )
                        or (color_frac >= cont.min_direct_color_fraction and color_match_px >= cont.min_direct_color_pixels)
                    )
                    if not verified:
                        continue

                    cur_conf = load_confirmed(idx) > 0
                    add_mask = cand & (~cur_conf)
                    if not np.any(add_mask):
                        continue
                    merged_u8 = (cur_conf | add_mask).astype(np.uint8) * 255
                    if float(np.count_nonzero(exclusion_mask(merged_u8, c_cfg))) / float(pixels) > c_cfg.mask_fraction_limit:
                        continue

                    # 写出更新后的组掩膜、联合掩膜及（必要时）释放的非对话字幕保护
                    new_gm = cur_bool | add_mask
                    save_group(g, idx, new_gm.astype(np.uint8) * 255)
                    g_masks[idx] = new_gm
                    save_confirmed(idx, merged_u8)
                    rel_px = int(np.count_nonzero(sub_prot & add_mask))
                    if rel_px > 0:
                        new_sub = (sub_prot & (~add_mask)).astype(np.uint8) * 255
                        save_sub_prot(idx, new_sub)
                        released_sub_total += rel_px

                    add_n = int(np.count_nonzero(add_mask))
                    added_total += add_n
                    pass_added += add_n
                    records.append(
                        dict(
                            frame=int(idx),
                            group=g,
                            added=add_n,
                            released_sub=rel_px,
                            ncc=round(eff_ncc, 4),
                            color_frac=round(color_frac, 4),
                            hp_frac=round(hp_frac, 4),
                        )
                    )
                if pass_added == 0:
                    break

    return dict(新增像素=added_total, 释放字幕保护像素=released_sub_total, 修复组帧记录数=len(records), 记录=records[:200])


def expand_adaptive_halo(
    image: np.ndarray,
    group_mask: np.ndarray,
    confirmed_mask: np.ndarray,
    protected_mask: np.ndarray,
    edge_cfg: EdgeSettings,
    cont_cfg: ContinuitySettings | None = None,
) -> tuple[np.ndarray, dict]:
    """在确认水印组周围按区别色抗锯齿低饱和阈值扩展窄外晕，严守保护区与高亮物理边界。"""
    cont = cont_cfg or ContinuitySettings()
    if cont.halo_radius <= 0 or not np.any(group_mask):
        return np.zeros_like(confirmed_mask, dtype=np.uint8), dict(hues=[], 新增像素=0)

    group_b = group_mask > 0
    prot_b = protected_mask > 0
    dist_peaks, hsv, outer = _distinguishing_peaks(image, group_b, prot_b, edge_cfg)
    if not dist_peaks:
        return np.zeros_like(confirmed_mask, dtype=np.uint8), dict(hues=[], 新增像素=0)

    # 排除外环背景自身已高密度包含的色相，避免向纯同色背景无界扩散
    kept_peaks: list[int] = []
    outer_n = int(np.count_nonzero(outer))
    for pk in dist_peaks:
        cm_rel = _relaxed_hue_match(
            hsv, [pk], edge_cfg.hue_radius + cont.halo_hue_padding, cont.halo_min_saturation, cont.halo_min_value
        )
        outer_frac = float(cm_rel[outer].mean()) if outer_n > 0 else 1.0
        if outer_frac <= cont.halo_max_outer_fraction:
            kept_peaks.append(pk)
    if not kept_peaks:
        return np.zeros_like(confirmed_mask, dtype=np.uint8), dict(hues=[], 新增像素=0)

    r = cont.halo_radius
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    near = cv2.dilate(group_b.astype(np.uint8), ker) > 0
    guard = prot_b | highlight_guard(image)
    matched = _relaxed_hue_match(
        hsv,
        kept_peaks,
        edge_cfg.hue_radius + cont.halo_hue_padding,
        cont.halo_min_saturation,
        cont.halo_min_value,
    )
    cand = near & (confirmed_mask == 0) & (~guard) & matched
    if not np.any(cand):
        return np.zeros_like(confirmed_mask, dtype=np.uint8), dict(hues=kept_peaks, 新增像素=0)

    # 仅保留与确认组保持8连通的路径，不吸纳外环孤立背景噪点
    region = group_b | cand
    num, labels = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    touch_labels = set(np.unique(labels[group_b])) - {0}
    connected = np.isin(labels, list(touch_labels)) & cand
    if np.any(connected & guard):
        raise PipelineError('MASK_UNCERTAIN', '自适应外晕扩展侵入保护区')
    return connected.astype(np.uint8) * 255, dict(hues=kept_peaks, 新增像素=int(np.count_nonzero(connected)))


def prune_isolated_specks(
    total_frames: int,
    scenes: list[list[int]] | list[tuple[int, int]],
    groups: list[str],
    load_confirmed: Callable[[int], np.ndarray],
    save_confirmed: Callable[[int, np.ndarray], None],
    load_edge: Callable[[int], np.ndarray],
    save_edge: Callable[[int, np.ndarray], None],
    load_group: Callable[[str, int], np.ndarray],
    save_group: Callable[[str, int, np.ndarray], None],
    cfg: ContinuitySettings | None = None,
) -> dict:
    """清理单帧孤立微小掩膜噪点（< speck_min_area 且在相邻帧邻域内完全不存在）。"""
    cont = cfg or ContinuitySettings()
    if total_frames <= 1 or cont.speck_min_area <= 1:
        return dict(清理像素=0, 清理连通域=0, 涉及帧数=0)

    rad = cont.speck_neighbor_radius
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))

    # 缓存全部二值联合掩膜以便跨相邻帧检查
    masks_bool: list[np.ndarray] = [load_confirmed(i) > 0 for i in range(total_frames)]
    removed_px = 0
    removed_ccs = 0
    touched_frames = 0

    for sc in (scenes or [[0, total_frames]]):
        s0, s1 = max(0, int(sc[0])), min(total_frames, int(sc[1]))
        for idx in range(s0, s1):
            cur = masks_bool[idx]
            if not np.any(cur):
                continue
            num, labels, stats, _ = cv2.connectedComponentsWithStats(cur.astype(np.uint8), connectivity=8)
            small_ids = [k for k in range(1, num) if int(stats[k, cv2.CC_STAT_AREA]) < cont.speck_min_area]
            if not small_ids:
                continue
            nbr_union = np.zeros_like(cur)
            if idx - 1 >= s0:
                nbr_union |= cv2.dilate(masks_bool[idx - 1].astype(np.uint8), ker) > 0
            if idx + 1 < s1:
                nbr_union |= cv2.dilate(masks_bool[idx + 1].astype(np.uint8), ker) > 0

            drop = np.zeros_like(cur)
            for k in small_ids:
                x0 = int(stats[k, cv2.CC_STAT_LEFT])
                y0 = int(stats[k, cv2.CC_STAT_TOP])
                x1 = x0 + int(stats[k, cv2.CC_STAT_WIDTH])
                y1 = y0 + int(stats[k, cv2.CC_STAT_HEIGHT])
                comp = labels[y0:y1, x0:x1] == k
                if not np.any(comp & nbr_union[y0:y1, x0:x1]):
                    drop[y0:y1, x0:x1] |= comp
                    removed_ccs += 1
            if np.any(drop):
                keep = (~drop)
                new_conf = (cur & keep).astype(np.uint8) * 255
                save_confirmed(idx, new_conf)
                eb = load_edge(idx)
                save_edge(idx, (eb & (keep.astype(np.uint8) * 255)))
                for g in groups:
                    gm = load_group(g, idx)
                    if np.any(gm > 0):
                        save_group(g, idx, gm & (keep.astype(np.uint8) * 255))
                removed_px += int(np.count_nonzero(drop))
                touched_frames += 1

    return dict(清理像素=removed_px, 清理连通域=removed_ccs, 涉及帧数=touched_frames)
