"""通用半透明叠加水印检测与安全掩膜补充模块。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np

from .edge_residue import highlight_guard
from .trusted_context import ContextSettings, exclusion_mask
from .v12_masks import Settings as V12Settings, detect_text_rows


@dataclass(frozen=True)
class FaintSettings:
    """通用半透明叠加水印检测参数配置。"""

    sample_frames: int = 32
    min_track_frames: int = 12
    min_unmasked_samples: int = 6
    min_track_bg_mad: float = 8.0
    min_pixel_bg_mad: float = 5.5
    signed_highpass_min: float = 4.5
    consistency_min: float = 0.65
    max_saturation: float = 90.0
    cross_track_corr_min: float = 0.55
    cross_track_hp_min: float = 1.5
    cross_track_consistency_min: float = 0.50
    static_scene_min_frames: int = 16
    static_scene_bg_mad_min: float = 5.0
    static_scene_moving_fraction_min: float = 0.65
    static_scene_hp_min: float = 5.5
    static_scene_consistency_min: float = 0.65
    min_cc_area: int = 10
    static_min_cc_area: int = 16
    stroke_dilate_radius: int = 1
    max_ocr_box_width_ratio: float = 0.20
    max_ocr_box_height_ratio: float = 0.10
    ocr_overlap_min_pixels: int = 6
    halo_rel: tuple[float, float, float, float] = (-0.25, -0.30, 1.30, 1.85)
    gaussian_sigma: float = 4.0
    core_refine_radius: int = 12
    core_match_min_corr: float = 0.20
    min_track_box_w: int = 24
    min_track_box_h: int = 12

    def __post_init__(self) -> None:
        if self.sample_frames < 6 or self.min_track_frames < 4:
            raise ValueError('轨迹采样帧数参数无效')
        if self.min_unmasked_samples < 3:
            raise ValueError('未遮蔽样本数下限不能低于3')
        if not 0.0 <= self.consistency_min <= 1.0 or not 0.0 <= self.cross_track_consistency_min <= 1.0:
            raise ValueError('时域一致率阈值必须在[0, 1]之间')
        if self.stroke_dilate_radius < 0 or self.stroke_dilate_radius > 3:
            raise ValueError('半透明笔画膨胀半径必须在[0, 3]之间')


def signed_highpass(gray: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    """计算有符号局部高通响应（突出背景上叠加的半透明亮笔画）。"""
    gray_f = gray.astype(np.float32, copy=False)
    blur = cv2.GaussianBlur(gray_f, (0, 0), float(sigma))
    return gray_f - blur


def filter_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """保留面积不少于阈值的八连通区域。"""
    src = (mask > 0).astype(np.uint8)
    if not np.any(src):
        return np.zeros_like(src, dtype=bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(src)
    keep = np.zeros(num, dtype=bool)
    for k in range(1, num):
        if int(stats[k, cv2.CC_STAT_AREA]) >= min_area:
            keep[k] = True
    return keep[labels]


def dialogue_row_mask(image: np.ndarray, cfg: V12Settings | None = None) -> np.ndarray:
    """提取单帧中形态学字幕行检测到的真实对话字幕外接区域。"""
    settings = cfg or V12Settings()
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for item in detect_text_rows(image, settings):
        box = item['box'] if isinstance(item, dict) else item
        x0 = max(0, int(np.floor(box[0])))
        y0 = max(0, int(np.floor(box[1])))
        x1 = min(w, int(np.ceil(box[2])))
        y1 = min(h, int(np.ceil(box[3])))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def _track_frame_boxes(track: dict) -> list[tuple[int, tuple[float, float, float, float]]]:
    items = track.get('hints') or track.get('observations') or []
    out: list[tuple[int, tuple[float, float, float, float]]] = []
    for item in items:
        box = item.get('box')
        if box is None or len(box) != 4:
            continue
        out.append((int(item['frame']), (float(box[0]), float(box[1]), float(box[2]), float(box[3]))))
    out.sort(key=lambda row: row[0])
    return out


def _refine_core_anchor(
    hp: np.ndarray,
    box: tuple[float, float, float, float],
    core: np.ndarray,
    mw: int,
    mh: int,
    radius: int,
    min_corr: float,
) -> tuple[float, float]:
    bx = int(round(box[0]))
    by = int(round(box[1]))
    h, w = hp.shape[:2]
    cw = min(core.shape[1], max(8, int(round(0.8 * mw))))
    ch = min(core.shape[0], max(6, mh))
    tpl = core[:ch, :cw]
    if tpl.size == 0 or float(np.std(tpl)) < 1e-4:
        return float(bx), float(by)
    sx0 = max(0, bx - radius)
    sy0 = max(0, by - radius)
    sx1 = min(w, bx + cw + radius)
    sy1 = min(h, by + ch + radius)
    patch = hp[sy0:sy1, sx0:sx1]
    if patch.shape[0] < ch or patch.shape[1] < cw:
        return float(bx), float(by)
    res = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if np.isfinite(max_val) and float(max_val) >= min_corr:
        return float(sx0 + max_loc[0]), float(sy0 + max_loc[1])
    return float(bx), float(by)


def _compute_conditional_metrics(
    hp_arr: np.ndarray,
    mk_arr: np.ndarray,
    min_unmasked: int,
    pre_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """向量化计算非遮蔽帧上的条件高通中值与一致率为正的比例。"""
    import warnings

    unmk_cnt = np.sum(~mk_arr, axis=0)
    ch, cw = hp_arr.shape[1:]
    cond_med = np.zeros((ch, cw), dtype=np.float32)
    cond_pos = np.zeros((ch, cw), dtype=np.float32)
    eligible = unmk_cnt >= min_unmasked
    if pre_mask is not None:
        eligible &= pre_mask
    if not np.any(eligible):
        return cond_med, cond_pos, unmk_cnt
    vals = np.where(~mk_arr[:, eligible], hp_arr[:, eligible], np.nan)
    pos_vals = np.where(~mk_arr[:, eligible], (hp_arr[:, eligible] > 2.0).astype(np.float32), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        cond_med[eligible] = np.nan_to_num(np.nanmedian(vals, axis=0), nan=0.0)
        cond_pos[eligible] = np.nan_to_num(np.nanmean(pos_vals, axis=0), nan=0.0)
    return cond_med, cond_pos, unmk_cnt


def _analyze_single_track(
    tid: int,
    track: dict,
    load_frame: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    load_mask: Callable[[int], np.ndarray],
    cfg: FaintSettings,
) -> dict | None:
    """分析单条跟踪轨迹在规范局部坐标下的半透明持久叠加笔画。"""
    entries = _track_frame_boxes(track)
    if len(entries) < cfg.min_track_frames:
        return None
    mw = int(round(float(np.median([b[2] - b[0] for _, b in entries]))))
    mh = int(round(float(np.median([b[3] - b[1] for _, b in entries]))))
    if mw < cfg.min_track_box_w or mh < cfg.min_track_box_h:
        return None

    rx0, ry0, rx1, ry1 = cfg.halo_rel
    px0_off = int(round(max(0.0, -rx0) * mw))
    px1_off = int(round(max(0.0, rx1 - 1.0) * mw))
    py0_off = int(round(max(0.0, -ry0) * mh))
    py1_off = int(round(max(0.0, ry1 - 1.0) * mh))
    cw = mw + px0_off + px1_off
    ch = mh + py0_off + py1_off

    sample_idx = np.linspace(0, len(entries) - 1, min(cfg.sample_frames, len(entries)), dtype=int)
    samples = [entries[int(k)] for k in sample_idx]
    mid_f, mid_box = samples[len(samples) // 2]
    _, _, _, mid_hp = load_frame(mid_f)
    h_img, w_img = mid_hp.shape[:2]
    mbx = max(0, min(w_img - 1, int(round(mid_box[0]))))
    mby = max(0, min(h_img - 1, int(round(mid_box[1]))))
    mcw = max(8, min(w_img - mbx, int(round(0.8 * mw))))
    mch = max(6, min(h_img - mby, mh))
    core = mid_hp[mby : mby + mch, mbx : mbx + mcw].copy()

    gr_list, hp_list, sat_list, mk_list = [], [], [], []
    for f, box in samples:
        bgr, gray, sat, hp = load_frame(f)
        mk = load_mask(f)
        ax, ay = _refine_core_anchor(hp, box, core, mw, mh, cfg.core_refine_radius, cfg.core_match_min_corr)
        mat = np.array([[1.0, 0.0, -(ax - px0_off)], [0.0, 1.0, -(ay - py0_off)]], dtype=np.float32)
        gr_list.append(cv2.warpAffine(gray, mat, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        hp_list.append(cv2.warpAffine(hp, mat, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        sat_list.append(cv2.warpAffine(sat, mat, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        mk_list.append(cv2.warpAffine(mk.astype(np.float32), mat, (cw, ch), flags=cv2.INTER_NEAREST) > 0.5)

    gr_arr = np.stack(gr_list)
    hp_arr = np.stack(hp_list)
    sat_arr = np.stack(sat_list)
    mk_arr = np.stack(mk_list)

    bg_mad = np.median(np.abs(gr_arr - np.median(gr_arr, axis=0)), axis=0)
    track_bg_mad = float(np.median(bg_mad))
    cond_med_hp, cond_pos, unmk_cnt = _compute_conditional_metrics(hp_arr, mk_arr, cfg.min_unmasked_samples)
    med_sat = np.median(sat_arr, axis=0)
    med_hp_all = np.median(hp_arr, axis=0)
    core_tmpl = med_hp_all[py0_off : py0_off + mh, px0_off : px0_off + max(8, int(round(0.85 * mw)))].copy()

    proven = (
        (unmk_cnt >= cfg.min_unmasked_samples)
        & (bg_mad >= cfg.min_pixel_bg_mad)
        & (cond_med_hp >= cfg.signed_highpass_min)
        & (cond_pos >= cfg.consistency_min)
        & (med_sat <= cfg.max_saturation)
    )
    if track_bg_mad < cfg.min_track_bg_mad:
        proven = np.zeros_like(proven, dtype=bool)
    else:
        proven = filter_small_components(proven, cfg.min_cc_area)

    return dict(
        tid=tid,
        track_id=track.get('id', tid + 1),
        scene=track.get('scene'),
        entries=entries,
        mw=mw,
        mh=mh,
        cw=cw,
        ch=ch,
        offs=(px0_off, py0_off),
        core=core,
        core_tmpl=core_tmpl,
        bg_mad_med=track_bg_mad,
        cond_med_hp=cond_med_hp,
        cond_pos=cond_pos,
        med_sat=med_sat,
        unmk_cnt=unmk_cnt,
        proven=proven,
        certified_direct=bool(track_bg_mad >= cfg.min_track_bg_mad and np.any(proven)),
        transferred_from=None,
        cross_corr=None,
    )


def _match_core_ncc(core_lo: np.ndarray, core_hi: np.ndarray) -> tuple[float, int, int]:
    """用较小步长二维归一化互相关匹配低运动轨迹与已证实高运动轨迹的核心笔画模板。"""
    if core_lo.size == 0 or core_hi.size == 0:
        return -1.0, 0, 0
    if float(np.std(core_lo)) < 1e-4 or float(np.std(core_hi)) < 1e-4:
        return -1.0, 0, 0
    # 轻微留边搜索以容忍 OCR 框左上角±12px 偏移
    pad = 12
    hi_r = cv2.resize(core_hi, (core_lo.shape[1], core_lo.shape[0]), interpolation=cv2.INTER_LINEAR)
    hi_pad = cv2.copyMakeBorder(hi_r, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0.0)
    res = cv2.matchTemplate(hi_pad, core_lo, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if not np.isfinite(max_val):
        return -1.0, 0, 0
    dx = int(max_loc[0] - pad)
    dy = int(max_loc[1] - pad)
    return float(max_val), dx, dy


def _apply_cross_track_transfer(models: list[dict], cfg: FaintSettings) -> None:
    high_motion = [m for m in models if m['certified_direct']]
    if not high_motion:
        return
    for lo in models:
        if lo['bg_mad_med'] >= cfg.min_track_bg_mad:
            continue
        best_corr = -1.0
        best_model = None
        best_shift = (0, 0)
        for hi in high_motion:
            # 长宽比差距过大时不跨类迁移
            ratio_w = hi['mw'] / max(1.0, float(lo['mw']))
            ratio_h = hi['mh'] / max(1.0, float(lo['mh']))
            if not (0.6 <= ratio_w <= 1.65 and 0.6 <= ratio_h <= 1.65):
                continue
            corr, dx, dy = _match_core_ncc(lo['core_tmpl'], hi['core_tmpl'])
            if corr > best_corr:
                best_corr = corr
                best_model = hi
                best_shift = (dx, dy)
        if best_model is None or best_corr < cfg.cross_track_corr_min:
            continue
        sx, sy = best_model['cw'] / float(lo['cw']), best_model['ch'] / float(lo['ch'])
        warped = cv2.resize(
            best_model['proven'].astype(np.float32),
            (lo['cw'], lo['ch']),
            interpolation=cv2.INTER_NEAREST,
        ) > 0.5
        if best_shift != (0, 0):
            dx, dy = best_shift
            # 对齐核心偏移后把 hi 规范模板平移映射到 lo
            dx_c = int(round(dx / max(1e-3, sx * 0.5 + 0.5)))
            dy_c = int(round(dy / max(1e-3, sy * 0.5 + 0.5)))
            mat = np.array([[1.0, 0.0, float(-dx_c)], [0.0, 1.0, float(-dy_c)]], dtype=np.float32)
            warped = cv2.warpAffine(warped.astype(np.float32), mat, (lo['cw'], lo['ch']), flags=cv2.INTER_NEAREST) > 0.5
        verified = (
            warped
            & (lo['unmk_cnt'] >= cfg.min_unmasked_samples)
            & (lo['cond_pos'] >= cfg.cross_track_consistency_min)
            & (lo['cond_med_hp'] >= cfg.cross_track_hp_min)
            & (lo['med_sat'] <= cfg.max_saturation + 10.0)
        )
        verified = filter_small_components(verified, cfg.min_cc_area)
        if np.any(verified):
            lo['proven'] = verified
            lo['transferred_from'] = int(best_model['track_id'])
            lo['cross_corr'] = round(best_corr, 4)


def detect_translucent_candidates(
    total_frames: int,
    detections: dict,
    load_bgr: Callable[[int], np.ndarray],
    load_confirmed: Callable[[int], np.ndarray],
    cfg: FaintSettings | None = None,
    v12_cfg: V12Settings | None = None,
    enable_static_scene: bool = True,
) -> dict:
    """在整段帧序列中通用检测未被确认掩膜覆盖的稳定半透明水印笔画。"""
    from collections import OrderedDict

    settings = cfg or FaintSettings()
    sub_cfg = v12_cfg or V12Settings()
    frame_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()
    mask_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get_frame(f: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if f in frame_cache:
            frame_cache.move_to_end(f)
            return frame_cache[f]
        bgr = load_bgr(f)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
        hp = signed_highpass(gray, settings.gaussian_sigma)
        frame_cache[f] = (bgr, gray, sat, hp)
        if len(frame_cache) > 96:
            frame_cache.popitem(last=False)
        return frame_cache[f]

    def get_mask(f: int) -> np.ndarray:
        if f in mask_cache:
            mask_cache.move_to_end(f)
            return mask_cache[f]
        mask_cache[f] = load_confirmed(f) > 0
        if len(mask_cache) > 192:
            mask_cache.popitem(last=False)
        return mask_cache[f]

    # 1. 轨迹对齐的时域半透明笔画检测
    models: list[dict] = []
    for tid, tr in enumerate(detections.get('tracks', [])):
        model = _analyze_single_track(tid, tr, get_frame, get_mask, settings)
        if model is not None:
            models.append(model)
    _apply_cross_track_transfer(models, settings)

    # 2. 镜头级静止半透明叠加检测（补充无轨迹覆盖的固定位置淡字）
    scenes = detections.get('scenes') or [[0, total_frames]]
    static_scene_masks: dict[int, np.ndarray] = {}
    if enable_static_scene:
        for sc_idx, sc in enumerate(scenes):
            s0, s1 = int(sc[0]), int(sc[1])
            if s1 - s0 < settings.static_scene_min_frames:
                continue
            s_idx = np.linspace(s0, s1 - 1, min(settings.sample_frames, s1 - s0), dtype=int)
            gr_l, hp_l, sat_l, mk_l, row_l = [], [], [], [], []
            for f in s_idx:
                bgr, gray, sat, hp = get_frame(int(f))
                gr_l.append(gray)
                hp_l.append(hp)
                sat_l.append(sat)
                mk_l.append(get_mask(int(f)))
                row_l.append(dialogue_row_mask(bgr, sub_cfg))
            gr_arr = np.stack(gr_l)
            hp_arr = np.stack(hp_l)
            sat_arr = np.stack(sat_l)
            mk_arr = np.stack(mk_l)
            row_union = np.any(np.stack(row_l), axis=0)
            bg_mad = np.median(np.abs(gr_arr - np.median(gr_arr, axis=0)), axis=0)
            # 仅当场景大面积背景处于平移/运动状态（而非仅行人走过静止背景）时才直接采信单镜头静态叠加，避免静止物体误报
            if float(np.mean(bg_mad >= settings.static_scene_bg_mad_min)) < settings.static_scene_moving_fraction_min:
                continue
            med_sat = np.median(sat_arr, axis=0)
            pre_ok = (bg_mad >= settings.static_scene_bg_mad_min) & (med_sat <= settings.max_saturation) & (~row_union)
            cond_med, cond_pos, unmk_cnt = _compute_conditional_metrics(
                hp_arr, mk_arr, settings.min_unmasked_samples, pre_mask=pre_ok
            )
            st_cand = (
                (unmk_cnt >= settings.min_unmasked_samples)
                & (bg_mad >= settings.static_scene_bg_mad_min)
                & (cond_med >= settings.static_scene_hp_min)
                & (cond_pos >= settings.static_scene_consistency_min)
                & (med_sat <= settings.max_saturation)
                & (~row_union)
            )
            st_cand = filter_small_components(st_cand, settings.static_min_cc_area)
            if np.any(st_cand):
                static_scene_masks[sc_idx] = st_cand

    # 3. 逐帧投射回全画幅并生成候选汇总
    sample_bgr = load_bgr(0)
    h_img, w_img = sample_bgr.shape[:2]
    per_frame_masks: list[np.ndarray] = [np.zeros((h_img, w_img), dtype=bool) for _ in range(total_frames)]
    per_frame_provenance: list[list[dict]] = [[] for _ in range(total_frames)]

    # 按帧索引避免对同一帧重复解码与计算高通
    frame_tracks: list[list[tuple[dict, tuple[float, float, float, float]]]] = [[] for _ in range(total_frames)]
    for model in models:
        if not np.any(model['proven']):
            continue
        model['_proven_u8'] = model['proven'].astype(np.uint8)
        model['_mean_hp'] = float(model['cond_med_hp'][model['proven']].mean())
        for f, box in model['entries']:
            if 0 <= f < total_frames:
                frame_tracks[f].append((model, box))

    for f in range(total_frames):
        items = frame_tracks[f]
        if not items:
            continue
        _, _, _, hp = get_frame(f)
        for model, box in items:
            px0_off, py0_off = model['offs']
            mw, mh = model['mw'], model['mh']
            ax, ay = _refine_core_anchor(
                hp, box, model['core'], mw, mh, settings.core_refine_radius, settings.core_match_min_corr
            )
            inv_mat = np.array([[1.0, 0.0, ax - px0_off], [0.0, 1.0, ay - py0_off]], dtype=np.float32)
            projected = cv2.warpAffine(model['_proven_u8'], inv_mat, (w_img, h_img), flags=cv2.INTER_NEAREST) > 0
            if not np.any(projected):
                continue
            per_frame_masks[f] |= projected
            ys, xs = np.where(projected)
            per_frame_provenance[f].append(
                dict(
                    来源='cross_track_transfer' if model['transferred_from'] is not None else 'track_temporal_median',
                    track_id=int(model['track_id']),
                    migrated_from=model['transferred_from'],
                    cross_corr=model['cross_corr'],
                    bg_mad_med=round(model['bg_mad_med'], 2),
                    mean_canonical_hp=round(model['_mean_hp'], 2),
                    box=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    pixels=int(projected.sum()),
                )
            )

    for sc_idx, st_mask in static_scene_masks.items():
        s0, s1 = int(scenes[sc_idx][0]), int(scenes[sc_idx][1])
        ys, xs = np.where(st_mask)
        st_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        px = int(st_mask.sum())
        for f in range(max(0, s0), min(total_frames, s1)):
            per_frame_masks[f] |= st_mask
            per_frame_provenance[f].append(
                dict(
                    来源='scene_static_temporal_median',
                    scene=int(sc_idx),
                    box=st_box,
                    pixels=px,
                )
            )

    track_summary = [
        dict(
            track_id=int(m['track_id']),
            scene=m['scene'],
            frames=[int(m['entries'][0][0]), int(m['entries'][-1][0])],
            bg_mad_med=round(m['bg_mad_med'], 2),
            canonical_proven_pixels=int(m['proven'].sum()),
            certified_direct=bool(m['certified_direct']),
            transferred_from=m['transferred_from'],
            cross_corr=m['cross_corr'],
        )
        for m in models
    ]

    return dict(
        per_frame_masks=per_frame_masks,
        per_frame_provenance=per_frame_provenance,
        track_summary=track_summary,
        static_scenes={int(k): int(v.sum()) for k, v in static_scene_masks.items()},
    )


def reclassify_watermark_subtitles(
    frame_idx: int,
    image: np.ndarray,
    sub_prot_mask: np.ndarray,
    ocr_subtitles: Iterable[dict],
    translucent_mask: np.ndarray,
    cfg: FaintSettings | None = None,
    v12_cfg: V12Settings | None = None,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """仅释放非真实对话字幕行且与稳定半透明水印笔画重合的误分类 OCR 字幕保护框。"""
    settings = cfg or FaintSettings()
    sub_bool = sub_prot_mask > 0
    if not np.any(sub_bool) or not np.any(translucent_mask):
        return sub_prot_mask.copy(), [], dialogue_row_mask(image, v12_cfg)

    h, w = image.shape[:2]
    row_mask = dialogue_row_mask(image, v12_cfg)
    trans_dil = cv2.dilate((translucent_mask > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    updated = sub_bool.copy()
    audit: list[dict] = []
    for item in ocr_subtitles:
        box = item.get('box')
        if box is None or len(box) != 4:
            continue
        x0 = max(0, int(np.floor(box[0])))
        y0 = max(0, int(np.floor(box[1])))
        x1 = min(w, int(np.ceil(box[2])))
        y1 = min(h, int(np.ceil(box[3])))
        if x1 <= x0 or y1 <= y0:
            continue
        bw, bh = x1 - x0, y1 - y0
        if bw >= settings.max_ocr_box_width_ratio * w or bh >= settings.max_ocr_box_height_ratio * h:
            continue
        if np.any(row_mask[y0:y1, x0:x1]):
            continue
        # 需在框内部或紧邻区域与独立验证过的半透明水印笔画重合
        ex0, ey0 = max(0, x0 - 2), max(0, y0 - 2)
        ex1, ey1 = min(w, x1 + 2), min(h, y1 + 2)
        ov_px = int(np.count_nonzero(trans_dil[ey0:ey1, ex0:ex1]))
        if ov_px < settings.ocr_overlap_min_pixels:
            continue
        # 进一步确认待释放框内确有保护像素被取消，且不触及对话字幕掩膜
        region_sub = updated[ey0:ey1, ex0:ex1] & (~row_mask[ey0:ey1, ex0:ex1])
        cleared = int(np.count_nonzero(region_sub))
        if cleared <= 0:
            continue
        updated[ey0:ey1, ex0:ex1] &= row_mask[ey0:ey1, ex0:ex1]
        audit.append(
            dict(
                frame=int(frame_idx),
                box=[int(x0), int(y0), int(x1), int(y1)],
                text=str(item.get('text', '')),
                translucent_overlap_pixels=ov_px,
                cleared_protection_pixels=cleared,
                原因='非形态学对话字幕行且与稳定半透明水印笔画重合的孤立OCR碎片',
            )
        )
    return updated.astype(np.uint8) * 255, audit, row_mask


def build_faint_frame_addition(
    image: np.ndarray,
    translucent_mask: np.ndarray,
    confirmed_mask: np.ndarray,
    protected_mask: np.ndarray,
    cfg: FaintSettings | None = None,
    context_cfg: ContextSettings | None = None,
    max_layer_fraction: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """将已证实半透明候选安全转换为可加入确认掩膜的新增笔画，受保护区与面积上限约束。"""
    settings = cfg or FaintSettings()
    ctx_cfg = context_cfg or ContextSettings()
    cand = translucent_mask > 0
    conf = confirmed_mask > 0
    guard = (protected_mask > 0) | highlight_guard(image)

    raw_uncovered = cand & (~conf)
    if not np.any(raw_uncovered):
        return np.zeros(conf.shape, dtype=np.uint8), dict(raw_uncovered=0, added=0, guard_blocked=0, budget_capped=False)

    if settings.stroke_dilate_radius > 0:
        r = settings.stroke_dilate_radius
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        expanded = cv2.dilate(raw_uncovered.astype(np.uint8), ker) > 0
    else:
        expanded = raw_uncovered

    safe = expanded & (~conf) & (~guard)
    safe = filter_small_components(safe, 6)
    guard_blocked = int(np.count_nonzero(expanded & (~conf) & guard))

    # 严格遵守每层 <= max_layer_fraction (20%) 施工面积保护
    merged_u8 = (conf | safe).astype(np.uint8) * 255
    frac = float(np.count_nonzero(exclusion_mask(merged_u8, ctx_cfg))) / float(merged_u8.size)
    capped = False
    if frac > max_layer_fraction:
        # 仅保留紧贴原未覆盖核心（不额外膨胀），若仍超限则本帧放弃补充以避免触发越界
        strict_safe = raw_uncovered & (~conf) & (~guard)
        merged_strict = (conf | strict_safe).astype(np.uint8) * 255
        frac_strict = float(np.count_nonzero(exclusion_mask(merged_strict, ctx_cfg))) / float(merged_strict.size)
        if frac_strict <= max_layer_fraction:
            safe = strict_safe
        else:
            safe = np.zeros_like(safe, dtype=bool)
        capped = True

    return safe.astype(np.uint8) * 255, dict(
        raw_uncovered=int(np.count_nonzero(raw_uncovered)),
        added=int(np.count_nonzero(safe)),
        guard_blocked=guard_blocked,
        budget_capped=capped,
    )
