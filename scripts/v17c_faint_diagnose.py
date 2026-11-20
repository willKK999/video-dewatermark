"""v17c 只读半透明淡字水印诊断脚本（仅输出数值 JSON，不查看/打印图像内容）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from worker.faint_watermark import FaintSettings, detect_translucent_candidates, dialogue_row_mask, signed_highpass
from worker.trusted_context import ContextSettings, exclusion_mask


def _load_video_frames(video_path: Path | None, indices: set[int]) -> dict[int, np.ndarray]:
    if video_path is None or not video_path.exists() or not indices:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    out: dict[int, np.ndarray] = {}
    max_idx = max(indices)
    idx = 0
    while idx <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in indices:
            out[idx] = frame
        idx += 1
    cap.release()
    return out


def _read_png_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(shape, dtype=bool)
    return img > 0


def _union_box(boxes: list[list[int]]) -> list[int]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _box_intersection_area(b1: list[int], b2: list[int]) -> int:
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)


def _summarize_mask_coverage(
    cand: np.ndarray,
    frame_idx: int,
    fname: str,
    bgr: np.ndarray,
    masks_dir: Path,
    detections: dict,
    layers_json: dict,
    roi_report: dict | None,
    group_names: list[str],
    layer_dirs: list[Path],
) -> dict:
    h, w = cand.shape[:2]
    total = int(np.count_nonzero(cand))
    if total == 0:
        return {}
    ys, xs = np.where(cand)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    masks = _read_png_mask(masks_dir / 'masks' / fname, (h, w))
    edge_band = _read_png_mask(masks_dir / 'edge-band' / fname, (h, w))
    sub_prot = _read_png_mask(masks_dir / 'subtitle-protection' / fname, (h, w))
    scene_prot = _read_png_mask(masks_dir / 'scene-protection' / fname, (h, w))
    dlg_rows = dialogue_row_mask(bgr)

    cov_masks = int(np.count_nonzero(cand & masks))
    cov_edge = int(np.count_nonzero(cand & edge_band))
    cov_core = int(np.count_nonzero(cand & masks & (~edge_band)))
    cov_sub = int(np.count_nonzero(cand & sub_prot))
    cov_sub_dlg = int(np.count_nonzero(cand & sub_prot & dlg_rows))
    cov_sub_nondlg = int(np.count_nonzero(cand & sub_prot & (~dlg_rows)))
    cov_scene = int(np.count_nonzero(cand & scene_prot))

    group_cov = {}
    for gname in group_names:
        gm = _read_png_mask(masks_dir / 'group-objects' / gname / fname, (h, w))
        ov = int(np.count_nonzero(cand & gm))
        if ov > 0:
            group_cov[gname] = dict(像素=ov, 比例=round(ov / total, 4))

    layer_cov = {}
    if layer_dirs:
        layer_union = np.zeros((h, w), dtype=bool)
        for idx, ldir in enumerate(layer_dirs):
            lm = _read_png_mask(ldir / fname, (h, w))
            ov = int(np.count_nonzero(cand & lm))
            if ov > 0:
                layer_cov[f'layer_{idx:02d}'] = dict(像素=ov, 比例=round(ov / total, 4))
            layer_union |= lm
        layer_union_ov = int(np.count_nonzero(cand & layer_union))
    else:
        layer_union_ov = cov_masks
        if cov_masks > 0 and layers_json.get('layers'):
            layer_cov['layer_00_masks_proxy'] = dict(像素=cov_masks, 比例=round(cov_masks / total, 4))

    # detections 跟踪框与 OCR 字幕框覆盖
    det_track_mask = np.zeros((h, w), dtype=bool)
    hit_tracks = []
    for tr in detections.get('tracks', []):
        for item in tr.get('hints', []):
            if int(item['frame']) == frame_idx:
                tb = [int(round(v)) for v in item['box']]
                x0, y0, x1, y1 = max(0, tb[0]), max(0, tb[1]), min(w, tb[2]), min(h, tb[3])
                if x1 > x0 and y1 > y0:
                    tm = np.zeros((h, w), dtype=bool)
                    tm[y0:y1, x0:x1] = True
                    ov = int(np.count_nonzero(cand & tm))
                    if ov > 0:
                        det_track_mask |= tm
                        hit_tracks.append(dict(track_id=int(tr.get('id', -1)), box=tb, 像素=ov, 比例=round(ov / total, 4)))
                break

    det_sub_mask = np.zeros((h, w), dtype=bool)
    hit_ocr_subs = []
    for item in detections.get('subtitles', {}).get(str(frame_idx), []):
        sb = [int(round(v)) for v in item['box']]
        x0, y0, x1, y1 = max(0, sb[0]), max(0, sb[1]), min(w, sb[2]), min(h, sb[3])
        if x1 > x0 and y1 > y0:
            sm = np.zeros((h, w), dtype=bool)
            sm[y0:y1, x0:x1] = True
            ov = int(np.count_nonzero(cand & sm))
            if ov > 0:
                det_sub_mask |= sm
                hit_ocr_subs.append(dict(text=str(item.get('text', '')), box=sb, 像素=ov, 比例=round(ov / total, 4)))

    # ROI 任务覆盖（按 roi-report 的帧区间与施工扩展口径）
    ctx_cfg = ContextSettings()
    roi_union = np.zeros((h, w), dtype=bool)
    hit_rois = []
    if roi_report:
        for tidx, task in enumerate(roi_report.get('任务', [])):
            f0, f1 = int(task['开始帧']), int(task['结束帧'])
            if not (f0 <= frame_idx < f1):
                continue
            rx0, ry0, rx1, ry1 = [int(v) for v in task['ROI']]
            rm = np.zeros((h, w), dtype=bool)
            rm[max(0, ry0) : min(h, ry1), max(0, rx0) : min(w, rx1)] = True
            ov_box = int(np.count_nonzero(cand & rm))
            # 确认掩膜交集在 ROI 内才真正在该任务里作为水印被补画
            conf_in_roi = (exclusion_mask(masks.astype(np.uint8) * 255, ctx_cfg) > 0) & rm
            ov_active = int(np.count_nonzero(cand & conf_in_roi))
            if ov_box > 0:
                roi_union |= rm
                hit_rois.append(
                    dict(
                        任务序号=tidx + 1,
                        帧区间=[f0, f1],
                        ROI=[rx0, ry0, rx1, ry1],
                        ROI框覆盖像素=ov_box,
                        ROI框覆盖比例=round(ov_box / total, 4),
                        含确认施工域覆盖像素=ov_active,
                        含确认施工域覆盖比例=round(ov_active / total, 4),
                        原任务通过=bool(task.get('通过')),
                    )
                )

    return dict(
        frame=frame_idx,
        外接框=box,
        面积=total,
        masks=dict(像素=cov_masks, 比例=round(cov_masks / total, 4)),
        edge_band=dict(像素=cov_edge, 比例=round(cov_edge / total, 4)),
        masks_core=dict(像素=cov_core, 比例=round(cov_core / total, 4)),
        subtitle_protection=dict(
            像素=cov_sub,
            比例=round(cov_sub / total, 4),
            真实对话行重叠像素=cov_sub_dlg,
            非对话孤立OCR重叠像素=cov_sub_nondlg,
        ),
        scene_protection=dict(像素=cov_scene, 比例=round(cov_scene / total, 4)),
        group_objects=group_cov,
        layers=dict(总覆盖像素=layer_union_ov, 总覆盖比例=round(layer_union_ov / total, 4), 分层=layer_cov),
        detections=dict(
            track框总覆盖像素=int(np.count_nonzero(cand & det_track_mask)),
            track框总覆盖比例=round(int(np.count_nonzero(cand & det_track_mask)) / total, 4),
            命中轨迹=hit_tracks,
            OCR字幕总覆盖像素=int(np.count_nonzero(cand & det_sub_mask)),
            OCR字幕总覆盖比例=round(int(np.count_nonzero(cand & det_sub_mask)) / total, 4),
            命中OCR条目=hit_ocr_subs,
        ),
        roi_tasks=dict(
            ROI总覆盖像素=int(np.count_nonzero(cand & roi_union)),
            ROI总覆盖比例=round(int(np.count_nonzero(cand & roi_union)) / total, 4),
            命中任务=hit_rois,
        ),
    )


def _compute_video_diffs(
    cand: np.ndarray,
    src_bgr: np.ndarray,
    version_frames: dict[str, np.ndarray],
) -> dict:
    if not np.any(cand):
        return {}
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    src_hp = signed_highpass(src_gray, 4.0)
    src_hp_mean = float(np.mean(src_hp[cand]))
    out = dict(源片高通均值=round(src_hp_mean, 3))
    for ver, img in version_frames.items():
        if img is None:
            continue
        diff = np.abs(src_bgr.astype(np.float32) - img.astype(np.float32))[cand]
        out_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        out_hp = signed_highpass(out_gray, 4.0)
        out_hp_mean = float(np.mean(out_hp[cand]))
        hp_diff = float(np.mean(np.abs(src_hp[cand] - out_hp[cand])))
        out[ver] = dict(
            源片与成片MAE=round(float(np.mean(diff)), 3),
            源片与成片P90=round(float(np.percentile(diff, 90)), 3),
            成片高通均值=round(out_hp_mean, 3),
            高通差均值=round(hp_diff, 3),
            高通残留比例=round(out_hp_mean / src_hp_mean, 4) if src_hp_mean > 1e-3 else None,
        )
    return out


def diagnose_faint(
    masks_dir: Path,
    frames_dir: Path,
    source_video: Path | None = None,
    v13e_video: Path | None = None,
    v16_video: Path | None = None,
    v17b_video: Path | None = None,
    roi_report_path: Path | None = None,
    probe_frames: list[int] | None = None,
    cfg: FaintSettings | None = None,
) -> dict:
    """运行通用半透明叠加诊断并汇总全片与重点核查帧报告。"""
    det = json.loads((masks_dir / 'detections.json').read_text(encoding='utf-8'))
    layers_json = json.loads((masks_dir / 'layers.json').read_text(encoding='utf-8')) if (masks_dir / 'layers.json').exists() else {}
    roi_report = json.loads(roi_report_path.read_text(encoding='utf-8')) if roi_report_path and roi_report_path.exists() else None

    frame_files = sorted((masks_dir / 'masks').glob('*.png'))
    total_frames = len(frame_files)
    if total_frames == 0:
        raise FileNotFoundError(f'掩膜目录为空: {masks_dir / "masks"}')

    group_root = masks_dir / 'group-objects'
    group_names = sorted(d.name for d in group_root.iterdir() if d.is_dir()) if group_root.exists() else []
    layer_dirs = sorted(d for d in (masks_dir / 'layers').glob('*') if d.is_dir()) if (masks_dir / 'layers').exists() else []

    cap_src = None
    def load_bgr(fidx: int) -> np.ndarray:
        p = frames_dir / f'{fidx:06d}.png'
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
        nonlocal cap_src
        if source_video is not None and source_video.exists():
            if cap_src is None:
                cap_src = cv2.VideoCapture(str(source_video))
            cap_src.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frm = cap_src.read()
            if ok and frm is not None:
                return frm
        raise FileNotFoundError(f'无法读取帧 {fidx}')

    def load_confirmed(fidx: int) -> np.ndarray:
        m = cv2.imread(str(masks_dir / 'masks' / f'{fidx:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f'无法读取确认掩膜 {fidx}')
        return m

    detected = detect_translucent_candidates(
        total_frames=total_frames,
        detections=det,
        load_bgr=load_bgr,
        load_confirmed=load_confirmed,
        cfg=cfg,
    )
    if cap_src is not None:
        cap_src.release()

    per_masks = detected['per_frame_masks']
    per_prov = detected['per_frame_provenance']
    probe = set(probe_frames if probe_frames is not None else [157, 411])

    # 按来源（track_id / static scene）聚类候选段并选代表帧计算完整覆盖与差异
    grouped: dict[tuple[str, int], list[tuple[int, np.ndarray, dict]]] = {}
    for fidx in range(total_frames):
        m = per_masks[fidx]
        if not np.any(m):
            continue
        conf = load_confirmed(fidx) > 0
        uncov = m & (~conf)
        for prov in per_prov[fidx]:
            key = (str(prov['来源']), int(prov.get('track_id', prov.get('scene', -1))))
            grouped.setdefault(key, []).append((fidx, uncov if np.any(uncov) else m, prov))

    # 收集需要从对比视频读取的帧（各簇代表帧 + probe 帧）
    sample_indices = set(f for f in probe if 0 <= f < total_frames)
    cluster_reps: list[tuple[tuple[str, int], list[tuple[int, np.ndarray, dict]], int]] = []
    for key, rows in sorted(grouped.items()):
        # 选择未覆盖像素最多的帧作代表帧
        best_row = max(rows, key=lambda r: int(np.count_nonzero(r[1])))
        rep_f = int(best_row[0])
        sample_indices.add(rep_f)
        cluster_reps.append((key, rows, rep_f))

    v13e_map = _load_video_frames(v13e_video, sample_indices)
    v16_map = _load_video_frames(v16_video, sample_indices)
    v17b_map = _load_video_frames(v17b_video, sample_indices)

    candidates_report = []
    for key, rows, rep_f in cluster_reps:
        frames_hit = [r[0] for r in rows]
        boxes = [r[2]['box'] for r in rows]
        raw_areas = [int(r[2]['pixels']) for r in rows]
        uncov_areas = [int(np.count_nonzero(r[1])) for r in rows]
        fname = f'{rep_f:06d}.png'
        rep_bgr = load_bgr(rep_f)
        rep_mask = per_masks[rep_f]
        # 同时报告整体候选像素覆盖 + 未覆盖像素面积
        cov = _summarize_mask_coverage(
            rep_mask, rep_f, fname, rep_bgr, masks_dir, det, layers_json, roi_report, group_names, layer_dirs
        )
        uncov_mask = rep_mask & (~(load_confirmed(rep_f) > 0))
        diffs = _compute_video_diffs(
            uncov_mask if np.any(uncov_mask) else rep_mask,
            rep_bgr,
            dict(v13e=v13e_map.get(rep_f), v16=v16_map.get(rep_f), v17b=v17b_map.get(rep_f)),
        )
        candidates_report.append(
            dict(
                来源=key[0],
                轨迹或镜头编号=key[1],
                帧区间=[int(min(frames_hit)), int(max(frames_hit))],
                命中帧数=len(frames_hit),
                全部候选外接框=_union_box(boxes),
                平均候选面积=round(float(np.mean(raw_areas)), 1),
                平均未覆盖面积=round(float(np.mean(uncov_areas)), 1),
                代表帧=rep_f,
                代表帧覆盖=cov,
                代表帧源与成片差异=diffs,
            )
        )

    def inspect_half_frame(fidx: int, half: str = 'right') -> dict:
        if not (0 <= fidx < total_frames):
            return dict(frame=fidx, 存在=False)
        bgr = load_bgr(fidx)
        h, w = bgr.shape[:2]
        cand_full = per_masks[fidx]
        half_mask = np.zeros((h, w), dtype=bool)
        if half == 'right':
            half_mask[:, w // 2 :] = True
        else:
            half_mask[:, : w // 2] = True
        cand_half = cand_full & half_mask
        fname = f'{fidx:06d}.png'
        conf = _read_png_mask(masks_dir / 'masks' / fname, (h, w))
        subp = _read_png_mask(masks_dir / 'subtitle-protection' / fname, (h, w))

        # 拆解连通块逐个报告（用少量膨胀将同字笔画归组便于计数，但面积统计仍按精确笔画像素）
        if np.any(cand_half):
            cc_src = cv2.dilate(cand_half.astype(np.uint8), np.ones((5, 5), np.uint8))
            ncc, lbl, st, _ = cv2.connectedComponentsWithStats(cc_src)
        else:
            ncc, lbl = 1, np.zeros((h, w), dtype=np.int32)

        cc_list = []
        for k in range(1, ncc):
            sub_c = (lbl == k) & cand_half
            if not np.any(sub_c):
                continue
            cov_k = _summarize_mask_coverage(
                sub_c, fidx, fname, bgr, masks_dir, det, layers_json, roi_report, group_names, layer_dirs
            )
            diff_k = _compute_video_diffs(
                sub_c, bgr, dict(v13e=v13e_map.get(fidx), v16=v16_map.get(fidx), v17b=v17b_map.get(fidx))
            )
            cc_list.append(dict(连通组=k, 覆盖=cov_k, 源与成片差异=diff_k))

        total_cov = _summarize_mask_coverage(
            cand_half, fidx, fname, bgr, masks_dir, det, layers_json, roi_report, group_names, layer_dirs
        ) if np.any(cand_half) else {}
        total_diff = _compute_video_diffs(
            cand_half, bgr, dict(v13e=v13e_map.get(fidx), v16=v16_map.get(fidx), v17b=v17b_map.get(fidx))
        ) if np.any(cand_half) else {}

        uncov_px = int(np.count_nonzero(cand_half & (~conf)))
        sub_block_px = int(np.count_nonzero(cand_half & (~conf) & subp))
        return dict(
            frame=fidx,
            半区=half,
            半区起点x=w // 2 if half == 'right' else 0,
            候选总面积=int(np.count_nonzero(cand_half)),
            未被确认掩膜覆盖像素=uncov_px,
            未覆盖中落入字幕保护像素=sub_block_px,
            总体覆盖=total_cov,
            总体源与成片差异=total_diff,
            分块明细=cc_list,
            来源追踪=per_prov[fidx],
            覆盖结论=(
                f'第{fidx}帧{half}半区检测到{int(np.count_nonzero(cand_half))}个稳定半透明候选像素，'
                f'其中{uncov_px}像素（{round(uncov_px / max(1, int(np.count_nonzero(cand_half))) * 100, 1)}%）未被masks覆盖，'
                f'{sub_block_px}像素同时落入非对话OCR字幕保护区。'
            ) if np.any(cand_half) else f'第{fidx}帧{half}半区未检测到未覆盖半透明候选。',
        )

    total_frames_with_uncovered = sum(
        1 for fidx in range(total_frames) if np.any(per_masks[fidx] & (~load_confirmed(fidx) > 0 if False else ~(load_confirmed(fidx) > 0)))
    )

    return dict(
        方法说明='通用轨迹对齐有符号高通时域中值 + 跨镜头核心互相关迁移 + 镜头静态低饱和高通共识（无针对片名/坐标/帧号规则）',
        总帧数=total_frames,
        存在未覆盖半透明候选帧数=total_frames_with_uncovered,
        轨迹统计=detected['track_summary'],
        镜头静态候选像素=detected['static_scenes'],
        候选簇汇总=candidates_report,
        第157帧右半画面=inspect_half_frame(157, 'right'),
        第411帧右半画面=inspect_half_frame(411, 'right'),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='v17c 通用半透明淡字水印只读诊断脚本')
    parser.add_argument('--masks-dir', type=Path, required=True)
    parser.add_argument('--frames-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=None)
    parser.add_argument('--v13e', type=Path, default=None)
    parser.add_argument('--v16', type=Path, default=None)
    parser.add_argument('--v17b', type=Path, default=None)
    parser.add_argument('--roi-report', type=Path, default=None)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    result = diagnose_faint(
        masks_dir=args.masks_dir,
        frames_dir=args.frames_dir,
        source_video=args.source,
        v13e_video=args.v13e,
        v16_video=args.v16,
        v17b_video=args.v17b,
        roi_report_path=args.roi_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(
        dict(
            已写入=str(args.output),
            总帧数=result['总帧数'],
            存在未覆盖半透明候选帧数=result['存在未覆盖半透明候选帧数'],
            第157帧右半=result['第157帧右半画面'].get('覆盖结论'),
            第411帧右半=result['第411帧右半画面'].get('覆盖结论'),
        ),
        ensure_ascii=False,
    ))


if __name__ == '__main__':
    main()
