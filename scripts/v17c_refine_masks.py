"""v17c 通用确认掩膜构建脚本：半透明淡字补充、OCR误保护重分类、同色孔洞补全与窄收边。"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from worker.edge_residue import EdgeSettings, complete_group_color_holes, refine_group
from worker.faint_watermark import (
    FaintSettings,
    build_faint_frame_addition,
    detect_translucent_candidates,
    reclassify_watermark_subtitles,
)
from worker.hls_layers import VerifiedLayers, write_manifest
from worker.mask_continuity import (
    ContinuitySettings,
    expand_adaptive_halo,
    prune_isolated_specks,
    repair_group_temporal_dropouts,
)
from worker.process import PipelineError
from worker.trusted_context import ContextSettings, exclusion_mask


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, dtype=np.uint8)
    return m


def _atomic_write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.tmp-{path.name}')
    ok = cv2.imwrite(str(tmp), arr)
    if not ok:
        raise PipelineError('IO_ERROR', f'写出PNG失败: {path}')
    tmp.replace(path)


def refine_v17c_masks(
    base_dir: Path,
    out_dir: Path,
    edge_cfg: EdgeSettings | None = None,
    faint_cfg: FaintSettings | None = None,
    frames_dir: Path | None = None,
    neighbor_window: int = 8,
) -> dict:
    """从基础确认掩膜目录通用生成 v17c 确认掩膜（含 group-faint、同色孔洞补全与窄收边）。"""
    e_cfg = edge_cfg or EdgeSettings()
    f_cfg = faint_cfg or FaintSettings()
    ctx_cfg = ContextSettings()

    det = json.loads((base_dir / 'detections.json').read_text(encoding='utf-8'))
    VerifiedLayers(base_dir / 'layers.json', base_dir / 'detections.json')

    files = sorted((base_dir / 'masks').glob('*.png'))
    if not files:
        raise PipelineError('MASK_INVALID', f'基础确认掩膜目录为空: {base_dir / "masks"}')
    src_frames_dir = frames_dir if frames_dir is not None else (base_dir / 'frames')
    if not src_frames_dir.exists():
        raise PipelineError('INPUT_INVALID', f'缺少源帧目录: {src_frames_dir}')

    total = len(files)
    sample_bgr = cv2.imread(str(src_frames_dir / files[0].name))
    if sample_bgr is None:
        raise PipelineError('INPUT_INVALID', f'无法读取首帧: {src_frames_dir / files[0].name}')
    shape = sample_bgr.shape[:2]
    pixels = shape[0] * shape[1]

    out_dir.mkdir(parents=True, exist_ok=True)
    for folder in ('masks', 'edge-band', 'subtitle-protection', 'group-objects'):
        (out_dir / folder).mkdir(parents=True, exist_ok=True)
    for link_name in ('frames', 'scene-protection'):
        dst = out_dir / link_name
        src_target = src_frames_dir if link_name == 'frames' else (base_dir / link_name)
        if src_target.exists() and not dst.exists():
            dst.symlink_to(src_target.resolve())
    if (base_dir / 'animation-events.json').exists():
        old_anim = json.loads((base_dir / 'animation-events.json').read_text(encoding='utf-8'))
        events = old_anim.get('events', [])
        unified_anim = dict(settings=old_anim.get('settings', {}), events=events, layers={'0': events})
        (out_dir / 'animation-events.json').write_text(
            json.dumps(unified_anim, ensure_ascii=False, indent=2), encoding='utf-8'
        )

    groups = sorted(p.name for p in (base_dir / 'group-objects').iterdir() if p.is_dir()) if (base_dir / 'group-objects').exists() else []
    for g in groups + ['group-faint']:
        (out_dir / 'group-objects' / g).mkdir(parents=True, exist_ok=True)

    def load_bgr(idx: int) -> np.ndarray:
        im = cv2.imread(str(src_frames_dir / files[idx].name))
        if im is None:
            raise PipelineError('INPUT_INVALID', f'无法读取帧: {files[idx].name}')
        return im

    def load_confirmed(idx: int) -> np.ndarray:
        return _read_mask(files[idx], shape)

    # 1. 通用半透明叠加检测
    faint_res = detect_translucent_candidates(
        total_frames=total,
        detections=det,
        load_bgr=load_bgr,
        load_confirmed=load_confirmed,
        cfg=f_cfg,
    )
    faint_masks = faint_res['per_frame_masks']
    faint_prov = faint_res['per_frame_provenance']

    # 预构建各组在邻近窗口内的滑动缓存，避免重复IO或全集驻留内存
    grp_caches: dict[str, deque[tuple[int, np.ndarray]]] = {g: deque() for g in groups}
    grp_loaded_until = -1

    def ensure_grp_window(center: int) -> None:
        nonlocal grp_loaded_until
        hi = min(total - 1, center + neighbor_window)
        while grp_loaded_until < hi:
            nxt = grp_loaded_until + 1
            fn = files[nxt].name
            for g in groups:
                gp = base_dir / 'group-objects' / g / fn
                if gp.exists():
                    gm = _read_mask(gp, shape)
                    if np.any(gm):
                        grp_caches[g].append((nxt, gm))
            grp_loaded_until = nxt
        lo = center - neighbor_window
        for g in groups:
            dq = grp_caches[g]
            while dq and dq[0][0] < lo:
                dq.popleft()

    reclassified_audit: list[dict] = []
    faint_records: list[dict] = []
    hole_records: list[dict] = []
    edge_records: list[dict] = []
    total_base_px = 0
    total_faint_px = 0
    total_hole_px = 0
    total_edge_px = 0
    max_excl_frac = 0.0
    subtitles_dict = det.get('subtitles', {})

    for idx, fp in enumerate(files):
        fname = fp.name
        image = load_bgr(idx)
        base_conf = load_confirmed(idx)
        total_base_px += int(np.count_nonzero(base_conf))
        scene_prot = _read_mask(base_dir / 'scene-protection' / fname, shape)
        sub_prot = _read_mask(base_dir / 'subtitle-protection' / fname, shape)

        # 1a. 非对话孤立 OCR 碎片重分类（仅在真实对话行外且与稳定半透明笔画重合时释放保护）
        updated_sub_prot, sub_audit, _ = reclassify_watermark_subtitles(
            idx,
            image,
            sub_prot,
            subtitles_dict.get(str(idx), []),
            faint_masks[idx],
            cfg=f_cfg,
        )
        if sub_audit:
            reclassified_audit.extend(sub_audit)
        _atomic_write_png(out_dir / 'subtitle-protection' / fname, updated_sub_prot)
        protected = scene_prot | updated_sub_prot

        # 1b. 新增经独立证实且未被覆盖的半透明淡字组（group-faint）
        faint_add, f_info = build_faint_frame_addition(
            image, faint_masks[idx], base_conf, protected, cfg=f_cfg, context_cfg=ctx_cfg
        )
        _atomic_write_png(out_dir / 'group-objects' / 'group-faint' / fname, faint_add)
        conf_step1 = base_conf | faint_add
        faint_n = int(np.count_nonzero(faint_add))
        total_faint_px += faint_n
        if faint_n > 0:
            faint_records.append(dict(frame=idx, 新增像素=faint_n, 详情=f_info, 来源=faint_prov[idx]))

        # 2. 同色孔洞补全（修补邻帧可见而单帧漏掉的非保护同色角块，如120-124蓝标左下角）
        ensure_grp_window(idx)
        hole_frame = np.zeros(shape, dtype=np.uint8)
        for g in groups:
            cur_gm = next((m for f_i, m in grp_caches[g] if f_i == idx), None)
            if cur_gm is None or not np.any(cur_gm):
                continue
            nbrs = [m for f_i, m in grp_caches[g] if f_i != idx]
            h_add, h_info = complete_group_color_holes(image, cur_gm, conf_step1, protected, nbrs, e_cfg)
            h_n = int(np.count_nonzero(h_add))
            if h_n > 0:
                # 若加入后越过20%单层保护则拒绝本次孔洞追加
                cand_union = conf_step1 | hole_frame | h_add
                if float(np.count_nonzero(exclusion_mask(cand_union, ctx_cfg))) / pixels <= ctx_cfg.mask_fraction_limit:
                    updated_gm = cur_gm | h_add
                    _atomic_write_png(out_dir / 'group-objects' / g / fname, updated_gm)
                    hole_frame |= h_add
                    total_hole_px += h_n
                    hole_records.append(dict(frame=idx, group=g, **h_info))
                    continue
            _atomic_write_png(out_dir / 'group-objects' / g / fname, cur_gm)
        # 其余未出现在本帧的组写空掩膜或保留一致文件列表
        for g in groups:
            out_gp = out_dir / 'group-objects' / g / fname
            if not out_gp.exists():
                src_gp = base_dir / 'group-objects' / g / fname
                if src_gp.exists():
                    shutil.copy2(src_gp, out_gp)
                else:
                    _atomic_write_png(out_gp, np.zeros(shape, dtype=np.uint8))

        conf_step2 = conf_step1 | hole_frame

        # 3. 窄收边（shape_radius=1，仅区别色且满足外环样本下限）
        edge_frame = np.zeros(shape, dtype=np.uint8)
        for g in groups + ['group-faint']:
            gm = _read_mask(out_dir / 'group-objects' / g / fname, shape)
            if not np.any(gm):
                continue
            e_add, e_info = refine_group(image, gm, conf_step2 | edge_frame, protected, e_cfg)
            if np.any(e_add):
                cand_final = conf_step2 | edge_frame | e_add
                if float(np.count_nonzero(exclusion_mask(cand_final, ctx_cfg))) / pixels <= ctx_cfg.mask_fraction_limit:
                    edge_frame |= e_add
                    edge_records.append(dict(frame=idx, group=g, **e_info))

        final_mask = conf_step2 | edge_frame
        excl_frac = float(np.count_nonzero(exclusion_mask(final_mask, ctx_cfg))) / float(pixels)
        if excl_frac > ctx_cfg.mask_fraction_limit:
            raise PipelineError(
                'MASK_UNCERTAIN',
                f'第{idx}帧确认水印排除区占比{excl_frac:.4f}超过20%保护上限',
                details=dict(frame=idx, exclusion_fraction=excl_frac),
            )
        max_excl_frac = max(max_excl_frac, excl_frac)
        total_edge_px += int(np.count_nonzero(edge_frame))
        _atomic_write_png(out_dir / 'masks' / fname, final_mask)
        _atomic_write_png(out_dir / 'edge-band' / fname, edge_frame)

    # 4. v23 通用时域掩膜连续性修复、低饱和抗锯齿外晕扩展与单帧微噪点清理
    cont_cfg = ContinuitySettings()
    scenes = det.get('scenes') or [[0, total]]
    all_groups = groups + ['group-faint']

    def _load_out_group(g_name: str, frame_i: int) -> np.ndarray:
        return _read_mask(out_dir / 'group-objects' / g_name / files[frame_i].name, shape)

    def _save_out_group(g_name: str, frame_i: int, arr: np.ndarray) -> None:
        _atomic_write_png(out_dir / 'group-objects' / g_name / files[frame_i].name, arr)

    def _load_out_conf(frame_i: int) -> np.ndarray:
        return _read_mask(out_dir / 'masks' / files[frame_i].name, shape)

    def _save_out_conf(frame_i: int, arr: np.ndarray) -> None:
        _atomic_write_png(out_dir / 'masks' / files[frame_i].name, arr)

    def _load_out_edge(frame_i: int) -> np.ndarray:
        return _read_mask(out_dir / 'edge-band' / files[frame_i].name, shape)

    def _save_out_edge(frame_i: int, arr: np.ndarray) -> None:
        _atomic_write_png(out_dir / 'edge-band' / files[frame_i].name, arr)

    def _load_scene_prot(frame_i: int) -> np.ndarray:
        return _read_mask(base_dir / 'scene-protection' / files[frame_i].name, shape)

    def _load_out_sub_prot(frame_i: int) -> np.ndarray:
        return _read_mask(out_dir / 'subtitle-protection' / files[frame_i].name, shape)

    def _save_out_sub_prot(frame_i: int, arr: np.ndarray) -> None:
        _atomic_write_png(out_dir / 'subtitle-protection' / files[frame_i].name, arr)

    continuity_report = repair_group_temporal_dropouts(
        total_frames=total,
        scenes=scenes,
        groups=groups,
        load_bgr=load_bgr,
        load_group=_load_out_group,
        save_group=_save_out_group,
        load_confirmed=_load_out_conf,
        save_confirmed=_save_out_conf,
        load_scene_prot=_load_scene_prot,
        load_sub_prot=_load_out_sub_prot,
        save_sub_prot=_save_out_sub_prot,
        edge_cfg=e_cfg,
        ctx_cfg=ctx_cfg,
        cfg=cont_cfg,
    )

    halo_added_px = 0
    for idx, fp in enumerate(files):
        fname = fp.name
        image = load_bgr(idx)
        cur_conf = _load_out_conf(idx)
        cur_edge = _load_out_edge(idx)
        protected = _load_scene_prot(idx) | _load_out_sub_prot(idx)
        halo_frame = np.zeros(shape, dtype=np.uint8)
        for g in all_groups:
            gm = _load_out_group(g, idx)
            if not np.any(gm):
                continue
            h_add, _ = expand_adaptive_halo(image, gm, cur_conf | halo_frame, protected, e_cfg, cont_cfg)
            if np.any(h_add):
                cand = cur_conf | halo_frame | h_add
                if float(np.count_nonzero(exclusion_mask(cand, ctx_cfg))) / pixels <= ctx_cfg.mask_fraction_limit:
                    halo_frame |= h_add
        if np.any(halo_frame):
            _save_out_conf(idx, cur_conf | halo_frame)
            _save_out_edge(idx, cur_edge | halo_frame)
            halo_added_px += int(np.count_nonzero(halo_frame))

    speck_report = prune_isolated_specks(
        total_frames=total,
        scenes=scenes,
        groups=all_groups,
        load_confirmed=_load_out_conf,
        save_confirmed=_save_out_conf,
        load_edge=_load_out_edge,
        save_edge=_save_out_edge,
        load_group=_load_out_group,
        save_group=_save_out_group,
        cfg=cont_cfg,
    )

    max_excl_frac = 0.0
    total_final_px = 0
    total_edge_px = 0
    for idx in range(total):
        fm = _load_out_conf(idx)
        eb = _load_out_edge(idx)
        excl_frac = float(np.count_nonzero(exclusion_mask(fm, ctx_cfg))) / float(pixels)
        if excl_frac > ctx_cfg.mask_fraction_limit:
            raise PipelineError(
                'MASK_UNCERTAIN',
                f'第{idx}帧确认水印排除区占比{excl_frac:.4f}超过20%保护上限',
                details=dict(frame=idx, exclusion_fraction=excl_frac),
            )
        max_excl_frac = max(max_excl_frac, excl_frac)
        total_final_px += int(np.count_nonzero(fm))
        total_edge_px += int(np.count_nonzero(eb))

    # 更新 detections.json 与 layers.json
    det_out = json.loads(json.dumps(det))
    det_out['edge_settings'] = asdict(e_cfg)
    det_out['faint_settings'] = asdict(f_cfg)
    det_out['reclassified_subtitles'] = reclassified_audit
    det_out['faint_summary'] = dict(
        track_summary=faint_res['track_summary'],
        static_scenes=faint_res['static_scenes'],
        added_pixels=total_faint_px,
    )
    (out_dir / 'detections.json').write_text(json.dumps(det_out, ensure_ascii=False, indent=2), encoding='utf-8')

    refined_files = [out_dir / 'masks' / p.name for p in files]
    write_manifest(out_dir, out_dir / 'detections.json', [out_dir / 'masks'], refined_files)
    VerifiedLayers(out_dir / 'layers.json', out_dir / 'detections.json')

    provenance = {
        '方法说明': 'v23通用掩膜精修（半透明轨迹对齐/跨镜头迁移 + 非对话OCR重分类 + 同色孔洞 + 时域连续性恢复 + 自适应低饱和外晕 + 单帧微噪点清理）',
        '总帧数': total,
        '原掩膜总像素': total_base_px,
        '新增淡字像素': total_faint_px,
        '新增同色孔洞像素': total_hole_px,
        '新增时域补全像素': continuity_report['新增像素'],
        '释放时域OCR字幕保护像素': continuity_report['释放字幕保护像素'],
        '新增自适应外晕像素': halo_added_px,
        '新增收边带像素': total_edge_px,
        '清理孤立微噪点像素': speck_report['清理像素'],
        '清理孤立微噪点连通域': speck_report['清理连通域'],
        '最终掩膜总像素': total_final_px,
        '重分类OCR字幕条目数': len(reclassified_audit),
        '最大单帧模型排除占比': round(max_excl_frac, 6),
        '全部帧满足20%保护': max_excl_frac <= ctx_cfg.mask_fraction_limit,
        '淡字帧记录数': len(faint_records),
        '同色孔洞记录': hole_records,
        '时域连续性修复': continuity_report,
        '重分类OCR明细': reclassified_audit,
        '淡字轨迹汇总': faint_res['track_summary'],
    }
    (out_dir / 'v17c-mask-provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    (out_dir / 'edge-refinement.json').write_text(
        json.dumps(
            dict(
                settings=asdict(e_cfg),
                faint_settings=asdict(f_cfg),
                added_pixels=total_edge_px,
                faint_added_pixels=total_faint_px,
                hole_added_pixels=total_hole_px,
                maximum_exclusion_fraction=max_excl_frac,
                edge_records=edge_records[:200],
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description='v17c 通用确认掩膜精修入口')
    parser.add_argument('--input', type=Path, required=True, help='基础确认掩膜目录（含 masks/detections.json/layers.json）')
    parser.add_argument('--output', type=Path, required=True, help='输出 v17c 确认掩膜目录')
    parser.add_argument('--frames', type=Path, default=None, help='源帧目录（若 --input 下不含 frames）')
    parser.add_argument('--config', type=Path, default=Path('config/v17c-edge-residue.json'))
    parser.add_argument('--faint-config', type=Path, default=None)
    args = parser.parse_args()

    e_cfg = EdgeSettings(**json.loads(args.config.read_text(encoding='utf-8')))
    f_cfg = FaintSettings(**json.loads(args.faint_config.read_text(encoding='utf-8'))) if args.faint_config else FaintSettings()
    summary = refine_v17c_masks(args.input, args.output, edge_cfg=e_cfg, faint_cfg=f_cfg, frames_dir=args.frames)
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ('同色孔洞记录', '时域连续性修复', '重分类OCR明细', '淡字轨迹汇总')
            },
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
