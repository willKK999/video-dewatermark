"""v17c 离线全量复验脚本（仅生成数值 JSON 报告，不读取/打印图像内容）。"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from worker.edge_residue import EdgeSettings, measure
from worker.faint_watermark import FaintSettings, dialogue_row_mask
from worker.matte import temporal_detail
from worker.restoration import CleanReferences, restore_orthogonal_luma_detail
from worker.roi_inpaint import ROISettings, crop as roi_crop
from worker.roi_spatial_layers import (
    _scene_static_highpass,
    _scene_track_highpass,
    protection_mask,
    residue_summary,
    seam_variants,
)
from worker.spatial_roi import SpatialSettings, plan as spatial_plan
from worker.trusted_context import ContextSettings, exclusion_mask, read_confirmed
from worker.vision import windows


def _eval_faint_coverage(v17b_dir: Path, v17c_dir: Path, frames_dir: Path, diag_path: Path | None) -> dict:
    """(1) 对比两处重点淡字区域（第157帧、第411帧右半画面）在 v17b 与 v17c 掩膜中的覆盖变化，并核验真实对话字幕零丢失。"""
    files = sorted((v17b_dir / 'masks').glob('*.png'))
    det_c = json.loads((v17c_dir / 'detections.json').read_text(encoding='utf-8'))
    diag = json.loads(diag_path.read_text(encoding='utf-8')) if diag_path and diag_path.exists() else None

    probes = {}
    for fidx in (157, 411):
        fname = f'{fidx:06d}.png'
        bgr = cv2.imread(str(frames_dir / fname))
        h, w = bgr.shape[:2]
        half_x = w // 2
        m_b = cv2.imread(str(v17b_dir / 'masks' / fname), 0) > 0
        m_c = cv2.imread(str(v17c_dir / 'masks' / fname), 0) > 0
        sub_b = cv2.imread(str(v17b_dir / 'subtitle-protection' / fname), 0) > 0
        sub_c = cv2.imread(str(v17c_dir / 'subtitle-protection' / fname), 0) > 0
        scene_p = cv2.imread(str(v17c_dir / 'scene-protection' / fname), 0) > 0
        from worker.edge_residue import highlight_guard
        hg = highlight_guard(bgr)
        gf_p = v17c_dir / 'group-objects/group-faint' / fname
        gf = (cv2.imread(str(gf_p), 0) > 0) if gf_p.exists() else np.zeros((h, w), dtype=bool)

        # 使用诊断结果精确外接框或右半区新增淡字评估覆盖率
        diag_key = f'第{fidx}帧右半画面'
        cand_union = np.zeros((h, w), dtype=bool)
        if diag and diag_key in diag:
            for cc in diag[diag_key].get('分块明细', []):
                cov = cc.get('覆盖', {})
                if cov.get('masks', {}).get('比例', 0.0) < 0.99:
                    bx = cov['外接框']
                    cand_union[bx[1] : bx[3], bx[0] : bx[2]] = True
        gf_right = gf.copy()
        gf_right[:, :half_x] = False
        target_mask = gf_right.copy()
        target_box_list = []
        if np.any(cand_union):
            from worker.faint_watermark import signed_highpass

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hp_pos = (signed_highpass(gray, 4.0) >= 4.0) & cand_union
            if np.any(hp_pos):
                target_mask = hp_pos | gf_right
            ys_c, xs_c = np.where(cand_union)
            target_box_list = [int(xs_c.min()), int(ys_c.min()), int(xs_c.max()) + 1, int(ys_c.max()) + 1]

        t_n = int(np.count_nonzero(target_mask))
        blocked_guard = scene_p | hg
        eligible_mask = target_mask & (~blocked_guard) & (~sub_c)
        e_n = int(np.count_nonzero(eligible_mask))
        cov_b = int(np.count_nonzero(target_mask & m_b))
        cov_c = int(np.count_nonzero(target_mask & m_c))
        cov_eligible_c = int(np.count_nonzero(eligible_mask & m_c))
        sub_block_b = int(np.count_nonzero(target_mask & sub_b))
        sub_block_c = int(np.count_nonzero(target_mask & sub_c))
        probes[f'frame_{fidx}_right'] = dict(
            frame=fidx,
            诊断候选总外接框=target_box_list,
            目标高通像素数=t_n,
            其中受场景与高亮保护像素=int(np.count_nonzero(target_mask & blocked_guard)),
            其中非保护可覆盖淡字像素=e_n,
            group_faint右半新增像素=int(np.count_nonzero(gf_right)),
            v17b_masks覆盖像素=cov_b,
            v17b_masks总覆盖比例=round(cov_b / max(1, t_n), 4),
            v17c_masks覆盖像素=cov_c,
            v17c_masks总覆盖比例=round(cov_c / max(1, t_n), 4),
            v17c可覆盖淡字覆盖比例=round(cov_eligible_c / max(1, e_n), 4),
            v17b字幕保护阻挡像素=sub_block_b,
            v17c字幕保护阻挡像素=sub_block_c,
        )

    # 验证整段全集真实对话字幕（detect_text_rows 且原受 subtitle-protection 保护）零受损
    reclassified = det_c.get('reclassified_subtitles', [])
    dlg_loss_px = 0
    for fidx in sorted({int(rec['frame']) for rec in reclassified}):
        fname = f'{fidx:06d}.png'
        bgr = cv2.imread(str(frames_dir / fname))
        dlg = dialogue_row_mask(bgr)
        sub_b = cv2.imread(str(v17b_dir / 'subtitle-protection' / fname), 0) > 0
        sub_c = cv2.imread(str(v17c_dir / 'subtitle-protection' / fname), 0) > 0
        dlg_loss_px += int(np.count_nonzero(dlg & sub_b & (~sub_c)))

    return dict(
        两处重点淡字覆盖对比=probes,
        整集重分类孤立OCR数=len(reclassified),
        真实对话字幕行保护丢失像素=dlg_loss_px,
    )


def _eval_mask_changes_and_tasks(
    v17b_dir: Path,
    v17c_dir: Path,
    roi_report_path: Path,
) -> tuple[dict, list[dict]]:
    """(3) 对比 v17b 与 v17c 全集掩膜面积、20%保护、收边带变化，并精确规划受影响任务列表。"""
    files = sorted((v17b_dir / 'masks').glob('*.png'))
    det_b = json.loads((v17b_dir / 'detections.json').read_text(encoding='utf-8'))
    rep = json.loads(roi_report_path.read_text(encoding='utf-8'))
    ctx_cfg = ContextSettings()

    roi_cfg = ROISettings(**rep['配置'])
    spatial = SpatialSettings(**rep['空间配置'])
    win_frames = int(rep.get('窗口', 36))
    ov_frames = int(rep.get('重叠', 8))
    from worker.hls_settings import HLSSettings

    hls = HLSSettings.from_env()
    hls = HLSSettings(
        window_frames=win_frames,
        overlap_frames=ov_frames,
        overlap_mae_limit=hls.overlap_mae_limit,
        mask_fraction_limit=hls.mask_fraction_limit,
    )

    m_b_list, m_c_list = [], []
    sum_b = sum_c = sum_eb_b = sum_eb_c = sum_gf = 0
    max_mask_b = max_mask_c = max_excl_b = max_excl_c = 0.0
    sub_b_sum = sub_c_sum = 0
    hole_frames_120_124 = {}

    for i, fp in enumerate(files):
        fn = fp.name
        mb = cv2.imread(str(v17b_dir / 'masks' / fn), 0)
        mc = cv2.imread(str(v17c_dir / 'masks' / fn), 0)
        ebb = cv2.imread(str(v17b_dir / 'edge-band' / fn), 0)
        ebc = cv2.imread(str(v17c_dir / 'edge-band' / fn), 0)
        sb = cv2.imread(str(v17b_dir / 'subtitle-protection' / fn), 0)
        sc = cv2.imread(str(v17c_dir / 'subtitle-protection' / fn), 0)
        gf_p = v17c_dir / 'group-objects/group-faint' / fn
        gf = cv2.imread(str(gf_p), 0) if gf_p.exists() else np.zeros_like(mb)

        m_b_list.append(mb)
        m_c_list.append(mc)
        px = mb.size
        nb = int(np.count_nonzero(mb))
        nc = int(np.count_nonzero(mc))
        sum_b += nb
        sum_c += nc
        sum_eb_b += int(np.count_nonzero(ebb))
        sum_eb_c += int(np.count_nonzero(ebc))
        sum_gf += int(np.count_nonzero(gf))
        sub_b_sum += int(np.count_nonzero(sb))
        sub_c_sum += int(np.count_nonzero(sc))
        max_mask_b = max(max_mask_b, nb / px)
        max_mask_c = max(max_mask_c, nc / px)
        max_excl_b = max(max_excl_b, float(np.count_nonzero(exclusion_mask(mb, ctx_cfg))) / px)
        max_excl_c = max(max_excl_c, float(np.count_nonzero(exclusion_mask(mc, ctx_cfg))) / px)

        if 118 <= i <= 128:
            g8b = cv2.imread(str(v17b_dir / 'group-objects/group-8' / fn), 0) > 0
            g8c = cv2.imread(str(v17c_dir / 'group-objects/group-8' / fn), 0) > 0
            diff_g8 = g8c & (~g8b)
            if np.any(diff_g8):
                ys, xs = np.where(diff_g8)
                hole_frames_120_124[str(i)] = dict(
                    补全像素=int(np.count_nonzero(diff_g8)),
                    外接框=[int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                )

    from worker.watermark_groups import AnimationSettings, planned_windows

    anim_path = v17b_dir / 'animation-events.json'
    anim_plan = json.loads(anim_path.read_text(encoding='utf-8')) if anim_path.exists() else None
    anim_cfg = AnimationSettings(**anim_plan['settings']) if anim_plan else None

    def get_schedule(s0: int, s1: int):
        if anim_plan and anim_cfg:
            events = [e['interval'] for e in anim_plan['layers'].get('0', []) if s0 <= e['interval'][0] < s1]
            return planned_windows(s0, s1, hls.window_frames, hls.overlap_frames, events, anim_cfg)
        return list(windows(s0, s1, hls.window_frames, hls.overlap_frames))

    # 使用与 v17b 完全一致的 spatial_roi.plan 还原45个任务与施工域并检查施工域变动
    affected_tasks = []
    v17b_tasks = rep['任务']
    plan_tasks = []
    for scene_start, scene_end in det_b['scenes']:
        for start, end in get_schedule(scene_start, scene_end):
            footprint = np.bitwise_or.reduce(m_b_list[start:end])
            budget = max(r - l for l, r, _ in seam_variants(start, end, scene_start, scene_end, hls, roi_cfg))
            jobs = spatial_plan(footprint, budget, roi_cfg, spatial)
            for frag in jobs:
                tidx = len(plan_tasks) + 1
                own = frag['ownership'] > 0
                box = [int(v) for v in frag['box']]
                a, b, c, d = box
                add_px = 0
                del_px = 0
                faint_px = 0
                changed_frames = 0
                for j in range(start, end):
                    fn = files[j].name
                    ob = (m_b_list[j][b:d, a:c] > 0) & own[b:d, a:c]
                    nc = (m_c_list[j][b:d, a:c] > 0) & own[b:d, a:c]
                    gf_p = v17c_dir / 'group-objects/group-faint' / fn
                    if gf_p.exists():
                        gf_j = (cv2.imread(str(gf_p), 0)[b:d, a:c] > 0) & own[b:d, a:c]
                    else:
                        gf_j = np.zeros_like(ob)
                    d_add = int(np.count_nonzero(nc & (~ob)))
                    d_del = int(np.count_nonzero(ob & (~nc)))
                    if d_add or d_del:
                        changed_frames += 1
                    add_px += d_add
                    del_px += d_del
                    faint_px += int(np.count_nonzero(gf_j))

                plan_tasks.append(
                    dict(
                        task_idx=tidx,
                        scene_start=int(scene_start),
                        scene_end=int(scene_end),
                        start=int(start),
                        end=int(end),
                        box=box,
                        ownership=frag['ownership'],
                    )
                )
                orig_t = v17b_tasks[tidx - 1]
                # 检查新增掩膜是否落在任务框 ROI 之内即使超出了原 own（如旁边新增 group-faint！）
                roi_box_add = 0
                for j in range(start, end):
                    mb_r = m_b_list[j][b:d, a:c] > 0
                    mc_r = m_c_list[j][b:d, a:c] > 0
                    roi_box_add += int(np.count_nonzero(mc_r & (~mb_r)))
                if add_px > 0 or del_px > 0 or roi_box_add > 0:
                    affected_tasks.append(
                        dict(
                            任务序号=tidx,
                            原v17b通过=bool(orig_t.get('通过')),
                            原v17b判定=orig_t.get('残留判定') or ('通过' if orig_t.get('通过') else orig_t.get('错误码')),
                            帧区间=[int(start), int(end)],
                            ROI=box,
                            归属域新增像素=add_px,
                            归属域减少收边像素=del_px,
                            ROI框内净新增像素=roi_box_add,
                            新增淡字像素=faint_px,
                            变化帧数=changed_frames,
                        )
                    )

    # 也检查用 v17c 掩膜重新执行 spatial_roi.plan 会产生多少个任务，及 ROI 布局是否一致
    new_plan_count = 0
    for scene_start, scene_end in det_b['scenes']:
        for start, end in get_schedule(scene_start, scene_end):
            footprint = np.bitwise_or.reduce(m_c_list[start:end])
            budget = max(r - l for l, r, _ in seam_variants(start, end, scene_start, scene_end, hls, roi_cfg))
            new_plan_count += len(spatial_plan(footprint, budget, roi_cfg, spatial))

    passed_changed = [t['任务序号'] for t in affected_tasks if t['原v17b通过']]
    mask_report = dict(
        总帧数=len(files),
        v17b掩膜总像素=sum_b,
        v17c掩膜总像素=sum_c,
        掩膜净变化像素=sum_c - sum_b,
        掩膜变化比例=round((sum_c - sum_b) / max(1, sum_b), 6),
        v17b收边带总像素=sum_eb_b,
        v17c收边带总像素=sum_eb_c,
        收边带减少像素=sum_eb_b - sum_eb_c,
        收边带变化比例=round((sum_eb_c - sum_eb_b) / max(1, sum_eb_b), 4),
        group_faint总像素=sum_gf,
        字幕保护v17b总像素=sub_b_sum,
        字幕保护v17c总像素=sub_c_sum,
        字幕保护释放像素=sub_b_sum - sub_c_sum,
        v17b单帧最大掩膜占比=round(max_mask_b, 6),
        v17c单帧最大掩膜占比=round(max_mask_c, 6),
        v17b单帧最大排除占比=round(max_excl_b, 6),
        v17c单帧最大排除占比=round(max_excl_c, 6),
        v17c满足每层20百分比保护=max_excl_c <= ctx_cfg.mask_fraction_limit,
        公告板段118至128帧group8同色孔洞补全=hole_frames_120_124,
        v17b任务总数=len(plan_tasks),
        v17c新规划任务总数=new_plan_count,
        预计受影响任务数=len(affected_tasks),
        原v17b已通过但掩膜变化任务数=len(passed_changed),
        原v17b已通过但掩膜变化任务序号=passed_changed,
        建议重跑策略='整块重跑（因为group-faint与收边收窄改动覆盖原已通过任务的施工与上下文域）' if passed_changed else '可只重跑未过任务',
        受影响任务明细=affected_tasks,
    )
    return mask_report, plan_tasks


def _eval_45_tasks_on_v17b(
    v17b_masks_dir: Path,
    v17b_run_dir: Path,
    roi_report_path: Path,
    plan_tasks: list[dict],
) -> dict:
    """(2) 在 v17b 已有 roi-layer-00 结果上，用 v17c 局部干净供体 + 新 measure 复检全部45个任务。"""
    files = sorted((v17b_masks_dir / 'masks').glob('*.png'))
    rep = json.loads(roi_report_path.read_text(encoding='utf-8'))
    v17b_tasks = rep['任务']
    edge_cfg = EdgeSettings(shape_radius=2)  # 按 v17b 已有结果的输入掩膜与收边带复检！
    shape = cv2.imread(str(v17b_run_dir / 'frames' / files[0].name)).shape[:2]

    # 轻量 LRU 缓存帧和掩膜避免重复读盘
    src_cache: OrderedDict[int, np.ndarray] = OrderedDict()
    pnt_cache: OrderedDict[int, np.ndarray] = OrderedDict()
    msk_cache: OrderedDict[int, np.ndarray] = OrderedDict()
    bnd_cache: OrderedDict[int, np.ndarray] = OrderedDict()
    prt_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get_src(i: int) -> np.ndarray:
        if i not in src_cache:
            src_cache[i] = cv2.imread(str(v17b_run_dir / 'frames' / files[i].name))
            if len(src_cache) > 260:
                src_cache.popitem(last=False)
        return src_cache[i]

    def get_pnt(i: int) -> np.ndarray:
        if i not in pnt_cache:
            pnt_cache[i] = cv2.imread(str(v17b_run_dir / 'roi-layer-00' / files[i].name))
            if len(pnt_cache) > 260:
                pnt_cache.popitem(last=False)
        return pnt_cache[i]

    def get_msk(i: int) -> np.ndarray:
        if i not in msk_cache:
            msk_cache[i] = read_confirmed(v17b_masks_dir / 'masks' / files[i].name)
            if len(msk_cache) > 260:
                msk_cache.popitem(last=False)
        return msk_cache[i]

    def get_bnd(i: int) -> np.ndarray:
        if i not in bnd_cache:
            p = v17b_masks_dir / 'edge-band' / files[i].name
            bnd_cache[i] = read_confirmed(p) if p.exists() else np.zeros(shape, dtype=np.uint8)
            if len(bnd_cache) > 260:
                bnd_cache.popitem(last=False)
        return bnd_cache[i]

    def get_prt(i: int) -> np.ndarray:
        if i not in prt_cache:
            # 按实际补画 run 目录读取保护掩膜，与 v17b 运行内复检及 v17b_offline_recheck 口径一致
            prt_cache[i] = protection_mask(v17b_run_dir, files[i].name, shape)
            if len(prt_cache) > 260:
                prt_cache.popitem(last=False)
        return prt_cache[i]

    task_results = []
    verdict_before = {'通过': 0, 'RESIDUE_FOUND': 0, 'RESIDUE_UNCERTAIN': 0}
    verdict_after = {'通过': 0, 'RESIDUE_FOUND': 0, 'RESIDUE_UNCERTAIN': 0}
    total_before_res = total_after_res = 0
    total_before_pend = total_after_pend = 0
    total_false_excluded_t_proof = 0
    total_dense_frames_used = 0

    for t in plan_tasks:
        tidx = t['task_idx']
        orig_t = v17b_tasks[tidx - 1]
        s0, s1 = t['scene_start'], t['scene_end']
        start, end = t['start'], t['end']
        box = t['box']
        a, b, c, d = box
        own = t['ownership'] > 0
        own_crop = own[b:d, a:c]

        orig_status = '通过' if orig_t.get('通过') else (orig_t.get('残留判定') or orig_t.get('错误码') or 'RESIDUE_UNCERTAIN')
        if orig_status not in verdict_before:
            orig_status = 'RESIDUE_FOUND' if orig_t.get('错误码') == 'RESIDUE_FOUND' else 'RESIDUE_UNCERTAIN'
        verdict_before[orig_status] += 1

        orig_last = orig_t['尝试'][-1] if orig_t.get('尝试') else {}
        orig_sum = orig_last.get('残留汇总', {})
        b_res_cnt = int(orig_sum.get('确认残留', 0))
        b_pend_cnt = int(orig_sum.get('待目检', 0))
        total_before_res += b_res_cnt
        total_before_pend += b_pend_cnt

        base_refs = np.unique(np.linspace(s0, s1 - 1, min(24, s1 - s0), dtype=int))
        # 构造最多40帧窗口加密参考列表
        extra_grid = np.linspace(max(s0, start - 12), min(s1 - 1, end + 15), 32, dtype=int)
        ref_ids = sorted(dict.fromkeys([int(x) for x in base_refs] + [int(x) for x in extra_grid]))[:40]
        cr_holder: list[CleanReferences] = []

        def get_cr() -> CleanReferences:
            if not cr_holder:
                cr_holder.append(
                    CleanReferences(
                        [roi_crop(get_src(r), box) for r in ref_ids],
                        [roi_crop(get_msk(r), box, True) for r in ref_ids],
                    )
                )
            return cr_holder[0]

        scores = []
        dense_used_task = 0
        frame_details_special = []
        for j in range(start, end):
            mk_full = get_msk(j)
            active = (mk_full[b:d, a:c] > 0) & own_crop
            if not np.any(active):
                continue
            active_u8 = active.astype(np.uint8) * 255
            band_c = get_bnd(j)[b:d, a:c] & active_u8
            src_c = roi_crop(get_src(j), box)[: d - b, : c - a]
            pnt_c = get_pnt(j)[b:d, a:c]
            conf_c = roi_crop(mk_full, box, True)[: d - b, : c - a]
            prot_c = get_prt(j)[b:d, a:c]

            # 先快速初筛无时域参考是否存在颜色候选或无模板组；无任何候选且无模板则时域不会改判
            sc0 = measure(src_c, pnt_c, active_u8, band_c, edge_cfg, confirmed=conf_c, protected=prot_c)
            if (
                sc0.get('颜色候选', 0) == 0
                and sc0.get('无模板遮罩内像素', 0) == 0
                and sc0.get('无模板收边带像素', 0) == 0
                and sc0.get('判定') == '通过'
            ):
                sc_final = sc0
            else:
                cr = get_cr()
                src_crop_pad = roi_crop(get_src(j), box)
                msk_crop_pad = roi_crop(mk_full, box, True)
                ev = cr.reference_evidence(src_crop_pad, msk_crop_pad, confirmed=msk_crop_pad)
                ev = {k: (v[: d - b, : c - a] if hasattr(v, 'shape') else v) for k, v in ev.items()}
                sc_final = measure(src_c, pnt_c, active_u8, band_c, edge_cfg, reference_evidence=ev, confirmed=conf_c, protected=prot_c)
                if sc_final.get('判定') in ('RESIDUE_FOUND', 'RESIDUE_UNCERTAIN'):
                    d0, d1 = max(s0, j - 18), min(s1, j + 19)
                    d_ids = [r for r in range(d0, d1, 2) if r != j]
                    if len(d_ids) >= 6:
                        dense_cr = CleanReferences(
                            [roi_crop(get_src(r), box) for r in d_ids],
                            [roi_crop(get_msk(r), box, True) for r in d_ids],
                        )
                        dev = dense_cr.reference_evidence(src_crop_pad, msk_crop_pad, confirmed=msk_crop_pad)
                        dev = {k: (v[: d - b, : c - a] if hasattr(v, 'shape') else v) for k, v in dev.items()}
                        sc_dense = measure(
                            src_c, pnt_c, active_u8, band_c, edge_cfg, reference_evidence=dev, confirmed=conf_c, protected=prot_c
                        )
                        if (sc_dense.get('确认残留', 0), sc_dense.get('待目检', 0)) < (
                            sc_final.get('确认残留', 0),
                            sc_final.get('待目检', 0),
                        ):
                            sc_final = sc_dense
                            dense_used_task += 1

            scores.append(dict(frame=j, **sc_final))
            if tidx in (3, 4, 6, 7) and (102 <= j <= 127 or (tidx == 6 and 141 <= j <= 167)):
                frame_details_special.append(
                    dict(
                        frame=j,
                        判定=sc_final['判定'],
                        确认残留=sc_final['确认残留'],
                        待目检=sc_final['待目检'],
                        时域证实背景本色=sc_final['时域证实背景本色'],
                        外环证实背景本色=sc_final['外环证实背景本色'],
                    )
                )

        sm = residue_summary(scores)
        new_status = sm['判定'] if scores else '通过'
        if new_status not in verdict_after:
            new_status = 'RESIDUE_UNCERTAIN'
        verdict_after[new_status] += 1
        total_after_res += int(sm.get('确认残留', 0))
        total_after_pend += int(sm.get('待目检', 0))
        total_false_excluded_t_proof += int(sm.get('时域证实背景本色', 0))
        total_dense_frames_used += dense_used_task

        row = dict(
            任务序号=tidx,
            帧区间=[start, end],
            ROI=box,
            v17b状态=orig_status,
            v17c状态=new_status,
            v17b确认残留=b_res_cnt,
            v17c确认残留=int(sm.get('确认残留', 0)),
            v17b待目检=b_pend_cnt,
            v17c待目检=int(sm.get('待目检', 0)),
            时域误报排除像素=int(sm.get('时域证实背景本色', 0)),
            外环证实像素=int(sm.get('外环证实背景本色', 0)),
            确认残留帧=sm.get('确认残留帧', []),
            密集参考复核帧数=dense_used_task,
        )
        if frame_details_special:
            row['重点帧段明细'] = frame_details_special
        task_results.append(row)

    # 汇总任务6与任务3/4/7专项焦点
    focus = {}
    for row in task_results:
        tid = row['任务序号']
        if tid in (3, 4, 6, 7):
            sub_rows = row.get('重点帧段明细', [])
            if tid == 6:
                p_frames = [r for r in sub_rows if 141 <= r['frame'] <= 158]
                b_frames = [r for r in sub_rows if 102 <= r['frame'] <= 127]
                focus['任务6'] = dict(
                    整体=dict(
                        v17b状态=row['v17b状态'],
                        v17c状态=row['v17c状态'],
                        v17b确认残留=row['v17b确认残留'],
                        v17c确认残留=row['v17c确认残留'],
                        v17b待目检=row['v17b待目检'],
                        v17c待目检=row['v17c待目检'],
                        时域误报排除像素=row['时域误报排除像素'],
                    ),
                    蓝紫P图案141至158帧=dict(
                        帧数=len(p_frames),
                        v17c总确认残留=sum(r['确认残留'] for r in p_frames),
                        v17c总时域证实=sum(r['时域证实背景本色'] for r in p_frames),
                        未清零帧=[r for r in p_frames if r['确认残留'] > 0],
                    ),
                    公告板102至127帧=dict(
                        帧数=len(b_frames),
                        v17c总确认残留=sum(r['确认残留'] for r in b_frames),
                        未清零帧=[r for r in b_frames if r['确认残留'] > 0],
                    ),
                )
            else:
                focus[f'任务{tid}_102至127帧'] = dict(
                    v17b状态=row['v17b状态'],
                    v17c状态=row['v17c状态'],
                    v17b确认残留=row['v17b确认残留'],
                    v17c确认残留=row['v17c确认残留'],
                    确认残留帧=[r['frame'] for r in sub_rows if r['确认残留'] > 0],
                    时域误报排除像素=row['时域误报排除像素'],
                )

    return dict(
        复检样本基线='v17b 已有成片输出 roi-layer-00（尚未按 v17c 掩膜与确认残留补修重新跑 GPU 模型）',
        任务总数=len(plan_tasks),
        v17b状态分布=verdict_before,
        v17c新口径复检状态分布=verdict_after,
        v17b累计确认残留像素=total_before_res,
        v17c新口径累计确认残留像素=total_after_res,
        v17b累计待目检像素=total_before_pend,
        v17c新口径累计待目检像素=total_after_pend,
        时域证据误报排除总像素=total_false_excluded_t_proof,
        密集邻近帧二次复核触发次数=total_dense_frames_used,
        重点关注任务分析=focus,
        全45任务逐项结果=[{k: v for k, v in r.items() if k != '重点帧段明细'} for r in task_results],
    )


def _eval_clarity_proxy(
    v17b_masks_dir: Path,
    v17c_masks_dir: Path,
    v17b_run_dir: Path,
    acceptance_json_path: Path,
    out_v17b_mp4: Path,
) -> dict:
    """(4) 计算清晰度代理指标与 51 帧样本清晰度保留对比。"""
    files = sorted((v17b_masks_dir / 'masks').glob('*.png'))
    acc = json.loads(acceptance_json_path.read_text(encoding='utf-8'))
    samples = acc['版本']['v17b']['清晰度']['全部样本']
    sample_frames = [int(s['frame']) for s in samples]
    det = json.loads((v17b_masks_dir / 'detections.json').read_text(encoding='utf-8'))
    edge_cfg = EdgeSettings(shape_radius=1)

    # A. 全集780帧清晰度代理指标：掩膜施工面积、收边新增像素、落入掩膜内的源图高细节像素数
    hi_detail_b = hi_detail_c = area_b = area_c = eb_b = eb_c = 0
    for fp in files:
        fn = fp.name
        src_g = cv2.imread(str(v17b_run_dir / 'frames' / fn), cv2.IMREAD_GRAYSCALE)
        gx = cv2.Sobel(src_g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(src_g, cv2.CV_32F, 0, 1, ksize=3)
        high_detail = np.hypot(gx, gy) > 40.0
        mb = cv2.imread(str(v17b_masks_dir / 'masks' / fn), 0) > 0
        mc = cv2.imread(str(v17c_masks_dir / 'masks' / fn), 0) > 0
        ebb = cv2.imread(str(v17b_masks_dir / 'edge-band' / fn), 0) > 0
        ebc = cv2.imread(str(v17c_masks_dir / 'edge-band' / fn), 0) > 0
        area_b += int(np.count_nonzero(mb))
        area_c += int(np.count_nonzero(mc))
        eb_b += int(np.count_nonzero(ebb))
        eb_c += int(np.count_nonzero(ebc))
        hi_detail_b += int(np.count_nonzero(mb & high_detail))
        hi_detail_c += int(np.count_nonzero(mc & high_detail))

    # B. 在51个清晰度抽样帧上对比：(i) v17b基线, (ii) 仅收窄边界退回源像素, (iii) +局部干净供体回填 + 正交亮度细节恢复
    cap_v17b = cv2.VideoCapture(str(out_v17b_mp4))
    scenes = det['scenes']
    scene_static_hps = {}
    scene_track_hps = {}
    src_paths = [v17b_run_dir / 'frames' / f.name for f in files]
    for s0, s1 in scenes:
        ids = np.linspace(s0, s1 - 1, min(16, s1 - s0), dtype=int)
        scene_static_hps[(s0, s1)] = _scene_static_highpass([cv2.imread(str(src_paths[i])) for i in ids])
        scene_track_hps.update(_scene_track_highpass(det, src_paths, s0, s1, (1280, 720)))

    r_base, r_narrow, r_full = [], [], []
    cr_restored_px_total = 0
    for f in sample_frames:
        s0 = cv2.imread(str(src_paths[f - 1]))
        s1 = cv2.imread(str(src_paths[f]))
        m_b0 = cv2.imread(str(v17b_masks_dir / 'masks' / f'{f-1:06d}.png'), 0) > 0
        m_b1 = cv2.imread(str(v17b_masks_dir / 'masks' / f'{f:06d}.png'), 0) > 0
        m_c0 = cv2.imread(str(v17c_masks_dir / 'masks' / f'{f-1:06d}.png'), 0) > 0
        m_c1 = cv2.imread(str(v17c_masks_dir / 'masks' / f'{f:06d}.png'), 0) > 0

        cap_v17b.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        _, p0 = cap_v17b.read()
        cap_v17b.set(cv2.CAP_PROP_POS_FRAMES, f)
        _, p1 = cap_v17b.read()

        both_b = (m_b0 & m_b1).astype(np.uint8) * 255
        uni_b = (m_b0 | m_b1).astype(np.uint8) * 255
        d_b = temporal_detail(s0, s1, p0, p1, both_b, uni_b)
        if d_b.get('measurable'):
            r_base.append(float(d_b['retention_ratio']))

        # 模拟 (ii)：在 v17c 施工掩膜外严格保持源像素（消除多余收边带模糊）
        sim0_n = p0.copy()
        sim1_n = p1.copy()
        sim0_n[~m_c0] = s0[~m_c0]
        sim1_n[~m_c1] = s1[~m_c1]
        both_c = (m_c0 & m_c1).astype(np.uint8) * 255
        uni_c = (m_c0 | m_c1).astype(np.uint8) * 255
        d_n = temporal_detail(s0, s1, sim0_n, sim1_n, both_b, uni_b)
        if d_n.get('measurable'):
            r_narrow.append(float(d_n['retention_ratio']))

        # 模拟 (iii)：+ 局部 CleanReferences 背景直填 + 排除水印笔画的正交亮度细节恢复
        sc_k0 = next((s0_i, s1_i) for s0_i, s1_i in scenes if s0_i <= f - 1 < s1_i)
        sc_k1 = next((s0_i, s1_i) for s0_i, s1_i in scenes if s0_i <= f < s1_i)
        ids0 = [r for r in range(max(sc_k0[0], f - 1 - 16), min(sc_k0[1], f - 1 + 17), 4) if r != f - 1]
        ids1 = [r for r in range(max(sc_k1[0], f - 16), min(sc_k1[1], f + 17), 4) if r != f]
        cr0 = CleanReferences(
            [cv2.imread(str(src_paths[r])) for r in ids0],
            [cv2.imread(str(v17c_masks_dir / 'masks' / f'{r:06d}.png'), 0) for r in ids0],
        )
        cr1 = CleanReferences(
            [cv2.imread(str(src_paths[r])) for r in ids1],
            [cv2.imread(str(v17c_masks_dir / 'masks' / f'{r:06d}.png'), 0) for r in ids1],
        )
        m_c0_u8 = m_c0.astype(np.uint8) * 255
        m_c1_u8 = m_c1.astype(np.uint8) * 255
        res0 = cr0.restore(s0, m_c0_u8, sim0_n, confirmed=m_c0_u8)
        cr_restored_px_total += int(cr0.stats.get('restored_pixels', 0))
        res1 = cr1.restore(s1, m_c1_u8, sim1_n, confirmed=m_c1_u8)
        cr_restored_px_total += int(cr1.stats.get('restored_pixels', 0))

        wm0 = scene_static_hps[sc_k0]
        if (f - 1) in scene_track_hps:
            t0_hp = scene_track_hps[f - 1]
            wm0 = np.where(np.abs(t0_hp) > np.abs(wm0), t0_hp, wm0)
        wm1 = scene_static_hps[sc_k1]
        if f in scene_track_hps:
            t1_hp = scene_track_hps[f]
            wm1 = np.where(np.abs(t1_hp) > np.abs(wm1), t1_hp, wm1)
        out0 = restore_orthogonal_luma_detail(s0, res0, m_c0_u8, wm0, edge_cfg)
        out1 = restore_orthogonal_luma_detail(s1, res1, m_c1_u8, wm1, edge_cfg)
        d_f = temporal_detail(s0, s1, out0, out1, both_b, uni_b)
        if d_f.get('measurable'):
            r_full.append(float(d_f['retention_ratio']))

    cap_v17b.release()

    return dict(
        全集代理指标=dict(
            v17b施工面积像素=area_b,
            v17c施工面积像素=area_c,
            施工面积变化比例=round((area_c - area_b) / max(1, area_b), 6),
            v17b收边带新增像素=eb_b,
            v17c收边带新增像素=eb_c,
            收边带缩减像素=eb_b - eb_c,
            v17b落入掩膜高细节源像素=hi_detail_b,
            v17c落入掩膜高细节源像素=hi_detail_c,
            高细节源像素变化=hi_detail_c - hi_detail_b,
        ),
        样本51帧清晰度对比=dict(
            v13e历史中位数=acc['版本']['v13e']['清晰度'].get('中位数') or float(np.median([s['retention_ratio'] for s in acc['版本']['v13e']['清晰度']['全部样本']])),
            v16历史中位数=acc['版本']['v16']['清晰度'].get('中位数') or float(np.median([s['retention_ratio'] for s in acc['版本']['v16']['清晰度']['全部样本']])),
            v17b实测中位数=round(float(np.median(r_base)), 4),
            v17b最低值=round(float(np.min(r_base)), 4),
            v17c仅缩窄收边模拟中位数=round(float(np.median(r_narrow)), 4),
            v17c仅缩窄收边模拟最低值=round(float(np.min(r_narrow)), 4),
            v17c完整策略模拟中位数=round(float(np.median(r_full)), 4),
            v17c完整策略模拟最低值=round(float(np.min(r_full)), 4),
            样本局部参考真实直填像素=cr_restored_px_total,
            说明='模拟保持了v17b既有模型内部区、仅反映缩窄收边带与局部干净参考/安全正交细节恢复带来的保守清晰度增益；实际重跑ProPainter时预期更高。',
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='v17c 离线综合复验与对比评估')
    parser.add_argument('--v17b-masks', type=Path, required=True)
    parser.add_argument('--v17c-masks', type=Path, required=True)
    parser.add_argument('--v17b-run', type=Path, required=True)
    parser.add_argument('--v17b-mp4', type=Path, required=True)
    parser.add_argument('--v17b-acceptance', type=Path, required=True)
    parser.add_argument('--faint-diag', type=Path, default=None)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()

    t0 = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    roi_rep = args.v17b_run / 'roi-report.json'

    r1 = _eval_faint_coverage(args.v17b_masks, args.v17c_masks, args.v17b_run / 'frames', args.faint_diag)
    (args.out_dir / '1-两处淡字覆盖对比.json').write_text(json.dumps(r1, ensure_ascii=False, indent=2), encoding='utf-8')

    r3, plan_tasks = _eval_mask_changes_and_tasks(args.v17b_masks, args.v17c_masks, roi_rep)
    (args.out_dir / '3-掩膜变化与受影响任务.json').write_text(json.dumps(r3, ensure_ascii=False, indent=2), encoding='utf-8')

    r2 = _eval_45_tasks_on_v17b(args.v17b_masks, args.v17b_run, roi_rep, plan_tasks)
    (args.out_dir / '2-全45任务新口径复检.json').write_text(json.dumps(r2, ensure_ascii=False, indent=2), encoding='utf-8')

    r4 = _eval_clarity_proxy(args.v17b_masks, args.v17c_masks, args.v17b_run, args.v17b_acceptance, args.v17b_mp4)
    (args.out_dir / '4-清晰度代理指标.json').write_text(json.dumps(r4, ensure_ascii=False, indent=2), encoding='utf-8')

    summary = dict(
        耗时秒=round(time.monotonic() - t0, 2),
        淡字覆盖=r1,
        全45任务复检摘要={k: v for k, v in r2.items() if k != '全45任务逐项结果'},
        掩膜与任务变化摘要={k: v for k, v in r3.items() if k != '受影响任务明细'},
        清晰度代理指标=r4,
    )
    (args.out_dir / 'offline-eval-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(耗时秒=summary['耗时秒'], 任务分布_b=r2['v17b状态分布'], 任务分布_c=r2['v17c新口径复检状态分布'], 最大排除=r3['v17c单帧最大排除占比'], 清晰度=r4['样本51帧清晰度对比']), ensure_ascii=False))


if __name__ == '__main__':
    main()
