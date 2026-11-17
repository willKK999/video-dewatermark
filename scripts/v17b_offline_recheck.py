"""离线复验脚本：基于旧运行的 staged-delta 或已通过层输出重建补画像素，用新口径 measure 复检并可输出四栏取证拼图。"""
import argparse
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import time
import cv2
import numpy as np
from worker.edge_residue import EdgeSettings, measure
from worker.hls_settings import HLSSettings
from worker.json_types import json_native
from worker.restoration import CleanReferences
from worker.roi_inpaint import ROISettings, crop
from worker.roi_spatial_layers import protection_mask, residue_summary, seam_variants
from worker.spatial_roi import SpatialSettings, plan
from worker.trusted_context import read_confirmed


PANEL_LABEL_HEIGHT = 28


def parse_tasks(spec, available=None):
    """解析任务编号范围（如 '1-8'、'1,3,5-6' 或 'all'）。"""
    text = (spec or '').strip()
    if not text:
        raise ValueError('任务范围不能为空')
    if text.lower() == 'all':
        if available is None:
            raise ValueError('all 仅在存在报告任务列表时可用')
        return sorted(int(t) for t in available)
    selected = []
    seen = set()
    for part in text.split(','):
        part = part.strip()
        if not part:
            raise ValueError(f'无效任务范围片段: {spec}')
        if '-' in part:
            bounds = part.split('-')
            if len(bounds) != 2 or not bounds[0].strip().isdigit() or not bounds[1].strip().isdigit():
                raise ValueError(f'无效任务区间: {part}')
            lo, hi = int(bounds[0]), int(bounds[1])
            if lo < 1 or lo > hi:
                raise ValueError(f'任务区间越界: {part}')
            for idx in range(lo, hi + 1):
                if idx not in seen:
                    seen.add(idx)
                    selected.append(idx)
        else:
            if not part.isdigit() or int(part) < 1:
                raise ValueError(f'无效任务编号: {part}')
            idx = int(part)
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)
    return selected


def _init_worker():
    cv2.setNumThreads(1)


def _old_blue_counts(painted_roi, mask_roi, band_roi):
    """旧口径对照：HSV H in [85, 130], S >= 60, V >= 60。"""
    hsv = cv2.cvtColor(painted_roi, cv2.COLOR_BGR2HSV)
    blue = (hsv[..., 0] >= 85) & (hsv[..., 0] <= 130) & (hsv[..., 1] >= 60) & (hsv[..., 2] >= 60)
    mask_bool = mask_roi > 0
    band_bool = band_roi > 0
    return (
        int(np.count_nonzero(blue & mask_bool)),
        int(np.count_nonzero(blue & mask_bool & ~band_bool)),
        int(np.count_nonzero(blue & band_bool)),
    )


def _select_best_attempt(task):
    """返回任务的最佳尝试编号：优先取通过的那次尝试，否则取 roi-report 记录的‘最佳尝试’。"""
    trials = task.get('尝试', [])
    passed = [int(t['尝试']) for t in trials if t.get('通过')]
    if passed:
        return passed[-1]
    if task.get('最佳尝试') is not None:
        best = int(task['最佳尝试'])
        if any(int(t['尝试']) == best for t in trials):
            return best
        raise ValueError(f"任务={task.get('任务')} 最佳尝试={best} 不在尝试列表中")
    raise ValueError(f"任务={task.get('任务')} 缺少通过或最佳尝试记录，无法确定目标尝试")


def _is_passed_trial(task, trial, trials):
    if trial.get('通过'):
        return True
    return (
        bool(task.get('通过'))
        and not trial.get('错误码')
        and bool(trials)
        and int(trial['尝试']) == int(trials[-1]['尝试'])
    )


def _resolve_layer_mask_dir(root, layers, layer_index):
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(f'图层索引越界: layer={layer_index}, layers={len(layers)}')
    layer_dir = Path(layers[layer_index]['directory'])
    if layer_dir.is_dir():
        return layer_dir
    if len(layers) == 1 and (root / 'masks').is_dir():
        return root / 'masks'
    raise ValueError(f'缺少图层 {layer_index} 掩膜目录: {layer_dir}')


def _load_plan_context(root, report):
    spatial_path = root / 'spatial-roi.json'
    if spatial_path.is_file():
        spatial_dict = json.loads(spatial_path.read_text())
    elif '空间配置' in report:
        spatial_dict = report['空间配置']
    else:
        raise ValueError('缺少空间ROI配置（spatial-roi.json 或 roi-report.json 空间配置），无法重建施工掩膜')

    profile_path = root.parent / 'profile.json'
    profile = json.loads(profile_path.read_text()) if profile_path.is_file() else {}
    if '配置' in report:
        roi_dict = report['配置']
    elif 'ROI' in profile:
        roi_dict = profile['ROI']
    else:
        raise ValueError('缺少ROI配置（roi-report.json 配置 或 profile.json），无法重建施工掩膜')

    layers_path = root / 'layers.json'
    if not layers_path.is_file():
        raise ValueError('缺少 layers.json，无法重建施工掩膜')
    layers = json.loads(layers_path.read_text())['layers']

    window_frames = int(report.get('窗口', profile.get('窗口', HLSSettings().window_frames)))
    overlap_frames = int(report.get('重叠', profile.get('重叠', HLSSettings().overlap_frames)))
    return (
        SpatialSettings(**spatial_dict),
        ROISettings(**roi_dict),
        HLSSettings(window_frames=window_frames, overlap_frames=overlap_frames),
        layers,
    )


def _reconstruct_task_ownership(root, report, scenes, files, task):
    """按 worker/roi_spatial_layers.py 同口径重建已通过任务的图层掩膜目录与 ownership。"""
    spatial, roi_cfg, hls_settings, layers = _load_plan_context(root, report)
    layer_num = int(task.get('图层', 0))
    layer_mask_dir = _resolve_layer_mask_dir(root, layers, layer_num)
    start, end = int(task['开始帧']), int(task['结束帧'])
    scene_start, scene_end = next(((int(x), int(y)) for x, y in scenes if int(x) <= start < int(y)), (None, None))
    if scene_start is None:
        raise ValueError(f"任务={task['任务']} 开始帧={start} 未命中任何镜头区间")
    if start < 0 or start >= end or end > len(files):
        raise ValueError(f"任务={task['任务']} 帧范围越界: {start}:{end} (总帧数={len(files)})")

    layer_masks = []
    for idx in range(start, end):
        mask_path = layer_mask_dir / files[idx].name
        if not mask_path.is_file():
            raise ValueError(f'缺少图层掩膜文件: {mask_path}')
        layer_masks.append(read_confirmed(mask_path))
    footprint = np.bitwise_or.reduce(layer_masks)
    if not np.any(footprint):
        raise ValueError(f"任务={task['任务']} 图层={layer_num} 窗口={start}:{end} 足迹为空")
    budget = max(r - l for l, r, _ in seam_variants(start, end, scene_start, scene_end, hls_settings, roi_cfg))
    jobs = plan(footprint, budget, roi_cfg, spatial)

    same_group = [
        t for t in report['任务']
        if int(t.get('图层', 0)) == layer_num and int(t['开始帧']) == start and int(t['结束帧']) == end
    ]
    if len(same_group) != len(jobs):
        raise ValueError(f"任务={task['任务']} 规划块数不匹配: 报告={len(same_group)} 重建={len(jobs)}")
    target_id = int(task['任务'])
    for item, job in zip(same_group, jobs):
        if job['box'] != item['ROI']:
            raise ValueError(f"任务={item['任务']} ROI 不匹配: 报告={item['ROI']} 重建={job['box']}")
        area = int(np.count_nonzero(job['ownership']))
        if '施工面积像素' in item and area != int(item['施工面积像素']):
            raise ValueError(f"任务={item['任务']} 施工面积像素不匹配: 报告={item['施工面积像素']} 重建={area}")
        if int(item['任务']) == target_id:
            return layer_mask_dir, job['ownership']
    raise ValueError(f'未能在规划结果中找到任务={target_id}')


def _iter_staged_delta_frames(root, files, task, k):
    """按 staged-delta + baseline 重建单次尝试各施工帧。"""
    taskid = int(task['任务'])
    start, end = int(task['开始帧']), int(task['结束帧'])
    a, b, c, d = task['ROI']
    cache = root / f'roi-{taskid:04d}-attempt-{k:02d}' / 'staged-delta'
    baseline = root / f'roi-{taskid:04d}-baseline'
    if not baseline.is_dir():
        raise ValueError(f'任务={taskid} 尝试={k} 缺少 baseline 目录: {baseline}')
    npz_files = sorted(cache.glob('*.npz'))
    if not npz_files:
        raise ValueError(f'任务={taskid} 尝试={k} staged-delta 目录为空: {cache}')

    for path in npz_files:
        idx = int(path.stem)
        if not (start <= idx < end):
            continue
        name = files[idx].name if 0 <= idx < len(files) and files[idx].stem == path.stem else f'{path.stem}.png'
        original = cv2.imread(str(root / 'frames' / name))
        image = cv2.imread(str(baseline / name))
        if original is None or image is None or original.shape != image.shape:
            raise ValueError(f'任务={taskid} 尝试={k} 帧={name} 源帧或基线无效')
        shape = original.shape[:2]
        with np.load(path, allow_pickle=False) as delta:
            indices = delta['indices']
            pixels = delta['pixels']
        if pixels.shape != (len(indices), 3) or len(np.unique(indices)) != len(indices):
            raise ValueError(f'任务={taskid} 尝试={k} 帧={name} 增量缓存形状或索引无效')
        if np.any((indices < 0) | (indices >= shape[0] * shape[1])):
            raise ValueError(f'任务={taskid} 尝试={k} 帧={name} 增量索引越界')
        if not indices.size:
            continue

        image.reshape(-1, 3)[indices] = pixels
        full = np.zeros(shape, np.uint8)
        full.ravel()[indices] = 255
        if full[:b].any() or full[d:].any() or full[b:d, :a].any() or full[b:d, c:].any():
            raise ValueError(f'任务={taskid} 尝试={k} 帧={name} 增量像素超出ROI边界')

        yield idx, name, original, image[b:d, a:c], full[b:d, a:c]


def _iter_passed_layer_frames(root, report, scenes, files, task, k):
    """对无 staged-delta 的已通过尝试，从 roi-layer-XX 与 masks&ownership 重建施工帧并核验 roi-coverage。"""
    taskid = int(task['任务'])
    layer_num = int(task.get('图层', 0))
    start, end = int(task['开始帧']), int(task['结束帧'])
    a, b, c, d = task['ROI']
    layer_out_dir = root / f'roi-layer-{layer_num:02d}'
    if not layer_out_dir.is_dir():
        raise ValueError(f'任务={taskid} 尝试={k} 已通过但缺少图层输出目录: {layer_out_dir}')
    coverage_dir = root / 'roi-coverage'
    if not coverage_dir.is_dir():
        raise ValueError(f'任务={taskid} 尝试={k} 缺少覆盖核验目录: {coverage_dir}')

    layer_mask_dir, ownership = _reconstruct_task_ownership(root, report, scenes, files, task)
    yielded = False
    for idx in range(start, end):
        name = files[idx].name
        original = cv2.imread(str(root / 'frames' / name))
        painted_full = cv2.imread(str(layer_out_dir / name))
        if original is None or painted_full is None or original.shape != painted_full.shape:
            raise ValueError(f'任务={taskid} 尝试={k} 帧={name} 源帧或图层输出无效: {layer_out_dir / name}')
        shape = original.shape[:2]
        layer_mask_path = layer_mask_dir / name
        coverage_path = coverage_dir / name
        if not layer_mask_path.is_file() or not coverage_path.is_file():
            raise ValueError(f'任务={taskid} 尝试={k} 帧={name} 缺少图层掩膜或 roi-coverage 文件')
        layer_mask = read_confirmed(layer_mask_path)
        covered = read_confirmed(coverage_path)
        if layer_mask.shape != shape or covered.shape != shape:
            raise ValueError(f'任务={taskid} 尝试={k} 帧={name} 掩膜与源帧几何不一致')
        active_full = layer_mask & ownership
        if np.any((active_full > 0) & (covered == 0)):
            raise ValueError(f'任务={taskid} 尝试={k} 帧={name} roi-coverage 未覆盖全部施工像素')
        if active_full[:b].any() or active_full[d:].any() or active_full[b:d, :a].any() or active_full[b:d, c:].any():
            raise ValueError(f'任务={taskid} 尝试={k} 帧={name} 施工像素超出ROI边界')
        mask_roi = active_full[b:d, a:c]
        if not mask_roi.any():
            continue
        yielded = True
        yield idx, name, original, painted_full[b:d, a:c], mask_roi
    if not yielded:
        raise ValueError(f'任务={taskid} 尝试={k} 已通过但施工区间内无有效施工像素')


def _write_panel_png(panels_dir, taskid, idx, source_roi, painted_roi, residue_mask, pending_mask, residue_px, pending_px):
    """生成四栏横向取证拼图（原图 | 补画 | 确认残留红涂 | 待目检黄涂）并校验输出尺寸。"""
    ch, cw = source_roi.shape[:2]
    p2 = painted_roi.copy()
    p2[residue_mask] = (0, 0, 255)
    p3 = painted_roi.copy()
    p3[pending_mask] = (0, 255, 255)
    body = np.hstack([source_roi, painted_roi, p2, p3])
    header = np.zeros((PANEL_LABEL_HEIGHT, 4 * cw, 3), np.uint8)
    scale = min(0.55, max(0.32, cw / 360.0))
    labels = [
        f'task={taskid:02d} source',
        f'frame={idx:06d} painted',
        f'residue={residue_px}',
        f'pending={pending_px}',
    ]
    for col, text in enumerate(labels):
        cv2.putText(
            header,
            text,
            (col * cw + 6, PANEL_LABEL_HEIGHT - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    canvas = np.vstack([header, body])
    expected_w = 4 * cw
    expected_h = ch + PANEL_LABEL_HEIGHT
    assert canvas.shape == (expected_h, expected_w, 3), f'拼图尺寸异常: {canvas.shape}'

    filename = f'task{taskid:02d}-frame{idx:06d}-residue{residue_px}-pending{pending_px}.png'
    out_path = panels_dir / filename
    if not cv2.imwrite(str(out_path), canvas):
        raise ValueError(f'拼图写出失败: {out_path}')
    saved = cv2.imread(str(out_path))
    if saved is None:
        raise ValueError(f'拼图读回校验失败: {out_path}')
    assert saved.shape[1] == expected_w and saved.shape[0] == expected_h, (
        f'拼图写盘尺寸不符: 期望=({expected_h}, {expected_w}) 实际={saved.shape[:2]}'
    )
    return dict(
        文件名=filename,
        任务=taskid,
        帧=idx,
        确认残留像素=residue_px,
        待目检像素=pending_px,
        宽=int(expected_w),
        高=int(expected_h),
        裁块宽=int(cw),
        裁块高=int(ch),
        标签条高度=PANEL_LABEL_HEIGHT,
    )


def _skipped_attempt_row(taskid, k, trial):
    return dict(
        任务=taskid,
        尝试=k,
        状态='缺少暂存或基线',
        有效施工帧数=0,
        有效施工像素=0,
        颜色候选=0,
        区别色候选数=0,
        遮罩内颜色候选=0,
        收边带颜色候选=0,
        时域证实背景本色=0,
        外环证实背景本色=0,
        背景本色证实=0,
        外环证实但时域反证=0,
        确认残留=0,
        遮罩内确认残留=0,
        收边带确认残留=0,
        待目检=0,
        遮罩内待目检=0,
        收边带待目检=0,
        未解释候选=0,
        遮罩内残留=0,
        收边带残留=0,
        无模板遮罩内像素=0,
        无模板收边带像素=0,
        同色不适用组数=0,
        同色不适用施工像素=0,
        单帧最大未解释=0,
        判定='跳过',
        判定计数={'通过': 0, 'RESIDUE_FOUND': 0, 'RESIDUE_UNCERTAIN': 0},
        旧口径蓝色候选=0,
        旧口径蓝色遮罩内=0,
        旧口径蓝色收边带=0,
        重叠最大差异=trial.get('重叠最大差异'),
        原运行错误码=trial.get('错误码'),
        新复验通过=False,
        帧=[],
    )


def _recheck_task(payload):
    cv2.setNumThreads(1)
    run_dir, report, task, scenes, config_dict, attempt_mode, panels_dir_str, panel_top = payload
    root = Path(run_dir)
    panels_dir = Path(panels_dir_str) if panels_dir_str is not None else None
    cfg = EdgeSettings(**config_dict)
    taskid = int(task['任务'])
    box = task['ROI']
    a, b, c, d = box
    start, end = int(task['开始帧']), int(task['结束帧'])
    files = sorted((root / 'frames').glob('*.png'))
    if not files:
        raise ValueError(f'缺少冻结源帧目录: {root / "frames"}')
    scene = next(((int(x), int(y)) for x, y in scenes if int(x) <= start < int(y)), None)
    if scene is None:
        raise ValueError(f'任务={taskid} 开始帧={start} 未命中任何镜头区间')
    refs = np.unique(np.linspace(scene[0], scene[1] - 1, min(24, scene[1] - scene[0]), dtype=int))
    ref_images = [crop(cv2.imread(str(files[j])), box) for j in refs]
    ref_masks = [crop(read_confirmed(root / 'masks' / files[j].name), box, True) for j in refs]
    references = CleanReferences(ref_images, ref_masks)
    evidence_cache = {}
    attempt_rows = []
    panel_records = []

    trials = task.get('尝试', [])
    need_best = attempt_mode == 'best' or panels_dir is not None
    best_attempt = _select_best_attempt(task) if need_best else None
    selected_trials = [t for t in trials if int(t['尝试']) == best_attempt] if attempt_mode == 'best' else trials

    for trial in selected_trials:
        k = int(trial['尝试'])
        cache = root / f'roi-{taskid:04d}-attempt-{k:02d}' / 'staged-delta'
        passed_trial = _is_passed_trial(task, trial, trials)
        collect_panels = panels_dir is not None and k == best_attempt

        if cache.is_dir():
            source_mode = 'staged-delta'
            frame_iter = _iter_staged_delta_frames(root, files, task, k)
        elif passed_trial:
            layer_num = int(task.get('图层', 0))
            source_mode = f'roi-layer-{layer_num:02d}'
            frame_iter = _iter_passed_layer_frames(root, report, scenes, files, task, k)
        elif trial.get('完整暂存') or attempt_mode == 'best' or collect_panels:
            raise ValueError(f'任务={taskid} 尝试={k} 缺少 staged-delta 且非已通过尝试，无法重建')
        else:
            attempt_rows.append(_skipped_attempt_row(taskid, k, trial))
            continue

        rows = []
        residue_ranks = []
        pending_ranks = []
        cached_frames = {}

        for idx, name, original, painted_roi, mask_roi in frame_iter:
            source_roi = original[b:d, a:c]
            confirmed_full = read_confirmed(root / 'masks' / name)
            confirmed_roi = confirmed_full[b:d, a:c]
            band_path = root / 'edge-band' / name
            band_roi = (read_confirmed(band_path)[b:d, a:c] & mask_roi) if band_path.exists() else np.zeros(mask_roi.shape, np.uint8)
            protected_roi = protection_mask(root, name, original.shape[:2])[b:d, a:c]

            if idx not in evidence_cache:
                source_padded = crop(original, box)
                union_padded = crop(confirmed_full, box, True)
                evidence_cache[idx] = {
                    key: value[:d - b, :c - a].copy()
                    for key, value in references.reference_evidence(source_padded, union_padded, confirmed=union_padded).items()
                }
            evidence = evidence_cache[idx]

            score = measure(
                source_roi,
                painted_roi,
                mask_roi,
                band_roi,
                cfg,
                reference_evidence=evidence,
                confirmed=confirmed_roi,
                protected=protected_roi,
                return_masks=collect_panels,
            )
            masks_dict = score.pop('_masks') if collect_panels else None
            old_total, old_interior, old_band = _old_blue_counts(painted_roi, mask_roi, band_roi)
            rows.append(dict(
                frame=idx,
                旧口径蓝色候选=old_total,
                旧口径蓝色遮罩内=old_interior,
                旧口径蓝色收边带=old_band,
                **score,
            ))

            if collect_panels:
                residue_px = int(score['确认残留'])
                pending_px = int(score['待目检'])
                if residue_px > 0 or pending_px > 0:
                    if residue_px > 0:
                        residue_ranks.append((-residue_px, idx))
                        residue_ranks.sort()
                        del residue_ranks[panel_top:]
                    if pending_px > 0:
                        pending_ranks.append((-pending_px, idx))
                        pending_ranks.sort()
                        del pending_ranks[panel_top:]
                    keep_ids = {item[1] for item in residue_ranks} | {item[1] for item in pending_ranks}
                    if idx in keep_ids:
                        cached_frames[idx] = (
                            source_roi.copy(),
                            painted_roi.copy(),
                            masks_dict['确认残留'],
                            masks_dict['待目检'],
                            residue_px,
                            pending_px,
                        )
                    for stale in list(cached_frames):
                        if stale not in keep_ids:
                            del cached_frames[stale]

        if collect_panels and cached_frames:
            for idx in sorted(cached_frames):
                src_crop, paint_crop, res_mask, pend_mask, residue_px, pending_px = cached_frames[idx]
                panel_records.append(
                    _write_panel_png(
                        panels_dir, taskid, idx, src_crop, paint_crop, res_mask, pend_mask, residue_px, pending_px
                    )
                )

        summary = residue_summary(rows)
        verdict_counts = {
            '通过': summary['通过帧数'],
            'RESIDUE_FOUND': len(summary['确认残留帧']),
            'RESIDUE_UNCERTAIN': len(summary['待目检帧']),
        }
        attempt_rows.append(dict(
            任务=taskid,
            尝试=k,
            状态='已复验' if rows else '无有效施工帧',
            重建来源=source_mode,
            重叠最大差异=trial.get('重叠最大差异'),
            原运行错误码=trial.get('错误码'),
            **summary,
            有效施工像素=sum(r['有效施工像素'] for r in rows),
            帧判定计数=verdict_counts,
            判定计数=verdict_counts,
            旧口径蓝色候选=sum(r['旧口径蓝色候选'] for r in rows),
            旧口径蓝色遮罩内=sum(r['旧口径蓝色遮罩内'] for r in rows),
            旧口径蓝色收边带=sum(r['旧口径蓝色收边带'] for r in rows),
            新复验通过=bool(rows) and summary['判定'] == '通过',
            帧=rows,
        ))

    attempt_verdicts = {'通过': 0, 'RESIDUE_FOUND': 0, 'RESIDUE_UNCERTAIN': 0, '跳过': 0}
    frame_verdicts = {'通过': 0, 'RESIDUE_FOUND': 0, 'RESIDUE_UNCERTAIN': 0}
    for row in attempt_rows:
        if row['状态'] == '已复验':
            attempt_verdicts[row['判定']] = attempt_verdicts.get(row['判定'], 0) + 1
            for key in frame_verdicts:
                frame_verdicts[key] += row['帧判定计数'][key]
        else:
            attempt_verdicts['跳过'] += 1

    return dict(
        任务=taskid,
        开始帧=start,
        结束帧=end,
        ROI=box,
        最佳尝试=task.get('最佳尝试') if task.get('最佳尝试') is not None else best_attempt,
        尝试数=len(attempt_rows),
        判定计数=attempt_verdicts,
        帧判定计数=frame_verdicts,
        尝试=attempt_rows,
        _panels=panel_records,
    )


def _slim_report(report):
    meta = {key: report[key] for key in ('空间配置', '配置', '窗口', '重叠') if key in report}
    meta['任务'] = [
        {key: t[key] for key in ('任务', '图层', '开始帧', '结束帧', 'ROI', '施工面积像素') if key in t}
        for t in report['任务']
    ]
    return meta


def _slim_task(task):
    slim = {
        key: task[key]
        for key in ('任务', '图层', '开始帧', '结束帧', 'ROI', '施工面积像素', '最佳尝试', '通过', '采用未过结果')
        if key in task
    }
    slim['尝试'] = [
        {
            key: trial[key]
            for key in ('尝试', '重叠最大差异', '错误码', '通过', '完整暂存')
            if key in trial
        }
        for trial in task.get('尝试', [])
    ]
    return slim


def execute(args):
    started = time.monotonic()
    cv2.setNumThreads(1)
    if args.workers < 1:
        raise ValueError('workers 必须 >= 1')
    attempt_mode = getattr(args, 'attempt_mode', 'all')
    if attempt_mode not in {'all', 'best'}:
        raise ValueError(f'无效 attempt-mode: {attempt_mode}')
    panels_dir = getattr(args, 'panels_dir', None)
    panel_top = getattr(args, 'panel_top', 3)
    if panel_top < 1:
        raise ValueError('panel-top 必须 >= 1')
    if panels_dir is not None:
        panels_dir = Path(panels_dir)
        panels_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.run)
    cfg = EdgeSettings(**json.loads(Path(args.config).read_text()))
    report = json.loads((root / 'roi-report.json').read_text())
    detections = json.loads((root / 'detections.json').read_text())
    scenes = detections['scenes']
    tasks_by_id = {int(t['任务']): t for t in report['任务']}
    selected_ids = parse_tasks(args.tasks, available=tasks_by_id.keys())
    missing = [tid for tid in selected_ids if tid not in tasks_by_id]
    if missing:
        raise ValueError(f'任务编号不存在于报告中: {missing}')

    config_dict = asdict(cfg)
    report_meta = _slim_report(report)
    panels_dir_str = str(panels_dir) if panels_dir is not None else None
    payloads = [
        (str(root), report_meta, _slim_task(tasks_by_id[tid]), scenes, config_dict, attempt_mode, panels_dir_str, panel_top)
        for tid in selected_ids
    ]

    if args.workers == 1 or len(payloads) <= 1:
        _init_worker()
        task_results = [_recheck_task(p) for p in payloads]
    else:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=min(args.workers, len(payloads)), initializer=_init_worker) as pool:
            task_results = pool.map(_recheck_task, payloads)

    task_results.sort(key=lambda t: t['任务'])
    all_panels = []
    for t in task_results:
        all_panels.extend(t.pop('_panels', []))
    all_panels.sort(key=lambda p: (p['任务'], p['帧']))

    attempt_verdicts = {'通过': 0, 'RESIDUE_FOUND': 0, 'RESIDUE_UNCERTAIN': 0, '跳过': 0}
    frame_verdicts = {'通过': 0, 'RESIDUE_FOUND': 0, 'RESIDUE_UNCERTAIN': 0}
    sum_keys = [
        '有效施工帧数', '有效施工像素',
        '颜色候选', '区别色候选数', '遮罩内颜色候选', '收边带颜色候选',
        '时域证实背景本色', '外环证实背景本色', '背景本色证实', '外环证实但时域反证',
        '确认残留', '遮罩内确认残留', '收边带确认残留',
        '待目检', '遮罩内待目检', '收边带待目检',
        '未解释候选', '遮罩内残留', '收边带残留',
        '无模板遮罩内像素', '无模板收边带像素',
        '同色不适用组数', '同色不适用施工像素',
        '旧口径蓝色候选', '旧口径蓝色遮罩内', '旧口径蓝色收边带',
    ]
    totals = {k: 0 for k in sum_keys}
    max_single_frame = 0
    checked_attempts = []

    for t in task_results:
        for key in attempt_verdicts:
            attempt_verdicts[key] += t['判定计数'].get(key, 0)
        for key in frame_verdicts:
            frame_verdicts[key] += t['帧判定计数'].get(key, 0)
        for a in t['尝试']:
            if a['状态'] == '已复验':
                checked_attempts.append(a)
                for key in sum_keys:
                    totals[key] += a.get(key, 0)
                max_single_frame = max(max_single_frame, a.get('单帧最大未解释', 0))
            print(
                f"任务={a['任务']} 尝试={a['尝试']} 状态={a['状态']} 判定={a['判定']} "
                f"有效施工帧={a['有效施工帧数']} 颜色候选={a['颜色候选']}(遮罩内{a['遮罩内颜色候选']}/收边带{a['收边带颜色候选']}) "
                f"区别色候选数={a['区别色候选数']} 同色不适用组数={a['同色不适用组数']} 同色不适用施工像素={a['同色不适用施工像素']} "
                f"时域证实={a['时域证实背景本色']} 外环证实={a['外环证实背景本色']} "
                f"确认残留={a['确认残留']}(遮罩内{a['遮罩内确认残留']}/收边带{a['收边带确认残留']}) "
                f"待目检={a['待目检']}(遮罩内{a['遮罩内待目检']}/收边带{a['收边带待目检']}) "
                f"单帧最大={a['单帧最大未解释']} 判定计数={json.dumps(a['帧判定计数'], ensure_ascii=False)} "
                f"旧口径蓝色候选={a['旧口径蓝色候选']}",
                flush=True,
            )

    totals['单帧最大未解释'] = max_single_frame
    output_data = dict(
        run=str(root),
        config=config_dict,
        attempt_mode=attempt_mode,
        tasks=selected_ids,
        已复验尝试数=len(checked_attempts),
        判定计数=attempt_verdicts,
        帧判定计数=frame_verdicts,
        汇总=totals,
        任务=task_results,
        耗时秒=time.monotonic() - started,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_native(output_data), ensure_ascii=False, indent=2))

    if panels_dir is not None:
        for item in all_panels:
            assert item['宽'] == 4 * item['裁块宽'] and item['高'] == item['裁块高'] + item['标签条高度']
        manifest = dict(
            panels_dir=str(panels_dir),
            panel_top=panel_top,
            label_height=PANEL_LABEL_HEIGHT,
            总数=len(all_panels),
            panels=all_panels,
        )
        manifest_path = panels_dir / 'panels.json'
        manifest_path.write_text(json.dumps(json_native(manifest), ensure_ascii=False, indent=2))
        print(f'取证拼图完成：生成={len(all_panels)} 清单={manifest_path}', flush=True)

    print(
        f"离线复验完成：任务={selected_ids} 模式={attempt_mode} 已复验尝试={len(checked_attempts)} "
        f"尝试判定计数={json.dumps(attempt_verdicts, ensure_ascii=False)} "
        f"帧判定计数={json.dumps(frame_verdicts, ensure_ascii=False)} "
        f"旧口径蓝色候选={totals['旧口径蓝色候选']} 颜色候选={totals['颜色候选']} "
        f"区别色候选数={totals['区别色候选数']} 同色不适用组数={totals['同色不适用组数']} 同色不适用施工像素={totals['同色不适用施工像素']} "
        f"时域证实={totals['时域证实背景本色']} 外环证实={totals['外环证实背景本色']} "
        f"确认残留={totals['确认残留']} 待目检={totals['待目检']} "
        f"输出={out_path} 耗时秒={output_data['耗时秒']:.3f}",
        flush=True,
    )
    return output_data


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='旧运行 run 目录')
    parser.add_argument('--tasks', default='1-8', help='复验任务编号范围，如 1-8 或 1,3,5')
    parser.add_argument('--config', type=Path, default=Path('config/v17-edge-residue.json'), help='EdgeSettings 配置文件路径')
    parser.add_argument('--output', type=Path, required=True, help='输出汇总 JSON 路径')
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1), help='多进程 worker 数')
    parser.add_argument('--attempt-mode', choices=['all', 'best'], default='all', help='尝试复检模式：all 全部尝试；best 仅最佳或已通过尝试')
    parser.add_argument('--panels-dir', type=Path, default=None, help='可选四栏对比拼图输出目录')
    parser.add_argument('--panel-top', type=int, default=3, help='每任务按确认残留与待目检各自取前 N 帧生成拼图（默认 3）')
    return parser


if __name__ == '__main__':
    execute(build_parser().parse_args())
