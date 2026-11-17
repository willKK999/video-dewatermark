"""离线复验脚本：基于旧运行的 staged-delta 与基线帧重建补画像素，用新口径 measure 复检并对照旧蓝色口径（只输出数字 JSON，不写图片）。"""
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
from worker.json_types import json_native
from worker.restoration import CleanReferences
from worker.roi_inpaint import crop
from worker.roi_spatial_layers import protection_mask, residue_summary
from worker.trusted_context import read_confirmed


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


def _recheck_task(payload):
    cv2.setNumThreads(1)
    run_dir, task, scenes, config_dict = payload
    root = Path(run_dir)
    cfg = EdgeSettings(**config_dict)
    taskid = int(task['任务'])
    box = task['ROI']
    a, b, c, d = box
    start, end = int(task['开始帧']), int(task['结束帧'])
    files = sorted((root / 'frames').glob('*.png'))
    if not files:
        raise ValueError(f'缺少冻结源帧目录: {root / "frames"}')
    scene = next((x, y) for x, y in scenes if x <= start < y)
    refs = np.unique(np.linspace(scene[0], scene[1] - 1, min(24, scene[1] - scene[0]), dtype=int))
    ref_images = [crop(cv2.imread(str(files[j])), box) for j in refs]
    ref_masks = [crop(read_confirmed(root / 'masks' / files[j].name), box, True) for j in refs]
    references = CleanReferences(ref_images, ref_masks)
    evidence_cache = {}
    attempt_rows = []

    for trial in task.get('尝试', []):
        k = int(trial['尝试'])
        cache = root / f'roi-{taskid:04d}-attempt-{k:02d}' / 'staged-delta'
        baseline = root / f'roi-{taskid:04d}-baseline'
        if not cache.is_dir() or not baseline.is_dir():
            attempt_rows.append(dict(
                任务=taskid,
                尝试=k,
                状态='缺少暂存或基线',
                有效施工帧数=0,
                有效施工像素=0,
                颜色候选=0,
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
            ))
            continue

        rows = []
        for path in sorted(cache.glob('*.npz')):
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

            source_roi = original[b:d, a:c]
            painted_roi = image[b:d, a:c]
            mask_roi = full[b:d, a:c]
            confirmed_full = read_confirmed(root / 'masks' / name)
            confirmed_roi = confirmed_full[b:d, a:c]
            band_path = root / 'edge-band' / name
            band_roi = (read_confirmed(band_path)[b:d, a:c] & mask_roi) if band_path.exists() else np.zeros(mask_roi.shape, np.uint8)
            protected_roi = protection_mask(root, name, shape)[b:d, a:c]

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
            )
            old_total, old_interior, old_band = _old_blue_counts(painted_roi, mask_roi, band_roi)
            rows.append(dict(
                frame=idx,
                旧口径蓝色候选=old_total,
                旧口径蓝色遮罩内=old_interior,
                旧口径蓝色收边带=old_band,
                **score,
            ))

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
        最佳尝试=task.get('最佳尝试'),
        尝试数=len(attempt_rows),
        判定计数=attempt_verdicts,
        帧判定计数=frame_verdicts,
        尝试=attempt_rows,
    )


def execute(args):
    started = time.monotonic()
    cv2.setNumThreads(1)
    if args.workers < 1:
        raise ValueError('workers 必须 >= 1')
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
    payloads = [(str(root), tasks_by_id[tid], scenes, config_dict) for tid in selected_ids]

    if args.workers == 1 or len(payloads) <= 1:
        _init_worker()
        task_results = [_recheck_task(p) for p in payloads]
    else:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=min(args.workers, len(payloads)), initializer=_init_worker) as pool:
            task_results = pool.map(_recheck_task, payloads)

    task_results.sort(key=lambda t: t['任务'])

    attempt_verdicts = {'通过': 0, 'RESIDUE_FOUND': 0, 'RESIDUE_UNCERTAIN': 0, '跳过': 0}
    frame_verdicts = {'通过': 0, 'RESIDUE_FOUND': 0, 'RESIDUE_UNCERTAIN': 0}
    sum_keys = [
        '有效施工帧数', '有效施工像素',
        '颜色候选', '遮罩内颜色候选', '收边带颜色候选',
        '时域证实背景本色', '外环证实背景本色', '背景本色证实', '外环证实但时域反证',
        '确认残留', '遮罩内确认残留', '收边带确认残留',
        '待目检', '遮罩内待目检', '收边带待目检',
        '未解释候选', '遮罩内残留', '收边带残留',
        '无模板遮罩内像素', '无模板收边带像素',
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
    print(
        f"离线复验完成：任务={selected_ids} 已复验尝试={len(checked_attempts)} "
        f"尝试判定计数={json.dumps(attempt_verdicts, ensure_ascii=False)} "
        f"帧判定计数={json.dumps(frame_verdicts, ensure_ascii=False)} "
        f"旧口径蓝色候选={totals['旧口径蓝色候选']} 颜色候选={totals['颜色候选']} "
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
    return parser


if __name__ == '__main__':
    execute(build_parser().parse_args())
