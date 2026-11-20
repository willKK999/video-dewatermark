"""本地复放单个逐层ROI任务的CPU合成与残留复检阶段，并输出各子步骤耗时JSON用于定位热点。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from worker import edge_residue, restoration, roi_inpaint as roi, roi_spatial_layers as rsl
from worker.hls_settings import HLSSettings
from worker.spatial_roi import SpatialSettings, plan
from worker.trusted_context import ContextSettings, exclusion_mask, overlap_weight, read_confirmed, verify_from_confirmed
from worker.watermark_groups import AnimationSettings, planned_windows


class _TimedCv2:
    """轻量代理 OpenCV 调用以精确统计各算子耗时，不改变任何计算与返回结果。"""

    def __init__(self, real_cv2, bucket: dict[str, float], counts: dict[str, int], prefix: str):
        self._cv2 = real_cv2
        self._bucket = bucket
        self._counts = counts
        self._prefix = prefix

    def __getattr__(self, name: str):
        attr = getattr(self._cv2, name)
        if name in {
            'imread',
            'imwrite',
            'cvtColor',
            'GaussianBlur',
            'dilate',
            'morphologyEx',
            'distanceTransform',
            'boxFilter',
            'connectedComponents',
            'connectedComponentsWithStats',
            'goodFeaturesToTrack',
            'calcOpticalFlowPyrLK',
            'estimateAffinePartial2D',
            'warpAffine',
            'addWeighted',
            'copyMakeBorder',
            'matchTemplate',
            'inpaint',
        }:
            key = f'{self._prefix}.{name}'

            def wrapped(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return attr(*args, **kwargs)
                finally:
                    self._bucket[key] += time.perf_counter() - t0
                    self._counts[key] += 1

            return wrapped
        return attr


def _locate_task(
    masks_dir: Path,
    baseline_path: Path,
    spatial_cfg_path: Path,
    trusted_cfg_path: Path,
    target_task: int,
):
    """按冻结掩膜与配置定位第 target_task 号任务的场景、窗格与 ROI 切块。"""
    det = json.loads((masks_dir / 'detections.json').read_text())
    layers = json.loads((masks_dir / 'layers.json').read_text())['layers']
    anim_path = masks_dir / 'animation-events.json'
    anim_plan = json.loads(anim_path.read_text()) if anim_path.exists() else None
    anim_cfg = AnimationSettings(**anim_plan['settings']) if anim_plan else None
    spatial = SpatialSettings(**json.loads(spatial_cfg_path.read_text()))
    baseline = json.loads(baseline_path.read_text())
    cfg = roi.ROISettings(**baseline['ROI']['配置'])
    settings = HLSSettings.from_env()
    trusted = ContextSettings(**json.loads(trusted_cfg_path.read_text()))

    files = sorted((masks_dir / 'frames').glob('*.png'))
    layer = layers[0]
    layer_dir = Path(layer['directory'])
    if not layer_dir.exists():
        layer_dir = masks_dir / 'masks'
    masks = [cv2.imread(str(layer_dir / p.name), 0) for p in files]

    tid = 0
    layer_done = {}
    for s0, s1 in det['scenes']:
        refs = np.unique(np.linspace(s0, s1 - 1, min(24, s1 - s0), dtype=int))
        sched = (
            planned_windows(
                s0,
                s1,
                settings.window_frames,
                settings.overlap_frames,
                [e['interval'] for e in anim_plan['layers'].get('0', []) if s0 <= e['interval'][0] < s1],
                anim_cfg,
            )
            if anim_plan
            else list(rsl.windows(s0, s1, settings.window_frames, settings.overlap_frames))
        )
        for start, end in sched:
            footprint = np.bitwise_or.reduce(masks[start:end])
            budget = max(r - l for l, r, _ in rsl.seam_variants(start, end, s0, s1, settings, cfg))
            for frag in plan(footprint, budget, cfg, spatial):
                tid += 1
                if tid == target_task:
                    return dict(
                        task_id=tid,
                        det=det,
                        files=files,
                        masks=masks,
                        scene=(s0, s1),
                        refs=refs,
                        start=start,
                        end=end,
                        fragment=frag,
                        layer_done=layer_done,
                        cfg=cfg,
                        settings=settings,
                        spatial=spatial,
                        trusted=trusted,
                    )
                for j in range(start, end):
                    layer_done.setdefault(j, np.zeros(masks[0].shape, np.uint8))[:] |= masks[j] & frag['ownership']
    raise ValueError(f'未找到任务编号 {target_task}（总任务数={tid}）')


def profile_task(args) -> dict:
    info = _locate_task(args.masks, args.baseline, args.spatial_config, args.trusted_context_config, args.task)
    files = info['files']
    masks = info['masks']
    s0, s1 = info['scene']
    start, end = info['start'], info['end']
    frag = info['fragment']
    box = frag['box']
    a, b, c, d = box
    ownership = frag['ownership']
    cfg = info['cfg']
    settings = info['settings']
    trusted = info['trusted']
    edge_cfg = edge_residue.EdgeSettings(**json.loads(args.edge_config.read_text()))
    variants = rsl.seam_variants(start, end, s0, s1, settings, cfg)
    left, right, length = variants[args.attempt]
    if args.max_frames and right - left > args.max_frames:
        right_eval = left + args.max_frames
    else:
        right_eval = right

    cv_timings: dict[str, float] = defaultdict(float)
    cv_counts: dict[str, int] = defaultdict(int)
    step_timings: dict[str, float] = defaultdict(float)
    step_counts: dict[str, int] = defaultdict(int)

    real_cv2 = cv2
    orig_rest_cv2 = restoration.cv2
    orig_edge_cv2 = edge_residue.cv2
    orig_rsl_cv2 = rsl.cv2
    orig_track = restoration.CleanReferences._track
    orig_local = restoration.CleanReferences._collect_local_donors
    orig_fill = restoration.fill_small_holes
    orig_orth = restoration.restore_orthogonal_luma_detail
    orig_peaks = edge_residue.color_peaks
    orig_eval_peak = edge_residue._evaluate_peak
    orig_box_count = edge_residue._box_count
    orig_measure = edge_residue.measure
    orig_build_eval = restoration._build_donor_evaluator

    def wrap_fn(fn, name: str):
        def inner(*f_args, **f_kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*f_args, **f_kwargs)
            finally:
                step_timings[name] += time.perf_counter() - t0
                step_counts[name] += 1

        return inner

    def wrap_donor_builder(shape, donor_crops):
        fn = orig_build_eval(shape, donor_crops)
        step_counts['donor_crops_total'] += len(donor_crops)

        def timed_eval(hue, ecfg, painted_hsv=None):
            t0 = time.perf_counter()
            try:
                return fn(hue, ecfg, painted_hsv=painted_hsv)
            finally:
                step_timings['edge.donor_eval'] += time.perf_counter() - t0
                step_counts['edge.donor_eval'] += 1

        return timed_eval

    with tempfile.TemporaryDirectory(prefix='v17c-prof-') as tmp:
        work = Path(tmp)
        for name in ('frames', 'masks', 'edge-band', 'scene-protection', 'subtitle-protection'):
            src = args.masks / name
            if src.exists():
                (work / name).symlink_to(src.resolve(), target_is_directory=True)
        dest = work / 'roi-layer-00'
        dest.mkdir()
        for p in files:
            os.link(p, dest / p.name)
        task_dir = work / f'roi-{args.task:04d}-attempt-{args.attempt:02d}'
        staged = task_dir / 'staged'
        (task_dir / 'frames').mkdir(parents=True)
        (task_dir / 'masks').mkdir(parents=True)
        (task_dir / 'out/frames/frames').mkdir(parents=True)
        staged.mkdir()
        (task_dir / 'seam-difference').mkdir()
        (task_dir / 'seam-shared').mkdir()

        # 准备模型输出（优先使用真实模型输出镜像，否则用 fallback 或高斯平滑模拟预测裁块）
        reuse_dir = args.model_output / task_dir.name / 'frames' if args.model_output else None
        predicted = []
        if reuse_dir is not None and reuse_dir.is_dir():
            predicted = sorted(reuse_dir.glob('*.png'))
            pred_source = str(reuse_dir)
        else:
            pred_source = str(args.fallback_frames) if args.fallback_frames else 'synthetic_blur'
            for idx, j in enumerate(range(left, right)):
                if args.fallback_frames and (args.fallback_frames / files[j].name).exists():
                    im = real_cv2.imread(str(args.fallback_frames / files[j].name))
                else:
                    im = real_cv2.imread(str(files[j]))
                    im = real_cv2.GaussianBlur(im, (9, 9), 2.5)
                crop_im = roi.crop(im, box)
                out_p = task_dir / 'out/frames/frames' / f'{idx:06d}.png'
                real_cv2.imwrite(str(out_p), crop_im)
                predicted.append(out_p)
            crop0 = roi.crop(real_cv2.imread(str(files[left])), box)
            real_cv2.imwrite(str(task_dir / 'frames/000000.png'), crop0)

        if not (task_dir / 'frames/000000.png').exists():
            real_cv2.imwrite(str(task_dir / 'frames/000000.png'), roi.crop(real_cv2.imread(str(files[left])), box))

        # 注入计时器
        restoration.cv2 = _TimedCv2(real_cv2, cv_timings, cv_counts, 'restoration.cv2')
        edge_residue.cv2 = _TimedCv2(real_cv2, cv_timings, cv_counts, 'edge_residue.cv2')
        rsl.cv2 = _TimedCv2(real_cv2, cv_timings, cv_counts, 'roi_spatial.cv2')
        restoration.CleanReferences._track = wrap_fn(orig_track, 'restoration._track')
        restoration.CleanReferences._collect_local_donors = wrap_fn(orig_local, 'restoration._collect_local_donors')
        restoration.fill_small_holes = wrap_fn(orig_fill, 'restoration.fill_small_holes')
        restoration.restore_orthogonal_luma_detail = wrap_fn(orig_orth, 'restoration.restore_orthogonal_luma_detail')
        edge_residue.color_peaks = wrap_fn(orig_peaks, 'edge.color_peaks')
        edge_residue._evaluate_peak = wrap_fn(orig_eval_peak, 'edge._evaluate_peak')
        edge_residue._box_count = wrap_fn(orig_box_count, 'edge._box_count')
        edge_residue.measure = wrap_fn(orig_measure, 'edge.measure')
        restoration._build_donor_evaluator = wrap_donor_builder

        try:
            shape = masks[0].shape
            pixels = shape[0] * shape[1]
            t0 = time.perf_counter()
            if args.include_prep:
                scene_track_hps = (
                    rsl._scene_track_highpass(info['det'], files, s0, s1, shape)
                    if trusted.detail_preserve_mode == 'orthogonal_luma'
                    else {}
                )
                step_timings['scene_track_highpass'] = time.perf_counter() - t0
            else:
                scene_track_hps = {}

            if args.include_prep:
                tp0 = time.perf_counter()
                for j in range(left, right_eval):
                    roi.atomic_png(task_dir / 'frames' / f'{j - left:06d}.png', roi.crop(real_cv2.imread(str(files[j])), box))
                    excl = exclusion_mask(read_confirmed(work / 'masks' / files[j].name), trusted)
                    mm = roi.crop(excl, box, True)
                    mp = task_dir / 'masks' / f'{j - left:06d}.png'
                    roi.atomic_png(mp, mm)
                    verify_from_confirmed(work / 'masks' / files[j].name, box, real_cv2.imread(str(mp), 0), roi.crop(masks[j] & ownership, box, True), pixels, trusted)
                step_timings['input_prep_and_audit'] = time.perf_counter() - tp0

            task_refs = rsl._window_refs(s0, s1, start, end, info['refs'], trusted.dense_reference_frames)
            crop_shape = real_cv2.imread(str(task_dir / 'frames/000000.png')).shape
            reference_holder = []
            scene_hp_holder = []

            def clean_references():
                if not reference_holder:
                    t_read = time.perf_counter()
                    ref_imgs = [roi.crop(real_cv2.imread(str(files[j])), box) for j in task_refs]
                    ref_msks = [roi.crop(real_cv2.imread(str(work / 'masks' / files[j].name), 0), box, True) for j in task_refs]
                    step_timings['clean_refs_init.imread'] += time.perf_counter() - t_read
                    t_ctor = time.perf_counter()
                    reference_holder.append(restoration.CleanReferences(ref_imgs, ref_msks))
                    step_timings['clean_refs_init.ctor'] += time.perf_counter() - t_ctor
                    if trusted.detail_preserve_mode == 'orthogonal_luma':
                        t_shp = time.perf_counter()
                        scene_hp_holder.append(rsl._scene_static_highpass(ref_imgs))
                        step_timings['clean_refs_init.static_hp'] += time.perf_counter() - t_shp
                return reference_holder[0]

            layer_done = info['layer_done']
            completed = {j for j in range(s0, s1) if j in layer_done and np.any(layer_done[j] & ownership)}
            active_frames = [j for j in range(start, end) if np.any(masks[j] & ownership)]
            sample_frames = (
                {active_frames[q] for q in np.linspace(0, len(active_frames) - 1, min(trusted.model_sample_count, len(active_frames)), dtype=int)}
                if active_frames
                else set()
            )
            stage_frames = set(sorted(sample_frames)[: trusted.retain_stage_sample_count]) if sample_frames else set()

            maximum = 0.0
            residue_scores = []
            processed_frames = 0
            loop_t0 = time.perf_counter()
            for j in range(left, right_eval):
                t_io = time.perf_counter()
                repair = real_cv2.imread(str(predicted[j - left]))
                step_timings['frame_io.read_repair'] += time.perf_counter() - t_io
                construction = start <= j < end
                active = (masks[j][b:d, a:c] > 0) & (ownership[b:d, a:c] > 0)
                prior_done = layer_done.get(j)
                shared = (prior_done[b:d, a:c] > 0) & (masks[j][b:d, a:c] > 0) if prior_done is not None else np.zeros(active.shape, bool)
                if not construction and not shared.any():
                    continue
                if not active.any() and not shared.any() and j not in sample_frames:
                    continue
                processed_frames += 1

                t_io2 = time.perf_counter()
                original = roi.crop(real_cv2.imread(str(files[j])), box)
                mask = roi.crop(masks[j], box, True)
                confirmed = roi.crop(read_confirmed(work / 'masks' / files[j].name), box, True)
                step_timings['frame_io.read_orig_mask'] += time.perf_counter() - t_io2

                references = clean_references()
                t_res = time.perf_counter()
                restored = references.restore(original, mask, repair, confirmed=confirmed)
                step_timings['clean_refs.restore_total'] += time.perf_counter() - t_res
                step_counts['clean_refs.restore_total'] += 1

                if trusted.detail_preserve_mode == 'orthogonal_luma' and scene_hp_holder and scene_hp_holder[0] is not None:
                    wm_hp = scene_hp_holder[0]
                    if j in scene_track_hps:
                        trk_crop = roi.crop(scene_track_hps[j], box, True)
                        wm_hp = np.where(np.abs(trk_crop) > np.abs(wm_hp), trk_crop, wm_hp)
                    restored = restoration.restore_orthogonal_luma_detail(original, restored, mask, wm_hp, edge_cfg)

                t_comp = time.perf_counter()
                painted = edge_residue.composite(original, restored, mask)[: d - b, : c - a]
                step_timings['composite'] += time.perf_counter() - t_comp

                t_dest = time.perf_counter()
                image = real_cv2.imread(str(dest / files[j].name))
                old = image[b:d, a:c]
                step_timings['frame_io.read_dest'] += time.perf_counter() - t_dest

                if shared.any():
                    t_seam = time.perf_counter()
                    disagreement = float(np.abs(old.astype(float) - painted.astype(float))[shared].mean())
                    maximum = max(maximum, disagreement)
                    step_timings['seam_check'] += time.perf_counter() - t_seam
                if construction:
                    t_w = time.perf_counter()
                    difference = np.where(shared, np.abs(old.astype(float) - painted.astype(float)).mean(axis=2), 0.0)
                    roi.atomic_png(task_dir / 'seam-difference' / files[j].name, np.rint(difference * 256).astype(np.uint16))
                    roi.atomic_png(task_dir / 'seam-shared' / files[j].name, shared.astype(np.uint8) * 255)
                    if j in completed:
                        weight = overlap_weight(j, start, end, completed)
                        mixed = real_cv2.addWeighted(painted, weight, old, 1 - weight, 0)
                        painted[active & shared] = mixed[active & shared]
                    old[active] = painted[active]
                    roi.atomic_png(staged / files[j].name, image)
                    step_timings['frame_io.write_staged_and_seam'] += time.perf_counter() - t_w

                    if active.any():
                        t_m_io = time.perf_counter()
                        band_path = work / 'edge-band' / files[j].name
                        band = read_confirmed(band_path)[b:d, a:c] if band_path.exists() else np.zeros(active.shape, np.uint8)
                        source_color = roi.crop(real_cv2.imread(str(files[j])), box)[: d - b, : c - a]
                        evidence = getattr(references, 'last_evidence', None)
                        if evidence is not None:
                            evidence = {k: (v[: d - b, : c - a] if hasattr(v, 'shape') else v) for k, v in evidence.items()}
                        active_mask = active.astype(np.uint8) * 255
                        conf_crop = confirmed[: d - b, : c - a]
                        prot_crop = rsl.protection_mask(work, files[j].name, shape)[b:d, a:c]
                        step_timings['frame_io.read_measure_inputs'] += time.perf_counter() - t_m_io

                        t_m = time.perf_counter()
                        score = edge_residue.measure(
                            source_color,
                            old,
                            active_mask,
                            band & active_mask,
                            edge_cfg,
                            reference_evidence=evidence,
                            confirmed=conf_crop,
                            protected=prot_crop,
                        )
                        step_timings['edge.measure_total'] += time.perf_counter() - t_m
                        step_counts['edge.measure_total'] += 1

                        if score.get('判定') in rsl.RESIDUE_ERRORS and trusted.dense_reference_frames > len(task_refs):
                            t_d = time.perf_counter()
                            dense_ev = rsl._dense_frame_evidence(j, s0, s1, box, files, work, files, task_refs, original, mask, confirmed)
                            step_timings['dense_frame_evidence'] += time.perf_counter() - t_d
                            step_counts['dense_frame_evidence'] += 1
                            if dense_ev is not None:
                                dense_ev = {k: (v[: d - b, : c - a] if hasattr(v, 'shape') else v) for k, v in dense_ev.items()}
                                dense_sc = edge_residue.measure(
                                    source_color,
                                    old,
                                    active_mask,
                                    band & active_mask,
                                    edge_cfg,
                                    reference_evidence=dense_ev,
                                    confirmed=conf_crop,
                                    protected=prot_crop,
                                )
                                if (dense_sc.get('确认残留', 0), dense_sc.get('待目检', 0)) < (score.get('确认残留', 0), score.get('待目检', 0)):
                                    score = dense_sc
                        residue_scores.append(dict(frame=j, **score))

            loop_total = time.perf_counter() - loop_t0
            ref_timings = dict(reference_holder[0].timings) if reference_holder else {}
            ref_stats = dict(reference_holder[0].stats) if reference_holder else {}
        finally:
            restoration.cv2 = orig_rest_cv2
            edge_residue.cv2 = orig_edge_cv2
            rsl.cv2 = orig_rsl_cv2
            restoration.CleanReferences._track = orig_track
            restoration.CleanReferences._collect_local_donors = orig_local
            restoration.fill_small_holes = orig_fill
            restoration.restore_orthogonal_luma_detail = orig_orth
            edge_residue.color_peaks = orig_peaks
            edge_residue._evaluate_peak = orig_eval_peak
            edge_residue._box_count = orig_box_count
            edge_residue.measure = orig_measure
            restoration._build_donor_evaluator = orig_build_eval

    sorted_steps = {k: round(v, 4) for k, v in sorted(step_timings.items(), key=lambda kv: -kv[1])}
    sorted_cv = {k: round(v, 4) for k, v in sorted(cv_timings.items(), key=lambda kv: -kv[1])}
    return dict(
        任务=args.task,
        尝试=args.attempt,
        场景=[s0, s1],
        施工区间=[start, end],
        输入区间=[left, right],
        实际剖析区间=[left, right_eval],
        处理帧数=processed_frames,
        ROI=box,
        裁块尺寸=[c - a, d - b],
        参考帧数=len(task_refs),
        预测来源=pred_source,
        CPU循环总耗时秒=round(loop_total, 4),
        单帧平均耗时秒=round(loop_total / max(1, processed_frames), 4),
        CleanReferences内部计时={k: round(v, 4) for k, v in ref_timings.items()},
        CleanReferences统计=ref_stats,
        子步骤耗时秒=sorted_steps,
        子步骤调用次数=dict(sorted(step_counts.items())),
        OpenCV算子耗时秒=sorted_cv,
        OpenCV算子调用次数=dict(sorted(cv_counts.items())),
        残留汇总=rsl.residue_summary(residue_scores) if residue_scores else None,
        接缝最大差异=round(maximum, 4),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--masks', type=Path, default=Path('/workspace/watermark-test/878ffd51/out-v17c/masks-run'))
    p.add_argument('--baseline', type=Path, default=Path('/workspace/watermark-test/878ffd51/v17c-离线数据/v6/report.json'))
    p.add_argument('--spatial-config', type=Path, default=Path('config/v13-spatial-roi.json'))
    p.add_argument('--trusted-context-config', type=Path, default=Path('config/v17c-trusted-context.json'))
    p.add_argument('--edge-config', type=Path, default=Path('config/v17c-edge-residue.json'))
    p.add_argument('--model-output', type=Path, default=None, help='含 roi-000N-attempt-00/frames 的目录')
    p.add_argument('--fallback-frames', type=Path, default=Path('/workspace/watermark-test/878ffd51/v17c-离线数据/v17b/inpaint/run/roi-layer-00'))
    p.add_argument('--task', type=int, default=1)
    p.add_argument('--attempt', type=int, default=0)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--include-prep', action='store_true')
    p.add_argument('--output', type=Path, default=None)
    args = p.parse_args()
    if args.fallback_frames and not args.fallback_frames.exists():
        args.fallback_frames = None
    res = profile_task(args)
    text = json.dumps(res, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
