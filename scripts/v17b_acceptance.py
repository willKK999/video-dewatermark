"""v17b 只读验收脚本：输出三版同口径闪烁候选、清晰度与原生保护汇总 JSON（不写图片）。"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from worker.edge_residue import highlight_guard
from worker.hls_acceptance import clarity_summary
from worker.json_types import json_native
from worker.matte import temporal_detail
from worker.roi_spatial_layers import protection_mask


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inputs(args):
    base = args.base
    v13e = args.v13e if args.v13e is not None else (base / 'v13/inpaint-v6-retries/第1块.mp4' if base is not None else None)
    v16 = args.v16 if args.v16 is not None else (base / 'v16/inpaint-v6-retries-context/第1块.mp4' if base is not None else None)
    masks_dir = args.masks if args.masks is not None else (base / 'v13/masks-run-strokes' if base is not None else None)
    edge_band_dir = args.edge_band if args.edge_band is not None else (base / 'v17/masks-run/edge-band' if base is not None else None)
    missing = [name for name, val in [('v13e', v13e), ('v16', v16), ('masks', masks_dir), ('edge-band', edge_band_dir)] if val is None]
    if missing:
        raise ValueError(f'缺少输入路径（请提供 --base 或显式参数）: {missing}')
    return v13e, v16, masks_dir, edge_band_dir


def _read_guard(masks_dir, name, shape):
    for folder in ['scene-protection', 'subtitle-protection']:
        subdir = masks_dir / folder
        if subdir.is_dir() and not (subdir / name).is_file():
            raise ValueError(f'缺少保护掩膜文件: {subdir / name}')
    return protection_mask(masks_dir, name, shape)


def execute(args):
    started = time.monotonic()
    cv2.setNumThreads(1)
    run_dir = Path(args.run)
    v13e_path, v16_path, masks_dir, edge_band_dir = _resolve_inputs(args)
    paths = {'v13e': Path(v13e_path), 'v16': Path(v16_path), 'v17b': Path(args.video)}

    for label, path in paths.items():
        if not path.is_file():
            raise ValueError(f'成片文件不存在 ({label}): {path}')
    for label, folder in [('run', run_dir), ('masks', masks_dir), ('edge-band', edge_band_dir)]:
        if not folder.is_dir():
            raise ValueError(f'输入目录不存在 ({label}): {folder}')

    files = sorted((run_dir / 'frames').glob('*.png'))
    if not files:
        raise ValueError(f'run/frames 为空或缺失: {run_dir / "frames"}')
    n = len(files)
    detections_path = run_dir / 'detections.json'
    if not detections_path.is_file():
        raise ValueError(f'缺少 detections.json: {detections_path}')
    detections = json.loads(detections_path.read_text())
    cuts = {int(s[0]) for s in detections['scenes']}

    caps = {k: cv2.VideoCapture(str(v)) for k, v in paths.items()}
    metrics = {k: dict(时域全部样本=[], 闪烁候选=[], 清晰度样本=[]) for k in paths}
    protection = []
    previous = None

    try:
        for k, cap in caps.items():
            if not cap.isOpened():
                raise ValueError(f'无法打开视频 ({k}): {paths[k]}')
        fps = float(caps['v17b'].get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'v17b 视频帧率无效: {fps}')

        for i, frame_path in enumerate(files):
            name = frame_path.name
            source = cv2.imread(str(frame_path))
            if source is None:
                raise ValueError(f'源帧读取失败: {frame_path}')
            shape = source.shape[:2]

            oldmask_path = masks_dir / 'masks' / name
            newmask_path = run_dir / 'masks' / name
            band_path = edge_band_dir / name
            composite_path = run_dir / 'composite' / name

            oldmask = cv2.imread(str(oldmask_path), 0)
            newmask = cv2.imread(str(newmask_path), 0)
            band_img = cv2.imread(str(band_path), 0)
            composite = cv2.imread(str(composite_path))
            if oldmask is None or oldmask.shape != shape:
                raise ValueError(f'冻结旧掩膜无效: {oldmask_path}')
            if newmask is None or newmask.shape != shape:
                raise ValueError(f'新确认掩膜无效: {newmask_path}')
            if band_img is None or band_img.shape != shape:
                raise ValueError(f'收边带掩膜无效: {band_path}')
            if composite is None or composite.shape != source.shape:
                raise ValueError(f'合成帧无效: {composite_path}')

            band = band_img > 0
            guard = _read_guard(masks_dir, name, shape)
            protection.append(dict(
                frame=i,
                新增像素=int(band.sum()),
                场景字幕保护交集=int(np.count_nonzero(band & (guard > 0))),
                高亮边保护交集=int(np.count_nonzero(band & highlight_guard(source))),
                未编码确认域外修改像素=int(np.count_nonzero(np.any(composite != source, axis=2) & (newmask == 0))),
                原确认域丢失像素=int(np.count_nonzero((oldmask > 0) & (newmask == 0))),
            ))

            images = {}
            for k, cap in caps.items():
                ok, image = cap.read()
                if not ok or image is None or image.shape != source.shape:
                    raise ValueError(f'视频读取失败或几何不符 ({k}, frame={i}): {paths[k]}')
                images[k] = image

            if previous is not None:
                ox, om, oldimages = previous
                support = (oldmask | om) > 0
                sx = source.astype(np.float32)
                ps = ox.astype(np.float32)
                source_mae = float(np.abs(sx - ps)[support].mean()) if support.any() else 0.0
                for k, image in images.items():
                    y = image.astype(np.float32)
                    oy = oldimages[k].astype(np.float32)
                    result_mae = float(np.abs(y - oy)[support].mean()) if support.any() else 0.0
                    delta_mae = float(np.abs((y - sx) - (oy - ps))[support].mean()) if support.any() else 0.0
                    row = dict(
                        frame=i,
                        秒=i / fps,
                        原片镜头切换=i in cuts,
                        源片差异=source_mae,
                        成片差异=result_mae,
                        修复差异=delta_mae,
                    )
                    metrics[k]['时域全部样本'].append(row)
                    if i not in cuts and result_mae > max(source_mae + 6, source_mae * 1.8) and delta_mae > 12:
                        metrics[k]['闪烁候选'].append(row)
                    if i not in cuts and i % max(1, round(fps)) in (0, 1):
                        detail = temporal_detail(ox, source, oldimages[k], image, oldmask & om, oldmask | om)
                        detail.pop('passed', None)
                        detail.pop('threshold', None)
                        metrics[k]['清晰度样本'].append(dict(frame=i, time=i / fps, **detail))

            previous = (source, oldmask, images)
            if i % 30 == 0:
                print(f'三版同口径时域及原生保护核验 {i}/{n}', flush=True)
    finally:
        for cap in caps.values():
            cap.release()

    for k, m in metrics.items():
        m['清晰度'] = clarity_summary(m.pop('清晰度样本'))
        m['视频摘要'] = _sha256_file(paths[k])
        m['闪烁候选数'] = len(m['闪烁候选'])

    protection_summary = {key: sum(r[key] for r in protection) for key in protection[0] if key != 'frame'}
    elapsed = time.monotonic() - started
    result = dict(
        帧数=n,
        帧率=fps,
        口径='三个完整版本均使用v13e冻结联合掩膜；逐帧差分与源片比较，非光流校正指标；原片切镜单列并排除闪烁报警。主色统计与闪烁候选均非语义真值；清晰度仅记录。',
        版本=metrics,
        原生保护=protection,
        原生保护汇总=protection_summary,
        源片镜头切换=sorted(cuts),
        路径={k: str(v) for k, v in paths.items()},
        run=str(run_dir),
        masks=str(masks_dir),
        edge_band=str(edge_band_dir),
        耗时秒=elapsed,
        通过=False,
        完整连续目检完成=False,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(json_native(result), ensure_ascii=False, indent=2))
    print(
        json.dumps(
            json_native(
                dict(
                    帧数=n,
                    原生保护=protection_summary,
                    版本={
                        k: dict(
                            闪烁候选数=m['闪烁候选数'],
                            清晰度最低=m['清晰度']['最小保留比'],
                            低于07样本数=m['清晰度']['低于0.7样本数'],
                        )
                        for k, m in metrics.items()
                    },
                    耗时秒=elapsed,
                )
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=None, help='数据根目录（用于推导未显式提供的默认路径）')
    parser.add_argument('--run', type=Path, required=True, help='v17b 运行 run 目录')
    parser.add_argument('--video', type=Path, required=True, help='v17b 成片路径')
    parser.add_argument('--v13e', type=Path, default=None, help='v13e 成片路径')
    parser.add_argument('--v16', type=Path, default=None, help='v16 成片路径')
    parser.add_argument('--masks', type=Path, default=None, help='统一冻结掩膜目录（如 base/v13/masks-run-strokes）')
    parser.add_argument('--edge-band', type=Path, default=None, help='v17 收边带目录')
    parser.add_argument('--output', type=Path, required=True, help='输出验收 JSON 路径')
    return parser


if __name__ == '__main__':
    execute(build_parser().parse_args())
