"""显式启用的 SAM 3.1 文字视频分割来源；只写掩膜，不调用补画。"""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    prompts: tuple = ("watermark", "logo", "semi-transparent logo", "website url text")
    score_threshold: float = 0.5
    min_track_frames: int = 3
    min_track_fraction: float = 0.15
    window_frames: int = 120
    max_objects: int = 16
    max_layer_fraction: float = 0.2

    def __post_init__(self):
        if not self.prompts or any(not isinstance(p, str) or not p.strip() for p in self.prompts):
            raise ValueError("文字提示不能为空")
        if not 0 <= self.score_threshold <= 1 or not 0 < self.min_track_fraction <= 1:
            raise ValueError("阈值必须为有效比例")
        if self.min_track_frames < 1 or self.window_frames < self.min_track_frames or self.max_objects < 1:
            raise ValueError("持续帧数、窗口和对象数无效")
        if not 0 < self.max_layer_fraction <= 0.2:
            raise ValueError("不得放宽每层20%面积保护")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def binary(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != shape or not set(np.unique(mask)) <= {0, 1, 255}:
        raise ValueError("掩膜形状或二值取值无效")
    return (mask > 0).astype(np.uint8) * 255


def accepted_tracks(observations, config):
    """按模型对象ID及其生命周期过滤，允许移动，不作固定位置或影片判断。"""
    return {key for key, frames in observations.items()
            if len(set(frames)) >= config.min_track_frames
            and len(set(frames)) / (max(frames) - min(frames) + 1) >= config.min_track_fraction}


def checked_union(masks, shape):
    result = np.zeros(shape, np.uint8)
    for mask in masks:
        result |= binary(mask, shape)
    return result


def write_png(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise OSError("掩膜写入失败")


def run(frames_dir, output, checkpoint, config, baseline=None, predictor=None):
    if not config.enabled:
        raise ValueError("必须在配置中显式启用SAM3.1来源")
    started = time.perf_counter()
    files = sorted(Path(frames_dir).glob('*.png'))
    if not files:
        raise ValueError("没有输入帧")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录必须为空，避免混用旧掩膜")
    output.mkdir(parents=True, exist_ok=True)
    shape = cv2.imread(str(files[0])).shape[:2]
    if baseline:
        for file in files:
            p = Path(baseline) / file.name
            m = cv2.imread(str(p), 0)
            if m is None:
                raise ValueError("基线逐帧掩膜缺失")
            binary(m, shape)
    binding = {'config': asdict(config), 'frames': {f.name: digest(f) for f in files},
               'checkpoint_sha256': digest(checkpoint), 'source': 'sam3.1-text-video'}
    (output / 'source.json').write_text(json.dumps(binding, ensure_ascii=False, indent=2))
    loaded = time.perf_counter()
    if predictor is None:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint), use_fa3=False, compile=False, warm_up=False,
            max_num_objects=config.max_objects, default_output_prob_thresh=config.score_threshold,
            async_loading_frames=False)
    load_seconds = time.perf_counter() - loaded
    timings, tracks, rows = [], {}, []
    raw = output / 'raw-object-masks'
    for start in range(0, len(files), config.window_frames):
        end = min(len(files), start + config.window_frames)
        with tempfile.TemporaryDirectory(prefix='sam31-frames-') as temp:
            # 官方视频目录读取器要求JPEG；仅作输入转换，不裁剪、不降采样。
            for i, f in enumerate(files[start:end]):
                image = cv2.imread(str(f))
                if image is None or image.shape[:2] != shape:
                    raise ValueError("输入帧缺失或尺寸不一致")
                if not cv2.imwrite(str(Path(temp) / f'{i:06d}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 100]):
                    raise OSError("输入转换失败")
            for pi, prompt in enumerate(config.prompts):
                t = time.perf_counter()
                sid = predictor.handle_request(dict(type='start_session', resource_path=temp,
                                                     offload_video_to_cpu=True))['session_id']
                seen, observations = set(), defaultdict(list)
                try:
                    predictor.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0,
                                                  text=prompt, output_prob_thresh=config.score_threshold))
                    for response in predictor.handle_stream_request(dict(type='propagate_in_video', session_id=sid)):
                        idx = int(response['frame_index'])
                        if not 0 <= idx < end - start or idx in seen:
                            raise ValueError("模型返回重复或越界帧")
                        seen.add(idx)
                        data = response['outputs']
                        ids, probs, masks = data['out_obj_ids'], data['out_probs'], data['out_binary_masks']
                        if not len(ids) == len(probs) == len(masks):
                            raise ValueError("模型对象输出不齐全")
                        for obj, prob, mask in zip(ids, probs, masks):
                            if not np.isfinite(prob) or float(prob) < config.score_threshold:
                                continue
                            mask = binary(mask, shape)
                            if not np.any(mask):
                                continue
                            key = f'w{start:06d}-p{pi}-o{int(obj)}'
                            observations[key].append(start + idx)
                            write_png(raw / key / files[start + idx].name, mask)
                    if len(seen) != end - start:
                        raise ValueError("视频传播未返回全部输入帧")
                finally:
                    predictor.handle_request(dict(type='close_session', session_id=sid))
                keep = accepted_tracks(observations, config)
                for key, indices in observations.items():
                    tracks[key] = dict(frames=indices, retained=key in keep, prompt=prompt)
                timings.append(dict(start=start, end=end, prompt=prompt, seconds=time.perf_counter()-t))
                print(f'窗口 {start}:{end} 提示 {prompt} 完成，保留{len(keep)}/{len(observations)}条轨迹', flush=True)
    t = time.perf_counter()
    layer_max = {}
    for key, track in tracks.items():
        if not track['retained']:
            continue
        maximum = max(float(np.count_nonzero(cv2.imread(str(raw/key/files[i].name), 0))) / np.prod(shape)
                      for i in track['frames'])
        track['maximum_fraction'] = maximum
        if maximum > config.max_layer_fraction:
            # 不拆分过大对象来绕过20%限制，保留原始候选以供审查。
            track['retained'] = False
            track['rejection'] = '单对象超过每层20%面积保护'
        else:
            layer_max[key] = maximum
    # 将对象按各自最大面积装箱为图层，窗口间无需复用对象ID。
    bins, totals = [], []
    for key in sorted(layer_max, key=lambda k: (-layer_max[k], k)):
        j = next((j for j, total in enumerate(totals) if total + layer_max[key] <= config.max_layer_fraction), len(bins))
        if j == len(bins):
            bins.append([])
            totals.append(0.)
        bins[j].append(key)
        totals[j] += layer_max[key]
    layer_rows = [dict(directory=f'layers/{j:02d}', objects=keys, maximum_fraction=0., sha256={}) for j, keys in enumerate(bins)]
    for i, file in enumerate(files):
        layers = []
        for layer in layer_rows:
            masks = [cv2.imread(str(raw/key/file.name), 0) for key in layer['objects'] if i in tracks[key]['frames']]
            mask = checked_union(masks, shape)
            fraction = float(np.count_nonzero(mask) / mask.size)
            if fraction > config.max_layer_fraction:
                raise ValueError("图层超过20%保护")
            path = output / layer['directory'] / file.name
            write_png(path, mask)
            layer['maximum_fraction'] = max(layer['maximum_fraction'], fraction)
            layer['sha256'][file.name] = digest(path)
            layers.append(mask)
        mask = checked_union(layers, shape)
        write_png(output/'masks'/file.name, mask)
        row = dict(frame=file.name, sam_pixels=int(np.count_nonzero(mask)))
        if baseline:
            base = binary(cv2.imread(str(Path(baseline)/file.name), 0), shape)
            merged = mask | base
            write_png(output/'merged-masks'/file.name, merged)
            row.update(baseline_pixels=int(np.count_nonzero(base)), merged_pixels=int(np.count_nonzero(merged)),
                       added_pixels=int(np.count_nonzero((mask>0)&(base==0))),
                       removed_pixels=int(np.count_nonzero((base>0)&(merged==0))))
        rows.append(row)
    data = dict(config=asdict(config), tracks=tracks, layers=layer_rows, frames=rows,
                timings=timings, model_load_seconds=load_seconds,
                postprocess_seconds=time.perf_counter()-t, total_seconds=time.perf_counter()-started,
                input_frame_count=len(files), area_protection_passed=True)
    (output/'metrics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline')
    args = parser.parse_args()
    config = Config(**json.loads(Path(args.config).read_text()))
    from .gpu_lock import device_lock
    import torch
    with device_lock(), torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        run(args.frames, args.output, args.checkpoint, config, args.baseline)


if __name__ == '__main__':
    main()
