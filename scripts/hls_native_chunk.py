"""视频去水印系统：原分辨率、原帧率合成单块；只在自动支持范围内替换像素。"""

import argparse, json, time, subprocess, math
from pathlib import Path
from fractions import Fraction
from functools import lru_cache, wraps
import cv2, numpy as np
from worker.fine_masks import Supports
from worker.matte import (
    projected_texture_composite,
    temporal_detail,
    stable_detail_mask,
)
from worker.restoration import soft_composite
from worker.media import probe
from worker.speedup import bounded_map, encoder_args
from scripts.hls_chunk_check import mux_core_audio, audio_hash


def source_frame_for_output(frame, source_fps, output_fps):
    if not 0 < output_fps <= source_fps:
        raise ValueError("输出帧率必须大于零且不高于原片")
    return math.floor(frame * source_fps / output_fps)


def serialized_cuda_render(operation):
    """只在合成期间持有显卡锁，后续七版补修仍可正常调用模型。"""

    @wraps(operation)
    def execute(*args, **kwargs):
        if kwargs.get("backend", "numpy") == "torch-cuda":
            from worker.gpu_lock import device_lock

            with device_lock(check=kwargs.get("check")):
                return operation(*args, **kwargs)
        return operation(*args, **kwargs)

    return execute


@serialized_cuda_render
def render(
    source,
    coarse,
    detected,
    prior,
    calibration,
    template,
    graphic,
    output,
    duration=25,
    clean_reference_frames=None,
    projection_calibration=None,
    output_fps=None,
    detail_sigma=1.2,
    video_crf=16,
    final_recompose=False,
    layer_manifest=None,
    *,
    backend="numpy",
    fused_detail=False,
    encoder_name="libx264",
    frames_only=False,
    source_start_frame=0,
    check=lambda: None,
):
    if not 0 < duration <= 30 or source_start_frame < 0:
        raise ValueError("原生块必须为三十秒以内且起始帧非负")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    projector = projected_texture_composite
    soft = soft_composite
    torch_engine = None
    if backend != "numpy":
        from worker.torch_composite import TorchCompositor

        torch_engine = TorchCompositor("cpu" if backend == "torch-cpu" else "cuda")
        projector, soft = torch_engine.project, torch_engine.soft
    timings = dict(decode=0.0, composite=0.0, detail=0.0, encode=0.0, mux=0.0)
    metadata = probe(source)
    video = next(v for v in metadata["streams"] if v["codec_type"] == "video")
    source_fps = float(Fraction(video["avg_frame_rate"]))
    fps = output_fps or source_fps
    source_frame_for_output(0, source_fps, fps)
    width, height = video["width"], video["height"]
    count = round(duration * fps)
    detections = json.loads(detected.read_text())
    from worker.hls_layers import VerifiedLayers

    layer_proof = VerifiedLayers(layer_manifest, detected) if layer_manifest else None
    alpha = np.load(calibration)["alpha"]
    supports = Supports(prior, detections, alpha, graphic)
    native = np.load(template)["template"]
    coefficients = (
        json.loads(projection_calibration.read_text())
        if projection_calibration
        else None
    )
    mh, mw = supports.median.shape

    source_positions = None
    alignment = (
        clean_reference_frames / "alignment.json" if clean_reference_frames else None
    )
    if alignment and not alignment.exists():
        raise ValueError("原生参考对齐报告尚未完成，禁止使用未经校准的参考帧")
    if alignment and alignment.exists():
        source_positions = np.array(
            [p["source_frame"] for p in json.loads(alignment.read_text())["帧对应"]]
        )
    native_refs = None
    if clean_reference_frames:
        from worker.texture_refs import NativeReferences

        native_refs = NativeReferences(
            coarse,
            clean_reference_frames,
            detections["scenes"],
            match_low_frequency=True,
            detail_sigma=detail_sigma,
            blur_backend=torch_engine,
        )

    @lru_cache(maxsize=8)
    def reference(index):
        return cv2.imread(str(coarse / "frames" / f"{index:06d}.png"))

    @lru_cache(maxsize=8)
    def repaired(index):
        return cv2.resize(
            cv2.imread(str(coarse / "composite" / f"{index:06d}.png")),
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )

    silent = output / "native-video.mp4"
    errors = (output / "encode.log").open("wb")
    if frames_only:
        (output / "frames").mkdir(exist_ok=True)
    encoder = (
        None
        if frames_only
        else subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                *encoder_args(encoder_name, video_crf),
                "-threads",
                "4",
                "-progress",
                str(output / "encode-progress.log"),
                "-nostats",
                "-pix_fmt",
                "yuv420p",
                str(silent),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=errors,
        )
    )
    cap = cv2.VideoCapture(str(source))
    if source_start_frame and (
        not cap.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)
        or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - source_start_frame) > 0.01
    ):
        cap.release()
        raise ValueError("无法精确定位原片块起点")
    stats = []
    reference_counts = []
    previous = None
    mapping = []
    max_area = 0.0
    source_cursor = -1
    source_indices = []

    def decode(frame):
        nonlocal source_cursor
        source_index = source_frame_for_output(frame, source_fps, fps)
        checked = time.monotonic()
        while source_cursor < source_index:
            ok, original = cap.read()
            if not ok:
                raise ValueError("原片主体帧不足")
            source_cursor += 1
        timings["decode"] += time.monotonic() - checked
        return frame, source_index, original

    decoded = bounded_map(decode, range(count), depth=2)
    try:
        for frame, source_index, original in decoded:
            check()
            source_indices.append(source_index)
            frame_started = time.monotonic()
            frame_detail_seconds = 0.0
            small = cv2.resize(original, (mw, mh))
            nominal = (
                int(np.argmin(np.abs(source_positions - source_index)))
                if source_positions is not None
                else round(source_index * 30 / source_fps)
            )
            candidates = [
                i
                for i in [nominal - 1, nominal, nominal + 1]
                if 0
                <= i
                < (
                    len(source_positions)
                    if source_positions is not None
                    else round((duration + 2) * 30)
                )
            ]
            index = min(
                candidates,
                key=lambda i: float(
                    np.abs(
                        small[::8, ::8].astype(np.float32) - reference(i)[::8, ::8]
                    ).mean()
                ),
            )
            mapping.append(index)
            parts = supports.at(index, small)
            base = repaired(index)
            reference_valid = None
            if native_refs:
                base, reference_valid = native_refs.restore(index)
                reference_counts.append(int(np.count_nonzero(reference_valid)))
            result = original.copy()
            faint = {}
            faint_union = np.zeros((height, width), np.uint8)
            occupied = np.zeros_like(faint_union)
            opaque_union = np.zeros_like(faint_union)
            other_marks = np.zeros_like(faint_union)
            templates = []
            for part in parts:
                mask = cv2.resize(
                    part["mask"], (width, height), interpolation=cv2.INTER_NEAREST
                )
                if part["opaque"]:
                    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
                    other_marks |= mask
                    opaque_union |= mask
                    occupied |= mask
                    painted = soft(original, base, mask, 2)
                    result[mask > 0] = painted[mask > 0]
                    continue
                if part["kind"] == "fixed":
                    g = native
                else:
                    region = next(
                        r for r in supports.regions if r["id"] == part["region"]
                    )
                    x, y, a, b = [round(q * width / mw) for q in region["box"]]
                    g0 = native[y:b, x:a]
                    if part["track"]["flip"]:
                        g0 = cv2.flip(g0, 1)
                    x, y, a, b = [round(q * width / mw) for q in part["point"]["box"]]
                    g = np.zeros((height, width), np.float32)
                    g[y:b, x:a] = cv2.resize(g0, (a - x, b - y))
                faint_union |= mask
                occupied |= mask
                if part["kind"] == "fixed":
                    faint[part["id"]] = mask
                else:
                    other_marks |= mask
                    templates.append(g)
            faint_union[opaque_union > 0] = 0
            if faint:
                templates.insert(0, native)
            if templates and np.any(faint_union):
                painted = projector(
                    original,
                    base,
                    faint_union,
                    templates,
                    calibration=coefficients if faint else None,
                    detail_sigma=detail_sigma,
                )
                result[faint_union > 0] = painted[faint_union > 0]
            if reference_valid is not None:
                trusted = reference_valid & (occupied > 0)
                result[trusted] = base[trusted]
            if final_recompose and templates and np.any(faint_union):
                detail_started = time.monotonic()
                # v7 的第二遍联合纹理合成；在同一原生帧上完成，免去中间有损编码。
                painted = projector(
                    original,
                    result,
                    faint_union,
                    templates,
                    calibration=coefficients if faint else None,
                    detail_sigma=detail_sigma,
                )
                result[faint_union > 0] = painted[faint_union > 0]
                frame_detail_seconds = time.monotonic() - detail_started
                timings["detail"] += frame_detail_seconds
            area = float(np.count_nonzero(occupied) / occupied.size)
            max_area = max(max_area, area)
            if layer_proof:
                layer_proof.validate(index, occupied)
            elif area > 0.2:
                raise ValueError("联合掩膜超过20%且缺少逐层保护与来源证明")
            timings["composite"] += (
                time.monotonic() - frame_started - frame_detail_seconds
            )
            if (
                fused_detail
                and not final_recompose
                and templates
                and np.any(faint_union)
            ):
                detail_started = time.monotonic()
                proposed = projector(
                    original,
                    result,
                    faint_union,
                    templates,
                    calibration=coefficients if faint else None,
                    detail_sigma=3,
                )
                result[faint_union > 0] = proposed[faint_union > 0]
                timings["detail"] += time.monotonic() - detail_started
            encode_started = time.monotonic()
            if frames_only:
                if not cv2.imwrite(
                    str(output / "frames" / f"{frame:06d}.png"),
                    result,
                    [cv2.IMWRITE_PNG_COMPRESSION, 2],
                ):
                    raise ValueError("原生帧检查点写入失败")
            else:
                encoder.stdin.write(result.tobytes())
            timings["encode"] += time.monotonic() - encode_started
            if previous is not None and frame % 6 == 0:
                old_source, old_result, old_faint, old_others = previous
                for identity, mask in faint.items():
                    if identity in old_faint:
                        stable = stable_detail_mask(
                            mask & old_faint[identity], old_others, other_marks
                        )
                        metric = temporal_detail(
                            old_source,
                            original,
                            old_result,
                            result,
                            stable,
                            occupied | old_others,
                        )
                        metric["stable_mask_fraction"] = float(
                            np.count_nonzero(stable) / max(1, np.count_nonzero(mask))
                        )
                        stats.append(
                            dict(
                                frame=frame,
                                time=frame / fps,
                                track_id=identity,
                                **metric,
                            )
                        )
            previous = (original, result, faint, other_marks)
            if frame % 60 == 0:
                print("高清合成帧数：" + str(frame) + "/" + str(count), flush=True)
        encode_started = time.monotonic()
        if encoder:
            encoder.stdin.close()
        if encoder and encoder.wait() != 0:
            raise ValueError("原规格视频编码失败")
        timings["encode"] += time.monotonic() - encode_started
    finally:
        decoded.close()
        cap.release()
        errors.close()
        if encoder and encoder.poll() is None:
            encoder.stdin.close()
            encoder.terminate()
            try:
                encoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
        if torch_engine is not None and torch_engine.device.type == "cuda":
            # 原生渲染结束即归还缓存显存，给下一块串行模型留足空间。
            torch_engine.torch.cuda.synchronize()
            torch_engine.torch.cuda.empty_cache()
    final = output / "chunk1.mp4"
    mux_started = time.monotonic()
    if not frames_only:
        mux_core_audio(silent, source, final, duration)
    timings["mux"] = time.monotonic() - mux_started
    if frames_only:
        timings["frame_cache_io"] = timings["encode"]
        timings["encode"] = 0.0
    report = dict(
        视频规格=[width, height, fps],
        分步耗时秒=timings,
        合成后端=backend,
        内存连续细节融合=fused_detail,
        七版原细节重合成=final_recompose,
        编码器=None if frames_only else encoder_name,
        清晰度仅记录=True,
        低于零点七的样本=[
            s for s in stats if s["measurable"] and s["retention_ratio"] < 0.7
        ],
        主体秒数=duration,
        帧数=count,
        音频内容一致=None
        if frames_only
        else audio_hash(source, duration) == audio_hash(final),
        无损帧交付=frames_only,
        原片起始帧=source_start_frame,
        清晰度阈值=0.8,
        清晰度统计=stats,
        清晰度通过=None,
        历史清晰度参考线满足=all(s["passed"] for s in stats)
        and sum(s["measurable"] for s in stats) >= max(3, min(50, round(duration * 2))),
        可测样本数=sum(s["measurable"] for s in stats),
        清晰度最小保留比=min(
            (s["retention_ratio"] for s in stats if s["measurable"]), default=None
        ),
        渲染耗时秒=time.monotonic() - started,
        原生干净参考回填像素总数=sum(reference_counts),
        模型推理规格说明="补画沿用720p30模型；在原分辨率中保留未处理像素，优先回填经校验的原生纹理，缺失部分才使用补画",
        原片帧率=source_fps,
        分频半径像素=detail_sigma,
        视频量化等级=video_crf,
        原片帧索引=source_indices,
        未处理后续块=True,
        帧映射=mapping,
        参考逐帧对齐=source_positions is not None,
        跨帧固定字形系数=coefficients is not None,
        联合掩膜最大面积比例=max_area,
    )
    (output / "native-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        "高清合成完成，清晰度最低记录：" + str(report["清晰度最小保留比"]), flush=True
    )


def main():
    cv2.setNumThreads(2)
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "source",
        "coarse",
        "detected",
        "prior",
        "calibration",
        "template",
        "graphic",
        "output",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--duration", type=float, default=25)
    p.add_argument("--clean-reference-frames", type=Path)
    p.add_argument("--projection-calibration", type=Path)
    p.add_argument("--output-fps", type=float)
    p.add_argument("--detail-sigma", type=float, default=1.2)
    p.add_argument("--crf", type=int, default=16)
    p.add_argument(
        "--final-recompose",
        action="store_true",
        help="执行主分支七版原始第二遍细节合成",
    )
    p.add_argument(
        "--backend",
        choices=["numpy", "torch-cpu", "torch-cuda"],
        default="numpy",
        help="纹理合成后端",
    )
    p.add_argument(
        "--fused-detail",
        action="store_true",
        help="在内存中继续执行七版标准差三的细节融合",
    )
    p.add_argument(
        "--encoder",
        choices=["libx264", "h264_nvenc"],
        default="libx264",
        help="最终高质量编码器",
    )
    a = p.parse_args()
    if not 0 < a.duration <= 30:
        p.error("仅允许30秒以内的单块")
    render(
        a.source,
        a.coarse,
        a.detected,
        a.prior,
        a.calibration,
        a.template,
        a.graphic,
        a.output,
        a.duration,
        a.clean_reference_frames,
        a.projection_calibration,
        a.output_fps,
        a.detail_sigma,
        a.crf,
        final_recompose=a.final_recompose,
        backend=a.backend,
        fused_detail=a.fused_detail,
        encoder_name=a.encoder,
    )


if __name__ == "__main__":
    main()
