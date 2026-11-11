"""视频去水印系统：原分辨率、原帧率合成单块；只在自动支持范围内替换像素。"""

import argparse, json, time, subprocess, math
from pathlib import Path
from fractions import Fraction
from functools import lru_cache
import cv2, numpy as np
from worker.fine_masks import Supports
from worker.matte import (
    projected_texture_composite,
    temporal_detail,
    stable_detail_mask,
)
from worker.restoration import soft_composite
from worker.media import probe
from scripts.hls_chunk_check import mux_core_audio, audio_hash


def source_frame_for_output(frame, source_fps, output_fps):
    if not 0 < output_fps <= source_fps:
        raise ValueError("输出帧率必须大于零且不高于原片")
    return math.floor(frame * source_fps / output_fps)


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
):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
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
    encoder = subprocess.Popen(
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
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            str(video_crf),
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
    cap = cv2.VideoCapture(str(source))
    stats = []
    reference_counts = []
    previous = None
    mapping = []
    max_area = 0.0
    source_cursor = -1
    source_indices = []
    try:
        for frame in range(count):
            source_index = source_frame_for_output(frame, source_fps, fps)
            source_indices.append(source_index)
            while source_cursor < source_index:
                ok, original = cap.read()
                if not ok:
                    raise ValueError("原片主体帧不足")
                source_cursor += 1
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
                    painted = soft_composite(original, base, mask, 2)
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
                painted = projected_texture_composite(
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
                # v7 的第二遍联合纹理合成；在同一原生帧上完成，免去中间有损编码。
                painted = projected_texture_composite(
                    original,
                    result,
                    faint_union,
                    templates,
                    calibration=coefficients if faint else None,
                    detail_sigma=detail_sigma,
                )
                result[faint_union > 0] = painted[faint_union > 0]
            area = float(np.count_nonzero(occupied) / occupied.size)
            max_area = max(max_area, area)
            if layer_proof:
                layer_proof.validate(index, occupied)
            elif area > 0.2:
                raise ValueError("联合掩膜超过20%且缺少逐层保护与来源证明")
            encoder.stdin.write(result.tobytes())
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
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise ValueError("原规格视频编码失败")
    finally:
        cap.release()
        errors.close()
        if encoder.poll() is None:
            encoder.stdin.close()
            encoder.terminate()
            try:
                encoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
    final = output / "chunk1.mp4"
    mux_core_audio(silent, source, final, duration)
    report = dict(
        视频规格=[width, height, fps],
        主体秒数=duration,
        帧数=count,
        音频内容一致=audio_hash(source, duration) == audio_hash(final),
        清晰度仅记录=True,
        清晰度历史参考阈值=0.8,
        清晰度统计=stats,
        清晰度通过=all(s["passed"] for s in stats)
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
        "高清合成完成，清晰度历史参考结果（仅记录）：" + str(report["清晰度通过"]),
        flush=True,
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
    )


if __name__ == "__main__":
    main()
