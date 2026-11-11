"""视频去水印系统：只运行一个带上下文的处理块，绝不调度后续块。"""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from worker import media, vision
from worker.pipeline import execute
from worker.process import run, PipelineError
from worker.hls import seam_metrics


def audio_hash(path, duration=None):
    args = ["ffmpeg", "-v", "error", "-i", path]
    if duration is not None:
        args += ["-t", str(duration)]
    return hashlib.sha256(
        run(args + ["-map", "0:a:0", "-c:a", "copy", "-f", "adts", "-"])
    ).hexdigest()


def mux_core_audio(video, source, destination, duration):
    """视频帧数限制不能参与音频封装，否则可能提前截掉最后一个AAC包。"""
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            video,
            "-i",
            source,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c",
            "copy",
            "-t",
            str(duration),
            "-movflags",
            "+faststart",
            destination,
        ]
    )


def cut_references(detections, first, last):
    result = dict(detections)
    result["scenes"] = [
        [max(a, first) - first, min(b, last) - first]
        for a, b in detections["scenes"]
        if a < last and b > first
    ]
    result["tracks"] = [
        dict(
            t,
            hints=[
                dict(p, frame=p["frame"] - first)
                for p in t["hints"]
                if first <= p["frame"] < last
            ],
        )
        for t in detections["tracks"]
    ]
    result["tracks"] = [t for t in result["tracks"] if t["hints"]]
    result["hints"] = sorted(
        [p for t in result["tracks"] for p in t["hints"]],
        key=lambda p: (p["frame"], p["track_id"]),
    )
    return result


def run_one(source, prior, output, duration=25, context=2, resume_root=None):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    work = resume_root or output / "work"
    if not resume_root:
        work.mkdir()
    source_info = media.probe(source)
    video = next(s for s in source_info["streams"] if s["codec_type"] == "video")
    fps = min(30, float(Fraction(video["avg_frame_rate"])))
    core = round(duration * fps)
    count = round((duration + context) * fps)
    clip = work / "input.mp4"
    refinements = []
    extraction_seconds = 0
    phases = {}
    if resume_root:
        files = sorted((work / "frames").glob("*.png"))
        if len(files) != count:
            raise ValueError("续跑帧数与处理块上下文不一致")
        detections = json.loads((work / "detections.json").read_text())
        if not detections.get("multitrack"):
            raise ValueError("续跑必须包含完整多轨迹检测")
        checked = time.monotonic()
        manifest = vision.validate_masks(work, files, detections)
        (work / "validated-masks.json").write_text(json.dumps(manifest))
        phases["mask_validation"] = round((time.monotonic() - checked) * 1000)
        model = json.loads((work / "metrics.sam.json").read_text())
        phases["mask"] = round(model["segmentation_seconds"] * 1000)
        checked = time.monotonic()
        print("单块阶段：inpaint", flush=True)
        vision.inpaint(work, files, detections["scenes"], "propainter", False)
        phases["inpaint"] = round((time.monotonic() - checked) * 1000)
    else:
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                source,
                "-an",
                "-vf",
                f"fps={fps}",
                "-frames:v",
                str(count),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                clip,
            ]
        )
        extraction_seconds = time.monotonic() - started
        try:
            result, info, phases = execute(
                clip,
                work,
                dict(hls=True, multitrack_prior=str(prior.resolve())),
                "propainter",
                False,
                lambda s: print("单块阶段：" + s, flush=True),
            )
        except PipelineError as exc:
            if exc.code != "RESIDUAL_UNCERTAIN" or not (work / "quality.json").exists():
                raise
            phases = json.loads((work / "phase_ms.json").read_text())
        files = sorted((work / "frames").glob("*.png"))
        detections = json.loads((work / "detections.json").read_text())
    try:
        vision.residual_quality(work, files, detections)
    except PipelineError as exc:
        if exc.code != "RESIDUAL_UNCERTAIN":
            raise
        from worker.hls_repair import refine

        refinements = refine(
            work,
            files,
            detections,
            "propainter",
            False,
            fps,
            lambda s: print("单块复修阶段：" + s, flush=True),
            lambda: None,
        )
        phases["residual_refine"] = sum(p["elapsed_ms"] for p in refinements)
    (work / "phase_ms.json").write_text(json.dumps(phases))
    refs = cut_references(detections, 0, core)
    final = output / "chunk1.mp4"
    silent = output / "core-video.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            work / "composite/%06d.png",
            "-map",
            "0:v:0",
            "-frames:v",
            str(core),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            silent,
        ]
    )
    mux_core_audio(silent, source, final, duration)
    return verify_one(
        source,
        prior,
        output,
        duration,
        context,
        fps,
        work,
        started,
        phases,
        refinements,
        extraction_seconds,
        resume_root,
    )


def verify_one(
    source,
    prior,
    output,
    duration,
    context,
    fps,
    work,
    started,
    phases,
    refinements,
    extraction_seconds,
    resume_root,
    previous=None,
):
    core = round(duration * fps)
    refs = cut_references(json.loads((work / "detections.json").read_text()), 0, core)
    final = output / "chunk1.mp4"
    encoded = output / "encoded"
    encoded.mkdir(exist_ok=True)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            final,
            "-start_number",
            "0",
            encoded / "%06d.png",
        ]
    )
    audit = output / "audit"
    if audit.exists():
        audit.rename(output / f"audit-before-{time.time_ns()}")
    audit.mkdir()
    (audit / "composite").symlink_to(encoded.resolve(), target_is_directory=True)
    (audit / "detections.json").write_text(json.dumps(refs))
    vision.residual_quality(audit, sorted(encoded.glob("*.png")), refs)
    quality = json.loads((audit / "quality.json").read_text())
    out_info = media.probe(final, count=True)
    ov = next(s for s in out_info["streams"] if s["codec_type"] == "video")
    oa = next(s for s in out_info["streams"] if s["codec_type"] == "audio")
    audio_equal = audio_hash(source, duration) == audio_hash(final)

    def audio_packets(path):
        return json.loads(
            run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_packets",
                    "-show_entries",
                    "packet=pts_time,duration_time",
                    "-of",
                    "json",
                    path,
                ]
            )
        )["packets"]

    baseline = [
        p for p in audio_packets(source) if 0 <= float(p["pts_time"]) < duration
    ]
    actual = audio_packets(final)
    pts_error = max(
        (
            abs(float(a["pts_time"]) - float(b["pts_time"]))
            for a, b in zip(baseline, actual)
        ),
        default=0,
    )
    av_end = abs(float(ov["duration"]) - float(oa["duration"]))
    performance = json.loads((work / "performance.json").read_text())
    boundaries = []
    for a, b in refs["scenes"]:
        boundaries += [left for left, right in vision.windows(a, b, 36, 8) if left > a]
    selected = sorted(
        set(
            [
                boundaries[i]
                for i in np.linspace(
                    0, len(boundaries) - 1, min(6, len(boundaries)), dtype=int
                )
            ]
            if boundaries
            else []
        )
    )
    changes = []
    for track in refs["tracks"]:
        if track["kind"] == "moving":
            a, b = track["hints"][0]["frame"], track["hints"][-1]["frame"] + 1
            changes += [i for i in [a, b] if 0 < i < core]
    selected = sorted(set(selected + changes))
    seams = seam_metrics(
        source,
        encoded,
        [dict(start=0)] + [dict(start=i) for i in selected],
        fps,
        audit,
        lambda: None,
        refs["hints"],
        final,
    )
    comparisons = output / "对比图"
    comparisons.mkdir(exist_ok=True)
    seconds = sorted(
        set(list(range(int(np.ceil(duration)))) + [round(i / fps, 3) for i in changes])
    )
    cap = cv2.VideoCapture(str(source))
    for second in seconds:
        frame = min(core - 1, round(second * fps))
        cap.set(cv2.CAP_PROP_POS_MSEC, frame * 1000 / fps)
        ok, image = cap.read()
        if not ok:
            raise ValueError("原片对应时间戳取帧失败")
        clean = cv2.imread(str(encoded / f"{frame:06d}.png"))
        h, w = clean.shape[:2]
        image = cv2.resize(image, (w, h))
        cv2.imwrite(
            str(comparisons / f"全帧-{frame/fps:06.3f}秒.jpg"),
            np.hstack([image, clean]),
        )
        for label, (a, b, c, d) in {
            "左上": (0, 0, w // 3, h // 3),
            "右下": (w * 2 // 3, h * 2 // 3, w, h),
        }.items():
            cv2.imwrite(
                str(comparisons / f"{label}-{frame/fps:06.3f}秒.jpg"),
                np.hstack([image[b:d, a:c], clean[b:d, a:c]]),
            )
        for track in refs["tracks"]:
            point = next((p for p in track["hints"] if p["frame"] == frame), None)
            if point is None:
                continue
            a, b, c, d = point["box"]
            a, b = max(0, int(a) - 12), max(0, int(b) - 12)
            c, d = min(w, int(np.ceil(c)) + 12), min(h, int(np.ceil(d)) + 12)
            cv2.imwrite(
                str(comparisons / f'轨迹{track["id"]:02d}-{frame/fps:06.3f}秒.jpg'),
                np.hstack([image[b:d, a:c], clean[b:d, a:c]]),
            )
    cap.release()
    passed = (
        quality["residual_count"] == 0
        and not quality["needs_review"]
        and int(ov["nb_read_frames"]) == core
        and audio_equal
        and len(baseline) == len(actual)
        and pts_error <= 0.001
        and av_end <= 0.08
        and performance.get("overlap_max_mae", 0) <= 35
    )
    report = {
        "通过": passed,
        "范围": "仅第1块；没有调度或处理后续块",
        "主体时长秒": duration,
        "输入上下文时长秒": duration + context,
        "右侧上下文秒": context,
        "帧数": int(ov["nb_read_frames"]),
        "帧率": fps,
        "输出尺寸": [ov["width"], ov["height"]],
        "视频时长秒": float(ov["duration"]),
        "音频时长秒": float(oa["duration"]),
        "音频逐字节复制": audio_equal,
        "音频包数原片与成品": [len(baseline), len(actual)],
        "音频最大时间戳差秒": pts_error,
        "音画尾差秒": av_end,
        "编码后复检": quality,
        "模型窗口指标": performance,
        "时域检查": seams,
        "接缝范围说明": "仅验证单块内部模型窗和水印出现消失处；未检验相邻HLS处理块接缝",
        "源视频摘要": vision.sha256(source),
        "先验摘要": vision.sha256(prior),
        "阶段耗时毫秒": phases,
        "自动复修": refinements,
        "输入提取耗时秒": extraction_seconds,
        "总耗时秒": time.monotonic() - started,
        "续跑复用准备": bool(resume_root),
        "耗时口径": (
            "总耗时为本次执行实测墙时；续跑时不含已经完成的提取、检测和SAM分割，SAM实测耗时另列于阶段数据"
            if resume_root
            else "总耗时包含完整单块处理及验收"
        ),
        "本次与已完成分割累计实测秒": time.monotonic()
        - started
        + (phases.get("mask", 0) / 1000 if resume_root else 0),
        "输出文件": "chunk1.mp4",
        "目检": "待检查对比图及视频；算法结果不能代替目检",
    }
    if previous:
        elapsed = time.monotonic() - started
        report["前次实测续跑耗时秒"] = previous["总耗时秒"]
        report["本次重新封装复检耗时秒"] = elapsed
        report["总耗时秒"] = previous["总耗时秒"] + elapsed
        report["本次与已完成分割累计实测秒"] = (
            report["总耗时秒"] + phases.get("mask", 0) / 1000
        )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(
        "单块自动验收"
        + ("通过" if passed else "未通过")
        + "；耗时秒："
        + str(round(report["总耗时秒"], 3)),
        flush=True,
    )
    if not passed:
        raise ValueError("单块未达到原有验收门槛")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=25)
    p.add_argument("--context", type=float, default=2)
    p.add_argument("--resume-root", type=Path)
    p.add_argument("--recheck", action="store_true")
    a = p.parse_args()
    if a.duration <= 0 or a.context < 0 or a.duration + a.context > 30:
        p.error("主体与上下文总时长必须在30秒以内")
    if a.recheck:
        if not a.resume_root:
            p.error("重新封装复检需要提供原处理目录")
        started = time.monotonic()
        previous = json.loads((a.output / "report.json").read_text())
        (a.output / f"report-before-{time.time_ns()}.json").write_text(
            json.dumps(previous, ensure_ascii=False, indent=2)
        )
        silent = a.output / "core-video.mp4"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                a.output / "chunk1.mp4",
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-an",
                silent,
            ]
        )
        mux_core_audio(silent, a.source, a.output / "chunk1.mp4", a.duration)
        verify_one(
            a.source,
            a.prior,
            a.output,
            a.duration,
            a.context,
            previous["帧率"],
            a.resume_root,
            started,
            previous["阶段耗时毫秒"],
            previous["自动复修"],
            previous["输入提取耗时秒"],
            a.resume_root,
            previous,
        )
    else:
        run_one(a.source, a.prior, a.output, a.duration, a.context, a.resume_root)


if __name__ == "__main__":
    main()
