"""沿用既有校验规则，把原规格视频封装为保持原分片时间轴的 HLS。"""

import json, time, zipfile
from .hls import audio_timeline_metrics, seam_metrics
from .media import probe
from .process import PipelineError, run
from .vision import sha256


def package(
    full,
    output,
    root,
    segments,
    lines,
    source_pts,
    plans,
    fps,
    source_fps,
    times,
    digest,
    cache_hash,
    started,
    references,
    check,
):
    durations = [d for d, _ in segments]
    cuts = [sum(durations[:i]) for i in range(1, len(durations))]
    info = probe(full, check)
    delivery = root / "hls"
    delivery.mkdir()
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-i",
        output,
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "disabled",
        "-f",
        "segment",
        "-segment_time_delta",
        str(0.5 / fps),
        "-segment_list",
        delivery / "actual.csv",
        "-segment_list_type",
        "csv",
        "-segment_format",
        "mpegts",
        "-segment_format_options",
        "mpegts_copyts=1:avoid_negative_ts=disabled:pes_payload_size=0",
        "-reset_timestamps",
        "0",
    ]
    if cuts:
        args += ["-segment_times", ",".join(str(x) for x in cuts)]
    run(args + [delivery / "%06d.ts"], check)
    emitted = sorted(delivery.glob("*.ts"))
    if len(emitted) != len(segments):
        raise PipelineError("OUTPUT_INVALID", "输出分片数量不一致")
    timing = []
    for i, path in enumerate(emitted):
        remux = delivery / "shifted.ts"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-copyts",
                "-i",
                path,
                "-map",
                "0",
                "-c",
                "copy",
                "-output_ts_offset",
                str(source_pts),
                "-avoid_negative_ts",
                "disabled",
                "-mpegts_copyts",
                "1",
                "-mpegts_flags",
                "+initial_discontinuity",
                "-pes_payload_size",
                "0",
                "-muxdelay",
                "0",
                remux,
            ],
            check,
        )
        remux.replace(path)
        packets = json.loads(
            run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_packets",
                    "-show_entries",
                    "packet=pts_time,duration_time,flags",
                    "-of",
                    "json",
                    path,
                ],
                check,
            )
        )["packets"]
        first_pts = float(packets[0]["pts_time"])
        expected = source_pts + sum(durations[:i])
        if (
            "K" not in packets[0]["flags"]
            or abs(first_pts - expected) > 1 / fps + 0.001
        ):
            raise PipelineError("OUTPUT_INVALID", "分片起点关键帧或 PTS 校验失败")
        pts = sorted(float(x["pts_time"]) for x in packets)
        if i and abs(pts[0] - timing[-1]["end_pts"]) > 1 / fps + 0.001:
            raise PipelineError("OUTPUT_INVALID", "分片 PTS 不连续")
        timing.append(
            dict(
                index=i,
                start_pts=pts[0],
                end_pts=pts[-1] + 1 / fps,
                expected_pts=expected,
            )
        )
    # 保留媒体序号、目标时长、EXTINF 等元信息，仅替换分片 URI。
    out_lines = []
    n = 0
    for line in lines:
        if not line.startswith("#"):
            out_lines.append(emitted[n].name)
            n += 1
        else:
            out_lines.append(line)
    (delivery / "index.m3u8").write_text("\n".join(out_lines) + "\n")
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-i",
            delivery / "index.m3u8",
            "-f",
            "null",
            "-",
        ],
        check,
    )
    # 交付 MP4 明确由新播放列表无损合并，保持源片相对时间轴。
    remuxed = root / "result-from-hls.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-copyts",
            "-i",
            delivery / "index.m3u8",
            "-map",
            "0",
            "-c",
            "copy",
            "-output_ts_offset",
            str(-source_pts),
            "-avoid_negative_ts",
            "disabled",
            "-movflags",
            "+faststart",
            remuxed,
        ],
        check,
        timeout=1800,
    )
    remuxed.replace(output)
    output_info = probe(output, check, count=True)
    output_video = next(x for x in output_info["streams"] if x["codec_type"] == "video")
    if (
        int(output_video["nb_read_frames"]) != round(sum(durations) * fps)
        or abs(float(output_video["duration"]) - sum(durations)) > 1 / fps + 0.001
    ):
        raise PipelineError("OUTPUT_INVALID", "HLS 成品帧数或时长不一致")
    audio_copy = True
    audio_timing = {}
    if any(x["codec_type"] == "audio" for x in info["streams"]):

        def audio_hash(path):
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    path,
                    "-map",
                    "0:a:0",
                    "-c",
                    "copy",
                    "-f",
                    "adts",
                    root / "audio-check.aac",
                ],
                check,
            )
            return sha256(root / "audio-check.aac")

        # 有界音频内容摘要，确保没有转码或丢包。
        source_audio = audio_hash(full)
        audio_copy = (
            source_audio == audio_hash(output) == audio_hash(delivery / "index.m3u8")
        )
        if not audio_copy:
            raise PipelineError("OUTPUT_INVALID", "压缩音频内容发生变化")
    if any(x["codec_type"] == "audio" for x in info["streams"]):
        audio_timing = audio_timeline_metrics(
            full, output, delivery / "index.m3u8", source_pts, check
        )
    seams = seam_metrics(full, None, plans, fps, root, check, references, output)
    report = dict(
        source_fps=source_fps,
        processing_fps=fps,
        chunks=times,
        seams=seams,
        elapsed_ms=round((time.monotonic() - started) * 1000),
        source_sha256=digest,
        checkpoint_fingerprint=cache_hash,
        segment_timing=timing,
        source_start_pts=source_pts,
        residual_count=0,
        clarity_record_only=True,
        output_fps=fps,
        audio_copy_verified=audio_copy,
        **audio_timing,
    )
    (root / "metrics-hls.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    with zipfile.ZipFile(
        root / "hls.zip", "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for path in sorted(delivery.iterdir()):
            if path.suffix in (".ts", ".m3u8"):
                archive.write(path, path.name)
    result = dict(
        hls=True,
        chunk_count=len(plans),
        duration_ms=round(sum(durations) * 1000),
        fps=fps,
        width=output_video["width"],
        height=output_video["height"],
        seams=seams,
        audio_copy_verified=audio_copy,
        residual_count=0,
    )
    return output, result, {"hls_total": report["elapsed_ms"]}
