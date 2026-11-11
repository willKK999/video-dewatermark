import json
from fractions import Fraction
from .process import PipelineError, run


def probe(path, check=lambda: None, count=False):
    args = ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe"]
    if count:
        args += ["-count_frames"]
    try:
        return json.loads(
            run(
                args + ["-show_streams", "-show_format", "-of", "json", path],
                check,
                timeout=60,
            )
        )
    except PipelineError as e:
        if e.code != "PROCESS_FAILED":
            raise
        raise PipelineError("INVALID_MEDIA", "media cannot be probed") from e
    except ValueError as e:
        raise PipelineError("INVALID_MEDIA", "invalid probe output") from e


def plan(info, short_edge=720):
    videos = [s for s in info["streams"] if s["codec_type"] == "video"]
    audio = [s for s in info["streams"] if s["codec_type"] == "audio"]
    fmt = info.get("format", {})
    if (
        not set(fmt.get("format_name", "").split(",")) & {"mov", "mp4"}
        or len(videos) != 1
        or len(audio) > 1
    ):
        raise PipelineError(
            "INVALID_MEDIA",
            "only MP4/MOV with one video and at most one audio track supported",
        )
    v = videos[0]
    try:
        duration = float(fmt["duration"])
        rate = float(Fraction(v.get("avg_frame_rate", "0/1")))
        w, h = int(v["width"]), int(v["height"])
    except (KeyError, ValueError, ZeroDivisionError, OverflowError) as e:
        raise PipelineError(
            "INVALID_MEDIA", "missing media timing or dimensions"
        ) from e
    if (
        v.get("codec_name") != "h264"
        or v.get("pix_fmt") not in ("yuv420p", "yuvj420p")
        or v.get("color_transfer") in ("smpte2084", "arib-std-b67")
    ):
        raise PipelineError("INVALID_MEDIA", "only H.264 8-bit SDR 4:2:0 is supported")
    if not (0 < duration <= 30 and 0 < rate <= 240 and 0 < w <= 8192 and 0 < h <= 8192):
        raise PipelineError(
            "INVALID_MEDIA", "duration, frame rate or dimensions exceed limits"
        )
    # Non-square pixel material requires an explicit display-aspect policy.
    if v.get("sample_aspect_ratio", "1:1") not in ("1:1", "0:1", "N/A"):
        raise PipelineError(
            "INVALID_MEDIA", "non-square pixels are not supported in v1"
        )
    rotation = float(v.get("tags", {}).get("rotate", 0))
    for side in v.get("side_data_list", []):
        rotation = float(side.get("rotation", rotation))
    if rotation % 90 != 0:
        raise PipelineError("INVALID_MEDIA", "only orthogonal rotations supported")
    if int(abs(rotation)) % 180 == 90:
        w, h = h, w
    scale = min(1, 1280 / max(w, h), short_edge / min(w, h))
    width, height = max(2, int(w * scale) // 2 * 2), max(2, int(h * scale) // 2 * 2)
    fps = min(rate, 30)
    return dict(
        width=width,
        height=height,
        source_width=w,
        source_height=h,
        fps=fps,
        duration=duration,
        has_audio=bool(audio),
        source_start_ms=round(float(fmt.get("start_time", 0)) * 1000),
    )


def normalize(source, root, short_edge=720, check=lambda: None):
    if source.stat().st_size > 200 * 1024 * 1024:
        raise PipelineError("INVALID_MEDIA", "input exceeds 200 MiB")
    info = plan(probe(source, check), short_edge)
    normalized = root / "normalized.mp4"
    fps = info["fps"]
    # Default ffmpeg timestamp rebasing preserves relative audio/video offsets.
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-protocol_whitelist",
        "file,pipe",
        "-i",
        source,
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        f"scale={info['width']}:{info['height']},fps={fps}:start_time=0,setsar=1",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-af",
        "aresample=async=1:first_pts=0",
        "-map_metadata",
        "-1",
        normalized,
    ]
    try:
        run(args, check)
    except PipelineError as exc:
        if exc.code != "PROCESS_FAILED":
            raise
        raise PipelineError("INVALID_MEDIA", "input cannot be normalized") from exc
    frames = root / "frames"
    frames.mkdir()
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            normalized,
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            "901",
            "-start_number",
            "0",
            frames / "%06d.png",
        ],
        check,
    )
    files = sorted(frames.glob("*.png"))
    if not 1 <= len(files) <= 900:
        raise PipelineError("INVALID_MEDIA", "normalized frame count must be 1–900")
    info["frame_count"] = len(files)
    info["duration_ms"] = round(len(files) * 1000 / fps)
    info["timeline"] = {
        "source_start_ms": info["source_start_ms"],
        "normalized_start_ms": 0,
        "fps": fps,
        "mapping": "source_start_ms + frame_index * 1000 / fps",
    }
    return info


def validate_output(path, info, check=lambda: None):
    try:
        _validate_output(path, info, check)
    except PipelineError as exc:
        if exc.code not in ("INVALID_MEDIA", "PROCESS_FAILED"):
            raise
        raise PipelineError("OUTPUT_INVALID", "output is not decodable") from exc
    except (KeyError, ValueError, StopIteration) as exc:
        raise PipelineError("OUTPUT_INVALID", "output metadata is incomplete") from exc


def _validate_output(path, info, check):
    output = probe(path, check, count=True)
    video = next(s for s in output["streams"] if s["codec_type"] == "video")
    count = int(video.get("nb_read_frames", 0))
    if (
        count != info["frame_count"]
        or video["width"] != info["width"]
        or video["height"] != info["height"]
        or video.get("codec_name") != "h264"
        or video.get("pix_fmt") != "yuv420p"
        or abs(float(video["duration"]) * 1000 - info["duration_ms"]) > 80
    ):
        raise PipelineError(
            "OUTPUT_INVALID", "output frames, geometry or duration differ"
        )
    audio = [s for s in output["streams"] if s["codec_type"] == "audio"]
    if bool(audio) != info["has_audio"]:
        raise PipelineError("OUTPUT_INVALID", "main audio track missing")
    if audio:
        a = audio[0]
        if (
            abs(float(a.get("start_time", 0)) - float(video.get("start_time", 0)))
            > 0.08
            or abs(float(a["duration"]) - float(video["duration"])) > 0.08
        ):
            raise PipelineError(
                "OUTPUT_INVALID", "audio/video timing differs by over 80 ms"
            )
    run(
        ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", path, "-f", "null", "-"],
        check,
    )


def encode(root, info, check=lambda: None):
    dest = root / "result.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-framerate",
            str(info["fps"]),
            "-start_number",
            "0",
            "-i",
            root / "composite" / "%06d.png",
            "-i",
            info.get("audio_source", root / "normalized.mp4"),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "copy",
            "-t",
            str(info["frame_count"] / info["fps"]),
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
            dest,
        ],
        check,
    )
    validate_output(dest, info, check)
    return dest


def normalize_direct(source, root, short_edge=720, check=lambda: None):
    """直接解码为无损帧；音频保留源引用，省去中间视频和音频转码。"""
    if source.stat().st_size > 200 * 1024 * 1024:
        raise PipelineError("INVALID_MEDIA", "输入文件超过二百兆字节")
    info = plan(probe(source, check), short_edge)
    frames = root / "frames"
    frames.mkdir()
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            f"scale={info['width']}:{info['height']},fps={info['fps']}:start_time=0,setsar=1",
            "-frames:v",
            "901",
            "-start_number",
            "0",
            frames / "%06d.png",
        ],
        check,
    )
    files = sorted(frames.glob("*.png"))
    if not 1 <= len(files) <= 900:
        raise PipelineError("INVALID_MEDIA", "直接解码的帧数必须为一至九百")
    info.update(
        frame_count=len(files),
        duration_ms=round(len(files) * 1000 / info["fps"]),
        audio_source=str(source),
    )
    info["timeline"] = dict(
        source_start_ms=info["source_start_ms"],
        normalized_start_ms=0,
        fps=info["fps"],
        mapping="source_start_ms + frame_index * 1000 / fps",
    )
    return info
