"""视频去水印系统：有界 HLS 快照、带上下文分块与连续时间轴封装。"""

import http.client
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import socket
import time
import urllib.parse
import urllib.request
import zipfile

import cv2
import numpy as np
from botocore.exceptions import ClientError
from .media import probe
from .pipeline import execute
from .process import LeaseLost, PipelineError, run


def invalid(message):
    return PipelineError("HLS_UNSUPPORTED", message)


def resolve(base, child):
    if urllib.parse.urlsplit(base).scheme in ("http", "https"):
        return urllib.parse.urljoin(base, child)
    parsed = urllib.parse.urlsplit(child)
    if parsed.scheme:
        return child
    if parsed.netloc:
        return "https:" + child
    import posixpath

    path = urllib.parse.unquote(parsed.path)
    if path.startswith("/"):
        return posixpath.normpath(path).lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(base), path))


def public_url(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme not in ("http", "https") or p.username or p.password or not p.hostname:
        raise invalid("仅支持公开 HTTP/HTTPS 地址")
    try:
        addresses = socket.getaddrinfo(
            p.hostname, p.port or (443 if p.scheme == "https" else 80)
        )
        if not addresses or any(
            not ipaddress.ip_address(a[4][0]).is_global for a in addresses
        ):
            raise invalid("不允许访问本机或内网地址")
    except (ValueError, OSError):
        raise invalid("无法解析公开媒体地址") from None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise invalid("媒体地址重定向不受支持，请提交最终地址")


def public_connection(
    address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None
):
    """直连固定到已验证 IP；使用环境代理时由既有代理执行目的地隔离。"""
    host, port = address
    proxy_hosts = {
        urllib.parse.urlsplit(v).hostname
        for v in urllib.request.getproxies().values()
        if "://" in v
    }
    if host in proxy_hosts:
        return socket.create_connection(address, timeout, source_address)
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(a[4][0]).is_global for a in addresses
    ):
        raise invalid("不允许连接本机或内网地址")
    for family, kind, protocol, _, sockaddr in addresses:
        connection = socket.socket(family, kind, protocol)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                connection.settimeout(timeout)
            if source_address:
                connection.bind(source_address)
            connection.connect(sockaddr)
            return connection
        except OSError:
            connection.close()
    raise OSError("公开媒体连接失败")


class PublicHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = public_connection


class PublicHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = public_connection


class PublicHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, request):
        return self.do_open(PublicHTTPConnection, request)


class PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(PublicHTTPSConnection, request, context=self._context)


class Reader:
    def __init__(self, s3, bucket, check, key_prefix=None):
        self.s3, self.bucket, self.check = s3, bucket, check
        self.total = 0
        self.key_prefix = key_prefix

    def read(self, source, limit):
        self.check()
        try:
            parsed = urllib.parse.urlsplit(source)
            if parsed.scheme in ("http", "https"):
                endpoints = [
                    urllib.parse.urlsplit("//" + os.getenv(name, "")).hostname
                    for name in ("S3_ENDPOINT", "S3_PUBLIC_ENDPOINT")
                    if os.getenv(name)
                ]
                object_path = urllib.parse.unquote(parsed.path)
                if parsed.hostname in endpoints and object_path.startswith(
                    "/" + self.bucket + "/"
                ):
                    source = object_path[len(self.bucket) + 2 :]
                elif parsed.hostname in [
                    self.bucket + "." + host for host in endpoints
                ]:
                    source = object_path.lstrip("/")
            if urllib.parse.urlsplit(source).scheme:
                public_url(source)
                stream = urllib.request.build_opener(
                    NoRedirect(), PublicHTTPHandler(), PublicHTTPSHandler()
                ).open(source, timeout=30)
            else:
                if (
                    source.startswith("/")
                    or source.startswith("../")
                    or any(part in (".", "..") for part in source.split("/"))
                    or (self.key_prefix and not source.startswith(self.key_prefix))
                ):
                    raise invalid("对象键无效")
                stream = self.s3.get_object(Bucket=self.bucket, Key=source)["Body"]
            data = bytearray()
            try:
                while chunk := stream.read(min(1024 * 1024, limit + 1 - len(data))):
                    self.check()
                    data.extend(chunk)
                    self.total += len(chunk)
                    if len(data) > limit or self.total > 2 * 1024**3:
                        raise invalid("HLS 素材超过下载大小限制")
            finally:
                stream.close()
            return bytes(data)
        except (PipelineError, LeaseLost):
            raise
        except Exception:
            raise PipelineError(
                "HLS_DOWNLOAD", "HLS 素材读取失败", retryable=True
            ) from None


def parse(text):
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines or lines[0] != "#EXTM3U":
        raise invalid("无效的 m3u8 播放列表")
    variants, segments = [], []
    duration = None
    for i, line in enumerate(lines):
        if line.startswith("#EXT-X-STREAM-INF:"):
            rate = re.search(r"(?:[:,])BANDWIDTH=(\d+)", line)
            if not rate or i + 1 == len(lines) or lines[i + 1].startswith("#"):
                raise invalid("主播放列表缺少码率或地址")
            if "AUDIO=" in line:
                raise invalid("暂不支持独立音频播放列表")
            variants.append((int(rate[1]), lines[i + 1]))
        if line.startswith(("#EXT-X-KEY:", "#EXT-X-SESSION-KEY:")) and not re.search(
            r"(?:^|,)METHOD=NONE(?:,|$)", line.split(":", 1)[1]
        ):
            raise invalid("暂不支持加密 HLS（包括 AES-128），请提供解密后的素材")
        if line.startswith(
            (
                "#EXT-X-BYTERANGE",
                "#EXT-X-MAP",
                "#EXT-X-DISCONTINUITY",
                "#EXT-X-GAP",
                "#EXT-X-DEFINE",
            )
        ):
            raise invalid("暂不支持字节范围、fMP4、时间轴中断或缺失分片")
        if line.startswith("#EXTINF:"):
            try:
                duration = float(line.split(":", 1)[1].split(",")[0])
            except ValueError:
                raise invalid("分片时长无效") from None
            if not math.isfinite(duration) or duration <= 0:
                raise invalid("分片时长无效")
        elif not line.startswith("#") and duration is not None:
            segments.append((duration, line))
            duration = None
    if variants:
        return lines, variants, []
    if (
        "#EXT-X-ENDLIST" not in lines
        or not segments
        or len(segments) > 2000
        or duration is not None
    ):
        raise invalid("仅支持完整的点播媒体播放列表（最多 2000 分片）")
    if sum(d for d, _ in segments) > 3600:
        raise invalid("HLS 总时长不得超过一小时")
    return lines, [], segments


def snapshot(source, reader, root, bandwidth=0):
    for _ in range(5):
        try:
            text = reader.read(source, 2 * 1024**2).decode("utf-8-sig")
        except UnicodeError:
            raise invalid("播放列表编码必须为 UTF-8") from None
        lines, variants, segments = parse(text)
        if not variants:
            break
        choices = [v for v in variants if not bandwidth or v[0] <= bandwidth]
        selected = max(choices, default=min(variants), key=lambda x: x[0])
        source = resolve(source, selected[1])
    else:
        raise invalid("播放列表嵌套过深")
    local = root / "source-hls"
    local.mkdir()
    names = []
    digest = hashlib.sha256()
    digest.update(text.encode())
    for i, (_, uri) in enumerate(segments):
        data = reader.read(resolve(source, uri), 200 * 1024**2)
        digest.update(data)
        name = f"{i:06d}.ts"
        (local / name).write_bytes(data)
        names.append(name)
    output, n = [], 0
    for line in lines:
        if not line.startswith("#"):
            output.append(names[n])
            n += 1
        else:
            output.append(line)
    (local / "index.m3u8").write_text("\n".join(output) + "\n")
    return local / "index.m3u8", lines, segments, digest.hexdigest()


def validate_segments(playlist, segments, check):
    """在修复前拒绝未声明的时间轴间断，不把错误时长写回成品。"""
    first_pts = None
    elapsed = 0
    for i, (duration, _) in enumerate(segments):
        info = probe(playlist.parent / f"{i:06d}.ts", check)
        videos = [x for x in info["streams"] if x["codec_type"] == "video"]
        audios = [x for x in info["streams"] if x["codec_type"] == "audio"]
        if len(videos) != 1 or len(audios) > 1:
            raise invalid("每个 HLS 分片必须有一条视频、至多一条音频")
        video = videos[0]
        from fractions import Fraction

        fps = float(Fraction(video["avg_frame_rate"]))
        if not fps > 0:
            raise invalid("分片帧率无效")
        pts = float(video["start_time"])
        if first_pts is None:
            first_pts = pts
        if (
            abs(pts - first_pts - elapsed) > 1 / fps + 0.001
            or abs(float(video["duration"]) - duration) > 1 / fps + 0.001
        ):
            raise invalid("原分片 PTS 或实际时长与播放列表不一致")
        elapsed += duration
    return first_pts


def chunks(durations, fps, context=2, target=24):
    """所有边界在同一帧网格，主体按分片对齐；过长分片在帧边界拆开。"""
    total = round(sum(durations) * fps)
    boundaries = [round(sum(durations[: i + 1]) * fps) for i in range(len(durations))]
    pad = round(context * fps)
    limit = min(math.floor((target + 2) * fps), math.floor(30 * fps) - 2 * pad)
    start, plans = 0, []
    while start < total:
        candidates = [b for b in boundaries if start < b <= start + limit]
        end = (
            min(candidates, key=lambda b: (abs((b - start) / fps - target), -b))
            if candidates
            else min(total, start + min(limit, math.floor(target * fps)))
        )
        plans.append(
            dict(
                start=start,
                end=end,
                left=max(0, start - pad),
                right=min(total, end + pad),
            )
        )
        start = end
    if len(plans) > 1:
        previous, tail = plans[-2:]
        if tail["end"] - previous["start"] <= limit and tail["right"] - previous[
            "left"
        ] <= math.floor(30 * fps):
            previous.update(end=tail["end"], right=tail["right"])
            plans.pop()
    return plans


def record_failed_chunk(root, work, index, chunk_started, started):
    """失败块也保留可审计耗时，不写入输入地址或异常详情。"""
    try:
        path = work / "phase_ms.json"
        phase = json.loads(path.read_text()) if path.exists() else {}
        measured = {
            f"hls_chunk_{index:02d}_{name}": value for name, value in phase.items()
        }
        measured[f"hls_chunk_{index:02d}_total"] = round(
            (time.monotonic() - chunk_started) * 1000
        )
        measured["hls_total_to_failure"] = round((time.monotonic() - started) * 1000)
        (root / "phase_ms.json").write_text(json.dumps(measured))
    except (OSError, ValueError):
        pass


def merge_frames(destination, incoming, plan, previous_right, fps, check):
    """对同一时间戳的修复画面渐变融合；不会混合不同时间的运动内容。"""
    files = sorted(incoming.glob("*.png"))
    expected = plan["right"] - plan["left"]
    if len(files) != expected:
        raise PipelineError("OUTPUT_INVALID", "处理块帧数与全片时间轴不符")
    overlap_end = min(previous_right, plan["right"])
    for i, path in enumerate(files):
        check()
        index = plan["left"] + i
        out = destination / f"{index:08d}.png"
        if index < previous_right and out.exists():
            weight = (index - plan["left"] + 1) / (overlap_end - plan["left"] + 1)
            image = cv2.addWeighted(
                cv2.imread(str(out)), 1 - weight, cv2.imread(str(path)), weight, 0
            )
            if not cv2.imwrite(str(out), image):
                raise PipelineError("OUTPUT_INVALID", "接缝画面写入失败")
        else:
            shutil.copyfile(path, out)


def seam_metrics(
    source, frames, plans, fps, root, check, references=None, result_path=None
):
    records = []
    for i, p in enumerate(plans[1:], 1):
        n = p["start"]
        if result_path is None:
            a, b = [cv2.imread(str(frames / f"{j:08d}.png")) for j in (n - 1, n)]
        else:
            encoded = root / f"seam-result-{i}"
            encoded.mkdir()
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    result_path,
                    "-vf",
                    f"select=between(n\\,{n-1}\\,{n})",
                    "-fps_mode",
                    "passthrough",
                    "-frames:v",
                    "2",
                    encoded / "%d.png",
                ],
                check,
            )
            a, b = [cv2.imread(str(encoded / f"{j}.png")) for j in (1, 2)]
        sample = root / f"seam-source-{i}"
        sample.mkdir()
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                source,
                "-vf",
                f"select=between(n\\,{n-1}\\,{n}),scale={a.shape[1]}:{a.shape[0]}",
                "-fps_mode",
                "passthrough",
                "-frames:v",
                "2",
                sample / "%d.png",
            ],
            check,
        )
        raw = [cv2.imread(str(sample / f"{j}.png")) for j in (1, 2)]
        delta = lambda x, y: float(
            np.abs(x.astype(np.float32) - y.astype(np.float32)).mean()
        )
        out_delta, in_delta = delta(a, b), delta(*raw)
        residual_delta = delta(
            b.astype(np.float32) - raw[1], a.astype(np.float32) - raw[0]
        )
        cv2.imwrite(
            str(root / f"seam-{i:02d}.jpg"),
            np.vstack([np.hstack(raw), np.hstack([a, b])]),
        )
        record = dict(
            frame=n,
            time=n / fps,
            source_mae=in_delta,
            result_mae=out_delta,
            repair_delta_mae=residual_delta,
        )
        if references:
            hints = sorted(references, key=lambda h: abs(h["frame"] - n))[:2]
            x1, y1, x2, y2 = [
                int(v)
                for v in [
                    min(h["box"][0] for h in hints),
                    min(h["box"][1] for h in hints),
                    max(h["box"][2] for h in hints),
                    max(h["box"][3] for h in hints),
                ]
            ]
            pad = 12
            x1, y1, x2, y2 = (
                max(0, x1 - pad),
                max(0, y1 - pad),
                min(a.shape[1], x2 + pad),
                min(a.shape[0], y2 + pad),
            )
            roi = lambda image: image[y1:y2, x1:x2]
            record.update(
                roi=[x1, y1, x2, y2],
                roi_source_mae=delta(*[roi(x) for x in raw]),
                roi_result_mae=delta(roi(a), roi(b)),
                roi_repair_delta_mae=delta(
                    roi(b).astype(np.float32) - roi(raw[1]),
                    roi(a).astype(np.float32) - roi(raw[0]),
                ),
            )
            cv2.imwrite(
                str(root / f"seam-{i:02d}-crop.jpg"),
                np.vstack(
                    [np.hstack([roi(x) for x in raw]), np.hstack([roi(a), roi(b)])]
                ),
            )
        records.append(record)
    return records


def audio_timeline_metrics(source, output, playlist, source_pts, check=lambda: None):
    """比较逐包 PTS，容许 MPEG-TS 时间基舍入，不以音频摘要替代同步验证。"""

    def packets(path):
        data = json.loads(
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
                ],
                check,
            )
        )["packets"]
        return np.array(
            [[float(p["pts_time"]), float(p.get("duration_time", 0))] for p in data]
        )

    original = packets(source)
    max_error = 0.0
    for path, offset in ((output, 0), (playlist, source_pts)):
        actual = packets(path)
        if actual.shape != original.shape or len(actual) == 0:
            raise PipelineError("OUTPUT_INVALID", "音频包数量发生变化")
        actual[:, 0] -= offset
        error = float(np.abs(actual[:, 0] - original[:, 0]).max())
        max_error = max(max_error, error)
        if error > 0.001 or np.any(np.diff(actual[:, 0]) <= 0):
            raise PipelineError("OUTPUT_INVALID", "音频 PTS 发生变化")
    info = probe(source, check)
    video = next(x for x in info["streams"] if x["codec_type"] == "video")
    start_offset = original[0, 0] - float(video.get("start_time", 0))
    end_offset = (
        original[-1].sum()
        - float(video.get("start_time", 0))
        - float(video["duration"])
    )
    if abs(start_offset) > 0.08 or abs(end_offset) > 0.08:
        raise PipelineError("OUTPUT_INVALID", "源片音画起止偏差超过 80 毫秒")
    return dict(
        audio_timeline_max_error_ms=round(max_error * 1000, 6),
        av_start_offset_ms=round(start_offset * 1000, 6),
        av_end_offset_ms=round(end_offset * 1000, 6),
    )


def checkpoint_hash(digest, options, engine, low_memory, execution_fingerprint=""):
    settings = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(
            (
                "BOX_",
                "MASK_",
                "SHOT_",
                "DETECTION_",
                "YOLO_",
                "HLS_",
                "RESIDUAL_",
                "REFERENCE_",
                "RESTORE_",
                "SAM_",
                "PROPAINTER_",
            )
        )
    }
    return hashlib.sha256(
        json.dumps(
            dict(
                strategy="hls-coverage-and-short-tail-v6",
                source=digest,
                options=options,
                models=os.getenv("MODEL_LOCK_HASH", ""),
                pipeline=os.getenv("PIPELINE_VERSION", ""),
                engine=engine,
                low_memory=low_memory,
                execution_fingerprint=execution_fingerprint,
                settings=settings,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def execute_hls(
    source,
    root,
    options,
    engine,
    low_memory,
    stage,
    check,
    s3,
    bucket,
    checkpoint_prefix,
    progress,
    execution_fingerprint="",
):
    started = time.monotonic()
    full = root / "source.mp4"
    descriptor_key = checkpoint_prefix + "/source.json"
    try:
        body = s3.get_object(Bucket=bucket, Key=descriptor_key)["Body"]
        try:
            saved = json.loads(body.read(2 * 1024**2))
        finally:
            body.close()
        s3.download_file(
            bucket,
            checkpoint_prefix + "/source.mp4",
            str(full),
            Callback=lambda _: check(),
        )
        from .vision import sha256

        if sha256(full) != saved["mp4_sha256"]:
            raise PipelineError("OUTPUT_INVALID", "冻结的 HLS 输入摘要不一致")
        lines, segments, digest, source_pts = (
            saved["lines"],
            saved["segments"],
            saved["digest"],
            saved["source_pts"],
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        playlist, lines, segments, digest = snapshot(
            source,
            Reader(
                s3,
                bucket,
                check,
                (
                    checkpoint_prefix.split("/", 1)[0] + "/"
                    if "/jobs/" in checkpoint_prefix
                    else None
                ),
            ),
            root,
            options.get("hls_bandwidth", 0),
        )
        source_pts = validate_segments(playlist, segments, check)
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-copyts",
                "-protocol_whitelist",
                "file,crypto,data",
                "-i",
                playlist,
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c",
                "copy",
                "-output_ts_offset",
                str(-source_pts),
                "-avoid_negative_ts",
                "disabled",
                "-movflags",
                "+faststart",
                full,
            ],
            check,
        )
        # 清除原始 URI，检查点与日志均不保存签名地址。
        lines = [line if line.startswith("#") else "segment.ts" for line in lines]
        segments = [(d, "segment.ts") for d, _ in segments]
        from .vision import sha256

        saved = dict(
            lines=lines,
            segments=segments,
            digest=digest,
            source_pts=source_pts,
            mp4_sha256=sha256(full),
        )
        s3.upload_file(
            str(full),
            bucket,
            checkpoint_prefix + "/source.mp4",
            ExtraArgs={"ContentType": "video/mp4"},
            Callback=lambda _: check(),
        )
        s3.put_object(
            Bucket=bucket,
            Key=descriptor_key,
            Body=json.dumps(saved).encode(),
            ContentType="application/json",
        )
    info = probe(full, check)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    from fractions import Fraction

    source_fps = float(Fraction(video["avg_frame_rate"]))
    if (
        video["codec_name"] != "h264"
        or not 0 < source_fps <= 240
        or abs(float(Fraction(video["r_frame_rate"])) - source_fps) > 0.001
    ):
        raise invalid("HLS 仅支持帧率不超过 240 的恒定帧率 H.264 视频")
    fps = min(source_fps, 30)
    if any(
        x["codec_type"] == "audio" and x["codec_name"] != "aac" for x in info["streams"]
    ):
        raise invalid("HLS 仅支持内嵌 AAC 音轨，原样复制不转码")
    durations = [d for d, _ in segments]
    if abs(float(info["format"]["duration"]) - sum(durations)) > 0.1:
        raise invalid("播放列表时长与实际媒体时间轴不一致")
    plans = chunks(durations, fps)
    frames = root / "joined"
    frames.mkdir()
    times, previous_right = [], 0
    references = []
    previous_hints = []
    cache_hash = checkpoint_hash(
        digest, options, engine, low_memory, execution_fingerprint
    )
    for i, p in enumerate(plans):
        check()
        progress(i + 1, len(plans), "probe")
        chunk_start = time.monotonic()
        work = root / f"chunk-{i:04d}"
        work.mkdir()
        clip = work / "input.mp4"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(p["left"] / fps),
                "-i",
                full,
                "-an",
                "-vf",
                f"fps={fps}",
                "-frames:v",
                str(p["right"] - p["left"]),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                clip,
            ],
            check,
        )
        cache = f"{checkpoint_prefix}/{cache_hash}/{i:04d}.mkv"
        cached = work / "cached.mkv"
        reused = False
        refinements = []
        restored_draft = False
        try:
            s3.download_file(bucket, cache, str(cached), Callback=lambda _: check())
            reference_key = cache + ".json"
            s3.download_file(bucket, reference_key, str(work / "detections.json"))
            try:
                s3.download_file(
                    bucket, cache + ".timing.json", str(work / "checkpoint-timing.json")
                )
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in (
                    "404",
                    "NoSuchKey",
                    "NotFound",
                ):
                    raise
                (work / "checkpoint-timing.json").write_text("{}")
            reused = True
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
        if reused:
            composite = work / "composite"
            composite.mkdir()
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    cached,
                    "-start_number",
                    "0",
                    composite / "%06d.png",
                ],
                check,
            )
            meta = {"cached": True}
            phase = json.loads((work / "checkpoint-timing.json").read_text())
            try:
                s3.download_file(
                    bucket,
                    cache + ".refinements.json",
                    str(work / "hls-refinements.json"),
                )
                refinements = json.loads((work / "hls-refinements.json").read_text())
            except ClientError as exc:
                if exc.response["Error"]["Code"] not in (
                    "404",
                    "NoSuchKey",
                    "NotFound",
                ):
                    raise

        else:
            # 每帧检测与相同参数保证移动水印轨迹；重叠帧的结果在合并时共享。
            block_options = {
                **options,
                "hls_reference": [
                    dict(h, frame=h["frame"] - p["left"])
                    for h in previous_hints
                    if p["left"] <= h["frame"] < min(previous_right, p["right"])
                ],
            }
            from .hls_checkpoint import restore_draft, save_draft

            phase = restore_draft(
                s3, bucket, cache, work, clip, options, low_memory, check
            )
            restored_draft = phase is not None
            if restored_draft:
                from . import vision

                try:
                    vision.residual_quality(
                        work,
                        sorted((work / "frames").glob("*.png")),
                        json.loads((work / "detections.json").read_text()),
                        check,
                    )
                except PipelineError as exc:
                    if exc.code != "RESIDUAL_UNCERTAIN":
                        raise
            else:
                try:
                    result, meta, phase = execute(
                        clip,
                        work,
                        block_options,
                        engine,
                        low_memory,
                        lambda name: (
                            stage("encode" if name == "quality" else name),
                            progress(i + 1, len(plans), name),
                        ),
                        check,
                    )
                except PipelineError as exc:
                    if (
                        exc.code != "RESIDUAL_UNCERTAIN"
                        or not (work / "quality.json").exists()
                    ):
                        record_failed_chunk(root, work, i + 1, chunk_start, started)
                        raise
                    phase = json.loads((work / "phase_ms.json").read_text())
            if (work / "quality.json").exists():
                quality = json.loads((work / "quality.json").read_text())
                if quality["residual_count"] or quality["needs_review"]:
                    if not restored_draft:
                        save_draft(s3, bucket, cache, work, fps, check)
                    from .hls_repair import refine

                    detected = json.loads((work / "detections.json").read_text())
                    refinements = refine(
                        work,
                        sorted((work / "frames").glob("*.png")),
                        detected,
                        engine,
                        low_memory,
                        fps,
                        lambda name: (
                            stage("encode" if name == "quality" else name),
                            progress(i + 1, len(plans), name),
                        ),
                        check,
                    )
                    phase["residual_refine"] = sum(x["elapsed_ms"] for x in refinements)
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    work / "composite" / "%06d.png",
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",
                    cached,
                ],
                check,
            )
            s3.upload_file(
                str(cached),
                bucket,
                cache,
                ExtraArgs={"ContentType": "video/x-matroska"},
                Callback=lambda _: check(),
            )
            s3.upload_file(
                str(work / "detections.json"),
                bucket,
                cache + ".json",
                ExtraArgs={"ContentType": "application/json"},
            )
            s3.put_object(
                Bucket=bucket,
                Key=cache + ".timing.json",
                Body=json.dumps(phase).encode(),
                ContentType="application/json",
            )
            s3.put_object(
                Bucket=bucket,
                Key=cache + ".refinements.json",
                Body=json.dumps(refinements).encode(),
                ContentType="application/json",
            )
            composite = work / "composite"
        if (work / "detections.json").exists():
            previous_hints = [
                dict(h, frame=h["frame"] + p["left"])
                for h in json.loads((work / "detections.json").read_text())["hints"]
            ]
            references.extend(
                h for h in previous_hints if p["start"] <= h["frame"] < p["end"]
            )
        merge_frames(frames, composite, p, previous_right, fps, check)
        previous_right = p["right"]
        elapsed = round((time.monotonic() - chunk_start) * 1000)
        times.append(
            dict(
                index=i + 1,
                elapsed_ms=elapsed,
                reused=reused,
                residual_count=0,
                restored_draft=restored_draft,
                refinements=refinements,
                phase_ms=phase,
                **p,
            )
        )
        (root / "metrics-hls.json").write_text(
            json.dumps(
                dict(
                    chunks=times,
                    source_sha256=digest,
                    checkpoint_fingerprint=cache_hash,
                    completed_chunks=i + 1,
                    total_chunks=len(plans),
                ),
                ensure_ascii=False,
            )
        )
        shutil.rmtree(work)
    if os.getenv("DETECTOR") == "yolo":
        stage("detect")
        progress(len(plans), len(plans), "quality")
        from . import vision

        (root / "composite").symlink_to(frames, target_is_directory=True)
        (root / "frames").symlink_to(frames, target_is_directory=True)
        files = sorted(frames.glob("*.png"))
        detections = {
            "hls": True,
            "scenes": vision.shots(files, check),
            "hints": sorted(references, key=lambda x: x["frame"]),
        }
        (root / "detections.json").write_text(json.dumps(detections))
    stage("encode")
    progress(len(plans), len(plans), "encode")
    cuts = [sum(durations[:i]) for i in range(1, len(durations))]
    output = root / "result.mp4"
    keyframes = ",".join(str(x) for x in [0] + cuts)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            frames / "%08d.png",
            "-i",
            full,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-force_key_frames",
            keyframes,
            "-sc_threshold",
            "0",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            output,
        ],
        check,
    )
    if os.getenv("DETECTOR") == "yolo":
        stage("detect")
        progress(len(plans), len(plans), "quality")
        encoded_frames = root / "encoded-quality"
        encoded_frames.mkdir()
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                output,
                "-start_number",
                "0",
                encoded_frames / "%08d.png",
            ],
            check,
        )
        (root / "composite").unlink()
        (root / "composite").symlink_to(encoded_frames, target_is_directory=True)
        files = sorted(encoded_frames.glob("*.png"))
        vision.residual_quality(root, files, detections, check)
        quality = json.loads((root / "quality.json").read_text())
        if quality["residual_count"] or quality["needs_review"]:
            raise PipelineError(
                "RESIDUAL_UNCERTAIN", "HLS 合并结果存在残留证据", review=True
            )
    stage("encode")
    progress(len(plans), len(plans), "encode")
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
    output_info = probe(output, check, count=True)
    output_video = next(x for x in output_info["streams"] if x["codec_type"] == "video")
    if (
        int(output_video["nb_read_frames"]) != plans[-1]["end"]
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
    seams = seam_metrics(full, frames, plans, fps, root, check, references, output)
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
