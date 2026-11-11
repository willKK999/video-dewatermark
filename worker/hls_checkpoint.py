"""视频去水印系统：原始检测及 SAM 已完成时保存可复修的无损草稿。"""

import json
from pathlib import Path
import re
import zipfile
from botocore.exceptions import ClientError
from . import media
from .process import PipelineError, run
from .vision import sha256


def save_draft(s3, bucket, key, root, fps, check):
    movie = root / "draft.mkv"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            root / "composite" / "%06d.png",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            movie,
        ],
        check,
    )
    archive = root / "draft.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("draft.json", json.dumps(dict(sha256=sha256(movie))))
        for name in (
            "detections.json",
            "phase_ms.json",
            "quality.json",
            "quality-detections.json",
        ):
            out.write(root / name, name)
        for mask in sorted((root / "masks").glob("*.png")):
            check()
            out.write(mask, "masks/" + mask.name)
    s3.upload_file(str(movie), bucket, key + ".draft.mkv", Callback=lambda _: check())
    check()
    s3.upload_file(str(archive), bucket, key + ".draft.zip", Callback=lambda _: check())


def restore_draft(s3, bucket, key, root, clip, options, low_memory, check):
    archive = root / "draft.zip"
    movie = root / "draft.mkv"
    try:
        s3.download_file(
            bucket, key + ".draft.zip", str(archive), Callback=lambda _: check()
        )
        s3.download_file(
            bucket, key + ".draft.mkv", str(movie), Callback=lambda _: check()
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    with zipfile.ZipFile(archive) as source:
        if sum(x.file_size for x in source.infolist()) > 50 * 1024**2:
            raise PipelineError("OUTPUT_INVALID", "草稿掩膜归档超限")
        meta = json.loads(source.read("draft.json"))
        if sha256(movie) != meta["sha256"]:
            raise PipelineError("OUTPUT_INVALID", "草稿视频摘要不一致")
        for member in source.infolist():
            if member.filename == "draft.json":
                continue
            if member.filename not in (
                "detections.json",
                "phase_ms.json",
                "quality.json",
                "quality-detections.json",
            ) and not re.fullmatch(r"masks/[0-9]{6}\.png", member.filename):
                raise PipelineError("OUTPUT_INVALID", "草稿归档路径无效")
            target = root / member.filename
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(source.read(member))
    short = options.get("max_short_edge", 720)
    if low_memory and options.get("allow_downgrade"):
        short = min(short, 540)
    normalizer = (
        media.normalize_direct if options.get("speedup_v2") else media.normalize
    )
    normalizer(clip, root, short, check)
    (root / "composite").mkdir()
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            movie,
            "-start_number",
            "0",
            root / "composite" / "%06d.png",
        ],
        check,
    )
    phase = json.loads((root / "phase_ms.json").read_text())
    return phase
