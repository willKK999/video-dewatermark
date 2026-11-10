"""视频去水印系统：按任务导出可离线复验的 HLS、MP4 和安全审计记录。"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys

from botocore.exceptions import ClientError
from worker import store
from worker.main import object_store, verify_result_head
from worker.process import PipelineError
from worker.vision import sha256
from scripts.hls_verify import verify


def export(job_id, output, detect):
    with store.connect(os.environ["DATABASE_URL"]) as db:
        job = db.execute(
            "SELECT id,tenant_id,status,attempt,result_key,result_meta,created_at,updated_at FROM jobs WHERE id=%s",
            (job_id,),
        ).fetchone()
        attempts = db.execute(
            "SELECT attempt_no,started_at,ended_at,failure_code,phase_ms,peak_vram_mib FROM attempts WHERE job_id=%s ORDER BY attempt_no",
            (job_id,),
        ).fetchall()
    if not job or job["status"] != "SUCCEEDED" or not job["result_meta"].get("hls"):
        raise PipelineError("OUTPUT_INVALID", "仅能导出已成功的 HLS 任务")
    output.mkdir(parents=True, exist_ok=True)
    hls = output / "hls"
    hls.mkdir(exist_ok=True)
    s3 = object_store()
    bucket = os.environ["S3_BUCKET"]
    meta = job["result_meta"]
    prefix = job["result_key"].rsplit("/", 1)[0]

    def download(key, target):
        s3.download_file(bucket, key, str(target))

    download(job["result_key"], output / "result.mp4")
    verify_result_head(
        s3.head_object(Bucket=bucket, Key=job["result_key"]),
        (output / "result.mp4").stat().st_size,
        sha256(output / "result.mp4"),
    )
    download(meta["source_key"], output / "source.mp4")
    download(prefix + "/metrics-hls.json", output / "metrics-hls.json")
    download(meta["hls_prefix"] + "index.m3u8", hls / "index.m3u8")
    names = [
        line.strip()
        for line in (hls / "index.m3u8").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not all(re.fullmatch(r"[0-9]{6}\.ts", name) for name in names):
        raise PipelineError("OUTPUT_INVALID", "成品播放列表路径不符合输出规范")
    for name in names:
        download(meta["hls_prefix"] + name, hls / name)
    report = json.loads((output / "metrics-hls.json").read_text())
    ref_file = output / "references.json"
    try:
        download(prefix + "/detections.json", ref_file)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
            raise
        checkpoint_prefix = f"{job['tenant_id']}/jobs/{job_id}/checkpoints/"
        candidates = [
            obj["Key"]
            for page in s3.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=checkpoint_prefix
            )
            for obj in page.get("Contents", [])
        ]
        hints = []
        for i, plan in enumerate(report["chunks"]):
            matches = [
                key
                for key in candidates
                if key.endswith(f"/{i:04d}.mkv.json")
                and (
                    not report.get("checkpoint_fingerprint")
                    or "/" + report["checkpoint_fingerprint"] + "/" in key
                )
            ]
            if len(matches) != 1:
                raise PipelineError(
                    "OUTPUT_INVALID", "检查点版本存在歧义，无法重建自动检测轨迹"
                )
            temporary = output / "checkpoint-reference.json"
            download(matches[0], temporary)
            detected = json.loads(temporary.read_text())
            hints.extend(
                dict(h, frame=h["frame"] + plan["left"])
                for h in detected["hints"]
                if plan["start"] <= h["frame"] + plan["left"] < plan["end"]
            )
        ref_file.write_text(
            json.dumps(
                dict(
                    hls=True,
                    scenes=[[0, report["chunks"][-1]["end"]]],
                    hints=sorted(hints, key=lambda h: h["frame"]),
                )
            )
        )
    references = json.loads(ref_file.read_text())
    safe = dict(
        id=job_id,
        status=job["status"],
        attempt=job["attempt"],
        created_at=job["created_at"].isoformat(),
        updated_at=job["updated_at"].isoformat(),
        wall_seconds=(job["updated_at"] - job["created_at"]).total_seconds(),
        attempts=attempts,
    )
    (output / "timing.json").write_text(
        json.dumps(
            safe,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.isoformat() if isinstance(value, datetime) else str(value)
            ),
        )
    )
    verify(
        output / "source.mp4",
        output / "result.mp4",
        hls / "index.m3u8",
        report,
        references,
        output / "audit",
        detect,
    )
    print("HLS 成品和审计证据导出完成，分片数：" + str(len(names)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detect", action="store_true")
    args = parser.parse_args()
    export(args.job, args.output, args.detect)


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        print("验收导出失败：" + exc.code + "，" + str(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print("验收导出失败：" + type(exc).__name__, file=sys.stderr)
        sys.exit(1)
