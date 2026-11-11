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


def export_native_artifacts(s3, bucket, prefix, output):
    """导出原生报告与对比，不回退到旧的模型帧率检查点解释。"""
    s3.download_file(bucket, prefix + "/report.json", str(output / "report.json"))
    report = json.loads((output / "report.json").read_text())
    pairs = report.get("对比产物", {})
    names = ["progress.log", "merged-from-hls.mp4"]
    comparisons = [pairs.get("并排视频"), *pairs.get("图片", [])]
    if any(
        not isinstance(name, str)
        or not re.fullmatch(r"comparisons/[^/\\]+\.(?:mp4|png)", name)
        or ".." in Path(name).parts
        for name in comparisons
    ):
        raise PipelineError("OUTPUT_INVALID", "原生对比产物路径无效")
    names.extend(comparisons)
    evidence = []
    for chunk in report.get("原始残留检查", []):
        directory = chunk.get("原始证据目录")
        if not isinstance(directory, str) or not re.fullmatch(
            r"native-chunk-[0-9]{4}/audit/", directory
        ):
            raise PipelineError("OUTPUT_INVALID", "原生核验证据路径无效")
        evidence.extend(
            directory + name
            for name in (
                "quality.json",
                "quality-detections.json",
                "native-quality.json",
                "raw-residual-candidates.json",
                "raw-residual-review.json",
            )
        )
    saved, missing = [], []
    for name in [*names, *evidence]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(bucket, prefix + "/" + name, str(target))
            saved.append(name)
        except ClientError as error:
            if name not in evidence or error.response["Error"]["Code"] not in (
                "404",
                "NoSuchKey",
                "NotFound",
            ):
                raise
            missing.append(name)
    result = dict(
        已导出=saved,
        缺失核验证据=missing,
        说明="原始报告和候选不改写；缺失证据不得宣称已完成核验",
    )
    (output / "export-native.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    return result


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
    if meta.get("native_path"):
        export_native_artifacts(s3, bucket, prefix, output)
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
    if references.get("multitrack"):
        prior = output / "multitrack-prior.npz"
        download(prefix + "/multitrack-prior.npz", prior)
        references["prior"] = str(prior.resolve())
        ref_file.write_text(json.dumps(references))
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
