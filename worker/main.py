import fcntl
import json
import logging
import os
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
import boto3
import redis
from botocore.config import Config
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import BotoCoreError, ClientError
from . import store
from .delivery import ack_after_miss, reclaim_batch
from .pipeline import execute
from .publication import upload_artifacts
from .process import LeaseLost, PipelineError, run
from .provenance import assert_catalog_locked, validate_provenance
from .vision import verify_models, sha256

log = logging.getLogger("worker")
stop = threading.Event()


def verify_result_head(head, size, digest):
    meta = {
        str(key).lower(): value for key, value in (head.get("Metadata") or {}).items()
    }
    if head.get("ContentLength") != size or meta.get("sha256") != digest:
        raise PipelineError("OUTPUT_INVALID", "uploaded result verification failed")


class Guard:
    def __init__(self, dsn, job):
        self.dsn, self.job = dsn, job
        self.state = "RUNNING"
        self.done = threading.Event()
        self.peak = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join(timeout=10)

    def loop(self):
        while not self.done.is_set():
            try:
                self.state = store.heartbeat(self.dsn, self.job)
                if self.state is None:
                    return
            except Exception:
                self.state = None
                return
            try:
                usage = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if usage.returncode == 0:
                    self.peak = max(
                        self.peak or 0, max(int(x) for x in usage.stdout.splitlines())
                    )
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.done.wait(1)

    def check(self):
        if stop.is_set() or self.state is None:
            raise LeaseLost("lease or worker lost")
        if self.state == "CANCEL_REQUESTED":
            raise PipelineError("CANCELED", "cancellation requested")

    def stage(self, name):
        self.check()
        if not store.set_stage(self.dsn, self.job, name):
            # Resolve the cancel-versus-stage race before treating it as lost.
            self.state = store.heartbeat(self.dsn, self.job)
            self.check()
            raise LeaseLost()


def object_store():
    return boto3.client(
        "s3",
        endpoint_url=("https://" if os.getenv("S3_SECURE") == "true" else "http://")
        + os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name=os.getenv("S3_REGION", "us-east-1"),
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=5,
            read_timeout=20,
            retries={"max_attempts": 2},
        ),
    )


def upload_options(kind, **options):
    """兼容不实现对象标签的存储；是否使用标签由部署能力明确配置。"""
    if os.environ.get("S3_OBJECT_TAGGING", "true").lower() == "true":
        options["Tagging"] = "kind=" + kind
    else:
        options["Metadata"] = {**options.get("Metadata", {}), "kind": kind}
    return options


def process(job, dsn, s3, bucket, workdir, engine):
    with (
        tempfile.TemporaryDirectory(prefix=job["id"] + "-", dir=workdir) as temp,
        Guard(dsn, job) as guard,
    ):
        root = Path(temp).resolve()
        timings = {}
        result_key = None
        result = None
        error = None
        try:
            guard.stage("probe")
            source = root / "source"
            if job["options"].get("hls"):
                from .hls import execute_hls

                def progress(current, total, phase, **details):
                    store.set_progress(
                        dsn,
                        job,
                        dict(
                            current_chunk=current,
                            total_chunks=total,
                            phase=phase,
                            **details,
                        ),
                    )

                result_path, result, timings = execute_hls(
                    job["input_key"],
                    root,
                    job["options"],
                    engine,
                    job["low_memory"],
                    guard.stage,
                    guard.check,
                    s3,
                    bucket,
                    f"{job['tenant_id']}/jobs/{job['id']}/checkpoints",
                    progress,
                    execution_fingerprint=job["execution_config"]["base_config_hash"],
                )
            else:
                # Bounded stream and lease checks, including during downloads.
                body = s3.get_object(Bucket=bucket, Key=job["input_key"])["Body"]
                try:
                    with source.open("wb") as out:
                        total = 0
                        while chunk := body.read(1024 * 1024):
                            guard.check()
                            total += len(chunk)
                            if total > 200 * 1024 * 1024:
                                raise PipelineError(
                                    "INVALID_MEDIA", "input exceeds 200 MiB"
                                )
                            out.write(chunk)
                finally:
                    body.close()
                result_path, result, timings = execute(
                    source,
                    root,
                    job["options"],
                    engine,
                    job["low_memory"],
                    guard.stage,
                    guard.check,
                )
            guard.stage("publish")
            started = time.monotonic()
            prefix = f"{job['tenant_id']}/jobs/{job['id']}/attempts/{job['attempt']}-{job['lease_token']}"
            result_key = prefix + "/result.mp4"
            if result.get("hls"):
                progress(result["chunk_count"], result["chunk_count"], "publish")
                report_path = root / "report.json"
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                    report["安全对象前缀"] = prefix + "/"
                    report_path.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2)
                    )
                for path in [
                    *sorted((root / "hls").glob("*.ts")),
                    root / "hls" / "index.m3u8",
                    root / "hls.zip",
                    root / "source.mp4",
                    root / "metrics-hls.json",
                    *(
                        [root / "detections.json"]
                        if (root / "detections.json").exists()
                        else []
                    ),
                    *sorted(root.glob("seam-*.jpg")),
                    *sorted(root.glob("multitrack-*.npz")),
                    *sorted(root.glob("multitrack-*.json")),
                    *[
                        p
                        for p in (
                            root / "report.json",
                            root / "progress.log",
                            root / "merged-from-hls.mp4",
                        )
                        if p.exists()
                    ],
                    *sorted((root / "comparisons").glob("*.mp4")),
                    *sorted((root / "comparisons").glob("*.png")),
                    *sorted(root.glob("native-chunk-*/audit/*.json")),
                ]:
                    relative = path.relative_to(root).as_posix()
                    content_type = {
                        "ts": "video/mp2t",
                        "m3u8": "application/vnd.apple.mpegurl",
                        "zip": "application/zip",
                        "mp4": "video/mp4",
                        "json": "application/json",
                        "jpg": "image/jpeg",
                        "npz": "application/octet-stream",
                        "png": "image/png",
                        "log": "text/plain; charset=utf-8",
                    }[path.suffix[1:]]
                    s3.upload_file(
                        str(path),
                        bucket,
                        prefix + "/" + relative,
                        ExtraArgs=upload_options("result", ContentType=content_type),
                        Callback=lambda _: guard.check(),
                    )
                result["hls_prefix"] = prefix + "/hls/"
                result["bundle_key"] = prefix + "/hls.zip"
                result["source_key"] = prefix + "/source.mp4"
            digest = sha256(result_path)
            s3.upload_file(
                str(result_path),
                bucket,
                result_key,
                ExtraArgs=upload_options(
                    "result",
                    ContentType="video/mp4",
                    ChecksumAlgorithm="SHA256",
                    Metadata={"sha256": digest},
                ),
                Callback=lambda _: guard.check(),
            )
            verify_result_head(
                s3.head_object(Bucket=bucket, Key=result_key),
                result_path.stat().st_size,
                digest,
            )
            result_ms = round((time.monotonic() - started) * 1000)
            artifacts_ms = (
                0
                if result.get("hls")
                else upload_artifacts(
                    s3, bucket, prefix, root, upload_options, guard.check
                )
            )
            (root / "publish_ms.json").write_text(
                json.dumps(dict(result=result_ms, artifacts=artifacts_ms))
            )
            timings["publish"] = round((time.monotonic() - started) * 1000)
            result.update(
                engine=engine,
                pipeline_version=job["pipeline_version"],
            )
            guard.check()
        except LeaseLost:
            log.info(
                "attempt abandoned",
                extra={"job_id": job["id"], "attempt": job["attempt"]},
            )
            return
        except PipelineError as exc:
            error = exc
            result_key = None
            result = None
        except (BotoCoreError, ClientError, S3UploadFailedError, OSError):
            error = PipelineError(
                "STORAGE_UNAVAILABLE", "storage or local IO failed", retryable=True
            )
            result_key = None
            result = None
        except Exception as exc:
            # Deliberately omit exception text: SDK errors may contain presigned URLs.
            log.error("unexpected pipeline failure: %s", type(exc).__name__)
            error = PipelineError("INTERNAL_ERROR", "unexpected pipeline failure")
            result_key = None
            result = None
        if (root / "phase_ms.json").exists():
            saved = json.loads((root / "phase_ms.json").read_text())
            saved.update(timings)
            timings = saved
        metrics = {
            f.stem: json.loads(f.read_text()) for f in root.glob("metrics*.json")
        }
        for filename in ("performance.json", "publish_ms.json"):
            path = root / filename
            if path.exists():
                metrics[path.stem] = json.loads(path.read_text())
        metrics["execution_config"] = {
            **job["execution_config"],
            "config_hash": job["config_hash"],
            "image_digest": os.getenv("IMAGE_DIGEST", "unrecorded"),
            "model_lock_hash": os.getenv("MODEL_LOCK_HASH", "dev-no-models"),
            "pipeline_version": job["pipeline_version"],
        }
        accepted = store.finish(
            dsn,
            job,
            error=error,
            result_key=result_key,
            result=result,
            timings=timings,
            peak=guard.peak,
            metrics=metrics,
        )
        log.info(
            json.dumps(
                {
                    "event": "attempt_finished",
                    "job_id": job["id"],
                    "attempt": job["attempt"],
                    "code": error.code if error else "OK",
                    "accepted": accepted,
                    "phase_ms": timings,
                    "peak_vram_mib": guard.peak,
                }
            )
        )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    workdir = Path(os.getenv("WORK_DIR", "work")).resolve()
    workdir.mkdir(exist_ok=True, parents=True)
    # Shared mount across worker processes on this GPU host. One active task per card.
    lock = (workdir / "gpu.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    engine = os.getenv("INPAINT_ENGINE", "propainter")
    version = os.getenv("PIPELINE_VERSION", "mvp-1")
    if engine not in ("propainter", "telea-dev"):
        raise ValueError("unknown inpaint engine")
    if engine == "telea-dev" and not version.endswith("-dev"):
        raise ValueError("CPU smoke engine requires a distinct -dev pipeline version")
    if engine == "propainter" and version.endswith("-dev"):
        raise ValueError("production engine cannot use a -dev pipeline version")
    run(["ffprobe", "-version"])
    encoders = set(run(["ffmpeg", "-encoders"]).decode().split())
    if "libx264" not in encoders or "aac" not in encoders:
        raise RuntimeError("FFmpeg H.264 or AAC encoder missing")
    if engine == "propainter":
        # Official catalog first. An operator lock cannot paper over UNRESOLVED digests.
        catalog = validate_provenance(
            os.getenv("MODEL_CATALOG", "deploy/models.json"),
            os.getenv("WEIGHTS_DIR"),
        )
        model_hash = verify_models(os.environ["MODEL_MANIFEST"])
        assert_catalog_locked(
            catalog, json.loads(Path(os.environ["MODEL_MANIFEST"]).read_text())
        )
    else:
        model_hash = "dev-no-models"
    os.environ["MODEL_LOCK_HASH"] = model_hash
    log.info(
        json.dumps(
            {
                "event": "worker_started",
                "pipeline_version": version,
                "engine": engine,
                "image_digest": os.getenv("IMAGE_DIGEST", "unrecorded"),
                "model_lock_hash": model_hash,
            }
        )
    )
    if engine == "propainter":
        for key in ("TRACK_PYTHON", "INPAINT_PYTHON"):
            run(
                [
                    os.environ[key],
                    "-c",
                    'import torch; assert torch.cuda.is_available(); x=torch.ones(8,device="cuda"); assert x.sum().item()==8',
                ]
            )
    dsn = os.environ["DATABASE_URL"]
    s3 = object_store()
    bucket = os.getenv("S3_BUCKET", "watermark")
    s3.head_bucket(Bucket=bucket)
    probe_key = "health/" + os.urandom(16).hex()
    s3.put_object(Bucket=bucket, Key=probe_key, Body=b"health")
    assert s3.get_object(Bucket=bucket, Key=probe_key)["Body"].read() == b"health"
    s3.delete_object(Bucket=bucket, Key=probe_key)
    queue = redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    queue.ping()
    with store.connect(dsn) as db:
        db.execute("SELECT 1 FROM jobs LIMIT 1")
    group = "workers"
    consumer = socket.gethostname() + "-" + str(os.getpid())
    try:
        queue.xgroup_create("jobs", group, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
    image_digest = os.getenv("IMAGE_DIGEST", "unrecorded")
    config = store.config_hash(
        {
            "pipeline": version,
            "engine": engine,
            "models": model_hash,
            "detector": os.getenv("DETECTOR", "template-or-prompt"),
            "template_sha256": (
                sha256(os.environ["TEMPLATE_PATH"])
                if os.getenv("TEMPLATE_PATH")
                else None
            ),
            "image": image_digest,
            "fp16": os.getenv("PROPAINTER_FP16", "false"),
            "bf16": os.getenv("SAM_BF16", "false"),
        }
    )
    reclaim_cursor = "0-0"
    while not stop.is_set():
        try:
            reclaimed = queue.xautoclaim(
                "jobs",
                group,
                consumer,
                min_idle_time=60000,
                start_id=reclaim_cursor,
                count=1,
            )
            reclaim_cursor, messages = reclaim_batch(reclaim_cursor, reclaimed)
            if not messages:
                streams = queue.xreadgroup(
                    group, consumer, {"jobs": ">"}, count=1, block=1000
                )
                messages = streams[0][1] if streams else []
            for mid, fields in messages:
                jid = fields.get("job_id", "")
                job = store.claim(
                    dsn,
                    jid,
                    consumer,
                    version,
                    config,
                    image_digest,
                    {"pipeline_version": version, "engine": engine},
                )
                if job:
                    queue.xack("jobs", group, mid)
                    process(job, dsn, s3, bucket, workdir, engine)
                    continue
                # Another pipeline version may be waiting for its own worker.
                row = store.job_delivery_row(dsn, jid)
                if ack_after_miss(row, version):
                    queue.xack("jobs", group, mid)
        except Exception as exc:
            log.error("worker loop failed: %s", type(exc).__name__)
            stop.wait(2)


if __name__ == "__main__":
    main()
