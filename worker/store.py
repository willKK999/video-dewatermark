"""PostgreSQL is authoritative; Redis messages are only delivery hints."""

import hashlib
import json
import random
import secrets
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def connect(dsn):
    return psycopg.connect(dsn, row_factory=dict_row, connect_timeout=5)


def claim(dsn, jid, worker, version, config_hash):
    token = secrets.token_hex(24)
    with connect(dsn) as db:
        job = db.execute(
            """UPDATE jobs SET status='RUNNING',attempt=attempt+1,lease_token=%s,
            lease_until=now()+interval '60 seconds',updated_at=now(),error_code=NULL,error_message=NULL
            WHERE id=%s AND status='QUEUED' AND attempt<3 AND pipeline_version=%s RETURNING *""",
            (token, jid, version),
        ).fetchone()
        if job:
            job["low_memory"] = bool(
                db.execute(
                    "SELECT 1 FROM attempts WHERE job_id=%s AND failure_code='GPU_OOM' LIMIT 1",
                    (jid,),
                ).fetchone()
            )
            job["execution_config"] = {
                "base_config_hash": config_hash,
                "options": job["options"],
                "low_memory": job["low_memory"],
                "window": 36 if job["low_memory"] else 60,
                "overlap": 8 if job["low_memory"] else 12,
            }
            job["config_hash"] = hashlib.sha256(
                json.dumps(job["execution_config"], sort_keys=True).encode()
            ).hexdigest()
            db.execute(
                "INSERT INTO attempts(job_id,attempt_no,worker_id,config_hash) VALUES(%s,%s,%s,%s)",
                (jid, job["attempt"], worker, job["config_hash"]),
            )
        return job


def heartbeat(dsn, job):
    with connect(dsn) as db:
        row = db.execute(
            """UPDATE jobs SET lease_until=now()+interval '60 seconds'
            WHERE id=%s AND lease_token=%s AND lease_until>now() AND status IN ('RUNNING','CANCEL_REQUESTED') RETURNING status""",
            (job["id"], job["lease_token"]),
        ).fetchone()
        return row["status"] if row else None


def set_stage(dsn, job, stage):
    with connect(dsn) as db:
        return (
            db.execute(
                """UPDATE jobs SET stage=%s,updated_at=now() WHERE id=%s AND lease_token=%s
             AND lease_until>now() AND status='RUNNING'""",
                (stage, job["id"], job["lease_token"]),
            ).rowcount
            == 1
        )


def finish(
    dsn,
    job,
    *,
    error=None,
    result_key=None,
    result=None,
    timings=None,
    peak=None,
    metrics=None,
):
    with connect(dsn) as db:
        current = db.execute(
            "SELECT status FROM jobs WHERE id=%s AND lease_token=%s AND lease_until>now() FOR UPDATE",
            (job["id"], job["lease_token"]),
        ).fetchone()
        if not current or current["status"] not in ("RUNNING", "CANCEL_REQUESTED"):
            return False
        status = "SUCCEEDED"
        code = None
        message = None
        if current["status"] == "CANCEL_REQUESTED":
            status = "CANCELED"
            result_key = None
            result = None
        elif error:
            code = error.code
            message = str(error)
            can_retry = (
                error.retryable
                and job["attempt"] < 3
                and not (code == "GPU_OOM" and job["low_memory"])
            )
            status = (
                "QUEUED"
                if can_retry
                else ("NEEDS_REVIEW" if error.review else "FAILED")
            )
        db.execute(
            """UPDATE jobs SET status=%s,stage=NULL,lease_token=NULL,lease_until=NULL,
            error_code=%s,error_message=%s,result_key=%s,result_meta=%s,
            result_expires_at=CASE WHEN %s::text IS NOT NULL THEN now()+interval '7 days' ELSE NULL END,
            updated_at=now() WHERE id=%s""",
            (
                status,
                code,
                message,
                result_key,
                Jsonb(result) if result else None,
                result_key,
                job["id"],
            ),
        )
        db.execute(
            """UPDATE attempts SET ended_at=now(),failure_code=%s,phase_ms=%s,peak_vram_mib=%s,metrics=%s
            WHERE job_id=%s AND attempt_no=%s""",
            (
                code,
                Jsonb(timings or {}),
                peak,
                Jsonb(metrics or {}),
                job["id"],
                job["attempt"],
            ),
        )
        if status == "QUEUED":
            delay = (2 ** job["attempt"]) + random.uniform(0, 2)
            db.execute(
                "INSERT INTO outbox(job_id,dispatch_no,available_at) VALUES(%s,%s,now()+%s*interval '1 second') ON CONFLICT DO NOTHING",
                (job["id"], job["attempt"], delay),
            )
        return True


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
