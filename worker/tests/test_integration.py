"""Opt-in tests against real PostgreSQL/Redis and a local S3-compatible service."""

import concurrent.futures
import json
import os
import secrets
import urllib.error
import urllib.request
import pytest
from worker import store
from worker.process import PipelineError
from worker.main import object_store

pytestmark = pytest.mark.skipif(
    os.getenv("INTEGRATION") != "1", reason="set INTEGRATION=1 with running API/infra"
)
BASE = os.getenv("API_URL", "http://localhost:8080")
TOKEN = "local-development-token-change-me"


def call(path, body=None, method=None, token=TOKEN, headers=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        response = e
    with response:
        return response.status, json.load(response)


def uploaded():
    status, upload = call(
        "/v1/uploads",
        {"filename": "test.mp4", "size_bytes": 5, "content_type": "video/mp4"},
    )
    assert status == 201
    with urllib.request.urlopen(
        urllib.request.Request(
            upload["put_url"],
            data=b"input",
            method="PUT",
            headers=upload["required_headers"],
        )
    ) as r:
        assert r.status == 200
    return upload


def create():
    upload = uploaded()
    key = secrets.token_hex(16)
    body = {"upload_id": upload["upload_id"], "options": {"max_short_edge": 720}}
    status, job = call("/v1/jobs", body, headers={"Idempotency-Key": key})
    assert status == 202
    return upload, key, body, job


def test_concurrent_idempotency_input_freeze_tenant_and_cancel():
    upload = uploaded()
    key = secrets.token_hex(16)
    body = {"upload_id": upload["upload_id"], "options": {"max_short_edge": 720}}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(
            pool.map(
                lambda _: call("/v1/jobs", body, headers={"Idempotency-Key": key}),
                range(8),
            )
        )
    assert all(status == 202 for status, _ in responses)
    assert len({j["id"] for _, j in responses}) == 1
    jid = responses[0][1]["id"]
    status, _ = call(
        "/v1/jobs",
        dict(body, options={"max_short_edge": 540}),
        headers={"Idempotency-Key": key},
    )
    assert status == 409
    status, _ = call("/v1/jobs/" + jid, token="second-tenant-development-token")
    assert status == 404
    status, _ = call(
        "/v1/jobs",
        body,
        token="second-tenant-development-token",
        headers={"Idempotency-Key": key},
    )
    assert status == 404
    # Client may overwrite its presigned staging key, but the worker input is frozen.
    with urllib.request.urlopen(
        urllib.request.Request(
            upload["put_url"],
            data=b"other",
            method="PUT",
            headers=upload["required_headers"],
        )
    ) as r:
        assert r.status == 200
    with store.connect(os.environ["DATABASE_URL"]) as db:
        row = db.execute("SELECT input_key FROM jobs WHERE id=%s", (jid,)).fetchone()
    s3 = object_store()
    assert (
        s3.get_object(Bucket=os.environ["S3_BUCKET"], Key=row["input_key"])[
            "Body"
        ].read()
        == b"input"
    )
    status, job = call("/v1/jobs/" + jid + "/cancel", {}, method="POST")
    assert status == 202 and job["status"] == "CANCELED"
    status, job = call("/v1/jobs/" + jid + "/cancel", {}, method="POST")
    assert status == 202 and job["status"] == "CANCELED"


def test_stale_lease_cannot_publish_and_cancel_wins():
    _, _, _, job = create()
    dsn = os.environ["DATABASE_URL"]
    claimed = store.claim(dsn, job["id"], "test-worker", "mvp-1-dev", "test-config")
    assert claimed and not store.claim(
        dsn, job["id"], "duplicate", "mvp-1-dev", "test-config"
    )
    stale = dict(claimed, lease_token="obsolete")
    assert not store.finish(dsn, stale, result_key="illegal", result={"width": 1})
    assert not store.set_stage(dsn, stale, "publish")
    status, canceled = call("/v1/jobs/" + job["id"] + "/cancel", {}, method="POST")
    assert status == 202 and canceled["status"] == "CANCEL_REQUESTED"
    assert store.finish(
        dsn, claimed, result_key="must-not-be-published", result={"width": 1}
    )
    status, current = call("/v1/jobs/" + job["id"])
    assert current["status"] == "CANCELED" and current["result"] is None
    assert not store.finish(dsn, claimed, result_key="late", result={"width": 1})


def test_oom_retries_once_and_attempt_cap():
    _, _, _, job = create()
    dsn = os.environ["DATABASE_URL"]
    a = store.claim(dsn, job["id"], "test", "mvp-1-dev", "config")
    assert store.finish(
        dsn, a, error=PipelineError("GPU_OOM", "out of memory", retryable=True)
    )
    b = store.claim(dsn, job["id"], "test", "mvp-1-dev", "config")
    assert b["attempt"] == 2 and b["low_memory"]
    assert store.finish(
        dsn, b, error=PipelineError("GPU_OOM", "out of memory", retryable=True)
    )
    _, j = call("/v1/jobs/" + job["id"])
    assert j["status"] == "FAILED" and j["attempt"] == 2
    assert not store.claim(dsn, job["id"], "test", "mvp-1-dev", "config")


def test_expired_lease_reclaimed_without_stale_publish():
    import time

    _, _, _, job = create()
    dsn = os.environ["DATABASE_URL"]
    a = store.claim(dsn, job["id"], "test", "mvp-1-dev", "config")
    with store.connect(dsn) as db:
        db.execute(
            "UPDATE jobs SET lease_until=now()-interval '1 second' WHERE id=%s",
            (job["id"],),
        )
    assert not store.finish(dsn, a, result_key="late", result={"width": 1})
    until = time.monotonic() + 5
    while time.monotonic() < until:
        _, current = call("/v1/jobs/" + job["id"])
        if current["status"] == "QUEUED":
            break
        time.sleep(0.1)
    assert current["status"] == "QUEUED"
    b = store.claim(dsn, job["id"], "new-worker", "mvp-1-dev", "config")
    assert b["attempt"] == 2 and b["lease_token"] != a["lease_token"]
    assert not store.finish(dsn, a, error=PipelineError("INTERNAL", "late"))
    assert store.finish(dsn, b, error=PipelineError("TEST_DONE", "test done"))


def test_upload_validation_and_error_contract():
    status, error = call(
        "/v1/uploads",
        {"filename": "a.mp4", "size_bytes": 209715201, "content_type": "video/mp4"},
    )
    assert (
        status == 413
        and error["error"]["code"] == "UPLOAD_TOO_LARGE"
        and error["request_id"]
    )
    status, _ = call(
        "/v1/uploads",
        {"filename": "a.exe", "size_bytes": 1, "content_type": "video/mp4"},
    )
    assert status == 400
    status, _ = call("/v1/jobs/not-found", token="invalid")
    assert status == 401


def test_success_cancel_race_has_one_terminal_winner():
    import threading

    _, _, _, job = create()
    dsn = os.environ["DATABASE_URL"]
    claimed = store.claim(dsn, job["id"], "racer", "mvp-1-dev", "config")
    gate = threading.Barrier(2)

    def cancel():
        gate.wait()
        return call("/v1/jobs/" + job["id"] + "/cancel", {}, method="POST")

    def succeed():
        gate.wait()
        return store.finish(
            dsn,
            claimed,
            result_key="missing-test-result",
            result={"width": 320, "height": 180},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(cancel)
        b = pool.submit(succeed)
        status, _ = a.result()
        assert b.result()
    _, final = call("/v1/jobs/" + job["id"])
    assert (status, final["status"]) in [(202, "CANCELED"), (409, "SUCCEEDED")]
    assert not store.finish(dsn, claimed, result_key="duplicate", result={"width": 1})
    if final["status"] == "SUCCEEDED":
        assert final["result"]["expired"]


def test_pending_quota_preserves_idempotent_replay():
    jobs = []
    try:
        # Isolated test tenant, API runs without a consuming worker in this suite.
        for _ in range(10):
            jobs.append(create())
        upload, key, body, job = jobs[0]
        status, replayed = call("/v1/jobs", body, headers={"Idempotency-Key": key})
        assert status == 202 and replayed["id"] == job["id"]
        status, error = call(
            "/v1/jobs", body, headers={"Idempotency-Key": secrets.token_hex(16)}
        )
        assert status == 429 and error["error"]["code"] == "QUOTA_EXCEEDED"
    finally:
        for _, _, _, job in jobs:
            call("/v1/jobs/" + job["id"] + "/cancel", {}, method="POST")
