"""使用已部署服务验收操作台；仅在显式启用时运行，不输出凭证或签名地址。"""
import json
import os
import secrets
import urllib.request
import urllib.error

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("CONSOLE_INTEGRATION") != "1", reason="需要显式启用操作台部署集成测试"
)


def request(path, token, method="GET", data=None, headers=None):
    req = urllib.request.Request(
        os.getenv("API_URL", "http://127.0.0.1:18080") + path,
        data=data, method=method,
        headers={"Authorization": "Bearer " + token, **(headers or {})},
    )
    try:
        response = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return response.status, response.read(), response.headers


@pytest.fixture
def console():
    import psycopg
    from psycopg.rows import dict_row
    tokens = json.loads(os.environ["API_TOKENS"])
    token = next(k for k, v in tokens.items() if v == "kelin")
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as db:
        yield token, db


def test_console_tenant_boundary_and_list(console):
    token, db = console
    status, raw, headers = request("/v1/jobs", token)
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    items = json.loads(raw)["items"]
    ids = [e["job"]["id"] for e in items]
    expected = db.execute("SELECT id FROM jobs WHERE tenant_id='kelin' ORDER BY created_at DESC,id DESC LIMIT 100").fetchall()
    assert ids == [row["id"] for row in expected]
    other = db.execute("SELECT id,upload_id FROM jobs WHERE tenant_id!='kelin' LIMIT 1").fetchone()
    assert other is not None
    assert other["id"] not in ids
    for path in ["/v1/jobs/" + other["id"], "/v1/jobs/" + other["id"] + "/media/source", "/v1/jobs/" + other["id"] + "/media/result"]:
        assert request(path, token)[0] == 404
    assert request("/v1/uploads/" + other["upload_id"] + "/content", token, "PUT", b"x")[0] == 404
    assert request("/v1/jobs", "invalid-token")[0] == 401


def test_console_invalid_media_and_size(console):
    token, _ = console
    body = json.dumps({"filename": "invalid.mp4", "size_bytes": 5, "content_type": "video/mp4"}).encode()
    status, raw, _ = request("/v1/uploads", token, "POST", body, {"Content-Type": "application/json"})
    assert status == 201
    reservation = json.loads(raw)
    uid = reservation["upload_id"]
    if os.getenv("S3_OBJECT_TAGGING", "true").lower() == "false":
        assert not any(k.lower() == "x-amz-tagging" for k in reservation["required_headers"])
        assert any(k.lower() == "x-amz-meta-kind" for k in reservation["required_headers"])
    status, raw, _ = request("/v1/uploads/" + uid + "/content", token, "PUT", b"wrong", {"Content-Type": "video/mp4"})
    assert status == 422 and json.loads(raw)["error"]["code"] == "UNSUPPORTED_VIDEO"
    assert request("/v1/uploads/" + uid + "/content", token, "PUT", b"size", {"Content-Type": "video/mp4"})[0] == 422
    body = json.dumps({"filename": "large.mp4", "size_bytes": 209715201, "content_type": "video/mp4"}).encode()
    assert request("/v1/uploads", token, "POST", body, {"Content-Type": "application/json"})[0] == 413


def test_console_cancel_and_media_range(console):
    token, db = console
    job = db.execute("SELECT * FROM jobs WHERE tenant_id='kelin' AND status='SUCCEEDED' ORDER BY created_at DESC LIMIT 1").fetchone()
    assert job is not None
    for kind in ["source", "result"]:
        status, body, headers = request("/v1/jobs/" + job["id"] + "/media/" + kind, token, headers={"Range": "bytes=0-31"})
        assert status == 206 and len(body) == 32 and headers["Content-Range"].startswith("bytes 0-31/")
    jid = "job_console_test_" + secrets.token_hex(8)
    fixture_ids = [jid]
    try:
        db.execute("INSERT INTO jobs(id,tenant_id,upload_id,input_key,idempotency_key,request_hash,status,options,pipeline_version) VALUES(%s,'kelin',%s,%s,%s,'test','QUEUED','{}',%s)", (jid, job["upload_id"], job["input_key"], jid, job["pipeline_version"]))
        db.commit()
        for _ in range(2):
            status, raw, _ = request("/v1/jobs/" + jid + "/cancel", token, "POST", b"")
            assert status == 202 and json.loads(raw)["status"] == "CANCELED"
        # 终态不可修改，使用独立的测试记录分别检查缺失和已过期的保留时间。
        for expiry in [None, "2000-01-01T00:00:00Z"]:
            expired_id = "job_console_test_" + secrets.token_hex(8)
            fixture_ids.append(expired_id)
            db.execute("INSERT INTO jobs(id,tenant_id,upload_id,input_key,idempotency_key,request_hash,status,options,pipeline_version,result_key,result_meta,result_expires_at) VALUES(%s,'kelin',%s,%s,%s,'test','SUCCEEDED','{}',%s,%s,'{}',%s)", (expired_id, job["upload_id"], job["input_key"], expired_id, job["pipeline_version"], job["result_key"], expiry))
            db.commit()
            assert request("/v1/jobs/" + expired_id + "/media/result", token)[0] == 404
    finally:
        db.rollback()
        db.execute("DELETE FROM jobs WHERE id=ANY(%s)", (fixture_ids,))
        db.commit()
