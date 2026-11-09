"""Exercise upload -> immutable job -> worker -> downloadable MP4 using synthetic owned media."""

import argparse
import json
import os
import tempfile
import time
import urllib.request
from pathlib import Path
from scripts.fixtures import generate

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://localhost:8080")
p.add_argument("--token", default="local-development-token-change-me")
p.add_argument("--output", default="work/smoke-result.mp4")
args = p.parse_args()


def request(path, body=None, method=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        args.url + path,
        data=data,
        method=method,
        headers={
            "Authorization": "Bearer " + args.token,
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


with tempfile.TemporaryDirectory() as temp:
    source = generate(Path(temp))
    data = source.read_bytes()
    upload = request(
        "/v1/uploads",
        {
            "filename": "synthetic.mp4",
            "size_bytes": len(data),
            "content_type": "video/mp4",
        },
    )
    with urllib.request.urlopen(
        urllib.request.Request(
            upload["put_url"],
            data=data,
            method="PUT",
            headers=upload["required_headers"],
        ),
        timeout=30,
    ) as response:
        assert response.status == 200
    body = {
        "upload_id": upload["upload_id"],
        "options": {
            "max_short_edge": 720,
            "allow_downgrade": False,
            "prompts": [
                {"frame": 0, "box": [20, 30, 62, 52]},
                {"frame": 11, "box": [53, 30, 95, 52]},
            ],
        },
    }
    job = request(
        "/v1/jobs", body, headers={"Idempotency-Key": "smoke-" + os.urandom(8).hex()}
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = request("/v1/jobs/" + job["id"])
        if job["status"] in ("SUCCEEDED", "FAILED", "NEEDS_REVIEW", "CANCELED"):
            break
        time.sleep(0.5)
    assert job["status"] == "SUCCEEDED", job
    with urllib.request.urlopen(job["result"]["download_url"], timeout=30) as response:
        data = response.read()
        assert len(data) > 1000
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    print(
        json.dumps(
            {
                "id": job["id"],
                "status": job["status"],
                "result": {
                    k: v for k, v in job["result"].items() if k != "download_url"
                },
            },
            indent=2,
        )
    )
