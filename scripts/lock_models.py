"""Generate a reviewable local lock after downloading and reviewing model sources."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--repo", action="append", default=[], required=True)
p.add_argument("--file", action="append", default=[], required=True)
p.add_argument(
    "--source",
    required=True,
    help="Source/release/HF revision provenance recorded in the lock",
)
p.add_argument("--output", default="models/model-lock.json")
a = p.parse_args()
files = []
for name in a.file:
    path = Path(name).resolve()
    digest = (
        hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        if hasattr(hashlib, "file_digest")
        else hashlib.sha256(path.read_bytes()).hexdigest()
    )
    files.append({"path": str(path), "sha256": digest, "source": a.source})
repos = []
for name in a.repo:
    path = Path(name).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip():
        raise ValueError("review and commit model source patches before locking")
    repos.append({"path": str(path), "commit": commit})
Path(a.output).write_text(
    json.dumps({"files": files, "repositories": repos}, indent=2) + "\n"
)
print(a.output)
