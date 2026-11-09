"""Run a frozen, authorized dataset and report observed latency, VRAM and mask quality."""

import argparse
import fcntl
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
import cv2
import numpy as np
from worker.pipeline import execute
from worker.process import PipelineError
from worker.vision import sha256, verify_models


def mask_metrics(pred, truth):
    pred, truth = pred > 0, truth > 0
    tp = np.count_nonzero(pred & truth)
    return {
        "recall": tp / max(1, np.count_nonzero(truth)),
        "precision": tp / max(1, np.count_nonzero(pred)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default="benchmark.json")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--freeze-acceptance", action="store_true")
    a = p.parse_args()
    data = json.loads(Path(a.dataset).read_text())
    samples = data["samples"]
    if a.freeze_acceptance:
        assert len(samples) == 30 and sum(s["supported"] for s in samples) == 20
        assert sum(bool(s.get("truth_masks")) for s in samples) >= 10
    for s in samples:
        assert sha256(s["path"]) == s["sha256"], "dataset hash differs"
    work = Path(os.getenv("WORK_DIR", "work")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    gpu_lock = (work / "gpu.lock").open("a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    model_hash = verify_models(os.environ["MODEL_MANIFEST"])
    rows = []
    for s in samples:
        # Warm-up is recorded separately and excluded from the latency summary.
        for repeat in range(-1, a.repeats):
            peak = [0]
            done = threading.Event()

            def monitor():
                while not done.is_set():
                    try:
                        r = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=2,
                        )
                        peak[0] = max(peak[0], max(map(int, r.split())))
                    except Exception:
                        pass
                    done.wait(0.2)

            t = threading.Thread(target=monitor, daemon=True)
            t.start()
            started = time.monotonic()
            row = {
                "sample": s["id"],
                "repeat": repeat,
                "warmup": repeat < 0,
                "supported": s["supported"],
                "model_lock": model_hash,
                "pipeline_version": os.getenv("PIPELINE_VERSION", "mvp-1"),
            }
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    _, media, phases = execute(s["path"], root, s.get("options", {}))
                    row.update(status="SUCCEEDED", media=media, phase_ms=phases)
                    scores = []
                    for truth in (
                        sorted(Path(s["truth_masks"]).glob("*.png"))
                        if s.get("truth_masks")
                        else []
                    ):
                        pred = cv2.imread(str(root / "masks" / truth.name), 0)
                        gt = cv2.imread(str(truth), 0)
                        assert (
                            pred is not None
                            and gt is not None
                            and pred.shape == gt.shape
                        )
                        scores.append(mask_metrics(pred, gt))
                    if scores:
                        row["mask"] = {
                            key: float(np.mean([score[key] for score in scores]))
                            for key in ("recall", "precision")
                        }
                    row["rtf"] = (time.monotonic() - started) / (
                        media["duration_ms"] / 1000
                    )
            except PipelineError as e:
                row.update(status="NEEDS_REVIEW" if e.review else "FAILED", code=e.code)
            finally:
                done.set()
                t.join(timeout=3)
            row.update(
                elapsed_ms=round((time.monotonic() - started) * 1000),
                peak_vram_mib=peak[0] or None,
            )
            rows.append(row)
            Path(a.output).write_text(json.dumps({"runs": rows}, indent=2))
    measured = [r for r in rows if not r["warmup"]]
    groups = {}
    for name, group in [
        (
            "720p",
            [
                r
                for r in measured
                if r["status"] == "SUCCEEDED"
                and min(r["media"]["width"], r["media"]["height"]) == 720
            ],
        ),
        (
            "lower_resolution",
            [
                r
                for r in measured
                if r["status"] == "SUCCEEDED"
                and min(r["media"]["width"], r["media"]["height"]) < 720
            ],
        ),
        ("failed", [r for r in measured if r["status"] != "SUCCEEDED"]),
    ]:
        times = [r["elapsed_ms"] for r in group]
        groups[name] = {
            "count": len(group),
            "p50_ms": float(np.percentile(times, 50)) if times else None,
            "p95_ms": float(np.percentile(times, 95)) if times else None,
        }
    Path(a.output).write_text(
        json.dumps(
            {"runs": rows, "summary": groups, "total_measured": len(measured)}, indent=2
        )
    )
    print(a.output)


if __name__ == "__main__":
    main()
