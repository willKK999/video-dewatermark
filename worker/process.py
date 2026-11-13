"""Interruptible subprocesses. No shell expansion; kill descendants on lease loss."""

import os
import signal
import subprocess
import tempfile
import time
import json
from pathlib import Path


class PipelineError(Exception):
    def __init__(self, code, message, retryable=False, review=False, details=None):
        super().__init__(message)
        self.code, self.retryable, self.review = code, retryable, review
        self.details = details or {}


class LeaseLost(Exception):
    pass


def run(args, check=lambda: None, cwd=None, timeout=900, *, stage=None, event_path=None, workload=None):
    check()
    # File-backed capture prevents pipe deadlock and unbounded RAM consumption.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        p = subprocess.Popen(
            [str(a) for a in args],
            cwd=cwd,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        started = time.monotonic()
        event = dict(stage=stage, timeout_seconds=timeout, status="running",
                     workload=workload or {}, command=Path(str(args[0])).name)
        def record():
            if event_path is not None:
                Path(event_path).write_text(json.dumps(event, ensure_ascii=False, indent=2))
        try:
            record()
            while p.poll() is None:
                check()
                if time.monotonic() - started > timeout:
                    raise PipelineError(
                        "PROCESS_TIMEOUT", "stage execution timed out", retryable=True,
                        details=dict(stage=stage, timeout_seconds=timeout,
                                     elapsed_seconds=time.monotonic()-started,
                                     workload=workload or {})
                    )
                time.sleep(0.1)
            check()
            err.seek(max(0, err.tell() - 8192))
            detail = err.read().decode(errors="replace")
            if p.returncode:
                if "out of memory" in detail.lower():
                    raise PipelineError(
                        "GPU_OOM", "GPU memory exhausted", retryable=True
                    )
                # Do not surface source paths, media contents or signed URLs.
                raise PipelineError(
                    "PROCESS_FAILED",
                    f"{os.path.basename(str(args[0]))} exited {p.returncode}",
                )
            out.seek(0)
            event["status"] = "complete"
            return out.read()
        except BaseException as exc:
            event.update(status="failed", error_code=getattr(exc,"code",type(exc).__name__))
            raise
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            event["elapsed_seconds"] = time.monotonic()-started
            record()
