"""Interruptible subprocesses. No shell expansion; kill descendants on lease loss."""

import os
import signal
import subprocess
import tempfile
import time


class PipelineError(Exception):
    def __init__(self, code, message, retryable=False, review=False):
        super().__init__(message)
        self.code, self.retryable, self.review = code, retryable, review


class LeaseLost(Exception):
    pass


def run(args, check=lambda: None, cwd=None, timeout=900):
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
        try:
            while p.poll() is None:
                check()
                if time.monotonic() - started > timeout:
                    raise PipelineError(
                        "PROCESS_TIMEOUT", "stage execution timed out", retryable=True
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
            return out.read()
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
