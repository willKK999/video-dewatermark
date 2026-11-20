"""视频去水印系统：跨进程串行使用单卡模型，允许准备及合成并行。"""

from contextlib import contextmanager
import fcntl
import os
import time
from pathlib import Path


@contextmanager
def device_lock(path=None, check=None):
    path = Path(
        path or os.getenv("GPU_STAGE_LOCK", "/tmp/video-dewatermark-gpu-stage.lock")
    )
    with path.open("a") as handle:
        if check is None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        else:
            while True:
                check()
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
