"""多进程模型队列不能同时持有单卡推理槽位。"""

from concurrent.futures import ProcessPoolExecutor
import time
import multiprocessing
from worker.gpu_lock import device_lock


def locked_interval(path):
    with device_lock(path):
        first = time.monotonic()
        time.sleep(0.05)
        last = time.monotonic()
    return first, last


def test_gpu_slot_is_exclusive_across_processes(tmp_path):
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        a, b = sorted(pool.map(locked_interval, [str(tmp_path / "gpu.lock")] * 2))
    assert a[1] <= b[0]
