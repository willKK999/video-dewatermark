"""发布必须保留完整产物、并发上限和取消/故障传播。"""
import threading
from pathlib import Path

import pytest

from worker.publication import upload_artifacts
from worker.process import LeaseLost


def artifacts(root, count=25):
    (root/'masks').mkdir()
    for name in ['manifest.json', 'detections.json']:
        (root/name).write_text('{}')
    for i in range(count):
        (root/'masks'/f'{i:06d}.png').write_bytes(b'mask')


def test_bounded_concurrency_preserves_every_key_and_callback(tmp_path):
    artifacts(tmp_path)
    lock = threading.Lock()
    barrier = threading.Barrier(3)
    active = peak = 0
    keys = []
    checks = []

    class Storage:
        def upload_file(self, path, bucket, key, ExtraArgs, Callback):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                first = len(keys) < 3
                keys.append(key)
            if first:
                barrier.wait(timeout=5)
            assert bucket == 'bucket'
            assert ExtraArgs['kind'] == 'intermediate'
            assert Path(path).is_file()
            Callback(4)
            with lock:
                active -= 1

    upload_artifacts(Storage(), 'bucket', 'attempt', tmp_path,
                     lambda kind, **kw: dict(kind=kind, **kw),
                     lambda: checks.append(True), workers=3)
    expected = {'attempt/'+p.relative_to(tmp_path).as_posix()
                for p in tmp_path.rglob('*') if p.is_file()}
    assert set(keys) == expected and len(keys) == len(expected)
    assert peak == 3 and active == 0 and checks


@pytest.mark.parametrize('failure', [OSError, LeaseLost])
def test_failure_never_returns_success_or_leaves_uploads_running(tmp_path, failure):
    artifacts(tmp_path)
    active = 0
    lock = threading.Lock()

    class Storage:
        def upload_file(self, *args, **kwargs):
            nonlocal active
            with lock:
                active += 1
            try:
                raise failure('中止发布')
            finally:
                with lock:
                    active -= 1

    with pytest.raises(failure):
        upload_artifacts(Storage(), 'bucket', 'attempt', tmp_path,
                         lambda kind, **kw: kw, lambda: None, workers=2)
    assert active == 0
