"""有界并发发布中间产物，全部完成后才允许提交任务成功。"""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time


def upload_artifacts(s3, bucket, prefix, root, options, check, workers=8):
    paths = [root / 'manifest.json', root / 'detections.json']
    paths.extend(sorted((root / 'masks').glob('*.png')))

    def upload(path):
        check()
        relative = path.relative_to(root).as_posix()
        content_type = 'image/png' if path.suffix == '.png' else 'application/json'
        s3.upload_file(str(path), bucket, prefix+'/'+relative,
                       ExtraArgs=options('intermediate', ContentType=content_type),
                       Callback=lambda _: check())
        check()

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        try:
            for path in paths:
                check()
                pending.add(pool.submit(upload, path))
                if len(pending) >= workers*2:
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in finished:
                        future.result()
            for future in pending:
                future.result()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return round((time.monotonic()-started)*1000)
