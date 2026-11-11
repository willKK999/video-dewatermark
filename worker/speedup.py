"""提速路径的有界预取与明确编码策略；不探测或占用设备。"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor


def bounded_map(function, iterable, depth=2):
    if not 1 <= depth <= 8:
        raise ValueError("预取队列深度必须为一至八")
    iterator = iter(iterable)
    with ThreadPoolExecutor(max_workers=1) as executor:
        queue = deque()
        for _ in range(depth):
            try:
                queue.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while queue:
            result = queue.popleft().result()
            try:
                queue.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
            yield result


def encoder_args(encoder="libx264", crf=10):
    if not 0 <= crf <= 51:
        raise ValueError("视频量化参数无效")
    if encoder == "libx264":
        return ["-c:v", encoder, "-preset", "fast", "-crf", str(crf)]
    if encoder == "h264_nvenc":
        # 恒定量化模式避免低码率上限；与七版的视觉等效性仍须实测。
        return [
            "-c:v",
            encoder,
            "-preset",
            "p7",
            "-tune",
            "hq",
            "-rc",
            "constqp",
            "-qp",
            str(crf),
            "-multipass",
            "fullres",
            "-spatial-aq",
            "1",
            "-temporal-aq",
            "1",
            "-forced-idr",
            "1",
            "-no-scenecut",
            "1",
        ]
    raise ValueError("仅支持软件编码或显卡高质量编码")
