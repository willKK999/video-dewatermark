"""按解码内容校准模型序列与原片帧号，保留无重编码的原生参考像素。"""

import argparse, json, time
from pathlib import Path
from fractions import Fraction
import cv2, numpy as np
from worker.media import probe


def align(
    source,
    frames,
    output,
    model_fps=30,
    source_start_frame=0,
    check=lambda: None,
    source_end_frame=None,
):
    output.mkdir(parents=True, exist_ok=True)
    video = next(s for s in probe(source)["streams"] if s["codec_type"] == "video")
    ratio = float(Fraction(video["avg_frame_rate"])) / model_fps
    cap = cv2.VideoCapture(str(source))
    if source_start_frame and (
        not cap.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)
        or abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - source_start_frame) > 0.01
    ):
        cap.release()
        raise ValueError("原生参考无法精确定位块起点")
    cache = {}
    read = -1
    last = 0
    records = []
    started = time.monotonic()
    for i, path in enumerate(sorted(frames.glob("*.png"))):
        check()
        center = round(i * ratio)
        lo = max(last, center - 6)
        hi = (
            min(center + 6, source_end_frame - source_start_frame - 1)
            if source_end_frame is not None
            else center + 6
        )
        while read < hi:
            ok, image = cap.read()
            if not ok:
                break
            read += 1
            cache[read] = (image, cv2.resize(image, (160, 90)).astype(np.float32))
        target = cv2.resize(cv2.imread(str(path)), (160, 90)).astype(np.float32)
        candidates = [
            (float(np.abs(small - target).mean()), j)
            for j, (image, small) in cache.items()
            if lo <= j <= hi
        ]
        if not candidates:
            raise ValueError("原片帧不足，参考校准失败")
        error, chosen = min(candidates)
        if error > 8:
            raise ValueError("原生参考与模型帧无法可靠对齐，禁止错位回填")
        last = chosen
        cv2.imwrite(
            str(output / path.name), cache[chosen][0], [cv2.IMWRITE_PNG_COMPRESSION, 2]
        )
        records.append(dict(model_frame=i, source_frame=chosen, mean_error=error))
        cache = {j: v for j, v in cache.items() if j >= max(0, center - 8)}
        if i % 120 == 0:
            print("原生参考校准帧数", i, flush=True)
    cap.release()
    report = dict(
        帧对应=records,
        原片起始帧=source_start_frame,
        耗时秒=time.monotonic() - started,
        方法="有限时域窗口逐帧内容匹配，保留原始解码像素，帧号单调不减",
    )
    (output / "alignment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print("原生参考时间校准完成", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["source", "frames", "output"]:
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(2)
    align(a.source, a.frames, a.output)


if __name__ == "__main__":
    main()
