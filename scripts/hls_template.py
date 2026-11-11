"""从多时段共识建立原分辨率分频模板，不依赖影片坐标。"""

import argparse, time
from pathlib import Path
import cv2, numpy as np


def build(frames, output, sigma):
    files = sorted(frames.glob("*.jpg"))
    if len(files) < 15:
        raise ValueError("模板时域采样不足")
    started = time.monotonic()
    groups = []
    for indices in np.array_split(np.arange(len(files)), 5):
        values = []
        for i in indices:
            gray = cv2.cvtColor(cv2.imread(str(files[i])), cv2.COLOR_BGR2GRAY).astype(
                np.float32
            )
            values.append(gray - cv2.GaussianBlur(gray, (0, 0), sigma))
        groups.append(np.median(np.stack(values), axis=0))
    np.savez_compressed(
        output, template=np.median(np.stack(groups), axis=0), detail_sigma=sigma
    )
    print(
        "原分辨率多时段模板完成，分频半径、耗时秒",
        sigma,
        time.monotonic() - started,
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sigma", type=float, default=3)
    a = p.parse_args()
    cv2.setNumThreads(2)
    build(a.frames, a.output, a.sigma)


if __name__ == "__main__":
    main()
