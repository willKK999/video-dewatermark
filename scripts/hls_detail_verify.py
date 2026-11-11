"""对编码后的全部相邻帧核验清晰度，同时覆盖原始帧率的两个相位。"""

import argparse, json, time
from pathlib import Path
import cv2, numpy as np
from worker.fine_masks import Supports
from worker.matte import temporal_detail, stable_detail_mask


def verify(source, result, mapping, detected, prior, alpha, graphic, output):
    started = time.monotonic()
    indices = json.loads(mapping.read_text())["帧映射"]
    supports = Supports(
        prior, json.loads(detected.read_text()), np.load(alpha)["alpha"], graphic
    )
    mh, mw = supports.median.shape
    a = cv2.VideoCapture(str(source))
    b = cv2.VideoCapture(str(result))
    fps = b.get(5)
    w, h = int(b.get(3)), int(b.get(4))
    previous = None
    stats = []
    if abs(a.get(5) - fps) > 0.001:
        raise ValueError("清晰度验证要求原片与成品帧率相同")
    for n, i in enumerate(indices):
        ok, x = a.read()
        good, y = b.read()
        if not ok or not good:
            raise ValueError("编码视频帧数不足")
        faint = {}
        occupied = np.zeros((h, w), np.uint8)
        other = np.zeros_like(occupied)
        for p in supports.at(i, cv2.resize(x, (mw, mh))):
            mask = cv2.resize(p["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
            occupied |= mask
            if p["kind"] == "fixed" and not p["opaque"]:
                faint[p["id"]] = mask
            else:
                other |= mask
        if previous is not None:
            old_x, old_y, old_faint, old_other, old_occupied = previous
            for identity, mask in faint.items():
                if identity not in old_faint:
                    continue
                stable = stable_detail_mask(
                    mask & old_faint[identity], other, old_other
                )
                metric = temporal_detail(
                    old_x, x, old_y, y, stable, occupied | old_occupied
                )
                stats.append(
                    dict(
                        frame=n,
                        time=n / fps,
                        track_id=identity,
                        phase=n % 2,
                        stable_fraction=np.count_nonzero(stable)
                        / max(1, np.count_nonzero(mask)),
                        **metric,
                    )
                )
        previous = x, y, faint, other, occupied
        if n % 300 == 0:
            print("编码后清晰度复检帧数", n, flush=True)
    a.release()
    b.release()
    valid = [s for s in stats if s["measurable"]]
    failed = [s for s in valid if not s["passed"]]
    report = dict(
        通过=None,
        清晰度仅记录=True,
        历史参考阈值=0.8,
        低于零点七的样本=[s for s in valid if s["retention_ratio"] < 0.7],
        帧数=len(indices),
        有效样本数=len(valid),
        低于历史参考线样本数=len(failed),
        最小保留比=min((s["retention_ratio"] for s in valid), default=None),
        耗时秒=time.monotonic() - started,
        核验方式="全部相邻帧、包含两个帧率相位；排除其他移动水印，清晰度只记录，绝对值不判失败",
        低于历史参考线样本=failed,
        全部样本=stats,
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        "编码后清晰度记录",
        "历史参考线以下样本",
        len(failed),
        "最小保留比",
        report["最小保留比"],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "source",
        "result",
        "mapping",
        "detected",
        "prior",
        "alpha",
        "graphic",
        "output",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(2)
    verify(
        a.source, a.result, a.mapping, a.detected, a.prior, a.alpha, a.graphic, a.output
    )


if __name__ == "__main__":
    main()
