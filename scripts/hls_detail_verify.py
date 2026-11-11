"""记录编码后的清晰度样本；覆盖两个帧率相位及生命周期边界，不作为验收硬门槛。"""

import argparse, json, time
from pathlib import Path
import cv2, numpy as np
from worker.hls_acceptance import clarity_summary
from worker.fine_masks import Supports
from worker.matte import temporal_detail, stable_detail_mask


def sample_pairs(indices, detected, stride=30):
    if stride < 1:
        raise ValueError("采样步长必须为正数")
    count = len(indices)
    selected = {n for n in range(1, count) if n % stride in (0, 1)}
    selected.update(n for n in [1, 2, count - 2, count - 1] if 0 < n < count)
    boundaries = {a for a, b in detected["scenes"]} | {b for a, b in detected["scenes"]}
    for track in detected["tracks"]:
        if track["kind"] == "moving" and track["hints"]:
            boundaries.update([track["hints"][0]["frame"], track["hints"][-1]["frame"]])
    for n, index in enumerate(indices):
        if index in boundaries:
            selected.update(k for k in range(n - 2, n + 3) if 0 < k < count)
    return selected


def verify(
    source, result, mapping, detected, prior, alpha, graphic, output, sample_stride=30
):
    started = time.monotonic()
    indices = json.loads(mapping.read_text())["帧映射"]
    detections = json.loads(detected.read_text())
    supports = Supports(prior, detections, np.load(alpha)["alpha"], graphic)
    selected = sample_pairs(indices, detections, sample_stride)
    mh, mw = supports.median.shape
    a = cv2.VideoCapture(str(source))
    b = cv2.VideoCapture(str(result))
    fps = b.get(5)
    w, h = int(b.get(3)), int(b.get(4))
    previous = None
    stats = []
    if abs(a.get(5) - fps) > 0.001:
        raise ValueError("清晰度记录要求原片与成品帧率相同")

    def masks(i, x):
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
        return faint, occupied, other

    try:
        for n, i in enumerate(indices):
            ok, x = a.read()
            good, y = b.read()
            if not ok or not good:
                raise ValueError("清晰度记录视频帧数不足")
            if previous is not None and n in selected:
                old_x, old_y, old_i = previous
                faint, occupied, other = masks(i, x)
                old_faint, old_occupied, old_other = masks(old_i, old_x)
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
                            **metric
                        )
                    )
            previous = x, y, i
            if n % 300 == 0:
                print("编码后清晰度记录帧数", n, flush=True)
    finally:
        a.release()
        b.release()
    report = dict(
        **clarity_summary(stats),
        帧数=len(indices),
        采样步长=sample_stride,
        选择帧号=sorted(selected),
        耗时秒=time.monotonic() - started,
        核验方式="每半秒相邻两对覆盖帧率两个相位，并加采转场、移动目标生命周期边界；清晰度只记录，低于0.7单列；残留和闪烁另做全帧复检"
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        "编码后清晰度记录完成，低于0.7样本数",
        report["低于0.7样本数"],
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
    p.add_argument(
        "--sample-stride",
        type=int,
        default=30,
        help="1表示记录全部相邻帧；默认30覆盖半秒内两个相位并加采边界",
    )
    a = p.parse_args()
    cv2.setNumThreads(2)
    verify(
        a.source,
        a.result,
        a.mapping,
        a.detected,
        a.prior,
        a.alpha,
        a.graphic,
        a.output,
        a.sample_stride,
    )


if __name__ == "__main__":
    main()
