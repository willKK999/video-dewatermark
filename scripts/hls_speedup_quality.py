"""用相同时间相位比较七版与候选的真实纹理，记录绝对值和相对损失。"""

import json
import cv2
import numpy as np
from worker.fine_masks import Supports
from worker.matte import temporal_detail, stable_detail_mask
from scripts.hls_detail_verify import occluder_mask


def relative_clarity(args, output):
    native = json.loads(args.native_report.read_text())
    mapping = native["帧映射"]
    supports = Supports(
        args.prior,
        json.loads(args.detected.read_text()),
        np.load(args.alpha)["alpha"],
        args.graphic,
    )
    mh, mw = supports.median.shape
    captures = [
        cv2.VideoCapture(str(p)) for p in (args.source, args.baseline, args.candidate)
    ]
    rates = [c.get(cv2.CAP_PROP_FPS) for c in captures]
    if not rates[0] == rates[2] == 60 or rates[1] not in (30, 60):
        raise ValueError("相对清晰度需要六十帧源片与候选，以及三十或六十帧七版")
    cursors = [-1, -1, -1]
    samples = []
    previous = None
    try:
        for n in range(round(args.duration * rates[1])):
            frames = []
            indices = [round(n * rate / rates[1]) for rate in rates]
            for k, target in enumerate(indices):
                image = None
                while cursors[k] < target:
                    ok, image = captures[k].read()
                    if not ok:
                        raise ValueError("相对清晰度输入帧不足")
                    cursors[k] += 1
                frames.append(image)
            x, baseline, candidate = frames
            h, w = x.shape[:2]
            faint = {}
            other = occluder_mask(
                [args.graphic] if args.graphic else [], mapping[indices[2]], (h, w)
            )
            occupied = other.copy()
            for part in supports.at(mapping[indices[2]], cv2.resize(x, (mw, mh))):
                mask = cv2.resize(part["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
                occupied |= mask
                if part["kind"] == "fixed" and not part["opaque"]:
                    faint[part["id"]] = mask
                else:
                    other |= mask
            if previous is not None:
                (
                    old_x,
                    old_baseline,
                    old_candidate,
                    old_faint,
                    old_other,
                    old_occupied,
                ) = previous
                for identity, mask in faint.items():
                    if identity not in old_faint:
                        continue
                    stable = stable_detail_mask(
                        mask & old_faint[identity], other, old_other
                    )
                    before = temporal_detail(
                        old_x,
                        x,
                        old_baseline,
                        baseline,
                        stable,
                        occupied | old_occupied,
                    )
                    after = temporal_detail(
                        old_x,
                        x,
                        old_candidate,
                        candidate,
                        stable,
                        occupied | old_occupied,
                    )
                    comparable = before["measurable"] and after["measurable"]
                    samples.append(
                        dict(
                            帧=n,
                            秒数=n / rates[1],
                            轨迹=identity,
                            七版=before,
                            候选=after,
                            可比较=comparable,
                            保留比差值=after["retention_ratio"]
                            - before["retention_ratio"]
                            if comparable
                            else None,
                        )
                    )
            previous = x, baseline, candidate, faint, other, occupied
    finally:
        for cap in captures:
            cap.release()
    valid = [s for s in samples if s["可比较"]]
    delta = [s["保留比差值"] for s in valid]
    report = dict(
        全部样本=samples,
        有效样本数=len(valid),
        平均保留比差值=float(np.mean(delta)) if delta else None,
        保留比差值第五百分位=float(np.percentile(delta, 5)) if delta else None,
        下降超过零点零五的样本=[s for s in valid if s["保留比差值"] < -0.05],
        说明="比较与七版相同时间相位；六十帧额外相位另做全部相邻帧检查，不使用绝对零点八门槛",
    )
    (output / "relative-clarity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return report
