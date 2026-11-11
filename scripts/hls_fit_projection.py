"""跨背景共同估计固定字形系数，避免逐帧投影抹掉人物运动纹理。"""

import argparse, json
from pathlib import Path
import cv2, numpy as np
from worker.fine_masks import Supports
from worker.matte import projected_texture_composite, solve_projection


def fit(
    coarse,
    source_frames,
    detected,
    prior,
    alpha,
    template,
    graphic,
    output,
    detail_sigma=1.2,
):
    d = json.loads(detected.read_text())
    supports = Supports(prior, d, np.load(alpha)["alpha"], graphic)
    native = np.load(template)["template"]
    h, w = native.shape
    files = sorted((coarse / "frames").glob("*.png"))
    acc = {}
    used = []
    for i in np.linspace(0, len(files) - 1, min(30, len(files))).astype(int):
        small = cv2.imread(str(files[i]))
        parts = supports.at(int(i), small)
        mask = np.zeros((h, w), np.uint8)
        exclude = np.zeros_like(mask)
        for part in parts:
            m = cv2.resize(part["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
            if part["kind"] == "fixed" and not part["opaque"]:
                mask |= m
            else:
                exclude |= m
        mask[exclude > 0] = 0
        if not mask.any():
            continue
        source = cv2.imread(str(source_frames / files[i].name))
        base = cv2.resize(
            cv2.imread(str(coarse / "composite" / files[i].name)),
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        projected_texture_composite(
            source, base, mask, native, accumulator=acc, detail_sigma=detail_sigma
        )
        used.append(int(i))
    if len(used) < 15:
        raise ValueError("跨背景校准样本不足，禁止逐帧过拟合")
    output.write_text(json.dumps(solve_projection(acc)))
    output.with_suffix(".samples.json").write_text(
        json.dumps(
            dict(模型采样帧=used, 样本数=len(used)), ensure_ascii=False, indent=2
        )
    )
    print("固定字形跨背景校准完成，样本数", len(used), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in [
        "coarse",
        "source-frames",
        "detected",
        "prior",
        "alpha",
        "template",
        "graphic",
        "output",
    ]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--detail-sigma", type=float, default=1.2)
    a = p.parse_args()
    cv2.setNumThreads(2)
    fit(
        a.coarse,
        a.source_frames,
        a.detected,
        a.prior,
        a.alpha,
        a.template,
        a.graphic,
        a.output,
        a.detail_sigma,
    )


if __name__ == "__main__":
    main()
