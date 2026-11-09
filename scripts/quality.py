"""Mask-region PSNR/SSIM on aligned clean-master and repaired PNG frames."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def score(clean, repaired, mask):
    if (
        clean.shape != repaired.shape
        or clean.shape[:2] != mask.shape
        or not (mask > 0).any()
    ):
        raise ValueError("aligned frames and a nonempty mask required")
    region = mask > 0
    x, y = clean.astype(np.float64), repaired.astype(np.float64)
    mse = np.square(x - y)[region].mean()
    psnr = float(10 * np.log10(255**2 / mse)) if mse else None
    # Standard 11x11 Gaussian local SSIM, averaged over the mask and channels.
    blur = lambda a: cv2.GaussianBlur(a, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    vx, vy = blur(x * x) - mx * mx, blur(y * y) - my * my
    cov = blur(x * y) - mx * my
    ssim = ((2 * mx * my + 6.5025) * (2 * cov + 58.5225)) / (
        (mx * mx + my * my + 6.5025) * (vx + vy + 58.5225)
    )
    return {"psnr_db": psnr, "identical": mse == 0, "ssim": float(ssim[region].mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean", required=True)
    p.add_argument("--repaired", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--output", default="quality.json")
    a = p.parse_args()
    rows = []
    for mask in sorted(Path(a.masks).glob("*.png")):
        clean = cv2.imread(str(Path(a.clean) / mask.name))
        repaired = cv2.imread(str(Path(a.repaired) / mask.name))
        m = cv2.imread(str(mask), 0)
        if clean is None or repaired is None or m is None:
            raise ValueError("missing frame")
        if (m > 0).any():
            rows.append({"frame": mask.name, **score(clean, repaired, m)})
    if not rows:
        raise ValueError("no scored frames")
    finite = [r["psnr_db"] for r in rows if r["psnr_db"] is not None]
    Path(a.output).write_text(
        json.dumps(
            {
                "frames": rows,
                "mean_psnr_db": float(np.mean(finite)) if finite else None,
                "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
            },
            indent=2,
        )
    )
    print(a.output)


if __name__ == "__main__":
    main()
