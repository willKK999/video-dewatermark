"""在已确认淡水印笔画内校正编码后残留投影；保留低频和时变纹理。"""

import argparse, json, shutil, subprocess, time
from pathlib import Path
import cv2, numpy as np
from .multitrack import load_prior, highpass, component_scores
from .matte import ink_support
from .process import PipelineError
from .vision import sha256
from scripts.hls_chunk_check import mux_core_audio, audio_hash
from .hls_settings import is_residual_score


def fit_correction(template, consensus, mask):
    """沿用v7的字形及一阶导数基、64像素窗和32像素步长；仅拟合时域共识。"""
    g = template.astype(np.float32)
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
    h, w = g.shape
    out = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w), np.float32)
    for y in range(0, h, 32):
        for x in range(0, w, 32):
            a, b = min(w, x + 64), min(h, y + 64)
            active = mask[y:b, x:a] > 0
            if active.sum() < 20:
                continue
            design = np.stack([v[y:b, x:a] for v in [g, dx, dy]], axis=2)
            X = design[active].astype(np.float64)
            if np.sum(X * X) < 1e-4:
                continue
            normal = X.T @ X
            normal += np.eye(3) * max(np.trace(normal) * 0.0001, 1e-6)
            coeff = np.linalg.solve(
                normal, X.T @ consensus[y:b, x:a][active].astype(np.float64)
            )
            weight = (
                np.maximum(0.05, np.hanning(b - y))[:, None]
                * np.maximum(0.05, np.hanning(a - x))[None, :]
            )
            out[y:b, x:a] += (
                np.einsum("ijk,kc->ijc", design, coeff) * weight[:, :, None]
            )
            weights[y:b, x:a] += weight
    out /= np.maximum(weights[:, :, None], 1e-6)
    # 只减去字形方向，不模糊或替换背景，不接触掩膜外。
    feather = np.clip(
        cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5) / 2, 0, 1
    )
    return out * feather[:, :, None]


def apply_correction(image, correction, mask):
    result = np.clip(np.rint(image.astype(np.float32) - correction), 0, 255).astype(
        np.uint8
    )
    result[mask == 0] = image[mask == 0]
    return result


def scan(video, prior, regions):
    cap = cv2.VideoCapture(str(video))
    fps = round(cap.get(5))
    gray = []
    rgb = []
    try:
        n = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if n % 15 == 0:
                gray.append(highpass(cv2.resize(im, (prior.shape[1], prior.shape[0]))))
            if n % fps == 0:
                rgb.append(
                    im.astype(np.float32)
                    - cv2.GaussianBlur(im.astype(np.float32), (0, 0), 3)
                )
            n += 1
    finally:
        cap.release()
    common = np.median(np.stack(gray), axis=0)
    scores = []
    for region in regions:
        x, y, a, b = region["box"]
        parts = component_scores(
            prior[y:b, x:a], common[y:b, x:a], region["mask"][y:b, x:a]
        )
        scores.append(
            dict(
                region=region["id"],
                components=parts,
                residual=any(is_residual_score(p) for p in parts),
            )
        )
    return scores, rgb


def patch(work, assets_path):
    cv2.setNumThreads(2)
    started = time.monotonic()
    assets = json.loads(assets_path.read_text())
    info = json.loads((work / "prepared.json").read_text())
    native = work / "native"
    video = native / "chunk1.mp4"
    digest = sha256(video)
    root = work / "residual-projection"
    root.mkdir(exist_ok=True)
    report_path = root / "report.json"
    if (
        report_path.exists()
        and json.loads(report_path.read_text()).get("输出摘要") == digest
    ):
        return
    prior, signal, regions, _, _ = load_prior(Path(assets["prior"]))
    before, images = scan(video, prior, regions)
    wanted = {p["region"] for p in before if p["residual"]}
    # 专用面板已有独立补修；字形投影不能代替清除不透明整块图形。
    opaque = work / "opaque-panels/masks"
    union = np.zeros(prior.shape, np.uint8)
    if opaque.exists():
        for p in sorted(opaque.glob("*.png"))[::30]:
            union |= cv2.imread(str(p), 0)
    wanted = {
        r["id"]
        for r in regions
        if r["id"] in wanted
        and np.count_nonzero(union & (r["mask"])) < np.count_nonzero(r["mask"]) * 0.25
    }
    if not wanted:
        report_path.write_text(
            json.dumps(
                dict(无需校正=True, 修复前=before, 输出摘要=digest),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if len(images) < 15:
        raise PipelineError(
            "TEMPORAL_UNCERTAIN", "残留投影的独立时域样本不足", review=True
        )
    h, w = images[0].shape[:2]
    template = np.load(assets["template"])["template"]
    assert template.shape == (h, w)
    fields = np.zeros((h, w, 3), np.float32)
    mask = np.zeros((h, w), np.uint8)
    statistics = []
    for region in regions:
        if region["id"] not in wanted:
            continue
        support = cv2.resize(
            ink_support(prior, signal, region["mask"], 3),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        x, y, a, b = [
            (
                round(v * w / prior.shape[1])
                if i % 2 == 0
                else round(v * h / prior.shape[0])
            )
            for i, v in enumerate(region["box"])
        ]
        values = np.stack([im[y:b, x:a] for im in images])
        common = np.median(values, axis=0)
        field = fit_correction(template[y:b, x:a], common, support[y:b, x:a])
        fields[y:b, x:a] += field
        mask |= support
        statistics.append(
            dict(
                区域=region["id"],
                样本数=len(images),
                校正均方根=float(np.sqrt(np.mean(field**2))),
            )
        )
    del images
    np.savez_compressed(root / "correction.npz", correction=fields, mask=mask)
    old = root / f"before-{digest[:12]}.mp4"
    shutil.copyfile(video, old)
    cap = cv2.VideoCapture(str(old))
    count = round(cap.get(7))
    fps = cap.get(5)
    silent = root / "video.mp4"
    log = (root / "encode.log").open("wb")
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            "-threads",
            "4",
            str(silent),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=log,
    )
    mapping = json.loads((native / "native-report.json").read_text())["帧映射"]
    from functools import lru_cache

    @lru_cache(maxsize=8)
    def permitted(index):
        allowed = mask.copy()
        for directory in [
            Path(info["graphics"]) / "masks",
            work / "opaque-panels/masks",
        ]:
            path = directory / f"{index:06d}.png"
            if path.exists():
                occupied = cv2.resize(
                    cv2.imread(str(path), 0), (w, h), interpolation=cv2.INTER_NEAREST
                )
                allowed[cv2.dilate(occupied, np.ones((5, 5), np.uint8)) > 0] = 0
        return allowed

    try:
        for n in range(count):
            ok, image = cap.read()
            if not ok:
                raise PipelineError("OUTPUT_INVALID", "残留投影输入帧不足")
            encoder.stdin.write(
                apply_correction(image, fields, permitted(mapping[n])).tobytes()
            )
        encoder.stdin.close()
        if encoder.wait():
            raise PipelineError("OUTPUT_INVALID", "残留投影无损编码失败")
    finally:
        cap.release()
        log.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    output = root / "patched.mp4"
    mux_core_audio(silent, Path(info["source"]), output, count / fps)
    if audio_hash(old) != audio_hash(output):
        raise PipelineError("OUTPUT_INVALID", "残留投影改变了音频")
    after, samples = scan(output, prior, regions)
    del samples
    report = dict(
        方法="v7编码后时域字形投影校正",
        修复区域=sorted(wanted),
        修复前=before,
        修复后=after,
        校正统计=statistics,
        耗时秒=time.monotonic() - started,
        输入摘要=digest,
        输出摘要=sha256(output),
        音频原样复制=True,
        低频策略="只校正已证实的字形方向，保留v7低频基底及无关时变纹理",
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    output.replace(video)
    for name in ["audit", "verified.json"]:
        if (work / name).exists():
            (work / name).rename(
                work / f"{name}-before-residual-projection-{time.time_ns()}"
            )
    if any(p["residual"] for p in after if p["region"] in wanted):
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "残留字形校正后仍未通过原阈值", review=True
        )
    print(
        "编码后时域字形校正完成，区域",
        sorted(wanted),
        "耗时秒",
        report["耗时秒"],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    a = p.parse_args()
    patch(a.work, a.assets)


if __name__ == "__main__":
    main()
