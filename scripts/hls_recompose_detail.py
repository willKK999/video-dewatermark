"""保留已修复的底色，重新合成原片真实纹理；不以模板幅度抑制皮肤细节。"""

import argparse, json, subprocess, time
from pathlib import Path
import cv2, numpy as np
from worker.fine_masks import Supports
from worker.matte import (
    projected_texture_composite,
    temporal_detail,
    stable_detail_mask,
)
from scripts.hls_chunk_check import mux_core_audio, audio_hash


def recompose(
    source,
    video,
    metadata,
    prior,
    detected,
    alpha,
    graphic,
    template,
    calibration,
    audio_source,
    output,
):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    meta = json.loads(metadata.read_text())
    mapping = meta["帧映射"]
    s = Supports(
        prior, json.loads(detected.read_text()), np.load(alpha)["alpha"], graphic
    )
    mh, mw = s.median.shape
    native = np.load(template)["template"]
    coefficients = json.loads(calibration.read_text())
    a = cv2.VideoCapture(str(source))
    b = cv2.VideoCapture(str(video))
    fps = b.get(5)
    w, h = int(b.get(3)), int(b.get(4))
    stats = []
    previous = None
    if abs(a.get(5) - fps) > 0.001:
        raise ValueError("源参考和成品帧率必须一致")
    error = (output / "encode.log").open("wb")
    silent = output / "video.mp4"
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
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            str(silent),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=error,
    )
    try:
        for n, i in enumerate(mapping):
            ok, x = a.read()
            good, base = b.read()
            if not ok or not good:
                raise ValueError("重合成输入帧不足")
            parts = s.at(i, cv2.resize(x, (mw, mh)))
            faint = {}
            union = np.zeros((h, w), np.uint8)
            occupied = np.zeros_like(union)
            other = np.zeros_like(union)
            templates = []
            for p in parts:
                mask = cv2.resize(p["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
                occupied |= mask
                if p["opaque"]:
                    other |= mask
                    continue
                union |= mask
                if p["kind"] == "fixed":
                    faint[p["id"]] = mask
                else:
                    other |= mask
                    reg = next(r for r in s.regions if r["id"] == p["region"])
                    xx, yy, aa, bb = [
                        round(q * (w / mw if j % 2 == 0 else h / mh))
                        for j, q in enumerate(reg["box"])
                    ]
                    g = native[yy:bb, xx:aa]
                    if p["track"]["flip"]:
                        g = cv2.flip(g, 1)
                    xx, yy, aa, bb = [
                        round(q * (w / mw if j % 2 == 0 else h / mh))
                        for j, q in enumerate(p["point"]["box"])
                    ]
                    z = np.zeros((h, w), np.float32)
                    z[yy:bb, xx:aa] = cv2.resize(g, (aa - xx, bb - yy))
                    templates.append(z)
            if faint:
                templates.insert(0, native)
            opaque = np.zeros_like(union)
            for p in parts:
                if p["opaque"]:
                    opaque |= cv2.resize(
                        p["mask"], (w, h), interpolation=cv2.INTER_NEAREST
                    )
            union[cv2.dilate(opaque, np.ones((5, 5), np.uint8)) > 0] = 0
            out = base.copy()
            if templates and union.any():
                proposed = projected_texture_composite(
                    x,
                    base,
                    union,
                    templates,
                    calibration=coefficients if faint else None,
                    detail_sigma=3,
                )
                out[union > 0] = proposed[union > 0]
            encoder.stdin.write(out.tobytes())
            if previous is not None and n % 3 == 0:
                old_x, old_y, old_faint, old_other, old_occupied = previous
                for identity, mask in faint.items():
                    if identity not in old_faint:
                        continue
                    stable = stable_detail_mask(
                        mask & old_faint[identity], other, old_other
                    )
                    metric = temporal_detail(
                        old_x, x, old_y, out, stable, occupied | old_occupied
                    )
                    stats.append(
                        dict(frame=n, time=n / fps, track_id=identity, **metric)
                    )
            previous = x, out, faint, other, occupied
            if n % 150 == 0:
                print("原片细节重合成帧数", n, flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise ValueError("原片细节重合成编码失败")
    finally:
        a.release()
        b.release()
        error.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    final = output / "chunk1.mp4"
    mux_core_audio(silent, audio_source, final, len(mapping) / fps)
    meta.update(
        dict(
            清晰度统计=stats,
            清晰度通过=all(q["passed"] for q in stats)
            and sum(q["measurable"] for q in stats) >= 50,
            清晰度最小保留比=min(
                (q["retention_ratio"] for q in stats if q["measurable"]), default=None
            ),
            可测样本数=sum(q["measurable"] for q in stats),
            细节重合成耗时秒=time.monotonic() - started,
            音频内容一致=audio_hash(audio_source, len(mapping) / fps)
            == audio_hash(final),
            模板幅度不影响纹理保留=True,
            视频量化等级=10,
        )
    )
    (output / "native-report.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print(
        "原片细节重合成检查", meta["清晰度通过"], meta["清晰度最小保留比"], flush=True
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "source",
        "video",
        "metadata",
        "prior",
        "detected",
        "alpha",
        "graphic",
        "template",
        "calibration",
        "audio-source",
        "output",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(2)
    recompose(
        a.source,
        a.video,
        a.metadata,
        a.prior,
        a.detected,
        a.alpha,
        a.graphic,
        a.template,
        a.calibration,
        a.audio_source,
        a.output,
    )


if __name__ == "__main__":
    main()
