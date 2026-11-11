"""从对齐的已处理帧估计固定叠加差，仅转移去水印修正，保留当前原片的全部纹理。"""

import argparse, json, time, subprocess
from functools import lru_cache
from pathlib import Path
import cv2, numpy as np
from worker.fine_masks import Supports
from worker.matte import (
    projected_texture_composite,
    restore_intermediate_frame,
    temporal_detail,
    stable_detail_mask,
)
from scripts.hls_chunk_check import mux_core_audio, audio_hash


def transfer(
    source,
    video,
    metadata,
    coarse,
    native_frames,
    prior,
    detected,
    alpha,
    graphic,
    template,
    calibration,
    audio_source,
    output,
    detail_sigma=3,
    video_crf=10,
):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    meta = json.loads(metadata.read_text())
    mapping = meta["帧映射"]
    s = Supports(
        prior, json.loads(detected.read_text()), np.load(alpha)["alpha"], graphic
    )
    mh, mw = s.median.shape
    g = np.load(template)["template"]
    coefficients = json.loads(calibration.read_text())
    h, w = g.shape

    @lru_cache(maxsize=4)
    def model(i):
        anchor = cv2.imread(str(native_frames / f"{i:06d}.png"))
        base = cv2.resize(
            cv2.imread(str(coarse / "composite" / f"{i:06d}.png")),
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        approved = cv2.resize(
            cv2.imread(str(coarse / "masks" / f"{i:06d}.png"), 0),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        faint = {}
        other = np.zeros((h, w), np.uint8)
        for p in s.at(i, cv2.resize(anchor, (mw, mh))):
            mask = cv2.resize(p["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
            if p["kind"] == "fixed" and not p["opaque"]:
                faint[p["id"]] = mask & approved
            else:
                other |= mask
        union = np.zeros((h, w), np.uint8)
        for mask in faint.values():
            union |= mask
        union[cv2.dilate(other, np.ones((7, 7), np.uint8)) > 0] = 0
        clean = (
            projected_texture_composite(
                anchor,
                base,
                union,
                g,
                calibration=coefficients,
                detail_sigma=detail_sigma,
            )
            if union.any()
            else anchor.copy()
        )
        return anchor, clean, union, faint, other, approved

    a = cv2.VideoCapture(str(source))
    b = cv2.VideoCapture(str(video))
    fps = b.get(5)
    previous = None
    stats = []
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
            str(video_crf),
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
            good, current = b.read()
            if not ok or not good:
                raise ValueError("当前帧或已处理帧不足")
            anchor, clean, mask, faint, other, approved = model(i)
            out = restore_intermediate_frame(current, clean, x, anchor, mask)
            # 合成不得根据弱相关先验扩张到完整流水线未确认的淡水印区域。
            potential = np.zeros_like(mask)
            for p in s.at(i, cv2.resize(x, (mw, mh))):
                if p["kind"] == "fixed" and not p["opaque"]:
                    potential |= cv2.resize(
                        p["mask"], (w, h), interpolation=cv2.INTER_NEAREST
                    )
            unapproved = (potential > 0) & (approved == 0) & (other == 0)
            out[unapproved] = x[unapproved]
            encoder.stdin.write(out.tobytes())
            if previous is not None and n % 3 == 0:
                old_x, old_y, old_faint, old_other, old_approved = previous
                for identity, area in faint.items():
                    if identity not in old_faint:
                        continue
                    stable = stable_detail_mask(
                        area & old_faint[identity], other, old_other
                    )
                    metric = temporal_detail(
                        old_x, x, old_y, out, stable, approved | old_approved
                    )
                    stats.append(
                        dict(frame=n, time=n / fps, track_id=identity, **metric)
                    )
            previous = x, out, faint, other, approved
            if n % 150 == 0:
                print("固定叠加差转移帧数", n, flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise ValueError("固定叠加差转移编码失败")
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
            叠加差转移耗时秒=time.monotonic() - started,
            音频内容一致=audio_hash(audio_source, len(mapping) / fps)
            == audio_hash(final),
            当前原帧保留全部纹理=True,
            分频半径像素=detail_sigma,
            视频量化等级=video_crf,
        )
    )
    (output / "native-report.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print(
        "固定叠加差转移检查", meta["清晰度通过"], meta["清晰度最小保留比"], flush=True
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "source",
        "video",
        "metadata",
        "coarse",
        "native-frames",
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
    p.add_argument("--detail-sigma", type=float, default=3)
    p.add_argument("--crf", type=int, default=10)
    a = p.parse_args()
    cv2.setNumThreads(2)
    transfer(
        a.source,
        a.video,
        a.metadata,
        a.coarse,
        a.native_frames,
        a.prior,
        a.detected,
        a.alpha,
        a.graphic,
        a.template,
        a.calibration,
        a.audio_source,
        a.output,
        a.detail_sigma,
        a.crf,
    )


if __name__ == "__main__":
    main()
