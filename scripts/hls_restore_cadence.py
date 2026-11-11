"""以固定水印轨迹内的真实帧差恢复原始帧率，禁止把移动水印差分加回。"""

import argparse, json, subprocess, time
from pathlib import Path
import cv2, numpy as np
from worker.fine_masks import Supports
from worker.matte import restore_intermediate_frame, temporal_detail, stable_detail_mask
from scripts.hls_chunk_check import mux_core_audio, audio_hash


def restore(
    source,
    video,
    coarse,
    detected,
    prior,
    alpha,
    graphic,
    alignment,
    native_report,
    output,
):
    output.mkdir(exist_ok=True, parents=True)
    started = time.monotonic()
    report = json.loads(native_report.read_text())
    mapping = report["帧映射"]
    positions = [p["source_frame"] for p in json.loads(alignment.read_text())["帧对应"]]
    d = json.loads(detected.read_text())
    supports = Supports(prior, d, np.load(alpha)["alpha"], graphic)
    mh, mw = supports.median.shape
    a = cv2.VideoCapture(str(source))
    b = cv2.VideoCapture(str(video))
    fps = b.get(cv2.CAP_PROP_FPS)
    w, h = int(b.get(3)), int(b.get(4))
    cache = {}
    read = -1
    stats = []
    previous = None
    changed = 0
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
            "12",
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
            anchor = min(len(mapping) - 1, positions[i])
            while read < max(n, anchor):
                ok, x = a.read()
                good, y = b.read()
                if not ok or not good:
                    raise ValueError("原片或成品帧数不足")
                read += 1
                cache[read] = (x, y)
            original, current = cache[n]
            anchor_source, anchor_clean = cache[anchor]
            parts = supports.at(i, cv2.resize(original, (mw, mh)))
            faint = {}
            others = np.zeros((h, w), np.uint8)
            for p in parts:
                mask = cv2.resize(p["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
                if p["kind"] == "fixed" and not p["opaque"]:
                    faint[p["id"]] = mask
                else:
                    others |= mask
            union = np.zeros((h, w), np.uint8)
            for mask in faint.values():
                union |= mask
            union[cv2.dilate(others, np.ones((7, 7), np.uint8)) > 0] = 0
            result = (
                restore_intermediate_frame(
                    current, anchor_clean, original, anchor_source, union
                )
                if anchor != n and union.any()
                else current.copy()
            )
            changed += int(anchor != n and union.any())
            encoder.stdin.write(result.tobytes())
            if previous is not None and n % 6 == 0:
                old_source, old_result, old_faint, old_others = previous
                for identity, mask in faint.items():
                    if identity not in old_faint:
                        continue
                    stable = stable_detail_mask(
                        mask & old_faint[identity], old_others, others
                    )
                    metric = temporal_detail(
                        old_source,
                        original,
                        old_result,
                        result,
                        stable,
                        others | old_others | union,
                    )
                    stats.append(
                        dict(frame=n, time=n / fps, track_id=identity, **metric)
                    )
            previous = (original, result, faint, others)
            cache = {j: v for j, v in cache.items() if j >= n - 12}
            if n % 300 == 0:
                print("原生时序恢复帧数", n, flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise ValueError("时序恢复编码失败")
    finally:
        a.release()
        b.release()
        error.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    final = output / "chunk1.mp4"
    mux_core_audio(silent, source, final, len(mapping) / fps)
    report.update(
        dict(
            上一版清晰度通过=report["清晰度通过"],
            清晰度统计=stats,
            清晰度通过=all(s["passed"] for s in stats)
            and sum(s["measurable"] for s in stats) >= 50,
            可测样本数=sum(s["measurable"] for s in stats),
            清晰度最小保留比=min(
                (s["retention_ratio"] for s in stats if s["measurable"]), default=None
            ),
            原生帧间变化恢复帧数=changed,
            时序恢复耗时秒=time.monotonic() - started,
            音频内容一致=audio_hash(source, len(mapping) / fps) == audio_hash(final),
            未处理后续块=True,
        )
    )
    (output / "native-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print("原生时序恢复完成，清晰度检查", report["清晰度通过"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "source",
        "video",
        "coarse",
        "detected",
        "prior",
        "alpha",
        "graphic",
        "alignment",
        "native-report",
        "output",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(2)
    restore(
        a.source,
        a.video,
        a.coarse,
        a.detected,
        a.prior,
        a.alpha,
        a.graphic,
        a.alignment,
        a.native_report,
        a.output,
    )


if __name__ == "__main__":
    main()
