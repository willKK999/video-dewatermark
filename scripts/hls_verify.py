"""视频去水印系统：对最终 MP4 与 HLS 做离线时序、音频、残留及接缝审计。"""

import argparse
import hashlib
from fractions import Fraction
import json
from pathlib import Path
import shutil
import cv2
import numpy as np

from worker.workflow import interpolated_box

from worker.hls import audio_timeline_metrics, seam_metrics
from worker.media import probe
from worker.process import run
from worker.vision import sha256


def temporal_windows(source, result, plans, fps, references, output):
    """用接缝附近普通帧对作基线，镜头切换及不可靠对齐不强行打分。"""
    from scripts.quality_v3 import align_previous

    records = []
    info = probe(result)
    video = next(x for x in info["streams"] if x["codec_type"] == "video")
    width, height = video["width"], video["height"]
    for index, plan in enumerate(plans[1:], 1):
        boundary = plan["start"]
        first = max(0, boundary - 15)
        folder = output / f"temporal-{index:02d}"
        folder.mkdir()
        for label, path in (("source", source), ("result", result)):
            target = folder / label
            target.mkdir()
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    path,
                    "-vf",
                    f"select=between(n\\,{first}\\,{first+30}),scale={width}:{height}",
                    "-fps_mode",
                    "passthrough",
                    "-frames:v",
                    "31",
                    "-start_number",
                    "0",
                    target / "%04d.png",
                ]
            )
        originals = [
            cv2.imread(str(p)) for p in sorted((folder / "source").glob("*.png"))
        ]
        results = [
            cv2.imread(str(p)) for p in sorted((folder / "result").glob("*.png"))
        ]
        pairs = []
        cuts = {s for s, e in references["scenes"]}
        for j in range(1, min(len(originals), len(results))):
            frame = first + j
            if frame in cuts:
                continue
            boxes = [
                interpolated_box(references["hints"], k) for k in (frame - 1, frame)
            ]
            x1 = max(0, int(min(b[0] for b in boxes)) - 12)
            y1 = max(0, int(min(b[1] for b in boxes)) - 12)
            x2 = min(width, int(np.ceil(max(b[2] for b in boxes))) + 12)
            y2 = min(height, int(np.ceil(max(b[3] for b in boxes))) + 12)
            mask = np.zeros((height, width), np.uint8)
            mask[y1:y2, x1:x2] = 255
            aligned = align_previous(
                originals[j],
                originals[j - 1],
                [results[j - 1], originals[j - 1]],
                mask,
                mask,
            )
            if aligned is None:
                continue
            core = mask > 0
            ring = (cv2.dilate(mask, np.ones((17, 17), np.uint8)) > 0) & ~core
            delta = np.abs(
                results[j].astype(np.float32) - aligned[0].astype(np.float32)
            ).mean(axis=2)
            repair_delta = np.abs(
                (results[j].astype(np.float32) - originals[j])
                - (aligned[0].astype(np.float32) - aligned[1])
            ).mean(axis=2)
            pairs.append(
                dict(
                    frame=frame,
                    is_seam=frame == boundary,
                    result_mae=float(delta[core].mean()),
                    ring_mae=float(delta[ring].mean()),
                    repair_delta_mae=float(repair_delta[core].mean()),
                )
            )
        seam = next((p for p in pairs if p["is_seam"]), None)
        ordinary = [p["result_mae"] for p in pairs if not p["is_seam"]]
        record = dict(
            frame=boundary, time=boundary / fps, aligned_pairs=len(pairs), seam=seam
        )
        if ordinary:
            record.update(
                local_median_mae=float(np.median(ordinary)),
                local_p95_mae=float(np.percentile(ordinary, 95)),
            )
            if seam:
                record["seam_over_local_p95"] = seam["result_mae"] / max(
                    record["local_p95_mae"], 0.01
                )
        records.append(record)
        shutil.rmtree(folder)
    return records


def sample_sheets(source, result, plans, fps, references, output):
    """每秒自动裁取水印轨迹，便于覆盖整片目检；左原片、右成品。"""
    caps = [cv2.VideoCapture(str(path)) for path in (source, result)]
    rows, full_rows = {}, {}
    sample_frames = {
        round(t * fps) for t in range(int(np.ceil(plans[-1]["end"] / fps)))
    }
    total = 0
    try:
        for frame in range(plans[-1]["end"]):
            images = [cap.read()[1] for cap in caps]
            if any(image is None for image in images):
                raise ValueError("抽样视频帧数不一致")
            if frame not in sample_frames:
                continue
            original, clean = images
            h, w = clean.shape[:2]
            original = cv2.resize(original, (w, h))
            a, b, c, d = interpolated_box(references["hints"], frame)
            x1, y1 = max(0, int(a) - 24), max(0, int(b) - 24)
            x2, y2 = min(w, int(np.ceil(c)) + 24), min(h, int(np.ceil(d)) + 24)
            pair = np.hstack([original[y1:y2, x1:x2], clean[y1:y2, x1:x2]])
            scale = min(560 / pair.shape[1], 100 / pair.shape[0])
            pair = cv2.resize(pair, None, fx=scale, fy=scale)
            tile = np.zeros((122, 560, 3), np.uint8)
            tile[22 : 22 + pair.shape[0], : pair.shape[1]] = pair
            cv2.putText(
                tile,
                f"{frame/fps:.2f}s",
                (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            index = next(
                i for i, p in enumerate(plans) if p["start"] <= frame < p["end"]
            )
            rows.setdefault(index, []).append(tile)
            second = round(frame / fps)
            if second % 5 == 0:
                cv2.imwrite(
                    str(output / f"full-{second:04d}s.jpg"),
                    np.hstack([original, clean]),
                )
                full = np.zeros((449, 480, 3), np.uint8)
                scale = min(240 / w, 427 / h)
                for column, image in enumerate((original, clean)):
                    thumb = cv2.resize(image, None, fx=scale, fy=scale)
                    full[
                        22 : 22 + thumb.shape[0],
                        column * 240 : column * 240 + thumb.shape[1],
                    ] = thumb
                cv2.putText(
                    full,
                    f"{frame/fps:.2f}s",
                    (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )
                full_rows.setdefault(index, []).append(full)
            total += 1
    finally:
        for cap in caps:
            cap.release()
    for index, tiles in rows.items():
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack(
            [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
        )
        cv2.imwrite(str(output / f"block-{index+1:02d}-samples.jpg"), sheet)
    for index, tiles in full_rows.items():
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack(
            [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
        )
        cv2.imwrite(str(output / f"block-{index+1:02d}-full-samples.jpg"), sheet)
    return total


def verify(source, result, playlist, report, references, output, detect=False):
    output.mkdir(parents=True, exist_ok=True)
    source_info = probe(source)
    result_info = probe(result, count=True)
    hls_info = probe(playlist, count=True)
    sv = next(x for x in source_info["streams"] if x["codec_type"] == "video")
    rv = next(x for x in result_info["streams"] if x["codec_type"] == "video")
    hv = next(x for x in hls_info["streams"] if x["codec_type"] == "video")
    fps = float(Fraction(rv["avg_frame_rate"]))
    assert (
        int(rv["nb_read_frames"])
        == int(hv["nb_read_frames"])
        == report["chunks"][-1]["end"]
    ), "成品帧数不一致"
    assert (
        abs(float(rv["duration"]) - float(sv["duration"])) <= 1 / fps + 0.001
    ), "成品时长不一致"
    for path in (result, playlist):
        run(["ffmpeg", "-v", "error", "-xerror", "-i", path, "-f", "null", "-"])

    def decoded_hashes(path):
        text = run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                path,
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "framemd5",
                "-",
            ]
        ).decode()
        return [
            line.rsplit(",", 1)[-1].strip()
            for line in text.splitlines()
            if line and not line.startswith("#")
        ]

    mp4_hashes, hls_hashes = decoded_hashes(result), decoded_hashes(playlist)
    assert mp4_hashes == hls_hashes and len(mp4_hashes) == int(
        rv["nb_read_frames"]
    ), "HLS 切分改变了解码像素"
    video_digest = hashlib.sha256("".join(mp4_hashes).encode()).hexdigest()
    timing = {}
    if any(x["codec_type"] == "audio" for x in source_info["streams"]):
        hashes = []
        for path in (source, result, playlist):
            audio = output / "audio-check.aac"
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    path,
                    "-map",
                    "0:a:0",
                    "-c",
                    "copy",
                    "-f",
                    "adts",
                    audio,
                ]
            )
            hashes.append(sha256(audio))
        assert len(set(hashes)) == 1, "压缩音频内容不一致"
        timing = audio_timeline_metrics(
            source, result, playlist, report["source_start_pts"]
        )
        timing["audio_sha256"] = hashes[0]
    frames = output / "frames"
    quality = {}
    if detect:
        from worker import vision

        frames.mkdir()
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                result,
                "-start_number",
                "0",
                frames / "%08d.png",
            ]
        )
        (output / "composite").symlink_to(frames.resolve(), target_is_directory=True)
        (output / "detections.json").write_text(json.dumps(references))
        references = {
            **references,
            "scenes": vision.shots(sorted(frames.glob("*.png"))),
        }
        (output / "detections.json").write_text(json.dumps(references))
        vision.residual_quality(output, sorted(frames.glob("*.png")), references)
        quality = json.loads((output / "quality.json").read_text())
        assert (
            quality["residual_count"] == 0 and not quality["needs_review"]
        ), "编码成品存在残留证据"
    seams = seam_metrics(
        source,
        frames,
        report["chunks"],
        fps,
        output,
        lambda: None,
        references["hints"],
        result,
    )
    sampled = sample_sheets(source, result, report["chunks"], fps, references, output)
    motion = temporal_windows(source, result, report["chunks"], fps, references, output)
    summary = {
        **report,
        "seams": seams,
        "seams_from_encoded_video": True,
        **timing,
        "source_duration_ms": round(float(sv["duration"]) * 1000),
        "result_duration_ms": round(float(rv["duration"]) * 1000),
        "hls_frame_count": int(hv["nb_read_frames"]),
        "hls_mp4_decoded_frames_equal": True,
        "decoded_frame_digest": video_digest,
        "encoded_quality": quality,
        "motion_compensated_seams": motion,
        "sampled_frames": sampled,
        "full_sampled_frames": len(list(output.glob("full-*s.jpg"))),
    }
    (output / "verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    if frames.exists():
        (output / "composite").unlink()
        shutil.rmtree(frames)
    print(
        "最终成品帧数、时长、解码、音频与接缝审计通过；接缝数：" + str(len(seams)),
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "result", "playlist", "report", "references", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--detect", action="store_true", help="在 GPU 运行环境中逐帧复检最终编码成品"
    )
    args = parser.parse_args()
    verify(
        args.source,
        args.result,
        args.playlist,
        json.loads(args.report.read_text()),
        json.loads(args.references.read_text()),
        args.output,
        args.detect,
    )


if __name__ == "__main__":
    main()
