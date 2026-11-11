"""视频去水印系统：复用已接受的通用先验，有界并行准备、补画、原生合成与核验。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import threading
import cv2
import numpy as np
from .process import PipelineError, run
from .vision import sha256
from .hls_acceptance import clarity_summary


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def load_assets(path, source_digest):
    assets = json.loads(path.read_text())
    if assets["source_sha256"] != source_digest or assets.get("method") != "v7":
        raise PipelineError("OUTPUT_INVALID", "复用先验与冻结源片或已接受方案不一致")
    for name, digest in assets["sha256"].items():
        if sha256(Path(assets[name])) != digest:
            raise PipelineError("OUTPUT_INVALID", "复用先验文件摘要不一致")
    return assets


def link(path, target):
    target = Path(target).resolve()
    if path.is_symlink():
        if path.resolve() == target:
            return
        path.unlink()
    if not path.exists():
        path.symlink_to(target, target_is_directory=target.is_dir())


def graphic_masks(files, assets, root, check):
    """位置由完整图案多尺度匹配产生；没有影片名称、坐标或人工框。"""
    from .brand_templates import variants, match, mask_for

    color = cv2.imread(assets["color_seed_image"])
    color_mask = cv2.imread(assets["color_seed_mask"], 0)
    color_bank = variants(
        color, color_mask, [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0]
    )
    white = cv2.imread(assets["variant_seed_image"])
    white_mask = cv2.imread(assets["variant_seed_mask"], 0)
    white_bank = variants(white, white_mask, [0.95, 1, 1.05])
    for name in ["color", "variant", "graphics"]:
        (root / name / "masks").mkdir(parents=True, exist_ok=True)
    for name in ["seed-image.png", "seed-mask.png"]:
        shutil.copyfile(
            assets["variant_" + name.replace("-", "_").replace(".png", "")],
            root / "variant" / name,
        )
    records, intervals = [], []
    for index, file in enumerate(files):
        check()
        image = cv2.imread(str(file))
        c = match(image, color_bank, 0.55, 2)
        v = match(image, white_bank, 0.5, 2)
        cm = mask_for(image.shape[:2], color_bank, c)
        vm = mask_for(image.shape[:2], white_bank, v)
        cv2.imwrite(str(root / "color/masks" / file.name), cm)
        cv2.imwrite(str(root / "variant/masks" / file.name), vm)
        cv2.imwrite(str(root / "graphics/masks" / file.name), cm | vm)
        if c:
            if intervals and intervals[-1][1] == index:
                intervals[-1][1] += 1
            else:
                intervals.append([index, index + 1])
        records.append(dict(frame=index, matches=v, color_matches=c))
    write_json(root / "variant/tracks.json", dict(frames=records))
    # 固定标识的显隐判定只在彩色开场图案期间启用，不因白色水印全片存在而错误关掉固定轨迹。
    write_json(root / "color/detections.json", dict(scenes=intervals))
    write_json(root / "graphics/detections.json", dict(scenes=intervals))


def prepare_block(full, work, plan, assets, check):
    started = time.monotonic()
    work.mkdir(parents=True, exist_ok=True)
    descriptor = work / "prepared.json"
    if descriptor.exists():
        return json.loads(descriptor.read_text())
    seconds = (plan["right"] - plan["left"]) / 30
    clip = work / "source.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-threads",
            "2",
            "-ss",
            str(plan["left"] / 30),
            "-i",
            full,
            "-t",
            str(seconds),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "0",
            "-threads",
            "2",
            "-c:a",
            "copy",
            "-avoid_negative_ts",
            "disabled",
            clip,
        ],
        check,
        timeout=1800,
    )
    for name in ["frames", "reference"]:
        (work / name).mkdir(exist_ok=True)
    count = plan["right"] - plan["left"]
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-threads",
            "2",
            "-i",
            clip,
            "-filter_complex",
            "fps=30,split[a][b];[a]scale=1280:720[x];[b]null[y]",
            "-map",
            "[x]",
            "-frames:v",
            str(count),
            "-threads",
            "2",
            "-start_number",
            "0",
            work / "frames/%06d.png",
            "-map",
            "[y]",
            "-frames:v",
            str(count),
            "-threads",
            "2",
            "-start_number",
            "0",
            work / "reference/%06d.png",
        ],
        check,
        timeout=1800,
    )
    files = sorted((work / "frames").glob("*.png"))
    if len(files) != count:
        raise PipelineError("OUTPUT_INVALID", "处理块规范化帧数不一致")
    from . import vision
    from .multitrack import make_tracks

    detected = make_tracks(
        files, vision.shots(files, check), Path(assets["prior"]), work, check
    )
    write_json(
        work / "reference/alignment.json",
        {"帧对应": [{"source_frame": i * 2} for i in range(count)]},
    )
    graphic_masks(files, assets, work, check)
    info = dict(
        reused_v7=False,
        coarse=str(work / "fine"),
        detected=str(work / "detections.json"),
        graphics=str(work / "graphics"),
        color=str(work / "color"),
        variant=str(work / "variant"),
        reference=str(work / "reference"),
        source=str(clip),
        seconds=seconds,
        prepare_seconds=time.monotonic() - started,
    )
    write_json(descriptor, info)
    return info


def gpu_block(work, assets, info, engine, stage, check):
    from .gpu_lock import device_lock

    work.mkdir(parents=True, exist_ok=True)
    with device_lock(work / "gpu-stage.lock", check):
        return _gpu_block(work, assets, info, engine, stage, check)


def _gpu_block(work, assets, info, engine, stage, check):
    marker = work / "gpu.json"
    if marker.exists():
        return json.loads(marker.read_text())
    if info["reused_v7"]:
        stats = dict(reused_v7=True, gpu_seconds=0, accepted_intermediate=True)
        write_json(marker, stats)
        return stats
    from . import vision
    from .fine_masks import Supports

    files = sorted((work / "frames").glob("*.png"))
    d = json.loads(Path(info["detected"]).read_text())
    started = time.monotonic()
    stage("detect")
    if not (work / "all-yolo.json").exists():
        run(
            [
                os.environ["DETECT_PYTHON"],
                "-m",
                "scripts.hls_candidates",
                "--frames",
                work / "frames",
                "--output",
                work / "all-yolo.json",
            ],
            check,
            timeout=1800,
        )
    stage("mask")
    if not (work / "sam-complete.json").exists():
        # SAM 对每个镜头及目标生命周期独立初始化，已有多轨迹不会阻止新增目标。
        if (work / "masks").exists():
            shutil.rmtree(work / "masks")
        vision.masks(work, files, d, engine, check)
        write_json(work / "sam-complete.json", dict(完成=True))
    from .colour_tracks import complete as complete_colour
    from .hls_layers import track_bins, write_manifest
    from .hls_repair import intervals

    colour_tracking = complete_colour(work, assets, d, check)
    fine = work / "fine"
    if (
        fine.exists()
        and not (fine / "complete.json").exists()
        and not (fine / "layer-schema.json").exists()
    ):
        fine.rename(work / f"fine-before-layering-{time.time_ns()}")
    fine.mkdir(exist_ok=True)
    link(fine / "frames", work / "frames")
    (fine / "masks").mkdir(exist_ok=True)
    support = Supports(
        Path(assets["prior"]), d, np.load(assets["alpha"])["alpha"], Path(info["color"])
    )
    areas = []
    maximum_by_track = {}
    for i, file in enumerate(files):
        check()
        mask = np.zeros((720, 1280), np.uint8)
        for part in support.at(i, cv2.imread(str(file))):
            if part["kind"] == "graphic":
                continue
            mask |= part["mask"]
            maximum_by_track[part["id"]] = max(
                maximum_by_track.get(part["id"], 0),
                float(np.count_nonzero(part["mask"]) / mask.size),
            )
        areas.append(float(np.count_nonzero(mask) / mask.size))
        cv2.imwrite(str(fine / "masks" / file.name), mask)
    groups = (
        [list(maximum_by_track)] if max(areas) <= 0.2 else track_bins(maximum_by_track)
    )
    write_json(
        fine / "layer-schema.json", dict(method="v7-verified-layers-1", groups=groups)
    )
    layer_dirs = []
    if len(groups) == 1:
        base_roots = [fine]
    else:
        base_roots = [fine] + [
            work / f"base-layer-{i:02d}" for i in range(1, len(groups))
        ]
        for item in base_roots:
            item.mkdir(exist_ok=True)
            (item / "masks").mkdir(exist_ok=True)
        for i, file in enumerate(files):
            parts = [
                p
                for p in support.at(i, cv2.imread(str(file)))
                if p["kind"] != "graphic"
            ]
            for item, identities in zip(base_roots, groups):
                mask = np.zeros((720, 1280), np.uint8)
                for part in parts:
                    if part["id"] in identities:
                        mask |= part["mask"]
                cv2.imwrite(str(item / "masks" / file.name), mask)
    stage("inpaint")
    performance_paths = []

    def apply_layer(target, source_frames, mask_dir, short_context=False):
        target.mkdir(exist_ok=True)
        link(target / "frames", source_frames)
        if mask_dir != target / "masks":
            link(target / "masks", mask_dir)
        if not (target / "complete.json").exists():
            for folder in [target / "composite", *target.glob("window-*")]:
                if folder.exists():
                    shutil.rmtree(folder)
            active = []
            for i, file in enumerate(files):
                mask = cv2.imread(str(target / "masks" / file.name), 0)
                if np.count_nonzero(mask) / mask.size > 0.2:
                    raise PipelineError(
                        "MASK_UNCERTAIN", "单遍自动图层超过20%保护", review=True
                    )
                if mask.any():
                    active.append(i)
            scenes = (
                intervals(active, d["scenes"], len(files), 30)
                if short_context
                else [(a, b) for a, b in d["scenes"] if any(a <= i < b for i in active)]
            )
            if scenes:
                vision.inpaint(
                    target,
                    sorted((target / "frames").glob("*.png")),
                    scenes,
                    engine,
                    True,
                    check,
                )
            else:
                (target / "composite").mkdir()
            for file in files:
                if not (target / "composite" / file.name).exists():
                    os.link(source_frames / file.name, target / "composite" / file.name)
            write_json(
                target / "complete.json",
                dict(完成=True, 包含目标帧=len(active), 单遍保护=0.2),
            )
        layer_dirs.append(target / "masks")
        performance_paths.append(target / "performance.json")
        return target / "composite"

    previous = work / "frames"
    for target in base_roots:
        previous = apply_layer(target, previous, target / "masks")
    # 与已接受方案一致，彩色完整图形和白色独立品牌分遍；不因真实多水印的总面积拒绝。
    previous = apply_layer(
        work / "color-repair", previous, Path(info["color"]) / "masks", True
    )
    extra = work / "variant-repair"
    previous = apply_layer(extra, previous, Path(info["variant"]) / "masks")
    merged = work / "coarse"
    merged.mkdir(exist_ok=True)
    link(merged / "frames", work / "frames")
    link(merged / "composite", previous)
    (merged / "masks").mkdir(exist_ok=True)
    union_max = 0.0
    for file in files:
        mask = np.zeros((720, 1280), np.uint8)
        for directory in layer_dirs:
            mask |= cv2.imread(str(directory / file.name), 0)
        union_max = max(union_max, float(np.count_nonzero(mask) / mask.size))
        cv2.imwrite(str(merged / "masks" / file.name), mask)
    write_manifest(merged, Path(info["detected"]), layer_dirs, files)
    info["coarse"] = str(merged)
    write_json(work / "prepared.json", info)
    stats = dict(
        reused_v7=False,
        gpu_seconds=time.monotonic() - started,
        mask_max_fraction=union_max,
        single_pass_limit=0.2,
        verified_layers=len(layer_dirs),
        colour_tracking=colour_tracking,
        sam=(
            json.loads((work / "metrics.sam.json").read_text())
            if (work / "metrics.sam.json").exists()
            else {}
        ),
        stages=[json.loads(p.read_text()) for p in performance_paths if p.exists()],
    )
    write_json(marker, stats)
    return stats


def native_block(work, assets_path):
    from .gpu_lock import device_lock

    with device_lock(work / "native-stage.lock"):
        return _native_block(work, assets_path)


def _native_block(work, assets_path):
    """独立进程运行，避免多块光流缓存叠加常驻；参数与 v7 相同。"""
    cv2.setNumThreads(2)
    from scripts.hls_native_chunk import render
    from .hls_settings import HLSSettings

    settings = HLSSettings.from_env()

    info = json.loads((work / "prepared.json").read_text())
    assets = json.loads(assets_path.read_text())
    native = work / "native"
    if not (native / "native-report.json").exists():
        render(
            Path(info["source"]),
            Path(info["coarse"]),
            Path(info["detected"]),
            Path(assets["prior"]),
            Path(assets["alpha"]),
            Path(assets["template"]),
            Path(info["graphics"]),
            native,
            info["seconds"],
            Path(info["reference"]),
            Path(assets["projection"]),
            60,
            settings.detail_sigma,
            settings.video_crf,
            True,
            layer_manifest=(
                (Path(info["coarse"]) / "layers.json")
                if (Path(info["coarse"]) / "layers.json").exists()
                else None
            ),
        )
    from .opaque_panels import patch as complete_panels
    from .hls_graphic_repair import patch as complete_graphics
    from .hls_round import execute_round, legacy_attempts

    def low_stage():
        if assets.get("low_band_template"):
            complete_graphics(work, assets_path, Path(assets["low_band_template"]))

    attempts = legacy_attempts(work)
    attempts["v7原生去水印"] = dict(
        通过=True,
        证据文件=str(native / "native-report.json"),
        耗时秒=json.loads((native / "native-report.json").read_text()).get(
            "渲染耗时秒"
        ),
    )
    report = execute_round(
        work,
        [
            ("v7原生去水印", lambda: None),
            ("低频检查与补修", low_stage),
            ("不透明面板补修", lambda: complete_panels(work, assets_path)),
            ("编码后复检", lambda: verify_block(work, assets, info)),
        ],
        native / "chunk1.mp4",
        attempts,
    )
    if not (work / "verified.json").exists():
        write_json(
            work / "verified.json",
            dict(
                通过=False,
                残留=None,
                低频残留=None,
                清晰度最低值=None,
                清晰度低于0点7样本数=None,
                核验耗时秒=0,
                候选数=None,
                视频摘要=report["视频摘要"],
                未通过原因=report["未通过原因"],
            ),
        )
    return report


def verify_block(work, assets, info):
    from scripts.hls_native_verify import remap_references
    from scripts.hls_brand_verify import verify as brand_verify
    from scripts.hls_detail_verify import verify as detail_verify
    from .multitrack import residual_quality

    started = time.monotonic()
    native = work / "native"
    report_path = native / "native-report.json"
    metadata = json.loads(report_path.read_text())
    result = native / "chunk1.mp4"
    audit = work / "audit"
    audit.mkdir(exist_ok=True)
    encoded = audit / "composite"
    encoded.mkdir(exist_ok=True)
    mapping = metadata["帧映射"]
    detections = json.loads(Path(info["detected"]).read_text())
    references = remap_references(detections, mapping)
    references["prior"] = assets["prior"]
    if not (audit / "quality.json").exists():
        cap = cv2.VideoCapture(str(result))

        def reader(index):
            ok, image = cap.read()
            if not ok:
                raise PipelineError("OUTPUT_INVALID", "编码后原规格帧数不足")
            return cv2.resize(image, (1280, 720))

        try:
            residual_quality(
                audit,
                [Path(f"{i:06d}.png") for i in range(len(mapping))],
                references,
                frame_reader=reader,
            )
            if cap.read()[0]:
                raise PipelineError("OUTPUT_INVALID", "编码后原规格帧数多于映射")
        except PipelineError as exc:
            if exc.code != "RESIDUAL_UNCERTAIN":
                raise
        finally:
            cap.release()
    if not (audit / "brand.json").exists():
        brand_verify(
            result,
            Path(info["coarse"]) / "frames",
            report_path,
            Path(info["color"]),
            Path(info["variant"]),
            Path(assets["palette"]),
            audit / "brand.json",
        )
    if not (audit / "clarity.json").exists():
        detail_verify(
            Path(info["source"]),
            result,
            report_path,
            Path(info["detected"]),
            Path(assets["prior"]),
            Path(assets["alpha"]),
            Path(info["graphics"]),
            audit / "clarity.json",
        )
    quality = json.loads((audit / "quality.json").read_text())
    brand = json.loads((audit / "brand.json").read_text())
    clarity = json.loads((audit / "clarity.json").read_text())
    low = None
    current_assets = (
        json.loads(Path(os.environ["HLS_V7_ASSETS"]).read_text())
        if os.getenv("HLS_V7_ASSETS")
        else assets
    )
    if current_assets.get("low_band_template"):
        from .hls_graphic_repair import verify_low
        from .multitrack import load_prior

        low = verify_low(
            result,
            np.load(current_assets["low_band_template"])["template"],
            load_prior(Path(assets["prior"]))[2],
        )
        write_json(audit / "low-band.json", low)
    from .opaque_panels import verify as panel_verify

    panel = panel_verify(result, work)
    write_json(audit / "opaque.json", panel)
    candidates = quality["residual_count"] + brand["残留证据数"] + panel["候选数"]
    low_count = low["残留区域数"] if low else 0
    previous_summary = (
        json.loads((work / "verified.json").read_text())
        if (work / "verified.json").exists()
        else {}
    )
    summary = dict(
        候选数=candidates + low_count,
        高通及品牌候选数=candidates,
        残留=candidates + low_count,
        低频残留=low_count,
        不透明面板复检=panel,
        低频复检=low,
        清晰度最低值=clarity["最小保留比"],
        清晰度低于0点7样本数=clarity["低于0.7样本数"],
        核验耗时秒=previous_summary.get("核验耗时秒", 0) + time.monotonic() - started,
        核验轮次=previous_summary.get("核验轮次", 1) + 1 if previous_summary else 1,
        视频摘要=sha256(result),
        通过=candidates == 0 and low_count == 0,
    )
    # 自动候选原样保留。误报只能由与当前视频摘要绑定的逐项人工证据排除。
    review_path = work / "review.json"
    if candidates and review_path.exists():
        review = json.loads(review_path.read_text())
        items = review.get("逐项复核", [])
        if (
            review.get("视频摘要") == summary["视频摘要"]
            and len(items) == candidates
            and all(
                p.get("结论") == "误报"
                and p.get("证据文件")
                and Path(p["证据文件"]).is_file()
                for p in items
            )
        ):
            summary.update(残留=low_count, 通过=low_count == 0, 人工误报复核=review)
    write_json(work / "verified.json", summary)
    shutil.rmtree(encoded)
    print(
        "单块编码后核验完成："
        + json.dumps(
            {
                k: summary.get(k)
                for k in [
                    "通过",
                    "残留",
                    "低频残留",
                    "清晰度最低值",
                    "清晰度低于0点7样本数",
                ]
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def scale_plan(plan, factor):
    return {k: round(v * factor) for k, v in plan.items()}


def join_videos(works, plans, output, full, cuts, check):
    """在同一时间戳线性融合上下文；只缓存交界处，音频从整片复制。"""
    from .media import probe

    video = next(
        s
        for s in probe(works[0] / "native/chunk1.mp4")["streams"]
        if s["codec_type"] == "video"
    )
    fps = float(Fraction(video["avg_frame_rate"]))
    from .hls_settings import HLSSettings

    settings = HLSSettings.from_env()
    w, h = video["width"], video["height"]
    silent = output.parent / "joined-video.mp4"
    args = [
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
        str(settings.video_crf),
        "-threads",
        "4",
        "-pix_fmt",
        "yuv420p",
        "-force_key_frames",
        ",".join(str(x) for x in [0] + cuts),
        "-sc_threshold",
        "0",
        str(silent),
    ]
    errors = (output.parent / "join-encode.log").open("wb")
    encoder = subprocess.Popen(
        args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
    )
    pending = {}
    cursor = 0
    previous_right = 0
    try:
        for i, (work, plan) in enumerate(zip(works, plans)):
            cap = cv2.VideoCapture(str(work / "native/chunk1.mp4"))
            for local in range(plan["right"] - plan["left"]):
                check()
                ok, image = cap.read()
                if not ok:
                    raise PipelineError("OUTPUT_INVALID", "接缝合并输入帧不足")
                n = plan["left"] + local
                if n in pending:
                    weight = (n - plan["left"] + 1) / (
                        min(previous_right, plan["right"]) - plan["left"] + 1
                    )
                    image = cv2.addWeighted(pending[n], 1 - weight, image, weight, 0)
                pending[n] = image
                # 当前块后续帧不会再修改已越过的时间戳。
                flush_before = (
                    plans[i + 1]["left"] if i + 1 < len(plans) else plan["right"]
                )
                while cursor < min(flush_before, n + 1) and cursor in pending:
                    encoder.stdin.write(pending.pop(cursor).tobytes())
                    cursor += 1
            cap.release()
            previous_right = plan["right"]
        if pending or cursor != plans[-1]["end"]:
            raise PipelineError("OUTPUT_INVALID", "合并时间轴有缺帧或多余帧")
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise PipelineError("OUTPUT_INVALID", "整集原规格编码失败")
    finally:
        errors.close()
        if encoder.poll() is None:
            encoder.stdin.close()
            encoder.terminate()
            try:
                encoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            silent,
            "-i",
            full,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output,
        ],
        check,
        timeout=1800,
    )


def execute_native(
    full,
    root,
    options,
    engine,
    stage,
    check,
    s3,
    bucket,
    checkpoint_prefix,
    progress,
    prior_path,
    segments,
    lines,
    digest,
    source_pts,
    source_fps,
    started,
):
    if abs(source_fps - 60) > 0.001:
        raise PipelineError(
            "HLS_UNSUPPORTED",
            "当前原生 v7 验收模式要求60帧源片，其他帧率使用常规 HLS 流程",
        )
    cv2.setNumThreads(2)
    assets_path = Path(os.environ["HLS_V7_ASSETS"])
    source_digest = sha256(full)
    assets = load_assets(assets_path, source_digest)
    from .hls import chunks
    from .hls_package import package

    plans30 = chunks([d for d, _ in segments], 30)
    plans = [scale_plan(p, 2) for p in plans30]
    fingerprint = hashlib.sha256(
        (
            source_digest
            + json.dumps(assets["sha256"], sort_keys=True)
            + "v7-native60-record-clarity-1"
        ).encode()
    ).hexdigest()[:24]
    persistent = Path(os.environ["HLS_V7_WORKDIR"]) / fingerprint
    persistent.mkdir(parents=True, exist_ok=True)
    write_json(
        persistent / "run.json",
        dict(
            方案="v7",
            清晰度仅记录=True,
            源摘要=source_digest,
            计划=plans,
            原始计划=plans30,
            开始时间=datetime.now(timezone.utc).isoformat(),
            分频半径=3,
            量化等级=10,
        ),
    )
    full_saved = persistent / "source.mp4"
    if not full_saved.exists():
        shutil.copyfile(full, full_saved)
    works = [persistent / f"chunk-{i+1:02d}" for i in range(len(plans))]
    times = []
    completed = set()
    stage_failures = {}
    native_futures = {}
    completion_lock = threading.RLock()
    active = {"index": 0, "phase": "probe"}
    pipeline_started = time.monotonic()
    # 两个准备槽位、两个原生合成槽位，GPU 只有主循环一条队列。
    with ThreadPoolExecutor(max_workers=2) as preparers, ThreadPoolExecutor(
        max_workers=2
    ) as natives:
        prepared = {}

        def submit_prepare(i):
            if i < len(works) and i not in prepared:
                prepared[i] = preparers.submit(
                    prepare_block, full_saved, works[i], plans30[i], assets, check
                )

        submit_prepare(0)
        submit_prepare(1)

        def record_failure(index, exc):
            with completion_lock:
                stage_failures[index] = dict(index=index + 1, code=exc.code)
                write_json(
                    persistent / "stage-failures.json", list(stage_failures.values())
                )
                print(
                    f"块号={index+1}/{len(works)} 阶段失败={exc.code}；保留检查点，继续其余块",
                    flush=True,
                )

        def collect_finished(wait_all=False):
            # 等待必须在锁外，完成回调才能释放线程并启动排队的下一块。
            if wait_all:
                for k, future in list(native_futures.items()):
                    try:
                        future.result()
                    except PipelineError as exc:
                        record_failure(k, exc)
            with completion_lock:
                for k, future in list(native_futures.items()):
                    if (
                        k in completed
                        or k in stage_failures
                        or (not wait_all and not future.done())
                    ):
                        continue
                    try:
                        future.result()
                    except PipelineError as exc:
                        record_failure(k, exc)
                        continue
                    verified = json.loads((works[k] / "verified.json").read_text())
                    gpu = json.loads((works[k] / "gpu.json").read_text())
                    info = json.loads((works[k] / "prepared.json").read_text())
                    native = json.loads(
                        (works[k] / "native/native-report.json").read_text()
                    )
                    phases = dict(
                        准备=info["prepare_seconds"],
                        补画=gpu["gpu_seconds"],
                        原生合成=native["渲染耗时秒"],
                        核验=verified["核验耗时秒"],
                    )
                    for name, label in [
                        ("complete-graphics/report.json", "低频补修"),
                        ("opaque-panels/report.json", "不透明面板补修"),
                    ]:
                        path = works[k] / name
                        if path.exists():
                            phases[label] = json.loads(path.read_text()).get(
                                "耗时秒", 0
                            )
                    elapsed = sum(phases.values())
                    record = dict(
                        index=k + 1,
                        elapsed_ms=round(elapsed * 1000),
                        phase_seconds=phases,
                        residual_count=verified["残留"],
                        clarity_min=verified["清晰度最低值"],
                        clarity_record_only=True,
                        reused_v7=gpu["reused_v7"],
                        **plans[k],
                    )
                    times.append(record)
                    completed.add(k)
                    line = f"{datetime.now(timezone.utc).isoformat()} 块号={k+1}/{len(works)} 耗时秒={elapsed:.3f} 残留={verified['残留']} 清晰度最低值={verified['清晰度最低值']} 复检候选={verified['候选数']}\n"
                    with (persistent / "progress.log").open("a") as stream:
                        stream.write(line)
                    write_json(
                        persistent / "completed.json",
                        sorted(times, key=lambda x: x["index"]),
                    )
                    print(line.strip(), flush=True)
                    progress(
                        active["index"] or len(works),
                        len(works),
                        active["phase"],
                        completed_chunks=len(completed),
                        last_completed_chunk=k + 1,
                    )

        for i, work in enumerate(works):
            check()
            active.update(index=i + 1, phase="probe")
            collect_finished()
            progress(i + 1, len(works), "probe", completed_chunks=len(completed))
            submit_prepare(i + 2)
            try:
                info = prepared[i].result()
            except PipelineError as exc:
                record_failure(i, exc)
                continue

            def chunk_stage(name):
                active["phase"] = name
                stage(name)
                progress(i + 1, len(works), name, completed_chunks=len(completed))

            try:
                gpu_block(work, assets, info, engine, chunk_stage, check)
            except PipelineError as exc:
                record_failure(i, exc)
                continue
            if (work / "verified.json").exists():
                from concurrent.futures import Future

                f = Future()
                f.set_result(None)
                native_futures[i] = f
            else:
                native_futures[i] = natives.submit(
                    run,
                    [
                        os.sys.executable,
                        "-m",
                        "worker.hls_v7",
                        "--native",
                        str(work),
                        "--assets",
                        str(assets_path),
                    ],
                    check,
                    timeout=14400,
                )
            native_futures[i].add_done_callback(lambda future: collect_finished())
        active.update(index=len(works), phase="quality")
        progress(len(works), len(works), "quality", completed_chunks=len(completed))
        collect_finished(True)
    if stage_failures and not all(
        (work / "native/native-report.json").exists()
        and (work / "native/chunk1.mp4").exists()
        for work in works
    ):
        raise PipelineError(
            "HLS_BLOCK_FAILED", "部分块缺少完整原生产物，检查点已保留", review=True
        )
    times.sort(key=lambda t: t["index"])
    failures = [
        i + 1
        for i, work in enumerate(works)
        if not (
            json.loads((work / "verified.json").read_text())
            if (work / "verified.json").exists()
            else {}
        ).get("通过")
    ]
    references = []
    combined = {}
    scene_cuts = {0, plans[-1]["end"]}
    for block_index, (work, plan) in enumerate(zip(works, plans)):
        info = json.loads((work / "prepared.json").read_text())
        meta = json.loads((work / "native/native-report.json").read_text())
        from scripts.hls_native_verify import remap_references

        detected = json.loads(Path(info["detected"]).read_text())
        mapping = meta["帧映射"]
        d = remap_references(detected, mapping)
        for scene_start, _ in detected["scenes"]:
            if scene_start:
                positions = [
                    n for n, value in enumerate(mapping) if value >= scene_start
                ]
                if positions:
                    scene_cuts.add(plan["left"] + positions[0])
        for track in d["tracks"]:
            identity = (
                track["id"]
                if track["kind"] == "fixed"
                else 1000 * (block_index + 1) + track["id"]
            )
            target = combined.setdefault(identity, dict(track, id=identity, hints=[]))
            for hint in track["hints"]:
                n = hint["frame"] + plan["left"]
                if plan["start"] <= n < plan["end"]:
                    point = dict(
                        hint,
                        frame=n,
                        track_id=identity,
                        box=[q * 1.5 for q in hint["box"]],
                    )
                    references.append(point)
                    target["hints"].append(point)
        brand_frames = json.loads((Path(info["variant"]) / "tracks.json").read_text())[
            "frames"
        ]
        color_boxes = {}
        for model_frame in set(mapping):
            file = Path(info["color"]) / "masks" / f"{model_frame:06d}.png"
            mask = cv2.imread(str(file), 0) if file.exists() else None
            if mask is not None and mask.any():
                yy, xx = np.where(mask > 0)
                color_boxes[model_frame] = [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max() + 1),
                    int(yy.max() + 1),
                ]
        for n, model_frame in enumerate(mapping):
            global_frame = n + plan["left"]
            if not plan["start"] <= global_frame < plan["end"]:
                continue
            if model_frame in color_boxes:
                identity = 200000 + block_index
                target = combined.setdefault(
                    identity, dict(id=identity, kind="brand_color", hints=[])
                )
                hint = dict(
                    frame=global_frame,
                    track_id=identity,
                    box=[q * 1.5 for q in color_boxes[model_frame]],
                )
                target["hints"].append(hint)
                references.append(hint)
            for slot, point in enumerate(brand_frames[model_frame]["matches"]):
                identity = 100000 + block_index * 100 + slot
                target = combined.setdefault(
                    identity, dict(id=identity, kind="brand", hints=[])
                )
                hint = dict(
                    frame=global_frame,
                    track_id=identity,
                    confidence=point["confidence"],
                    box=[q * 1.5 for q in point["box"]],
                )
                target["hints"].append(hint)
                references.append(hint)
    stage("encode")
    progress(len(works), len(works), "encode")
    cuts = [sum(d for d, _ in segments[:i]) for i in range(1, len(segments))]
    output = root / "result.mp4"
    join_videos(works, plans, output, full_saved, cuts, check)
    if failures or stage_failures:
        # 先交付完整预览；正式发布仍保留原零残留与无闪烁门槛。
        preview_dir = persistent / "v0-preview"
        preview_dir.mkdir(exist_ok=True)
        shutil.copyfile(output, preview_dir / "整集-v0.mp4")
        from scripts.hls_v0 import block_report

        write_json(
            preview_dir / "report.json",
            dict(
                系统="视频去水印系统",
                版本="整集v0",
                流程完成=True,
                通过=False,
                各块=[
                    block_report(work, i + 1, plans[i]) for i, work in enumerate(works)
                ],
                未通过块=failures,
            ),
        )
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                output,
                "-vf",
                "scale=-2:720,fps=30",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-threads",
                "4",
                "-c:a",
                "copy",
                preview_dir / "整集-v0-预览.mp4",
            ],
            check,
            timeout=1800,
        )
        raise PipelineError(
            "RESIDUAL_UNCERTAIN",
            "整集v0已生成，部分块待复核；未作为正式成品发布",
            review=True,
        )
    shutil.copyfile(assets["prior"], root / "multitrack-prior.npz")
    cuts_sorted = sorted(scene_cuts)
    write_json(
        root / "detections.json",
        dict(
            hls=True,
            multitrack=True,
            scenes=list(zip(cuts_sorted, cuts_sorted[1:])),
            hints=references,
            tracks=list(combined.values()),
            prior=str(root / "multitrack-prior.npz"),
        ),
    )
    # 对实际交付编码再检查一次；HLS 后续只做无损封装。
    from .hls_final_audit import audit as final_audit

    final_references = json.loads((root / "detections.json").read_text())
    final_references["prior"] = assets["prior"]
    shutil.copyfile(output, persistent / "joined-result.mp4")
    final_check = final_audit(
        full_saved, output, final_references, works, plans, assets, persistent, check
    )
    result = package(
        full_saved,
        output,
        root,
        segments,
        lines,
        source_pts,
        plans,
        60,
        source_fps,
        times,
        digest,
        fingerprint,
        started,
        references,
        check,
    )
    metric_path = root / "metrics-hls.json"
    metrics = json.loads(metric_path.read_text())
    metrics.update(
        dict(
            final_encoded_audit=final_check,
            method="v7",
            source_mp4_sha256=source_digest,
            processing_fps=30,
            output_fps=60,
            clarity_record_only=True,
            clarity_low_threshold=0.7,
            block_audits=[
                dict(
                    index=i + 1,
                    residual=json.loads((work / "audit/quality.json").read_text()),
                    brand=json.loads((work / "audit/brand.json").read_text()),
                    clarity=json.loads((work / "audit/clarity.json").read_text()),
                    verified=json.loads((work / "verified.json").read_text()),
                    opaque_panels=(
                        json.loads((work / "audit/opaque.json").read_text())
                        if (work / "audit/opaque.json").exists()
                        else {}
                    ),
                    low_band=(
                        json.loads((work / "audit/low-band.json").read_text())
                        if (work / "audit/low-band.json").exists()
                        else {}
                    ),
                    graphic_repair=(
                        json.loads((work / "complete-graphics/report.json").read_text())
                        if (work / "complete-graphics/report.json").exists()
                        else {}
                    ),
                    panel_repair=(
                        json.loads((work / "opaque-panels/report.json").read_text())
                        if (work / "opaque-panels/report.json").exists()
                        else {}
                    ),
                    native_clarity=clarity_summary(
                        json.loads((work / "native/native-report.json").read_text())[
                            "清晰度统计"
                        ]
                    ),
                )
                for i, work in enumerate(works)
            ],
        )
    )
    write_json(metric_path, metrics)
    write_json(
        persistent / "delivery.json",
        dict(
            检查点前缀=checkpoint_prefix,
            持久工作目录=str(persistent),
            总流水线耗时秒=time.monotonic() - pipeline_started,
            技术封装完成=True,
        ),
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    args = p.parse_args()
    native_block(args.native, args.assets)


if __name__ == "__main__":
    main()
