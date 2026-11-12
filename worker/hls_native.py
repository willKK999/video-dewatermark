"""整集默认原生六十帧交付；提速与七版共用边界、质量检查和最终封装。"""

from fractions import Fraction
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from . import media
from .native_assets import prepare_assets
from .native_chunk import process_chunk, decode_chunk, audit_chunk
from .native_checkpoint import BundleStore
from .process import PipelineError, run
from .roi_inpaint import digest
from .speedup import encoder_args


def configuration(options):
    mode = options.get("hls_native_path") or os.getenv("HLS_NATIVE_PATH", "speedup-v2")
    if mode not in ("speedup-v2", "v7"):
        raise PipelineError("INVALID_ARGUMENT", "原生路径只支持提速第二版或七版")
    backend = (
        "numpy" if mode == "v7" else os.getenv("HLS_COMPOSITE_BACKEND", "torch-cuda")
    )
    encoder = options.get("hls_encoder") or os.getenv("HLS_VIDEO_ENCODER", "libx264")
    if backend not in ("numpy", "torch-cpu", "torch-cuda"):
        raise PipelineError("INVALID_ARGUMENT", "原生合成后端无效")
    # 一键回退同时恢复软件编码，不受提速环境开关污染。
    if mode == "v7":
        encoder = "libx264"
    encoder_args(encoder, 10)
    return dict(mode=mode, fast=mode == "speedup-v2", backend=backend, encoder=encoder)


def validate_source(info):
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    if (video["width"], video["height"]) != (1920, 1080) or abs(
        float(Fraction(video["avg_frame_rate"])) - 60
    ) > 0.001:
        raise PipelineError(
            "HLS_UNSUPPORTED",
            "原生交付要求一九二零乘一零八零、六十帧源片，禁止放大凑规格",
        )
    if any(
        s["codec_type"] == "audio" and s["codec_name"] != "aac" for s in info["streams"]
    ):
        raise PipelineError("HLS_UNSUPPORTED", "原生交付音频仅支持直接复制原有压缩流")


def identity(source_hash, options, config, execution_fingerprint):
    # 编码器仅影响最终编码，切换编码器不得重跑模型及原生帧。
    settings = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(
            (
                "BOX_",
                "MASK_",
                "SHOT_",
                "DETECTION_",
                "YOLO_",
                "RESIDUAL_",
                "REFERENCE_",
                "RESTORE_",
                "SAM_",
                "PROPAINTER_",
            )
        )
        or k == "MODEL_LOCK_HASH"
    }
    code = {path.name: digest(path) for path in Path(__file__).parent.glob("*.py")}
    code.update(
        {
            "scripts/" + path.name: digest(path)
            for path in Path(__file__)
            .parent.parent.joinpath("scripts")
            .glob("hls_*.py")
        }
    )
    return hashlib.sha256(
        json.dumps(
            dict(
                source=source_hash,
                options={k: v for k, v in options.items() if k != "hls_encoder"},
                mode=config["mode"],
                backend=config["backend"],
                settings=settings,
                asset_recipe="v7-source-rebuild-1",
                execution=execution_fingerprint,
                code=code,
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def assets_from(root):
    assets = {
        name: root / file
        for name, file in dict(
            prior="multitrack-prior.npz",
            alpha="alpha.npz",
            template="template.npz",
            palette="palette.npy",
        ).items()
    }
    if (root / "projection.json").is_file():
        assets["projection"] = root / "projection.json"
    if (root / "low-band.npz").is_file():
        assets["low_band_template"] = root / "low-band.npz"
    for name in (
        "color_seed_image",
        "color_seed_mask",
        "variant_seed_image",
        "variant_seed_mask",
    ):
        prefix, _, kind = name.partition("_seed_")
        file = root / f"{prefix}-seed-{kind}.png"
        if file.is_file():
            assets[name] = file
    return assets


def cache_chunk(work, store, name, key, check):
    movie = work / "native.mkv"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            "60",
            "-i",
            work / "rendered/frames/%06d.png",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            movie,
        ],
        check,
        timeout=7200,
    )
    paths = [
        movie,
        work / "chunk.json",
        work / "detections.json",
        work / "masks",
        work / "graphic",
        work / "rendered/native-report.json",
        work / "performance.json",
        work / "projection.json",
    ]
    if (work / "v7-assets.json").is_file():
        paths = canonical_checkpoint_files(work, movie, check)
    store.save(
        name,
        work,
        paths,
        key,
    )
    movie.unlink()


def restore_chunk(work, store, name, key, source, plan, assets, check):
    if not store.restore(name, work, key):
        return False
    canonical = (work / "v7-assets.json").is_file()
    if canonical:
        restore_canonical_inputs(work, assets, check)
    else:
        decode_chunk(source, work, plan, check)
    (work / "rendered/frames").mkdir(exist_ok=True)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            work / "native.mkv",
            "-start_number",
            "0",
            work / "rendered/frames/%06d.png",
        ],
        check,
        timeout=7200,
    )
    frames = sorted((work / "rendered/frames").glob("*.png"))
    if len(frames) != plan["right"] - plan["left"]:
        raise PipelineError("OUTPUT_INVALID", "原生检查点帧数不一致")
    for path in (frames[0], frames[-1]):
        if cv2.imread(str(path)).shape != (1080, 1920, 3):
            raise PipelineError("OUTPUT_INVALID", "原生检查点尺寸不一致")
    detections = json.loads((work / "detections.json").read_text())
    detections["prior"] = str(assets["prior"])
    (work / "detections.json").write_text(json.dumps(detections))
    if canonical:
        for path in work.rglob("layers.json"):
            data = json.loads(path.read_text())
            data["detected_sha256"] = digest(work / "detections.json")
            path.write_text(json.dumps(data))
    (work / "native.mkv").unlink()
    return True


def overlap_metrics(joined, incoming, plan, previous_right, check):
    maximum = 0.0
    count = 0
    for n in range(plan["left"], min(previous_right, plan["right"])):
        check()
        old = cv2.imread(str(joined / f"{n:08d}.png"))
        new = cv2.imread(str(incoming / f"{n - plan['left']:06d}.png"))
        if old is None or new is None or old.shape != new.shape:
            raise PipelineError("OUTPUT_INVALID", "原生重叠帧缺失或尺寸不符")
        active = np.any(old != new, axis=2)
        error = (
            float(np.abs(old.astype(np.float32) - new)[active].mean())
            if active.any()
            else 0.0
        )
        maximum = max(maximum, error)
        count += 1
    if maximum > 35:
        raise PipelineError(
            "SEAM_UNCERTAIN", "原生重叠差异超过原有三十五门槛", review=True
        )
    return dict(samples=count, max_mae=maximum, threshold=35)


def execute_native_hls(
    full,
    info,
    lines,
    segments,
    source_hash,
    source_pts,
    root,
    options,
    stage,
    check,
    s3,
    bucket,
    prefix,
    progress,
    execution_fingerprint,
    started,
):
    from .hls import chunks, merge_frames, deliver_hls

    validate_source(info)
    durations = [d for d, _ in segments]
    if abs(float(info["format"]["duration"]) - sum(durations)) > 0.1:
        raise PipelineError("HLS_UNSUPPORTED", "原生源片与原分片时长不一致")
    if os.getenv("DETECTOR") != "yolo":
        raise PipelineError("MODEL_INVALID", "整集原生交付需要固定的专用检测模型")
    config = configuration(options)
    key = identity(source_hash, options, config, execution_fingerprint)
    asset_root = root / "native-assets"
    if asset_root.exists() and any(asset_root.iterdir()):
        asset_root = root / f"native-assets-rebuilt-{time.time_ns()}"
    asset_root.mkdir(exist_ok=True)
    stage("detect")
    # 公共输入每次由原片重建，不从旧整集或对象存储恢复本集素材。
    assets = prepare_assets(full, asset_root, source_hash, check)
    key = hashlib.sha256(
        json.dumps(
            [key, {name: digest(path) for name, path in sorted(assets.items())}],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    store = BundleStore(s3, bucket, prefix + "/native-v2/" + key, check)
    plans = chunks(durations, 60)
    frames = root / "joined"
    frames.mkdir()
    times, references = [], []
    source_cuts = set()
    previous_right = 0
    for index, plan in enumerate(plans):
        check()
        progress(index + 1, len(plans), "probe")
        work = root / f"native-chunk-{index:04d}"
        work.mkdir()
        name = f"chunk-{index:04d}"
        chunk_key = hashlib.sha256(json.dumps([key, plan]).encode()).hexdigest()
        begun = time.monotonic()
        reused = restore_chunk(work, store, name, chunk_key, full, plan, assets, check)
        if reused:
            summary = json.loads((work / "chunk.json").read_text())
        else:
            try:
                summary = process_chunk(
                    full,
                    work,
                    plan,
                    assets,
                    config,
                    source_hash,
                    check,
                    lambda step: (stage(step), progress(index + 1, len(plans), step)),
                )
                cache_chunk(work, store, name, chunk_key, check)
            except BaseException:
                phase = work / "stages/timings.json"
                failed = dict(
                    index=index + 1,
                    elapsed_ms=round((time.monotonic() - begun) * 1000),
                    phase=json.loads(phase.read_text()) if phase.exists() else {},
                )
                (root / "metrics-hls.json").write_text(
                    json.dumps(
                        dict(
                            chunks=times,
                            failed_chunk=failed,
                            completed_chunks=index,
                            total_chunks=len(plans),
                            checkpoint_fingerprint=key,
                        ),
                        ensure_ascii=False,
                    )
                )
                raise
        incoming = work / "rendered/frames"
        overlap = overlap_metrics(frames, incoming, plan, previous_right, check)
        merge_frames(frames, incoming, plan, previous_right, 60, check)
        previous_right = plan["right"]
        native = summary["native"]
        mapping = native["帧映射"]
        lo, hi = plan["start"] - plan["left"], plan["end"] - plan["left"]
        # 供接缝检查使用原生坐标与原生帧号，不混用模型帧率。
        detection = json.loads((work / "detections.json").read_text())
        for cut, _ in detection["scenes"]:
            absolute = plan["left"] + cut * 2
            if cut and plan["start"] <= absolute < plan["end"]:
                source_cuts.add(absolute)
        by_frame = {}
        for hint in detection["hints"]:
            by_frame.setdefault(hint["frame"], []).append(hint)
        for local in range(lo, hi):
            for hint in by_frame.get(mapping[local], []):
                references.append(
                    dict(
                        hint,
                        frame=plan["left"] + local,
                        box=[value * 1.5 for value in hint["box"]],
                    )
                )
        record = dict(
            index=index + 1,
            elapsed_ms=round((time.monotonic() - begun) * 1000),
            reused=reused,
            phase_ms={
                n: round(v["seconds"] * 1000) for n, v in summary["stages"].items()
            },
            overlap=overlap,
            residual_count=None,
            clarity_min=native["清晰度最小保留比"],
            **plan,
        )
        times.append(record)
        (root / "metrics-hls.json").write_text(
            json.dumps(
                dict(
                    chunks=times,
                    completed_chunks=index + 1,
                    total_chunks=len(plans),
                    checkpoint_fingerprint=key,
                ),
                ensure_ascii=False,
            )
        )
        with (root / "progress.log").open("a") as log:
            log.write(
                f"块号={index + 1} 耗时毫秒={record['elapsed_ms']} 残留=待编码后核验 清晰度最低值={record['clarity_min']}\n"
            )
        shutil.rmtree(incoming)
        for folder in ("native-reference", "composite", "roi-cache"):
            if (work / folder).exists():
                shutil.rmtree(work / folder)
    # 发布原生帧号/坐标的完整全局轨迹，供既有离线导出与目检工具复用。
    from .multitrack import combine_tracks

    prior_output = root / "multitrack-prior.npz"
    shutil.copyfile(assets["prior"], prior_output)
    cuts = [0, *sorted(source_cuts), plans[-1]["end"]]
    detected = combine_tracks(
        dict(
            hls=True,
            scenes=[[a, b] for a, b in zip(cuts, cuts[1:])],
            hints=sorted(references, key=lambda p: (p["frame"], p["track_id"])),
        ),
        prior_output,
    )
    (root / "detections.json").write_text(json.dumps(detected))
    native = dict(
        mode=config["mode"],
        encoder_args=encoder_args(config["encoder"], 10),
        config=config,
        assets=assets,
        samples=[],
        evidence=[],
        flicker=[],
        chunks=times,
        started=started,
    )

    def verify_encoded(output):
        for index, plan in enumerate(plans):
            progress(index + 1, len(plans), "quality")
            work = root / f"native-chunk-{index:04d}"
            try:
                summary = audit_chunk(full, output, work, plan, assets, check)
            finally:
                # 失败也提交原始证据，防止临时工作目录被回收后丢失复核材料。
                evidence_paths = sorted((work / "audit").glob("*.json"))
                if evidence_paths:
                    store.save(
                        f"audit-{index:04d}",
                        work,
                        evidence_paths,
                        hashlib.sha256(
                            json.dumps([key, config["encoder"], plan]).encode()
                        ).hexdigest(),
                    )
            native["samples"].extend(summary["清晰度统计"])
            native["evidence"].append(
                dict(
                    模板检查=summary["原始残留检查"],
                    品牌残留候选=summary["品牌残留候选"],
                    全画面模型复检=summary.get("全画面模型复检"),
                    原始证据目录=f"native-chunk-{index:04d}/audit/",
                )
            )
            native["flicker"].extend(summary["闪烁指标"])
            times[index]["residual_count"] = summary["原始残留检查"][
                "residual_count"
            ] + len(summary["品牌残留候选"])
            with (root / "progress.log").open("a") as log:
                log.write(
                    f"块号={index + 1} 编码后残留={times[index]['residual_count']} 清晰度最低值={min((s['retention_ratio'] for s in summary['清晰度统计'] if s['measurable']), default=None)}\n"
                )

    # 原有最终编码、分片时间轴、音频包与接缝校验共同复用。
    return deliver_hls(
        full,
        info,
        lines,
        segments,
        plans,
        60,
        frames,
        references,
        times,
        source_hash,
        key,
        source_pts,
        root,
        check,
        stage,
        progress,
        started,
        native,
        verify_encoded,
    )


def finish_report(root, output, delivery, info, metrics, native, check):
    """由新分片无损合并成品，导出对比；待目检项不自动写为通过。"""
    from scripts.hls_speedup_bench import comparisons

    merged = root / "merged-from-hls.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-copyts",
            "-i",
            delivery / "index.m3u8",
            "-map",
            "0",
            "-c",
            "copy",
            "-output_ts_offset",
            str(-metrics["source_start_pts"]),
            "-avoid_negative_ts",
            "disabled",
            "-movflags",
            "+faststart",
            merged,
        ],
        check,
        timeout=7200,
    )
    a, b = (
        media.probe(output, check, count=True),
        media.probe(merged, check, count=True),
    )
    videos = [
        next(s for s in x["streams"] if s["codec_type"] == "video") for x in (a, b)
    ]
    if (
        any(
            (v["width"], v["height"]) != (1920, 1080)
            or abs(float(Fraction(v["avg_frame_rate"])) - 60) > 0.001
            for v in videos
        )
        or videos[0]["nb_read_frames"] != videos[1]["nb_read_frames"]
    ):
        raise PipelineError("OUTPUT_INVALID", "新分片合并成品规格或帧数不一致")
    duration = float(videos[0]["duration"])
    from scripts.hls_chunk_check import audio_hash

    merged_audio = not any(
        s["codec_type"] == "audio" for s in info["streams"]
    ) or audio_hash(root / "source.mp4") == audio_hash(merged)
    if not merged_audio:
        raise PipelineError("OUTPUT_INVALID", "新分片无损合并后的音频压缩数据不一致")
    if any(s["codec_type"] == "audio" for s in info["streams"]):
        from .hls import audio_timeline_metrics

        audio_timeline_metrics(
            root / "source.mp4",
            merged,
            delivery / "index.m3u8",
            metrics["source_start_pts"],
            check,
        )
    if abs(float(videos[1]["duration"]) - duration) > 1 / 60 + 0.001:
        raise PipelineError("OUTPUT_INVALID", "新分片无损合并后的时长不一致")
    # 主下载成品也必须来自新分片无损合并，不只生成旁支验收文件。
    shutil.copyfile(merged, output)
    samples = native["samples"]
    # 整集生成原片与成品两列；真实七版对照由单块基准单独执行。
    captures = comparisons(
        root / "source.mp4",
        output,
        None,
        root / "comparisons",
        duration,
        [5, 60, 120, 200],
    )
    captures["图片顺序"] = "原片、整集成品"
    captures["并排视频"] = Path(captures["并排视频"]).relative_to(root).as_posix()
    captures["图片"] = [Path(p).relative_to(root).as_posix() for p in captures["图片"]]
    metrics["elapsed_ms"] = round((time.monotonic() - native["started"]) * 1000)
    report = dict(
        系统="视频去水印系统",
        原生路径=native["mode"],
        输出规格=[1920, 1080, 60],
        原片时长秒=float(info["format"]["duration"]),
        最终编码器=native["config"]["encoder"],
        最终编码耗时秒=native.get("encode_seconds"),
        时长秒=duration,
        各块=native["chunks"],
        总耗时毫秒=metrics["elapsed_ms"],
        所有清晰度样本=samples,
        清晰度最低值=min(
            (s["retention_ratio"] for s in samples if s["measurable"]), default=None
        ),
        低于零点七的样本=[
            s for s in samples if s["measurable"] and s["retention_ratio"] < 0.7
        ],
        清晰度仅记录=True,
        原始残留检查=native["evidence"],
        残留复核依据="完整轨迹时域共识与品牌部件阈值不变；原始候选另存各块核验目录",
        接缝指标=metrics["seams"],
        分片时间轴=metrics["segment_timing"],
        音频压缩数据一致=metrics["audio_copy_verified"],
        音频逐包时序={
            k: metrics[k]
            for k in (
                "audio_timeline_max_error_ms",
                "av_start_offset_ms",
                "av_end_offset_ms",
            )
            if k in metrics
        },
        无损合并音频压缩数据一致=merged_audio,
        播放检查="新播放列表已完整解码；交互播放器检查未进行",
        闪烁指标=native["flicker"],
        无闪烁目检="未进行",
        目检="未进行",
        通过=None,
        安全对象前缀=None,
        新分片无损合并成品=str(merged.name),
        主下载成品来源="新播放列表分片无损合并",
        对比产物=captures,
    )
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def canonical_checkpoint_files(work, movie, check):
    """保存七版真正使用的模型输入与图形证据，恢复时不重新采样或改变相位。"""
    from .native_checkpoint import allowed

    model = work / "model-input.mkv"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            "30",
            "-i",
            work / "frames/%06d.png",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "bgr0",
            model,
        ],
        check,
        timeout=7200,
    )
    source = work / "source-input.mkv"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            work / "source.mp4",
            "-map",
            "0",
            "-c",
            "copy",
            source,
        ],
        check,
        timeout=7200,
    )
    paths = [movie, model, source]
    metadata = (
        "chunk.json",
        "detections.json",
        "prepared.json",
        "v7-assets.json",
        "decision-provenance.json",
        "performance.json",
        "native/native-report.json",
        "rendered/native-report.json",
    )
    paths += [work / name for name in metadata if (work / name).is_file()]
    for folder in (
        "color",
        "variant",
        "graphics",
        "masks",
        "fine",
        "coarse",
        "opaque-panels",
        "brand-ink-repair",
        "complete-graphics",
    ):
        root = work / folder
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and allowed(path.relative_to(work).as_posix())
            ):
                if (
                    path.suffix == ".json"
                    or "masks" in path.parts
                    or path.name in ("seed-image.png", "seed-mask.png")
                ):
                    paths.append(path)
    return paths


def restore_canonical_inputs(work, assets, check):
    """只作路径适配与无损解码；不调用检测、分割、模板匹配或掩膜判定。"""
    info_path = work / "prepared.json"
    old = json.loads(info_path.read_text())
    old_root = str(Path(old["detected"]).parent)

    def rebase(value):
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, str) and (
            value == old_root or value.startswith(old_root + "/")
        ):
            return str(work) + value[len(old_root) :]
        return value

    for file in work.rglob("*.json"):
        if file.is_symlink():
            continue
        file.write_text(
            json.dumps(rebase(json.loads(file.read_text())), ensure_ascii=False)
        )
    frames = work / "frames"
    frames.mkdir(exist_ok=True)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            work / "model-input.mkv",
            "-start_number",
            "0",
            frames / "%06d.png",
        ],
        check,
        timeout=7200,
    )
    coarse = work / "coarse"
    coarse.mkdir(exist_ok=True)
    (coarse / "frames").symlink_to(frames.resolve(), target_is_directory=True)
    if not (work / "graphic").exists():
        (work / "graphic").symlink_to(
            (work / "graphics").resolve(), target_is_directory=True
        )
    info = rebase(old)
    info["source"] = str(work / "source-input.mkv")
    info_path.write_text(json.dumps(info))
    manifest = work / "v7-assets.json"
    data = json.loads(manifest.read_text())
    data.update({name: str(path) for name, path in assets.items()})
    manifest.write_text(json.dumps(data))
