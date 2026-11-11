"""视频去水印系统：空闲设备上的单块提速基准与七版质量对照。"""

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
from worker import media, vision
from worker.process import run
from worker.speedup_cache import StageCache, fingerprint

BASELINE = Path("/workspace/dewatermark-data/hls-huangguo/chunk1-v7/chunk1.mp4")
BASELINE_SECONDS = dict(detect=34, sam=356, inpaint=1061, composite=1070, detail=479)


def clarity_summary(samples):
    valid = [
        s
        for s in samples
        if s.get("measurable") and s.get("retention_ratio") is not None
    ]
    return dict(
        全部样本=samples,
        可测样本数=len(valid),
        最低值=min((s["retention_ratio"] for s in valid), default=None),
        低于零点七的样本=[s for s in valid if s["retention_ratio"] < 0.7],
        仅记录不作绝对门槛=True,
    )


def parse_metrics(ssim, psnr):
    s = [float(v) for v in re.findall(r"All:([\d.eE+-]+)", ssim.read_text())]
    errors = [float(v) for v in re.findall(r"mse_avg:([\d.eE+-]+)", psnr.read_text())]
    if not s or len(s) != len(errors):
        raise ValueError("对照指标没有覆盖相同的全部帧")
    mse = sum(errors) / len(errors)
    return dict(
        帧数=len(s),
        SSIM均值=sum(s) / len(s),
        SSIM最低=min(s),
        PSNR分贝=10 * math.log10(255**2 / mse) if mse else None,
        均方误差日志为零=mse == 0,
        均方误差=mse,
    )


def compare(candidate, baseline, output, duration, crops=()):
    """采用七版原帧率对齐所有帧；六十帧的额外相位由原片清晰度检查覆盖。"""
    output.mkdir(parents=True, exist_ok=True)
    a = next(s for s in media.probe(candidate)["streams"] if s["codec_type"] == "video")
    b = next(s for s in media.probe(baseline)["streams"] if s["codec_type"] == "video")
    if (a["width"], a["height"]) != (b["width"], b["height"]):
        raise ValueError("七版与候选成品的分辨率不一致，禁止缩放掩盖差异")
    rate = float(Fraction(b["avg_frame_rate"]))
    values = {}
    for index, crop in enumerate([None, *crops]):
        directory = output / str(index)
        directory.mkdir(exist_ok=True)
        crop_filter = ""
        if crop:
            x, y, w, h = crop
            if (
                x < 0
                or y < 0
                or w <= 0
                or h <= 0
                or x + w > a["width"]
                or y + h > a["height"]
            ):
                raise ValueError("对照裁图范围超出视频")
            crop_filter = f",crop={w}:{h}:{x}:{y}"
        normalize = f"fps={rate}:start_time=0,trim=duration={duration},setpts=PTS-STARTPTS{crop_filter}"
        filters = f"[0:v]{normalize},split[c1][c2];[1:v]{normalize},split[v1][v2];[c1][v1]ssim=stats_file=ssim.log[s];[c2][v2]psnr=stats_file=psnr.log[p]"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                candidate.resolve(),
                "-i",
                baseline.resolve(),
                "-filter_complex",
                filters,
                "-map",
                "[s]",
                "-map",
                "[p]",
                "-an",
                "-f",
                "null",
                "-",
            ],
            cwd=directory,
            timeout=7200,
        )
        metric = parse_metrics(directory / "ssim.log", directory / "psnr.log")
        if abs(metric["帧数"] - round(duration * rate)) > 1:
            raise ValueError("七版对照时长或帧数不足")
        values["全画面" if crop is None else ",".join(map(str, crop))] = metric
    return values


def comparisons(source, candidate, baseline, output, duration, times):
    """并排视频及自动角部截图仅供查看，不参与水印框规划。"""
    output.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            source,
            "-i",
            candidate,
            "-filter_complex",
            "[0:v]fps=60,setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];[a][b]hstack=shortest=1[v]",
            "-map",
            "[v]",
            "-an",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "10",
            output / "source-candidate.mp4",
        ],
        timeout=7200,
    )
    captures = [
        cv2.VideoCapture(str(path))
        for path in (
            (source, baseline, candidate)
            if baseline is not None
            else (source, candidate)
        )
    ]
    images = []
    skipped = []
    try:
        for second in times:
            if not 0 <= second < duration:
                skipped.append(second)
                continue
            frames = []
            for cap in captures:
                cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                ok, image = cap.read()
                if not ok:
                    raise ValueError("对比截图解码失败")
                frames.append(image)
            h, w = frames[0].shape[:2]
            for label, x, y in [("左上", 0, 0), ("右下", w * 2 // 3, h * 2 // 3)]:
                pieces = [f[y : y + h // 3, x : x + w // 3] for f in frames]
                path = output / f"{second:g}-{label}.png"
                cv2.imwrite(str(path), np.hstack(pieces))
                images.append(str(path))
    finally:
        for cap in captures:
            cap.release()
    return dict(
        并排视频=str(output / "source-candidate.mp4"),
        图片=images,
        图片顺序="原片、七版、提速成品" if baseline is not None else "原片、整集成品",
        超出单块时长未生成的时间点=skipped,
    )


def automatic_crops(args):
    if not args.prior or not args.detected:
        return []
    from worker.multitrack import load_prior

    median, _, _, _, _ = load_prior(args.prior)
    mh, mw = median.shape
    detected = json.loads(args.detected.read_text())
    boxes = []
    for track in detected.get("tracks", []):
        hints = [p["box"] for p in track["hints"]]
        if not hints:
            continue
        x = max(0, math.floor(min(b[0] for b in hints) * 1920 / mw) - 12)
        y = max(0, math.floor(min(b[1] for b in hints) * 1080 / mh) - 12)
        a = min(1920, math.ceil(max(b[2] for b in hints) * 1920 / mw) + 12)
        b = min(1080, math.ceil(max(b[3] for b in hints) * 1080 / mh) + 12)
        if a > x and b > y:
            boxes.append((x, y, a - x, b - y))
    if args.graphic:
        union = np.zeros((mh, mw), np.uint8)
        for path in sorted((args.graphic / "masks").glob("*.png")):
            image = cv2.imread(str(path), 0)
            if image is None:
                raise ValueError("品牌支持图像无法读取")
            union |= cv2.resize(image, (mw, mh), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(union > 0)
        if len(xs):
            x, y = (
                max(0, math.floor(xs.min() * 1920 / mw) - 12),
                max(0, math.floor(ys.min() * 1080 / mh) - 12),
            )
            a, b = (
                min(1920, math.ceil((xs.max() + 1) * 1920 / mw) + 12),
                min(1080, math.ceil((ys.max() + 1) * 1080 / mh) + 12),
            )
            boxes.append((x, y, a - x, b - y))
    return sorted(set(boxes))


def prepare_run(args, phases):
    """直接复用整集分块执行器，默认自动准备全片先验与原生素材。"""
    from worker.hls_native import configuration, validate_source
    from worker.native_assets import prepare_assets
    from worker.native_chunk import process_chunk
    from worker.speedup import encoder_args
    from scripts.hls_chunk_check import mux_core_audio

    if os.getenv("DETECTOR") != "yolo" or not os.getenv("MODEL_MANIFEST"):
        raise ValueError("真实基准需要专用检测环境和固定模型清单")
    os.environ["MODEL_LOCK_HASH"] = vision.verify_models(os.environ["MODEL_MANIFEST"])
    info = media.probe(args.source)
    validate_source(info)
    if args.duration + args.context > min(30, float(info["format"]["duration"])):
        raise ValueError("主体加上下文超过三十秒或源片时长")
    from worker.roi_inpaint import digest

    source_hash = digest(args.source)
    config = configuration(
        dict(
            hls_native_path=getattr(args, "path", "speedup-v2"),
            hls_encoder=args.encoder,
        )
    )
    if config["fast"]:
        config["backend"] = args.backend
    assets = prepare_assets(
        args.source, args.output / "native-assets", source_hash, prior=args.prior
    )
    asset_times = args.output / "native-assets/stages/timings.json"
    if asset_times.exists():
        phases.timings.update(
            {
                "assets_" + name: value
                for name, value in json.loads(asset_times.read_text()).items()
            }
        )
    supplied = [
        getattr(args, name, None)
        for name in ("alpha", "template", "graphic", "projection_calibration")
    ]
    if any(supplied):
        if not all(p is not None and p.exists() for p in supplied):
            raise ValueError("复用七版自动素材时须提供完整透明度、模板、图形与投影标定")
        assets.update(
            alpha=args.alpha,
            template=args.template,
            graphic=args.graphic,
            projection=args.projection_calibration,
        )
    work = args.output / "work"
    plan = dict(
        left=0,
        right=round((args.duration + args.context) * 60),
        start=0,
        end=round(args.duration * 60),
    )
    summary = process_chunk(args.source, work, plan, assets, config, source_hash)
    phases.timings.update(summary["stages"])
    args._input_fingerprint = fingerprint(
        list(assets.values()), [source_hash, plan, config["mode"]]
    )
    rendered = args.output / "rendered"
    rendered.mkdir(exist_ok=True)

    def encode():
        silent = rendered / "video.mp4"
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
                "-frames:v",
                str(plan["end"]),
                *encoder_args(config["encoder"], 10),
                "-pix_fmt",
                "yuv420p",
                silent,
            ],
            timeout=7200,
        )
        mux_core_audio(silent, args.source, rendered / "chunk1.mp4", args.duration)

    key = fingerprint(
        [work / "rendered/native-report.json"], [summary, config["encoder"]]
    )
    phases.run("encode", key, encode, lambda: [rendered / "chunk1.mp4"])
    native = dict(summary["native"])
    native["帧映射"] = native["帧映射"][: plan["end"]]
    native["原片帧索引"] = native["原片帧索引"][: plan["end"]]
    native["帧数"] = plan["end"]
    native["主体秒数"] = args.duration
    native["编码器"] = config["encoder"]
    native["最终编码耗时秒"] = phases.timings["encode"]["seconds"]
    (rendered / "native-report.json").write_text(
        json.dumps(native, ensure_ascii=False, indent=2)
    )
    args.candidate, args.native_report = (
        rendered / "chunk1.mp4",
        rendered / "native-report.json",
    )
    args.detected = work / "detections.json"
    args.prior, args.alpha, args.template = (
        assets["prior"],
        assets["alpha"],
        assets["template"],
    )
    args.graphic = work / "graphic"
    args._native_run = dict(work=work, plan=plan, assets=assets)
    return work


def assess(args, phases):
    from scripts.hls_chunk_check import audio_hash

    report = dict(
        系统="视频去水印系统",
        基线分步耗时秒=BASELINE_SECONDS,
        基线总耗时秒=sum(BASELINE_SECONDS.values()),
        分步计时=phases.timings,
        通过=None,
        目检="未进行",
        闪现闪烁="待结合原片镜头切换逐帧检查",
        接缝检查="单块内部窗口见补画指标；整集分片时间轴与接缝待整集基准",
        安全对象前缀=None,
    )
    args._partial_report = report
    verification_started = time.monotonic()
    verification_times = {}
    report["核验分步耗时秒"] = verification_times
    candidate_info = media.probe(args.candidate, count=True)
    v = next(s for s in candidate_info["streams"] if s["codec_type"] == "video")
    fps = float(Fraction(v["avg_frame_rate"]))
    frames = int(v.get("nb_read_frames", 0))
    report["输出规格"] = [v["width"], v["height"], fps]
    report["输出秒数"] = float(v["duration"])
    report["规格通过"] = (
        report["输出规格"] == [1920, 1080, 60]
        and frames == round(args.duration * 60)
        and abs(float(v["duration"]) - args.duration) <= 1 / 60 + 0.001
    )
    source_info = media.probe(args.source)
    report["原片总秒数"] = float(source_info["format"]["duration"])
    report["本块主体秒数"] = args.duration
    report["音画同步检查"] = "压缩音频一致性另有检查；逐包时序与播放器同步复核尚未进行"
    has_audio = any(s["codec_type"] == "audio" for s in source_info["streams"])
    result_audio = any(s["codec_type"] == "audio" for s in candidate_info["streams"])
    report["音频压缩数据一致"] = (
        audio_hash(args.source, args.duration) == audio_hash(args.candidate)
        if has_audio and result_audio
        else has_audio == result_audio
    )
    run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", args.candidate, "-f", "null", "-"],
        timeout=7200,
    )
    report["完整解码检查"] = True
    report["播放器检查"] = "未进行"
    crops = sorted(set([*args.crop, *automatic_crops(args)]))
    report["自动及额外对比区域"] = crops
    measured = time.monotonic()
    report["相对七版"] = compare(
        args.candidate, args.baseline, args.output / "metrics", args.duration, crops
    )
    verification_times["SSIM_PSNR"] = time.monotonic() - measured
    report["数值对齐复核提示"] = dict(
        建议SSIM下限=0.98,
        建议PSNR下限分贝=38,
        待复核区域=[
            name
            for name, metric in report["相对七版"].items()
            if metric["SSIM均值"] < 0.98
            or (metric["PSNR分贝"] is not None and metric["PSNR分贝"] < 38)
        ],
        说明="建议相对七版的数值复核线；不能替代零残留、无闪烁和目检",
    )
    native = json.loads(args.native_report.read_text()) if args.native_report else {}
    for name in ("performance.json", "metrics.sam.json"):
        path = (
            args.output
            / "work"
            / ("tracking/metrics.sam.json" if name == "metrics.sam.json" else name)
        )
        if path.exists():
            report[name] = json.loads(path.read_text())
    report["渲染分步耗时秒"] = native.get("分步耗时秒")
    report["清晰度"] = clarity_summary(native.get("清晰度统计", []))
    audit = args.output / "audit"
    audit.mkdir(exist_ok=True)
    if getattr(args, "_native_run", None):
        from worker.native_chunk import audit_chunk
        from scripts.hls_speedup_quality import relative_clarity

        shared = args._native_run
        measured = time.monotonic()
        quality = audit_chunk(
            args.source,
            args.candidate,
            shared["work"],
            shared["plan"],
            shared["assets"],
        )
        verification_times["残留品牌清晰度与闪烁指标"] = time.monotonic() - measured
        report["清晰度"] = clarity_summary(quality["清晰度统计"])
        report["原始残留检查"] = quality["原始残留检查"]
        report["原始品牌残留候选"] = quality["品牌残留候选"]
        report["全画面模型复检"] = quality.get("全画面模型复检")
        report["闪烁指标"] = quality["闪烁指标"]
        report["品牌图形独立检查"] = "已执行编码后完整部件复检，仍需目检"
        measured = time.monotonic()
        report["相对七版清晰度"] = relative_clarity(args, audit)
        verification_times["相对七版清晰度"] = time.monotonic() - measured
    else:
        if (
            args.native_report
            and args.detected
            and args.prior
            and args.alpha
            and args.graphic
        ):
            from scripts.hls_detail_verify import verify as detail_verify

            detail_verify(
                args.source,
                args.candidate,
                args.native_report,
                args.detected,
                args.prior,
                args.alpha,
                args.graphic,
                audit / "detail.json",
            )
            samples = json.loads((audit / "detail.json").read_text())["全部样本"]
            report["清晰度"] = clarity_summary(samples)
            # 七版只具备三十帧，另以同相位序列比较相对清晰度。
            from scripts.hls_speedup_quality import relative_clarity

            report["相对七版清晰度"] = relative_clarity(args, audit)
        else:
            report["清晰度检查"] = "未完成：缺少映射或自动支持输入"
        if args.native_report and args.detected:
            from scripts.hls_native_verify import verify as native_verify

            native_verify(
                args.source,
                args.candidate,
                args.detected,
                args.native_report,
                audit / "residual",
            )
            report["原始残留检查"] = json.loads(
                (audit / "residual/quality.json").read_text()
            )
            raw = audit / "residual/quality-detections.json"
            report["原始残留候选"] = (
                json.loads(raw.read_text()) if raw.exists() else None
            )
            report["残留复核依据"] = (
                "保留原始候选；仍须短时图形与全画面新增标识复检和人工目检，不自动改判候选"
            )
        else:
            report["原始残留检查"] = None
            report["残留复核依据"] = "未完成：缺少检测与映射"
        report["品牌图形独立检查"] = "待运行品牌图形复检入口，未宣称零残留"
    if args.make_comparisons:
        report["对比产物"] = comparisons(
            args.source,
            args.candidate,
            args.baseline,
            args.output / "comparisons",
            args.duration,
            args.times,
        )
    else:
        report["对比产物"] = "未生成，使用并排输出参数生成"
    report["独立核验耗时秒"] = time.monotonic() - verification_started
    report["提速比较范围"] = (
        "原生块准备至最终编码；独立核验与七版对照耗时另记，不能冒称完整验收耗时"
    )
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run",
        action="store_true",
        help="运行检测、分割、局部补画和合成；仅在设备空闲后使用",
    )
    mode.add_argument(
        "--candidate",
        type=Path,
        help="只评估已有成品；提供检测映射时仍调用检测模型复检，需设备空闲",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=Path("/workspace/dewatermark-data/hls-huangguo/source.mp4"),
        help="已解密的原片",
    )
    p.add_argument("--baseline", type=Path, default=BASELINE, help="已接受的七版成品")
    p.add_argument(
        "--output", type=Path, required=True, help="独立的基准输出和检查点目录"
    )
    for name, label in [
        ("prior", "自动多轨迹先验"),
        ("alpha", "自动透明度标定"),
        ("template", "原生字形模板"),
        ("graphic", "完整品牌覆盖缓存目录"),
        ("projection-calibration", "七版固定字形投影系数"),
        ("native-report", "已有成品的原生帧映射报告"),
        ("detected", "已有多轨迹检测报告"),
    ]:
        p.add_argument("--" + name, type=Path, help=label)
    p.add_argument(
        "--path",
        choices=["speedup-v2", "v7"],
        default="speedup-v2",
        help="共用整集原生路径；七版开关同时恢复整帧补画与软件合成编码",
    )
    p.add_argument("--duration", type=float, default=25, help="主体秒数")
    p.add_argument(
        "--context", type=float, default=2, help="末尾上下文秒数，总输入不得超过三十秒"
    )
    p.add_argument(
        "--backend",
        choices=["numpy", "torch-cpu", "torch-cuda"],
        default="torch-cuda",
        help="原生合成和细节融合后端",
    )
    p.add_argument(
        "--encoder",
        choices=["libx264", "h264_nvenc"],
        default="libx264",
        help="首次质量对齐建议软件编码；显卡编码须另行对照",
    )
    p.add_argument(
        "--sam-prepare-max-edge",
        type=int,
        default=1024,
        help="分割准备图像长边；内部模型尺寸不变",
    )
    p.add_argument(
        "--sam-hint-stride",
        type=int,
        default=30,
        help="关键提示最大间隔，显著运动自动补提示",
    )
    p.add_argument(
        "--roi-margin", type=int, default=96, help="局部补画区域外扩像素，模型尺度不变"
    )
    p.add_argument(
        "--crop",
        type=lambda v: tuple(map(int, v.split(","))),
        action="append",
        default=[],
        help="额外比较区域，格式为横坐标、纵坐标、宽、高；可重复",
    )
    p.add_argument(
        "--make-comparisons",
        action="store_true",
        help="生成原片与成品并排视频及三列角部截图",
    )
    p.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[5, 60, 120, 200],
        help="截图秒数，超出单块的点明确记录为未生成",
    )
    return p


def validate_output(args):
    output = args.output.resolve()
    protected = [args.source, args.baseline]
    if args.run:
        protected += [
            args.prior,
            args.alpha,
            args.template,
            args.graphic,
            args.projection_calibration,
        ]
    for path in protected:
        if path is not None and Path(path).resolve().is_relative_to(output):
            raise ValueError("输出目录不得覆盖或包含基准输入及七版成品")
    for production in (
        Path("/workspace/video-dewatermark"),
        Path("/workspace/video-dewatermark-github"),
    ):
        if output == production or output.is_relative_to(production):
            raise ValueError("输出目录不得位于正在使用的运行或主工作目录")


def main():
    p = parser()
    args = p.parse_args()
    if (
        not 0 < args.duration <= 30
        or not 0 <= args.context
        or args.duration + args.context > 30
    ):
        p.error("主体加上下文必须在三十秒内")
    if any(len(c) != 4 for c in args.crop):
        p.error("比较区域必须有四个整数")
    args.output = args.output.resolve()
    try:
        validate_output(args)
    except ValueError as error:
        p.error(str(error))
    args.output.mkdir(parents=True, exist_ok=True)
    phases = StageCache(args.output / "stages")
    cv2.setNumThreads(2)
    started = time.monotonic()
    report = None
    try:
        if args.run:
            os.environ.update(
                PROPAINTER_ROI="true",
                PROPAINTER_ROI_MARGIN=str(args.roi_margin),
                SAM_SPEEDUP="true",
                SAM_PREPARE_MAX_EDGE=str(args.sam_prepare_max_edge),
                SAM_HINT_STRIDE=str(args.sam_hint_stride),
            )
            processing_started = time.monotonic()
            prepare_run(args, phases)
            args._process_wall_seconds = time.monotonic() - processing_started
        report = assess(args, phases)
    except Exception as error:
        # 不输出异常正文，避免第三方日志包含环境、签名地址或输入内容。
        report = getattr(args, "_partial_report", {})
        report.update(
            系统="视频去水印系统",
            通过=False,
            目检="未进行",
            失败类型=type(error).__name__,
            失败阶段="处理或核验",
            分步计时=phases.timings,
            未完成检查="不得宣称提速或质量达标",
        )
        raise SystemExit("基准未完成，阶段记录已保留，请检查独立输出目录") from None
    finally:
        if report is not None:
            report["本次总耗时秒"] = time.monotonic() - started
            report["分步耗时累计秒"] = sum(
                v["seconds"] for v in phases.timings.values()
            )
            report["本次处理耗时秒"] = getattr(args, "_process_wall_seconds", None)
            report["输入与配置指纹"] = getattr(args, "_input_fingerprint", None)
            report["复用检查点"] = (
                any(v.get("reused") for v in phases.timings.values())
                or report.get("performance.json", {}).get("cached_windows", 0) > 0
            )
            report["可用于冷启动提速结论"] = (
                args.run and not report["复用检查点"] and report.get("失败类型") is None
            )
            report["三倍提速目标达成"] = (
                report["本次处理耗时秒"] < sum(BASELINE_SECONDS.values()) / 3
                if report["可用于冷启动提速结论"]
                else None
            )
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            )
    print("基准与对照记录已保存；残留、闪烁与目检完成前，验收状态保持待核验。")


if __name__ == "__main__":
    main()
