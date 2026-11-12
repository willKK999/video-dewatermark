"""直接调用主分支七版完整流程；提速仅替换纹理合成的计算后端。"""

import json
import time
from pathlib import Path
import cv2
from . import hls_v7
from .process import PipelineError, run
from .roi_inpaint import digest

REQUIRED_ASSETS = (
    "prior",
    "alpha",
    "template",
    "palette",
    "projection",
    "color_seed_image",
    "color_seed_mask",
    "variant_seed_image",
    "variant_seed_mask",
)


def run_chunk(
    source,
    work,
    plan,
    assets,
    config,
    source_hash,
    check=lambda: None,
    stage=lambda _: None,
):
    """模型、图形、分层和补修全用七版入口，不启用实验提示或局部裁剪。"""
    missing = [key for key in REQUIRED_ASSETS if key not in assets]
    if missing:
        raise PipelineError(
            "MODEL_INVALID", "七版固定输入不完整，不能用新的图形判定代替"
        )
    if work.exists() and any(work.iterdir()):
        raise PipelineError(
            "OUTPUT_INVALID", "七版等价处理必须使用全新空块目录，不读取旧阶段检查点"
        )
    work.mkdir(parents=True, exist_ok=True)
    stages = work / "stages"
    stages.mkdir(exist_ok=True)
    timings = {}

    class PhaseTimes(dict):
        def __setitem__(self, name, seconds):
            timings[name] = dict(seconds=seconds, reused=False)
            (stages / "timings.json").write_text(json.dumps(timings))
            (stages / (name + ".json")).write_text(json.dumps(dict(method="v7")))
            super().__setitem__(name, seconds)

    detailed = PhaseTimes()

    def perform(name, operation):
        check()
        stage(name)
        start = time.monotonic()
        completed = False
        try:
            result = operation()
            completed = True
            return result
        finally:
            timings[name] = dict(seconds=time.monotonic() - start, reused=False)
            (stages / "timings.json").write_text(json.dumps(timings))
            if completed:
                (stages / (name + ".json")).write_text(json.dumps(dict(method="v7")))

    canonical_assets = {k: str(v) for k, v in assets.items()}
    asset_path = work / "v7-assets.json"
    asset_path.write_text(
        json.dumps(
            dict(
                method="v7",
                source_sha256=source_hash,
                **canonical_assets,
                sha256={k: digest(v) for k, v in assets.items() if Path(v).is_file()},
            )
        )
    )
    # 七版以三十帧规划；解码、帧率相位和参考对齐仍由原入口完成。
    model_plan = {key: value / 2 for key, value in plan.items()}
    info = perform(
        "v7_prepare",
        lambda: hls_v7.prepare_block(
            source, work, model_plan, canonical_assets, check, timings=detailed
        ),
    )
    performance = perform(
        "v7_gpu",
        lambda: hls_v7.gpu_block(
            work, canonical_assets, info, "propainter", stage, check, timings=detailed
        ),
    )
    perform(
        "v7_native",
        lambda: hls_v7.native_block(work, asset_path, backend=config["backend"]),
    )
    (work / "performance.json").write_text(json.dumps(performance))
    rendered = work / "rendered"
    rendered.mkdir(exist_ok=True)
    native = json.loads((work / "native/native-report.json").read_text())
    native["原片帧索引"] = [index + plan["left"] for index in native["原片帧索引"]]
    (rendered / "native-report.json").write_text(json.dumps(native, ensure_ascii=False))
    # 整集封装仍消费原生帧；基准可直接复制同一原生成品，避免重复编码。
    if config.get("materialize_frames", True):

        def decode_native():
            (rendered / "frames").mkdir(exist_ok=True)
            run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    work / "native/chunk1.mp4",
                    "-start_number",
                    "0",
                    rendered / "frames/%06d.png",
                ],
                check,
                timeout=7200,
            )
            if len(list((rendered / "frames").glob("*.png"))) != native["帧数"]:
                raise PipelineError("OUTPUT_INVALID", "七版原生成品与帧映射数量不符")

        perform("native_frame_io", decode_native)
    proof = dict(
        判定来源="主分支七版原函数",
        基准提交="0e4dc7b",
        图形="worker.hls_v7.graphic_masks、worker.colour_tracks.complete",
        分割="worker.sam_stage、worker.sam_schedule.sampled_hints",
        掩膜="worker.hls_v7._gpu_block、worker.hls_layers",
        局部补画=False,
        提示重采样=False,
        颜色判据改写=False,
    )
    (work / "decision-provenance.json").write_text(
        json.dumps(proof, ensure_ascii=False, indent=2)
    )
    # 旧导出工具所用名称仅作路径适配，不重新计算图形或判定。
    if not (work / "graphic").exists():
        (work / "graphic").symlink_to(work / "graphics", target_is_directory=True)
    summary = dict(
        plan=plan,
        mode=config["mode"],
        stages=timings,
        native=native,
        native_video=str(work / "native/chunk1.mp4"),
        判定来源=proof,
    )
    (work / "chunk.json").write_text(json.dumps(summary, ensure_ascii=False))
    return summary


def audit_chunk(source, result, work, plan, assets, check=lambda: None, enforce=True):
    """编码后核验复用七版函数，只适配整集主体帧范围。"""
    import shutil

    qa = work / "final-audit-v7"
    if qa.exists():
        shutil.rmtree(qa)
    qa.mkdir()
    for name in ("color", "variant", "graphics", "opaque-panels", "brand-ink-repair"):
        if (work / name).exists():
            (qa / name).symlink_to((work / name).resolve(), target_is_directory=True)
    (qa / "native").mkdir()
    lo, hi = plan["start"] - plan["left"], plan["end"] - plan["left"]
    report = json.loads((work / "native/native-report.json").read_text())
    report["帧映射"] = report["帧映射"][lo:hi]
    report["原片帧索引"] = report["原片帧索引"][lo:hi]
    report["帧数"] = hi - lo
    report["主体秒数"] = (hi - lo) / 60
    (qa / "native/native-report.json").write_text(
        json.dumps(report, ensure_ascii=False)
    )
    target = qa / "native/chunk1.mp4"
    cap = cv2.VideoCapture(str(result))
    count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if plan["start"] == 0 and count == hi - lo:
        target.symlink_to(Path(result).resolve())
    else:
        target = qa / "native/core.mkv"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                result,
                "-an",
                "-vf",
                f"trim=start_frame={plan['start']}:end_frame={plan['end']},setpts=PTS-STARTPTS",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-f",
                "matroska",
                target,
            ],
            check,
            timeout=7200,
        )
    info = json.loads((work / "prepared.json").read_text())
    canonical_assets = {k: str(v) for k, v in assets.items()}
    hls_v7.verify_block(
        qa, canonical_assets, info, source_frame_offset=lo, result_path=target
    )
    verified = json.loads((qa / "verified.json").read_text())
    raw = json.loads((qa / "audit/quality.json").read_text())
    brand = json.loads((qa / "audit/brand.json").read_text())
    clarity = json.loads((qa / "audit/clarity.json").read_text())
    from scripts.hls_speedup_quality import temporal_candidates

    temporal = temporal_candidates(
        Path(info["source"]),
        target,
        Path(info["detected"]),
        qa / "native/native-report.json",
        (hi - lo) / 60,
        qa / "audit/temporal.json",
        source_frame_offset=lo,
    )
    audit = work / "audit"
    audit.mkdir(exist_ok=True)
    for path in (qa / "audit").glob("*.json"):
        shutil.copyfile(path, audit / path.name)
    quality = dict(
        硬门槛通过=verified["通过"] and not temporal["异常候选"],
        清晰度统计=clarity["全部样本"],
        原始残留检查=raw,
        品牌残留候选=brand["证据"],
        原始候选=verified["原始候选"],
        完整图形低频复检=verified.get("低频复检"),
        闪烁指标=temporal["全体样本"],
        时域核验=temporal,
        七版核验=verified,
    )
    (audit / "native-quality.json").write_text(json.dumps(quality, ensure_ascii=False))
    # 视频子集只供核验，原始输出及其压缩音频不做任何变更。
    quality["核验使用同一成品"] = count == hi - lo and plan["start"] == 0
    if enforce and not quality["硬门槛通过"]:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "七版编码后复检存在原始候选", review=True
        )
    return quality
