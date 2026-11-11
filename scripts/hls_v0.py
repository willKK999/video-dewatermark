"""通用HLS检查点单轮预览：阶段失败不阻止拼接，预览不代表正式验收通过。"""

import argparse, json, time, threading, subprocess, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from worker.hls_v7 import native_block, join_videos, load_assets
from worker.vision import sha256
from worker.media import probe
from worker.process import run
from scripts.hls_chunk_check import audio_hash


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def block_report(work, index, plan, review=None):
    verified = read(work / "verified.json")
    round_info = read(work / "round.json")
    native = read(work / "native/native-report.json")
    gpu = read(work / "gpu.json")
    prepared = read(work / "prepared.json")
    phases = {
        "准备": prepared.get("prepare_seconds"),
        "基础补画": gpu.get("gpu_seconds"),
        "原生合成": native.get("渲染耗时秒"),
        "编码后核验": verified.get("核验耗时秒"),
    }
    for name, folder in [
        ("低频补修", "complete-graphics"),
        ("不透明面板补修", "opaque-panels"),
    ]:
        phases[name] = read(work / folder / "report.json").get("耗时秒")
    reasons = list(round_info.get("未通过原因", []))
    residual = verified.get("残留")
    low = verified.get("低频残留")
    review = review or {}
    flicker = False if review.get("无闪烁") is True else None
    if residual != 0:
        reasons.append("残留未清零" if residual is not None else "残留检查未完成")
    if low != 0:
        reasons.append("低频残留未清零" if low is not None else "低频复检未完成")
    if flicker is None:
        reasons.append("闪烁及目检尚未完成")
    passed = (
        verified.get("通过") is True and not reasons and review.get("目检通过") is True
    )
    return dict(
        块号=index,
        是否通过=passed,
        自动复检通过=verified.get("通过") is True,
        残留=residual,
        低频残留=low,
        闪烁=flicker,
        清晰度最低值=verified.get("清晰度最低值"),
        低于0点7数=verified.get("清晰度低于0点7样本数"),
        清晰度仅记录=True,
        耗时秒=sum(v for v in phases.values() if isinstance(v, (float, int))),
        阶段耗时秒=phases,
        未通过原因=reasons,
        当前完整产物=str(work / "native/chunk1.mp4"),
        选择依据="保留最近完整产物；历史检查口径不同，不伪称已证明全局最优",
        单轮=round_info,
        核验=verified,
        计划=plan,
    )


def deliver(root, source, assets_path, output, reviews=None):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    load_assets(assets_path, sha256(source))
    data = read(root / "run.json")
    if data.get("源摘要") != sha256(source):
        raise ValueError("内容检查点与冻结源片摘要不一致")
    plans = data["计划"]
    works = [root / f"chunk-{i+1:02d}" for i in range(len(plans))]
    reviews = reviews or {}
    lock = threading.Lock()
    records = []

    def log(message):
        line = datetime.now(timezone.utc).isoformat() + " " + message
        with lock:
            with (output / "progress.log").open("a") as stream:
                stream.write(line + "\n")
        print(line, flush=True)

    def one(i):
        work = works[i]
        waiting = time.monotonic()
        while not (work / "gpu.json").exists():
            if time.monotonic() - waiting > 14400:
                raise RuntimeError("基础补画检查点等待超时")
            time.sleep(2)
        try:
            native_block(work, assets_path)
        except Exception as exc:
            # 基础阶段异常也不影响其他块；缺少完整视频时最后明确拒绝拼接。
            (work / "v0-error.json").write_text(
                json.dumps(
                    dict(错误码=getattr(exc, "code", type(exc).__name__)),
                    ensure_ascii=False,
                )
            )
        video = work / "native/chunk1.mp4"
        digest = sha256(video) if video.exists() else None
        report = block_report(work, i + 1, plans[i], reviews.get(digest))
        error = read(work / "v0-error.json")
        if error:
            report["未通过原因"].append("原生阶段：" + error["错误码"])
            report["是否通过"] = False
        report["本轮墙时秒"] = time.monotonic() - waiting
        (work / "v0-block.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        with lock:
            records.append(report)
        log(
            f'块号={i+1}/{len(works)} 耗时秒={report["耗时秒"]:.3f} 残留={report["残留"]} 低频残留={report["低频残留"]} 清晰度最低值={report["清晰度最低值"]} 是否通过={report["是否通过"]}'
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, range(len(works))))
    missing = [
        i + 1 for i, w in enumerate(works) if not (w / "native/chunk1.mp4").exists()
    ]
    if missing:
        raise RuntimeError(
            "处理块缺少完整视频，不能拼接：" + ",".join(map(str, missing))
        )
    result = output / "整集-v0.mp4"
    join_videos(works, plans, result, source, [], lambda: None)
    preview = output / "整集-v0-预览.mp4"
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            result,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
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
            "-movflags",
            "+faststart",
            preview,
        ],
        timeout=1800,
    )
    source_video = next(
        s for s in probe(source)["streams"] if s["codec_type"] == "video"
    )
    meta = probe(result)
    video = next(s for s in meta["streams"] if s["codec_type"] == "video")
    audio = audio_hash(source) == audio_hash(result) == audio_hash(preview)
    technical = (
        video["width"] == source_video["width"]
        and video["height"] == source_video["height"]
        and int(video["nb_frames"]) == plans[-1]["end"]
        and audio
    )
    if not technical:
        raise RuntimeError("整集v0技术规格或音频复制检查未通过")
    report = dict(
        系统="视频去水印系统",
        版本="整集v0",
        阶段="第一阶段单轮预览",
        流程完成=True,
        通过=False,
        目检=dict(通过=False, 状态="待整集检查"),
        各块=sorted(records, key=lambda p: p["块号"]),
        视频规格=[
            video["width"],
            video["height"],
            float(Fraction(video["avg_frame_rate"])),
        ],
        原片时长秒=float(source_video["duration"]),
        成品时长秒=float(video["duration"]),
        帧数=int(video["nb_frames"]),
        音频原样复制=audio,
        技术检查通过=technical,
        本轮单次流程及拼接墙时秒=time.monotonic() - started,
        未通过块=[
            p["块号"]
            for p in sorted(records, key=lambda p: p["块号"])
            if not p["是否通过"]
        ],
        说明="v0用于跑通流程和预览；失败或未检查项均保留，不代表正式验收通过。各块阶段耗时可重叠，不能相加冒充整集墙时。",
        成品摘要=sha256(result),
        预览摘要=sha256(preview),
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    log("整集 v0 完成")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["work-root", "source", "assets", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--reviews", type=Path)
    a = p.parse_args()
    deliver(
        a.work_root, a.source, a.assets, a.output, read(a.reviews) if a.reviews else {}
    )


if __name__ == "__main__":
    main()
