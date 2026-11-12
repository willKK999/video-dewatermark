"""在其他磁盘续做已编码整集的技术检查和报告，不改变逐块验收结论。"""

import argparse
import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from scripts.hls_chunk_check import audio_hash
from scripts.hls_v0_evidence import enrich
from worker.media import probe
from worker.vision import sha256


def technical_check(source, result, preview, frames, audio_equal):
    video = lambda info: next(s for s in info["streams"] if s["codec_type"] == "video")
    a, b, c = map(video, (source, result, preview))
    return (
        (b["width"], b["height"]) == (a["width"], a["height"])
        and Fraction(b["avg_frame_rate"]) == Fraction(a["avg_frame_rate"])
        and int(b["nb_frames"]) == frames
        and abs(float(b["duration"]) - float(a["duration"])) <= 1 / float(Fraction(a["avg_frame_rate"]))
        and c["height"] == 720
        and Fraction(c["avg_frame_rate"]) == 30
        and abs(float(c["duration"]) - float(a["duration"])) <= 1 / 30
        and audio_equal
    )


def finalize(root, source, output, timing=None):
    data = json.loads((root / "run.json").read_text())
    if data["源摘要"] != sha256(source):
        raise ValueError("冻结源片与检查点摘要不一致")
    result, preview = [output / name for name in ("整集-v0.mp4", "整集-v0-预览.mp4")]
    infos = [probe(p) for p in (source, result, preview)]
    audio = len({audio_hash(p) for p in (source, result, preview)}) == 1
    if not technical_check(*infos, data["计划"][-1]["end"], audio):
        raise ValueError("视频规格、帧数、时长或音频复制检查失败")
    rows = []
    for index in range(1, len(data["计划"]) + 1):
        row = json.loads((root / f"chunk-{index:02d}/v0-block.json").read_text())
        if row["块号"] != index or row["计划"] != data["计划"][index - 1]:
            raise ValueError("单块报告与整集时间轴不一致")
        rows.append(row)
    now = datetime.now(timezone.utc)
    timing = timing or {}
    submitted = timing.get("任务提交时间")
    total = (now - datetime.fromisoformat(submitted)).total_seconds() if submitted else None
    sv, rv = [next(s for s in info["streams"] if s["codec_type"] == "video") for info in infos[:2]]
    report = dict(
        系统="视频去水印系统", 版本="整集v0", 阶段="第一阶段单轮预览",
        流程完成=True, 通过=False, 目检=dict(通过=False, 状态="待整集检查"),
        各块=rows, 未通过块=[r["块号"] for r in rows if not r["是否通过"]],
        视频规格=[rv["width"], rv["height"], float(Fraction(rv["avg_frame_rate"]))],
        原片时长秒=float(sv["duration"]), 成品时长秒=float(rv["duration"]),
        帧数=data["计划"][-1]["end"], 音频原样复制=audio, 技术检查通过=True,
        提交至交付墙时秒=total, 墙时说明="包含历史等待、诊断和检查点续跑，非纯推理耗时；各块阶段耗时可以重叠。",
        原任务时间=timing, 完成时间=now.isoformat(),
        成品摘要=sha256(result), 预览摘要=sha256(preview),
        接缝指标=None, 播放检查="待浏览器复验", 音画同步="音频压缩数据一致；主观同步待目检",
        结果R2前缀=None, 说明="v0未正式验收，不发布到正式结果前缀；GPU磁盘满后在本地无损封装，未重跑去水印。",
    )
    path = output / "report.json"
    if path.exists():
        backup = output / "report-封装前.json"
        if not backup.exists():
            backup.write_bytes(path.read_bytes())
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    enrich(root, output)
    with (output / "progress.log").open("a") as stream:
        stream.write(now.isoformat() + " 整集 v0 完成；技术规格及音频复制通过；未通过块=" + str(report["未通过块"]) + "\n")
    (output / "v0-local-ready.json").write_text(json.dumps(dict(完成时间=now.isoformat(), 成品摘要=report["成品摘要"], 预览摘要=report["预览摘要"]), ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work-root", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timing", type=Path)
    args = parser.parse_args()
    report = finalize(args.work_root, args.source, args.output, json.loads(args.timing.read_text()) if args.timing else None)
    print("整集v0技术检查及报告完成；未通过块：", report["未通过块"])


if __name__ == "__main__":
    main()
