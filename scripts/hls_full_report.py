"""视频去水印系统：汇总完整证据；清晰度只记录，待目检不能写成通过。"""

import argparse, json
from pathlib import Path
from worker.hls_acceptance import clarity_summary, hard_acceptance


def build(root):
    verification = json.loads((root / "audit/verification.json").read_text())
    browser_path = root / "browser.json"
    browser = json.loads(browser_path.read_text()) if browser_path.exists() else {}
    visual_path = root / "目检.json"
    visual = (
        json.loads(visual_path.read_text())
        if visual_path.exists()
        else {"通过": False, "状态": "待逐块目检"}
    )
    samples = []
    for block in verification.get("block_audits", []):
        index = block["index"]
        plan = verification["chunks"][index - 1]
        for label, values in [
            ("编码后成品", block["clarity"]["全部样本"]),
            (
                "v7基底合成（补修前）",
                block.get("native_clarity", {}).get("全部样本", []),
            ),
        ]:
            for sample in values:
                samples.append(
                    dict(
                        sample,
                        块号=index,
                        测量阶段=label,
                        全局时间=plan["left"] / 60 + sample["time"],
                    )
                )
    clarity = clarity_summary(samples)
    final = verification.get("final_encoded_audit", {})
    residual = 0 if final.get("通过") is True else None
    technical = (
        verification.get("hls_mp4_decoded_frames_equal") is True
        and verification.get("audio_copy_verified") is True
        and all(
            browser.get(k) is True
            for k in ("hls_playback", "seek", "mp4_download", "zip_download")
        )
    )
    passed = hard_acceptance(
        residual, visual.get("无闪烁"), technical, visual.get("通过")
    )
    report = dict(
        系统="视频去水印系统",
        通过=passed,
        目检=visual,
        方案="已接受的v7，1920×1080、60帧，清晰度只记录",
        残留=residual,
        清晰度=clarity,
        各块=verification["chunks"],
        总执行耗时秒=verification["elapsed_ms"] / 1000,
        原片时长秒=verification["source_duration_ms"] / 1000,
        成品时长秒=verification["result_duration_ms"] / 1000,
        帧数=verification["hls_frame_count"],
        音频原样复制=verification["audio_copy_verified"],
        音频时间戳最大误差毫秒=verification["audio_timeline_max_error_ms"],
        音画起点偏差毫秒=verification["av_start_offset_ms"],
        音画终点偏差毫秒=verification["av_end_offset_ms"],
        接缝=verification["seams"],
        运动补偿接缝=verification["motion_compensated_seams"],
        最终编码复检=final,
        浏览器=browser,
        说明="各块阶段耗时可重叠，不能把块耗时之和当作整集墙时；低于0.7的清晰度样本单列，不否决验收。",
    )
    timing = root / "timing.json"
    if timing.exists():
        timing_data = json.loads(timing.read_text())
        report["任务耗时"] = timing_data
        report["本次交付尝试耗时秒"] = report["总执行耗时秒"]
        report["总执行耗时秒"] = timing_data.get("wall_seconds", report["总执行耗时秒"])
    history = root / "任务历史.json"
    if history.exists():
        report["任务历史"] = json.loads(history.read_text())
    report["逐块完整证据"] = verification.get("block_audits", [])
    detection = []
    for name in ["源片逐秒检测统计.json", "源片逐秒增强检测统计.json"]:
        path = root / name
        if path.exists():
            detection.append(dict(证据文件=name, **json.loads(path.read_text())))
    report["源片检测统计"] = detection
    report["检测统计说明"] = (
        "候选数含字幕及误报，不能作为真实水印个数或召回率；模型曾漏检相邻播放图形和部分淡色笔画，处理结合自动共识、完整模板与独立跟踪补齐。最终零残留依据编码后复检与目检，不声称模型单独召回率100%。"
    )
    prefix = root / "结果前缀.json"
    if prefix.exists():
        report["结果R2前缀"] = json.loads(prefix.read_text())
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (root / "清晰度低于0.7样本.json").write_text(
        json.dumps(clarity["低于0.7样本"], ensure_ascii=False, indent=2)
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = build(a.output)
    print(
        "整集验收汇总，是否通过：",
        result["通过"],
        "，清晰度仅记录，低于0.7样本数：",
        result["清晰度"]["低于0.7样本数"],
    )


if __name__ == "__main__":
    main()
