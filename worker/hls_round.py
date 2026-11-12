"""单轮分块流程：失败保留产物、继续后续阶段，验收状态如实记录。"""

import json, time, os
from pathlib import Path
from .vision import sha256


def execute_round(work, stages, video, prior_attempts=None):
    work = Path(work)
    video = Path(video)
    path = work / "round.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            saved.get("完成")
            and video.exists()
            and saved.get("视频摘要") == sha256(video)
        ):
            return saved
    report = dict(版本="单轮不阻塞-1", 阶段=[], 完成=False, 未通过原因=[])
    prior_attempts = prior_attempts or {}
    for name, callback in stages:
        started = time.monotonic()
        row = dict(名称=name)
        try:
            previous = prior_attempts.get(name)
            if previous:
                row.update(状态="复用既有尝试", 既有证据=previous)
                if previous.get("通过") is not True:
                    report["未通过原因"].append(
                        name + "：既有尝试未通过或中止，本轮不重复补修"
                    )
            else:
                if video.exists():
                    snapshot = work / "round-inputs"
                    snapshot.mkdir(exist_ok=True)
                    copy = snapshot / (name + ".mp4")
                    if not copy.exists():
                        os.link(video, copy)
                callback()
                row["状态"] = "完成"
        except Exception as exc:
            row.update(状态="失败", 错误码=getattr(exc, "code", type(exc).__name__))
            report["未通过原因"].append(name + "：" + row["错误码"])
        row["本轮耗时秒"] = time.monotonic() - started
        report["阶段"].append(row)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    report.update(
        完成=True,
        产物可用=video.is_file() and video.stat().st_size > 0,
        视频摘要=sha256(video) if video.exists() else None,
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def legacy_attempts(work):
    """旧阶段证据只是检查点迁移；不含影片或块号规则。"""
    work = Path(work)
    result = {}
    for name, folder in [
        ("低频检查与补修", "complete-graphics"),
        ("不透明面板补修", "opaque-panels"),
    ]:
        root = work / folder
        if not root.exists() or not any(root.iterdir()):
            continue
        report = root / "report.json"
        data = json.loads(report.read_text()) if report.exists() else {}
        passed = None
        if name == "低频检查与补修":
            checks = data.get("修复后低频复检", data.get("低频复检", {}))
            passed = checks.get("残留区域数") == 0 if checks else None
        elif data:
            passed = True
        result[name] = dict(证据目录=str(root), 通过=passed, 已开始=True)
    return result
