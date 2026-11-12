"""视频去水印系统：清晰度只记录，残留及闪烁仍执行硬门槛。"""


def clarity_summary(samples):
    valid = [
        s
        for s in samples
        if s.get("measurable") and s.get("retention_ratio") is not None
    ]
    low = [s for s in valid if s["retention_ratio"] < 0.7]
    return {
        "仅记录": True,
        "不是验收硬门槛": True,
        "单列阈值": 0.7,
        "全部样本": samples,
        "有效样本数": len(valid),
        "最小保留比": min((s["retention_ratio"] for s in valid), default=None),
        "低于0.7样本数": len(low),
        "低于0.7样本": low,
    }


def hard_acceptance(residual_count, flicker_passed, technical_passed, visual_passed):
    return (
        residual_count == 0
        and flicker_passed is True
        and technical_passed is True
        and visual_passed is True
    )
