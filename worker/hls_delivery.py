"""按硬门槛选择完整候选；清晰度只记录，不参与版本排序。"""
import math


def verified_stages(quality, video_digest):
    """同摘要已通过复检的完整视频直接复用，不再对干净背景补画。"""
    if not (
        quality.get("视频摘要") == video_digest
        and quality.get("候选摘要")
        and quality.get("通过") is True
        and quality.get("残留") == 0
        and quality.get("低频残留") == 0
    ):
        return {}
    return {
        name: dict(通过=True, 视频摘要=video_digest, 候选摘要=quality["候选摘要"])
        for name in ["低频检查与补修", "不透明面板补修", "编码后复检"]
    }


def best_version(candidates):
    usable = [p for p in candidates if p.get("完整") is True]
    if not usable:
        raise ValueError("没有完整的可交付版本")

    def number(value):
        return value if isinstance(value, (int, float)) and value >= 0 else math.inf

    def rank(item):
        # 已确认闪烁的版本排后；未知不能冒充无闪烁或验收通过。
        return (
            item.get("闪烁") is True,
            number(item.get("残留")),
            number(item.get("低频残留")),
            item.get("闪烁") is not False,
            -item.get("生成时间", 0),
        )

    return min(usable, key=rank)
