"""按硬门槛选择完整候选；清晰度只记录，不参与版本排序。"""
import math


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
