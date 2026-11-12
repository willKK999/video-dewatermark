"""按目标生命周期安排分割，禁止在首次可靠提示之前传播目标。"""


def groups_for_scene(detections, start, end):
    hints = [p for p in detections["hints"] if start <= p["frame"] < end]
    if not detections.get("multitrack"):
        return [(start, end, hints)]
    by_id = {}
    for hint in hints:
        by_id.setdefault(hint["track_id"], []).append(hint)
    groups = {}
    for observations in by_id.values():
        first = min(p["frame"] for p in observations)
        last = max(p["frame"] for p in observations) + 1
        groups.setdefault((first, last), []).extend(observations)
    return [
        (a, b, sorted(points, key=lambda p: (p["frame"], p["track_id"])))
        for (a, b), points in sorted(groups.items())
    ]


def sampled_hints(hints, start, end, stride=30):
    first = {}
    for point in hints:
        first.setdefault(point.get("track_id", 1), point["frame"])
    return [
        p
        for p in hints
        if (p["frame"] - start) % stride == 0
        or p["frame"] == end - 1
        or p["frame"] == first[p.get("track_id", 1)]
    ]


def load_mask_prompt(root, relative, shape):
    from pathlib import Path
    import cv2, numpy as np

    path = Path(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[0] != "prompts"
        or path.suffix != ".png"
    ):
        raise ValueError("自动掩膜提示路径无效")
    mask = cv2.imread(str(Path(root) / path), 0)
    if mask is None or mask.shape != shape or not set(np.unique(mask)) <= {0, 255}:
        raise ValueError("自动掩膜提示尺寸或取值无效")
    return mask > 0


def adaptive_hints(hints, start, end, stride=30, motion_fraction=0.2):
    """每条轨迹保留首尾、最长间隔和显著运动提示，不跳过生命周期。"""
    import math

    if stride < 1 or not 0 < motion_fraction <= 1:
        raise ValueError("关键帧采样参数无效")
    groups = {}
    for point in sorted(hints, key=lambda p: p["frame"]):
        if start <= point["frame"] < end:
            groups.setdefault(point.get("track_id", 1), []).append(point)
    result = []
    for points in groups.values():
        previous = None
        for point in points:
            keep = previous is None or point is points[-1]
            if previous is not None:
                keep |= point["frame"] - previous["frame"] >= stride
                if "box" in point and "box" in previous:
                    x, y, a, b = previous["box"]
                    u, v, c, d = point["box"]
                    diagonal = max(math.hypot(a - x, b - y), 1)
                    motion = math.hypot((u + c - x - a) / 2, (v + d - y - b) / 2)
                    area_ratio = max((c - u) * (d - v), 1) / max((a - x) * (b - y), 1)
                    keep |= (
                        motion > diagonal * motion_fraction
                        or not 0.7 <= area_ratio <= 1.3
                    )
            if keep:
                result.append(point)
                previous = point
    return sorted(result, key=lambda p: (p["frame"], p.get("track_id", 1)))


def adaptive_mask_hints(hints, start, end, load_mask, fast=True):
    """框内形变也需要新提示；只在相对上次提示的像素形状稳定时稀疏传播。"""
    import numpy as np

    ordered = sorted(hints, key=lambda p: p["frame"])
    required = {p["frame"] for p in adaptive_hints(ordered, start, end)}
    previous = None
    last_valid = None
    last_emitted = None
    for point in ordered:
        mask = load_mask(point)
        if mask is None:
            continue
        last_valid = point, mask
        active = mask > 0
        changed = previous is None
        if previous is not None:
            union = np.count_nonzero(previous | active)
            changed = np.count_nonzero(previous & active) / max(1, union) < 0.7
        if not fast or point["frame"] in required or changed:
            yield point, mask
            previous = active
            last_emitted = point["frame"]
    if last_valid is not None and last_valid[0]["frame"] != last_emitted:
        yield last_valid


def segmentation_shape(height, width, max_edge):
    if max_edge < 128:
        raise ValueError("分割准备图像长边不得低于一百二十八像素")
    scale = min(1, max_edge / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))
