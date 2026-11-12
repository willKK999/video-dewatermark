"""从已由模型和品牌文本确认的完整图案学习翻转、缩放变体，独立跟踪图文部件。"""

import cv2
import numpy as np
from .multitrack import highpass


def variants(image, mask, scales):
    feature = highpass(image)
    yy, xx = np.where(mask > 0)
    if not len(xx):
        raise ValueError("自动品牌模板没有有效前景")
    x, y, a, b = int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)
    output = []
    for scale in scales:
        for flip in [False, True]:
            t = cv2.resize(feature[y:b, x:a], None, fx=scale, fy=scale)
            ink = cv2.resize(
                mask[y:b, x:a],
                (t.shape[1], t.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if flip:
                t, ink = cv2.flip(t, 1), cv2.flip(ink, 1)
            output.append(dict(template=t, mask=ink, scale=scale, flip=flip))
    return output


def match(image, bank, threshold=0.5, limit=8, allow_partial=False):
    feature = highpass(image)
    source_feature = feature
    pad_x = max(v["template"].shape[1] for v in bank) if allow_partial else 0
    pad_y = max(v["template"].shape[0] for v in bank) if allow_partial else 0
    if allow_partial:
        feature = np.pad(feature, ((pad_y, pad_y), (pad_x, pad_x)))
    found = []
    for identity, v in enumerate(bank):
        t = v["template"]
        mask = v["mask"]
        if t.shape[0] > feature.shape[0] or t.shape[1] > feature.shape[1]:
            continue
        ink = (
            (mask > 0) & (np.abs(t) >= max(0.5, float(np.median(np.abs(t[mask > 0])))))
        ).astype(np.uint8)
        if np.count_nonzero(ink) < 12:
            continue
        scores = cv2.matchTemplate(feature, t, cv2.TM_CCORR_NORMED, mask=ink)
        if allow_partial:
            # 只按实际可见的模板能量归一化，画外的文字不能稀释仍可见的图标。
            h, w = t.shape
            top = np.clip(pad_y - np.arange(scores.shape[0]), 0, h)
            bottom = np.clip(pad_y + image.shape[0] - np.arange(scores.shape[0]), 0, h)
            left = np.clip(pad_x - np.arange(scores.shape[1]), 0, w)
            right = np.clip(pad_x + image.shape[1] - np.arange(scores.shape[1]), 0, w)

            def visible(values):
                integral = cv2.integral(values.astype(np.float64))
                return (
                    integral[bottom[:, None], right[None, :]]
                    - integral[top[:, None], right[None, :]]
                    - integral[bottom[:, None], left[None, :]]
                    + integral[top[:, None], left[None, :]]
                )

            energy = (t * ink) ** 2
            seen = visible(energy)
            scores *= np.sqrt(float(energy.sum()) / np.maximum(seen, 1e-8))
            scores[(seen < float(energy.sum()) * 0.03) | (visible(ink) < 64)] = -1

        scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
        scores[(scores > 1.0001) | (scores < -1.0001)] = -1
        accepted = 0
        for _ in range(limit * 8):
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if score < threshold:
                break
            h, w = t.shape
            box = [x - pad_x, y - pad_y, x + w - pad_x, y + h - pad_y]
            # 用双精度直接复核峰值；掩膜相关在近常量画面上可能出现有限但大于1的伪峰。
            reliable = verified_score(image, v, box, source_feature)
            if reliable >= threshold:
                found.append(dict(variant=identity, confidence=reliable, box=box))
                accepted += 1
            scores[
                max(0, y - h // 2) : y + h // 2 + 1, max(0, x - w // 2) : x + w // 2 + 1
            ] = -1
            if accepted >= limit:
                break
    selected = []
    for candidate in sorted(found, key=lambda p: p["confidence"], reverse=True):
        x, y, a, b = candidate["box"]
        if any(
            max(0, min(a, q["box"][2]) - max(x, q["box"][0]))
            * max(0, min(b, q["box"][3]) - max(y, q["box"][1]))
            > 0.4
            * min(
                (a - x) * (b - y),
                (q["box"][2] - q["box"][0]) * (q["box"][3] - q["box"][1]),
            )
            for q in selected
        ):
            continue
        selected.append(candidate)
    return selected


def mask_for(shape, bank, matches):
    result = np.zeros(shape, np.uint8)
    for point in matches:
        x, y, a, b = point["box"]
        left, top, right, bottom = (
            max(0, x),
            max(0, y),
            min(shape[1], a),
            min(shape[0], b),
        )
        if left < right and top < bottom:
            result[top:bottom, left:right] |= bank[point["variant"]]["mask"][
                top - y : bottom - y, left - x : right - x
            ]
    return result


def verified_score(image, variant, box, feature=None):
    x, y, a, b = box
    left, top, right, bottom = (
        max(0, x),
        max(0, y),
        min(image.shape[1], a),
        min(image.shape[0], b),
    )
    if left >= right or top >= bottom:
        return -1.0
    template = variant["template"]
    mask = variant["mask"]
    ink = (mask > 0) & (
        np.abs(template) >= max(0.5, float(np.median(np.abs(template[mask > 0]))))
    )
    active = ink[top - y : bottom - y, left - x : right - x]
    if np.count_nonzero(active) < 64:
        return -1.0
    t = template[top - y : bottom - y, left - x : right - x][active].astype(np.float64)
    if feature is None:
        feature = highpass(image)
    z = feature[top:bottom, left:right][active].astype(np.float64)
    if np.sqrt(np.mean(z * z)) < 0.01:
        return -1.0
    return float(np.dot(t, z) / max(1e-12, np.linalg.norm(t) * np.linalg.norm(z)))


def temporal_brand_consensus(patches, mask):
    """跨独立背景建立标识模板，防止把单帧图案下的人物细节当成残留。"""
    if len(patches) < 15:
        raise ValueError("品牌时域共识样本不足")
    groups = np.stack(
        [
            np.median(np.stack([patches[i] for i in indices]), axis=0)
            for indices in np.array_split(np.arange(len(patches)), 5)
        ]
    )
    template = np.median(groups, axis=0).astype(np.float32)
    noise = (
        float(np.median(np.abs(template[mask == 0])))
        if np.any(mask == 0)
        else float(np.median(np.abs(template)))
    )
    agreement = np.sum(groups * np.sign(template)[None] > 0.3, axis=0)
    support = (
        (mask > 0) & (np.abs(template) > max(0.5, noise * 3)) & (agreement >= 4)
    ).astype(np.uint8) * 255
    if np.count_nonzero(support) < 64:
        raise ValueError("完整品牌的独立时域支持不足")
    return template, support
