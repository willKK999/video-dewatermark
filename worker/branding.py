"""以模型定位和自动学习的标识文本验证短时品牌，字幕不能仅因持久而成为水印。"""

import csv, io, re, subprocess
from difflib import SequenceMatcher
import cv2, numpy as np


def text_signature(value):
    return re.sub("[^a-z0-9]", "", value.lower())


def same_brand(value, references):
    queries = [
        text_signature(v) for v in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]{7,}", value)
    ]
    known = [
        text_signature(v)
        for r in references
        for v in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]{7,}", r)
    ]
    return any(
        SequenceMatcher(None, a, b).ratio() >= 0.8
        for a in queries
        for b in known
        if min(len(a), len(b)) >= 8
    )


def recognize(image, psm=7):
    # 只调用本机OCR，不访问外部识别服务或发送视频。
    ok, png = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("识别图像编码失败")
    result = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", str(psm)],
        input=png.tobytes(),
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def merge_text_rows(boxes):
    rows = []
    for item in sorted(boxes, key=lambda p: p["confidence"], reverse=True):
        box = list(item["box"])
        for row in rows:
            a = row["box"]
            h = min(a[3] - a[1], box[3] - box[1])
            gap = max(0, max(a[0], box[0]) - min(a[2], box[2]))
            if abs((a[1] + a[3]) - (box[1] + box[3])) / 2 < h * 0.6 and gap < h * 2:
                row["box"] = [
                    min(a[0], box[0]),
                    min(a[1], box[1]),
                    max(a[2], box[2]),
                    max(a[3], box[3]),
                ]
                break
        else:
            rows.append(dict(box=box, confidence=item["confidence"]))
    return rows


def grow_brand_components(signal, anchors):
    joined = cv2.morphologyEx(signal, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    components = []
    for identity, (x, y, w, h, area) in enumerate(stats[1:], 1):
        scale = max((b[2] - b[0] for b in anchors), default=0)
        envelope_ok = any(
            x + w >= b[0] - 0.65 * scale
            and x <= b[2] + 0.65 * scale
            and y >= b[1] - 0.65 * scale
            and y + h <= b[3] + 0.65 * scale
            for b in anchors
        )
        strip_ok = h <= max(w * 6, max((b[3] - b[1] for b in anchors), default=0) * 2)
        if (
            area >= 30
            and min(w, h) >= 3
            and h <= scale * 0.6
            and w <= scale * 2
            and envelope_ok
            and strip_ok
        ):
            components.append(
                dict(
                    id=identity,
                    box=[int(x), int(y), int(x + w), int(y + h)],
                    area=int(area),
                )
            )
    selected = set()
    for component in components:
        a = component["box"]
        if any(
            max(0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            >= component["area"] * 0.2
            for b in anchors
        ):
            selected.add(component["id"])
    for _ in range(5):
        added = set()
        for c in components:
            if c["id"] in selected:
                continue
            a = c["box"]
            aw, ah = a[2] - a[0], a[3] - a[1]
            for d in components:
                if d["id"] not in selected:
                    continue
                b = d["box"]
                bw, bh = b[2] - b[0], b[3] - b[1]
                gapx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
                gapy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
                if max(ah, bh) > 4 * min(ah, bh):
                    continue
                horizontal = (
                    gapx <= 1.5 * min(ah, bh)
                    and abs((a[1] + a[3]) - (b[1] + b[3])) / 2 <= max(ah, bh) * 0.65
                )
                vertical = (
                    gapy <= 1.5 * min(ah, bh)
                    and abs((a[0] + a[2]) - (b[0] + b[2])) / 2 <= max(aw, bw) * 0.65
                )
                if horizontal or vertical:
                    added.add(c["id"])
                    break
        if not added:
            break
        selected |= added
    return np.where(np.isin(labels, list(selected)), 255, 0).astype(np.uint8)


def learn_palette(images, positive_mask, colors=8):
    """仅学习跨背景稳定的已确认标识前景颜色，不采样字幕或人工框。"""
    stack = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2LAB) for im in images]).astype(
        np.float32
    )
    median = np.median(stack, axis=0)
    deviation = np.median(np.abs(stack - median), axis=0).mean(axis=2)
    active = (positive_mask > 0) & (deviation < 10)
    values = median[active]
    if len(values) < colors:
        raise ValueError("稳定品牌颜色样本不足")
    _, labels, centers = cv2.kmeans(
        values.astype(np.float32),
        colors,
        None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    # 正对比前景颜色中抛弃极暗的压缩边缘，避免把黑背景加入文字图形。
    centers = centers[centers[:, 0] > max(40, float(centers[:, 0].max()) * 0.35)]
    return centers


def palette_support(image, palette, distance=22):
    colors = cv2.cvtColor(palette.astype(np.uint8)[None], cv2.COLOR_LAB2BGR)[0]
    centers = cv2.cvtColor(colors[None], cv2.COLOR_BGR2HSV)[0].astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    mask = np.zeros(image.shape[:2], np.uint8)
    for h, s, v in centers:
        if s > 90:
            hue = np.minimum(np.abs(hsv[:, :, 0] - h), 180 - np.abs(hsv[:, :, 0] - h))
            selected = (hue < 7) & (hsv[:, :, 1] > s * 0.45) & (hsv[:, :, 2] > v * 0.6)
        elif v > 215:
            selected = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > v * 0.85)
        else:
            continue
        mask[selected] = 255
    colorful = ((hsv[:, :, 1] > 160) & (hsv[:, :, 2] > 90)).astype(np.uint8)
    n, groups, properties, _ = cv2.connectedComponentsWithStats(colorful)
    near = cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0
    capacity = max(30, np.count_nonzero(mask) * 0.4)
    for identity in range(1, n):
        component = groups == identity
        if (
            np.count_nonzero(component & near) >= 3
            and np.count_nonzero(component & (mask == 0)) <= capacity
        ):
            mask[component] = 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    for identity, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if (
            area < 12
            or x == 0
            or y == 0
            or x + w == mask.shape[1]
            or y + h == mask.shape[0]
        ):
            mask[labels == identity] = 0
    return mask


def opaque_brand_ink(image, box, palette):
    """在自动品牌框内覆盖所有独立前景笔画，不依赖SAM是否把小字分为同一对象。"""
    from .hls_settings import HLSSettings

    settings = HLSSettings.from_env()
    colors = cv2.cvtColor(palette.astype(np.uint8)[None], cv2.COLOR_LAB2BGR)[0]
    centers = cv2.cvtColor(colors[None], cv2.COLOR_BGR2HSV)[0].astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    selected = np.zeros(image.shape[:2], np.uint8)
    for h, s, v in centers:
        if s > 90:
            hue = np.minimum(np.abs(hsv[:, :, 0] - h), 180 - np.abs(hsv[:, :, 0] - h))
            active = (hue < 7) & (hsv[:, :, 1] > s * .45) & (hsv[:, :, 2] > v * .6)
        elif v > 215:
            active = (hsv[:, :, 1] < settings.opaque_white_saturation) & (hsv[:, :, 2] > max(settings.opaque_white_min_value, v - settings.opaque_palette_margin))
        else:
            continue
        selected[active] = 255
    x, y, a, b = map(round, box)
    pad=round((b-y)*settings.opaque_brand_padding)
    x,y,a,b=x-pad,y-pad,a+pad,b+pad
    x, y, a, b = max(0,x), max(0,y), min(image.shape[1],a), min(image.shape[0],b)
    bounded = np.zeros_like(selected)
    bounded[y:b,x:a] = selected[y:b,x:a]
    radius = settings.opaque_ink_radius
    return cv2.dilate(bounded,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))


def play_icons(image):
    """通用圆形播放标识：必须同时具有圆轮廓、内置三角形和合理面积比。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = min(gray.shape)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (5, 5), 1),
        cv2.HOUGH_GRADIENT,
        1.2,
        small * 0.08,
        param1=100,
        param2=30,
        minRadius=max(6, round(small * 0.01)),
        maxRadius=round(small * 0.3),
    )
    output = []
    edges = cv2.Canny(gray, 60, 120)
    edge_distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    if circles is None:
        return output
    for cx, cy, r in circles[0]:
        angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        xx = np.rint(cx + r * np.cos(angles)).astype(int)
        yy = np.rint(cy + r * np.sin(angles)).astype(int)
        valid = (xx >= 0) & (xx < gray.shape[1]) & (yy >= 0) & (yy < gray.shape[0])
        if (
            valid.mean() < 0.95
            or np.mean(edge_distance[yy[valid], xx[valid]] < 3) < 0.6
        ):
            continue
        x, y, a, b = (
            max(0, round(cx - r)),
            max(0, round(cy - r)),
            min(gray.shape[1], round(cx + r + 1)),
            min(gray.shape[0], round(cy + r + 1)),
        )
        patch = gray[y:b, x:a]
        bright = (patch > max(180, float(np.percentile(patch, 99)) * 0.97)).astype(
            np.uint8
        ) * 255
        contours, _ = cv2.findContours(
            bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if not 0.025 * np.pi * r * r < area < 0.45 * np.pi * r * r:
                continue
            polygon = cv2.approxPolyDP(
                contour, cv2.arcLength(contour, True) * 0.08, True
            )
            if len(polygon) != 3:
                continue
            center = polygon.reshape(-1, 2).mean(axis=0) + [x, y]
            if np.linalg.norm(center - [cx, cy]) > r * 0.4:
                continue
            output.append(
                dict(
                    box=[x, y, a, b],
                    circle=[float(cx), float(cy), float(r)],
                    confidence=1.0,
                )
            )
            break
    return output


def residual_components(result, palette, target_mask):
    """在自动目标近邻检查完整品牌色部件，不要求残留像素与原片颜色完全相同。"""
    if not np.any(target_mask):
        return np.zeros_like(target_mask)
    yy, _ = np.where(target_mask > 0)
    radius = max(3, round((yy.max() - yy.min() + 1) * 0.1))
    near = (
        cv2.dilate(
            target_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
            ),
        )
        > 0
    )
    possible = palette_support(result, palette)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(possible)
    mask = np.zeros_like(target_mask)
    for identity in range(1, count):
        component = labels == identity
        if np.count_nonzero(component & near) >= max(
            3, stats[identity, cv2.CC_STAT_AREA] * 0.2
        ):
            mask[component] = 255
    return cv2.dilate(mask, np.ones((7, 7), np.uint8))
