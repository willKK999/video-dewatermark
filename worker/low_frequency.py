"""完整图形低频填色的时域复检与自动圆形播放部件补全，不使用影片坐标。"""

from .hls_settings import is_residual_score

import cv2
import numpy as np
from .multitrack import component_scores


def band(image):
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if image.ndim == 3
        else image.astype(np.float32)
    )
    return cv2.GaussianBlur(gray, (0, 0), 4) - cv2.GaussianBlur(gray, (0, 0), 16)


def fixed_scores(template, consensus, regions):
    records = []
    for r in regions:
        x, y, a, b = r["box"]
        scores = component_scores(
            template[y:b, x:a], consensus[y:b, x:a], np.ones((b - y, a - x), np.uint8)
        )
        bad = [p for p in scores if is_residual_score(p)]
        records.append(dict(region=r["id"], components=scores, residual=bool(bad)))
    return records


def complete_play_graphics(template, signal, region):
    """模型确认的共识区域内，圆弧及居中近等边三角形共同印证完整图标。"""
    x, y, a, b = region["box"]
    z = template[y:b, x:a]
    active = (signal[y:b, x:a] > 0) & (region["mask"][y:b, x:a] > 0)
    result = np.zeros(template.shape, np.uint8)
    if not active.any():
        return result
    scale = max(0.5, float(np.percentile(np.abs(z[active]), 95)))
    patch = np.clip(128 - z * 64 / scale, 0, 255).astype(np.uint8)
    h, w = patch.shape
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    patch = cv2.copyMakeBorder(
        patch,
        top,
        side - h - top,
        left,
        side - w - left,
        cv2.BORDER_CONSTANT,
        value=128,
    )
    patch = cv2.resize(patch, None, fx=2, fy=2)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(patch, (5, 5), 1),
        cv2.HOUGH_GRADIENT,
        1.2,
        side * 0.16,
        param1=100,
        param2=30,
        minRadius=24,
        maxRadius=round(side * 0.6),
    )
    if circles is None:
        return result
    edges = cv2.Canny(patch, 60, 120)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 5)
    gy, gx = np.indices(z.shape)
    for cx, cy, radius in circles[0]:
        angles = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        xx = np.rint(cx + radius * np.cos(angles)).astype(int)
        yy = np.rint(cy + radius * np.sin(angles)).astype(int)
        valid = (
            (xx >= left * 2)
            & (xx < (left + w) * 2)
            & (yy >= top * 2)
            & (yy < (top + h) * 2)
        )
        if valid.mean() < 0.95 or np.mean(distance[yy[valid], xx[valid]] < 3) < 0.6:
            continue
        xc, yc = cx / 2 - left, cy / 2 - top
        r = radius / 2
        inner = (gx - xc) ** 2 + (gy - yc) ** 2 < (r * 0.75) ** 2
        confirmed = False
        for sign in [1, -1]:
            binary = ((z * sign > 0.3) & inner).astype(np.uint8) * 255
            for contour in cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]:
                poly = cv2.approxPolyDP(
                    contour, cv2.arcLength(contour, True) * 0.08, True
                )
                if len(poly) != 3:
                    continue
                points = poly.reshape(-1, 2)
                area = cv2.contourArea(contour)
                sides = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
                if (
                    0.025 * np.pi * r * r < area < 0.45 * np.pi * r * r
                    and sides.min() > 0
                    and sides.max() / sides.min() <= 1.35
                    and np.linalg.norm(points.mean(0) - [xc, yc]) <= r * 0.2
                ):
                    confirmed = True
                    break
            if confirmed:
                break
        if confirmed:
            cv2.circle(result, (round(x + xc), round(y + yc)), round(r + 3), 255, -1)
    return result


def repair_low_band(previous, old_base, new_base, mask, sigma=3):
    """仅补上原掩膜遗漏的低频填色，保持已接受的成品高频及掩膜外像素。"""
    before = cv2.GaussianBlur(old_base.astype(np.float32), (0, 0), sigma)
    after = cv2.GaussianBlur(new_base.astype(np.float32), (0, 0), sigma)
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip(distance / 2, 0, 1)[:, :, None]
    repaired = np.clip(
        np.rint(previous.astype(np.float32) + (after - before) * weight), 0, 255
    ).astype(np.uint8)
    repaired[mask == 0] = previous[mask == 0]
    return repaired
