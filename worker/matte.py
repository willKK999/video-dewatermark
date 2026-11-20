"""透明叠加拟合、稀疏笔画、高频保真合成与独立清晰度检查。"""

import cv2
import numpy as np


def fit_overlay(images, mask):
    active = mask > 0
    shape = mask.shape
    count = 0
    dd = np.zeros(int(active.sum()), np.float64)
    white_num = np.zeros_like(dd)
    white_den = np.zeros_like(dd)
    black_num = np.zeros_like(dd)
    black_den = np.zeros_like(dd)
    for image in images:
        background = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA).astype(np.float64)
        x = background[active]
        y = image[active].astype(np.float64)
        delta = y - x
        w = 255 - x
        k = -x
        dd += np.sum(delta * delta, axis=1)
        white_num += np.sum(delta * w, axis=1)
        white_den += np.sum(w * w, axis=1)
        black_num += np.sum(delta * k, axis=1)
        black_den += np.sum(k * k, axis=1)
        count += 1
    if count < 15:
        raise ValueError("透明叠加拟合至少需要15个不同背景样本")
    aw = np.clip(white_num / np.maximum(white_den, 1), 0, 0.98)
    ab = np.clip(black_num / np.maximum(black_den, 1), 0, 0.98)
    ew = dd - 2 * aw * white_num + aw * aw * white_den
    eb = dd - 2 * ab * black_num + ab * ab * black_den
    white = ew < eb
    alpha = np.zeros(shape, np.float32)
    offset = np.zeros((*shape, 3), np.float32)
    alpha[active] = np.where(white, aw, ab)
    offset[active] = np.where(white, 255 * aw, 0)[:, None]
    return alpha, offset


def invert_overlay(image, alpha, offset, limit=0.45):
    valid = (alpha <= limit) & ((alpha > 0) | np.any(offset != 0, axis=2))
    result = image.astype(np.float32)
    result[valid] = (result[valid] - offset[valid]) / np.maximum(
        1 - alpha[valid, None], 0.1
    )
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def temporal_detail(source_a, source_b, result_a, result_b, mask, exclude_mask=None):
    """静态水印在时间差中抵消；以周边未修改像素归一化编码损失。"""

    def hp(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray - cv2.GaussianBlur(gray, (0, 0), 1.2)

    a = hp(source_b) - hp(source_a)
    b = hp(result_b) - hp(result_a)
    core = mask > 0
    occupied = mask if exclude_mask is None else exclude_mask
    ring = (cv2.dilate(mask, np.ones((41, 41), np.uint8)) > 0) & (
        cv2.dilate(occupied, np.ones((9, 9), np.uint8)) == 0
    )

    def energy(data, where):
        return float(np.sqrt(np.mean(data[where] ** 2))) if where.any() else 0

    si, so = energy(a, core), energy(b, core)
    ri, ro = energy(a, ring), energy(b, ring)
    ratio = (so / max(si, 0.01)) / max(ro / max(ri, 0.01), 0.01)
    return dict(
        source_detail=si,
        result_detail=so,
        source_ring=ri,
        result_ring=ro,
        retention_ratio=ratio,
        measurable=si >= 0.2 and ri >= 0.2,
        threshold=0.8,
        passed=si < 0.2 or ri < 0.2 or ratio >= 0.8,
    )


def ink_support(template, signal, region_mask, radius=2):
    """保留独立笔画及圆形图标内部，不把字间皮肤闭合成整片。"""
    positive = ((signal > 0) & (region_mask > 0)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(positive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if (
            min(w, h) >= 25
            and 0.65 <= w / max(h, 1) <= 1.4
            and area / max(w * h, 1) > 0.45
            and 4 * np.pi * area / max(cv2.arcLength(contour, True) ** 2, 1) > 0.4
        ):
            cv2.drawContours(positive, [contour], -1, 255, cv2.FILLED)
    return cv2.dilate(
        positive,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
    )


def texture_composite(original, repaired, mask, watermark_highpass):
    """只补回与自动水印模板正交的原始高频，避免把文字边缘一起加回。"""
    source = original.astype(np.float32)
    base = repaired.astype(np.float32)
    high = source - cv2.GaussianBlur(source, (0, 0), 1.2)
    template = watermark_highpass.astype(np.float32)
    if template.ndim == 2:
        template = np.repeat(template[:, :, None], 3, axis=2)
    active = mask > 0
    low = base - cv2.GaussianBlur(base, (0, 0), 1.2)
    residual = high - low
    denom = float(np.sum(template[active] ** 2))
    beta = float(
        np.clip(np.sum(residual[active] * template[active]) / max(denom, 1e-6), 0, 16)
    )
    yy, xx = np.where(active)
    if len(xx):
        x, y, r, b = int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)
        grid = np.full((2, 4), beta, np.float32)
        for row in range(2):
            for column in range(4):
                x1 = x + round((r - x) * column / 4)
                x2 = x + round((r - x) * (column + 1) / 4)
                y1 = y + round((b - y) * row / 2)
                y2 = y + round((b - y) * (row + 1) / 2)
                valid = active[y1:y2, x1:x2]
                t = template[y1:y2, x1:x2][valid]
                z = residual[y1:y2, x1:x2][valid]
                energy = float(np.sum(t * t))
                if energy > denom * 0.005:
                    grid[row, column] = np.clip(np.sum(t * z) / energy, 0, 16)
        beta_map = np.full(mask.shape, beta, np.float32)
        beta_map[y:b, x:r] = cv2.resize(
            grid, (r - x, b - y), interpolation=cv2.INTER_LINEAR
        )
    else:
        beta_map = np.zeros(mask.shape, np.float32)
    detail = high - template * beta_map[:, :, None]
    # 无水印纹理来自原图，不使用锐化制造新细节。
    base[active] += detail[active] - low[active]
    distance = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip(distance / 2, 0, 1)[:, :, None]
    return (
        np.clip(np.rint(source * (1 - weight) + base * weight), 0, 255).astype(
            np.uint8
        ),
        beta,
    )


def projected_texture_composite(
    original,
    repaired,
    mask,
    watermark_highpass,
    detail_exclusion=None,
    accumulator=None,
    calibration=None,
    detail_sigma=1.2,
):
    """逐通道局部投影剔除字形频率，保留原图中与水印无关的高频。"""
    source = original.astype(np.float32)
    base = repaired.astype(np.float32)
    high = source - cv2.GaussianBlur(source, (0, 0), detail_sigma)
    base_high = base - cv2.GaussianBlur(base, (0, 0), detail_sigma)
    residual = high - base_high
    templates = (
        watermark_highpass
        if isinstance(watermark_highpass, (list, tuple))
        else [watermark_highpass]
    )
    glyphs = [
        (
            g.astype(np.float32),
            cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3) / 8,
            cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3) / 8,
        )
        for g in templates
    ]
    low = cv2.GaussianBlur(base, (0, 0), detail_sigma) / 255
    correction = np.zeros_like(source)
    weights = np.zeros(mask.shape, np.float32)
    height, width = mask.shape
    for y in range(0, height, 32):
        for x in range(0, width, 32):
            a, b = min(width, x + 64), min(height, y + 64)
            valid = mask[y:b, x:a] > 0
            if valid.sum() < 20:
                continue
            local = [
                (g[y:b, x:a], dx[y:b, x:a], dy[y:b, x:a])
                for g, dx, dy in glyphs
                if float(np.sum(g[y:b, x:a][valid] ** 2)) >= 1e-4
            ]
            if not local:
                continue
            wy = np.maximum(0.05, np.hanning(b - y))
            wx = np.maximum(0.05, np.hanning(a - x))
            window = wy[:, None] * wx[None, :]
            for channel in range(3):
                design = np.stack(
                    [
                        v
                        for g, dx, dy in local
                        for v in (g, g * low[y:b, x:a, channel], dx, dy)
                    ],
                    axis=2,
                )
                X = design[valid].astype(np.float64)
                Y = residual[y:b, x:a, channel][valid].astype(np.float64)
                key = f"{x},{y},{channel}"
                if accumulator is not None:
                    normal, rhs = accumulator.setdefault(
                        key, (np.zeros((X.shape[1], X.shape[1])), np.zeros(X.shape[1]))
                    )
                    normal += X.T @ X
                    rhs += X.T @ Y
                    continue
                fixed = np.zeros(valid.shape, np.float32)
                if calibration and key in calibration:
                    fixed = np.einsum(
                        "ijk,k->ij", design[:, :, :4], calibration[key]
                    ).astype(np.float32)
                    X = X[:, 4:]
                    Y = Y - fixed[valid]
                    design = design[:, :, 4:]
                if X.shape[1] == 0:
                    correction[y:b, x:a, channel] += fixed * window
                    continue
                normal = X.T @ X + np.eye(X.shape[1]) * max(
                    float(np.sum(X * X)) * 0.0001, 1e-6
                )
                coefficients = np.linalg.solve(normal, X.T @ Y)
                predicted = np.einsum("ijk,k->ij", design, coefficients).astype(
                    np.float32
                )
                correction[y:b, x:a, channel] += (predicted + fixed) * window
            weights[y:b, x:a] += window
    if accumulator is not None:
        return None
    correction /= np.maximum(weights[:, :, None], 1e-6)
    repaired_high = high - correction
    # 模板的任意数值缩放不能改变纹理保留；仅在显式确认的残留笔画上禁用原图细节。
    if detail_exclusion is not None:
        guard = cv2.GaussianBlur(
            (detail_exclusion > 0).astype(np.float32), (0, 0), 0.6
        )[:, :, None]
        repaired_high = repaired_high * (1 - guard) + base_high * guard
    rebuilt = base - base_high + repaired_high
    active = mask > 0
    distance = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(distance / 2, 0, 1)[:, :, None]
    result = source * (1 - alpha) + rebuilt * alpha
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def solve_projection(accumulator):
    """跨背景累计正规方程，只拟合一次字形系数，避免逐帧拟合吃掉移动皮肤纹理。"""
    result = {}
    for key, (normal, rhs) in accumulator.items():
        ridge = max(float(np.trace(normal)) * 0.0001, 1e-6)
        result[key] = np.linalg.solve(normal + np.eye(len(rhs)) * ridge, rhs).tolist()
    return result


def stable_detail_mask(mask, other_a, other_b):
    """固定标识的时间差核验须避开其他移动标识，避免把去掉的移动笔画当成细节损失。"""
    occupied = cv2.dilate(cv2.bitwise_or(other_a, other_b), np.ones((7, 7), np.uint8))
    return np.where((mask > 0) & (occupied == 0), 255, 0).astype(np.uint8)


def restore_intermediate_frame(current, anchor_clean, source, anchor_source, mask):
    """固定半透明叠加在相邻原帧差中抵消，用真实变化恢复中间帧，避免修复区重复冻结。"""
    delta = source.astype(np.float32) - anchor_source.astype(np.float32)
    target = np.clip(np.rint(anchor_clean.astype(np.float32) + delta), 0, 255).astype(
        np.uint8
    )
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip(distance / 2, 0, 1)[:, :, None]
    return np.clip(np.rint(current * (1 - weight) + target * weight), 0, 255).astype(
        np.uint8
    )
