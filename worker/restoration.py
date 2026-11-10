"""在原水印形状掩膜内融合；只用同镜头、经几何与颜色验证的原始背景。"""
import time
import cv2
import numpy as np


def context_mask(mask, radius=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(mask, kernel)


def soft_composite(original, repaired, mask, radius=3):
    # 现有形状掩膜已有至少 3 像素外扩；仅在其内侧安全边缘羽化。
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(distance/radius, 0, 1)
    support = mask > 0
    result = original.copy()
    weight = alpha[support, None]
    result[support] = np.rint(original[support]*(1-weight)+repaired[support]*weight).astype(np.uint8)
    return result


def fill_small_holes(background, confidence, mask):
    """只补平滑共识内部的小孔；不扩展回填到原水印掩膜以外。"""
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return background, confidence, 0
    h, w = mask.shape
    x1, x2 = max(0, xs.min()-4), min(w, xs.max()+5)
    y1, y2 = max(0, ys.min()-4), min(h, ys.max()+5)
    region = background[y1:y2, x1:x2].copy()
    reliable = confidence[y1:y2, x1:x2].copy()
    core = mask[y1:y2, x1:x2] > 0
    closed = cv2.morphologyEx(reliable, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    holes = ((closed > 0) & (reliable == 0) & core).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes)
    fill = np.zeros_like(holes)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] > 64:
            continue
        component = (labels == label).astype(np.uint8)
        border = (cv2.dilate(component, np.ones((3, 3), np.uint8)) > 0) & (component == 0)
        if (border.any() and np.mean(reliable[border] > 0) >= .7
                and region[border].astype(float).std(axis=0).max() <= 8):
            fill[component > 0] = 255
    if fill.any():
        region = cv2.inpaint(region, fill, 3, cv2.INPAINT_TELEA)
        reliable[fill > 0] = 1
        background[y1:y2, x1:x2] = region
        confidence[y1:y2, x1:x2] = reliable
    return background, confidence, int(np.count_nonzero(fill))


class CleanReferences:
    """保守的仿射背景回填；移动物体、遮挡和不一致参考自动退回模型。"""
    def __init__(self, images, masks):
        self.stats = dict(processed_frames=0, masked_pixels=0, restored_pixels=0, interpolated_pixels=0, reference_frames=len(images))
        self.timings = dict(tracking_seconds=0., alignment_seconds=0., median_seconds=0.)
        self.trusted = [255-context_mask(mask, 3) for mask in masks]
        self.images = images
        self.masks = masks
        self.grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]

    def _track(self, gray, refgray, points):
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray, refgray, points, None)
        if tracked is None:
            return None
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(refgray, gray, tracked, None)
        if back is None:
            return None
        good = (status.ravel() > 0) & (reverse.ravel() > 0) & (np.linalg.norm(back-points, axis=2).ravel() < 1)
        if good.sum() < 16:
            return None
        transform, inliers = cv2.estimateAffinePartial2D(points[good], tracked[good], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if transform is None or inliers.sum() < 12 or inliers.mean() < .65:
            return None
        return transform

    def restore(self, image, mask, repaired):
        self.stats["processed_frames"] += 1
        self.stats["masked_pixels"] += int(np.count_nonzero(mask))
        if not mask.any() or len(self.images) < 3:
            return repaired
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        valid = 255-context_mask(mask, 5)
        points = cv2.goodFeaturesToTrack(gray, 250, .01, 6, mask=valid)
        if points is None or len(points) < 16:
            return repaired
        h, w = mask.shape
        ring = (context_mask(mask, 8) > 0) & (context_mask(mask, 3) == 0)
        if ring.sum() < 32:
            return repaired
        donors = []
        support = mask > 0
        for reference, reftrusted, refgray in zip(self.images, self.trusted, self.grays):
            started = time.perf_counter()
            transform = self._track(gray, refgray, points)
            self.timings['tracking_seconds'] += time.perf_counter()-started
            if transform is None:
                continue
            started = time.perf_counter()
            aligned = cv2.warpAffine(reference, transform, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            trusted = cv2.warpAffine(reftrusted, transform, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
            self.timings['alignment_seconds'] += time.perf_counter()-started
            usable = ring & trusted
            if usable.sum() < .35*ring.sum():
                continue
            error = np.abs(aligned[usable].astype(np.float32)-image[usable].astype(np.float32)).mean(axis=1)
            if error.mean() > 6 or np.percentile(error, 90) > 14:
                continue
            donor = aligned[support].astype(np.float32)
            donor[~trusted[support]] = np.nan
            donors.append(donor)
        if len(donors) < 3:
            return repaired
        started = time.perf_counter()
        stack = np.stack(donors)
        count = np.isfinite(stack[..., 0]).sum(axis=0)
        # 不对缺少参考的像素求 nanmedian，避免警告与无效数值。
        eligible = np.zeros(mask.shape, bool)
        eligible[support] = count >= 3
        if not eligible.any():
            self.timings["median_seconds"] += time.perf_counter()-started
            return repaired
        values = stack[:, count >= 3]
        median = np.nanmedian(values, axis=0)
        agreement = np.nanmax(np.abs(values-median), axis=(0, 2)) <= 10
        self.timings['median_seconds'] += time.perf_counter()-started
        confidence = np.zeros(mask.shape, np.uint8)
        confidence[eligible] = agreement.astype(np.uint8)
        background = repaired.copy()
        background[eligible] = np.clip(np.rint(median), 0, 255).astype(np.uint8)
        background, confidence, filled = fill_small_holes(background, confidence, mask)
        self.stats["interpolated_pixels"] += filled
        # 仅在一致区域内部逐渐采用参考，避免局部回填产生新接缝。
        weight = np.clip(cv2.distanceTransform(confidence, cv2.DIST_L2, 5)/3, 0, 1)
        self.stats["restored_pixels"] += int(np.count_nonzero(weight >= .5))
        result = repaired.copy()
        active = weight > 0
        alpha = weight[active, None]
        result[active] = np.rint(background[active]*alpha+repaired[active]*(1-alpha)).astype(np.uint8)
        return result
