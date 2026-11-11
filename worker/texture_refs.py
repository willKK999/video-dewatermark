"""以去水印画面估计局部光流，只回填通过双向及颜色校验的原生干净参考像素。"""

from functools import lru_cache
from pathlib import Path
import cv2, numpy as np


class NativeReferences:
    def __init__(
        self,
        coarse,
        source_frames,
        scenes,
        match_low_frequency=False,
        detail_sigma=1.2,
        blur_backend=None,
    ):
        self.coarse = Path(coarse)
        self.source_frames = Path(source_frames)
        self.scenes = scenes
        self.match_low_frequency = match_low_frequency
        self.detail_sigma = detail_sigma
        self.blur_backend = blur_backend
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    @lru_cache(maxsize=64)
    def guide(self, index):
        image = cv2.imread(str(self.coarse / "composite" / f"{index:06d}.png"))
        return cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2GRAY)

    @lru_cache(maxsize=64)
    def source(self, index):
        return cv2.imread(str(self.source_frames / f"{index:06d}.png"))

    @lru_cache(maxsize=32)
    def source_low(self, index):
        return self.low_frequency(self.source(index))

    @lru_cache(maxsize=64)
    def mask(self, index):
        return cv2.imread(str(self.coarse / "masks" / f"{index:06d}.png"), 0)

    def low_frequency(self, image):
        if self.blur_backend is None:
            return cv2.GaussianBlur(image.astype(np.float32), (0, 0), self.detail_sigma)
        engine = self.blur_backend
        with engine.torch.inference_mode():
            return (
                engine.blur(engine.tensor(image).float(), self.detail_sigma)
                .cpu()
                .numpy()
            )

    @lru_cache(maxsize=64)
    def color(self, index):
        return cv2.resize(self.source(index), (640, 360)).astype(np.float32)

    @lru_cache(maxsize=64)
    def small_mask(self, index):
        return (
            cv2.resize(self.mask(index), (640, 360), interpolation=cv2.INTER_NEAREST)
            > 0
        )

    @lru_cache(maxsize=128)
    def motion(self, index, target):
        return self.flow.calc(self.guide(index), self.guide(target), None)

    @lru_cache(maxsize=4)
    def restore(self, index):
        original = self.source(index)
        h, w = original.shape[:2]
        base = cv2.resize(
            cv2.imread(str(self.coarse / "composite" / f"{index:06d}.png")),
            (w, h),
            interpolation=cv2.INTER_CUBIC,
        )
        mask = cv2.resize(self.mask(index), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        clean = base.copy()
        yy, xx = np.where(mask)
        filled = np.zeros((h, w), bool)
        if not len(xx):
            return clean, filled
        count = len(xx)
        rows = (count + 511) // 512
        pad = rows * 512 - count

        def pack(values):
            return np.pad(values.astype(np.float32), (0, pad), mode="edge").reshape(
                rows, 512
            )

        sx = pack((xx + 0.5) * 640 / w - 0.5)
        sy = pack((yy + 0.5) * 360 / h - 0.5)
        a, b = next((a, b) for a, b in self.scenes if a <= index < b)
        current = self.guide(index)
        gy, gx = np.indices(current.shape, dtype=np.float32)
        best = np.full(count, 1e6, np.float32)
        background = self.low_frequency(base)[yy, xx]
        current_color = self.color(index)
        current_occupied = self.small_mask(index)
        for offset in [-5, 5, -10, 10, -20, 20]:
            target = index + offset
            if not a <= target < b:
                continue
            other = self.guide(target)
            forward = self.motion(index, target)
            backward = self.motion(target, index)
            reverse = cv2.remap(
                backward,
                gx + forward[:, :, 0],
                gy + forward[:, :, 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=1e4,
            )
            consistency = np.linalg.norm(forward + reverse, axis=2)
            motion = cv2.remap(forward, sx, sy, cv2.INTER_LINEAR).reshape(-1, 2)[:count]
            mx = xx + motion[:, 0] * w / 640
            my = yy + motion[:, 1] * h / 360
            reference = cv2.remap(
                self.source(target),
                pack(mx),
                pack(my),
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT,
            ).reshape(-1, 3)[:count]
            occupied = (
                cv2.remap(
                    self.mask(target),
                    pack(mx * self.mask(target).shape[1] / w),
                    pack(my * self.mask(target).shape[0] / h),
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255,
                ).reshape(-1)[:count]
                > 0
            )
            # 在全部干净邻域像素上校验颜色，避免四个固定采样点恰好都落在其他笔画上。
            other_color = self.color(target)
            warped_color = cv2.remap(
                other_color,
                gx + forward[:, :, 0],
                gy + forward[:, :, 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=1e4,
            )
            other_occupied = (
                cv2.remap(
                    self.small_mask(target).astype(np.uint8),
                    gx + forward[:, :, 0],
                    gy + forward[:, :, 1],
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255,
                )
                > 0
            )
            blocked = (
                cv2.dilate(
                    (current_occupied | other_occupied).astype(np.uint8),
                    np.ones((3, 3), np.uint8),
                )
                > 0
            )
            valid_ring = (~blocked & (consistency < 0.6)).astype(np.float32)
            difference = np.abs(current_color - warped_color).mean(axis=2)
            kernel = max(15, round(24 * 640 / w) | 1)
            total = cv2.boxFilter(
                difference * valid_ring, -1, (kernel, kernel), normalize=False
            )
            samples = cv2.boxFilter(valid_ring, -1, (kernel, kernel), normalize=False)
            photometric = np.where(
                samples >= max(20, kernel * kernel * 0.09),
                total / np.maximum(samples, 1),
                1e6,
            )
            photo = cv2.remap(photometric, sx, sy, cv2.INTER_LINEAR).reshape(-1)[:count]
            error_fb = cv2.remap(consistency, sx, sy, cv2.INTER_LINEAR).reshape(-1)[
                :count
            ]
            error = error_fb * 10 + photo
            valid = (
                ~occupied
                & (error_fb < 0.6)
                & (photo < 12)
                & (mx >= 0)
                & (mx < w - 1)
                & (my >= 0)
                & (my < h - 1)
                & (error < best)
            )
            if self.match_low_frequency:
                # 参考帧只提供真实细节；亮度与底色沿用目标背景，避免曝光差形成半透明色块。
                reference_low = cv2.remap(
                    self.source_low(target),
                    pack(mx),
                    pack(my),
                    cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT,
                ).reshape(-1, 3)[:count]
                texture = reference.astype(np.float32) - reference_low + background
                clean[yy[valid], xx[valid]] = np.clip(
                    np.rint(texture[valid]), 0, 255
                ).astype(np.uint8)
            else:
                clean[yy[valid], xx[valid]] = reference[valid]
            best[valid] = error[valid]
            filled[yy[valid], xx[valid]] = True
        return clean, filled
