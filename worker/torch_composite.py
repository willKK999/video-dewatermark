"""与七版正规方程一致的批量纹理投影；设备显式选择，默认不使用半精度。"""

import cv2
import numpy as np


class TorchCompositor:
    def __init__(self, device="cuda", tile_batch=48):
        import torch

        if not 1 <= tile_batch <= 256:
            raise ValueError("投影批量必须为一至二百五十六")
        self.torch = torch
        self.device = torch.device(device)
        self.tile_batch = tile_batch
        self.glyph_cache = []

    def tensor(self, array):
        return self.torch.as_tensor(np.ascontiguousarray(array), device=self.device)

    def blur(self, data, sigma):
        """输入为高宽通道，反射边界与七版一致。"""
        torch = self.torch
        functional = torch.nn.functional
        size = int(round(sigma * 8 + 1)) | 1
        radius = size // 2
        kernel = self.tensor(cv2.getGaussianKernel(size, sigma, cv2.CV_32F).ravel())
        x = data.permute(2, 0, 1)[None]
        channels = x.shape[1]

        def indices(length):
            if length == 1:
                return torch.zeros(
                    length + 2 * radius, device=self.device, dtype=torch.long
                )
            idx = torch.arange(-radius, length + radius, device=self.device) % (
                2 * length - 2
            )
            return torch.minimum(idx, 2 * length - 2 - idx)

        x = x.index_select(3, indices(x.shape[3]))
        x = functional.conv2d(
            x, kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels
        )
        x = x.index_select(2, indices(x.shape[2]))
        x = functional.conv2d(
            x, kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels
        )
        return x[0].permute(1, 2, 0)

    def glyphs(self, templates):
        values = []
        for template in templates:
            for reference, glyph in self.glyph_cache:
                if reference is template:
                    break
            else:
                g = template.astype(np.float32)
                glyph = np.stack(
                    (
                        g,
                        cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8,
                        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8,
                    ),
                    axis=-1,
                )
                self.glyph_cache.append((template, glyph))
                self.glyph_cache = self.glyph_cache[-4:]
            values.append(glyph)
        return values

    def soft(self, original, repaired, mask, radius=3):
        torch = self.torch
        with torch.inference_mode():
            source = self.tensor(original).float()
            base = self.tensor(repaired).float()
            distance = cv2.distanceTransform(
                (mask > 0).astype(np.uint8), cv2.DIST_L2, 5
            )
            alpha = self.tensor(np.clip(distance / radius, 0, 1))[:, :, None]
            return (
                torch.clamp(torch.round(source * (1 - alpha) + base * alpha), 0, 255)
                .byte()
                .cpu()
                .numpy()
            )

    def project(
        self,
        original,
        repaired,
        mask,
        watermark_highpass,
        detail_exclusion=None,
        accumulator=None,
        calibration=None,
        detail_sigma=1.2,
    ):
        if accumulator is not None:
            raise ValueError("跨帧标定仍使用七版累计入口")
        torch = self.torch
        if not np.any(mask):
            return original.copy()
        templates = (
            watermark_highpass
            if isinstance(watermark_highpass, (list, tuple))
            else [watermark_highpass]
        )
        glyphs = self.glyphs(templates)
        with torch.inference_mode():
            source, base = self.tensor(original).float(), self.tensor(repaired).float()
            high = source - self.blur(source, detail_sigma)
            low_base = self.blur(base, detail_sigma)
            base_high = base - low_base
            residual, low = high - base_high, low_base / 255
            correction = torch.zeros_like(source)
            weights = torch.zeros(mask.shape, device=self.device)
            h, w = mask.shape
            # 分组消除可变矩阵阶数；每批上限控制显存，不构造整帧展开矩阵。
            groups = {}
            for y in range(0, h, 32):
                for x in range(0, w, 32):
                    a, b = min(w, x + 64), min(h, y + 64)
                    valid = mask[y:b, x:a] > 0
                    if valid.sum() < 20:
                        continue
                    local = [
                        g[y:b, x:a]
                        for g in glyphs
                        if float(np.sum(g[y:b, x:a, 0][valid] ** 2)) >= 1e-4
                    ]
                    if not local:
                        continue
                    window = (
                        np.maximum(0.05, np.hanning(b - y))[:, None]
                        * np.maximum(0.05, np.hanning(a - x))[None, :]
                    ).astype(np.float32)
                    weights[y:b, x:a] += self.tensor(window)
                    for channel in range(3):
                        fixed = (calibration or {}).get(f"{x},{y},{channel}")
                        key = (b - y, a - x, len(local), fixed is not None)
                        groups.setdefault(key, []).append(
                            (x, y, a, b, channel, local, valid, window, fixed)
                        )
            for jobs in groups.values():
                for offset in range(0, len(jobs), self.tile_batch):
                    batch = jobs[offset : offset + self.tile_batch]
                    # 每批一次传输字形及支持，避免每个通道都同步拷贝小数组。
                    glyph_batch = self.tensor(
                        np.stack([np.stack(job[5]) for job in batch])
                    )
                    low_batch = torch.stack(
                        [low[y:b, x:a, c] for x, y, a, b, c, *_ in batch]
                    )
                    g = glyph_batch[..., 0]
                    d = (
                        torch.stack(
                            (
                                g,
                                g * low_batch[:, None],
                                glyph_batch[..., 1],
                                glyph_batch[..., 2],
                            ),
                            dim=-1,
                        )
                        .permute(0, 2, 3, 1, 4)
                        .flatten(3)
                    )
                    prediction = torch.zeros_like(low_batch)
                    if batch[0][-1] is not None:
                        coefficients = self.tensor(
                            np.array([job[-1] for job in batch], np.float64)
                        )
                        prediction = (
                            (d[..., :4].double() * coefficients[:, None, None])
                            .sum(dim=-1)
                            .float()
                        )
                        d = d[..., 4:]
                    targets = (
                        torch.stack(
                            [residual[y:b, x:a, c] for x, y, a, b, c, *_ in batch]
                        )
                        - prediction
                    )
                    active = np.stack([job[6] for job in batch])
                    count = d.shape[-1]
                    if count:
                        valid = self.tensor(active).flatten(1)
                        X = d.flatten(1, 2).double() * valid[..., None]
                        Y = targets.flatten(1).double() * valid
                        normal = X.transpose(1, 2) @ X
                        ridge = torch.clamp(
                            X.square().sum(dim=(1, 2)) * 0.0001, min=1e-6
                        )
                        normal += (
                            torch.eye(count, device=self.device, dtype=torch.float64)[
                                None
                            ]
                            * ridge[:, None, None]
                        )
                        coef = torch.linalg.solve(
                            normal, (X.transpose(1, 2) @ Y[..., None])
                        )
                        prediction = (
                            prediction
                            + (d.flatten(1, 2).double() @ coef)
                            .reshape(prediction.shape)
                            .float()
                        )
                    for j, (x, y, a, b, c, _, _, window, _) in enumerate(batch):
                        correction[y:b, x:a, c] += prediction[j] * self.tensor(window)
            correction /= torch.clamp(weights[..., None], min=1e-6)
            repaired_high = high - correction
            if detail_exclusion is not None:
                guard = self.blur(
                    self.tensor((detail_exclusion > 0).astype(np.float32))[..., None],
                    0.6,
                )
                repaired_high = repaired_high * (1 - guard) + base_high * guard
            rebuilt = base - base_high + repaired_high
            distance = cv2.distanceTransform(
                (mask > 0).astype(np.uint8), cv2.DIST_L2, 5
            )
            alpha = self.tensor(np.clip(distance / 2, 0, 1))[..., None]
            result = source * (1 - alpha) + rebuilt * alpha
            return torch.clamp(torch.round(result), 0, 255).byte().cpu().numpy()
