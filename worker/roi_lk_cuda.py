"""逐层 ROI 批量稀疏金字塔 LK 实验路径。"""
import ctypes
from functools import lru_cache
import os

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _reflect(index, size):
    period = tl.maximum(1, 2 * size - 2)
    index = tl.abs(index) % period
    return tl.minimum(index, period - index)


@triton.jit
def _downsample(source, target, width, height,
                out_w, out_h, block: tl.constexpr):
    pixel = tl.program_id(0) * block + tl.arange(0, block)
    image = tl.program_id(1)
    x, y = 2 * (pixel % out_w), 2 * (pixel // out_w)
    value = tl.full((block,), 0, tl.int32)
    for dy in tl.static_range(-2, 3):
        for dx in tl.static_range(-2, 3):
            wx = 6 if dx == 0 else (4 if dx == 1 or dx == -1 else 1)
            wy = 6 if dy == 0 else (4 if dy == 1 or dy == -1 else 1)
            sample = tl.load(
                source + (image * height + _reflect(y + dy, height)) * width
                + _reflect(x + dx, width), pixel < out_w * out_h, 0,
            )
            value += sample.to(tl.int32) * (wx * wy)
    tl.store(target + image * out_w * out_h + pixel, (value + 128) >> 8,
             pixel < out_w * out_h)


@triton.jit
def _load(image, x, y, width, height, valid):
    return tl.load(image + _reflect(y, height) * width + _reflect(x, width),
                   valid, 0).to(tl.int32)


@triton.jit
def _scharr(image, x, y, width, height, valid):
    valid = valid & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    upper_left = _load(image, x - 1, y - 1, width, height, valid)
    upper_right = _load(image, x + 1, y - 1, width, height, valid)
    lower_left = _load(image, x - 1, y + 1, width, height, valid)
    lower_right = _load(image, x + 1, y + 1, width, height, valid)
    left = _load(image, x - 1, y, width, height, valid)
    right = _load(image, x + 1, y, width, height, valid)
    upper = _load(image, x, y - 1, width, height, valid)
    lower = _load(image, x, y + 1, width, height, valid)
    return (3 * (upper_right + lower_right - upper_left - lower_left) + 10 * (right - left),
            3 * (lower_left + lower_right - upper_left - upper_right) + 10 * (lower - upper))


@triton.jit
def _weights(x, y):
    floor_x, floor_y = tl.floor(x), tl.floor(y)
    ax, ay = x - floor_x, y - floor_y
    w00 = libdevice.float2int_rn((1. - ax) * (1. - ay) * 16384.)
    w01 = libdevice.float2int_rn(ax * (1. - ay) * 16384.)
    w10 = libdevice.float2int_rn((1. - ax) * ay * 16384.)
    return floor_x.to(tl.int32), floor_y.to(tl.int32), w00, w01, w10, 16384 - w00 - w01 - w10


@triton.jit
def _intensity(image, x, y, w00, w01, w10, w11, width,
               height, valid):
    return ((_load(image, x, y, width, height, valid) * w00
             + _load(image, x + 1, y, width, height, valid) * w01
             + _load(image, x, y + 1, width, height, valid) * w10
             + _load(image, x + 1, y + 1, width, height, valid) * w11 + 256) >> 9)


@triton.jit
def _inside(x, y, width, height):
    return (x >= -21) & (x < width) & (y >= -21) & (y < height)


@triton.jit
def _lane(values, index: tl.constexpr):
    return tl.sum(tl.gather(values, tl.full((1,), index, tl.int32), 0))


@triton.jit
def _ordered_sum(product, paired: tl.constexpr):
    # 复现基线 x86 OpenCV 逐行 SIMD 四通道与标量尾部的累加次序。
    lane = tl.arange(0, 8)
    total = tl.full((8,), 0., tl.float32)
    for row in range(21):
        base = row * 21
        first = tl.gather(product, base + lane, 0)
        second = tl.gather(product, base + lane + 4, 0)
        third = tl.gather(product, base + lane + 8, 0)
        fourth = tl.gather(product, base + lane + 12, 0)
        if paired:
            total += tl.where(lane < 4, (first + second).to(tl.float32), 0.)
            total += tl.where(lane < 4, (third + fourth).to(tl.float32), 0.)
        else:
            total += tl.where(lane < 4, first.to(tl.float32), 0.)
            total += tl.where(lane < 4, second.to(tl.float32), 0.)
            total += tl.where(lane < 4, third.to(tl.float32), 0.)
            total += tl.where(lane < 4, fourth.to(tl.float32), 0.)
        for x in tl.static_range(16, 21):
            tail = tl.gather(product, base + x + lane * 0, 0)
            total += tl.where(lane == 4, tail.to(tl.float32), 0.)
    return _lane(total, 4) + ((_lane(total, 0) + _lane(total, 2))
                             + (_lane(total, 1) + _lane(total, 3)))


@triton.jit
def _exact_sum(product, paired: tl.constexpr):
    # 所有部分和均为可精确表示的 float32 整数时，归约顺序不影响结果。
    if tl.sum(tl.abs(product).to(tl.int64)) <= (1 << 24):
        value = tl.sum(product).to(tl.float32)
    else:
        value = _ordered_sum(product, paired)
    return value


@triton.jit
def _track_level(images, points, result, status, width,
                 height, count, level: tl.constexpr,
                 first: tl.constexpr, reverse: tl.constexpr):
    point = tl.program_id(0) * 2
    ref = tl.program_id(1)
    pair = ref * count * 2 + point
    image0 = images + (ref + 1) * width * height if reverse else images
    image1 = images if reverse else images + (ref + 1) * width * height
    point += ref * count * 2 if reverse else 0
    px = tl.load(points + point) * (1. / (2 ** level))
    py = tl.load(points + point + 1) * (1. / (2 ** level))
    if first:
        nx, ny = px, py
    else:
        nx = tl.load(result + pair) * 2.
        ny = tl.load(result + pair + 1) * 2.
    offset = tl.arange(0, 512)
    valid = offset < 441
    ox, oy = offset % 21, offset // 21
    x, y, w00, w01, w10, w11 = _weights(px - 10., py - 10.)
    good = _inside(x, y, width, height)
    if good:
        patch_x, patch_y = x + ox, y + oy
        source = _intensity(image0, patch_x, patch_y, w00, w01, w10, w11, width, height, valid)
        dx00, dy00 = _scharr(image0, patch_x, patch_y, width, height, valid)
        dx01, dy01 = _scharr(image0, patch_x + 1, patch_y, width, height, valid)
        dx10, dy10 = _scharr(image0, patch_x, patch_y + 1, width, height, valid)
        dx11, dy11 = _scharr(image0, patch_x + 1, patch_y + 1, width, height, valid)
        dx = (dx00 * w00 + dx01 * w01 + dx10 * w10 + dx11 * w11 + 8192) >> 14
        dy = (dy00 * w00 + dy01 * w01 + dy10 * w10 + dy11 * w11 + 8192) >> 14
        scale: tl.constexpr = 1. / 1048576.
        a11 = _exact_sum(dx * dx, False) * scale
        a12 = _exact_sum(dx * dy, False) * scale
        a22 = _exact_sum(dy * dy, False) * scale
        determinant = a11 * a22 - a12 * a12
        difference = a11 - a22
        eig = (a11 + a22 - libdevice.sqrt(difference * difference + 4. * a12 * a12)) / 882.
        good = (eig >= 0.0001) & (determinant >= 1.1920928955078125e-7)
        if good:
            inv = tl.div_rn(1., determinant)
            qx, qy = nx - 10., ny - 10.
            last_dx, last_dy = 0., 0.
            iteration = 0
            converged = False
            while iteration < 30 and good and not converged:
                tx, ty, w00, w01, w10, w11 = _weights(qx, qy)
                good = _inside(tx, ty, width, height)
                if good:
                    target = _intensity(image1, tx + ox, ty + oy, w00, w01, w10, w11,
                                        width, height, valid)
                    diff = target - source
                    b1 = _exact_sum(diff * dx, True) * scale
                    b2 = _exact_sum(diff * dy, True) * scale
                    shift_x = (a12 * b2 - a22 * b1) * inv
                    shift_y = (a12 * b1 - a11 * b2) * inv
                    qx, qy = qx + shift_x, qy + shift_y
                    nx, ny = qx + 10., qy + 10.
                    sx64, sy64 = shift_x.to(tl.float64), shift_y.to(tl.float64)
                    converged = sx64 * sx64 + sy64 * sy64 <= 0.0001
                    oscillates = (iteration > 0 and not converged
                                  and tl.abs(shift_x + last_dx) < 0.01
                                  and tl.abs(shift_y + last_dy) < 0.01)
                    if oscillates:
                        nx, ny = nx - shift_x * 0.5, ny - shift_y * 0.5
                    converged |= oscillates
                    last_dx, last_dy = shift_x, shift_y
                iteration += 1
    tl.store(result + pair, nx)
    tl.store(result + pair + 1, ny)
    if level == 0:
        good &= _inside(tl.floor(nx - 10.), tl.floor(ny - 10.), width, height)
        tl.store(status + pair // 2, good)


@lru_cache(maxsize=1)
def _native_tracker(path):
    library = ctypes.CDLL(path)
    kernel = library.roi_lk_track_level
    kernel.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
    kernel.restype = ctypes.c_int
    return kernel


def track(gray, references, points, bidirectional=False):
    """逐参考图返回 OpenCV 形状的点坐标与有效标记。"""
    images = torch.from_numpy(np.stack([gray, *references])).to("cuda")
    pyramid = [images]
    count = len(points)
    while len(pyramid) < 4:
        n_images, height, width = pyramid[-1].shape
        out_h, out_w = (height + 1) // 2, (width + 1) // 2
        if min(out_h, out_w) <= 21:
            break
        output = torch.empty((n_images, out_h, out_w), dtype=torch.uint8, device=images.device)
        _downsample[(triton.cdiv(out_h * out_w, 128), n_images)](
            pyramid[-1], output, width, height, out_w, out_h, 128,
        )
        pyramid.append(output)
    start = torch.from_numpy(np.ascontiguousarray(points)).to(images.device)
    native_path = os.environ.get("ROI_CUDA_LK_LIBRARY")
    native = _native_tracker(native_path) if native_path else None

    def run(coords, reverse):
        result = torch.empty((len(references), count, 1, 2), device=images.device, dtype=torch.float32)
        status = torch.empty((len(references), count, 1), device=images.device, dtype=torch.uint8)
        for level, image in reversed(list(enumerate(pyramid))):
            _, height, width = image.shape
            first = level == len(pyramid) - 1
            if native is None:
                _track_level[(count, len(references))](
                    image, coords, result, status, width, height, count, level,
                    first, reverse, enable_fp_fusion=False,
                )
            else:
                error = native(
                    image.data_ptr(), coords.data_ptr(), result.data_ptr(), status.data_ptr(),
                    width, height, count, len(references), level, first, reverse,
                    torch.cuda.current_stream(images.device).cuda_stream,
                )
                if error:
                    raise RuntimeError(f"CUDA LK 启动失败（CUDA 错误码 {error}）")
        return result, status

    forward, valid = run(start, False)
    if not bidirectional:
        return forward.cpu().numpy(), valid.cpu().numpy()
    back, back_valid = run(forward, True)
    return forward.cpu().numpy(), valid.cpu().numpy(), back.cpu().numpy(), back_valid.cpu().numpy()
