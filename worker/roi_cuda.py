"""逐层 ROI 的可选 CUDA 运算，保留 CPU 路径供逐项对照。"""
import os
from functools import lru_cache


@lru_cache(maxsize=1)
def enabled():
    return os.environ.get("ROI_CUDA") == "1"


def donor_consensus(values):
    """供体为 8 位像素或 NaN，调用方保证每像素至少三个有效供体。"""
    import torch

    samples = torch.from_numpy(values).to("cuda")
    valid = ~samples.isnan()
    count = valid.sum(dim=0)
    ordered = samples.nan_to_num(nan=256).sort(dim=0).values
    lower = ordered.gather(0, ((count - 1) // 2).unsqueeze(0)).squeeze(0)
    upper = ordered.gather(0, (count // 2).unsqueeze(0)).squeeze(0)
    median = (lower + upper) * 0.5
    maximum = samples.nan_to_num(nan=-1).amax(dim=0)
    agreement = ((maximum - median <= 10) & (median - ordered[0] <= 10)).all(dim=1)
    return median.cpu().numpy(), agreement.cpu().numpy()


def box_count(values, size):
    """用整数积分图统计零填充窗口，最后返回浮点计数供原判定使用。"""
    import torch
    from torch.nn.functional import pad

    radius = size // 2
    samples = torch.from_numpy(values).to("cuda", dtype=torch.int32)
    integral = pad(samples, (radius + 1, radius, radius + 1, radius))
    integral = integral.cumsum(0, dtype=torch.int32).cumsum(1, dtype=torch.int32)
    totals = (
        integral[size:, size:] - integral[:-size, size:]
        - integral[size:, :-size] + integral[:-size, :-size]
    )
    return totals.to(dtype=torch.float32).cpu().numpy()
