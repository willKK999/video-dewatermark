"""视频去水印系统的通用HLS配置；默认值复用已接受的处理规格。"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class HLSSettings:
    block_seconds: float = 24
    context_seconds: float = 2
    model_fps: int = 30
    detail_sigma: float = 3
    video_crf: int = 10
    window_frames: int = 36
    overlap_frames: int = 8
    overlap_mae_limit: float = 35
    mask_fraction_limit: float = 0.2
    residual_correlation: float = 0.35
    residual_amplitude: float = 0.15
    clarity_record_threshold: float = 0.7
    panel_neighbor_reach: float = 1.5
    panel_neighbor_gap: float = 0.35
    panel_neighbor_min_height: float = 0.45
    panel_neighbor_max_height: float = 1.4
    panel_neighbor_bright_rim: float = 0.25
    panel_neighbor_dark_fill: float = 0.2

    @classmethod
    def from_env(cls):
        values = {}
        for name, item in cls.__dataclass_fields__.items():
            value = os.getenv("HLS_" + name.upper())
            if value is not None:
                values[name] = item.type(value)
        settings = cls(**values)
        if (
            settings.block_seconds <= 0
            or settings.context_seconds < 0
            or settings.block_seconds + 2 * settings.context_seconds > 30
        ):
            raise ValueError("处理块及上下文必须在30秒以内")
        if not 0 < settings.overlap_frames < settings.window_frames:
            raise ValueError("模型窗口与重叠配置无效")
        if not 0 < settings.mask_fraction_limit <= 0.2:
            raise ValueError("单遍掩膜保护不能超过20%")
        if (
            not 0 < settings.residual_correlation <= 0.35
            or not 0 < settings.residual_amplitude <= 0.15
        ):
            raise ValueError("残留复检配置无效")
        if not 0 < settings.overlap_mae_limit <= 35:
            raise ValueError("窗口差异保护不能放宽")
        if settings.model_fps != 30:
            raise ValueError("当前模型配置使用30帧采样")
        if settings.clarity_record_threshold != 0.7:
            raise ValueError("清晰度单列阈值固定为0.7，且不参与否决")
        if not (
            0 < settings.panel_neighbor_reach <= 2
            and 0 < settings.panel_neighbor_gap <= .5
            and 0 < settings.panel_neighbor_min_height < 1
            and 1 <= settings.panel_neighbor_max_height <= 2
            and 0 < settings.panel_neighbor_bright_rim <= 1
            and 0 < settings.panel_neighbor_dark_fill <= 1
        ):
            raise ValueError("相邻面板几何配置无效")
        return settings


def reusable_profile(path, source_digest, source_fps):
    """按内容而非片名选择可复用缓存；其他输入走自动检测的常规流程。"""
    import json
    from pathlib import Path

    if not path or abs(source_fps - 60) > 0.001:
        return False
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return data.get("method") == "v7" and data.get("source_sha256") == source_digest


def is_residual_score(point):
    settings = HLSSettings.from_env()
    return (
        abs(point["correlation"]) >= settings.residual_correlation
        and abs(point["relative_amplitude"]) >= settings.residual_amplitude
    )
