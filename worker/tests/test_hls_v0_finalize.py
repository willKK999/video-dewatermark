"""跨磁盘续封装必须保留原片规格、完整时间轴和压缩音频。"""

from copy import deepcopy
import pytest
from scripts.hls_v0_finalize import technical_check


def info(height=1080, fps=60):
    return {"streams": [dict(codec_type="audio"), dict(codec_type="video", width=height*16//9, height=height, avg_frame_rate=str(fps), nb_frames=str(2*fps), duration="2")]}


def test_complete_copied_media_accepts_audio_first_stream_order():
    assert technical_check(info(), info(), info(720, 30), 120, True)


@pytest.mark.parametrize("field,value", [("nb_frames", "119"), ("avg_frame_rate", "30"), ("width", 1280), ("duration", "1.9")])
def test_truncated_or_changed_result_cannot_be_reported_complete(field, value):
    result = deepcopy(info())
    result["streams"][1][field] = value
    assert not technical_check(info(), result, info(720, 30), 120, True)


def test_reencoded_or_missing_audio_cannot_be_reported_copied():
    assert not technical_check(info(), info(), info(720, 30), 120, False)
