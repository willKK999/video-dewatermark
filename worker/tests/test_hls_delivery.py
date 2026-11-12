import pytest
from worker.hls_delivery import best_version


def candidate(name, residual, **kw):
    return dict(版本=name, 完整=True, 残留=residual, 低频残留=0, **kw)


def test_select_residual_without_using_clarity():
    low = candidate("较少残留", 1, 清晰度最低值=0.1)
    high = candidate("较多残留", 2, 清晰度最低值=0.9)
    assert best_version([high, low]) is low


def test_confirmed_flash_is_not_preferred():
    flashing = candidate("闪烁", 0, 闪烁=True)
    clean = candidate("未闪烁", 1, 闪烁=False)
    assert best_version([flashing, clean]) is clean


def test_unknown_does_not_beat_measured_and_incomplete_is_excluded():
    incomplete = dict(candidate("不完整", 0), 完整=False)
    unknown = candidate("未知", 1)
    measured = candidate("已测", 1, 闪烁=False)
    assert best_version([incomplete, unknown, measured]) is measured
    with pytest.raises(ValueError):
        best_version([incomplete])
