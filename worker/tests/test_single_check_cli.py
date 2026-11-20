"""单块轮次仅标识版本，不限制已批准的新版本运行。"""
import sys
import pytest
from scripts import hls_single_check as single


def arguments(monkeypatch, iteration):
    monkeypatch.setattr(sys, 'argv', ['hls_single_check', '--source', 'source.mp4',
        '--prior', 'prior.npz', '--work', 'work', '--output', 'out',
        '--sha256', 'digest', '--iteration', str(iteration), '--roi-layered',
        '--roi-context-ratio', '.75', '--roi-min-context', '96'])


def test_new_iteration_preserves_explicit_roi_options(monkeypatch):
    arguments(monkeypatch, 5)
    received = []
    monkeypatch.setattr(single, 'execute_one', received.append)
    single.main()
    args = received[0]
    assert args.iteration == 5 and args.roi_layered
    assert args.roi_context_ratio == .75 and args.roi_min_context == 96


@pytest.mark.parametrize('iteration', [0, -1])
def test_nonpositive_iteration_rejected(monkeypatch, iteration):
    arguments(monkeypatch, iteration)
    monkeypatch.setattr(single, 'execute_one', lambda args: pytest.fail('无效轮次不可运行'))
    with pytest.raises(SystemExit) as error:
        single.main()
    assert error.value.code == 2
