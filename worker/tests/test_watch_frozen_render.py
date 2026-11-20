import json
import pytest
from scripts.watch_frozen_render import state


def test_never_started_is_immediate_error(tmp_path):
    with pytest.raises(RuntimeError,match='未启动'):state(tmp_path)


def test_exited_without_result_is_error(tmp_path,monkeypatch):
    (tmp_path/'worker-pid.json').write_text(json.dumps(dict(pid=1234)))
    def gone(*args):raise ProcessLookupError
    monkeypatch.setattr('scripts.watch_frozen_render.os.kill',gone)
    with pytest.raises(RuntimeError,match='已退出'):state(tmp_path)


@pytest.mark.parametrize('file,expected',[('failure.json','failed'),('render.json','rendered')])
def test_result_terminates_even_without_pid(tmp_path,file,expected):
    (tmp_path/file).write_text('{}')
    assert state(tmp_path)==expected
