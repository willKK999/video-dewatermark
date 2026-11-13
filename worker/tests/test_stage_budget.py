"""预算随真实工作量增长，旧调用不改变时限或模型参数，失败不得降级原帧。"""
import inspect,json
import pytest
from worker.process import run,PipelineError
from worker.stage_budget import sam_budget,BudgetSettings


def detections(count=30,tracks=1,fragmented=False):
    hints=[dict(frame=i,track_id=t) for t in range(tracks) for i in range(count)
           if not fragmented or i>=t]
    return dict(multitrack=True,scenes=[[0,count]],hints=hints)


def test_budget_is_monotonic_in_objects_resolution_and_initializations():
    a=sam_budget(detections(300,2),300,1280*720)
    b=sam_budget(detections(900,20),900,1280*720)
    c=sam_budget(detections(900,20,True),900,1920*1080)
    assert a['timeout_seconds']<b['timeout_seconds']<=c['timeout_seconds']
    assert c['workload']['initializations']>b['workload']['initializations']
    assert b['workload']['boxes']==18000


def test_budget_bounds_and_empty_work():
    cfg=BudgetSettings(minimum_seconds=15,maximum_seconds=30)
    result=sam_budget(detections(100,10),100,3840*2160,cfg)
    assert result['timeout_seconds']==30 and result['upper_bound_reached']
    assert sam_budget(dict(scenes=[],hints=[]),0,1)['timeout_seconds']==900
    with pytest.raises(ValueError):BudgetSettings(minimum_seconds=31,maximum_seconds=30)


def test_legacy_run_limit_and_success_are_unchanged():
    assert inspect.signature(run).parameters['timeout'].default==900
    assert run(['sh','-c','printf valid'])==b'valid'


def test_timeout_carries_stage_budget_and_workload_without_command_secrets(tmp_path):
    path=tmp_path/'event.json'
    with pytest.raises(PipelineError) as caught:
        run(['sh','-c','sleep 5'],timeout=.03,stage='mask',event_path=path,workload={'tracks':3})
    assert caught.value.code=='PROCESS_TIMEOUT'
    assert caught.value.details['stage']=='mask'
    assert caught.value.details['timeout_seconds']==.03
    event=json.loads(path.read_text())
    assert event['status']=='failed' and event['workload']=={'tracks':3}
    assert 'args' not in event and 'environment' not in event


def test_legacy_sam_call_does_not_enable_new_budget_or_detector(tmp_path,monkeypatch):
    from worker import vision
    calls=[]
    monkeypatch.setenv('TRACK_PYTHON','python')
    monkeypatch.setattr(vision,'run',lambda *a,**k:calls.append((a,k)))
    monkeypatch.setattr(vision,'validate_masks',lambda *a:[])
    root=tmp_path/'legacy';root.mkdir()
    vision.masks(root,[],dict(scenes=[],hints=[]),'propainter')
    assert calls[0][1]=={}
    assert not (root/'process-mask.json').exists()
    assert not (root/'mask-budget.json').exists()


def test_opt_in_sam_call_receives_exact_workload_budget(tmp_path,monkeypatch):
    from worker import vision
    calls=[]
    monkeypatch.setenv('TRACK_PYTHON','python')
    monkeypatch.setattr(vision,'run',lambda *a,**k:calls.append(k))
    monkeypatch.setattr(vision,'validate_masks',lambda *a:[])
    root=tmp_path/'new';root.mkdir()
    budget=sam_budget(detections(900,20),900,1280*720)
    vision.masks(root,[],dict(scenes=[],hints=[]),'propainter',budget=budget)
    assert calls[0]['timeout']==budget['timeout_seconds']
    assert calls[0]['stage']=='mask' and calls[0]['workload']==budget['workload']
