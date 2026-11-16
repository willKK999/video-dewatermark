from dataclasses import asdict
import pytest
from scripts.frozen_mask_inpaint import frozen_profile,RecoveryBudget
from worker.roi_inpaint import ROISettings
from worker.hls_settings import HLSSettings
from worker.process import PipelineError


def historical():
    return dict(ROI={'配置':asdict(ROISettings(context_ratio=.75,minimum_context=96,sequential_layers=True))},
                模型参数={'窗口':36,'重叠':8,'降分辨率':False},联合纹理投影=None)


def test_profile_preserves_historical_roi(monkeypatch):
    monkeypatch.setattr(HLSSettings,'from_env',lambda:HLSSettings())
    cfg=frozen_profile(historical())
    assert cfg.context_ratio==.75 and cfg.minimum_context==96
    assert cfg.fp16 and cfg.seam_retries==3 and cfg.sequential_layers


@pytest.mark.parametrize('override',[dict(overlap_mae_limit=36),dict(mask_fraction_limit=.21),dict(window_frames=18)])
def test_profile_rejects_changed_limits_or_window(monkeypatch,override):
    monkeypatch.setattr(HLSSettings,'from_env',lambda:HLSSettings(**override))
    with pytest.raises(ValueError):frozen_profile(historical())


def test_recovery_is_per_task_and_stops_after_one_retry():
    budget=RecoveryBudget()
    budget.check('逐层ROI 任务=1 尝试=0 参数=0:36/36 通过=False 错误码=SEAM_UNCERTAIN 耗时秒=2')
    budget.check('逐层ROI 任务=2 尝试=0 参数=0:36/36 通过=False 错误码=GPU_OOM 耗时秒=2')
    budget.check('逐层ROI 任务=1 尝试=1 参数=0:36/36 通过=True 错误码=无 耗时秒=2')
    with pytest.raises(PipelineError,match='一次通用恢复'):
        budget.check('逐层ROI 任务=2 尝试=1 参数=0:36/18 通过=False 错误码=GPU_OOM 耗时秒=2')


