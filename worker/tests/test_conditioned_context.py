"""旧的零掩膜重叠路径禁止绕过可信上下文契约。"""
import pytest
from worker import vision
from worker.process import PipelineError


@pytest.mark.parametrize('conditioned',[False,True])
@pytest.mark.parametrize('references',[None,'union'])
def test_legacy_context_is_explicitly_disabled(tmp_path,conditioned,references):
    with pytest.raises(PipelineError) as caught:
        vision.inpaint(tmp_path,[],[], 'propainter',condition_overlap=conditioned,reference_masks=references)
    assert caught.value.code=='UNTRUSTED_CONTEXT'
    assert not (tmp_path/'composite').exists()
