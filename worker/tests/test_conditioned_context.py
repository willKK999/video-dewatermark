"""重叠上下文仅改变模型输入；真实输出仍受原35差异保护。"""
from pathlib import Path
import json
import cv2,numpy as np,pytest
from worker import vision
from worker.process import PipelineError


def execute(tmp_path,monkeypatch,conditioned,ignore_context=False):
    monkeypatch.setenv('HLS_WINDOW_FRAMES','4');monkeypatch.setenv('HLS_OVERLAP_FRAMES','2')
    monkeypatch.setenv('INPAINT_PYTHON','python');monkeypatch.setenv('PROPAINTER_ROOT',str(tmp_path))
    for name in ['frames','masks']:(tmp_path/name).mkdir()
    mask=np.zeros((64,64),np.uint8);mask[20:44,20:44]=255
    files=[]
    for i in range(6):
        path=tmp_path/'frames'/f'{i:06d}.png';cv2.imwrite(str(path),np.full((64,64,3),10,np.uint8));files.append(path)
        cv2.imwrite(str(tmp_path/'masks'/path.name),mask)
    class References:
        def __init__(self,*a):self.stats={};self.timings={}
        def restore(self,original,mask,repaired):return repaired
    monkeypatch.setattr(vision,'CleanReferences',References)
    calls=[]
    def fake_run(args,*a,**kw):
        frames=Path(args[args.index('--video')+1]);masks=Path(args[args.index('--mask')+1]);out=Path(args[args.index('--output')+1])/'frames/frames';out.mkdir(parents=True)
        calls.append([])
        for p in sorted(frames.glob('*.png')):
            im=cv2.imread(str(p));m=cv2.imread(str(masks/p.name),0)
            calls[-1].append((im.copy(),m.copy()))
            if ignore_context:im[mask>0]=240 if len(calls)>1 else 160
            else:im[m>0]=240 if len(calls)>1 else 160
            cv2.imwrite(str(out/p.name),im)
    monkeypatch.setattr(vision,'run',fake_run)
    vision.inpaint(tmp_path,files,[(0,6)],'propainter',True,condition_overlap=conditioned)
    return calls


def test_context_uses_same_frame_previous_output_and_zero_mask(tmp_path,monkeypatch):
    calls=execute(tmp_path,monkeypatch,True)
    assert len(calls)==2
    for image,mask in calls[1][:2]:
        assert np.count_nonzero(mask)==0 and image[30,30,0]==160
    performance=json.loads((tmp_path/'performance.json').read_text())
    assert performance['conditioned_overlap'] is True
    assert performance['overlap_max_mae']<=35


@pytest.mark.parametrize('conditioned',[False,True])
def test_actual_disagreement_is_never_hidden_or_limit_relaxed(tmp_path,monkeypatch,conditioned):
    with pytest.raises(PipelineError) as caught:execute(tmp_path,monkeypatch,conditioned,ignore_context=True)
    assert caught.value.code=='SEAM_UNCERTAIN'
    event=json.loads((tmp_path/'overlap-failure.json').read_text())
    assert event['limit']==35 and event['disagreement_mae']>35
    assert event['conditioned_overlap'] is conditioned
