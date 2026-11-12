"""续修输入必须来自当前时间轴，并拒绝缺帧成品。"""
import cv2,numpy as np,pytest
from worker.hls_model_frames import capture


def test_current_video_mapping_keeps_first_corresponding_native_frame(tmp_path,monkeypatch):
    frames=[np.full((24,32,3),n*20,np.uint8) for n in range(6)]
    class Reader:
        def __init__(self,*args):self.frames=iter(frames)
        def read(self):
            image=next(self.frames,None)
            return image is not None,image
        def release(self):pass
    monkeypatch.setattr(cv2,'VideoCapture',Reader)
    output=tmp_path/'model'
    assert capture('current.mp4',[0,1,1,2,2,3],output,[])==[]
    assert [cv2.imread(str(output/f'{i:06d}.png'))[0,0,0] for i in range(4)]==[0,20,60,100]


def test_truncated_current_video_is_rejected(tmp_path,monkeypatch):
    class Empty:
        def __init__(self,*args):pass
        def read(self):return False,None
        def release(self):pass
    monkeypatch.setattr(cv2,'VideoCapture',Empty)
    with pytest.raises(ValueError):capture('broken.mp4',[0],tmp_path/'model',[])
