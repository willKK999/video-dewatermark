"""最终局部合成须保持原时间映射和掩膜外像素。"""
import subprocess
import cv2,numpy as np
from worker.hls_patch_video import blend


def test_lossless_blend_preserves_unmasked_pixels_and_native_frame_count(tmp_path):
    rng=np.random.default_rng(44)
    frames=rng.integers(60,180,(4,48,64,3),dtype=np.uint8)
    video=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','64x48','-r','30','-i','-','-c:v','libx264rgb','-crf','0',str(video)],input=frames.tobytes(),check=True,timeout=20)
    masks=tmp_path/'masks';masks.mkdir();paint=tmp_path/'paint';paint.mkdir()
    mask=np.zeros((48,64),np.uint8);mask[12:36,18:46]=255
    for i in range(2):
        cv2.imwrite(str(masks/f'{i:06d}.png'),mask)
        cv2.imwrite(str(paint/f'{i:06d}.png'),np.full((48,64,3),100+i*20,np.uint8))
    output=blend(video,video,[0,0,1,1],masks,paint,tmp_path/'result.mp4')
    cap=cv2.VideoCapture(str(output));decoded=[]
    while True:
        ok,image=cap.read()
        if not ok:break
        decoded.append(image)
    cap.release()
    assert len(decoded)==4
    for n,image in enumerate(decoded):
        assert np.array_equal(image[mask==0],frames[n][mask==0])
        assert np.all(image[20:28,25:35]==100+(n//2)*20)
