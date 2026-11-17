"""已有历史产物的只读编码诊断；仅向标准输出报告。"""
from pathlib import Path
import cv2,numpy as np,json,time,subprocess
import argparse
p=argparse.ArgumentParser(description='只读核验已有原生合成与编码视频，不写远端文件');p.add_argument('--run',required=True,type=Path);p.add_argument('--masks',required=True,type=Path);args=p.parse_args();out=args.run;root=args.masks;start=time.monotonic();rows=[];shape=cv2.imread(str(root/'frames/000000.png')).shape;frames=len(json.loads((out/'input-binding.json').read_text())['frames']);video=Path(json.loads((out/'render.json').read_text())['视频']['文件'])
p=subprocess.Popen(['timeout','500','ffmpeg','-v','error','-nostdin','-threads','1','-i',str(video),'-an','-frames:v',str(frames),'-vsync','0','-f','rawvideo','-pix_fmt','bgr24','-threads','1','pipe:1'],stdout=subprocess.PIPE)
def blue(x):
 h=cv2.cvtColor(x,cv2.COLOR_BGR2HSV);return (h[:,:,0]>=85)&(h[:,:,0]<=130)&(h[:,:,1]>=100)&(h[:,:,2]>=80)
try:
 for i in range(frames):
  data=p.stdout.read(int(np.prod(shape)))
  if len(data)!=int(np.prod(shape)):raise RuntimeError('编码视频不完整')
  encoded=np.frombuffer(data,np.uint8).reshape(shape);raw=cv2.imread(str(out/'run/composite'/f'{i:06d}.png'));source=cv2.imread(str(root/'frames'/f'{i:06d}.png'));mask=cv2.imread(str(root/'masks'/f'{i:06d}.png'),0)>0
  if raw is None or source is None:raise RuntimeError('原生合成或冻结源帧不可用')
  delta=np.abs(encoded.astype(np.int16)-raw);raw_blue=blue(raw);enc_blue=blue(encoded)
  rows.append(dict(frame=i,encoding_mae=float(delta.mean()),encoding_p99=float(np.percentile(delta,99)),encoding_max=int(delta.max()),raw_blue_inside=int(np.count_nonzero(raw_blue&mask)),encoded_blue_inside=int(np.count_nonzero(enc_blue&mask)),both_blue_inside=int(np.count_nonzero(raw_blue&enc_blue&mask)),raw_outside_modified=int(np.count_nonzero(np.any(raw!=source,axis=2)&~mask))))
  if i%60==0:print(f'原生与编码只读核验 {i}/{frames}',flush=True)
 if p.wait(timeout=20):raise RuntimeError('编码视频解码退出失败')
finally:
 if p.poll() is None:p.kill();p.wait(timeout=10)
report=json.loads((out/'run/roi-report.json').read_text());artifacts=[]
for t in report['任务']:
 for a in t['尝试']:
  folder=out/f"run/roi-{t['任务']:04d}-attempt-{a['尝试']:02d}"
  artifacts.append(dict(task=t['任务'],attempt=a['尝试'],passed=a['通过'],input_frames=len(list((folder/'frames').glob('*.png'))),input_masks=len(list((folder/'masks').glob('*.png'))),model_outputs=len(list((folder/'out/frames/frames').glob('*.png')))))
print('RESULT_JSON='+json.dumps(dict(rows=rows,artifacts=artifacts,seconds=time.monotonic()-start),ensure_ascii=False),flush=True)
