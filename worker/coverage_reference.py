"""复用同一输入经模型验证的历史掩膜；摘要不同即拒绝，不把缓存变成固定坐标规则。"""

try:
    from worker.json_types import json_native
except ModuleNotFoundError:
    from json_types import json_native
import hashlib,json,shutil
from pathlib import Path
import cv2,numpy as np
from .process import PipelineError
from .hls_layers import VerifiedLayers
from .model_tracks import patch,learn_opacity


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(work,reference,detected,progress=None):
    work=Path(work);reference=Path(reference)
    current=json.loads((work/'frames-binding.json').read_text());prior=json.loads((reference/'frames-binding.json').read_text())
    if current!=prior:raise PipelineError('SOURCE_CHANGED','覆盖参考与当前源摘要、帧范围或逐帧摘要不一致')
    summary=json.loads((reference/'model-mask-summary.json').read_text())
    binding=json.loads((reference/'model-prepared.json').read_text())
    if binding['detected_sha256']!=sha(reference/'detections.json'):raise PipelineError('SOURCE_CHANGED','历史检测摘要不一致')
    for name,digest in binding['assets'].items():
        if sha(reference/name)!=digest:raise PipelineError('SOURCE_CHANGED','历史模板或提示摘要不一致')
    for row in summary['逐帧']:
        if sha(reference/row['mask_key'])!=row['sha256']:raise PipelineError('MASK_INVALID','历史联合掩膜摘要发生变化')
    old=json.loads((reference/'detections.json').read_text());verified=VerifiedLayers(reference/'layers.json',reference/'detections.json')
    for row in summary['逐帧']:verified.validate(row['frame_index'],cv2.imread(str(reference/row['mask_key']),0))
    next_id=max((t['id'] for t in detected['tracks']),default=0)+1;precomputed=[]
    for old_track in old['tracks']:
        identity=next_id;next_id+=1;track=dict(old_track,id=identity,coverage_reference=True)
        track['hints']=[dict(p,track_id=identity) for p in old_track['hints']]
        for p in track['hints']:p.pop('mask_prompt',None)
        expected=binding['assets']['templates/'+Path(old_track['template']).name]
        if sha(old_track['template'])!=expected:raise PipelineError('SOURCE_CHANGED','历史实际模板路径摘要不一致')
        data=dict(np.load(old_track['template'],allow_pickle=False));base=data['mask'];size=(base.shape[1],base.shape[0])
        selected=np.unique(np.linspace(0,len(track['hints'])-1,min(15,len(track['hints'])),dtype=int))
        images=[patch(cv2.imread(str(work/'frames'/f'{track["hints"][k]["frame"]:06d}.png')),track['hints'][k]['box'],size) for k in selected]
        alpha=learn_opacity(images,base);data['alpha']=alpha;data['mask_alpha']=alpha
        if len(images)<15:
            saturation=cv2.cvtColor(np.median(np.stack(images),axis=0).astype(np.uint8),cv2.COLOR_BGR2HSV)[:,:,1]
            if np.median(saturation[base>0])<64:data['mask_alpha']=np.full(base.shape,.3,np.float32)
        template=work/'templates'/f'reference-{identity}.npz';np.savez_compressed(template,**data);track['template']=str(template)
        directory=work/'object-masks'/str(identity);directory.mkdir(parents=True)
        for p in track['hints']:
            name=f'{p["frame"]:06d}.png';source=reference/'validated-objects'/str(old_track['id'])/name
            if not source.exists():raise PipelineError('MASK_INVALID','历史轨迹掩膜不完整，禁止当作有效覆盖')
            shutil.copy2(source,directory/name)
        detected['tracks'].append(track);detected['hints'].extend(track['hints']);precomputed.append(identity)
        if progress and (len(precomputed)%5==0 or len(precomputed)==len(old['tracks'])):
            progress(f'同源基线掩膜已复核并合并 {len(precomputed)}/{len(old["tracks"])} 条轨迹')
    detected['hints'].sort(key=lambda p:(p['frame'],p['track_id']))
    detected['precomputed_track_ids']=precomputed
    detected['coverage_reference']=dict(源摘要=current['source']['source_sha256'],检测摘要=sha(reference/'detections.json'),
                                      联合掩膜摘要=sha(reference/'model-mask-summary.json'),轨迹数=len(precomputed),逐帧摘要=[p['sha256'] for p in summary['逐帧']])
    (work/'detections.json').write_text(json.dumps(json_native(detected)))
    (work/'coverage-reference.json').write_text(json.dumps(json_native(detected['coverage_reference']),ensure_ascii=False,indent=2))
    return detected


def verify_coverage(work,reference):
    work=Path(work);reference=Path(reference);rows=[]
    for path in sorted((reference/'masks').glob('*.png')):
        old=cv2.imread(str(path),0);new=cv2.imread(str(work/'masks'/path.name),0)
        if new is None or old.shape!=new.shape:raise PipelineError('COVERAGE_REGRESSION','基线覆盖核验帧缺失或几何不一致')
        missed=int(np.count_nonzero((old>0)&(new==0)))
        rows.append(dict(帧=int(path.stem),基线像素=int(np.count_nonzero(old)),本轮像素=int(np.count_nonzero(new)),缺失像素=missed))
    report=dict(逐帧=rows,缺失像素总数=sum(r['缺失像素'] for r in rows),通过=all(r['缺失像素']==0 for r in rows))
    (work/'coverage-comparison.json').write_text(json.dumps(json_native(report),ensure_ascii=False,indent=2))
    if not report['通过']:raise PipelineError('COVERAGE_REGRESSION','本轮丢失已验证的同源基线水印覆盖',details=report)
    return report
