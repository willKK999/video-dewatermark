"""稳定联合层的内容绑定续跑：复用完整前缀，不修改旧检查点。"""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,os,shutil,time
import cv2,numpy as np
from .process import PipelineError
from .restoration import CleanReferences
from .edge_residue import measure
from .roi_spatial_layers import protection_mask,residue_summary,residue_error
from .trusted_context import read_confirmed
from . import roi_inpaint as roi


def _read(path):return json.loads(Path(path).read_text())
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _link(source,dest):
    try:os.link(source,dest)
    except OSError:shutil.copy2(source,dest)


def prepare(work,old,files,cfg,settings,spatial,layers,log):
    old=Path(old).resolve()
    if old==work.resolve() or len(layers)!=1:raise PipelineError('CHECKPOINT_INCOMPATIBLE','续跑仅支持独立目录中的稳定联合层')
    for name in ['layers.json','detections.json','spatial-roi.json','trusted-context.json']:
        if _sha(old/name)!=_sha(work/name):raise PipelineError('CHECKPOINT_INCOMPATIBLE','续跑输入或模型保护配置与检查点不符')
    prior=_read(old.parent/'profile.json')
    if prior['ROI']!=asdict(cfg) or prior['窗口']!=settings.window_frames or prior['重叠']!=settings.overlap_frames or prior['接缝上限']!=settings.overlap_mae_limit or prior['每层面积上限']!=settings.mask_fraction_limit:
        raise PipelineError('CHECKPOINT_INCOMPATIBLE','续跑模型、接缝或面积配置不同')
    if _read(old.parent/'input-binding.json')!=_read(work.parent/'input-binding.json'):
        raise PipelineError('CHECKPOINT_INCOMPATIBLE','续跑冻结源帧或联合掩膜摘要不同')
    report=_read(old/'roi-report.json');prefix=[]
    for row in report['任务']:
        if not (row.get('通过') or row.get('采用未过结果')):break
        if row['任务']!=len(prefix)+1 or row['图层']!=0:raise PipelineError('CHECKPOINT_INCOMPATIBLE','检查点不是完整连续单层前缀')
        prefix.append(row)
    if not prefix:raise PipelineError('CHECKPOINT_INCOMPATIBLE','没有可复用的完整模型任务')
    expected={};plan=work/'resume-plan.json'
    # 调用方仍会按当前内容重建计划逐项比对；不能仅凭任务编号接纳缓存。
    for row in prefix:expected[row['任务']]=dict(开始帧=row['开始帧'],结束帧=row['结束帧'],ROI=row['ROI'],施工面积像素=row['施工面积像素'])
    dest=work/'roi-layer-00';dest.mkdir();coverage=work/'roi-coverage';coverage.mkdir();done={};digests=[]
    for i,p in enumerate(files):
        native=old/'roi-layer-00'/p.name;covered=old/'roi-coverage'/p.name
        image=cv2.imread(str(native));source=cv2.imread(str(p));mask=cv2.imread(str(covered),0);confirmed=cv2.imread(str(work/'masks'/p.name),0)
        if image is None or mask is None or confirmed is None or image.shape!=source.shape or mask.shape!=source.shape[:2] or not set(np.unique(mask))<={0,255}:
            raise PipelineError('CHECKPOINT_INCOMPATIBLE','检查点原生帧或覆盖掩膜缺失/无效')
        if np.any((mask>0)&(confirmed==0)) or np.any(np.any(image!=source,axis=2)&(mask==0)):
            raise PipelineError('CHECKPOINT_INCOMPATIBLE','检查点修改超出确认覆盖范围')
        _link(native,dest/p.name);_link(covered,coverage/p.name)
        if mask.any():done[i]=mask
        digests.append(dict(frame=i,native_sha256=_sha(native),coverage_sha256=_sha(covered)))
        if i%60==0:log(f'续跑检查点原生与覆盖核验 {i}/{len(files)}')
    plan.write_text(json.dumps(dict(来源=str(old),完整前缀数=len(prefix),计划=expected,帧=digests,说明='旧目录只读；新目录硬链接后以原子替换写入，逐任务重新检查颜色和背景证据。'),ensure_ascii=False,indent=2))
    return dict(rows=prefix,done=done,source=old,expected=expected)


def recheck(state,row,ownership,work,files,refs,box,start,end,edge_cfg,settings,log):
    number=row['任务'];old=state['rows'][number-1];expected=state['expected'][number]
    if any(row[k]!=v for k,v in expected.items()):raise PipelineError('CHECKPOINT_INCOMPATIBLE','当前内容派生ROI计划与旧检查点不一致')
    a,b,c,d=box
    references=CleanReferences([roi.crop(cv2.imread(str(files[j])),box) for j in refs],
                               [roi.crop(cv2.imread(str(work/'masks'/files[j].name),0),box,True) for j in refs])
    scores=[];begin=time.monotonic()
    for j in range(start,end):
        confirmed=read_confirmed(work/'masks'/files[j].name);active=(confirmed&ownership)[b:d,a:c]
        if not active.any():continue
        covered=state['done'].get(j)
        if covered is None or np.any((active>0)&(covered[b:d,a:c]==0)):
            raise PipelineError('CHECKPOINT_INCOMPATIBLE','旧任务有未完成施工像素，禁止用原帧伪装完整检查点')
        source=roi.crop(cv2.imread(str(files[j])),box);mask=roi.crop(confirmed,box,True)
        evidence={key:(value[:d-b,:c-a] if hasattr(value,'shape') else value) for key,value in references.reference_evidence(source,mask,confirmed=mask).items()}
        image=cv2.imread(str(work/'roi-layer-00'/files[j].name))[b:d,a:c]
        band_path=work/'edge-band'/files[j].name;band=(read_confirmed(band_path)[b:d,a:c]&active) if band_path.exists() else np.zeros(active.shape,np.uint8)
        protected=protection_mask(work,files[j].name,confirmed.shape)[b:d,a:c]
        scores.append(dict(frame=j,**measure(source[:d-b,:c-a],image,active,band,edge_cfg,reference_evidence=evidence,confirmed=confirmed[b:d,a:c],protected=protected)))
        if (j-start)%20==0:log(f'续跑旧任务={number} 只读复检帧={j}，未运行模型')
    chosen=next((t for t in old['尝试'] if t.get('通过')),None)
    if chosen is None:chosen=next((t for t in old['尝试'] if t['尝试']==old.get('最佳尝试')),None)
    if chosen is None or not scores:raise PipelineError('CHECKPOINT_INCOMPATIBLE','完整前缀缺少已采用尝试或非空施工样本')
    seam=chosen['重叠最大差异'];summary=residue_summary(scores);failure=residue_error(scores)
    passed=seam is not None and seam<=settings.overlap_mae_limit and failure is None
    row.update(尝试=old['尝试'],原运行通过=old['通过'],原运行未过=old.get('未过块',False),复用检查点=True,复用来源=str(state['source']),
               复用后复检=scores,残留汇总=summary,残留判定=summary['判定'],复检耗时秒=time.monotonic()-begin,历史补画耗时秒=old['耗时秒'],耗时秒=time.monotonic()-begin,
               重叠最大差异=seam,通过=bool(passed),原检查记录保留=True)
    if not passed:
        code='SEAM_UNCERTAIN' if seam is None or seam>settings.overlap_mae_limit else failure.code
        row.update(未过块=True,采用未过结果=True,错误码=code)
        if failure is not None:
            row['待目检说明']=('确认残留：背景证据表明该处背景不是水印主色，需补修；颜色问题换接缝窗口无效，不重试' if failure.code=='RESIDUE_FOUND'
                             else '待目检：主色候选或无模板组缺少可判背景证据，需人工目检，不算通过；颜色问题换接缝窗口无效，不重试')
    return row
