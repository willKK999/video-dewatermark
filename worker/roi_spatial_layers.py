"""空间连通ROI：冻结前层、独立施工域和上下文接缝核验。"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from multiprocessing import shared_memory
from pathlib import Path
from threading import Event
import hashlib,json,multiprocessing as mp,os,shutil,time
import cv2
import numpy as np
from . import edge_residue as _edge_residue_mod, roi_inpaint as roi, trusted_context as _trusted_context_mod
from .hls_settings import HLSSettings
from .json_types import json_native
from .process import PipelineError,run
from .restoration import CleanReferences,_build_donor_evaluator,restore_orthogonal_luma_detail,soft_composite
from .vision import windows
from .spatial_roi import SpatialSettings,plan,padded_area
from .roi_attempt_cache import compact,restore


def _available_cpus() -> int:
    """获取当前进程/容器 cgroup 实际可用的 CPU 核数。"""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    v2 = Path('/sys/fs/cgroup/cpu.max')
    if v2.exists():
        parts = v2.read_text().strip().split()
        if len(parts) == 2 and parts[0] != 'max':
            quota, period = int(parts[0]), int(parts[1])
            if quota > 0 and period > 0:
                count = min(count, max(1, quota // period))
    else:
        q_path = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        p_path = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if q_path.exists() and p_path.exists():
            quota, period = int(q_path.read_text().strip()), int(p_path.read_text().strip())
            if quota > 0 and period > 0:
                count = min(count, max(1, quota // period))
    return max(1, count)


def _cpu_workers() -> int:
    """读取 ROI_CPU_WORKERS（=1 退化为串行路径；默认 min(可用核数, 12)）。"""
    raw = os.environ.get('ROI_CPU_WORKERS', '').strip()
    if raw:
        return max(1, int(raw))
    return min(_available_cpus(), 12)


def _worker_init():
    """多进程工作器内固定 OpenCV 单线程，避免与进程池过度订阅。"""
    cv2.setNumThreads(1)


def _pack_bool(arr: np.ndarray) -> bytes:
    return np.packbits(np.ascontiguousarray(arr, dtype=bool).ravel(), bitorder='little').tobytes()


def _unpack_bool(packed: bytes, shape: tuple[int, int]) -> np.ndarray:
    n = shape[0] * shape[1]
    return np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder='little')[:n].reshape(shape).astype(bool)


class _LRUCache:
    """有界帧图像/掩膜缓存，避免同镜头多任务重复解码 PNG。"""

    def __init__(self, max_items: int = 160):
        self.max_items = max_items
        self._data: OrderedDict = OrderedDict()

    def get(self, key):
        val = self._data.get(key)
        if val is not None:
            self._data.move_to_end(key)
        return val

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()


def _model_stage(args,out_dir,key,**kw):
    """补画模型调用；验收夹具可用环境变量保存/复用模型原始输出（只做文件拷贝，不改变任何计算）。"""
    reuse=os.environ.get('ROI_MODEL_OUTPUT_REUSE');keep=os.environ.get('ROI_MODEL_OUTPUT_KEEP')
    out_dir=Path(out_dir)
    if reuse:
        src=Path(reuse)/key
        if not (src/'frames').is_dir():raise PipelineError('ROI_INCOMPLETE',f'复用模型输出缺失：{key}')
        (out_dir/'out/frames').mkdir(parents=True,exist_ok=True)
        shutil.copytree(src/'frames',out_dir/'out/frames/frames')
        if (src/'metrics.json').exists():shutil.copy2(src/'metrics.json',out_dir/'metrics.json')
        return
    run(args,**kw)
    if keep:
        dst=Path(keep)/key
        if dst.exists():shutil.rmtree(dst)
        dst.mkdir(parents=True)
        shutil.copytree(out_dir/'out/frames/frames',dst/'frames')
        if (out_dir/'metrics.json').exists():shutil.copy2(out_dir/'metrics.json',dst/'metrics.json')


def _scene_static_highpass(images, grays=None):
    """估算同镜头参考帧的静止水印高通成分，作为细节投影正交基底。"""
    if not images:
        return None
    hps=[]
    src_grays=grays if grays is not None else [cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in images]
    for g_u8 in src_grays:
        g=g_u8.astype(np.float32)
        hps.append(g-cv2.GaussianBlur(g,(0,0),3.0))
    return np.median(np.stack(hps),axis=0)


def _scene_track_highpass(detected,files,scene_start,scene_end,shape):
    """为同一镜头内的移动水印轨迹构建每帧高通笔画估计，防止正交细节恢复回注移动文字边缘。"""
    from .faint_watermark import _refine_core_anchor,_track_frame_boxes

    tracks=detected.get('tracks') or [] if isinstance(detected,dict) else []
    per_frame={}
    hp_cache={}
    def get_hp(fidx):
        if fidx not in hp_cache:
            g=cv2.imread(str(files[fidx]),0).astype(np.float32)
            hp_cache[fidx]=g-cv2.GaussianBlur(g,(0,0),3.0)
        return hp_cache[fidx]
    for tr in tracks:
        entries=[row for row in _track_frame_boxes(tr) if scene_start<=row[0]<scene_end]
        if len(entries)<12:
            continue
        mw=int(round(float(np.median([b[2]-b[0] for _,b in entries]))))
        mh=int(round(float(np.median([b[3]-b[1] for _,b in entries]))))
        if mw<24 or mh<12:
            continue
        pad=12;cw,ch=mw+2*pad,mh+2*pad
        s_idx=np.linspace(0,len(entries)-1,min(24,len(entries)),dtype=int)
        samples=[entries[int(q)] for q in s_idx]
        mf,mbox=samples[len(samples)//2]
        mhp=get_hp(mf);ih,iw=mhp.shape[:2]
        mbx=max(0,min(iw-1,int(round(mbox[0]))));mby=max(0,min(ih-1,int(round(mbox[1]))))
        core=mhp[mby:min(ih,mby+mh),mbx:min(iw,mbx+max(8,int(round(0.8*mw))))]
        stacks=[]
        for fidx,bx in samples:
            fhp=get_hp(fidx)
            ax,ay=_refine_core_anchor(fhp,bx,core,mw,mh,10,0.2)
            mat=np.array([[1.0,0.0,-(ax-pad)],[0.0,1.0,-(ay-pad)]],dtype=np.float32)
            stacks.append(cv2.warpAffine(fhp,mat,(cw,ch),flags=cv2.INTER_LINEAR))
        med=np.median(np.stack(stacks),axis=0)
        for fidx,bx in entries:
            fhp=get_hp(fidx)
            ax,ay=_refine_core_anchor(fhp,bx,core,mw,mh,10,0.2)
            inv=np.array([[1.0,0.0,ax-pad],[0.0,1.0,ay-pad]],dtype=np.float32)
            proj=cv2.warpAffine(med,inv,(shape[1],shape[0]),flags=cv2.INTER_LINEAR)
            prev=per_frame.get(fidx)
            per_frame[fidx]=proj if prev is None else np.where(np.abs(proj)>np.abs(prev),proj,prev)
    return per_frame


def _window_refs(scene_start,scene_end,start,end,base_refs,max_refs):
    """在场景均匀抽样基础上补充施工窗口邻近帧，便于运动背景下获取足够干净供体。"""
    if max_refs<=len(base_refs):
        return base_refs
    extra_budget=max_refs-len(base_refs)
    w0=max(scene_start,start-12);w1=min(scene_end,end+16)
    local_grid=np.linspace(w0,max(w0,w1-1),min(extra_budget*2,max(1,w1-w0)),dtype=int)
    merged=list(dict.fromkeys([int(x) for x in base_refs]+[int(x) for x in local_grid]))
    return np.array(sorted(merged[:max_refs]),dtype=int)


def _dense_frame_evidence(j,scene_start,scene_end,box,current,work,files,existing_refs,original,mask,confirmed,preloaded_refs=None,return_crops=False):
    """对存在未解释或确认残留候选的帧，按±18邻近帧补充密集干净参考复核，绝不捏造无证据背景。"""
    d0=max(scene_start,j-18);d1=min(scene_end,j+19)
    cand_ids=[idx for idx in range(d0,d1) if idx!=j]
    existing_set={int(x) for x in existing_refs}
    if len(cand_ids)<=len([idx for idx in existing_set if d0<=idx<d1]):
        return (None,[]) if return_crops else None
    if preloaded_refs is not None:
        dense_imgs=[];dense_tr=[];dense_gr=[];dense_msks=[]
        for idx in cand_ids:
            cached=preloaded_refs.get(idx)
            if cached is not None:
                im,tr,gr=cached
                dense_imgs.append(im);dense_tr.append(tr);dense_gr.append(gr);dense_msks.append(None)
            else:
                im=roi.crop(cv2.imread(str(current[idx])),box)
                mk=roi.crop(cv2.imread(str(work/'masks'/files[idx].name),0),box,True)
                dense_imgs.append(im);dense_msks.append(mk)
                dense_tr.append(255-cv2.dilate(mk,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))))
                dense_gr.append(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
        dense_refs=CleanReferences(dense_imgs,dense_msks,trusted=dense_tr,grays=dense_gr)
    else:
        dense_imgs=[roi.crop(cv2.imread(str(current[idx])),box) for idx in cand_ids]
        dense_msks=[roi.crop(cv2.imread(str(work/'masks'/files[idx].name),0),box,True) for idx in cand_ids]
        dense_refs=CleanReferences(dense_imgs,dense_msks)
    ev=dense_refs.reference_evidence(original,mask,confirmed=confirmed)
    return (ev,dense_refs.last_donor_crops) if return_crops else ev


def _repair_prep_worker(item):
    """确认残留补修帧级输入准备与掩膜核对。"""
    (j,left,box,shape,pixels,ctx_r,limit,in_path,mask_path_src,frame_dst,mask_dst,
     is_constr,in_completed,painted_packed,target_packed)=item
    a,b,c,d=box
    in_img=cv2.imread(in_path)
    roi.atomic_png(Path(frame_dst),roi.crop(in_img,box))
    full_conf=_trusted_context_mod.read_confirmed(mask_path_src)
    unpainted_conf=(full_conf>0)
    if is_constr or in_completed:
        painted_here=_unpack_bool(painted_packed,shape) if painted_packed is not None else np.zeros(shape,bool)
        unpainted_conf=unpainted_conf&(~painted_here)
    if target_packed is not None:
        unpainted_conf[b:d,a:c]|=_unpack_bool(target_packed,(d-b,c-a))
    excl_full=cv2.dilate(unpainted_conf.astype(np.uint8)*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*ctx_r+1,2*ctx_r+1)))
    if np.count_nonzero(excl_full)/pixels>limit:
        return ('LIMIT',j)
    roi.atomic_png(Path(mask_dst),roi.crop(excl_full,box,True))
    return ('OK',j)


def _repair_confirmed_residue_pass(
    task,work,files,current,dest,staged,baseline,completed,layer_done,
    masks,ownership,box,left,right,start,end,length,shape,pixels,
    edge_cfg,settings,cfg,trusted,policy,residue_scores,maximum,reference_holder,
    refs,detected,log,measure_fn,read_confirmed_fn,exclusion_fn,overlap_weight_fn,pool=None,
):
    """对判为RESIDUE_FOUND的尝试执行最多一轮有界确认残留补修，并严格重检残留与接缝。"""
    a,b,c,d=box
    res_targets={}
    before_summary=residue_summary(residue_scores)
    band_cache={}
    src_cache={}
    conf_cache={}
    prot_cache={}
    for sc in residue_scores:
        j=int(sc['frame'])
        if sc.get('判定')!='RESIDUE_FOUND':
            continue
        active=(masks[j][b:d,a:c]>0)&(ownership[b:d,a:c]>0)
        if not active.any():
            continue
        band_path=work/'edge-band'/files[j].name
        band=read_confirmed_fn(band_path)[b:d,a:c] if band_path.exists() else np.zeros(active.shape,np.uint8)
        band_cache[j]=band
        source_color=roi.crop(cv2.imread(str(files[j])),box)[:d-b,:c-a]
        src_cache[j]=source_color
        conf_crop=roi.crop(read_confirmed_fn(work/'masks'/files[j].name),box,True)[:d-b,:c-a]
        conf_cache[j]=conf_crop
        prot_crop=protection_mask(work,files[j].name,shape)[b:d,a:c]
        prot_cache[j]=prot_crop
        packed_residue = sc.get('_repair_residue')
        if packed_residue is not None:
            rm = _unpack_bool(packed_residue, active.shape)
        else:
            staged_img=cv2.imread(str(staged/files[j].name))
            if staged_img is None:
                continue
            active_mask=active.astype(np.uint8)*255
            full_sc=measure_fn(
                source_color,staged_img[b:d,a:c],active_mask,band&active_mask,edge_cfg,
                reference_evidence=sc.get('_evidence'),
                confirmed=conf_crop,
                protected=prot_crop,
                return_masks=True,
            )
            rm=full_sc.get('_masks',{}).get('确认残留')
        if rm is not None and np.any(rm>0):
            r_dil=edge_cfg.residue_repair_dilate_radius
            ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r_dil+1,2*r_dil+1))
            dilated=(cv2.dilate((rm>0).astype(np.uint8)*255,ker)>0)&active
            if np.any(dilated):
                res_targets[j]=dilated
    if not res_targets:
        return residue_scores,maximum,None

    rep_dir=task/'residue-repair-01'
    if rep_dir.exists():
        shutil.rmtree(rep_dir)
    (rep_dir/'frames').mkdir(parents=True);(rep_dir/'masks').mkdir(parents=True)
    ctx_r=trusted.boundary_exclusion_pixels if trusted else 3
    limit=trusted.mask_fraction_limit if trusted else settings.mask_fraction_limit
    total_target_px=sum(int(np.count_nonzero(m)) for m in res_targets.values())
    if pool is not None and right-left>1 and read_confirmed_fn is _ORIG_READ_CONFIRMED:
        items=[]
        for j in range(left,right):
            is_constr=start<=j<end
            if is_constr and (staged/files[j].name).exists():in_path=str(staged/files[j].name)
            elif j in completed:in_path=str(dest/files[j].name)
            else:in_path=str(current[j])
            done_j=layer_done.get(j)
            painted_packed=None
            if is_constr:painted_packed=_pack_bool((masks[j]>0)&(ownership>0))
            elif j in completed and done_j is not None:painted_packed=_pack_bool(done_j>0)
            target_packed=_pack_bool(res_targets[j]) if j in res_targets else None
            items.append((
                j,left,box,shape,pixels,ctx_r,limit,in_path,str(work/'masks'/files[j].name),
                str(rep_dir/'frames'/f'{j-left:06d}.png'),str(rep_dir/'masks'/f'{j-left:06d}.png'),
                is_constr,j in completed,painted_packed,target_packed,
            ))
        for status,_ in pool.imap(_repair_prep_worker,items):
            if status=='LIMIT':
                shutil.rmtree(rep_dir)
                return residue_scores,maximum,dict(跳过原因='补修排除掩膜接近20%保护上限',补修前残留=before_summary['确认残留'])
    else:
        for j in range(left,right):
            is_constr=start<=j<end
            if is_constr and (staged/files[j].name).exists():
                in_img=cv2.imread(str(staged/files[j].name))
            elif j in completed:
                in_img=cv2.imread(str(dest/files[j].name))
            else:
                in_img=cv2.imread(str(current[j]))
            roi.atomic_png(rep_dir/'frames'/f'{j-left:06d}.png',roi.crop(in_img,box))
            full_conf=read_confirmed_fn(work/'masks'/files[j].name)
            done_j=layer_done.get(j)
            # 对尚未完成施工的像素始终保持全确认排除，同时强制排除本轮确认残留补修目标
            unpainted_conf=(full_conf>0)
            if is_constr or j in completed:
                painted_here=(masks[j]>0)&(ownership>0) if is_constr else (done_j>0 if done_j is not None else np.zeros(shape,bool))
                unpainted_conf=unpainted_conf&(~painted_here)
            if j in res_targets:
                unpainted_conf[b:d,a:c]|=res_targets[j]
            excl_full=cv2.dilate(unpainted_conf.astype(np.uint8)*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*ctx_r+1,2*ctx_r+1)))
            if np.count_nonzero(excl_full)/pixels>limit:
                shutil.rmtree(rep_dir)
                return residue_scores,maximum,dict(跳过原因='补修排除掩膜接近20%保护上限',补修前残留=before_summary['确认残留'])
            roi.atomic_png(rep_dir/'masks'/f'{j-left:06d}.png',roi.crop(excl_full,box,True))

    singleton=(right-left)==1
    if singleton:
        for name in ['frames','masks']:
            shutil.copy2(rep_dir/name/'000000.png',rep_dir/name/'000001.png')
    args=[os.environ['INPAINT_PYTHON'],str(Path(__file__).with_name('torch_stage.py')),
          '--metrics',rep_dir/'metrics.json','--frames-only','--script',Path(os.environ['PROPAINTER_ROOT'])/'inference_propainter.py',
          '--video',rep_dir/'frames','--mask',rep_dir/'masks','--output',rep_dir/'out','--save_frames',
          '--mask_dilation','0','--subvideo_length',str(length),'--neighbor_length','6','--ref_stride','20']
    if cfg.fp16:
        args.append('--fp16')
    timeout=min(840,policy.get('model_timeout_seconds') or 840)
    _model_stage(args,rep_dir,task.name+'-'+rep_dir.name,cwd=os.environ['PROPAINTER_ROOT'],timeout=timeout,stage='逐层ROI-确认残留补修')
    predicted=sorted((rep_dir/'out/frames/frames').glob('*.png'))
    if len(predicted)!=(2 if singleton else right-left):
        shutil.rmtree(rep_dir)
        return residue_scores,maximum,dict(跳过原因='补修输出帧数不完整',补修前残留=before_summary['确认残留'])

    new_staged={};new_scores=[]
    new_max_seam=0.
    for j in range(left,right):
        if not (start<=j<end):
            continue
        prior_done=layer_done.get(j)
        shared=(prior_done[b:d,a:c]>0)&(masks[j][b:d,a:c]>0) if prior_done is not None else np.zeros((d-b,c-a),bool)
        if j not in res_targets and not shared.any():
            continue
        cur_staged=cv2.imread(str(staged/files[j].name))
        if cur_staged is None:
            continue
        if j in res_targets:
            rep2=cv2.imread(str(predicted[j-left]))
            if rep2 is None:
                continue
            cand_roi=cur_staged[b:d,a:c].copy()
            tmask=res_targets[j]
            cand_roi[tmask]=rep2[:d-b,:c-a][tmask]
            if j in completed:
                weight=overlap_weight_fn(j,start,end,completed)
                base_old=cv2.imread(str(dest/files[j].name))[b:d,a:c]
                mixed=cv2.addWeighted(cand_roi,weight,base_old,1-weight,0)
                cand_roi[tmask&shared]=mixed[tmask&shared]
            cur_staged[b:d,a:c]=cand_roi
            new_staged[j]=cur_staged
        if shared.any():
            base_old=cv2.imread(str(dest/files[j].name))[b:d,a:c]
            disagree=float(np.abs(base_old.astype(float)-cur_staged[b:d,a:c].astype(float))[shared].mean())
            new_max_seam=max(new_max_seam,disagree)

    if new_max_seam>settings.overlap_mae_limit:
        shutil.rmtree(rep_dir)
        return residue_scores,maximum,dict(
            补修前残留=before_summary['确认残留'],补修目标像素=total_target_px,
            补修后接缝=new_max_seam,采用补修=False,原因='补修后接缝超过上限，保留补修前暂存',
        )

    old_by_frame={int(s['frame']):s for s in residue_scores}
    for sc in residue_scores:
        j=int(sc['frame'])
        if j not in res_targets:
            new_scores.append(sc)
            continue
        active=(masks[j][b:d,a:c]>0)&(ownership[b:d,a:c]>0)
        active_mask=active.astype(np.uint8)*255
        band=band_cache[j];source_color=src_cache[j];ev=sc.get('_evidence')
        upd_img=new_staged[j][b:d,a:c]
        new_sc=measure_fn(
            source_color,upd_img,active_mask,band&active_mask,edge_cfg,
            reference_evidence=ev,
            confirmed=conf_cache[j],
            protected=prot_cache[j],
        )
        new_scores.append(dict(frame=j,**new_sc,_evidence=ev))

    after_summary=residue_summary(new_scores)
    improved=(after_summary['确认残留'],after_summary['待目检'])<(before_summary['确认残留'],before_summary['待目检'])
    if improved:
        for j,img_upd in new_staged.items():
            roi.atomic_png(staged/files[j].name,img_upd)
        res_out=new_scores;seam_out=max(maximum,new_max_seam)
    else:
        res_out=residue_scores;seam_out=maximum
    shutil.rmtree(rep_dir)
    info=dict(
        补修帧=sorted(res_targets.keys()),补修目标像素=total_target_px,
        补修前残留=before_summary['确认残留'],补修后残留=after_summary['确认残留'],
        补修前待目检=before_summary['待目检'],补修后待目检=after_summary['待目检'],
        补修后接缝=round(new_max_seam,4),采用补修=bool(improved),
    )
    log(f'逐层ROI 任务={task.name} 确认残留补修 补修前={before_summary["确认残留"]} 补修后={after_summary["确认残留"]} 采用={improved}')
    return res_out,seam_out,info


def seam_variants(start,end,scene_start,scene_end,settings,cfg):
    """最多三次恢复：平移起点增加重叠、双向上下文、缩短模型内部子窗口；参数完全相同的尝试只跑一次。"""
    count=cfg.seam_retries
    if not 0<=count<=3:raise ValueError('接缝重试次数必须在0到3之间')
    step=settings.overlap_frames
    variants=[(start,end,settings.window_frames),
              (max(scene_start,start-step),end,settings.window_frames),
              (max(scene_start,start-2*step),min(scene_end,end+step),settings.window_frames),
              (max(scene_start,start-step),min(scene_end,end+step),max(2,settings.window_frames//2))][:count+1]
    # 镜头起点处平移无效时，重复参数只会白跑GPU；去重不减少不同参数的恢复次数。
    return list(dict.fromkeys(variants))


# 只有接缝不确定可通过换上下文窗口恢复；颜色残留（确认残留/待目检）换窗口无效，不重试。
RETRYABLE_ERRORS={'SEAM_UNCERTAIN'}
RESIDUE_ERRORS={'RESIDUE_FOUND','RESIDUE_UNCERTAIN'}


def residue_summary(scores):
    """逐施工帧残留复检汇总；只汇总数字，不改写逐帧原始候选。"""
    keys=['颜色候选','区别色候选数','遮罩内颜色候选','收边带颜色候选','时域证实背景本色','外环证实背景本色','背景本色证实','外环证实但时域反证',
          '确认残留','遮罩内确认残留','收边带确认残留','待目检','遮罩内待目检','收边带待目检','未解释候选','遮罩内残留','收边带残留',
          '无模板遮罩内像素','无模板收边带像素','同色不适用组数','同色不适用施工像素']
    summary={k:sum(s.get(k,0) for s in scores) for k in keys}
    reasons=[]
    for s in scores:
        for reason in s.get('待目检原因',[]):
            if len(reasons)<20:reasons.append(f"帧{s['frame']}：{reason}")
    summary.update(有效施工帧数=len(scores),
                   通过帧数=sum(1 for s in scores if s.get('判定')=='通过'),
                   确认残留帧=[s['frame'] for s in scores if s.get('判定')=='RESIDUE_FOUND'],
                   待目检帧=[s['frame'] for s in scores if s.get('判定')=='RESIDUE_UNCERTAIN'],
                   单帧最大未解释=max((s.get('未解释候选',0) for s in scores),default=0),
                   待目检原因=reasons)
    summary['判定']=('RESIDUE_FOUND' if summary['确认残留帧'] else 'RESIDUE_UNCERTAIN' if summary['待目检帧'] else '通过') if scores else '无施工样本'
    return summary


def residue_error(scores):
    """把逐帧三态转换为错误；待目检绝不算通过。"""
    summary=residue_summary(scores)
    if summary['判定']=='RESIDUE_FOUND':
        return PipelineError('RESIDUE_FOUND','确认残留：存在有背景证据反证的水印主色候选',details=summary)
    if summary['判定']=='RESIDUE_UNCERTAIN':
        return PipelineError('RESIDUE_UNCERTAIN','待目检：主色候选或无模板组缺少可判背景证据，需人工目检，不算通过',details=summary)
    return None


def protection_mask(work,name,shape):
    """场景/字幕保护区并集；目录不存在时为空集。保护区只排除模板与外环取样，不缩小候选统计。"""
    result=np.zeros(shape,np.uint8)
    for folder in ['scene-protection','subtitle-protection']:
        path=work/folder/name
        if path.exists():
            mask=cv2.imread(str(path),0)
            if mask is None or mask.shape!=shape or np.any((mask!=0)&(mask!=255)):raise PipelineError('MASK_INVALID','保护掩膜缺失或几何无效')
            result|=mask
    return result


_ORIG_CLEAN_REFERENCES=CleanReferences
_ORIG_RESTORE_ORTHO=restore_orthogonal_luma_detail
_ORIG_SOFT_COMPOSITE=soft_composite
_ORIG_PROTECTION_MASK=protection_mask
_ORIG_MEASURE=_edge_residue_mod.measure
_ORIG_COMPOSITE=_edge_residue_mod.composite
_ORIG_VERIFY_CONFIRMED=_trusted_context_mod.verify_from_confirmed
_ORIG_EXCLUSION_MASK=_trusted_context_mod.exclusion_mask
_ORIG_READ_CONFIRMED=_trusted_context_mod.read_confirmed
_ORIG_OVERLAP_WEIGHT=_trusted_context_mod.overlap_weight


def _build_ref_shm(ref_imgs,ref_trusted,ref_grays,static_hp):
    """将当前任务的只读参考帧、可信掩膜、灰度图与高通基底打包进 SharedMemory，零拷贝共享给工作进程。"""
    n_refs=len(ref_imgs)
    if n_refs:
        h,w=ref_imgs[0].shape[:2]
        imgs_arr=np.stack(ref_imgs)
        tr_arr=np.stack(ref_trusted)
        gr_arr=np.stack(ref_grays)
    else:
        h,w=0,0
        imgs_arr=np.empty((0,0,0,3),dtype=np.uint8)
        tr_arr=np.empty((0,0,0),dtype=np.uint8)
        gr_arr=np.empty((0,0,0),dtype=np.uint8)
    has_hp=static_hp is not None
    hp_arr=np.ascontiguousarray(static_hp,dtype=np.float32) if has_hp else np.empty((0,0),dtype=np.float32)
    parts=[
        ('imgs',imgs_arr.shape,imgs_arr.dtype.str,imgs_arr.nbytes),
        ('tr',tr_arr.shape,tr_arr.dtype.str,tr_arr.nbytes),
        ('gr',gr_arr.shape,gr_arr.dtype.str,gr_arr.nbytes),
        ('hp',hp_arr.shape,hp_arr.dtype.str,hp_arr.nbytes),
    ]
    offsets={};cursor=0
    for name,shp,dt,nb in parts:
        cursor=(cursor+63)&~63
        offsets[name]=(cursor,shp,dt,nb)
        cursor+=nb
    shm=shared_memory.SharedMemory(create=True,size=max(1,cursor))
    try:
        for name,arr in [('imgs',imgs_arr),('tr',tr_arr),('gr',gr_arr),('hp',hp_arr)]:
            off,shp,dt,nb=offsets[name]
            if nb:
                view=np.ndarray(shp,dtype=dt,buffer=shm.buf,offset=off)
                np.copyto(view,arr)
    except Exception:
        shm.close();shm.unlink();raise
    return shm,dict(offsets=offsets,has_hp=has_hp,n_refs=n_refs,shape=(h,w))


def _attach_ref_shm(shm_name,spec):
    try:
        shm=shared_memory.SharedMemory(name=shm_name,create=False,track=False)
    except TypeError:
        # Python<3.13 无 track 参数：附着后从资源跟踪器注销，避免工作进程重复登记/误删共享内存（不影响计算）。
        shm=shared_memory.SharedMemory(name=shm_name,create=False)
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name,'shared_memory')
        except Exception:
            pass
    offsets=spec['offsets'];views={}
    for name in ('imgs','tr','gr','hp'):
        off,shp,dt,nb=offsets[name]
        if nb:
            arr=np.ndarray(shp,dtype=dt,buffer=shm.buf,offset=off)
            arr.flags.writeable=False
            views[name]=arr
        else:
            views[name]=None
    n_refs=spec['n_refs']
    ref_imgs=[views['imgs'][i] for i in range(n_refs)]
    ref_tr=[views['tr'][i] for i in range(n_refs)]
    ref_gr=[views['gr'][i] for i in range(n_refs)]
    static_hp=views['hp'] if spec['has_hp'] else None
    return shm,ref_imgs,ref_tr,ref_gr,static_hp


def _prep_frame_worker(item):
    """并行执行单帧模型输入裁块写出与独立层输入自检。"""
    (j,start,end,left,box,pixels,current_path,mask_path_src,frame_dst,mask_dst,
     active_packed,sub_shape,trusted_dict)=item
    roi.atomic_png(Path(frame_dst),roi.crop(cv2.imread(current_path),box))
    confirmed=_trusted_context_mod.read_confirmed(mask_path_src)
    if trusted_dict is None:
        return ('OK',j,None)
    trusted=_trusted_context_mod.ContextSettings(**trusted_dict)
    exclusion=_trusted_context_mod.exclusion_mask(confirmed,trusted)
    if np.count_nonzero(exclusion)/pixels>trusted.mask_fraction_limit:
        return ('MASK_UNCERTAIN',j,'MASK_UNCERTAIN','全确认水印排除区超过20%，必须重规划',None)
    model_mask=roi.crop(exclusion,box,True);mask_p=Path(mask_dst);roi.atomic_png(mask_p,model_mask)
    construction_crop=roi.crop(_unpack_bool(active_packed,sub_shape).astype(np.uint8)*255,[0,0,sub_shape[1],sub_shape[0]],True)
    try:
        checked=_trusted_context_mod.verify_from_confirmed(mask_path_src,box,cv2.imread(str(mask_p),0),construction_crop,pixels,trusted)
    except PipelineError as exc:
        return ('AUDIT_ERR',j,exc.code,str(exc),exc.details)
    entry=dict(frame=j,context=not(start<=j<end),模型掩膜sha256=hashlib.sha256(mask_p.read_bytes()).hexdigest(),**checked)
    return ('OK',j,entry)


def _prepare_parallel_inputs(pool, task, current, files, masks, ownership, box,
                             start, end, left, right, pixels, trusted, audit, check):
    """在冻结的上一图层上准备并自检模型输入。"""
    a, b, c, d = box
    trusted_dict = asdict(trusted) if trusted is not None else None
    items = [
        (j, start, end, left, box, pixels, str(current[j]), str(task.parent/'masks'/files[j].name),
         str(task/'frames'/f'{j-left:06d}.png'), str(task/'masks'/f'{j-left:06d}.png'),
         _pack_bool((masks[j][b:d,a:c]>0)&(ownership[b:d,a:c]>0)), (d-b,c-a), trusted_dict)
        for j in range(left, right)
    ]
    for status, frame, *details in pool.imap(_prep_frame_worker, items):
        check()
        if status == 'MASK_UNCERTAIN':
            raise PipelineError('MASK_UNCERTAIN', details[1])
        if status == 'AUDIT_ERR':
            code, message, evidence = details
            audit['帧'].append(dict(frame=frame, **evidence))
            audit['错误码'] = code
            raise PipelineError(code, message, details=evidence)
        if audit is not None and details[0] is not None:
            audit['帧'].append(details[0])


def _run_attempt_model(task, length, singleton, cfg, timeout, check):
    if singleton:
        for name in ('frames', 'masks'):
            shutil.copy2(task/name/'000000.png', task/name/'000001.png')
    root = Path(os.environ['PROPAINTER_ROOT'])
    args = [
        os.environ['INPAINT_PYTHON'], str(Path(__file__).with_name('torch_stage.py')),
        '--metrics', task/'metrics.json', '--frames-only', '--script', root/'inference_propainter.py',
        '--video', task/'frames', '--mask', task/'masks', '--output', task/'out', '--save_frames',
        '--mask_dilation', '0', '--subvideo_length', str(length), '--neighbor_length', '6', '--ref_stride', '20',
    ]
    if cfg.fp16:
        args.append('--fp16')
    _model_stage(args, task, task.name, check=check, cwd=os.environ['PROPAINTER_ROOT'],
                 timeout=timeout, stage='逐层ROI-ProPainter')


def _make_attempt_dirs(task):
    task.mkdir()
    (task/'frames').mkdir()
    (task/'masks').mkdir()


def _attempt_audit(task_number, layer, attempt, left, right, box):
    return dict(任务=task_number, 图层=layer, 尝试=attempt, 输入区间=[left, right], ROI=box, 通过=False, 帧=[])


def _commit_restore_worker(item):
    """并行回写未过块最佳暂存帧并更新覆盖掩膜 PNG。"""
    best_task_str,baseline_str,file_name,dest_path,cov_path,active_packed,shape=item
    roi.atomic_png(Path(dest_path),restore(Path(best_task_str),Path(baseline_str),file_name))
    done=cv2.imread(cov_path,0)
    roi.atomic_png(Path(cov_path),done|(_unpack_bool(active_packed,shape).astype(np.uint8)*255))


def _commit_pass_worker(item):
    """并行提交已通过施工帧至输出层并更新覆盖掩膜 PNG。"""
    staged_path,dest_path,cov_path,active_packed,shape=item
    Path(staged_path).replace(dest_path)
    done=cv2.imread(cov_path,0)
    roi.atomic_png(Path(cov_path),done|(_unpack_bool(active_packed,shape).astype(np.uint8)*255))


def _process_frame_worker(item):
    """并行执行单帧参考回填、正交亮度细节保留、接缝计算、暂存写出与主色残留复检。"""
    (j,box,crop_shape,shape,construction,in_completed,weight,dest_has_changes,in_sample,in_stage,
     continue_unpassed,need_repair_ev,detail_mode,edge_dict,active_packed,shared_packed,mask_crop_packed,
     trk_hp_sparse,shm_name,shm_spec,paths,dense_ctx)=item
    (pred_path,current_path,file_path,dest_path,mask_path_src,band_path,work_dir_str,file_name,
     staged_path,seam_diff_path,seam_shared_path,sample_dir_str,task_mask_path)=paths
    a,b,c,d=box;sub_shape=(d-b,c-a)
    repair=cv2.imread(pred_path)
    if repair is None or repair.shape!=crop_shape:
        return ('ERR',j,'OUTPUT_INVALID','逐层ROI模型改变几何')
    current_img=cv2.imread(current_path)
    original=roi.crop(current_img,box)
    if original.shape!=crop_shape:
        return ('ERR',j,'OUTPUT_INVALID','逐层ROI模型改变几何')
    mask=_unpack_bool(mask_crop_packed,crop_shape[:2]).astype(np.uint8)*255
    confirmed=roi.crop(_trusted_context_mod.read_confirmed(mask_path_src),box,True)
    edge_cfg=_edge_residue_mod.EdgeSettings(**edge_dict)

    shm,ref_imgs,ref_tr,ref_gr,static_hp=_attach_ref_shm(shm_name,shm_spec)
    try:
        references=CleanReferences(ref_imgs,[None]*len(ref_imgs),trusted=ref_tr,grays=ref_gr)
        restored=references.restore(original,mask,repair,confirmed=confirmed)
        if detail_mode=='orthogonal_luma' and static_hp is not None:
            wm_hp=static_hp
            if trk_hp_sparse is not None:
                ty0,ty1,tx0,tx1,patch=trk_hp_sparse
                trk_crop=np.zeros(crop_shape[:2],dtype=np.float32)
                trk_crop[ty0:ty1,tx0:tx1]=patch
                wm_hp=np.where(np.abs(trk_crop)>np.abs(wm_hp),trk_crop,wm_hp)
            restored=restore_orthogonal_luma_detail(original,restored,mask,wm_hp,edge_cfg)
        painted=_edge_residue_mod.composite(original,restored,mask)[:d-b,:c-a]
        active=_unpack_bool(active_packed,sub_shape)
        shared=_unpack_bool(shared_packed,sub_shape)
        if in_sample:
            sample=Path(sample_dir_str);sample.mkdir(parents=True,exist_ok=True)
            roi.atomic_png(sample/f'{j:06d}-raw-model.png',repair)
            shutil.copy2(task_mask_path,sample/f'{j:06d}-exclusion.png')
            roi.atomic_png(sample/f'{j:06d}-construction.png',roi.crop(active.astype(np.uint8)*255,[0,0,c-a,d-b],True))
            if in_stage:
                for label,value in [('source',original),('restored',restored),('composited',painted)]:
                    roi.atomic_png(sample/f'{j:06d}-{label}.png',value)
        image = cv2.imread(dest_path) if dest_has_changes else current_img
        old = image[b:d,a:c]
        shared_pixels = int(np.count_nonzero(shared))
        shared_difference = None
        if shared_pixels:
            shared_difference = np.abs(old[shared].astype(float)-painted[shared].astype(float))
        disagreement = float(shared_difference.mean()) if shared_difference is not None else None
        frame_score=None;residue_score=None;repair_ev_payload=None
        if construction:
            if continue_unpassed:
                difference = np.zeros(sub_shape, dtype=np.uint16)
                seam_mae = 0.
                if shared_difference is not None:
                    pixel_difference = shared_difference.mean(axis=1)
                    difference[shared] = np.rint(pixel_difference*256).astype(np.uint16)
                    seam_mae = float(pixel_difference.mean())
                roi.atomic_png(Path(seam_diff_path),difference)
                roi.atomic_png(Path(seam_shared_path),shared.astype(np.uint8)*255)
                frame_score=dict(frame=j,mae=seam_mae,pixels=shared_pixels)
            if in_completed:
                mixed=cv2.addWeighted(painted,weight,old,1-weight,0)
                painted[active&shared]=mixed[active&shared]
            old[active]=painted[active];roi.atomic_png(Path(staged_path),image)
            if active.any():
                band=_trusted_context_mod.read_confirmed(band_path)[b:d,a:c] if band_path else np.zeros(sub_shape,np.uint8)
                source_color=original[:d-b,:c-a] if current_path==file_path else roi.crop(cv2.imread(file_path),box)[:d-b,:c-a]
                evidence=getattr(references,'last_evidence',None)
                donor_crops=getattr(references,'last_donor_crops',[])
                if evidence is not None:
                    evidence={key:(value[:d-b,:c-a] if hasattr(value,'shape') else value) for key,value in evidence.items()}
                active_mask=active.astype(np.uint8)*255
                conf_crop=confirmed[:d-b,:c-a];prot_crop=protection_mask(Path(work_dir_str),file_name,shape)[b:d,a:c]
                score=_edge_residue_mod.measure(source_color,old,active_mask,band&active_mask,edge_cfg,
                                                reference_evidence=evidence,confirmed=conf_crop,protected=prot_crop,
                                                return_masks=need_repair_ev)
                if score.get('判定') in RESIDUE_ERRORS and dense_ctx is not None:
                    scene_start,scene_end,task_refs_tuple,cur_paths,fil_paths=dense_ctx
                    preloaded={int(rf): (ref_imgs[idx],ref_tr[idx],ref_gr[idx]) for idx,rf in enumerate(task_refs_tuple)}
                    dense_ev,dense_crops=_dense_frame_evidence(
                        j,scene_start,scene_end,box,[Path(p) for p in cur_paths],Path(work_dir_str),
                        [Path(p) for p in fil_paths],task_refs_tuple,original,mask,confirmed,
                        preloaded_refs=preloaded,return_crops=True,
                    )
                    if dense_ev is not None:
                        dense_ev={key:(value[:d-b,:c-a] if hasattr(value,'shape') else value) for key,value in dense_ev.items()}
                        dense_sc=_edge_residue_mod.measure(source_color,old,active_mask,band&active_mask,edge_cfg,
                                                           reference_evidence=dense_ev,confirmed=conf_crop,protected=prot_crop,
                                                           return_masks=need_repair_ev)
                        if (dense_sc.get('确认残留',0),dense_sc.get('待目检',0))<(score.get('确认残留',0),score.get('待目检',0)):
                            score=dense_sc;evidence=dense_ev;donor_crops=dense_crops
                repair_masks = score.pop('_masks', None)
                residue_score=dict(frame=j,**score)
                if repair_masks is not None and score.get('判定') == 'RESIDUE_FOUND':
                    residue_score['_repair_residue'] = _pack_bool(repair_masks['确认残留'])
                if need_repair_ev and score.get('判定')=='RESIDUE_FOUND' and evidence is not None:
                    repair_ev_payload=(evidence['background'],evidence['confidence'],evidence['counts'],donor_crops)
        return ('OK',j,disagreement,frame_score,residue_score,repair_ev_payload)
    finally:
        shm.close()


def execute(work,files,detected,log,cfg):
    settings=HLSSettings.from_env();started=time.monotonic()
    spatial=SpatialSettings(**json.loads((work/'spatial-roi.json').read_text()))
    policy_path=work/'inpaint-policy.json'
    policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
    continue_unpassed=policy.get('continue_unpassed',False)
    from .edge_residue import EdgeSettings,measure,composite
    edge_path=work/'edge-policy.json'
    edge_cfg=EdgeSettings(**json.loads(edge_path.read_text())) if edge_path.exists() else EdgeSettings()
    shape=cv2.imread(str(files[0])).shape[:2];pixels=shape[0]*shape[1]
    trusted_path=work/'trusted-context.json';trusted=None;input_audits=[]
    if not trusted_path.exists():raise PipelineError('UNTRUSTED_CONTEXT','缺少可信上下文配置，禁止退回当前层')
    if trusted_path.exists():
        from .trusted_context import ContextSettings,verify_from_confirmed,exclusion_mask,read_confirmed,overlap_weight
        trusted=ContextSettings(**json.loads(trusted_path.read_text()))
    workers=_cpu_workers()
    can_parallel=(
        workers>1
        and CleanReferences is _ORIG_CLEAN_REFERENCES
        and restore_orthogonal_luma_detail is _ORIG_RESTORE_ORTHO
        and soft_composite is _ORIG_SOFT_COMPOSITE
        and protection_mask is _ORIG_PROTECTION_MASK
        and measure is _ORIG_MEASURE
        and composite is _ORIG_COMPOSITE
        and verify_from_confirmed is _ORIG_VERIFY_CONFIRMED
        and exclusion_mask is _ORIG_EXCLUSION_MASK
        and read_confirmed is _ORIG_READ_CONFIRMED
        and overlap_weight is _ORIG_OVERLAP_WEIGHT
    )
    pool_holder=[None];normal_exit=False
    def get_pool():
        if pool_holder[0] is None and can_parallel:
            pool_holder[0]=mp.get_context('forkserver').Pool(processes=workers,initializer=_worker_init)
        return pool_holder[0]
    def save_input_audit():
        if trusted is None:return
        (work/'层输入自检.json').write_text(json.dumps(json_native(dict(
            口径='所有冻结确认水印并集与实际写入模型掩膜逐像素核对，覆盖重试和上下文帧；不是未检测水印的真值召回率',
            模型排除区='全部确认水印并集',回写域='当前层与当前ROI施工域交集',
            源水印未排除像素=sum(f.get('源水印未排除像素',0) for a in input_audits for f in a['帧']),
            全部已检查尝试通过=all(a['通过'] for a in input_audits),尝试=input_audits)),ensure_ascii=False,separators=(',',':')))
    save_input_audit()
    layers=json.loads((work/'layers.json').read_text())['layers']
    animation_path=work/'animation-events.json'
    animation_plan=json.loads(animation_path.read_text()) if animation_path.exists() else None
    if animation_plan:
        from .watermark_groups import AnimationSettings,planned_windows
        animation_cfg=AnimationSettings(**animation_plan['settings'])
        if animation_cfg.maximum_window_frames>cfg.maximum_frames:raise ValueError('动画窗口不得超过既有ROI最大帧数')
    resume=None
    if policy.get('resume_from'):
        from .roi_resume import prepare as resume_prepare,recheck as resume_recheck
        resume=resume_prepare(work,policy['resume_from'],files,cfg,settings,spatial,layers,log)
    coverage=work/'roi-coverage'
    if resume is None:
        coverage.mkdir()
        for p in files:roi.atomic_png(coverage/p.name,np.zeros(shape,np.uint8))
    current=files;records=[];layer_records=[];unpassed=[];unfillable=[]
    def save():
        summary=dict(空间配置=asdict(spatial),配置=asdict(cfg),任务=records,未过块=unpassed,不可填块=unfillable,全部ROI通过=len(layer_records)==len(layers) and all(r['通过'] for r in records),图层=layer_records,实际图层数=len(layers),
                     窗口=settings.window_frames,重叠=settings.overlap_frames,
                     总耗时秒=time.monotonic()-started,原帧回退数=0)
        (work/'roi-report.json').write_text(json.dumps(json_native(summary),ensure_ascii=False,separators=(',',':')))
        return summary
    try:
        for k,layer in enumerate(layers):
            begin=time.monotonic();dest=work/f'roi-layer-{k:02d}'
            if resume is None:dest.mkdir()
            masks=[cv2.imread(str(Path(layer['directory'])/p.name),0) for p in files]
            for i,m in enumerate(masks):
                if m is None or m.shape!=shape or np.any((m!=0)&(m!=255)):
                    raise PipelineError('MASK_INVALID','逐层ROI掩膜缺失或几何无效')
                if np.count_nonzero(m)/pixels>settings.mask_fraction_limit:
                    raise PipelineError('MASK_UNCERTAIN','逐层ROI单遍超过20%保护',details=dict(layer=k,frame=i))
            # 每层独立输出；本层任何窗口都只读取上一层，不能把当前层施工反馈进模型。
            if resume is None:
                for p,source in zip(files,current):
                    try:os.link(source,dest/p.name)
                    except OSError:shutil.copy2(source,dest/p.name)
            layer_done=resume['done'] if resume else {};log(f'逐层ROI 图层={k+1}/{len(layers)} 开始')
            for scene_start,scene_end in detected['scenes']:
                scene_img_cache=_LRUCache(160)
                scene_conf_cache=_LRUCache(260)
                def get_current_img(idx):
                    val=scene_img_cache.get(idx)
                    if val is None:
                        val=cv2.imread(str(current[idx]))
                        scene_img_cache.put(idx,val)
                    return val
                def get_confirmed_mask(idx):
                    if read_confirmed is not _ORIG_READ_CONFIRMED:
                        return read_confirmed(work/'masks'/files[idx].name)
                    val=scene_conf_cache.get(idx)
                    if val is None:
                        val=read_confirmed(work/'masks'/files[idx].name)
                        scene_conf_cache.put(idx,val)
                    return val
                refs=np.unique(np.linspace(scene_start,scene_end-1,min(24,scene_end-scene_start),dtype=int))
                scene_track_hps=_scene_track_highpass(detected,files,scene_start,scene_end,shape) if (trusted and trusted.detail_preserve_mode=='orthogonal_luma') else {}
                schedule=list(windows(scene_start,scene_end,settings.window_frames,settings.overlap_frames))
                if animation_plan:
                    events=[e['interval'] for e in animation_plan['layers'].get(str(k),[]) if scene_start<=e['interval'][0]<scene_end]
                    schedule=planned_windows(scene_start,scene_end,settings.window_frames,settings.overlap_frames,events,animation_cfg)
                prefetch_enabled = can_parallel and resume is None and os.environ.get('ROI_MODEL_PREFETCH') == '1'
                executor = ThreadPoolExecutor(max_workers=1) if prefetch_enabled else None
                prefetch_stop = Event()
                pending = None
                max_tasks = int(os.environ.get('ROI_MAX_TASKS') or 0)
                timeout = min(840, policy.get('model_timeout_seconds') or 840)

                def planned_jobs():
                    for job_start, job_end in schedule:
                        variants = seam_variants(job_start,job_end,scene_start,scene_end,settings,cfg)
                        footprint = np.bitwise_or.reduce(masks[job_start:job_end])
                        jobs = plan(footprint,max(r-l for l,r,_ in variants),cfg,spatial)
                        if jobs:
                            yield job_start, job_end, variants, jobs

                def prepare_next(task, job, pool, audit, left, right, length):
                    job_start, job_end, fragment = job

                    def check():
                        if prefetch_stop.is_set():
                            raise PipelineError('ROI_PREFETCH_CANCELLED', '逐层ROI预取已取消')

                    check()
                    _make_attempt_dirs(task)
                    _prepare_parallel_inputs(pool,task,current,files,masks,fragment['ownership'],fragment['box'],
                                             job_start,job_end,left,right,pixels,trusted,audit,check)
                    audit['通过'] = True
                    _run_attempt_model(task,length,False,cfg,timeout,check)

                try:
                    task_groups = planned_jobs()
                    queued = next(task_groups, None)
                    while queued is not None:
                        start, end, window_variants, jobs = queued
                        queued = next(task_groups, None) if executor else None
                        for job_index, fragment in enumerate(jobs):
                            ownership=fragment['ownership'];box=fragment['box'];a,b,c,d=box
                            completed={j for j in range(scene_start,scene_end) if j in layer_done and np.any(layer_done[j]&ownership)}
                            task_refs=_window_refs(scene_start,scene_end,start,end,refs,trusted.dense_reference_frames if trusted else 24)
                            row=dict(任务=len(records)+1,图层=k,开始帧=start,结束帧=end,ROI=box,fp16=cfg.fp16,
                                     施工面积像素=int(np.count_nonzero(ownership)),裁块像素=padded_area(box),像素帧预算=spatial.maximum_pixel_frames,输入图层=k-1,窗口=settings.window_frames,重叠=settings.overlap_frames,通过=False,尝试=[])
                            if max_tasks and len(records)>=max_tasks:
                                log(f'逐层ROI 验收夹具 ROI_MAX_TASKS={max_tasks} 已完成，停止后续任务');save();normal_exit=True;raise SystemExit(0)
                            records.append(row);tick=time.monotonic();save()
                            if resume and row['任务']<=len(resume['rows']):
                                resume_recheck(resume,row,ownership,work,files,refs,box,start,end,edge_cfg,settings,log)
                                if not row['通过']:unpassed.append(row['任务'])
                                save();log(f'逐层ROI 任务={row["任务"]} 检查点复用完成 通过={row["通过"]}；未运行补画模型')
                                continue
                            baseline=work/f'roi-{len(records):04d}-baseline'
                            variants=list(window_variants)
                            oom_used=False
                            task_shm=None;task_shm_spec=None
                            try:
                                for attempt,(left,right,length) in enumerate(variants):
                                    task=work/f'roi-{len(records):04d}-attempt-{attempt:02d}'
                                    prefetched = pending if pending is not None and pending[0] == task else None
                                    if prefetched is not None:
                                        pending = None
                                    else:
                                        _make_attempt_dirs(task)
                                    trial=dict(尝试=attempt,开始帧=left,结束帧=right,subvideo_length=length,
                                               前向重叠帧数=sum(j in completed for j in range(left,start+settings.overlap_frames)),
                                               通过=False,重叠最大差异=None)
                                    row['尝试'].append(trial);attempt_tick=time.monotonic();save()
                                    audit=None
                                    if trusted:
                                        audit = prefetched[1] if prefetched else _attempt_audit(row['任务'],k,attempt,left,right,box)
                                        input_audits.append(audit)
                                    log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 输入帧={left}:{right} 施工帧={start}:{end} subvideo={length} 裁块={c-a}×{d-b}')
                                    try:
                                        if padded_area(box)*(right-left)>spatial.maximum_pixel_frames:raise PipelineError('GPU_BUDGET_EXCEEDED','空间裁块超过配置像素帧预算')
                                        if prefetched is not None:
                                            try:
                                                prefetched[2].result()
                                            finally:
                                                if audit['通过']:
                                                    trial['超时秒'] = timeout
                                        elif can_parallel and right-left>1:
                                            _prepare_parallel_inputs(get_pool(),task,current,files,masks,ownership,box,
                                                                     start,end,left,right,pixels,trusted,audit,lambda: None)
                                        else:
                                            for j in range(left,right):
                                                roi.atomic_png(task/'frames'/f'{j-left:06d}.png',roi.crop(get_current_img(j),box))
                                                exclusion=exclusion_mask(get_confirmed_mask(j),trusted)
                                                if trusted and np.count_nonzero(exclusion)/pixels>trusted.mask_fraction_limit:
                                                    raise PipelineError('MASK_UNCERTAIN','全确认水印排除区超过20%，必须重规划')
                                                model_mask=roi.crop(exclusion,box,True);mask_path=task/'masks'/f'{j-left:06d}.png';roi.atomic_png(mask_path,model_mask)
                                                if trusted:
                                                    try:checked=verify_from_confirmed(work/'masks'/files[j].name,box,cv2.imread(str(mask_path),0),roi.crop(masks[j]&ownership,box,True),pixels,trusted)
                                                    except PipelineError as exc:
                                                        audit['帧'].append(dict(frame=j,**exc.details));audit['错误码']=exc.code;save_input_audit();raise
                                                    audit['帧'].append(dict(frame=j,context=not(start<=j<end),模型掩膜sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),**checked))
                                        if trusted:audit['通过']=True;save_input_audit();log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 层输入自检通过，未排除像素=0，检查帧数={right-left}')
                                        singleton=right-left==1
                                        trial['超时秒']=timeout
                                        if prefetched is None:
                                            heartbeat=time.monotonic()

                                            def check_model():
                                                nonlocal heartbeat
                                                if time.monotonic()-heartbeat>=30:
                                                    phase='模型指标已写出，等待子进程退出' if (task/'metrics.json').exists() else '模型进程仍存活，正在计算'
                                                    log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} {phase}')
                                                    heartbeat=time.monotonic()

                                            _run_attempt_model(task,length,singleton,cfg,timeout,check_model)
                                        next_number = len(records) + 1
                                        if executor and pending is None and (not max_tasks or next_number <= max_tasks):
                                            if job_index+1 < len(jobs):
                                                next_job = start, end, jobs[job_index+1]
                                                next_left, next_right, next_length = window_variants[0]
                                            elif queued is not None:
                                                next_start, next_end, next_variants, next_jobs = queued
                                                next_job = next_start, next_end, next_jobs[0]
                                                next_left, next_right, next_length = next_variants[0]
                                            else:
                                                next_job = None
                                            if next_job is not None and next_right-next_left > 1:
                                                next_box = next_job[2]['box']
                                                if padded_area(next_box)*(next_right-next_left) <= spatial.maximum_pixel_frames:
                                                    next_task = work/f'roi-{next_number:04d}-attempt-00'
                                                    next_audit = _attempt_audit(next_number,k,0,next_left,next_right,next_box)
                                                    future = executor.submit(prepare_next,next_task,next_job,get_pool(),next_audit,
                                                                             next_left,next_right,next_length)
                                                    pending = next_task, next_audit, future
                                                    log(f'逐层ROI 任务={next_number} 开始预取首轮模型')
                                        log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 模型已结束，开始CPU参考回填与残留复检')
                                        predicted=sorted((task/'out/frames/frames').glob('*.png'))
                                        if len(predicted)!=(2 if singleton else right-left):raise PipelineError('ROI_INCOMPLETE','逐层ROI输出帧数缺失，禁止原帧回退')
                                        crop_shape=cv2.imread(str(task/'frames'/'000000.png')).shape
                                        reference_holder=[];scene_hp_holder=[]
                                        def clean_references():
                                            # 惰性构建：只有需要参考回填/证据的帧才读取同镜头参考。
                                            if not reference_holder:
                                                ref_imgs=[roi.crop(get_current_img(j),box) for j in task_refs]
                                                ref_msks=[roi.crop(get_confirmed_mask(j),box,True) for j in task_refs]
                                                cr=CleanReferences(ref_imgs,ref_msks)
                                                reference_holder.append(cr)
                                                if trusted and trusted.detail_preserve_mode=='orthogonal_luma':
                                                    scene_hp_holder.append(_scene_static_highpass(ref_imgs,grays=cr.grays))
                                            return reference_holder[0]
                                        maximum=0.;staged=task/'staged';staged.mkdir();frame_scores=[];residue_scores=[];skipped=dict(上下文帧=0,无施工无重叠帧=0)
                                        sample_frames=set();stage_frames=set()
                                        if trusted:
                                            active_frames=[j for j in range(start,end) if np.any(masks[j]&ownership)]
                                            if active_frames:
                                                sample_frames={active_frames[q] for q in np.linspace(0,len(active_frames)-1,min(trusted.model_sample_count,len(active_frames)),dtype=int)}
                                                stage_frames=set(sorted(sample_frames)[:trusted.retain_stage_sample_count])
                                        if continue_unpassed:(task/'seam-difference').mkdir();(task/'seam-shared').mkdir()
                                        # 先完整核验，再提交；失败尝试不得修改覆盖记录或已通过窗口。
                                        cpu_heartbeat=time.monotonic()
                                        if can_parallel:
                                            work_frames=[]
                                            for j in range(left,right):
                                                construction=start<=j<end
                                                active=(masks[j][b:d,a:c]>0)&(ownership[b:d,a:c]>0)
                                                prior_done=layer_done.get(j)
                                                shared=(prior_done[b:d,a:c]>0)&(masks[j][b:d,a:c]>0) if prior_done is not None else np.zeros(active.shape,bool)
                                                if not construction and not shared.any():
                                                    r_chk=cv2.imread(str(predicted[j-left]))
                                                    if r_chk is None or r_chk.shape!=crop_shape:raise PipelineError('OUTPUT_INVALID','逐层ROI模型改变几何')
                                                    skipped['上下文帧']+=1;continue
                                                if not active.any() and not shared.any() and j not in sample_frames:
                                                    r_chk=cv2.imread(str(predicted[j-left]))
                                                    if r_chk is None or r_chk.shape!=crop_shape:raise PipelineError('OUTPUT_INVALID','逐层ROI模型改变几何')
                                                    skipped['无施工无重叠帧']+=1
                                                    image=cv2.imread(str(dest/files[j].name));roi.atomic_png(staged/files[j].name,image)
                                                    if continue_unpassed:
                                                        roi.atomic_png(task/'seam-difference'/files[j].name,np.zeros(active.shape,np.uint16))
                                                        roi.atomic_png(task/'seam-shared'/files[j].name,np.zeros(active.shape,np.uint8))
                                                        frame_scores.append(dict(frame=j,mae=0.,pixels=0))
                                                    continue
                                                work_frames.append((j,construction,active,shared))
                                            if work_frames:
                                                if task_shm is None:
                                                    ref_imgs=[roi.crop(get_current_img(rj),box) for rj in task_refs]
                                                    ref_msks=[roi.crop(get_confirmed_mask(rj),box,True) for rj in task_refs]
                                                    ker_tr=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
                                                    ref_trusted=[255-cv2.dilate(m,ker_tr) for m in ref_msks]
                                                    ref_grays=[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in ref_imgs]
                                                    static_hp=_scene_static_highpass(ref_imgs,grays=ref_grays) if (trusted and trusted.detail_preserve_mode=='orthogonal_luma') else None
                                                    task_shm,task_shm_spec=_build_ref_shm(ref_imgs,ref_trusted,ref_grays,static_hp)
                                                need_dense=bool(trusted and trusted.dense_reference_frames>len(task_refs))
                                                dense_ctx=(scene_start,scene_end,tuple(int(x) for x in task_refs),[str(p) for p in current],[str(p) for p in files]) if need_dense else None
                                                detail_mode=trusted.detail_preserve_mode if trusted else 'none'
                                                edge_dict=asdict(edge_cfg);need_repair_ev=edge_cfg.residue_repair_rounds>0
                                                sample_dir=work/'model-samples'/f'task-{row["任务"]:04d}'/f'attempt-{attempt:02d}'
                                                worker_items=[]
                                                for j,construction,active,shared in work_frames:
                                                    in_comp=j in completed
                                                    weight=overlap_weight(j,start,end,completed) if (construction and in_comp) else 1.0
                                                    mask_crop=roi.crop(masks[j],box,True)>0
                                                    trk_sparse=None
                                                    if detail_mode=='orthogonal_luma' and j in scene_track_hps:
                                                        trk_crop=roi.crop(scene_track_hps[j],box,True)
                                                        nz_y,nz_x=np.nonzero(trk_crop)
                                                        if len(nz_y):
                                                            ty0,ty1=int(nz_y.min()),int(nz_y.max())+1
                                                            tx0,tx1=int(nz_x.min()),int(nz_x.max())+1
                                                            trk_sparse=(ty0,ty1,tx0,tx1,np.ascontiguousarray(trk_crop[ty0:ty1,tx0:tx1],dtype=np.float32))
                                                    band_p=work/'edge-band'/files[j].name
                                                    paths=(
                                                        str(predicted[j-left]),str(current[j]),str(files[j]),str(dest/files[j].name),
                                                        str(work/'masks'/files[j].name),str(band_p) if band_p.exists() else '',
                                                        str(work),files[j].name,str(staged/files[j].name),
                                                        str(task/'seam-difference'/files[j].name),str(task/'seam-shared'/files[j].name),
                                                        str(sample_dir),str(task/'masks'/f'{j-left:06d}.png'),
                                                    )
                                                    worker_items.append((
                                                        j,box,crop_shape,shape,construction,in_comp,weight,resume is not None or j in layer_done,
                                                        j in sample_frames,j in stage_frames,continue_unpassed,need_repair_ev,
                                                        detail_mode,edge_dict,_pack_bool(active),_pack_bool(shared),_pack_bool(mask_crop),
                                                        trk_sparse,task_shm.name,task_shm_spec,paths,dense_ctx,
                                                    ))
                                                has_sampled=False
                                                # 按施工掩膜面积降序（LPT）派发，避免重帧落到尾部拖累整体并行耗时，再按帧号恢复严格顺序
                                                ordered_pairs=sorted(
                                                    zip(work_frames,worker_items),
                                                    key=lambda pair:-int(np.count_nonzero(pair[0][2])),
                                                )
                                                dispatched_items=[pair[1] for pair in ordered_pairs]
                                                raw_results=[]
                                                for res_w in get_pool().imap_unordered(_process_frame_worker,dispatched_items,chunksize=1):
                                                    if time.monotonic()-cpu_heartbeat>=30:
                                                        log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} CPU合成与复检进度={len(raw_results)+1}/{len(dispatched_items)}')
                                                        cpu_heartbeat=time.monotonic()
                                                    raw_results.append(res_w)
                                                raw_results.sort(key=lambda r:r[1])
                                                for res_w in raw_results:
                                                    if res_w[0]=='ERR':
                                                        raise PipelineError(res_w[2],res_w[3])
                                                    _,jw,disagr,f_sc,r_sc,rep_ev=res_w
                                                    if jw in sample_frames:has_sampled=True
                                                    if disagr is not None:maximum=max(maximum,disagr)
                                                    if f_sc is not None:frame_scores.append(f_sc)
                                                    if r_sc is not None:
                                                        ev_rebuilt=None
                                                        if rep_ev is not None:
                                                            bg,conf_arr,cnt_arr,dcrops=rep_ev
                                                            ev_rebuilt=dict(background=bg,confidence=conf_arr,counts=cnt_arr)
                                                            if dcrops:ev_rebuilt['evaluate_peak']=_build_donor_evaluator(crop_shape[:2],dcrops)
                                                        r_sc['_evidence']=ev_rebuilt
                                                        residue_scores.append(r_sc)
                                                if has_sampled:
                                                    (sample_dir/'metadata.json').write_text(json.dumps(dict(task=row['任务'],attempt=attempt,roi=box,input=[left,right],construction=[start,end],frames=sorted(sample_frames),说明='raw-model为模型原始预测；restored为参考回填后；composited为羽化后、窗口混合与施工域回写前。'),ensure_ascii=False,indent=2))
                                        else:
                                            for j in range(left,right):
                                                if time.monotonic()-cpu_heartbeat>=30:
                                                    log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} CPU合成与复检帧={j}/{right}')
                                                    cpu_heartbeat=time.monotonic()
                                                repair=cv2.imread(str(predicted[j-left]))
                                                if repair is None or repair.shape!=crop_shape:raise PipelineError('OUTPUT_INVALID','逐层ROI模型改变几何')
                                                construction=start<=j<end
                                                active=(masks[j][b:d,a:c]>0)&(ownership[b:d,a:c]>0)
                                                prior_done=layer_done.get(j)
                                                shared=(prior_done[b:d,a:c]>0)&(masks[j][b:d,a:c]>0) if prior_done is not None else np.zeros(active.shape,bool)
                                                # 纯上下文帧（不在start:end）且无接缝重叠时跳过CleanReferences与残留复检；施工窗内既无施工又无重叠且非抽样帧时，结果逐像素不变，同样跳过。
                                                if not construction and not shared.any():
                                                    skipped['上下文帧']+=1;continue
                                                if not active.any() and not shared.any() and j not in sample_frames:
                                                    skipped['无施工无重叠帧']+=1
                                                    image=cv2.imread(str(dest/files[j].name));roi.atomic_png(staged/files[j].name,image)
                                                    if continue_unpassed:
                                                        roi.atomic_png(task/'seam-difference'/files[j].name,np.zeros(active.shape,np.uint16))
                                                        roi.atomic_png(task/'seam-shared'/files[j].name,np.zeros(active.shape,np.uint8))
                                                        frame_scores.append(dict(frame=j,mae=0.,pixels=0))
                                                    continue
                                                original=roi.crop(get_current_img(j),box);mask=roi.crop(masks[j],box,True)
                                                if original.shape!=crop_shape:raise PipelineError('OUTPUT_INVALID','逐层ROI模型改变几何')
                                                confirmed=roi.crop(get_confirmed_mask(j),box,True)
                                                references=clean_references()
                                                restored=references.restore(original,mask,repair,confirmed=confirmed)
                                                if trusted and trusted.detail_preserve_mode=='orthogonal_luma' and scene_hp_holder and scene_hp_holder[0] is not None:
                                                    wm_hp=scene_hp_holder[0]
                                                    if j in scene_track_hps:
                                                        trk_crop=roi.crop(scene_track_hps[j],box,True)
                                                        wm_hp=np.where(np.abs(trk_crop)>np.abs(wm_hp),trk_crop,wm_hp)
                                                    restored=restore_orthogonal_luma_detail(original,restored,mask,wm_hp,edge_cfg)
                                                painted=composite(original,restored,mask)[:d-b,:c-a]
                                                if j in sample_frames:
                                                    sample=work/'model-samples'/f'task-{row["任务"]:04d}'/f'attempt-{attempt:02d}';sample.mkdir(parents=True,exist_ok=True)
                                                    roi.atomic_png(sample/f'{j:06d}-raw-model.png',repair)
                                                    shutil.copy2(task/'masks'/f'{j-left:06d}.png',sample/f'{j:06d}-exclusion.png')
                                                    roi.atomic_png(sample/f'{j:06d}-construction.png',roi.crop(masks[j]&ownership,box,True))
                                                    if j in stage_frames:
                                                        for label,value in [('source',original),('restored',restored),('composited',painted)]:roi.atomic_png(sample/f'{j:06d}-{label}.png',value)
                                                    (sample/'metadata.json').write_text(json.dumps(dict(task=row['任务'],attempt=attempt,roi=box,input=[left,right],construction=[start,end],frames=sorted(sample_frames),说明='raw-model为模型原始预测；restored为参考回填后；composited为羽化后、窗口混合与施工域回写前。'),ensure_ascii=False,indent=2))
                                                image=cv2.imread(str(dest/files[j].name));old=image[b:d,a:c]
                                                if shared.any():
                                                    disagreement=float(np.abs(old.astype(float)-painted.astype(float))[shared].mean())
                                                    maximum=max(maximum,disagreement)
                                                if construction:
                                                    if continue_unpassed:
                                                        difference=np.where(shared,np.abs(old.astype(float)-painted.astype(float)).mean(axis=2),0.)
                                                        roi.atomic_png(task/'seam-difference'/files[j].name,np.rint(difference*256).astype(np.uint16))
                                                        roi.atomic_png(task/'seam-shared'/files[j].name,shared.astype(np.uint8)*255)
                                                        frame_scores.append(dict(frame=j,mae=float(difference[shared].mean()) if shared.any() else 0.,pixels=int(np.count_nonzero(shared))))
                                                    if j in completed:
                                                        overlap=sum(t in completed for t in range(start,end))
                                                        weight=overlap_weight(j,start,end,completed)
                                                        mixed=cv2.addWeighted(painted,weight,old,1-weight,0)
                                                        painted[active&shared]=mixed[active&shared]
                                                    old[active]=painted[active];roi.atomic_png(staged/files[j].name,image)
                                                    if active.any():
                                                        band_path=work/'edge-band'/files[j].name
                                                        band=read_confirmed(band_path)[b:d,a:c] if band_path.exists() else np.zeros(active.shape,np.uint8)
                                                        source_color=original[:d-b,:c-a] if current[j]==files[j] else roi.crop(cv2.imread(str(files[j])),box)[:d-b,:c-a]
                                                        evidence=getattr(references,'last_evidence',None)
                                                        if evidence is not None:evidence={key:(value[:d-b,:c-a] if hasattr(value,'shape') else value) for key,value in evidence.items()}
                                                        active_mask=active.astype(np.uint8)*255
                                                        conf_crop=confirmed[:d-b,:c-a];prot_crop=protection_mask(work,files[j].name,shape)[b:d,a:c]
                                                        score=measure(source_color,old,active_mask,band & active_mask,edge_cfg,reference_evidence=evidence,
                                                                      confirmed=conf_crop,protected=prot_crop)
                                                        if score.get('判定') in RESIDUE_ERRORS and trusted and trusted.dense_reference_frames>len(task_refs):
                                                            dense_ev=_dense_frame_evidence(j,scene_start,scene_end,box,current,work,files,task_refs,original,mask,confirmed)
                                                            if dense_ev is not None:
                                                                dense_ev={key:(value[:d-b,:c-a] if hasattr(value,'shape') else value) for key,value in dense_ev.items()}
                                                                dense_sc=measure(source_color,old,active_mask,band & active_mask,edge_cfg,reference_evidence=dense_ev,confirmed=conf_crop,protected=prot_crop)
                                                                if (dense_sc.get('确认残留',0),dense_sc.get('待目检',0))<(score.get('确认残留',0),score.get('待目检',0)):
                                                                    score=dense_sc;evidence=dense_ev
                                                        residue_scores.append(dict(frame=j,**score,_evidence=evidence))
                                        if edge_cfg.residue_repair_rounds>0 and maximum<=settings.overlap_mae_limit:
                                            if any(s.get('判定')=='RESIDUE_FOUND' for s in residue_scores):
                                                residue_scores,maximum,repair_info=_repair_confirmed_residue_pass(
                                                    task,work,files,current,dest,staged,baseline,completed,layer_done,
                                                    masks,ownership,box,left,right,start,end,length,shape,pixels,
                                                    edge_cfg,settings,cfg,trusted,policy,residue_scores,maximum,reference_holder,
                                                    task_refs,detected,log,measure,read_confirmed,exclusion_mask,overlap_weight,
                                                    pool=get_pool() if can_parallel else None,
                                                )
                                                if repair_info is not None:trial['确认残留补修']=repair_info
                                        # 清理临时内存回调引用后再序列化 JSON
                                        for sc in residue_scores:
                                            sc.pop('_evidence',None)
                                            sc.pop('_repair_residue',None)
                                        trial['主色残留']=residue_scores
                                        trial['残留汇总']=residue_summary(residue_scores)
                                        trial['跳过参考回填帧']=skipped
                                        trial['重叠最大差异']=maximum
                                        if continue_unpassed:
                                            trial['施工帧差异']=frame_scores;trial['差图尺度']=256
                                            trial['代表施工帧']=max(frame_scores,key=lambda p:p['mae'])['frame']
                                        if maximum>settings.overlap_mae_limit:
                                            raise PipelineError('SEAM_UNCERTAIN','逐层ROI重叠差异超过原有保护',details=dict(差异=maximum,上限=settings.overlap_mae_limit))
                                        failure=residue_error(residue_scores)
                                        if failure is not None:raise failure
                                        if can_parallel and end-start>1:
                                            pass_items=[
                                                (str(staged/files[j].name),str(dest/files[j].name),str(coverage/files[j].name),
                                                 _pack_bool((masks[j]&ownership)>0),shape)
                                                for j in range(start,end)
                                            ]
                                            for _ in get_pool().imap_unordered(_commit_pass_worker,pass_items,chunksize=max(1,(end-start)//(workers*2))):
                                                pass
                                            for j in range(start,end):
                                                layer_done.setdefault(j,np.zeros(shape,np.uint8))[:] |= masks[j]&ownership
                                        else:
                                            for j in range(start,end):
                                                (staged/files[j].name).replace(dest/files[j].name)
                                                done=cv2.imread(str(coverage/files[j].name),0);roi.atomic_png(coverage/files[j].name,done|(masks[j]&ownership))
                                                layer_done.setdefault(j,np.zeros(shape,np.uint8))[:] |= masks[j]&ownership
                                        trial['通过']=True;row.update(通过=True,重叠最大差异=maximum)
                                    except Exception as exc:
                                        trial['错误码']=getattr(exc,'code',type(exc).__name__)
                                        if trial['错误码'] in {'UNTRUSTED_CONTEXT','CONTEXT_MASK_INVALID','MASK_UNCERTAIN','MASK_INVALID','RESIDUE_INPUT_INVALID'}:raise
                                        if continue_unpassed and trial['重叠最大差异'] is not None:
                                            trial['完整暂存']=compact(task,baseline,dest,files,masks,ownership,start,end,pool=get_pool() if can_parallel else None)
                                            trial['暂存路径']=str(task/'staged-delta');trial['基线路径']=str(baseline)
                                        retry=trial['错误码'] in RETRYABLE_ERRORS
                                        if trial['错误码'] in RESIDUE_ERRORS:
                                            trial['待目检说明']=('确认残留：背景证据表明该处背景不是水印主色，需补修；颜色问题换接缝窗口无效，不重试' if trial['错误码']=='RESIDUE_FOUND'
                                                             else '待目检：主色候选或无模板组缺少可判背景证据，需人工目检，不算通过；颜色问题换接缝窗口无效，不重试')
                                        if trial['错误码']=='GPU_OOM' and not oom_used:
                                            oom_used=True;retry=True
                                            if attempt+1<len(variants):
                                                variants[attempt+1]=(left,right,max(2,length//2))
                                            elif cfg.seam_retries>0:
                                                # 去重后已无后续参数时，仍保留原有的一次显存恢复。
                                                variants.append((left,right,max(2,length//2)))
                                        if not retry or attempt==len(variants)-1:
                                            row.update(错误码=trial['错误码'],耗时秒=time.monotonic()-tick)
                                            if '残留汇总' in trial:row['残留判定']=trial['残留汇总']['判定']
                                            if '待目检说明' in trial:row['待目检说明']=trial['待目检说明']
                                            if continue_unpassed:break
                                            raise
                                    finally:
                                        save_input_audit()
                                        trial['耗时秒']=time.monotonic()-attempt_tick;row['耗时秒']=time.monotonic()-tick;save()
                                        log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 参数={left}:{right}/{length} 接缝差异={trial["重叠最大差异"]} 上限={settings.overlap_mae_limit} 通过={trial["通过"]} 错误码={trial.get("错误码", "无")} 耗时秒={trial["耗时秒"]:.3f}')
                                        if '残留汇总' in trial:
                                            s=trial['残留汇总']
                                            log(f'逐层ROI 任务={row["任务"]} 尝试={attempt} 残留复检 判定={s["判定"]} 施工帧={s["有效施工帧数"]} 颜色候选={s["颜色候选"]}'
                                                f'(遮罩内{s["遮罩内颜色候选"]}/收边带{s["收边带颜色候选"]}) 区别色候选数={s["区别色候选数"]} '
                                                f'同色不适用组数={s["同色不适用组数"]} 同色不适用施工像素={s["同色不适用施工像素"]} '
                                                f'时域证实={s["时域证实背景本色"]} 外环证实={s["外环证实背景本色"]}'
                                                f' 确认残留={s["确认残留"]}(遮罩内{s["遮罩内确认残留"]}/收边带{s["收边带确认残留"]}) 待目检={s["待目检"]}(遮罩内{s["遮罩内待目检"]}/收边带{s["收边带待目检"]})'
                                                f' 无模板像素={s["无模板遮罩内像素"]+s["无模板收边带像素"]} 单帧最大未解释={s["单帧最大未解释"]} 跳过参考回填={trial.get("跳过参考回填帧")}')
                                        if continue_unpassed and not trial['通过']:
                                            for name in ['frames','masks','out']:
                                                if (task/name).exists():shutil.rmtree(task/name)
                                    if trial['通过']:
                                        for name in ['frames','masks','out','staged']:shutil.rmtree(task/name)
                                        break
                            finally:
                                if task_shm is not None:
                                    task_shm.close()
                                    task_shm.unlink()
                            if not row['通过'] and continue_unpassed:
                                candidates=[p for p in row['尝试'] if p.get('完整暂存') and p.get('重叠最大差异') is not None]
                                unpassed.append(row['任务']);row['未过块']=True
                                if candidates:
                                    # 择优：先要求接缝不超上限，再依次少确认残留、少未解释/无模板像素，最后看接缝差异；不因择优改判通过。
                                    best=min(candidates,key=lambda p:(p['重叠最大差异']>settings.overlap_mae_limit,
                                                                      p.get('残留汇总',{}).get('确认残留',0),
                                                                      p.get('残留汇总',{}).get('未解释候选',0)+p.get('残留汇总',{}).get('无模板遮罩内像素',0)+p.get('残留汇总',{}).get('无模板收边带像素',0),
                                                                      p['重叠最大差异']))
                                    best_task=work/f"roi-{row['任务']:04d}-attempt-{best['尝试']:02d}"
                                    if can_parallel and end-start>1:
                                        commit_items=[
                                            (str(best_task),str(baseline),files[j].name,str(dest/files[j].name),str(coverage/files[j].name),
                                             _pack_bool((masks[j]&ownership)>0),shape)
                                            for j in range(start,end)
                                        ]
                                        for _ in get_pool().imap_unordered(_commit_restore_worker,commit_items,chunksize=max(1,(end-start)//(workers*2))):
                                            pass
                                        for j in range(start,end):
                                            layer_done.setdefault(j,np.zeros(shape,np.uint8))[:] |= masks[j]&ownership
                                    else:
                                        for j in range(start,end):
                                            roi.atomic_png(dest/files[j].name,restore(best_task,baseline,files[j].name))
                                            done=cv2.imread(str(coverage/files[j].name),0);roi.atomic_png(coverage/files[j].name,done|(masks[j]&ownership))
                                            layer_done.setdefault(j,np.zeros(shape,np.uint8))[:] |= masks[j]&ownership
                                    row.update(最佳尝试=best['尝试'],最佳接缝=best['重叠最大差异'],采用未过结果=True,代表施工帧=best['代表施工帧'])
                                    log(f'逐层ROI 任务={row["任务"]} 未过块=True 采用最佳尝试={best["尝试"]} MAE={best["重叠最大差异"]}；继续其余任务，不标通过')
                                else:
                                    unfillable.append(row['任务']);row['采用未过结果']=False
                                    log(f'逐层ROI 任务={row["任务"]} 未过块=True 无完整模型候选；继续其余任务，禁止原帧伪造成片')
                                save()
                            log(f'逐层ROI 任务={row["任务"]} 完成 耗时秒={row["耗时秒"]:.3f}')
                        if executor is None:
                            queued = next(task_groups, None)
                finally:
                    if executor is not None:
                        prefetch_stop.set()
                        executor.shutdown(wait=True, cancel_futures=True)
                    scene_img_cache.clear()
                    scene_conf_cache.clear()
            layer_records.append(dict(图层=k,耗时秒=time.monotonic()-begin,活动帧数=sum(bool(m.any()) for m in masks)))
            current=[dest/p.name for p in files];save()
            log(f'逐层ROI 图层={k+1} 完成 耗时秒={time.monotonic()-begin:.3f}')
        normal_exit=True
    finally:
        if pool_holder[0] is not None:
            if normal_exit:
                pool_holder[0].close()
            else:
                pool_holder[0].terminate()
            pool_holder[0].join()
    (work/'composite').symlink_to(current[0].parent,target_is_directory=True)
    if unfillable:
        save();raise PipelineError('ROI_UNFILLABLE','其余任务已执行，但部分未过块无完整模型候选，禁止原帧回退',details=dict(tasks=unfillable))
    roi.assert_coverage(work,files)
    summary=save();summary.update(覆盖核验通过=True,任务数=len(records))
    (work/'roi-report.json').write_text(json.dumps(json_native(summary),ensure_ascii=False,separators=(',',':')))
    return summary
