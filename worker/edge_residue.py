"""确认形状约束的主色细边与按确认水印组的主色残留复检：不向全局颜色或高亮边搜索。"""
from dataclasses import dataclass
import cv2
import numpy as np
from .process import PipelineError


@dataclass(frozen=True)
class EdgeSettings:
    shape_radius:int=2
    hue_radius:int=12
    minimum_saturation:int=60
    minimum_value:int=60
    minimum_template_pixels:int=32
    template_contrast_ratio:float=2.0
    template_minimum_fraction:float=.1
    template_reference_radius:int=8
    # 每个确认组最多建立的源色色相峰数：第一峰即主色，其余峰用于防止大面积背景色抢走主色后漏检水印色。
    template_peaks:int=3
    # 外环背景举证：外环像素距所有已确认水印/收边带至少 gap 像素，且在组外 template_reference_radius 以内。
    background_ring_gap:int=3
    # 候选像素的“邻域外环”半径（方窗半边长）。
    background_ring_window:int=16
    background_ring_minimum_pixels:int=32
    # 邻域外环（以及其中远离掩膜的外半环）同色比例达到该值，才算外环证实背景本色。
    background_proof_fraction:float=.5
    # 邻域外环同色比例不高于该值，视为外环反证（背景不是该色）。
    background_refute_fraction:float=.1
    maximum_residue_pixels:int=0
    composite_mode:str='confirmed_replace'
    require_ring_for_refinement:bool=False
    hole_completion_radius:int=18
    hole_completion_min_area:int=12
    hole_completion_min_neighbors:int=3
    hole_completion_overlap_fraction:float=.5
    residue_repair_rounds:int=0
    residue_repair_dilate_radius:int=5

    def __post_init__(self):
        if not 0<=self.shape_radius<=3 or not 0<self.hue_radius<=15:raise ValueError('形状收边范围无效')
        if not 1<=self.minimum_saturation<=255 or not 1<=self.minimum_value<=255 or self.minimum_template_pixels<1:raise ValueError('主色配置无效')
        if self.template_contrast_ratio<2 or not 0<self.template_minimum_fraction<=1 or self.template_reference_radius<3:raise ValueError('模板区别背景保护无效')
        if not 3<=self.template_peaks<=6:raise ValueError('每组源色色相峰数必须在3到6之间，不得缩小复检口径')
        if not self.shape_radius+1<=self.background_ring_gap<=self.template_reference_radius-2:
            raise ValueError('外环必须与已确认水印和收边带至少相隔shape_radius+1像素，且保留外环宽度')
        if not self.template_reference_radius<=self.background_ring_window<=2*self.template_reference_radius:
            raise ValueError('邻域外环窗口不得超过两倍外环半径，禁止远处背景举证')
        if not 32<=self.background_ring_minimum_pixels<=(2*self.background_ring_window+1)**2:
            raise ValueError('外环举证样本不得少于32像素且不得超过邻域窗口面积')
        if not .5<=self.background_proof_fraction<=1:raise ValueError('外环证实比例不得低于50%')
        if not 0<=self.background_refute_fraction<=min(.25,self.background_proof_fraction/2):
            raise ValueError('外环反证比例必须在0到25%之间且不超过证实比例的一半')
        if self.maximum_residue_pixels!=0:raise ValueError('残留门槛不得放宽')
        if self.composite_mode!='confirmed_replace':raise ValueError('禁止将源水印混回确认域')
        if not 0<=self.hole_completion_radius<=32 or self.hole_completion_min_area<1 or self.hole_completion_min_neighbors<2:
            raise ValueError('同色孔洞补全参数无效')
        if not 0<self.hole_completion_overlap_fraction<=1:raise ValueError('同色孔洞邻帧重合比例无效')
        if not 0<=self.residue_repair_rounds<=2 or not 1<=self.residue_repair_dilate_radius<=12:
            raise ValueError('确认残留补修参数无效')


def highlight_guard(image):
    # 白衣、栏杆和纸张等低饱和高亮实物边界，保护其一像素邻域。
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gradient=cv2.morphologyEx(gray,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
    bright=(hsv[...,1]<60)&(hsv[...,2]>=180)&(gradient>=12)
    return cv2.dilate(bright.astype(np.uint8),np.ones((3,3),np.uint8))>0


def _ellipse(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))


def _hue_distance(hues,hue):
    delta=np.abs(hues.astype(np.int16)-int(hue));return np.minimum(delta,180-delta)


def _circular_mode(smooth):
    """环形直方图取最大值；若最大值是平顶，取最长平顶区间的环形中心，而不是最左端。"""
    flags=smooth==smooth.max();n=len(flags)
    if flags.all():return 0
    shift=int(np.argmin(flags));rolled=np.roll(flags,-shift)  # rolled[0] 必为非最大值
    best_length=best_start=0;i=0
    while i<n:
        if rolled[i]:
            j=i
            while j<n and rolled[j]:j+=1
            if j-i>best_length:best_length,best_start=j-i,i
            i=j
        else:i+=1
    return int((best_start+shift+(best_length-1)//2)%n)


def _smooth_histogram(hues,cfg):
    hist=np.bincount(hues,minlength=180)
    return sum(np.roll(hist,k) for k in range(-cfg.hue_radius,cfg.hue_radius+1))


def _colored(hsv,cfg):
    return (hsv[...,1]>=cfg.minimum_saturation)&(hsv[...,2]>=cfg.minimum_value)


def color_peaks(image,mask,cfg,limit=None,hsv=None):
    """源色色相峰：第一峰与 color_template 相同；后续峰在去掉已覆盖色相后重算，且自身须有足够样本。"""
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV) if hsv is None else hsv
    eligible=(mask>0)&_colored(hsv,cfg);peaks=[]
    for _ in range(cfg.template_peaks if limit is None else limit):
        if eligible.sum()<cfg.minimum_template_pixels:break
        smooth=_smooth_histogram(hsv[...,0][eligible],cfg);hue=_circular_mode(smooth)
        if peaks and smooth[hue]<cfg.minimum_template_pixels:break
        peaks.append(hue);eligible&=_hue_distance(hsv[...,0],hue)>cfg.hue_radius
    return peaks


def color_template(image,mask,cfg):
    peaks=color_peaks(image,mask,cfg,limit=1)
    return peaks[0] if peaks else None


def _matches_hsv(hsv,hue,cfg):
    if hue is None:return np.zeros(hsv.shape[:2],bool)
    return (_hue_distance(hsv[...,0],hue)<=cfg.hue_radius)&_colored(hsv,cfg)


def matches(image,hue,cfg):
    if hue is None:return np.zeros(image.shape[:2],bool)
    return _matches_hsv(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),hue,cfg)


def _is_distinguishing_color(inside,outside,ring_pixels,cfg):
    """按模板最小比例与外环对比度判断该色相峰是否为水印区别色。"""
    if outside is None or ring_pixels<=0:return False
    return inside>=cfg.template_minimum_fraction and inside>=cfg.template_contrast_ratio*max(outside,1/max(1,ring_pixels))


def _evaluate_peak(hue,s_hsv,template_region,ring,ring_pixels,ring_available,cfg):
    """计算单峰组内/外环密度并判定其是否满足水印区别色要求。"""
    source_match=_matches_hsv(s_hsv,hue,cfg)
    inside=float(source_match[template_region].mean()) if template_region.any() else 0.
    outside=float(source_match[ring].mean()) if ring_pixels>0 else None
    distinguishing=_is_distinguishing_color(inside,outside,ring_pixels,cfg)
    if distinguishing:
        reason='水印区别色'
    elif outside is None:
        reason='区别度不可判（外环为空）'
    elif not ring_available:
        reason='区别度不可判（外环像素不足）'
    elif inside<cfg.template_minimum_fraction:
        reason='非区别色（组内密度低于模板下限）'
    else:
        reason='非区别色（与背景同色）'
    entry=dict(色相=hue,组内密度=round(inside,4),外环密度=None if outside is None else round(outside,4),
               外环像素=ring_pixels,是否区别色=distinguishing,原因=reason)
    return distinguishing,source_match,entry


def refine_group(image,group,confirmed,protected,cfg):
    """只在本组形状的窄带寻找自身主色，禁跨越场景/字幕/高亮保护。"""
    if group.shape!=confirmed.shape or protected.shape!=confirmed.shape:raise PipelineError('MASK_INVALID','收边几何不一致')
    hue=color_template(image,group & ~protected,cfg)
    if hue is not None:
        colored=matches(image,hue,cfg);radius=cfg.template_reference_radius
        outer=(cv2.dilate(group,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0)&(group==0)&(protected==0)
        inner=(group>0)&(protected==0)
        outer_n=int(outer.sum())
        inside=float(colored[inner].mean()) if inner.any() else 0.
        outside=float(colored[outer].mean()) if outer_n>0 else None
        if (cfg.require_ring_for_refinement and outer_n<cfg.background_ring_minimum_pixels) or not _is_distinguishing_color(inside,outside,outer_n,cfg):hue=None
    r=cfg.shape_radius
    shape=cv2.dilate(group,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))>0
    candidate=shape&(confirmed==0)&matches(image,hue,cfg)
    guard=(protected>0)|highlight_guard(image)
    added=candidate&~guard
    if np.any(added&guard):raise PipelineError('MASK_UNCERTAIN','收边侵入保护区')
    return added.astype(np.uint8)*255,dict(hue=hue,候选像素=int(candidate.sum()),新增像素=int(added.sum()),保护拒绝像素=int((candidate&guard).sum()),保护交集像素=int((added&guard).sum()))


def composite(original,repaired,mask):
    # 已确认水印的内侧边缘也不可复用原水印颜色。施工域外由调用方逐像素保持。
    result=original.copy();result[mask>0]=repaired[mask>0];return result


def complete_group_color_holes(image,group,confirmed,protected,neighbor_groups,cfg):
    """在确认水印组附近按区别色模板与邻帧组掩膜重合证据，通用补全单帧漏检的非保护色块孔洞。"""
    if group.shape!=confirmed.shape or protected.shape!=confirmed.shape:raise PipelineError('MASK_INVALID','同色孔洞补全几何不一致')
    if cfg.hole_completion_radius<=0 or not neighbor_groups:
        return np.zeros_like(confirmed,dtype=np.uint8),dict(hues=[],候选像素=0,新增像素=0,保护交集像素=0)
    group_b=group>0;prot_b=protected>0;inner=group_b&~prot_b
    if not np.any(inner):
        return np.zeros_like(confirmed,dtype=np.uint8),dict(hues=[],候选像素=0,新增像素=0,保护交集像素=0)
    peaks=color_peaks(image,inner,cfg)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    rad=cfg.template_reference_radius
    outer=(cv2.dilate(group_b.astype(np.uint8)*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,2*rad+1)))>0)&(~group_b)&(~prot_b)
    outer_n=int(outer.sum())
    dist_peaks=[];colored_union=np.zeros(group.shape,dtype=bool)
    for pk in peaks:
        cm=_matches_hsv(hsv,int(pk),cfg)
        inside=float(cm[inner].mean())
        outside=float(cm[outer].mean()) if outer_n>0 else None
        if _is_distinguishing_color(inside,outside,outer_n,cfg):
            dist_peaks.append(int(pk));colored_union|=cm
    if not dist_peaks:
        return np.zeros_like(confirmed,dtype=np.uint8),dict(hues=[],候选像素=0,新增像素=0,保护交集像素=0)
    r=cfg.hole_completion_radius
    near=cv2.dilate(group_b.astype(np.uint8)*255,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))>0
    guard=prot_b|highlight_guard(image)
    cand=colored_union&near&(confirmed==0)&(~guard)
    if not np.any(cand):
        return np.zeros_like(confirmed,dtype=np.uint8),dict(hues=dist_peaks,候选像素=0,新增像素=0,保护交集像素=0)
    ys0,xs0=np.where(group_b);cx0,cy0=float(xs0.mean()),float(ys0.mean());area0=float(len(ys0))
    nbr_info=[]
    for nm in neighbor_groups:
        nb=nm>0
        if not np.any(nb):continue
        yn,xn=np.where(nb)
        nbr_info.append((nb,float(xn.mean()),float(yn.mean()),float(len(yn))))
    if len(nbr_info)<cfg.hole_completion_min_neighbors:
        return np.zeros_like(confirmed,dtype=np.uint8),dict(hues=dist_peaks,候选像素=int(cand.sum()),新增像素=0,保护交集像素=0)
    ncc,lbl,st,_=cv2.connectedComponentsWithStats(cand.astype(np.uint8))
    added=np.zeros(group.shape,dtype=bool)
    h,w=group.shape[:2]
    for k in range(1,ncc):
        if int(st[k,cv2.CC_STAT_AREA])<cfg.hole_completion_min_area:continue
        cc=(lbl==k);cy_c,cx_c=np.where(cc);c_area=float(len(cy_c))
        direct_hits=0;shifted_hits=0
        for nb,cxn,cyn,arean in nbr_info:
            if float(np.count_nonzero(nb&cc))/c_area>=cfg.hole_completion_overlap_fraction:direct_hits+=1
            if 0.3*area0<arean<3.0*area0:dx,dy=int(round(cxn-cx0)),int(round(cyn-cy0))
            else:dx,dy=0,0
            ny,nx=cy_c+dy,cx_c+dx
            ok=(ny>=0)&(ny<h)&(nx>=0)&(nx<w)
            if ok.any() and float(np.count_nonzero(nb[ny[ok],nx[ok]]))/c_area>=cfg.hole_completion_overlap_fraction:shifted_hits+=1
        if direct_hits>=cfg.hole_completion_min_neighbors or shifted_hits>=cfg.hole_completion_min_neighbors:added|=cc
    if np.any(added&guard):raise PipelineError('MASK_UNCERTAIN','同色孔洞补全侵入保护区')
    return added.astype(np.uint8)*255,dict(hues=dist_peaks,候选像素=int(cand.sum()),新增像素=int(added.sum()),保护交集像素=int((added&guard).sum()))


def _binary(value,name,shape):
    if value is None:return None
    if value.shape!=shape or value.dtype!=np.uint8 or not set(np.unique(value))<={0,255}:
        raise PipelineError('RESIDUE_INPUT_INVALID',f'残留复检{name}未与施工裁块对齐或取值无效')
    return value>0


def _box_count(values,size):
    return cv2.boxFilter(values.astype(np.float32),-1,(size,size),normalize=False,borderType=cv2.BORDER_CONSTANT)


RESIDUE_SCOPE=('判定口径：按确认水印组（施工掩膜与全部确认水印经shape_radius膨胀后的连通域）用冻结源图取最多template_peaks个色相峰（第一峰为主色，平顶取环形中心）；'
               '逐峰先做模板区别度检验：组内密度>=template_minimum_fraction且>=template_contrast_ratio*max(外环密度,1/max(1,外环像素))才为水印区别色，'
               '非区别色跳过；若组内无区别色且外环像素>=background_ring_minimum_pixels则记为同色不适用，外环不足则判RESIDUE_UNCERTAIN（待目检）。'
               '颜色候选为补画结果在组内施工域（含收边带）中与组区别色匹配的像素。背景本色证据二选一：'
               '时域=CleanReferences至少3份一致可信参考（confidence>0且counts>=3）的干净背景本身在hue_radius内匹配该色；'
               '外环=同帧源图中距全部确认水印/收边带至少background_ring_gap（>=shape_radius+1）像素、非保护区、在组外template_reference_radius内的'
               '邻域外环（半径background_ring_window，样本>=background_ring_minimum_pixels，且外半环样本>=一半）同色比例均>=background_proof_fraction。'
               '未解释候选若有时域反证或邻域外环同色比例<=background_refute_fraction则判RESIDUE_FOUND（确认残留），'
               '否则（无足够可判背景证据，或外环不足无法证明主色区别度，或组有>=10%彩色样本但不足建立主色模板）判RESIDUE_UNCERTAIN（待目检，绝不算通过）。'
               '允许残留门槛maximum_residue_pixels固定为0；遮罩内=施工掩膜去掉收边带，收边带单列。')


def measure(source,painted,mask,band,cfg,reference_evidence=None,confirmed=None,protected=None,return_masks=False):
    """按确认水印组复检主色残留，三态：通过 / RESIDUE_FOUND（确认残留）/ RESIDUE_UNCERTAIN（待目检）。

    source：冻结源图裁块（BGR）；painted：写回后的补画裁块；mask：当前施工掩膜；band：收边带（须在施工掩膜内）；
    reference_evidence：CleanReferences 时域证据（background/confidence/counts）；confirmed：全部确认水印并集（含收边带）；
    protected：场景/字幕保护区；return_masks：为 True 时额外返回 '_masks' 掩膜字典。后三者缺省分别退化为施工掩膜、空集与 False。
    """
    if source is None or painted is None or mask is None or band is None:
        raise PipelineError('RESIDUE_INPUT_INVALID','残留复检输入缺失')
    if source.ndim!=3 or source.shape[2]!=3 or painted.shape!=source.shape or mask.ndim!=2 or band.shape!=mask.shape or mask.shape!=source.shape[:2]:
        raise PipelineError('RESIDUE_INPUT_INVALID','残留复检要求对齐的三通道BGR和二维掩膜')
    if source.dtype!=np.uint8 or painted.dtype!=np.uint8 or mask.dtype!=np.uint8 or band.dtype!=np.uint8 or not set(np.unique(mask))<={0,255} or not set(np.unique(band))<={0,255}:
        raise PipelineError('RESIDUE_INPUT_INVALID','残留复检输入类型或掩膜取值无效')
    if not np.any(mask):raise PipelineError('RESIDUE_INPUT_INVALID','残留复检施工样本为空，禁止按零残留通过')
    if np.any((band>0)&(mask==0)):raise PipelineError('RESIDUE_INPUT_INVALID','收边带超出当前施工域')
    shape=mask.shape
    known=_binary(confirmed,'确认水印并集',shape);guard=_binary(protected,'保护区',shape)
    if guard is None:guard=np.zeros(shape,bool)
    temporal=None;donor_eval=None
    if reference_evidence is not None:
        if not isinstance(reference_evidence,dict) or not {'background','confidence','counts'}<=reference_evidence.keys():
            raise PipelineError('RESIDUE_INPUT_INVALID','干净参考证据字段缺失')
        background=reference_evidence['background'];confidence=reference_evidence['confidence'];counts=reference_evidence['counts']
        if background.shape!=source.shape or confidence.shape!=shape or counts.shape!=shape or background.dtype!=np.uint8 or confidence.dtype!=np.uint8 or counts.dtype!=np.uint8:
            raise PipelineError('RESIDUE_INPUT_INVALID','干净参考证据未与施工裁块对齐')
        trusted=(confidence>0)&(counts>=3)
        if trusted.any():temporal=(trusted,cv2.cvtColor(background,cv2.COLOR_BGR2HSV))
        if callable(reference_evidence.get('evaluate_peak')):
            donor_eval=reference_evidence['evaluate_peak']
    construction=mask>0;edge=band>0;interior=construction&~edge
    watermark=construction|edge|(known if known is not None else False)
    source_hsv=cv2.cvtColor(source,cv2.COLOR_BGR2HSV);painted_hsv=cv2.cvtColor(painted,cv2.COLOR_BGR2HSV)
    h,w=shape;R=cfg.template_reference_radius;L=cfg.background_ring_window;gap=cfg.background_ring_gap;middle=(gap+R+1)//2
    # 距全部已确认水印（含收边带）的距离；外环只取距离≥gap的像素，紧贴掩膜的光晕不能当背景证据。
    distance=cv2.distanceTransform((~watermark).astype(np.uint8),cv2.DIST_L2,5)
    far=(distance>=gap)&~guard;far_outer=(distance>=middle)&~guard
    count,labels=cv2.connectedComponents(cv2.dilate(watermark.astype(np.uint8),_ellipse(cfg.shape_radius)),connectivity=8)
    candidates=np.zeros(shape,bool);temporal_proved=np.zeros(shape,bool);ring_proved=np.zeros(shape,bool);conflict=np.zeros(shape,bool)
    unexplained=np.zeros(shape,bool);residue=np.zeros(shape,bool);untemplated=np.zeros(shape,bool);ring_uncertain=np.zeros(shape,bool)
    same_color_mask=np.zeros(shape,bool);groups=[];reasons=[]
    colorless=0;same_color_groups=0;minimum_outer=cfg.background_ring_minimum_pixels//2
    for label in range(1,count):
        whole=labels==label
        if not np.any(whole&construction):continue
        ys,xs=np.nonzero(whole);margin=R+L+1
        y0,y1=max(0,ys.min()-margin),min(h,ys.max()+margin+1);x0,x1=max(0,xs.min()-margin),min(w,xs.max()+margin+1)
        view=(slice(y0,y1),slice(x0,x1));group=whole[view];work=group&construction[view]
        template_region=group&watermark[view]&~guard[view];s_hsv=source_hsv[view];p_hsv=painted_hsv[view]
        row=dict(组=len(groups)+1,外接框=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1],施工像素=int(work.sum()),
                 遮罩内像素=int((work&interior[view]).sum()),收边带像素=int((work&edge[view]).sum()),模板像素=int(template_region.sum()))
        eligible=int((template_region&_colored(s_hsv,cfg)).sum());row['彩色样本']=eligible
        peaks=color_peaks(source[view],template_region,cfg,hsv=s_hsv);row['主色色相']=peaks
        if not peaks:
            row.update(区别色列表=[],非区别色列表=[])
            if eligible>0 and eligible>=cfg.template_minimum_fraction*max(1,row['模板像素']):
                row['状态']='待目检：组内有彩色样本但不足以建立主色模板';untemplated[view]|=work
                reasons.append(f"组{row['组']}彩色样本{eligible}<{cfg.minimum_template_pixels}，无有效主色模板")
            else:
                row['状态']='无彩色组：源组彩色样本不足10%且少于模板下限，主色复检不适用';colorless+=1
            groups.append(row);continue
        group_watermark=group&watermark[view]
        ring_distance=cv2.distanceTransform((~group_watermark).astype(np.uint8),cv2.DIST_L2,5)
        ring=(ring_distance<=R)&far[view];ring_outer=ring&far_outer[view]
        ring_pixels=int(ring.sum());ring_available=ring_pixels>=cfg.background_ring_minimum_pixels
        size=2*L+1;ring_n=_box_count(ring,size);outer_n=_box_count(ring_outer,size)
        judgeable=ring_n>=cfg.background_ring_minimum_pixels
        g_cand=np.zeros(group.shape,bool);g_tp=np.zeros(group.shape,bool);g_rp=np.zeros(group.shape,bool);g_conflict=np.zeros(group.shape,bool)
        g_unexplained=np.zeros(group.shape,bool);g_residue=np.zeros(group.shape,bool)
        contrast=[];distinguishing=[];non_distinguishing=[]
        for hue in peaks:
            is_distinguishing,source_match,entry=_evaluate_peak(hue,s_hsv,template_region,ring,ring_pixels,ring_available,cfg)
            contrast.append(entry)
            if not is_distinguishing:
                non_distinguishing.append(hue)
                continue
            distinguishing.append(hue)
            cand=work&_matches_hsv(p_hsv,hue,cfg)
            if not cand.any():continue
            ring_c=_box_count(ring&source_match,size);outer_c=_box_count(ring_outer&source_match,size)
            ring_proof=judgeable&(ring_c>=cfg.background_proof_fraction*ring_n)&(outer_n>=minimum_outer)&(outer_c>=cfg.background_proof_fraction*outer_n)
            ring_refute=judgeable&(ring_c<=cfg.background_refute_fraction*ring_n)
            t_proof=t_refute=np.zeros(group.shape,bool)
            if temporal is not None:
                trusted=temporal[0][view];background_match=_matches_hsv(temporal[1][view],hue,cfg)
                t_proof=trusted&background_match;t_refute=trusted&~background_match
            if donor_eval is not None:
                seen_f,exact_f,dil1_f=donor_eval(hue,cfg,painted_hsv)
                seen=seen_f[view];exact_h=exact_f[view];dil1_h=dil1_f[view]
                hue_proof=(seen>=3)&(exact_h>=3)&(exact_h*2>=seen)
                adj_strict=cv2.dilate(hue_proof.astype(np.uint8),np.ones((3,3),np.uint8))>0
                hue_proof|=adj_strict&(seen>=3)&(dil1_h>=3)&(dil1_h*2>=seen)
                # 仅允许严格RGB一致共识触发时域反证；局部色相举证仅作正例背景证实，避免薄边插值抖动误将待目检升级为确认残留。
                t_proof=t_proof|hue_proof
                t_refute=t_refute&~hue_proof
            # 时域干净参考反证优先：外环证实不能覆盖“该处背景本色不是主色”的时域证据。
            left=cand&~(t_proof|(ring_proof&~t_refute))
            g_cand|=cand;g_tp|=cand&t_proof;g_rp|=cand&ring_proof;g_conflict|=cand&ring_proof&t_refute&~t_proof
            g_unexplained|=left;g_residue|=left&(t_refute|ring_refute)
        g_pending=g_unexplained&~g_residue
        row.update(主色模板对比=contrast,区别色列表=distinguishing,非区别色列表=non_distinguishing,
                   外环像素=ring_pixels,颜色候选=int(g_cand.sum()),区别色候选数=int(g_cand.sum()),
                   时域证实=int(g_tp.sum()),外环证实=int(g_rp.sum()),
                   外环证实但时域反证=int(g_conflict.sum()),确认残留=int(g_residue.sum()),待目检=int(g_pending.sum()))
        if distinguishing:
            row['状态']='确认残留' if g_residue.any() else '待目检：候选处既无时域也无外环可判证据' if g_pending.any() else '通过'
            if g_pending.any():reasons.append(f"组{row['组']}有{int(g_pending.sum())}个主色候选无可判背景证据")
            candidates[view]|=g_cand;temporal_proved[view]|=g_tp;ring_proved[view]|=g_rp;conflict[view]|=g_conflict
            unexplained[view]|=g_unexplained;residue[view]|=g_residue
        elif ring_available:
            row['状态']='颜色复检不适用：水印源色与周围背景同色'
            same_color_groups+=1;same_color_mask[view]|=work
        else:
            row['状态']='待目检：外环不足无法证明主色区别度'
            row['待目检']=int(work.sum());ring_uncertain[view]|=work
            reasons.append(f"组{row['组']}外环像素{ring_pixels}<{cfg.background_ring_minimum_pixels}，无法证明主色区别度")
        groups.append(row)
    pending=(unexplained&~residue)|ring_uncertain
    def split(values):return int(np.count_nonzero(values&interior)),int(np.count_nonzero(values&edge))
    ci,ce=split(candidates);ri,re=split(residue);pi,pe=split(pending);ui,ue=split(unexplained);ti,te=split(untemplated)
    unexplained_total=ui+ue
    if residue.any():verdict='RESIDUE_FOUND'
    elif unexplained_total>cfg.maximum_residue_pixels or untemplated.any() or ring_uncertain.any():verdict='RESIDUE_UNCERTAIN'
    else:verdict='通过'
    result=dict(判定=verdict,通过=verdict=='通过',待目检原因=reasons,组=groups,组数=len(groups),无彩色组数=colorless,
                同色不适用组数=same_color_groups,同色不适用施工像素=int(np.count_nonzero(same_color_mask)),
                无模板组数=sum(1 for g in groups if g['状态'].startswith('待目检：组内')),无模板遮罩内像素=ti,无模板收边带像素=te,
                有效施工像素=int(np.count_nonzero(construction)),遮罩内像素=int(np.count_nonzero(interior)),边缘带像素=int(np.count_nonzero(edge)),统计形状=list(painted.shape),
                遮罩内颜色候选=ci,收边带颜色候选=ce,颜色候选=ci+ce,区别色候选数=ci+ce,
                时域证实背景本色=int(np.count_nonzero(temporal_proved)),外环证实背景本色=int(np.count_nonzero(ring_proved)),
                背景本色证实=int(np.count_nonzero(temporal_proved|ring_proved)),外环证实但时域反证=int(np.count_nonzero(conflict)),
                遮罩内确认残留=ri,收边带确认残留=re,确认残留=ri+re,遮罩内待目检=pi,收边带待目检=pe,待目检=pi+pe,
                遮罩内残留=ui,收边带残留=ue,未解释候选=unexplained_total,时域证据可用=(temporal is not None or donor_eval is not None),
                阈值=cfg.maximum_residue_pixels,阈值单位='像素/施工帧',口径=RESIDUE_SCOPE)
    if return_masks:
        result['_masks']=dict(确认残留=residue,待目检=pending,颜色候选=candidates)
    return result
