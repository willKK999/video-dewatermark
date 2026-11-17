"""在原水印形状掩膜内融合；只用同镜头、经几何与颜色验证的原始背景。"""
import time
import cv2
import numpy as np


def context_mask(mask, radius=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(mask, kernel)


def soft_composite(original, repaired, mask, radius=3):
    # 现有形状掩膜已有至少 3 像素外扩；仅在其内侧安全边缘羽化。
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(distance/radius, 0, 1)
    support = mask > 0
    result = original.copy()
    weight = alpha[support, None]
    result[support] = np.rint(original[support]*(1-weight)+repaired[support]*weight).astype(np.uint8)
    return result


def fill_small_holes(background, confidence, mask):
    """只补平滑共识内部的小孔；不扩展回填到原水印掩膜以外。"""
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return background, confidence, 0
    h, w = mask.shape
    x1, x2 = max(0, xs.min()-4), min(w, xs.max()+5)
    y1, y2 = max(0, ys.min()-4), min(h, ys.max()+5)
    region = background[y1:y2, x1:x2].copy()
    reliable = confidence[y1:y2, x1:x2].copy()
    core = mask[y1:y2, x1:x2] > 0
    closed = cv2.morphologyEx(reliable, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    holes = ((closed > 0) & (reliable == 0) & core).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes)
    fill = np.zeros_like(holes)
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] > 64:
            continue
        component = (labels == label).astype(np.uint8)
        border = (cv2.dilate(component, np.ones((3, 3), np.uint8)) > 0) & (component == 0)
        if (border.any() and np.mean(reliable[border] > 0) >= .7
                and region[border].astype(float).std(axis=0).max() <= 8):
            fill[component > 0] = 255
    if fill.any():
        region = cv2.inpaint(region, fill, 3, cv2.INPAINT_TELEA)
        reliable[fill > 0] = 1
        background[y1:y2, x1:x2] = region
        confidence[y1:y2, x1:x2] = reliable
    return background, confidence, int(np.count_nonzero(fill))


def _crop_affine(transform, ox, oy):
    """将全图逆向仿射矩阵平移到以 (ox, oy) 为原点的裁剪子窗口。"""
    mat = transform.astype(np.float64).copy()
    mat[:, 2] = transform[:, :2] @ np.array([float(ox), float(oy)], dtype=np.float64) + transform[:, 2]
    return mat


def _build_donor_evaluator(shape, donor_crops):
    """构造按色相峰值延迟评估已对齐干净供体的回调函数，避免重复计算HSV。"""
    from worker.edge_residue import _matches_hsv

    hsv_cache = []

    def evaluate_peak(hue, cfg, painted_hsv=None):
        if not hsv_cache:
            for y0, y1, x0, x1, bgr_crop, tr_crop in donor_crops:
                hsv_cache.append((y0, y1, x0, x1, cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV), tr_crop))
        seen = np.zeros(shape, dtype=np.uint8)
        exact = np.zeros(shape, dtype=np.uint8)
        dil1 = np.zeros(shape, dtype=np.uint8)
        ker3 = np.ones((3, 3), dtype=np.uint8)
        h_int = int(hue)
        local_hue_tol = max(6, cfg.hue_radius // 2 + 2)
        min_s = max(0, int(cfg.minimum_saturation) - 10)
        min_v = max(0, int(cfg.minimum_value) - 10)
        for y0, y1, x0, x1, hsv_crop, tr_crop in hsv_cache:
            hm = _matches_hsv(hsv_crop, h_int, cfg)
            # 若干净未遮挡供体在色相峰边界内与成片自身颜色高度一致（消除量化悬崖误反证），也计入匹配。
            if painted_hsv is not None:
                ph, pw = painted_hsv.shape[:2]
                iy1, ix1 = min(y1, ph), min(x1, pw)
                if iy1 > y0 and ix1 > x0:
                    p_hsv = painted_hsv[y0:iy1, x0:ix1]
                    hc = hsv_crop[: iy1 - y0, : ix1 - x0]
                    dh = np.abs(hc[:, :, 0].astype(np.int16) - p_hsv[:, :, 0].astype(np.int16))
                    dh = np.minimum(dh, 180 - dh)
                    self_match = (dh <= local_hue_tol) & (hc[:, :, 1] >= min_s) & (hc[:, :, 2] >= min_v)
                    hm[: iy1 - y0, : ix1 - x0] |= self_match
            hm_d = cv2.dilate(hm.astype(np.uint8), ker3) > 0
            seen[y0:y1, x0:x1] += tr_crop.astype(np.uint8)
            exact[y0:y1, x0:x1] += (tr_crop & hm).astype(np.uint8)
            dil1[y0:y1, x0:x1] += (tr_crop & hm_d).astype(np.uint8)
        return seen, exact, dil1

    return evaluate_peak


class CleanReferences:
    """保守的全局+局部仿射背景回填；移动物体、遮挡和不一致参考自动退回模型。"""
    def __init__(self, images, masks, enable_local_donors=True):
        self.stats = dict(processed_frames=0, masked_pixels=0, restored_pixels=0, interpolated_pixels=0, reference_frames=len(images))
        self.timings = dict(tracking_seconds=0., alignment_seconds=0., median_seconds=0.)
        self.trusted = [255-context_mask(mask, 3) for mask in masks]
        self.images = images
        self.masks = masks
        self.grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]
        self.enable_local_donors = enable_local_donors

    def _track(self, gray, refgray, points, reftrusted=None):
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray, refgray, points, None)
        if tracked is None:
            return None
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(refgray, gray, tracked, None)
        if back is None:
            return None
        good = (status.ravel() > 0) & (reverse.ravel() > 0) & (np.linalg.norm(back-points, axis=2).ravel() < 1)
        if reftrusted is not None:
            xy=np.rint(tracked.reshape(-1,2)).astype(int)
            h,w=reftrusted.shape
            inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            allowed=np.zeros(len(xy),bool)
            allowed[inside]=reftrusted[xy[inside,1],xy[inside,0]]>0
            good &= allowed
        if good.sum() < 16:
            return None
        transform, inliers = cv2.estimateAffinePartial2D(points[good], tracked[good], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if transform is None or inliers.sum() < 12 or inliers.mean() < .65:
            return None
        return transform

    def _collect_local_donors(self, image, gray, support, confirmed, missing, background, confidence, evidence_counts, global_seen_refs):
        """对全局仿射未能覆盖的局部组件，按多尺度局部光流与干净环验证收集同镜头供体。"""
        h, w = support.shape
        tgt_trusted = context_mask(confirmed, 3) == 0
        strict_excl = cv2.dilate(confirmed.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        global_ring = tgt_trusted & (~strict_excl)
        if int(np.count_nonzero(global_ring)) < 200:
            global_ring = tgt_trusted

        num, labels = cv2.connectedComponents(cv2.dilate(support.astype(np.uint8), np.ones((3, 3), np.uint8)))
        local_crops = []

        for k in range(1, num):
            cc_support = (labels == k) & support
            cc_miss = cc_support & missing
            if not np.any(cc_miss):
                continue
            ys_c, xs_c = np.where(cc_miss)
            mcx0, mcy0, mcx1, mcy1 = int(xs_c.min()), int(ys_c.min()), int(xs_c.max()) + 1, int(ys_c.max()) + 1
            # 仅对跨度超过128像素的大连通域分块，保持常规水印/背景块完整以避免破坏局部刚性特征环
            mw_cc, mh_cc = mcx1 - mcx0, mcy1 - mcy0
            tile_w = 96 if mw_cc > 128 else mw_cc
            tile_h = 96 if mh_cc > 128 else mh_cc
            for ty0 in range(mcy0, mcy1, tile_h):
                for tx0 in range(mcx0, mcx1, tile_w):
                    ty1 = min(mcy1, ty0 + tile_h)
                    tx1 = min(mcx1, tx0 + tile_w)
                    tile_miss = np.zeros_like(cc_miss)
                    tile_miss[ty0:ty1, tx0:tx1] = cc_miss[ty0:ty1, tx0:tx1]
                    if not np.any(tile_miss):
                        continue
                    tys, txs = np.where(tile_miss)
                    cx0, cy0, cx1, cy1 = int(txs.min()), int(tys.min()), int(txs.max()) + 1, int(tys.max()) + 1
                    cw, ch = cx1 - cx0, cy1 - cy0
                    cc_crop_mask = tile_miss[cy0:cy1, cx0:cx1]
                    used_by_ref = [
                        (global_seen_refs[rf_idx][cy0:cy1, cx0:cx1].copy() if global_seen_refs[rf_idx] is not None else np.zeros((ch, cw), dtype=bool))
                        for rf_idx in range(len(self.images))
                    ]
                    seen_count = evidence_counts[cy0:cy1, cx0:cx1].astype(np.int32).copy()
                    cc_rgb_stack = []
                    cc_valid_stack = []

                    for pad in (72, 128):
                        need_sub = cc_crop_mask & (seen_count < 16)
                        if not np.any(need_sub):
                            break
                        ny_l, nx_l = np.where(need_sub)
                        nx0, ny0 = cx0 + int(nx_l.min()), cy0 + int(ny_l.min())
                        nx1, ny1 = cx0 + int(nx_l.max()) + 1, cy0 + int(ny_l.max()) + 1
                        bx0, by0, bx1, by1 = max(0, nx0 - pad), max(0, ny0 - pad), min(w, nx1 + pad), min(h, ny1 + pad)
                        erad = min(pad, 64)
                        ex0, ey0, ex1, ey1 = max(0, nx0 - erad), max(0, ny0 - erad), min(w, nx1 + erad), min(h, ny1 + erad)
                        ew, eh = ex1 - ex0, ey1 - ey0
                        e_ring_crop = tgt_trusted[ey0:ey1, ex0:ex1]
                        e_n = int(np.count_nonzero(e_ring_crop))
                        if e_n < 24:
                            continue
                        min_both = min(120, max(24, int(0.30 * e_n)))

                        sx0, sy0 = max(0, bx0 - 96), max(0, by0 - 96)
                        sx1, sy1 = min(w, bx1 + 96), min(h, by1 + 96)
                        ring_local = global_ring[by0:by1, bx0:bx1]
                        if int(np.count_nonzero(ring_local)) < 180:
                            ring_local = tgt_trusted[by0:by1, bx0:bx1]
                        pts_box = cv2.goodFeaturesToTrack(gray[by0:by1, bx0:bx1], 180, 0.01, 5, mask=ring_local.astype(np.uint8) * 255)
                        if pts_box is None or len(pts_box) < 6:
                            continue
                        pts_full = pts_box.copy()
                        pts_full[:, 0, 0] += bx0
                        pts_full[:, 0, 1] += by0
                        off_pts = pts_full.copy()
                        off_pts[:, 0, 0] -= sx0
                        off_pts[:, 0, 1] -= sy0
                        src_win = gray[sy0:sy1, sx0:sx1]

                        for rf_idx, (ref_bgr, ref_tr, ref_gr) in enumerate(zip(self.images, self.trusted, self.grays)):
                            if np.all(~(cc_crop_mask & (~used_by_ref[rf_idx]))):
                                continue
                            nxt, st, _ = cv2.calcOpticalFlowPyrLK(src_win, ref_gr[sy0:sy1, sx0:sx1], off_pts, None)
                            if nxt is None or st is None:
                                continue
                            good = st.reshape(-1) == 1
                            if int(np.count_nonzero(good)) < 6:
                                continue
                            dst_pts = nxt[good].copy()
                            dst_pts[:, 0, 0] += sx0
                            dst_pts[:, 0, 1] += sy0
                            mat, _ = cv2.estimateAffinePartial2D(pts_full[good], dst_pts, method=cv2.RANSAC, ransacReprojThreshold=2.5)
                            if mat is None:
                                continue
                            m_e = _crop_affine(mat, ex0, ey0)
                            al_tr_e = cv2.warpAffine(ref_tr, m_e, (ew, eh), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
                            both_e = e_ring_crop & al_tr_e
                            if int(np.count_nonzero(both_e)) < min_both:
                                continue
                            al_gr_e = cv2.warpAffine(ref_gr, m_e, (ew, eh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                            diffs = np.abs(gray[ey0:ey1, ex0:ex1].astype(np.float32) - al_gr_e.astype(np.float32))[both_e]
                            if float(diffs.mean()) > 6.5 or float(np.percentile(diffs, 90)) > 15.0:
                                continue
                            m_c = _crop_affine(mat, cx0, cy0)
                            al_tr_c = cv2.warpAffine(ref_tr, m_c, (cw, ch), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
                            vc = cc_crop_mask & al_tr_c & (~used_by_ref[rf_idx])
                            if not np.any(vc):
                                continue
                            al_bgr_c = cv2.warpAffine(ref_bgr, m_c, (cw, ch), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                            used_by_ref[rf_idx] |= vc
                            seen_count += vc.astype(np.int32)
                            local_crops.append((cy0, cy1, cx0, cx1, al_bgr_c, vc))
                            donor_f = al_bgr_c.astype(np.float32)
                            donor_f[~vc] = np.nan
                            cc_rgb_stack.append(donor_f)
                            cc_valid_stack.append(vc)

                    if len(cc_rgb_stack) >= 3:
                        stack = np.stack(cc_rgb_stack)
                        valid_cnt = np.sum(np.stack(cc_valid_stack), axis=0)
                        miss_crop = cc_miss[cy0:cy1, cx0:cx1]
                        elig = miss_crop & (valid_cnt >= 3)
                        if np.any(elig):
                            vals = stack[:, elig]
                            med = np.nanmedian(vals, axis=0)
                            agree = np.nanmax(np.abs(vals - med), axis=(0, 2)) <= 10
                            bg_sub = background[cy0:cy1, cx0:cx1]
                            cf_sub = confidence[cy0:cy1, cx0:cx1]
                            cnt_sub = evidence_counts[cy0:cy1, cx0:cx1]
                            bg_sub[elig] = np.clip(np.rint(med), 0, 255).astype(np.uint8)
                            cf_sub[elig] = np.maximum(cf_sub[elig], agree.astype(np.uint8))
                            cnt_sub[elig] = np.minimum(255, cnt_sub[elig].astype(np.int32) + valid_cnt[elig]).astype(np.uint8)

        return local_crops

    def reference_evidence(self,image,mask,confirmed=None):
        self._evidence_only=True
        try:self.restore(image,mask,image,confirmed=confirmed)
        finally:self._evidence_only=False
        return self.last_evidence

    def restore(self, image, mask, repaired, confirmed=None):
        self.last_evidence=dict(background=np.zeros_like(image),confidence=np.zeros(mask.shape,np.uint8),counts=np.zeros(mask.shape,np.uint8))
        self.stats["processed_frames"] += 1
        self.stats["masked_pixels"] += int(np.count_nonzero(mask))
        if not mask.any() or len(self.images) < 3:
            return repaired
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        confirmed = mask if confirmed is None else confirmed
        h, w = mask.shape
        support = mask > 0
        ys_s, xs_s = np.where(support)
        sy0, sy1, sx0, sx1 = int(ys_s.min()), int(ys_s.max()) + 1, int(xs_s.min()), int(xs_s.max()) + 1

        valid = 255-context_mask(confirmed, 5)
        points = cv2.goodFeaturesToTrack(gray, 250, .01, 6, mask=valid)
        ring = (context_mask(mask, 8) > 0) & (context_mask(confirmed, 3) == 0)
        donors = []
        global_crops = []
        global_seen_refs = [None] * len(self.images)

        if points is not None and len(points) >= 16 and ring.sum() >= 32:
            for rf_idx, (reference, reftrusted, refgray) in enumerate(zip(self.images, self.trusted, self.grays)):
                started = time.perf_counter()
                transform = self._track(gray, refgray, points, reftrusted)
                self.timings['tracking_seconds'] += time.perf_counter()-started
                if transform is None:
                    continue
                started = time.perf_counter()
                aligned = cv2.warpAffine(reference, transform, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                trusted = cv2.warpAffine(reftrusted, transform, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
                self.timings['alignment_seconds'] += time.perf_counter()-started
                usable = ring & trusted
                if usable.sum() < .35*ring.sum():
                    continue
                error = np.abs(aligned[usable].astype(np.float32)-image[usable].astype(np.float32)).mean(axis=1)
                if error.mean() > 6 or np.percentile(error, 90) > 14:
                    continue
                donor = aligned[support].astype(np.float32)
                donor[~trusted[support]] = np.nan
                donors.append(donor)
                tr_crop = support[sy0:sy1, sx0:sx1] & trusted[sy0:sy1, sx0:sx1]
                if np.any(tr_crop):
                    global_crops.append((sy0, sy1, sx0, sx1, aligned[sy0:sy1, sx0:sx1].copy(), tr_crop))
                    global_seen_refs[rf_idx] = support & trusted

        confidence = np.zeros(mask.shape, np.uint8)
        background = repaired.copy()
        evidence_counts = np.zeros(mask.shape, np.uint8)

        if len(donors) >= 3:
            started = time.perf_counter()
            stack = np.stack(donors)
            count = np.isfinite(stack[..., 0]).sum(axis=0)
            eligible = np.zeros(mask.shape, bool)
            eligible[support] = count >= 3
            if eligible.any():
                values = stack[:, count >= 3]
                median = np.nanmedian(values, axis=0)
                agreement = np.nanmax(np.abs(values-median), axis=(0, 2)) <= 10
                confidence[eligible] = agreement.astype(np.uint8)
                background[eligible] = np.clip(np.rint(median), 0, 255).astype(np.uint8)
            evidence_counts[support] = np.minimum(count, 255).astype(np.uint8)
            self.timings['median_seconds'] += time.perf_counter()-started

        local_crops = []
        if self.enable_local_donors:
            missing = support & ((confidence == 0) | (evidence_counts < 3))
            if np.any(missing):
                local_crops = self._collect_local_donors(
                    image, gray, support, confirmed, missing, background, confidence, evidence_counts, global_seen_refs
                )

        all_crops = global_crops + local_crops
        if not np.any(evidence_counts >= 3) and not all_crops:
            return repaired

        # 仅三份以上真实供体的一致中值或严格几何验证的多帧色相举证可用作背景证据；不采用插值孔洞作证。
        evidence = dict(background=background.copy(), confidence=confidence.copy(), counts=evidence_counts.copy())
        if all_crops:
            evidence['evaluate_peak'] = _build_donor_evaluator(mask.shape, all_crops)
        self.last_evidence = evidence
        if getattr(self, '_evidence_only', False):
            return repaired
        if not np.any(confidence):
            return repaired
        background, confidence, filled = fill_small_holes(background, confidence, mask)
        self.stats["interpolated_pixels"] += filled
        # 仅在一致区域内部逐渐采用参考，避免局部回填产生新接缝。
        weight = np.clip(cv2.distanceTransform(confidence, cv2.DIST_L2, 5)/3, 0, 1)
        self.stats["restored_pixels"] += int(np.count_nonzero(weight >= .5))
        result = repaired.copy()
        active = weight > 0
        alpha = weight[active, None]
        result[active] = np.rint(background[active]*alpha+repaired[active]*(1-alpha)).astype(np.uint8)
        return result


def restore_orthogonal_luma_detail(
    source,
    repaired,
    mask,
    watermark_highpass,
    edge_cfg,
    max_delta=28.0,
):
    """在确认施工掩膜内严格排除水印主笔画后，通过正交亮度投影安全恢复被模型抹平的高频背景纹理。"""
    msk = mask > 0
    if not np.any(msk):
        result = source.copy()
        return result
    from worker.edge_residue import _matches_hsv, color_peaks

    hsv_src = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    colored_src = (hsv_src[:, :, 1] >= 45) & (hsv_src[:, :, 2] >= 50)
    src_y = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    pnt_y = cv2.cvtColor(repaired, cv2.COLOR_BGR2GRAY).astype(np.float32)
    wm_hp = watermark_highpass.astype(np.float32) if watermark_highpass is not None else np.zeros_like(src_y)
    # 同时排除静态/轨迹级水印笔画与逐帧强亮度跳变水印主体，禁止将原水印笔画回注成片。
    stroke_core = (np.abs(wm_hp) > 3.0) | colored_src | (np.abs(src_y - pnt_y) > 16.0)
    excluded = cv2.dilate(stroke_core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    active = msk & (~excluded)
    if not np.any(active):
        out = source.copy()
        out[msk] = repaired[msk]
        return out

    hp_s = src_y - cv2.GaussianBlur(src_y, (0, 0), 3.0)
    hp_p = pnt_y - cv2.GaussianBlur(pnt_y, (0, 0), 3.0)
    res_y = hp_s - hp_p
    b_norm = float(np.sum(wm_hp[msk] * wm_hp[msk])) + 1e-6
    alpha = float(np.sum(res_y[msk] * wm_hp[msk])) / b_norm
    delta_y = np.clip(res_y - alpha * wm_hp, -float(max_delta), float(max_delta))

    cand = repaired.astype(np.float32)
    for ch in range(3):
        cand[:, :, ch] = np.where(active, np.clip(cand[:, :, ch] + delta_y, 0.0, 255.0), cand[:, :, ch])
    out = np.clip(np.rint(cand), 0, 255).astype(np.uint8)

    peaks = color_peaks(source, msk, edge_cfg)
    if peaks:
        pnt_hsv = cv2.cvtColor(repaired, cv2.COLOR_BGR2HSV)
        out_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        revert = np.zeros(msk.shape, dtype=bool)
        for pk in peaks:
            revert |= _matches_hsv(out_hsv, int(pk), edge_cfg) & (~_matches_hsv(pnt_hsv, int(pk), edge_cfg))
        if np.any(revert):
            out[revert] = repaired[revert]

    final = source.copy()
    final[msk] = out[msk]
    return final
