"""用相同时间相位比较七版与候选的真实纹理，记录绝对值和相对损失。"""

import json
import cv2
import numpy as np
from worker.fine_masks import Supports
from worker.matte import temporal_detail, stable_detail_mask
from scripts.hls_detail_verify import occluder_mask


def relative_clarity(args, output):
    native = json.loads(args.native_report.read_text())
    mapping = native["帧映射"]
    supports = Supports(
        args.prior,
        json.loads(args.detected.read_text()),
        np.load(args.alpha)["alpha"],
        args.graphic,
    )
    mh, mw = supports.median.shape
    captures = [
        cv2.VideoCapture(str(p)) for p in (args.source, args.baseline, args.candidate)
    ]
    rates = [c.get(cv2.CAP_PROP_FPS) for c in captures]
    if not rates[0] == rates[2] == 60 or rates[1] not in (30, 60):
        raise ValueError("相对清晰度需要六十帧源片与候选，以及三十或六十帧七版")
    cursors = [-1, -1, -1]
    samples = []
    previous = None
    try:
        for n in range(round(args.duration * rates[1])):
            frames = []
            indices = [round(n * rate / rates[1]) for rate in rates]
            for k, target in enumerate(indices):
                image = None
                while cursors[k] < target:
                    ok, image = captures[k].read()
                    if not ok:
                        raise ValueError("相对清晰度输入帧不足")
                    cursors[k] += 1
                frames.append(image)
            x, baseline, candidate = frames
            h, w = x.shape[:2]
            faint = {}
            other = occluder_mask(
                [args.graphic] if args.graphic else [], mapping[indices[2]], (h, w)
            )
            occupied = other.copy()
            for part in supports.at(mapping[indices[2]], cv2.resize(x, (mw, mh))):
                mask = cv2.resize(part["mask"], (w, h), interpolation=cv2.INTER_NEAREST)
                occupied |= mask
                if part["kind"] == "fixed" and not part["opaque"]:
                    faint[part["id"]] = mask
                else:
                    other |= mask
            if previous is not None:
                (
                    old_x,
                    old_baseline,
                    old_candidate,
                    old_faint,
                    old_other,
                    old_occupied,
                ) = previous
                for identity, mask in faint.items():
                    if identity not in old_faint:
                        continue
                    stable = stable_detail_mask(
                        mask & old_faint[identity], other, old_other
                    )
                    before = temporal_detail(
                        old_x,
                        x,
                        old_baseline,
                        baseline,
                        stable,
                        occupied | old_occupied,
                    )
                    after = temporal_detail(
                        old_x,
                        x,
                        old_candidate,
                        candidate,
                        stable,
                        occupied | old_occupied,
                    )
                    comparable = before["measurable"] and after["measurable"]
                    samples.append(
                        dict(
                            帧=n,
                            秒数=n / rates[1],
                            轨迹=identity,
                            七版=before,
                            候选=after,
                            可比较=comparable,
                            保留比差值=after["retention_ratio"]
                            - before["retention_ratio"]
                            if comparable
                            else None,
                        )
                    )
            previous = x, baseline, candidate, faint, other, occupied
    finally:
        for cap in captures:
            cap.release()
    valid = [s for s in samples if s["可比较"]]
    delta = [s["保留比差值"] for s in valid]
    report = dict(
        全部样本=samples,
        有效样本数=len(valid),
        平均保留比差值=float(np.mean(delta)) if delta else None,
        保留比差值第五百分位=float(np.percentile(delta, 5)) if delta else None,
        下降超过零点零五的样本=[s for s in valid if s["保留比差值"] < -0.05],
        说明="比较与七版相同时间相位；六十帧额外相位另做全部相邻帧检查，不使用绝对零点八门槛",
    )
    (output / "relative-clarity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    return report


def temporal_candidates(
    source, result, detected, mapping, duration, output, source_frame_offset=0
):
    """沿用整集收尾的区域时域异常规则；候选需逐项复核，不等同真实闪烁。"""
    detected = json.loads(detected.read_text())
    indices = json.loads(mapping.read_text())["帧映射"]
    hints = {}
    for point in detected["hints"]:
        hints.setdefault(point["frame"], []).append(point)
    cuts = {start for start, end in detected["scenes"]}
    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(result))
    source_fps, fps = a.get(5), b.get(5)
    if source_fps != 60 or fps not in (30, 60):
        a.release()
        b.release()
        raise ValueError("时域核验需要六十帧源片及三十或六十帧成品")
    if source_frame_offset:
        a.set(cv2.CAP_PROP_POS_FRAMES, source_frame_offset)
    cursor = source_frame_offset - 1
    previous = None
    samples, candidates = [], []
    try:
        for n in range(round(duration * fps)):
            source_index = int(n * source_fps / fps)
            original = None
            while cursor < source_index + source_frame_offset:
                ok, original = a.read()
                if not ok:
                    raise ValueError("时域核验源片帧数不足")
                cursor += 1
            good, final = b.read()
            if not good or source_index >= len(indices):
                raise ValueError("时域核验成品或映射帧数不足")
            x, y = [
                cv2.resize(image, (1280, 720)).astype(np.float32)
                for image in (original, final)
            ]
            index = indices[source_index]
            if previous is not None:
                old_x, old_y, old_index = previous
                source_delta, result_delta = np.abs(x - old_x), np.abs(y - old_y)
                repair_delta = np.abs((y - x) - (old_y - old_x))
                source_cut = index in cuts and index != old_index
                for point in hints.get(index, []):
                    left, top, right, bottom = map(round, point["box"])
                    left, top, right, bottom = (
                        max(0, left),
                        max(0, top),
                        min(1280, right),
                        min(720, bottom),
                    )
                    if right <= left or bottom <= top:
                        continue
                    region = np.s_[top:bottom, left:right]
                    ds, dy, dr = [
                        float(v[region].mean())
                        for v in (source_delta, result_delta, repair_delta)
                    ]
                    row = dict(
                        frame=n,
                        time=n / fps,
                        track_id=point["track_id"],
                        source_mae=ds,
                        result_mae=dy,
                        repair_delta_mae=dr,
                        source_scene_cut=source_cut,
                        box=point["box"],
                    )
                    samples.append(row)
                    # 与worker.hls_final_audit一致，不放宽已有无闪烁门槛。
                    if not source_cut and dy > max(ds + 6, ds * 1.8) and dr > 12:
                        candidates.append(row)
            previous = x, y, index
    finally:
        a.release()
        b.release()
    report = dict(
        全体样本=samples,
        异常候选=candidates,
        异常候选数=len(candidates),
        采样帧率=fps,
        目检="未进行",
        说明="沿用整集区域时域规则；源片切镜不判修复闪现，未决候选不得自动清零",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
