"""v17c 只读补充验收：淡字残留数字、字幕保护核验、指定帧多列对比拼图（只写图片文件，不查看图片）。

淡字口径（通用，不含片名/坐标特例）：以确认掩膜目录中 group-faint 组作为半透明淡字域 F；
笔画像素 = 源帧有符号高通 > 阈值；某版本"残留像素" = 该版本与源帧在同一像素均高于阈值（笔画仍在）。
分 遮罩内(F∩确认掩膜)、收边带(edge-band∩F邻域)、遮罩外(F邻域∖确认掩膜) 三域统计。
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from worker.faint_watermark import dialogue_row_mask, signed_highpass


def _reader(path: Path, shape, nframes):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-an', '-frames:v', str(nframes), '-vsync', '0', '-f', 'rawvideo',
                             '-pix_fmt', 'bgr24', 'pipe:1'], stdout=subprocess.PIPE)


def _read(path: Path):
    m = cv2.imread(str(path), 0)
    return None if m is None else m > 0


def execute(a):
    labels = [x.split('=', 1)[0] for x in a.video]
    paths = [Path(x.split('=', 1)[1]) for x in a.video]
    files = sorted((a.masks / 'masks').glob('*.png'))
    n = len(files)
    h, w = cv2.imread(str(files[0]), 0).shape
    shape = (h, w, 3)
    procs = [_reader(a.source, shape, n)] + [_reader(p, shape, n) for p in paths]
    panel_frames = {int(x) for x in a.panel_frames.split(',') if x.strip()}
    box_specs = []
    for spec in a.box:
        f, rest = spec.split(':', 1)
        box_specs.append((int(f), [int(v) for v in rest.split(',')]))
    a.panels_dir.mkdir(parents=True, exist_ok=True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.ring + 1, 2 * a.ring + 1))
    faint = {lab: dict(遮罩内=0, 收边带=0, 遮罩外=0) for lab in ['source'] + labels}
    subt = {lab: dict(保护字幕区变化像素=0, 保护字幕区像素=0) for lab in labels}
    dialog_new_mask = 0
    dialog_new_mask_frames = []
    boxes_out = []
    panels = []
    faint_frames = 0
    for i, fp in enumerate(files):
        imgs = []
        for p in procs:
            data = p.stdout.read(h * w * 3)
            if len(data) != h * w * 3:
                raise ValueError(f'解码帧不完整 frame={i}')
            imgs.append(np.frombuffer(data, np.uint8).reshape(shape))
        src = imgs[0]
        m = _read(fp)
        band = _read(a.masks / 'edge-band' / fp.name)
        band = band if band is not None else np.zeros_like(m)
        F = _read(a.masks / 'group-objects' / 'group-faint' / fp.name)
        hp = {}
        if F is not None and F.any():
            faint_frames += 1
            near = cv2.dilate(F.astype(np.uint8), k) > 0
            src_hp = signed_highpass(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32))
            stroke = src_hp > a.hp_min
            doms = dict(遮罩内=F & m & stroke, 收边带=band & near & stroke, 遮罩外=near & ~m & stroke)
            for lab, img in zip(['source'] + labels, imgs):
                v_hp = src_hp if lab == 'source' else signed_highpass(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32))
                hp[lab] = v_hp
                res = v_hp > a.hp_min
                for dn, dm in doms.items():
                    faint[lab][dn] += int(np.count_nonzero(dm & res))
        for f, (x0, y0, x1, y1) in box_specs:
            if f != i:
                continue
            bm = np.zeros((h, w), bool); bm[y0:y1 + 1, x0:x1 + 1] = True
            src_hp = hp.get('source') if hp else signed_highpass(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32))
            stroke = bm & (src_hp > a.hp_min)
            row = dict(frame=f, box=[x0, y0, x1, y1], 源笔画像素=int(stroke.sum()), 确认掩膜覆盖笔画像素=int((stroke & m).sum()),
                       淡字组覆盖笔画像素=int((stroke & F).sum()) if F is not None else 0)
            for lab, img in zip(labels, imgs[1:]):
                v_hp = signed_highpass(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32))
                row[lab + '残留笔画像素'] = int(np.count_nonzero(stroke & (v_hp > a.hp_min)))
                row[lab + '框内平均高通'] = round(float(v_hp[stroke].mean()), 3) if stroke.any() else None
            row['源框内平均高通'] = round(float(src_hp[stroke].mean()), 3) if stroke.any() else None
            boxes_out.append(row)
        sub = _read(a.masks / 'subtitle-protection' / fp.name)
        if sub is not None and sub.any():
            for lab, img in zip(labels, imgs[1:]):
                d = np.abs(img.astype(np.int16) - src.astype(np.int16)).mean(axis=2) > a.change_min
                subt[lab]['保护字幕区变化像素'] += int(np.count_nonzero(d & sub))
                subt[lab]['保护字幕区像素'] += int(sub.sum())
        if a.baseline_masks is not None:
            b = _read(a.baseline_masks / 'masks' / fp.name)
            added = m & ~b if b is not None else m
            if added.any():
                dr = dialogue_row_mask(src) > 0
                c = int(np.count_nonzero(added & dr))
                dialog_new_mask += c
                if c:
                    dialog_new_mask_frames.append(i)
        if i in panel_frames:
            bar = 40
            cols = []
            plabs = ['source'] + ([x for x in a.panel_labels.split(',') if x] or labels)
            for lab, img in zip(['source'] + labels, imgs):
                if lab not in plabs:
                    continue
                c = np.zeros((h + bar, w, 3), np.uint8); c[bar:] = img
                cv2.putText(c, lab, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cols.append(c)
            out = a.panels_dir / (f'帧{i:06d}-' + '_'.join(plabs) + '.png')
            cv2.imwrite(str(out), np.hstack(cols))
            panels.append(dict(文件=str(out), 帧=i, 宽=w * len(cols), 高=h + bar))
    for p in procs:
        p.stdout.close(); p.wait(timeout=60)
    result = dict(
        口径=__doc__.strip(), 帧数=n, 含淡字组帧数=faint_frames, 高通阈值=a.hp_min, 邻域半径=a.ring,
        淡字残留像素=faint, 重点框=boxes_out,
        字幕保护=dict(说明=f'v17c 仍受保护的字幕区内，与源帧平均绝对差>{a.change_min} 的像素（编码噪声会贡献少量）', 各版本=subt,
                   新增掩膜落入真实对话行像素=dialog_new_mask, 新增掩膜落入真实对话行帧=dialog_new_mask_frames[:50]),
        拼图=panels)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(淡字残留像素=faint, 字幕保护=result['字幕保护'], 拼图数=len(panels)), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--masks', type=Path, required=True)
    p.add_argument('--baseline-masks', type=Path, default=None)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--video', action='append', required=True, help='标签=路径，可多次')
    p.add_argument('--panel-frames', default='')
    p.add_argument('--panel-labels', default='', help='拼图列（逗号分隔，默认全部）')
    p.add_argument('--box', action='append', default=[], help='帧:x0,y0,x1,y1（只用于验收定位）')
    p.add_argument('--panels-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hp-min', type=float, default=4.5)
    p.add_argument('--ring', type=int, default=9)
    p.add_argument('--change-min', type=float, default=20.0)
    execute(p.parse_args())
