"""视频去水印系统：从自动确认的完整品牌图案学习新变体，不使用影片坐标或品牌常量。"""

from pathlib import Path
import argparse, json, time
import cv2, numpy as np
from worker.brand_templates import variants, match, mask_for
from worker.branding import palette_support
from worker.matte import ink_support
from worker.multitrack import highpass

cv2.setNumThreads(2)
parser = argparse.ArgumentParser(
    description="从已确认的完整图案自动学习多尺度、翻转后的独立品牌轨迹"
)
for name in [
    "frames",
    "candidates",
    "known-masks",
    "confirmed-brand",
    "palette",
    "output",
]:
    parser.add_argument("--" + name, type=Path, required=True)
args = parser.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=True)
(root / "masks").mkdir(exist_ok=True)
f = args.frames
rows = json.loads(args.candidates.read_text())
seed = json.loads((args.confirmed_brand / "detections.json").read_text())
point = seed["hints"][0]
im = cv2.imread(str(f / f'{point["frame"]:06d}.png'))
sam = cv2.imread(str(args.confirmed_brand / "prompts/seed.png"), 0)
ink = palette_support(im, np.load(args.palette)) & sam
color_bank = variants(im, ink, [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0])
full_bank = variants(im, sam, [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0])
best = None
started = time.monotonic()
for row in rows[::15]:
    i = row["frame"]
    if any(a <= i < b for a, b in seed["scenes"]):
        continue
    evidence = [
        p for p in row["boxes"] if p["class_id"] in [0, 1] and p["confidence"] >= 0.65
    ]
    if not evidence:
        continue
    image = cv2.imread(str(f / f"{i:06d}.png"))
    known = cv2.imread(str(args.known_masks / f"{i:06d}.png"), 0)
    for p in match(image, color_bank, 0.35, 1):
        x, y, a, b = p["box"]
        # 两套模板都从同一自动目标裁剪，但前景与完整目标边界略有差别，覆盖率按完整框估算。
        if np.count_nonzero(known[y:b, x:a]) > (b - y) * (a - x) * 0.85:
            continue
        compatible = []
        for q in evidence:
            qx, qy, qa, qb = q["box"]
            inter = max(0, min(a, qa) - max(x, qx)) * max(0, min(b, qb) - max(y, qy))
            if inter >= 0.5 * (qa - qx) * (qb - qy):
                compatible.append(q)
        if not compatible:
            continue
        value = p["confidence"] * max(q["confidence"] for q in compatible)
        if best is None or value > best["value"]:
            best = dict(frame=i, point=p, value=value, evidence=compatible)
print("自动变体候选", best, flush=True)
if best is None:
    raise ValueError("未找到具有独立模型印证的品牌变体")
i = best["frame"]
p = best["point"]
image = cv2.imread(str(f / f"{i:06d}.png"))
x, y, a, b = p["box"]
g = highpass(image)
area = np.zeros(image.shape[:2], np.uint8)
area[y:b, x:a] = 255
# 匹配验证整个图文组合后，以其实际高频笔画生成稀疏掩膜，并填充独立圆形图标内部。
raw = (np.abs(g) > max(0.6, float(np.median(np.abs(g))) * 3)).astype(np.uint8) * 255
support = mask_for(image.shape[:2], color_bank, [p])
support = cv2.dilate(support, np.ones((11, 11), np.uint8))
raw &= support
mask = ink_support(g, raw, area, 2)
cv2.imwrite(str(root / "seed-mask.png"), mask)
cv2.imwrite(str(root / "seed-image.png"), image)
bank = variants(image, mask, [0.95, 1, 1.05])
records = []
for row in rows:
    index = row["frame"]
    image = cv2.imread(str(f / f"{index:06d}.png"))
    points = match(image, bank, 0.5, 2)
    mask = mask_for(image.shape[:2], bank, points)
    cv2.imwrite(str(root / "masks" / f"{index:06d}.png"), mask)
    records.append(
        dict(frame=index, matches=points, area=np.count_nonzero(mask) / mask.size)
    )
    if index % 120 == 0:
        print("完整品牌变体匹配帧数", index, flush=True)
(root / "tracks.json").write_text(
    json.dumps(
        dict(seed=best, frames=records, seconds=time.monotonic() - started),
        ensure_ascii=False,
        indent=2,
    )
)
print("新品牌变体覆盖完成", sum(bool(p["matches"]) for p in records), flush=True)
