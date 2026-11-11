"""发现全片共识之外的短时品牌：模型定位、自动品牌文本验证、图形前景补全。"""

import json
from pathlib import Path
import cv2, numpy as np
from .branding import (
    recognize,
    same_brand,
    merge_text_rows,
    palette_support,
    grow_brand_components,
)
from .multitrack import load_prior


def fraction_inside(box, other):
    a, b, c, d = box
    x, y, r, s = other
    return (
        max(0, min(c, r) - max(a, x))
        * max(0, min(d, s) - max(b, y))
        / max(1, (c - a) * (d - b))
    )


def tracking_interval(scene, last_observation, context_frames=30):
    """文字退场后仍跟踪图形尾部，避免淡出动画比文字存活更久而漏掉。"""
    return [scene[0], min(scene[1], last_observation + 1 + context_frames)]


def discover(files, detections, prior, rows, palette, root, stride=15):
    median, signal, regions, variants, metrics = load_prior(prior)
    references = []
    for region in regions:
        x, y, r, b = region["box"]
        image = np.clip(median[y:b, x:r] * 12 + 128, 0, 255).astype(np.uint8)
        references.append(recognize(cv2.resize(image, None, fx=3, fy=3), 11))
    by_frame = {i: [] for i in range(len(files))}
    for point in detections["hints"]:
        by_frame[point["frame"]].append(point["box"])
    matches = []
    for row in rows[::stride]:
        index = row["frame"]
        image = None
        for item in merge_text_rows([p for p in row["boxes"] if p["class_id"] == 2]):
            if any(fraction_inside(item["box"], b) > 0.5 for b in by_frame[index]):
                continue
            if image is None:
                image = cv2.imread(str(files[index]))
            x, y, r, b = map(round, item["box"])
            patch = image[
                max(0, y - 3) : min(image.shape[0], b + 3),
                max(0, x - 3) : min(image.shape[1], r + 3),
            ]
            if patch.size and same_brand(
                recognize(cv2.resize(patch, None, fx=3, fy=3)), references
            ):
                matches.append(dict(item, frame=index))
    if not matches:
        return None
    anchor = max(matches, key=lambda p: p["confidence"])
    related = []
    for row in rows:
        for item in row["boxes"]:
            if (
                item["class_id"] == 2
                and fraction_inside(item["box"], anchor["box"]) > 0.5
                and fraction_inside(anchor["box"], item["box"]) > 0.5
                and not any(
                    fraction_inside(item["box"], b) > 0.5
                    for b in by_frame[row["frame"]]
                )
            ):
                related.append(dict(item, frame=row["frame"]))
    seed = max(related or matches, key=lambda p: p["confidence"])
    image = cv2.imread(str(files[seed["frame"]]))
    mask = grow_brand_components(
        palette_support(image, palette), [anchor["box"], seed["box"]]
    )
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    yy, xx = np.where(mask > 0)
    if not len(xx):
        raise ValueError("模型验证了短时品牌，但图形支持不足，不能发布")
    scene = next((s, e) for s, e in detections["scenes"] if s <= seed["frame"] < e)
    start, end = tracking_interval(scene, max(p["frame"] for p in related or matches))
    prompts = root / "prompts"
    prompts.mkdir(exist_ok=True)
    cv2.imwrite(str(prompts / "seed.png"), mask)
    result = dict(
        scenes=[[start, end]],
        hints=[
            dict(
                frame=seed["frame"],
                box=[
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max() + 1),
                    int(yy.max() + 1),
                ],
                confidence=seed["confidence"],
                track_id=1,
                mask_prompt="prompts/seed.png",
            )
        ],
        automatic_short_brand=True,
        evidence=dict(
            model_anchor=seed,
            recognized_samples=len(matches),
            foreground_area=float(np.count_nonzero(mask) / mask.size),
        ),
    )
    (root / "detections.json").write_text(json.dumps(result))
    return result
