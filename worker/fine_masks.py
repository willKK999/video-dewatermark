"""分离透明笔画与不透明标识，所有位置来自自动轨迹及源片可见性证据。"""

from pathlib import Path
import json
import cv2, numpy as np
from .multitrack import load_prior, highpass
from .matte import ink_support


class Supports:
    def __init__(self, prior, detections, alpha, graphic_root=None):
        self.median, self.signal, self.regions, _, _ = load_prior(prior)
        self.detections = detections
        self.graphic_root = Path(graphic_root) if graphic_root else None
        self.graphic_intervals = []
        if self.graphic_root and (self.graphic_root / "detections.json").exists():
            self.graphic_intervals = json.loads(
                (self.graphic_root / "detections.json").read_text()
            )["scenes"]
        self.by_frame = {}
        self.opaque = set()
        for region in self.regions:
            footprint = (
                cv2.resize(
                    region["mask"],
                    (alpha.shape[1], alpha.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
            if np.percentile(alpha[footprint], 80) > 0.45:
                self.opaque.add(region["id"])
        self.static = {
            r["id"]: ink_support(self.median, self.signal, r["mask"], 2)
            for r in self.regions
        }
        for track in detections["tracks"]:
            for point in track["hints"]:
                self.by_frame.setdefault(point["frame"], []).append((track, point))

    def at(self, index, image):
        height, width = image.shape[:2]
        mh, mw = self.median.shape
        feature = highpass(cv2.resize(image, (mw, mh)))
        scores = []
        projections = []
        for region in self.regions:
            x, y, a, b = region["box"]
            active = region["mask"][y:b, x:a] > 0
            u = self.median[y:b, x:a][active]
            v = feature[y:b, x:a][active]
            projections.append(float(u @ v / max(float(u @ u), 1e-6)))
            energy = float(np.linalg.norm(v))
            scores.append(
                0.0
                if energy < 1e-3 * np.sqrt(len(v))
                else float(u @ v / max(1e-6, np.linalg.norm(u) * energy))
            )
        graphic_period = any(a <= index < b for a, b in self.graphic_intervals)
        result = []
        for track, point in self.by_frame.get(index, []):
            region = next(r for r in self.regions if r["id"] == track["region"])
            if track["kind"] == "fixed" and graphic_period:
                position = next(
                    i for i, r in enumerate(self.regions) if r["id"] == region["id"]
                )
                if scores[position] < 0.02 or projections[position] < 0.05:
                    continue
            if track["kind"] == "fixed":
                base = (
                    region["mask"]
                    if region["id"] in self.opaque
                    else self.static[region["id"]]
                )
                mask = cv2.resize(
                    base, (width, height), interpolation=cv2.INTER_NEAREST
                )
            else:
                x, y, a, b = region["box"]
                base = self.static[region["id"]][y:b, x:a]
                if track["flip"]:
                    base = cv2.flip(base, 1)
                x, y, a, b = [
                    round(q * (width / mw if j % 2 == 0 else height / mh))
                    for j, q in enumerate(point["box"])
                ]
                mask = np.zeros((height, width), np.uint8)
                mask[y:b, x:a] = cv2.resize(
                    base, (a - x, b - y), interpolation=cv2.INTER_NEAREST
                )
            result.append(
                dict(
                    id=track["id"],
                    kind=track["kind"],
                    region=region["id"],
                    opaque=region["id"] in self.opaque,
                    mask=mask,
                    point=point,
                    track=track,
                )
            )
        graphic_path = (
            self.graphic_root / "masks" / f"{index:06d}.png"
            if self.graphic_root
            else None
        )
        if graphic_path and graphic_path.exists():
            mask = cv2.imread(str(graphic_path), 0)
            if mask is not None and mask.any():
                mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
                result.append(dict(id=10000, kind="graphic", opaque=True, mask=mask))
        return result
