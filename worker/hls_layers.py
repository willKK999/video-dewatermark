"""视频去水印系统：每遍保持20%施工保护，多目标按已验证图层联合合成。"""

from functools import lru_cache
from pathlib import Path
import hashlib, json
import cv2, numpy as np
from .process import PipelineError


def track_bins(maximum_areas, limit=None):
    from .hls_settings import HLSSettings

    limit = HLSSettings.from_env().mask_fraction_limit if limit is None else limit
    groups = []
    totals = []
    for identity, area in sorted(maximum_areas.items(), key=lambda p: (-p[1], p[0])):
        if area > limit:
            raise PipelineError(
                "MASK_UNCERTAIN", "单条自动轨迹超过单遍面积保护", review=True
            )
        k = next(
            (i for i, total in enumerate(totals) if total + area <= limit), len(groups)
        )
        if k == len(groups):
            groups.append([])
            totals.append(0.0)
        groups[k].append(identity)
        totals[k] += area
    return groups


def validate_union(actual, layers, boxes=(), limit=None):
    from .hls_settings import HLSSettings

    limit = HLSSettings.from_env().mask_fraction_limit if limit is None else limit
    if not layers:
        raise PipelineError("MASK_UNCERTAIN", "缺少联合掩膜的图层来源", review=True)
    shape = layers[0].shape
    allowed = np.zeros(shape, np.uint8)
    for mask in layers:
        if (
            mask.shape != shape
            or not set(np.unique(mask)) <= {0, 255}
            or np.count_nonzero(mask) / mask.size > limit
        ):
            raise PipelineError(
                "MASK_UNCERTAIN", "单遍图层形状、取值或面积不符合保护要求", review=True
            )
        allowed |= mask
    for box in boxes:
        x, y, a, b = [round(v) for v in box]
        allowed[max(0, y) : min(shape[0], b), max(0, x) : min(shape[1], a)] = 255
    allowed = cv2.dilate(allowed, np.ones((9, 9), np.uint8))
    allowed = cv2.resize(
        allowed, (actual.shape[1], actual.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    if np.any((actual > 0) & (allowed == 0)):
        raise PipelineError(
            "MASK_UNCERTAIN", "联合合成超出自动轨迹与已验证图层范围", review=True
        )
    return True


def write_manifest(coarse, detected, directories, files):
    rows = []
    for directory in directories:
        maximum = 0.0
        digests = {}
        for file in files:
            path = Path(directory) / file.name
            image = cv2.imread(str(path), 0)
            if image is None or not set(np.unique(image)) <= {0, 255}:
                raise PipelineError("MASK_INVALID", "图层掩膜缺失或取值错误")
            maximum = max(maximum, float(np.count_nonzero(image) / image.size))
            if maximum > 0.2:
                raise PipelineError(
                    "MASK_UNCERTAIN", "单遍图层超过20%保护", review=True
                )
            digests[file.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            dict(
                directory=str(Path(directory).resolve()),
                maximum_fraction=maximum,
                sha256=digests,
            )
        )
    data = dict(
        method="v7-verified-layers-1",
        detected_sha256=hashlib.sha256(Path(detected).read_bytes()).hexdigest(),
        layers=rows,
    )
    (coarse / "layers.json").write_text(json.dumps(data))


class VerifiedLayers:
    def __init__(self, manifest, detected):
        self.data = json.loads(Path(manifest).read_text())
        if (
            self.data["method"] != "v7-verified-layers-1"
            or self.data["detected_sha256"]
            != hashlib.sha256(Path(detected).read_bytes()).hexdigest()
        ):
            raise PipelineError(
                "MASK_UNCERTAIN", "联合图层与自动检测版本不一致", review=True
            )
        d = json.loads(Path(detected).read_text())
        self.boxes = {}
        for p in d["hints"]:
            self.boxes.setdefault(p["frame"], []).append(p["box"])

    @lru_cache(maxsize=8)
    def masks(self, index):
        name = f"{index:06d}.png"
        result = []
        for layer in self.data["layers"]:
            path = Path(layer["directory"]) / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != layer["sha256"][name]:
                raise PipelineError(
                    "MASK_UNCERTAIN", "图层掩膜摘要发生变化", review=True
                )
            result.append(cv2.imread(str(path), 0))
        return result

    def validate(self, index, actual):
        return validate_union(actual, self.masks(index), self.boxes.get(index, []))
