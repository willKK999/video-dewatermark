"""七版原生交付的自动素材；只从源片、模型与时域先验提取，不接受人工框。"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
from .multitrack import load_prior
from .process import PipelineError, run


def prepare_assets(
    source, root, source_hash, check=lambda: None, prior=None, profile=None
):
    """本次从原片重建；冻结素材只能交给独立对照工具。"""
    from .v7_rebuild import rebuild

    if prior is not None or profile is not None or os.getenv("HLS_V7_ASSETS"):
        raise PipelineError("MODEL_INVALID", "禁止将旧先验或冻结素材清单作为处理输入")
    return rebuild(source, root, source_hash, check)


def brand_references(prior):
    from .branding import recognize

    median, _, regions, _, _ = load_prior(prior)
    references = []
    for region in regions:
        x, y, a, b = region["box"]
        patch = np.clip(median[y:b, x:a] * 12 + 128, 0, 255).astype(np.uint8)
        references.append(recognize(cv2.resize(patch, None, fx=3, fy=3), 11))
    return references


def prepare_graphics(work, detections, assets, fast, check=lambda: None):
    """只调用七版完整模板与逐镜头SAM，不建立新的图形判定。"""
    from .hls_v7 import graphic_masks
    from .colour_tracks import complete

    files = sorted((work / "frames").glob("*.png"))
    graphic_masks(
        files, {key: str(value) for key, value in assets.items()}, work, check
    )
    complete(
        work, {key: str(value) for key, value in assets.items()}, detections, check
    )
    if not (work / "graphic").exists():
        (work / "graphic").symlink_to(work / "graphics", target_is_directory=True)


def residual_candidates(audit, work, assets, check=lambda: None):
    """对全部编码帧保留原始模型候选；字幕仍须品牌文本印证，不能直接计作水印。"""
    from .branding import recognize, same_brand
    from .workflow import setting

    files = sorted((audit / "composite").glob("*.png"))
    indices = audit / "raw-indices.json"
    indices.write_text(json.dumps(list(range(len(files)))))
    threshold = min(
        setting("RESIDUAL_CONFIDENCE", 0.35), setting("HLS_RESIDUAL_CONFIDENCE", 0.15)
    )
    target = audit / "raw-residual-candidates.json"
    run(
        [
            os.environ["DETECT_PYTHON"],
            Path(__file__).with_name("yolo_stage.py"),
            "--frames",
            audit / "composite",
            "--weights",
            os.environ["YOLO_WEIGHTS"],
            "--indices",
            indices,
            "--all-candidates",
            "--confidence",
            str(threshold),
            "--output",
            target,
        ],
        check,
        timeout=7200,
    )
    rows = json.loads(target.read_text())
    if len(rows) != len(files) or [r["frame"] for r in rows] != list(range(len(files))):
        raise PipelineError("OUTPUT_INVALID", "编码后全画面复检候选帧数不符")
    references = brand_references(assets["prior"])
    confirmed, review = [], []
    cache = {}
    import hashlib

    for row in rows:
        check()
        image = None
        for candidate in row["boxes"]:
            if image is None:
                image = cv2.imread(str(files[row["frame"]]))
            x, y, a, b = map(round, candidate["box"])
            patch = image[
                max(0, y) : min(image.shape[0], b),
                max(0, x) : min(image.shape[1], a),
            ]
            if not patch.size:
                raise PipelineError("OUTPUT_INVALID", "文字候选超出编码画面")
            key = hashlib.sha256(patch.tobytes()).hexdigest()
            if key not in cache:
                if len(cache) >= 256:
                    cache.pop(next(iter(cache)))
                cache[key] = recognize(cv2.resize(patch, None, fx=3, fy=3))
            from .branding import readable_text

            if candidate["class_id"] in (0, 1) and not readable_text(cache[key]):
                confirmed.append(
                    dict(
                        candidate,
                        frame=row["frame"],
                        basis="专用模型非文字图标候选，须与原始证据复核",
                    )
                )
                continue
            is_brand = same_brand(cache[key], references)
            if is_brand:
                confirmed.append(
                    dict(candidate, frame=row["frame"], basis="自动品牌文本一致")
                )
            else:
                review.append(
                    dict(
                        candidate,
                        frame=row["frame"],
                        basis="文字不匹配已知品牌，不能把字幕直接计作水印",
                        original_logo_class=candidate["class_id"] in (0, 1),
                    )
                )
    result = dict(
        原始候选文件=target.name,
        原始候选总数=sum(len(row["boxes"]) for row in rows),
        需要复核=any(p.get("original_logo_class") for p in review),
        全帧数=len(files),
        检测置信度=threshold,
        残留候选=confirmed,
        文字复核依据=review,
    )
    (audit / "raw-residual-review.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    return result
