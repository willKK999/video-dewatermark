"""视频去水印系统：最终编码逐帧流式复检，不落全片高清图片。"""

from .hls_settings import is_residual_score

import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
from .process import PipelineError
from .multitrack import residual_quality
from .vision import sha256


def approved_review(path, video_digest, candidate_digest, count):
    if not path.exists():
        return False
    review = json.loads(path.read_text())
    items = review.get("逐项复核", [])
    return (
        review.get("视频摘要") == video_digest
        and review.get("候选摘要") == candidate_digest
        and sorted(p.get("候选序号", -1) for p in items) == list(range(count))
        and all(
            p.get("结论") == "误报"
            and p.get("证据文件")
            and Path(p["证据文件"]).is_file()
            and (not p.get("证据摘要") or sha256(Path(p["证据文件"])) == p["证据摘要"])
            for p in items
        )
    )


def audit(full, result, references, works, plans, assets, persistent, check):
    from scripts.hls_brand_verify import verify as brand_verify

    from .low_frequency import band, fixed_scores
    from .multitrack import load_prior

    current_assets = json.loads(Path(os.environ["HLS_V7_ASSETS"]).read_text())
    low_path = Path(current_assets["low_band_template"])
    if sha256(low_path) != current_assets["low_band_sha256"]:
        raise PipelineError("OUTPUT_INVALID", "低频复检先验摘要不一致")
    low_template = np.load(low_path)["template"]
    _, _, low_regions, _, _ = load_prior(Path(assets["prior"]))
    output = persistent / "final-audit"
    output.mkdir(exist_ok=True)
    digest = sha256(result)
    cached = output / "summary.json"
    if (
        cached.exists()
        and json.loads(cached.read_text()).get("视频摘要") == digest
        and json.loads(cached.read_text()).get("核验版本") == "v7-complete-ink-3"
    ):
        summary = json.loads(cached.read_text())
    else:
        a = cv2.VideoCapture(str(full))
        b = cv2.VideoCapture(str(result))
        count = round(b.get(cv2.CAP_PROP_FRAME_COUNT))
        hints = {}
        for p in references["hints"]:
            hints.setdefault(p["frame"], []).append(p)
        cuts = {s for s, e in references["scenes"]}
        previous = None
        metrics = []
        flashes = []
        low_samples = []

        def reader(i):
            nonlocal previous
            check()
            good, x = a.read()
            ok, y = b.read()
            if not good or not ok:
                raise PipelineError("OUTPUT_INVALID", "最终逐帧复检帧缺失")
            height, width = y.shape[:2]
            native_y = y
            x = cv2.resize(x, (1280, 720))
            y = cv2.resize(y, (1280, 720))
            if i % 60 == 0:
                low_samples.append(band(y))
            if previous is not None:
                old_x, old_y = previous
                ds = np.abs(x.astype(np.float32) - old_x)
                dy = np.abs(y.astype(np.float32) - old_y)
                dr = np.abs(
                    (y.astype(np.float32) - x) - (old_y.astype(np.float32) - old_x)
                )
                for p in hints.get(i, []):
                    xx, yy, aa, bb = [
                        round(q * (1280 / width if j % 2 == 0 else 720 / height))
                        for j, q in enumerate(p["box"])
                    ]
                    xx, yy, aa, bb = max(0, xx), max(0, yy), min(1280, aa), min(720, bb)
                    if xx >= aa or yy >= bb:
                        continue
                    source_mae = float(ds[yy:bb, xx:aa].mean())
                    result_mae = float(dy[yy:bb, xx:aa].mean())
                    repair_mae = float(dr[yy:bb, xx:aa].mean())
                    item = dict(
                        frame=i,
                        time=i / 60,
                        track_id=p["track_id"],
                        source_mae=source_mae,
                        result_mae=result_mae,
                        repair_delta_mae=repair_mae,
                        source_scene_cut=i in cuts,
                        box=p["box"],
                    )
                    metrics.append(item)
                    if (
                        i not in cuts
                        and result_mae > max(source_mae + 6, source_mae * 1.8)
                        and repair_mae > 12
                    ):
                        flashes.append(item)
            previous = (x, y)
            return native_y

        quality_file = output / "quality.json"
        try:
            residual_quality(
                output,
                [Path(f"{i:08d}.png") for i in range(count)],
                references,
                check,
                reader,
            )
        except PipelineError as exc:
            if exc.code != "RESIDUAL_UNCERTAIN":
                raise
        finally:
            a.release()
            b.release()
        quality = json.loads(quality_file.read_text())
        brands = []
        for i, (work, plan) in enumerate(zip(works, plans)):
            check()
            info = json.loads((work / "prepared.json").read_text())
            target = output / f"brand-{i+1:02d}.json"
            brand_verify(
                result,
                Path(info["coarse"]) / "frames",
                work / "native/native-report.json",
                Path(info["color"]),
                Path(info["variant"]),
                Path(assets["palette"]),
                target,
                frame_offset=plan["left"],
                extra_support=work/"brand-ink-repair/masks",
            )
            brands.append(dict(index=i + 1, **json.loads(target.read_text())))
        from .opaque_panels import verify as panel_verify

        panels = []
        for index, (work, plan) in enumerate(zip(works, plans)):
            panel = panel_verify(result, work, plan["left"], check)
            panels.append(dict(块号=index + 1, **panel))
        low_records = fixed_scores(
            low_template, np.median(np.stack(low_samples), axis=0), low_regions
        )
        candidates = []
        for item in low_records:
            if item["residual"]:
                candidates.append(dict(类型="完整图形低频", 详情=item))
        low_count = len(candidates)
        for item in quality["moving_evidence"]:
            candidates.append(dict(类型="移动模板", 详情=item))
        for item in quality["fixed_template_scores"]:
            if any(is_residual_score(p) for p in item["components"]):
                candidates.append(dict(类型="固定时域模板", 详情=item))
        if len(candidates) - low_count != quality["residual_count"]:
            raise PipelineError("OUTPUT_INVALID", "逐项残留证据计数不一致")
        for record in brands:
            for item in record["证据"]:
                candidates.append(
                    dict(类型="完整品牌", 块号=record["index"], 详情=item)
                )
        for record in panels:
            for item in record["证据"]:
                candidates.append(
                    dict(类型="不透明面板", 块号=record["块号"], 详情=item)
                )
        for item in flashes:
            candidates.append(dict(类型="修复时域突变", 详情=item))
        candidate_digest = hashlib.sha256(
            json.dumps(candidates, sort_keys=True).encode()
        ).hexdigest()
        summary = dict(
            视频摘要=digest,
            核验版本="v7-complete-ink-3",
            不透明面板核验=panels,
            低频先验摘要=current_assets["low_band_sha256"],
            低频复检=low_records,
            候选摘要=candidate_digest,
            原始候选=candidates,
            原始候选数=len(candidates),
            完整品牌核验=brands,
            多轨迹核验=quality,
            闪烁候选=flashes,
            时域全部样本=metrics,
            检查帧数=count,
        )
        cached.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    count = summary["原始候选数"]
    review = (
        approved_review(
            persistent / "final-review.json", digest, summary["候选摘要"], count
        )
        if count
        else False
    )
    summary.update(误报逐项复核通过=review, 通过=count == 0 or review)
    cached.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["通过"]:
        raise PipelineError(
            "RESIDUAL_UNCERTAIN",
            "最终编码仍有残留或时域突变候选，须按真实证据复核或重修失败块",
            review=True,
        )
    return summary
