"""模型确认的不透明面板独立补修；位置、轮廓和生命周期全部来自源片。"""

from .hls_settings import is_residual_score

import json, os, shutil, time, subprocess
from functools import lru_cache
from pathlib import Path
import cv2, numpy as np
from .colour_tracks import groups
from .multitrack import highpass, component_scores
from .process import PipelineError, run
from . import vision
from scripts.hls_chunk_check import mux_core_audio, audio_hash


def panel_support(image, box):
    h, w = image.shape[:2]
    x, y, a, b = [round(v) for v in box]
    pad = max(8, round((b - y) * 0.08))
    x, y, a, b = max(0, x - pad), max(0, y - pad), min(w, a + pad), min(h, b + pad)
    out = np.zeros((h, w), np.uint8)
    if x >= a or y >= b:
        return out
    patch = image[y:b, x:a]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if np.mean((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 90)) < 0.01:
        return out
    top, bottom, left, right = (
        (4 if y == 0 else 0),
        (4 if b == h else 0),
        (4 if x == 0 else 0),
        (4 if a == w else 0),
    )
    padded = cv2.copyMakeBorder(
        patch, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)
    )
    edges = cv2.Canny(padded, 40, 100)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    filled = np.zeros(edges.shape, np.uint8)
    area = (box[2] - box[0]) * (box[3] - box[1])
    for contour in [
        *cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0],
        *cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0],
    ]:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        ca = cv2.contourArea(contour)
        if (
            not 0.025 * area < ca < 1.4 * area
            or min(cw, ch) < 12
            or ca / (cw * ch) < 0.6
        ):
            continue
        poly = cv2.approxPolyDP(contour, 0.018 * cv2.arcLength(contour, True), True)
        if not 4 <= len(poly) <= 12:
            continue
        cv2.drawContours(filled, [cv2.convexHull(contour)], -1, 255, -1)
    out[y:b, x:a] = filled[top : top + b - y, left : left + a - x]
    # 面板中滚动文字会切断边缘；闭合面板相邻的饱和底色组件可补齐整条色带。
    local = out[y:b, x:a]
    near = cv2.dilate(local, np.ones((17, 17), np.uint8)) > 0
    coloured = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 90)).astype(np.uint8) * 255
    for contour in cv2.findContours(
        coloured, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]:
        cx, cy, cw, ch = cv2.boundingRect(contour)
        ca = cv2.contourArea(contour)
        if ca < area * 0.015 or ca / (cw * ch) < 0.4 or ch > (box[3] - box[1]) * 0.7:
            continue
        component = np.zeros(local.shape, np.uint8)
        cv2.drawContours(component, [cv2.convexHull(contour)], -1, 255, -1)
        if np.count_nonzero((component > 0) & near) >= min(20, ca * 0.05):
            local |= component
    if np.count_nonzero(out) < area * 0.35:
        return np.zeros_like(out)
    radius = max(4, round((box[3] - box[1]) * 0.06))
    return cv2.dilate(out, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8))


def observations(rows, files):
    result = []
    for row, file in zip(rows, files):
        boxes = [
            p for p in row["boxes"] if p["class_id"] == 0 and p["confidence"] >= 0.15
        ]
        selected = []
        while boxes:
            seed = max(boxes, key=lambda p: p["confidence"])
            boxes.remove(seed)
            if seed["confidence"] < 0.55:
                break
            box = list(seed["box"])
            joined = True
            while joined:
                joined = False
                for p in boxes[:]:
                    q = p["box"]
                    inter = max(0, min(q[2], box[2]) - max(q[0], box[0])) * max(
                        0, min(q[3], box[3]) - max(q[1], box[1])
                    )
                    h = min(q[3] - q[1], box[3] - box[1])
                    gap = max(0, max(q[0], box[0]) - min(q[2], box[2]))
                    near = (
                        gap <= h * 0.35
                        and abs((q[1] + q[3]) - (box[1] + box[3])) / 2 <= h * 0.6
                    )
                    if (inter > 0 or near) and max(
                        q[3] - q[1], box[3] - box[1]
                    ) < 2.5 * h:
                        box = [
                            min(box[0], q[0]),
                            min(box[1], q[1]),
                            max(box[2], q[2]),
                            max(box[3], q[3]),
                        ]
                        boxes.remove(p)
                        joined = True
            if (box[2] - box[0]) * (box[3] - box[1]) > 1280 * 720 * 0.2:
                continue
            image = cv2.imread(str(file))
            mask = panel_support(image, box)
            if mask.any():
                selected.append(dict(box=box, confidence=seed["confidence"]))
        result.append(dict(frame=row["frame"], color_matches=selected))
    return result


def track_envelope(points):
    boxes = [p["box"] for p in points]
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def pair_score(source, result, mask):
    yy, xx = np.where(mask > 0)
    if not len(xx):
        return dict(correlation=0.0, relative_amplitude=0.0, components=[])
    x, y, a, b = int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)
    parts = component_scores(
        highpass(source)[y:b, x:a], highpass(result)[y:b, x:a], mask[y:b, x:a]
    )
    bad = [p for p in parts if is_residual_score(p)]
    point = max(
        bad or parts,
        key=lambda p: abs(p["correlation"]),
        default=dict(correlation=0.0, relative_amplitude=0.0),
    )
    return dict(
        correlation=point["correlation"],
        relative_amplitude=point["relative_amplitude"],
        components=parts,
    )


def patch(work, assets_path):
    if not (work / "all-yolo.json").exists():
        return
    started = time.monotonic()
    info = json.loads((work / "prepared.json").read_text())
    native = work / "native"
    final = native / "chunk1.mp4"
    root = work / "opaque-panels"
    method = "完整面板分部件与时域包络-4"
    if root.exists() and any(root.iterdir()):
        marker = root / "method.json"
        if not marker.exists() or json.loads(marker.read_text()).get("方法") != method:
            root.rename(work / f"opaque-panels-before-v3-{time.time_ns()}")
    root.mkdir(exist_ok=True)
    (root / "method.json").write_text(json.dumps(dict(方法=method), ensure_ascii=False))
    report_path = root / "report.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text())
        current = vision.sha256(final)
        if previous.get("输出摘要") == current:
            return
        descendant = work / "complete-graphics/report.json"
        if descendant.exists():
            child = json.loads(descendant.read_text())
            if (
                child.get("基底视频摘要") == previous.get("输出摘要")
                and child.get("修复后视频摘要") == current
            ):
                return
    files = sorted((work / "frames").glob("*.png"))
    d = json.loads(Path(info["detected"]).read_text())
    meta = json.loads((native / "native-report.json").read_text())
    mapping = meta["帧映射"]
    detected_path = root / "observations.json"
    if not detected_path.exists():
        detected_path.write_text(
            json.dumps(
                observations(json.loads((work / "all-yolo.json").read_text()), files)
            )
        )
    tracks = groups(json.loads(detected_path.read_text()), d["scenes"])
    active = []
    evidence = []
    cap = cv2.VideoCapture(str(final))
    for number, track in enumerate(tracks):
        scores = []
        envelope = track_envelope(track["points"])
        first, last = track["points"][0]["frame"], track["points"][-1]["frame"]
        indices = sorted(
            {
                first,
                last,
                track["seed"]["frame"],
                *range(((first + 14) // 15) * 15, last + 1, 15),
            }
        )
        for idx in indices:
            positions = [n for n, m in enumerate(mapping) if m == idx]
            if not positions:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, positions[0])
            ok, image = cap.read()
            if not ok:
                raise PipelineError("OUTPUT_INVALID", "面板补修触发核验帧不足")
            source = cv2.imread(str(files[idx]))
            mask = panel_support(source, envelope)
            scores.append(
                dict(
                    frame=idx,
                    **pair_score(source, cv2.resize(image, (1280, 720)), mask),
                )
            )
        residual = any(is_residual_score(p) for p in scores)
        evidence.append(
            dict(
                轨迹=number,
                范围=track["scene"],
                模型观测数=len(track["points"]),
                复检=scores,
                需要补修=residual,
            )
        )
        if residual:
            active.append((number, track))
    cap.release()
    if not active:
        report_path.write_text(
            json.dumps(
                dict(无需补修=True, 证据=evidence, 输出摘要=vision.sha256(final)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    all_masks = root / "masks"
    all_masks.mkdir(exist_ok=True)
    for file in files:
        cv2.imwrite(str(all_masks / file.name), np.zeros((720, 1280), np.uint8))
    for number, track in active:
        target = root / f"track-{number:03d}"
        target.mkdir(exist_ok=True)
        (target / "masks").mkdir(exist_ok=True)
        (target / "prompts").mkdir(exist_ok=True)
        if not (target / "frames").exists():
            (target / "frames").symlink_to(work / "frames", target_is_directory=True)
        seed = track["seed"]
        envelope = track_envelope(track["points"])
        prompt = panel_support(cv2.imread(str(files[seed["frame"]])), envelope)
        cv2.imwrite(str(target / "prompts/seed.png"), prompt)
        start = max(track["scene"][0], track["points"][0]["frame"] - 15)
        end = min(track["scene"][1], track["points"][-1]["frame"] + 16)
        detect = dict(
            scenes=[[start, end]],
            hints=[
                dict(
                    frame=seed["frame"],
                    box=envelope,
                    track_id=1,
                    mask_prompt="prompts/seed.png",
                )
            ],
        )
        (target / "detections.json").write_text(json.dumps(detect))
        if not (target / "complete.json").exists():
            run(
                [
                    os.environ["TRACK_PYTHON"],
                    str(Path(__file__).with_name("sam_stage.py")),
                    "--root",
                    target,
                ],
                timeout=1800,
            )
            (target / "complete.json").write_text("{}")
        points = track["points"]
        for i in range(start, end):
            mask = cv2.imread(str(target / "masks" / files[i].name), 0)
            box = envelope
            x, y, a, b = [round(v) for v in box]
            pad = max(8, round((b - y) * 0.15))
            limit = np.zeros((720, 1280), np.uint8)
            limit[
                max(0, y - pad) : min(720, b + pad),
                max(0, x - pad) : min(1280, a + pad),
            ] = 255
            mask &= limit
            exact = panel_support(cv2.imread(str(files[i])), box)
            mask |= exact
            if np.count_nonzero(mask) / mask.size > 0.2:
                raise PipelineError(
                    "MASK_UNCERTAIN", "不透明面板单遍超过20%保护", review=True
                )
            cv2.imwrite(
                str(all_masks / files[i].name),
                cv2.imread(str(all_masks / files[i].name), 0) | mask,
            )
    # 自动目标邻域裁剪，不改变像素尺度和36/8时序窗口。
    union = np.zeros((720, 1280), np.uint8)
    for file in files:
        union |= cv2.imread(str(all_masks / file.name), 0)
    yy, xx = np.where(union > 0)
    pad = max(64, round(max(xx.max() - xx.min(), yy.max() - yy.min()) * 0.5))
    rx = max(0, (int(xx.min()) - pad) // 8 * 8)
    ry = max(0, (int(yy.min()) - pad) // 8 * 8)
    ra = min(1280, (int(xx.max()) + pad + 8) // 8 * 8)
    rb = min(720, (int(yy.max()) + pad + 8) // 8 * 8)
    crop = root / "crop"
    (crop / "frames").mkdir(parents=True, exist_ok=True)
    (crop / "masks").mkdir(exist_ok=True)
    coarse = Path(info["coarse"])
    if not (root / "gpu-complete.json").exists():
        for file in files:
            cv2.imwrite(
                str(crop / "frames" / file.name),
                cv2.imread(str(coarse / "composite" / file.name))[ry:rb, rx:ra],
            )
            cv2.imwrite(
                str(crop / "masks" / file.name),
                cv2.imread(str(all_masks / file.name), 0)[ry:rb, rx:ra],
            )
        for folder in [crop / "composite", *crop.glob("window-*")]:
            if folder.exists():
                shutil.rmtree(folder)
        vision.inpaint(
            crop,
            sorted((crop / "frames").glob("*.png")),
            d["scenes"],
            "propainter",
            True,
            condition_overlap=True,
        )
        (root / "gpu-complete.json").write_text("{}")
    old = root / "before.mp4"
    if not old.exists():
        shutil.copyfile(final, old)
    cap = cv2.VideoCapture(str(old))
    width, height = round(cap.get(3)), round(cap.get(4))
    silent = root / "video.mp4"
    log = (root / "encode.log").open("wb")
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            "60",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            "-threads",
            "4",
            str(silent),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=log,
    )

    @lru_cache(maxsize=8)
    def repaired(i):
        base = cv2.imread(str(coarse / "composite" / files[i].name))
        base[ry:rb, rx:ra] = cv2.imread(str(crop / "composite" / files[i].name))
        return cv2.resize(
            base, (width, height), interpolation=cv2.INTER_CUBIC
        ), cv2.resize(
            cv2.imread(str(all_masks / files[i].name), 0),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    try:
        for n, i in enumerate(mapping):
            ok, image = cap.read()
            if not ok:
                raise PipelineError("OUTPUT_INVALID", "不透明面板原生帧缺失")
            base, mask = repaired(i)
            weight = np.clip(
                cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5) / 3,
                0,
                1,
            )[:, :, None]
            image = np.rint(image * (1 - weight) + base * weight).astype(np.uint8)
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise PipelineError("OUTPUT_INVALID", "不透明面板无损合成失败")
    finally:
        cap.release()
        log.close()
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    output = root / "patched.mp4"
    mux_core_audio(silent, Path(info["source"]), output, len(mapping) / 60)
    if audio_hash(old) != audio_hash(output):
        raise PipelineError("OUTPUT_INVALID", "面板补修改变音频")
    output.replace(final)
    for name in ["audit", "verified.json"]:
        if (work / name).exists():
            (work / name).rename(work / f"{name}-before-opaque-{time.time_ns()}")
    report_path.write_text(
        json.dumps(
            dict(
                方法=method,
                证据=evidence,
                补修轨迹数=len(active),
                自动裁剪=[rx, ry, ra, rb],
                耗时秒=time.monotonic() - started,
                输出摘要=vision.sha256(final),
                音频原样复制=True,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("不透明面板独立补修完成，轨迹数", len(active), flush=True)


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    args = p.parse_args()
    patch(args.work, args.assets)


if __name__ == "__main__":
    main()


def verify(result, work, frame_offset=0, check=lambda: None):
    """对实际编码成品重新测量自动不透明面板，保留全部样本及候选。"""
    root = work / "opaque-panels"
    observed = root / "observations.json"
    if not observed.exists():
        return dict(检查样本数=0, 候选数=0, 全部样本=[], 证据=[])
    rows = json.loads(observed.read_text())
    detected = json.loads((work / "prepared.json").read_text())
    scenes = json.loads(Path(detected["detected"]).read_text())["scenes"]
    by_frame = {}
    for track in groups(rows, scenes):
        box = track_envelope(track["points"])
        start = max(track["scene"][0], track["points"][0]["frame"] - 15)
        end = min(track["scene"][1], track["points"][-1]["frame"] + 16)
        for i in range(start, end):
            by_frame.setdefault(i, []).append(
                dict(box=box, confidence=track["seed"]["confidence"])
            )
    mapping = json.loads((work / "native/native-report.json").read_text())["帧映射"]
    cap = cv2.VideoCapture(str(result))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
    samples = []
    candidates = []
    try:
        for n, index in enumerate(mapping):
            check()
            ok, image = cap.read()
            if not ok:
                raise PipelineError("OUTPUT_INVALID", "不透明面板编码后核验帧不足")
            if index not in by_frame or n % 30 not in [0, 1]:
                continue
            source = cv2.imread(str(work / "frames" / f"{index:06d}.png"))
            small = cv2.resize(image, (1280, 720))
            for point in by_frame[index]:
                mask = panel_support(source, point["box"])
                score = pair_score(source, small, mask)
                item = dict(
                    frame=n,
                    time=n / 60,
                    model_frame=index,
                    box=point["box"],
                    confidence=point["confidence"],
                    **score,
                )
                samples.append(item)
                if is_residual_score(score):
                    candidates.append(item)
    finally:
        cap.release()
    return dict(
        检查样本数=len(samples),
        候选数=len(candidates),
        全部样本=samples,
        证据=candidates,
        阈值=dict(绝对相关=0.35, 相对幅度=0.15),
    )
