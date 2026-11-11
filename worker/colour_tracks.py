"""从已确认品牌的完整模板定位独立目标，按镜头分别初始化 SAM 并补足动画前后图形。"""

import json, os, time
from pathlib import Path
import cv2, numpy as np
from .brand_templates import variants, mask_for
from .multitrack import load_prior
from .process import PipelineError, run


def groups(records, scenes):
    result = []
    for start, end in scenes:
        tracks = []
        for row in records[start:end]:
            frame = row["frame"]
            used = set()
            for point in row.get("color_matches", []):
                box = point["box"]
                center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
                options = [
                    (float(np.linalg.norm(t["center"] - center)), i)
                    for i, t in enumerate(tracks)
                    if i not in used and frame - t["last"] <= 60
                ]
                distance, k = min(options, default=(float("inf"), -1))
                if distance > max(128, (box[2] - box[0]) * 0.5):
                    k = len(tracks)
                    tracks.append(dict(points=[], center=center, last=frame))
                tracks[k]["points"].append(dict(point, frame=frame))
                tracks[k].update(center=center, last=frame)
                used.add(k)
        for track in tracks:
            if len(track["points"]) >= 3:
                result.append(
                    dict(
                        scene=[start, end],
                        seed=max(track["points"], key=lambda p: p["confidence"]),
                        points=track["points"],
                    )
                )
    return result


def complete(work, assets, detected, check):
    marker = work / "colour-sam-complete.json"
    if marker.exists():
        return json.loads(marker.read_text())
    started = time.monotonic()
    records = json.loads((work / "variant/tracks.json").read_text())["frames"]
    candidates = groups(records, detected["scenes"])
    files = sorted((work / "frames").glob("*.png"))
    bank = variants(
        cv2.imread(assets["color_seed_image"]),
        cv2.imread(assets["color_seed_mask"], 0),
        [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.75, 1.0],
    )
    _, _, known, _, _ = load_prior(assets["prior"])
    for number, track in enumerate(candidates):
        root = work / "colour-sam" / f"{number:03d}"
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "frames").exists():
            (root / "frames").symlink_to(work / "frames", target_is_directory=True)
        (root / "masks").mkdir(exist_ok=True)
        (root / "prompts").mkdir(exist_ok=True)
        seed = track["seed"]
        image = cv2.imread(str(files[seed["frame"]]))
        prompt = mask_for(image.shape[:2], bank, [seed])
        cv2.imwrite(str(root / "prompts/seed.png"), prompt)
        d = dict(
            scenes=[track["scene"]],
            hints=[
                dict(
                    frame=seed["frame"],
                    box=seed["box"],
                    confidence=seed["confidence"],
                    track_id=1,
                    mask_prompt="prompts/seed.png",
                )
            ],
        )
        (root / "detections.json").write_text(json.dumps(d))
        if not (root / "complete.json").exists():
            run(
                [
                    os.environ["TRACK_PYTHON"],
                    str(Path(__file__).with_name("sam_stage.py")),
                    "--root",
                    root,
                ],
                check,
                timeout=1800,
            )
            if (
                cv2.countNonZero(
                    cv2.imread(str(root / "masks" / files[seed["frame"]].name), 0)
                    & prompt
                )
                < cv2.countNonZero(prompt) * 0.6
            ):
                raise PipelineError(
                    "TRACKING_DRIFT", "完整图形种子未被分割覆盖", review=True
                )
            (root / "complete.json").write_text("{}")
        boxes = [p["box"] for p in track["points"]]
        pad = max(16, round((seed["box"][3] - seed["box"][1]) * 0.1))
        x = max(0, min(b[0] for b in boxes) - pad)
        y = max(0, min(b[1] for b in boxes) - pad)
        a = min(1280, max(b[2] for b in boxes) + pad)
        b = min(720, max(b[3] for b in boxes) + pad)
        envelope = np.zeros((720, 1280), np.uint8)
        envelope[round(y) : round(b), round(x) : round(a)] = 255
        for i in range(*track["scene"]):
            check()
            mask = cv2.imread(str(root / "masks" / files[i].name), 0) & envelope
            yy, xx = np.where(mask > 0)
            if len(xx):
                area = (xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1)
                # 跳到另一个已知固定副本时，该副本由其独立轨迹处理，不混淆身份。
                for region in known:
                    q, r, s, t = region["box"]
                    inter = max(0, min(xx.max() + 1, s) - max(xx.min(), q)) * max(
                        0, min(yy.max() + 1, t) - max(yy.min(), r)
                    )
                    if inter >= area * 0.85:
                        mask[:] = 0
                        break
            path = work / "color/masks" / files[i].name
            cv2.imwrite(str(path), cv2.imread(str(path), 0) | mask)
    intervals = []
    for i, file in enumerate(files):
        color = cv2.imread(str(work / "color/masks" / file.name), 0)
        white = cv2.imread(str(work / "variant/masks" / file.name), 0)
        cv2.imwrite(str(work / "graphics/masks" / file.name), color | white)
        if color.any():
            if intervals and intervals[-1][1] == i:
                intervals[-1][1] += 1
            else:
                intervals.append([i, i + 1])
    for name in ["color", "graphics"]:
        (work / name / "detections.json").write_text(json.dumps(dict(scenes=intervals)))
    info = dict(
        独立图形轨迹数=len(candidates),
        耗时秒=time.monotonic() - started,
        空帧允许=True,
        依据="完整品牌模板定位、逐镜头独立SAM状态、自动种子及既有固定副本排除",
    )
    marker.write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info
