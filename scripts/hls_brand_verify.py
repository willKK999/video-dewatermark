"""编码后逐帧核验完整图形及其局部部件，覆盖图标、文字和反色残留。"""

import argparse, json
from functools import lru_cache
from pathlib import Path
import cv2, numpy as np
from worker.multitrack import highpass, component_scores
from worker.brand_templates import variants
from worker.branding import palette_support


def clipped_scores(feature, template, mask, box):
    x, y, a, b = box
    left, top, right, bottom = (
        max(0, x),
        max(0, y),
        min(feature.shape[1], a),
        min(feature.shape[0], b),
    )
    if left >= right or top >= bottom:
        return []
    return component_scores(
        template[top - y : bottom - y, left - x : right - x],
        feature[top:bottom, left:right],
        mask[top - y : bottom - y, left - x : right - x],
    )


def verify(
    encoded, source_frames, mapping, short_brand, brand, palette, output, frame_offset=0
):
    native = json.loads(mapping.read_text())
    indices = native["帧映射"]
    fps = native["视频规格"][2]
    records = json.loads((brand / "tracks.json").read_text())["frames"]
    bank = variants(
        cv2.imread(str(brand / "seed-image.png")),
        cv2.imread(str(brand / "seed-mask.png"), 0),
        [0.95, 1, 1.05],
    )
    colors = np.load(palette)
    evidence = []
    checked = 0

    @lru_cache(maxsize=8)
    def source(i):
        return cv2.imread(str(source_frames / f"{i:06d}.png"))

    @lru_cache(maxsize=8)
    def graphic(i):
        path = short_brand / "masks" / f"{i:06d}.png"
        if not path.exists():
            return None
        sam = cv2.imread(str(path), 0)
        if not np.any(sam):
            return None
        image = source(i)
        mask = palette_support(image, colors) & sam
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        yy, xx = np.where(mask > 0)
        if not len(xx):
            return None
        x, y, a, b = int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)
        return highpass(image)[y:b, x:a], mask[y:b, x:a], [x, y, a, b]

    cap = cv2.VideoCapture(str(encoded)) if encoded.is_file() else None
    if cap is not None and frame_offset:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_offset)
    for n, i in enumerate(indices):
        if cap is not None:
            ok, image = cap.read()
            if not ok:
                raise ValueError("图形复检视频帧不足")
            image = cv2.resize(image, (1280, 720))
        else:
            image = cv2.imread(str(encoded / f"{n:06d}.png"))
        feature = highpass(image)
        targets = []
        first = graphic(i)
        if first:
            targets.append(("短时图形", *first))
        for point in records[i]["matches"]:
            v = bank[point["variant"]]
            t = v["template"]
            mask = (
                (v["mask"] > 0)
                & (np.abs(t) >= max(0.5, float(np.median(np.abs(t[v["mask"] > 0])))))
            ).astype(np.uint8) * 255
            targets.append(("品牌变体", t, mask, point["box"]))
        for kind, t, mask, box in targets:
            scores = clipped_scores(feature, t, mask, box)
            checked += 1
            bad = [
                p
                for p in scores
                if abs(p["correlation"]) >= 0.55
                and abs(p["relative_amplitude"]) >= 0.15
            ]
            if bad:
                evidence.append(
                    dict(
                        frame=n,
                        time=n / fps,
                        model_frame=i,
                        kind=kind,
                        box=box,
                        components=bad,
                    )
                )
        if n % 300 == 0:
            print("完整图形编码后复检帧数", n, flush=True)
    if cap is not None:
        cap.release()
    report = dict(
        帧数=len(indices),
        图形检查次数=checked,
        残留证据数=len(evidence),
        通过=not evidence,
        判定阈值=dict(绝对相关=0.55, 相对幅度=0.15),
        证据=evidence,
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("完整图形复检结果", report["通过"], "证据数", len(evidence), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "encoded",
        "source-frames",
        "mapping",
        "short-brand",
        "brand",
        "palette",
        "output",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(2)
    verify(
        a.encoded,
        a.source_frames,
        a.mapping,
        a.short_brand,
        a.brand,
        a.palette,
        a.output,
    )


if __name__ == "__main__":
    main()
