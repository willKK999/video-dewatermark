"""视频去水印系统：对已证实的完整图形低频残留进行小范围时序补修。"""

import argparse, json, os, shutil, subprocess, time
from pathlib import Path
from functools import lru_cache
import cv2, numpy as np
from . import vision
from .fine_masks import Supports
from .multitrack import load_prior
from .low_frequency import band, fixed_scores, complete_play_graphics
from .process import PipelineError
from scripts.hls_chunk_check import mux_core_audio, audio_hash


def residual_support(prior, signal, region):
    """完整圆形部件与真实文字笔画并集；普通文字不应因没有圆形而无法补修。"""
    from .matte import ink_support

    return complete_play_graphics(prior, signal, region) | ink_support(
        prior, signal, region["mask"], 3
    )


def verify_low(result, template, regions):
    cap = cv2.VideoCapture(str(result))
    fps = cap.get(5)
    count = round(cap.get(7))
    samples = []
    try:
        for n in range(count):
            ok, image = cap.read()
            if not ok:
                raise PipelineError("OUTPUT_INVALID", "低频复检帧不足")
            if n % round(fps) == 0:
                samples.append(
                    band(cv2.resize(image, (template.shape[1], template.shape[0])))
                )
    finally:
        cap.release()
    if not samples:
        raise PipelineError("OUTPUT_INVALID", "低频复检无样本")
    records = fixed_scores(template, np.median(np.stack(samples), axis=0), regions)
    return dict(
        样本数=len(samples),
        残留区域数=sum(p["residual"] for p in records),
        区域=records,
        阈值=dict(绝对相关=0.35, 相对幅度=0.15),
    )


def transformed(mask, region, part, shape):
    if part["kind"] == "fixed":
        return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    x, y, a, b = region["box"]
    piece = mask[y:b, x:a]
    if part["track"]["flip"]:
        piece = cv2.flip(piece, 1)
    x, y, a, b = [round(q) for q in part["point"]["box"]]
    output = np.zeros(shape, np.uint8)
    output[y:b, x:a] = cv2.resize(
        piece, (a - x, b - y), interpolation=cv2.INTER_NEAREST
    )
    return output


def patch(work, assets_path, low_template_path, complete_ink=False):
    cv2.setNumThreads(2)
    started = time.monotonic()
    assets = json.loads(assets_path.read_text())
    info = json.loads((work / "prepared.json").read_text())
    coarse = Path(info["coarse"])
    detected = json.loads(Path(info["detected"]).read_text())
    native = work / "native"
    final = native / "chunk1.mp4"
    meta = json.loads((native / "native-report.json").read_text())
    prior, signal, regions, _, _ = load_prior(Path(assets["prior"]))
    low = np.load(low_template_path)["template"]
    root = work / ("residual-ink" if complete_ink else "complete-graphics")
    method = "当前成品低频笔画补修-2"
    if root.exists() and any(root.iterdir()):
        marker = root / "method.json"
        if not marker.exists() or json.loads(marker.read_text()).get("方法") != method:
            root.rename(work / (root.name + "-previous-" + str(time.time_ns())))
    root.mkdir(exist_ok=True)
    (root / "method.json").write_text(json.dumps(dict(方法=method), ensure_ascii=False))
    report_path = root / "report.json"
    if report_path.exists():
        saved = json.loads(report_path.read_text())
        if saved.get("修复后视频摘要") == vision.sha256(final):
            checked = saved.get("修复后低频复检", saved.get("低频复检", {}))
            if checked.get("残留区域数") == 0:
                print("完整图形低频修复已完成，复用本块检查点", flush=True)
                return
            raise PipelineError(
                "RESIDUAL_UNCERTAIN",
                "上次低频补修仍未通过，须先诊断再重试",
                review=True,
            )
    before = verify_low(final, low, regions)
    (root / "low-before.json").write_text(
        json.dumps(before, ensure_ascii=False, indent=2)
    )
    identities = {r["region"] for r in before["区域"] if r["residual"]}
    if complete_ink:
        from .hls_residual_projection import scan

        high, temporary = scan(final, prior, regions)
        del temporary
        identities |= {p["region"] for p in high if p["residual"]}
    if not identities:
        report_path.write_text(
            json.dumps(
                dict(
                    无需复修=True, 低频复检=before, 修复后视频摘要=vision.sha256(final)
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    graphics = {
        r["id"]: residual_support(prior, signal, r)
        for r in regions
        if r["id"] in identities
    }
    if any(not graphics[i].any() for i in identities):
        raise PipelineError(
            "RESIDUAL_UNCERTAIN",
            "低频残留区域尚无可靠完整图形支持，不能编造掩膜",
            review=True,
        )
    support = Supports(
        Path(assets["prior"]),
        detected,
        np.load(assets["alpha"])["alpha"],
        Path(info["graphics"]),
    )
    frames = sorted((coarse / "frames").glob("*.png"))
    (root / "frames").mkdir(exist_ok=True)
    if not (root / "frames-complete.json").exists():
        cap = cv2.VideoCapture(str(final))
        mapping = meta["帧映射"]
        seen = set()
        for n, index in enumerate(mapping):
            ok, image = cap.read()
            if not ok:
                raise PipelineError("OUTPUT_INVALID", "残留笔画复修缺少原生帧")
            if index not in seen:
                cv2.imwrite(
                    str(root / "frames" / f"{index:06d}.png"),
                    cv2.resize(image, (1280, 720), interpolation=cv2.INTER_CUBIC),
                )
                seen.add(index)
        cap.release()
        missing = []
        for file in frames:
            if not (root / "frames" / file.name).exists():
                shutil.copyfile(
                    coarse / "composite" / file.name, root / "frames" / file.name
                )
                missing.append(file.name)
        (root / "frames-complete.json").write_text(
            json.dumps(dict(补齐无原生对应帧=missing))
        )
    (root / "masks").mkdir(exist_ok=True)
    areas = []
    for i, file in enumerate(frames):
        image = cv2.imread(str(file))
        mask = np.zeros(image.shape[:2], np.uint8)
        for part in support.at(i, image):
            if part.get("region") in identities:
                region = next(r for r in regions if r["id"] == part["region"])
                mask |= transformed(
                    graphics[part["region"]], region, part, image.shape[:2]
                )
        original_mask = cv2.imread(str(coarse / "masks" / file.name), 0)
        if np.count_nonzero(mask) / mask.size > 0.2:
            raise PipelineError(
                "MASK_UNCERTAIN", "完整图形补修单遍掩膜超过20%保护", review=True
            )
        if (
            np.count_nonzero(original_mask) / mask.size > 0.2
            and not (coarse / "layers.json").exists()
        ):
            raise PipelineError(
                "MASK_UNCERTAIN", "原基底联合面积过大且无分层来源证明", review=True
            )
        cv2.imwrite(str(root / "masks" / file.name), mask)
        areas.append(float(np.count_nonzero(mask) / mask.size))
    # 额外补修只裁自动图形及两倍直径的空间上下文，不缩放像素；仍保留整块全部时序。
    union = np.zeros((720, 1280), np.uint8)
    for file in frames:
        union |= cv2.imread(str(root / "masks" / file.name), 0)
    yy, xx = np.where(union > 0)
    if not len(xx):
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "残留存在但没有有效图形掩膜", review=True
        )
    pad = max(64, 2 * max(int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)))
    rx = max(0, (int(xx.min()) - pad) // 8 * 8)
    ry = max(0, (int(yy.min()) - pad) // 8 * 8)
    ra = min(1280, (int(xx.max()) + pad + 8) // 8 * 8)
    rb = min(720, (int(yy.max()) + pad + 8) // 8 * 8)
    crop = root / "crop"
    (crop / "frames").mkdir(parents=True, exist_ok=True)
    (crop / "masks").mkdir(exist_ok=True)
    if not (root / "gpu-complete.json").exists():
        for file in frames:
            image = cv2.imread(str(root / "frames" / file.name))
            mask = cv2.imread(str(root / "masks" / file.name), 0)
            cv2.imwrite(str(crop / "frames" / file.name), image[ry:rb, rx:ra])
            cv2.imwrite(str(crop / "masks" / file.name), mask[ry:rb, rx:ra])
        for folder in [crop / "composite", *crop.glob("window-*")]:
            if folder.exists():
                shutil.rmtree(folder)
        gpu_started = time.monotonic()
        vision.inpaint(
            crop,
            sorted((crop / "frames").glob("*.png")),
            detected["scenes"],
            "propainter",
            True,
            condition_overlap=complete_ink,
        )
        (root / "gpu-complete.json").write_text(
            json.dumps(
                dict(
                    耗时秒=time.monotonic() - gpu_started,
                    自动裁剪范围=[rx, ry, ra, rb],
                    像素缩放=1,
                ),
                ensure_ascii=False,
            )
        )
    old = (
        (root / "before.mp4")
        if complete_ink
        else (native / "chunk1-v7-before-graphic-completion.mp4")
    )
    if old.exists() and vision.sha256(old) != vision.sha256(final):
        old.rename(native / f"before-graphic-input-{time.time_ns()}.mp4")
    if not old.exists():
        shutil.copyfile(final, old)
    cap = cv2.VideoCapture(str(old))
    fps = cap.get(5)
    w, h = int(cap.get(3)), int(cap.get(4))
    mapping = meta["帧映射"]
    silent = root / "lossless-video.mp4"
    error = (root / "encode.log").open("wb")
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
            f"{w}x{h}",
            "-r",
            str(fps),
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
        stderr=error,
    )

    @lru_cache(maxsize=12)
    def correction(index):
        mask = cv2.imread(str(root / "masks" / f"{index:06d}.png"), 0)
        yy, xx = np.where(mask > 0)
        if not len(xx):
            return None
        mh, mw = mask.shape
        x = max(0, (int(xx.min()) - 24) // 2 * 2)
        y = max(0, (int(yy.min()) - 24) // 2 * 2)
        a = min(mw, (int(xx.max()) + 27) // 2 * 2)
        b = min(mh, (int(yy.max()) + 27) // 2 * 2)
        nx, ny, na, nb = (
            round(x * w / mw),
            round(y * h / mh),
            round(a * w / mw),
            round(b * h / mh),
        )
        shape = (na - nx, nb - ny)
        before = cv2.resize(
            cv2.imread(str(root / "frames" / f"{index:06d}.png"))[y:b, x:a],
            shape,
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        after = cv2.resize(
            cv2.imread(str(crop / "composite" / f"{index:06d}.png"))[
                y - ry : b - ry, x - rx : a - rx
            ],
            shape,
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        delta = cv2.GaussianBlur(after, (0, 0), 3) - cv2.GaussianBlur(before, (0, 0), 3)
        mm = cv2.resize(mask[y:b, x:a], shape, interpolation=cv2.INTER_NEAREST)
        weight = np.clip(
            cv2.distanceTransform((mm > 0).astype(np.uint8), cv2.DIST_L2, 5) / 2, 0, 1
        )[:, :, None]
        return (nx, ny, na, nb, after if complete_ink else delta * weight, mm)

    try:
        for n, index in enumerate(mapping):
            ok, image = cap.read()
            if not ok:
                raise PipelineError("OUTPUT_INVALID", "原生低频修复基底帧不足")
            data = correction(index)
            if data:
                x, y, a, b, delta, mask = data
                original = image[y:b, x:a].copy()
                if complete_ink:
                    weight = np.clip(
                        cv2.distanceTransform(
                            (mask > 0).astype(np.uint8), cv2.DIST_L2, 5
                        )
                        / 3,
                        0,
                        1,
                    )[:, :, None]
                    changed = np.clip(
                        np.rint(original * (1 - weight) + delta * weight), 0, 255
                    ).astype(np.uint8)
                else:
                    changed = np.clip(
                        np.rint(original.astype(np.float32) + delta), 0, 255
                    ).astype(np.uint8)
                changed[mask == 0] = original[mask == 0]
                image[y:b, x:a] = changed
            encoder.stdin.write(image.tobytes())
            if n % 300 == 0:
                print("完整图形低频合成帧数", n, flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise PipelineError("OUTPUT_INVALID", "低频修复无损中间编码失败")
    finally:
        cap.release()
        error.close()
        if encoder.poll() is None:
            encoder.stdin.close()
            encoder.terminate()
            try:
                encoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
    patched = root / "chunk.mp4"
    mux_core_audio(silent, Path(info["source"]), patched, len(mapping) / fps)
    if audio_hash(old) != audio_hash(patched):
        raise PipelineError("OUTPUT_INVALID", "低频修复改变压缩音频")
    after = verify_low(patched, low, regions)
    report = dict(
        方法=method,
        修复区域=sorted(identities),
        真实笔画保护=complete_ink,
        最大补全面积比例=max(areas),
        修复前低频复检=before,
        修复后低频复检=after,
        耗时秒=time.monotonic() - started,
        基底视频摘要=vision.sha256(old),
        修复后视频摘要=vision.sha256(patched),
        音频内容一致=True,
        掩膜外像素策略="复用v7成品；中间RGB无损编码，最终HLS统一编码",
        高频策略=("真实笔画内使用补画结果" if complete_ink else "保留v7成品高频，只加减当前基底的标准差3低频差分"),
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    patched.replace(final)
    # 旧版本检查不能冒充新视频的检查，保留后重新测量。
    if (work / "audit").exists():
        (work / "audit").rename(
            work / f"audit-before-graphic-completion-{time.time_ns()}"
        )
    if (work / "verified.json").exists():
        (work / "verified.json").rename(
            work / "verified-before-graphic-completion.json"
        )
    high_after = None
    if complete_ink:
        high_after, temporary = scan(final, prior, regions)
        del temporary
        report["修复后高通复检"] = high_after
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if after["残留区域数"] or (
        high_after
        and any(p["residual"] for p in high_after if p["region"] in identities)
    ):
        raise PipelineError(
            "RESIDUAL_UNCERTAIN", "完整图形低频补修后仍有残留证据", review=True
        )
    print("完整图形低频补修通过，耗时秒", report["耗时秒"], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["work", "assets", "low-template"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    patch(a.work, a.assets, a.low_template)


if __name__ == "__main__":
    main()
