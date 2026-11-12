"""通用检查点收尾：每块仅一轮，阶段失败继续，保留并选择完整版本。"""
import argparse
import json
import os
import time
from pathlib import Path
from worker.gpu_lock import device_lock
from worker.hls_delivery import best_version, verified_stages
from worker.hls_round import execute_round
from worker.hls_v7 import load_assets, verify_block
from worker.hls_graphic_repair import patch as graphics
from worker.opaque_panels import patch as panels
from worker.vision import sha256


def closeout(root, assets_path, blocks):
    assets = load_assets(assets_path, sha256(root / "source.mp4"))
    for index in blocks:
        work = root / f"chunk-{index:02d}"
        with device_lock(work / "native-stage.lock"):
            target = work / "closeout"
            marker = target / "selection.json"
            if marker.exists():
                print("收尾单轮已完成，复用记录", work.name, flush=True)
                continue
            target.mkdir(exist_ok=True)
            started = time.monotonic()
            info = json.loads((work / "prepared.json").read_text())
            candidates = []

            def snapshot(name):
                video = work / "native/chunk1.mp4"
                if not video.is_file():
                    return
                digest = sha256(video)
                qc = json.loads((work / "verified.json").read_text())
                if qc.get("视频摘要") != digest:
                    verify_block(work, assets, info)
                    qc = json.loads((work / "verified.json").read_text())
                path = target / (name + ".mp4")
                if not path.exists():
                    os.link(video, path)
                record = dict(
                    版本=name, 路径=str(path), 视频摘要=digest, 完整=True,
                    生成时间=video.stat().st_mtime, 残留=qc.get("残留"),
                    低频残留=qc.get("低频残留"), 闪烁=None,
                    清晰度最低值=qc.get("清晰度最低值"), 核验=qc,
                )
                (target / (name + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2))
                candidates.append(record)

            snapshot("本轮前")
            reused = verified_stages(candidates[-1]["核验"], candidates[-1]["视频摘要"])
            report = execute_round(target, [
                ("v7原生去水印", lambda: None),
                ("低频检查与补修", lambda: graphics(work, assets_path, Path(assets["low_band_template"]))),
                ("不透明面板补修", lambda: panels(work, assets_path)),
                ("编码后复检", lambda: verify_block(work, assets, info)),
            ], work / "native/chunk1.mp4", {
                "v7原生去水印": dict(通过=True, 证据文件=str(work / "native/native-report.json")),
                **reused,
            })
            try:
                snapshot("本轮后")
            except Exception as exc:
                report["未通过原因"].append("末尾核验：" + getattr(exc, "code", type(exc).__name__))
            selected = best_version(candidates)
            result = dict(块号=index, 已选=selected, 候选=candidates, 单轮=report,
                          耗时秒=time.monotonic()-started, 说明="仅比较本轮前后同口径完整版本；旧口径报告保留但不能直接冒充更优")
            marker.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print("收尾单轮完成", work.name, json.dumps({k: selected[k] for k in ["版本", "残留", "低频残留", "清晰度最低值"]}, ensure_ascii=False),
                  "耗时秒=" + str(result["耗时秒"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--blocks", type=int, nargs="+", required=True)
    args = parser.parse_args()
    for index in args.blocks:
        try:
            closeout(args.root, args.assets, [index])
        except Exception as exc:
            print("收尾单轮异常，继续下一块", index, getattr(exc, "code", type(exc).__name__), flush=True)
