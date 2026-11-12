"""在独立冻结源码上计算七版图形参考，只消费本次运行的相同输入。"""

import argparse
import json
from pathlib import Path
import sys
import hashlib

REFERENCE_COMMIT = "0e4dc7b9051388e251e13f50a1ee087cae7fb43b"


def verify_reference(root):
    root = Path(root).resolve()
    metadata = json.loads((root / "reference-code.json").read_text())
    required = {
        "worker/hls_v7.py",
        "worker/colour_tracks.py",
        "worker/brand_templates.py",
        "worker/sam_stage.py",
        "worker/sam_schedule.py",
        "worker/workflow.py",
    }
    if metadata.get("commit") != REFERENCE_COMMIT or not required <= set(
        metadata.get("sha256", {})
    ):
        raise ValueError("七版参考源码缺少完整来源证明")
    for name, expected in metadata["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.suffix != ".py":
            raise ValueError("参考源码清单出现非代码路径")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("冻结七版参考源码摘要不一致")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference-code", type=Path, required=True, help="主分支冻结源码目录"
    )
    p.add_argument("--work", type=Path, required=True, help="本次独立的参考输出目录")
    p.add_argument("--frames", type=Path, required=True, help="本次冷启动解码的输入帧")
    p.add_argument("--detected", type=Path, required=True, help="本次冷启动检测结果")
    p.add_argument("--assets", type=Path, required=True, help="双方相同的固定算法输入")
    a = p.parse_args()
    metadata = verify_reference(a.reference_code)
    if a.work.exists() and any(a.work.iterdir()):
        p.error("参考必须使用全新空目录，不读取旧检查点")
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "reference-source.json").write_text(
        json.dumps(metadata, ensure_ascii=False)
    )
    sys.path.insert(0, str(a.reference_code.resolve()))
    from worker.hls_v7 import graphic_masks
    from worker.colour_tracks import complete
    import cv2

    cv2.setNumThreads(2)
    (a.work / "frames").symlink_to(a.frames.resolve(), target_is_directory=True)
    assets = json.loads(a.assets.read_text())
    detected = json.loads(a.detected.read_text())
    files = sorted((a.work / "frames").glob("*.png"))
    graphic_masks(files, assets, a.work, lambda: None)
    complete(a.work, assets, detected, lambda: None)
    print("冻结七版图形参考已独立计算完成")


if __name__ == "__main__":
    main()
