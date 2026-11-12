"""逐帧核对七版与提速版掩膜；保留全部差异，不改修复范围或判定。"""

import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def compare(candidate, reference, output, names=("color", "variant", "graphics")):
    candidate, reference = Path(candidate).resolve(), Path(reference).resolve()
    if candidate == reference:
        raise ValueError("必须使用独立的七版参考，不能将同一输出与自身比较")
    differences, samples, input_differences = [], [], []
    cf, rf = candidate / "frames", reference / "frames"
    input_names = sorted({p.name for root in (cf, rf) for p in root.glob("*.png")})
    for name in input_names:
        fingerprints = []
        for root in (cf, rf):
            image = cv2.imread(str(root / name)) if (root / name).is_file() else None
            fingerprints.append(
                None if image is None else hashlib.sha256(image.tobytes()).hexdigest()
            )
        if fingerprints[0] is None or fingerprints[0] != fingerprints[1]:
            input_differences.append(dict(帧=name, 输入摘要=fingerprints))
    for stage in names:
        roots = [root / stage / "masks" for root in (candidate, reference)]
        frames = sorted({p.name for root in roots for p in root.glob("*.png")})
        if not frames:
            differences.append(dict(阶段=stage, 原因="双方均缺少输出，不算等价"))
        for name in frames:
            masks = [
                cv2.imread(str(root / name), 0) if (root / name).is_file() else None
                for root in roots
            ]
            row = dict(阶段=stage, 帧=name, 交并比=None)
            if any(mask is None for mask in masks):
                row["原因"] = "缺少对应帧"
            elif masks[0].shape != masks[1].shape or any(
                not set(np.unique(mask)) <= {0, 255} for mask in masks
            ):
                row["原因"] = "尺寸或二值掩膜取值不一致"
            else:
                a, b = [mask > 0 for mask in masks]
                intersection, union = np.count_nonzero(a & b), np.count_nonzero(a | b)
                row.update(
                    交集像素=int(intersection),
                    并集像素=int(union),
                    差异像素=int(np.count_nonzero(a != b)),
                    交并比=float(intersection / union) if union else 1.0,
                )
            samples.append(row)
            if row["交并比"] is None or row["交并比"] < 0.99:
                differences.append(row)
    result = dict(
        阈值=0.99,
        同一输入=bool(input_names) and not input_differences,
        输入差异=input_differences,
        全部样本=samples,
        非完全相同帧=[row for row in samples if row.get("差异像素", 1) != 0],
        掩膜差异=differences,
        最低交并比=min(
            (r["交并比"] for r in samples if r["交并比"] is not None), default=None
        ),
        通过=bool(input_names)
        and not input_differences
        and bool(samples)
        and not differences,
        说明="参考仅用于评价，不参与提速版的检测、分割、补画或缓存恢复",
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True, help="提速版工作目录")
    p.add_argument(
        "--reference", type=Path, required=True, help="独立七版参考工作目录，仅供比较"
    )
    p.add_argument("--output", type=Path, required=True, help="逐帧差异报告路径")
    a = p.parse_args()
    report = compare(a.candidate, a.reference, a.output)
    print("等价性检查完成，是否通过：" + str(report["通过"]))


if __name__ == "__main__":
    main()
