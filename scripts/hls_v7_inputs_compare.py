"""只读比较本次重建的九项输入与七版冻结素材，绝不返回处理素材。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from worker.roi_inpaint import digest
from worker.v7_equivalent import REQUIRED_ASSETS


def rebuilt_paths(root):
    paths = {
        name: root / file
        for name, file in dict(
            prior="prior-work/multitrack-prior.npz",
            alpha="alpha.npz",
            template="template.npz",
            palette="palette.npy",
            projection="projection.json",
            color_seed_image="color-seed-image.png",
            color_seed_mask="color-seed-mask.png",
            variant_seed_image="variant-bootstrap/seed-image.png",
            variant_seed_mask="variant-bootstrap/seed-mask.png",
        ).items()
    }
    return paths


def arrays(path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as values:
            return {key: values[key].copy() for key in values.files}
    if path.suffix == ".npy":
        return {"array": np.load(path, allow_pickle=False)}
    if path.suffix == ".png":
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("对照图像无法解码")
        return {"image": image}
    return {}


def delta(a, b):
    row = dict(重建形状=list(a.shape), 冻结形状=list(b.shape))
    if a.shape != b.shape:
        return dict(row, 状态="形状不同")
    if a.dtype.kind not in "biuf" or b.dtype.kind not in "biuf":
        return dict(row, 完全相同=bool(np.array_equal(a, b)), 状态="非数值元数据")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(row, 状态="存在非有限值")
    diff = a.astype(np.float64) - b
    row.update(
        状态="已比较",
        完全相同=bool(np.array_equal(a, b)),
        差异元素数=int(np.count_nonzero(diff)),
        平均绝对误差=float(np.abs(diff).mean()),
        最大绝对误差=float(np.abs(diff).max()),
        均方根误差=float(np.sqrt((diff**2).mean())),
    )
    if np.any(diff):
        position = np.unravel_index(np.abs(diff).argmax(), diff.shape)
        row["最大差异位置"] = list(map(int, position))
        row["该位置重建值"] = float(a[position])
        row["该位置冻结值"] = float(b[position])
    if set(np.unique(a)) <= {0, 255} and set(np.unique(b)) <= {0, 255}:
        union = np.count_nonzero((a > 0) | (b > 0))
        row["交并比"] = (
            float(np.count_nonzero((a > 0) & (b > 0)) / union) if union else 1.0
        )
    return row


def numeric_json(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(numeric_json(item, prefix + "/" + str(key)))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            result.update(numeric_json(item, prefix + "/" + str(i)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = value
    return result


def compare(root, frozen, source_hash):
    """仅返回差异报告；缺项也明确列出，不用冻结文件填补。"""
    digest(frozen)
    baseline = json.loads(Path(frozen).read_text())
    if baseline.get("method") != "v7" or baseline.get("source_sha256") != source_hash:
        raise ValueError("冻结对照与原片摘要不一致")
    rows = []
    for name, current in rebuilt_paths(Path(root)).items():
        row = dict(
            项目=name,
            重建路径=str(current),
            冻结摘要=baseline.get("sha256", {}).get(name),
        )
        reference = Path(baseline[name]) if name in baseline else None
        if reference is None or not reference.is_file():
            row["状态"] = "冻结项缺失"
        elif digest(reference) != row["冻结摘要"]:
            row["状态"] = "冻结项摘要不一致"
        elif not current.is_file():
            row["状态"] = "本次尚未重建"
        else:
            row.update(状态="已比较", 重建摘要=digest(current))
            row["字节相同"] = row["重建摘要"] == row["冻结摘要"]
            ca, ra = arrays(current), arrays(reference)
            row["数组差异"] = {
                key: delta(ca[key], ra[key])
                if key in ca and key in ra
                else dict(状态="缺少对应数组")
                for key in sorted(set(ca) | set(ra))
            }
            if "metadata" in ca and "metadata" in ra:
                a = numeric_json(json.loads(str(ca["metadata"])))
                b = numeric_json(json.loads(str(ra["metadata"])))
                row["元数据数值差异"] = [
                    dict(位置=key, 重建=a.get(key), 冻结=b.get(key))
                    for key in sorted(set(a) | set(b))
                    if a.get(key) != b.get(key)
                ]
            if current.suffix == reference.suffix == ".json":
                a, b = (
                    numeric_json(json.loads(current.read_text())),
                    numeric_json(json.loads(reference.read_text())),
                )
                row["系数差异"] = [
                    dict(
                        位置=key,
                        重建=a.get(key),
                        冻结=b.get(key),
                        绝对差异=abs(a[key] - b[key])
                        if key in a and key in b
                        else None,
                    )
                    for key in sorted(set(a) | set(b))
                    if a.get(key) != b.get(key)
                ]
            if (
                name == "palette"
                and ca["array"].ndim == ra["array"].ndim == 2
                and ca["array"].size
                and ra["array"].size
            ):
                distances = np.linalg.norm(
                    ca["array"][:, None].astype(float) - ra["array"][None], axis=2
                )
                row["颜色中心最近距离"] = distances.min(axis=1).tolist()
                row["冻结颜色中心最近距离"] = distances.min(axis=0).tolist()
        rows.append(row)
    return dict(
        仅供评价=True,
        冻结结果参与处理=False,
        项目=rows,
        完整重建=all(r["状态"] == "已比较" for r in rows),
        项目数=len(REQUIRED_ASSETS),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--rebuilt",
        type=Path,
        required=True,
        help="本次运行的素材根目录，可含失败前的部分结果",
    )
    p.add_argument(
        "--comparison-assets", type=Path, required=True, help="冻结九项清单，只读评价"
    )
    p.add_argument("--source", type=Path, required=True, help="原片，仅计算摘要")
    p.add_argument("--output", type=Path, required=True, help="独立差异报告")
    a = p.parse_args()
    report = compare(a.rebuilt, a.comparison_assets, digest(a.source))
    protected = [a.source, a.comparison_assets, *rebuilt_paths(a.rebuilt).values()]
    frozen = json.loads(a.comparison_assets.read_text())
    protected.extend(Path(frozen[name]) for name in REQUIRED_ASSETS if name in frozen)
    if any(a.output.resolve() == path.resolve() for path in protected):
        p.error("差异报告不得覆盖任一输入")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    )
    print("九项只读对照已保存")


if __name__ == "__main__":
    main()
