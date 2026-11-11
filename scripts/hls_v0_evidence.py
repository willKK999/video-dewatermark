"""补齐v0原始证据，不重算指标、不修改验收结论。"""

import argparse, json, shutil
from pathlib import Path


def enrich(work_root, delivery):
    report_path = delivery / "report.json"
    report = json.loads(report_path.read_text())
    original = delivery / "report-v0-原始.json"
    if not original.exists():
        shutil.copyfile(report_path, original)
    evidence = delivery / "逐块证据"
    evidence.mkdir(exist_ok=True)
    names = [
        "round.json",
        "verified.json",
        "v0-block.json",
        "native/native-report.json",
        "audit/quality.json",
        "audit/brand.json",
        "audit/low-band.json",
        "audit/opaque.json",
        "audit/clarity.json",
    ]
    for row in report["各块"]:
        work = work_root / f'chunk-{row["块号"]:02d}'
        target = evidence / f'块{row["块号"]:02d}'
        target.mkdir(exist_ok=True)
        for name in names:
            source = work / name
            if source.exists():
                shutil.copyfile(source, target / name.replace("/", "-"))
        clarity = work / "audit/clarity.json"
        native = work / "native/native-report.json"
        row["清晰度原始记录"] = (
            json.loads(clarity.read_text()) if clarity.exists() else None
        )
        row["原生基底清晰度原始记录"] = (
            json.loads(native.read_text()).get("清晰度统计")
            if native.exists()
            else None
        )
    report["残留计数说明"] = (
        "计数为自动复检候选，可能有误报及跨帧重复；未逐项确认的实际独立残留数量未知。保留原始候选，不能据此自动清零。"
    )
    report["指标说明"] = (
        "本文件保留v0原始测量及结论；清晰度只记录，接近0的值尚待第二阶段排查。未修改阈值或重算样本。"
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (delivery / "report-v0.json").write_text(report_path.read_text())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    args = parser.parse_args()
    enrich(args.work_root, args.delivery)
    print("整集v0原始证据整理完成")


if __name__ == "__main__":
    main()
