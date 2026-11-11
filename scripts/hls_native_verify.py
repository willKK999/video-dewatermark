"""对原规格单块编码后逐帧复检，并导出反馈时间点的真实对比图。"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from worker.multitrack import residual_quality
from worker.process import PipelineError
from scripts.hls_chunk_check import audio_hash


def remap_references(detections, mapping):
    by_frame = {}
    for index, model_frame in enumerate(mapping):
        by_frame.setdefault(model_frame, []).append(index)
    tracks = []
    for track in detections["tracks"]:
        hints = [
            dict(p, frame=i)
            for p in track["hints"]
            for i in by_frame.get(p["frame"], [])
        ]
        tracks.append(dict(track, hints=hints))
    references = dict(detections, tracks=tracks)
    return references


def verify(source, result, detected, native_report, output):
    output.mkdir(parents=True, exist_ok=True)
    encoded = output / "composite"
    encoded.mkdir(exist_ok=True)
    mapping = json.loads(native_report.read_text())["帧映射"]
    detections = json.loads(detected.read_text())
    references = remap_references(detections, mapping)
    (output / "detections.json").write_text(json.dumps(references))
    cap = cv2.VideoCapture(str(result))
    count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok, image = cap.read()
        if not ok:
            break
        cv2.imwrite(str(encoded / f"{count:06d}.png"), cv2.resize(image, (1280, 720)))
        count += 1
    cap.release()
    if count != len(mapping):
        raise ValueError("编码帧数与原生合成帧映射不一致")
    try:
        residual_quality(output, sorted(encoded.glob("*.png")), references)
    except PipelineError as error:
        if error.code != "RESIDUAL_UNCERTAIN":
            raise
    summary = json.loads((output / "quality.json").read_text())
    summary["音频压缩内容一致"] = audio_hash(source, count / fps) == audio_hash(result)
    summary["编码后帧数"] = count
    summary["编码后帧率"] = fps
    summary["检查说明"] = (
        "全部编码帧检查已知移动轨迹，固定轨迹检查时间共识；另须检查短时图形与全画面新候选，不能单凭本项声明零残留"
    )
    (output / "编码后检查.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"moving_evidence", "fixed_template_scores"}
            },
            ensure_ascii=False,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "result", "detected", "native-report", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    cv2.setNumThreads(2)
    verify(a.source, a.result, a.detected, a.native_report, a.output)


if __name__ == "__main__":
    main()
