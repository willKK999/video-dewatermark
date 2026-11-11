"""帧率相同也可能有起始重复帧，必须由内容恢复真实的帧对应。"""

import json
import cv2
import numpy as np
from scripts.hls_align_reference import align


def test_alignment_recovers_duplicated_start_and_shifted_reference(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 60, (100, 60)
    )
    for i in range(18):
        writer.write(np.full((60, 100, 3), i * 12, np.uint8))
    writer.release()
    cap = cv2.VideoCapture(str(source))
    images = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        images.append(image)
    cap.release()
    frames = tmp_path / "frames"
    frames.mkdir()
    expected = [0, 0, 1, 3, 5, 7]
    for i, j in enumerate(expected):
        cv2.imwrite(str(frames / f"{i:06d}.png"), images[j])
    output = tmp_path / "aligned"
    align(source, frames, output)
    rows = json.loads((output / "alignment.json").read_text())["帧对应"]
    assert [p["source_frame"] for p in rows] == expected
    for i, j in enumerate(expected):
        assert np.array_equal(cv2.imread(str(output / f"{i:06d}.png")), images[j])
