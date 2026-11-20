"""原生合成的帧率选择必须与源参考的标准重采样相位一致。"""

import subprocess
import cv2, numpy as np
from scripts.hls_native_chunk import source_frame_for_output


def test_sampling_phase_matches_ffmpeg_fps_filter(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (100, 60))
    for i in range(12):
        writer.write(np.full((60, 100, 3), i * 15, np.uint8))
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "fps=30",
            str(tmp_path / "frame-%02d.png"),
        ],
        check=True,
    )
    frames = sorted(tmp_path.glob("frame-*.png"))
    assert len(frames) == 6
    for n, frame in enumerate(frames):
        expected = source_frame_for_output(n, 60, 30) * 15
        assert abs(cv2.imread(str(frame)).mean() - expected) < 2
    assert source_frame_for_output(749, 60, 30) == 1498
