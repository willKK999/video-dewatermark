"""Owned synthetic moving watermark media used for tests and smoke runs."""

import cv2
import numpy as np
from worker.process import run


def generate(root, portrait=False, audio=True):
    frames = root / "input"
    frames.mkdir()
    width, height = (180, 320) if portrait else (320, 180)
    for i in range(12):
        y, x = np.mgrid[:height, :width]
        image = np.stack(
            [(x % 100) + 60, (y % 90) + 60, np.full_like(x, 80)], axis=-1
        ).astype(np.uint8)
        cv2.rectangle(image, (20 + i * 3, 30), (60 + i * 3, 50), (235, 235, 235), -1)
        cv2.putText(
            image,
            "WM",
            (21 + i * 3, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (10, 10, 10),
            1,
        )
        cv2.imwrite(str(frames / f"{i:06d}.png"), image)
    source = root / "source.mp4"
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-framerate",
        "12",
        "-i",
        frames / "%06d.png",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac"]
    run(args + ["-c:v", "libx264", "-pix_fmt", "yuv420p", source])
    return source
