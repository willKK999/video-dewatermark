"""从当前成品按既有时间映射提取模型输入，不能回退到旧补画结果覆盖新修复。"""

from pathlib import Path
import shutil
import cv2


def capture(video, mapping, output, reference_files):
    output = Path(output)
    output.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    seen = set()
    try:
        for index in mapping:
            ok, image = cap.read()
            if not ok:
                raise ValueError("当前成品的帧数少于模型时间映射")
            if index not in seen:
                if not cv2.imwrite(str(output / f"{index:06d}.png"), cv2.resize(image, (1280, 720), interpolation=cv2.INTER_CUBIC)):
                    raise ValueError("当前成品模型帧写入失败")
                seen.add(index)
        if cap.read()[0]:
            raise ValueError("当前成品的帧数多于模型时间映射")
    finally:
        cap.release()
    missing = []
    for file in reference_files:
        if not (output / file.name).exists():
            shutil.copyfile(file, output / file.name)
            missing.append(file.name)
    return missing
