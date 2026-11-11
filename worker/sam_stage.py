"""Run via TRACK_PYTHON with pinned SAM 2.1 installed. Exits after each job."""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor
from workflow import expand_box
from sam_schedule import groups_for_scene, sampled_hints, load_mask_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    detections = json.loads((root / "detections.json").read_text())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictor = build_sam2_video_predictor(
        os.environ["SAM_CONFIG"], os.environ["SAM_CHECKPOINT"], device="cuda"
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    dtype = torch.bfloat16 if os.environ.get("SAM_BF16") == "true" else torch.float32
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32),
    ):
        for scene_start, scene_end in detections["scenes"]:
            image = cv2.imread(str(root / "frames" / f"{scene_start:06d}.png"))
            height, width = image.shape[:2]
            for i in range(scene_start, scene_end):
                cv2.imwrite(
                    str(root / "masks" / f"{i:06d}.png"),
                    np.zeros((height, width), np.uint8),
                )
            for start, end, hints in groups_for_scene(
                detections, scene_start, scene_end
            ):
                with tempfile.TemporaryDirectory(dir=root) as tmp:
                    for i in range(start, end):
                        cv2.imwrite(
                            str(Path(tmp) / f"{i-start:06d}.jpg"),
                            cv2.imread(str(root / "frames" / f"{i:06d}.png")),
                        )
                    state = predictor.init_state(
                        video_path=tmp,
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                    )
                    if detections.get("multitrack"):
                        hints = sampled_hints(hints, start, end)
                    for point in hints:
                        if point.get("mask_prompt"):
                            predictor.add_new_mask(
                                state,
                                frame_idx=point["frame"] - start,
                                obj_id=point.get("track_id", 1),
                                mask=load_mask_prompt(
                                    root, point["mask_prompt"], (height, width)
                                ),
                            )
                        else:
                            predictor.add_new_points_or_box(
                                state,
                                frame_idx=point["frame"] - start,
                                obj_id=point.get("track_id", 1),
                                box=np.array(
                                    expand_box(point["box"], width, height),
                                    dtype=np.float32,
                                ),
                            )
                    first = min(p["frame"] - start for p in hints)
                    for reverse in [False, True]:
                        for idx, object_ids, logits in predictor.propagate_in_video(
                            state, start_frame_idx=first, reverse=reverse
                        ):
                            target = root / "masks" / f"{idx+start:06d}.png"
                            mask = cv2.imread(str(target), 0)
                            for k, object_id in enumerate(object_ids):
                                component = (logits[k, 0] > 0).cpu().numpy().astype(
                                    np.uint8
                                ) * 255
                                mask = cv2.bitwise_or(mask, component)
                                if detections.get("multitrack"):
                                    dest = root / "object-masks" / str(object_id)
                                    dest.mkdir(parents=True, exist_ok=True)
                                    cv2.imwrite(
                                        str(dest / f"{idx+start:06d}.png"), component
                                    )
                            cv2.imwrite(str(target), mask)
                    predictor.reset_state(state)
    (root / "metrics.sam.json").write_text(
        json.dumps(
            {
                "model_load_seconds": load_seconds,
                "segmentation_seconds": time.perf_counter() - started,
                "allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
            }
        )
    )


if __name__ == "__main__":
    main()
