"""Wrap an upstream script and capture PyTorch GPU high-water marks on exit."""

import argparse
import importlib.util
import json
import runpy
import sys
import time
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--frames-only", action="store_true")
    args, rest = parser.parse_known_args()
    if args.frames_only and "--save_frames" not in rest:
        parser.error("仅帧输出必须同时保存帧")
    torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0, str(Path(args.script).parent))
    hook = Path(__file__).with_name("propainter_offline.py")
    spec = importlib.util.spec_from_file_location("propainter_offline", hook)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_offline_hook()
    sys.argv = [args.script] + rest
    import imageio
    original_mimwrite = imageio.mimwrite
    previews = dict(seconds=0., skipped=0)

    def preview(*values, **options):
        if args.frames_only:
            previews['skipped'] += 1
            return
        started = time.perf_counter()
        try:
            return original_mimwrite(*values, **options)
        finally:
            previews['seconds'] += time.perf_counter()-started

    imageio.mimwrite = preview
    started = time.perf_counter()
    try:
        runpy.run_path(args.script, run_name="__main__")
    finally:
        imageio.mimwrite = original_mimwrite
        torch.cuda.synchronize()
        Path(args.metrics).write_text(
            json.dumps(
                {
                    "script_seconds": time.perf_counter()-started,
                    "preview_seconds": previews["seconds"],
                    "skipped_previews": previews["skipped"],
                    "allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                    "reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                }
            )
        )


if __name__ == "__main__":
    try:
        from .gpu_lock import device_lock
    except ImportError:
        from gpu_lock import device_lock
    with device_lock():
        main()
