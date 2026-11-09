"""Wrap an upstream script and capture PyTorch GPU high-water marks on exit."""

import argparse
import importlib.util
import json
import runpy
import sys
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--script", required=True)
    args, rest = parser.parse_known_args()
    torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0, str(Path(args.script).parent))
    hook = Path(__file__).with_name("propainter_offline.py")
    spec = importlib.util.spec_from_file_location("propainter_offline", hook)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.install_offline_hook()
    sys.argv = [args.script] + rest
    try:
        runpy.run_path(args.script, run_name="__main__")
    finally:
        Path(args.metrics).write_text(
            json.dumps(
                {
                    "allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                    "reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
                }
            )
        )


if __name__ == "__main__":
    main()
