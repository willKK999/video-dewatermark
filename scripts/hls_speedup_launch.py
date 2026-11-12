"""用模型解释器启动基准，后置复用只读纯Python依赖，禁止写入依赖目录缓存。"""

import argparse
import os
from pathlib import Path
import runpy
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--extra-python-path",
        type=Path,
        action="append",
        default=[],
        help="只读依赖目录，附加在现有搜索路径之后；其余参数传给基准脚本",
    )
    args, remaining = parser.parse_known_args(argv)
    for path in args.extra_python_path:
        if not path.is_dir():
            parser.error("额外Python依赖路径必须是已有目录")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for path in args.extra_python_path:
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.append(resolved)
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    sys.argv = ["scripts.hls_speedup_bench", *remaining]
    runpy.run_module("scripts.hls_speedup_bench", run_name="__main__")


if __name__ == "__main__":
    main()
