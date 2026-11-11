"""仅保存散列与阶段指标；检查点须验证内容，不依据文件存在就跳过。"""

import hashlib
import json
import time
from pathlib import Path
from .roi_inpaint import digest


def fingerprint(paths, parameters):
    entries = []
    for path in paths:
        path = Path(path)
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        entries.extend(
            (str(f.relative_to(path)) if path.is_dir() else str(path), digest(f))
            for f in files
            if f.is_file()
            and f.name != "key.bin"
            and not f.name.endswith(".env")
            and (
                not path.is_dir()
                or f.suffix in {".png", ".jpg", ".jpeg", ".json", ".npz"}
            )
        )
    return hashlib.sha256(
        json.dumps([entries, parameters], sort_keys=True).encode()
    ).hexdigest()


class StageCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.timings = {}
        self.tokens = {}

    def token(self, name, saved):
        self.tokens[name] = hashlib.sha256(
            json.dumps([saved["identity"], saved["outputs"]], sort_keys=True).encode()
        ).hexdigest()

    def run(self, name, identity, operation, outputs, clean=lambda: None):
        marker = self.root / (name + ".json")
        saved = None
        if marker.exists():
            try:
                saved = json.loads(marker.read_text())
                if (
                    saved["identity"] != identity
                    or not saved["outputs"]
                    or any(
                        not Path(p).is_file() or digest(p) != value
                        for p, value in saved["outputs"].items()
                    )
                ):
                    saved = None
            except (ValueError, KeyError, OSError):
                saved = None
        if saved:
            self.timings[name] = dict(
                seconds=0.0, reused=True, original_seconds=saved["seconds"]
            )
            self.token(name, saved)
            (self.root / "timings.json").write_text(
                json.dumps(self.timings, ensure_ascii=False, indent=2)
            )
            return
        marker.unlink(missing_ok=True)
        clean()
        started = time.perf_counter()
        try:
            operation()
            files = [
                f
                for path in outputs()
                for f in (
                    sorted(Path(path).rglob("*"))
                    if Path(path).is_dir()
                    else [Path(path)]
                )
                if f.is_file()
            ]
            if not files:
                raise ValueError("阶段没有可校验的输出")
            saved = dict(
                identity=identity,
                seconds=time.perf_counter() - started,
                outputs={str(p): digest(p) for p in files},
            )
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps(saved, ensure_ascii=False))
            temporary.replace(marker)
            self.token(name, saved)
        finally:
            self.timings[name] = dict(
                seconds=time.perf_counter() - started, reused=False
            )
            (self.root / "timings.json").write_text(
                json.dumps(self.timings, ensure_ascii=False, indent=2)
            )
