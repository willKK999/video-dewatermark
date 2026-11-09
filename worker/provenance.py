"""Official model catalog. Production refuses unresolved checksums."""

import hashlib
import json
from pathlib import Path


class ProvenanceError(RuntimeError):
    pass


def load_catalog(path):
    catalog = json.loads(Path(path).read_text())
    if not isinstance(catalog.get("models"), list) or not catalog["models"]:
        raise ProvenanceError("model catalog is empty")
    return catalog


def _digest_ok(value):
    text = str(value or "")
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def validate_provenance(path, weights_dir=None):
    """Fail closed before any job is consumed.

    Phase-2 entries are ignored. Every other asset must carry a 64-character
    SHA-256. UNRESOLVED, missing, or mismatched digests abort startup.
    weights_dir is checked only after every required digest is published.
    """
    catalog = load_catalog(path)
    for model in catalog["models"]:
        if model.get("phase") == 2:
            continue
        revision = str(model.get("source_revision") or "")
        if not revision or revision == "UNRESOLVED":
            raise ProvenanceError(f"{model.get('name', 'model')} source revision unresolved")
        assets = model.get("assets") or []
        if not assets:
            raise ProvenanceError(f"{model.get('name', 'model')} has no pinned assets")
        for asset in assets:
            digest = asset.get("sha256")
            if not _digest_ok(digest):
                raise ProvenanceError(
                    f"{asset.get('file', 'asset')} weight digest unresolved"
                )
            if weights_dir:
                file = Path(weights_dir) / asset["file"]
                if not file.is_file() or _sha256(file) != digest:
                    raise ProvenanceError(f"{asset['file']} weight checksum mismatch")
    return catalog


def assert_catalog_locked(catalog, lock):
    """The operator lock must cover every published catalog digest, including Tiny."""
    wanted = set()
    for model in catalog["models"]:
        if model.get("phase") == 2:
            continue
        for asset in model.get("assets") or []:
            if not _digest_ok(asset.get("sha256")):
                raise ProvenanceError("catalog digest unresolved")
            wanted.add(asset["sha256"])
    locked = {entry.get("sha256") for entry in lock.get("files", [])}
    if not wanted or not wanted <= locked:
        raise ProvenanceError("model lock does not cover catalog checksums")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
