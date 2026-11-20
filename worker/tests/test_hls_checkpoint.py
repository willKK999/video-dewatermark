"""失败块草稿必须无损恢复并拒绝损坏和路径穿越。"""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from botocore.exceptions import ClientError
from scripts.fixtures import generate
from worker.pipeline import execute
from worker.hls_checkpoint import save_draft, restore_draft
from worker.process import PipelineError


class MemoryStore:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key, **kwargs):
        self.objects[key] = Path(path).read_bytes()

    def download_file(self, bucket, key, path, **kwargs):
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        Path(path).write_bytes(self.objects[key])


def test_draft_restores_lossless_composite_masks_and_detection(tmp_path, monkeypatch):
    monkeypatch.delenv("DETECTOR", raising=False)
    monkeypatch.delenv("TEMPLATE_PATH", raising=False)
    source = generate(tmp_path)
    root = tmp_path / "work"
    root.mkdir()
    options = {
        "prompts": [
            dict(frame=0, box=[20, 30, 62, 52]),
            dict(frame=11, box=[53, 30, 95, 52]),
        ]
    }
    execute(source, root, options, "telea-dev")
    (root / "quality.json").write_text(
        json.dumps(dict(residual_count=1, needs_review=False))
    )
    (root / "quality-detections.json").write_text(
        json.dumps([dict(frame=4, box=[20, 30, 62, 52])])
    )
    s3 = MemoryStore()
    save_draft(s3, "bucket", "chunk", root, 12, lambda: None)
    restored = tmp_path / "restored"
    restored.mkdir()
    phase = restore_draft(
        s3, "bucket", "chunk", restored, source, options, False, lambda: None
    )
    assert phase["inpaint"] >= 0
    assert json.loads((restored / "quality.json").read_text())["residual_count"] == 1
    for path in (root / "composite").glob("*.png"):
        assert np.array_equal(
            cv2.imread(str(path)), cv2.imread(str(restored / "composite" / path.name))
        )
        assert (root / "masks" / path.name).read_bytes() == (
            restored / "masks" / path.name
        ).read_bytes()
    import io, zipfile

    original_archive = s3.objects["chunk.draft.zip"]
    buffer = io.BytesIO(original_archive)
    with zipfile.ZipFile(buffer, "a") as archive:
        archive.writestr("../escaped", "bad")
    s3.objects["chunk.draft.zip"] = buffer.getvalue()
    bad_path = tmp_path / "bad-path"
    bad_path.mkdir()
    with pytest.raises(PipelineError, match="路径"):
        restore_draft(
            s3, "bucket", "chunk", bad_path, source, options, False, lambda: None
        )
    assert not (tmp_path / "escaped").exists()
    s3.objects["chunk.draft.zip"] = original_archive
    s3.objects["chunk.draft.mkv"] += b"corruption"
    bad = tmp_path / "bad"
    bad.mkdir()
    with pytest.raises(PipelineError, match="摘要"):
        restore_draft(s3, "bucket", "chunk", bad, source, options, False, lambda: None)


def test_absent_draft_falls_back_to_full_pipeline(tmp_path):
    assert (
        restore_draft(
            MemoryStore(),
            "bucket",
            "missing",
            tmp_path,
            tmp_path / "source.mp4",
            {},
            False,
            lambda: None,
        )
        is None
    )
    assert not (tmp_path / "frames").exists()
