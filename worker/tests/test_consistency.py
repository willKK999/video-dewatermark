import json
from pathlib import Path

import pytest

from worker.delivery import ack_after_miss, reclaim_batch
from worker.main import verify_result_head
from worker.process import PipelineError
from worker.provenance import ProvenanceError, assert_catalog_locked, validate_provenance

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "deploy" / "models.json"


def test_shipped_catalog_blocks_production_startup():
    with pytest.raises(ProvenanceError, match="UNRESOLVED|digest unresolved"):
        validate_provenance(CATALOG)


def test_phase2_is_ignored_and_files_must_match(tmp_path):
    digest = "ab" * 32
    weight = tmp_path / "weights" / "demo.bin"
    weight.parent.mkdir()
    weight.write_bytes(b"model-bytes")
    import hashlib

    digest = hashlib.sha256(b"model-bytes").hexdigest()
    catalog = {
        "models": [
            {"name": "later", "phase": 2, "source_revision": "UNRESOLVED", "assets": []},
            {
                "name": "demo",
                "source_revision": "abc123",
                "assets": [{"file": "demo.bin", "sha256": digest}],
            },
        ]
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(catalog))
    loaded = validate_provenance(path, weight.parent)
    assert loaded["models"][0]["phase"] == 2
    weight.write_bytes(b"tampered")
    with pytest.raises(ProvenanceError, match="checksum mismatch"):
        validate_provenance(path, weight.parent)


def test_lock_must_cover_every_catalog_digest():
    catalog = {
        "models": [
            {
                "name": "demo",
                "assets": [{"file": "a.bin", "sha256": "cd" * 32}],
            }
        ]
    }
    with pytest.raises(ProvenanceError, match="does not cover"):
        assert_catalog_locked(catalog, {"files": [{"sha256": "11" * 32}]})
    assert_catalog_locked(catalog, {"files": [{"sha256": "cd" * 32}, {"sha256": "ee" * 32}]})


def test_reclaim_cursor_advances_and_version_mismatch_defers():
    cursor, messages = reclaim_batch("0-0", ["9-1", [("4-0", {"job_id": "job"})]])
    assert cursor == "9-1" and messages[0][0] == "4-0"
    again, empty = reclaim_batch(cursor, ["0-0", []])
    assert again == "0-0" and empty == []
    assert ack_after_miss(None, "mvp-1")
    assert ack_after_miss({"status": "RUNNING", "pipeline_version": "mvp-1"}, "mvp-1-dev")
    assert not ack_after_miss(
        {"status": "QUEUED", "pipeline_version": "mvp-1"}, "mvp-1-dev"
    )
    assert ack_after_miss({"status": "QUEUED", "pipeline_version": "mvp-1-dev"}, "mvp-1-dev")


def test_result_head_requires_length_and_sha256():
    verify_result_head({"ContentLength": 4, "Metadata": {"sha256": "abcd"}}, 4, "abcd")
    with pytest.raises(PipelineError) as exc:
        verify_result_head({"ContentLength": 4, "Metadata": {}}, 4, "abcd")
    assert exc.value.code == "OUTPUT_INVALID"
    with pytest.raises(PipelineError):
        verify_result_head({"ContentLength": 3, "Metadata": {"Sha256": "abcd"}}, 4, "abcd")
