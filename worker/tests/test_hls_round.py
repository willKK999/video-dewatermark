"""单块补修失败不得阻止后续块；未知指标不能伪造为零。"""

from worker.hls_round import execute_round, legacy_attempts
from worker.hls_settings import HLSSettings
from worker.process import PipelineError
import pytest


def test_failure_preserves_current_artifact_and_runs_all_later_stages(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes("完整基底".encode())
    called = []

    def fail():
        called.append("补修")
        raise PipelineError("SEAM_UNCERTAIN", "测试失败")

    def verify():
        called.append("复检")

    report = execute_round(tmp_path, [("补修", fail), ("复检", verify)], video)
    assert called == ["补修", "复检"] and video.read_bytes() == "完整基底".encode()
    assert (
        report["完成"]
        and report["产物可用"]
        and report["未通过原因"] == ["补修：SEAM_UNCERTAIN"]
    )
    assert (tmp_path / "round-inputs/补修.mp4").read_bytes() == "完整基底".encode()
    execute_round(tmp_path, [("补修", fail), ("复检", verify)], video)
    assert called == ["补修", "复检"]


def test_incomplete_legacy_attempt_is_recorded_without_retry(tmp_path):
    folder = tmp_path / "complete-graphics"
    folder.mkdir()
    (folder / "low-before.json").write_text("{}")
    video = tmp_path / "video.mp4"
    video.write_bytes("完成视频".encode())
    attempted = []
    report = execute_round(
        tmp_path,
        [
            ("低频检查与补修", lambda: attempted.append(1)),
            ("编码后复检", lambda: attempted.append(2)),
        ],
        video,
        legacy_attempts(tmp_path),
    )
    assert attempted == [2] and report["未通过原因"]


def test_generic_configuration_rejects_oversized_blocks_and_keeps_defaults(monkeypatch):
    config = HLSSettings.from_env()
    assert config.detail_sigma == 3 and config.video_crf == 10
    monkeypatch.setenv("HLS_CONTEXT_SECONDS", "4")
    with pytest.raises(ValueError):
        HLSSettings.from_env()


def test_v0_report_preserves_unknown_metrics_and_requires_review(tmp_path):
    import json
    from scripts.hls_v0 import block_report

    (tmp_path / "native").mkdir()
    (tmp_path / "native/native-report.json").write_text(json.dumps({"渲染耗时秒": 3}))
    first = block_report(tmp_path, 7, {"left": 0})
    assert first["残留"] is None and first["闪烁"] is None and not first["是否通过"]
    (tmp_path / "verified.json").write_text(
        json.dumps(
            {
                "通过": True,
                "残留": 0,
                "低频残留": 0,
                "清晰度最低值": 0.2,
                "清晰度低于0点7样本数": 9,
            }
        )
    )
    second = block_report(tmp_path, 7, {"left": 0}, {"无闪烁": True, "目检通过": True})
    assert second["是否通过"] and second["清晰度最低值"] == 0.2


def test_profile_is_optional_content_cache_not_film_rule(tmp_path):
    from worker.hls_settings import reusable_profile
    import json

    path = tmp_path / "cache.json"
    path.write_text(json.dumps(dict(method="v7", source_sha256="a")))
    assert reusable_profile(path, "a", 60)
    assert not reusable_profile(path, "b", 60)
    assert not reusable_profile(path, "a", 30)
    assert not reusable_profile(None, "a", 60)


def test_configuration_accepts_fractional_context_without_relaxing_gates(monkeypatch):
    monkeypatch.setenv("HLS_CONTEXT_SECONDS", "1.5")
    assert HLSSettings.from_env().context_seconds == 1.5
    monkeypatch.setenv("HLS_OVERLAP_MAE_LIMIT", "36")
    with pytest.raises(ValueError):
        HLSSettings.from_env()


def test_residual_configuration_can_only_become_stricter(monkeypatch):
    from worker.hls_settings import is_residual_score

    sample = dict(correlation=0.32, relative_amplitude=0.2)
    assert not is_residual_score(sample)
    monkeypatch.setenv("HLS_RESIDUAL_CORRELATION", ".3")
    assert is_residual_score(sample)
    monkeypatch.setenv("HLS_RESIDUAL_CORRELATION", ".36")
    with pytest.raises(ValueError):
        is_residual_score(sample)


def test_evidence_export_keeps_v0_values_and_failure_state(tmp_path):
    from scripts.hls_v0_evidence import enrich
    import json

    root = tmp_path / "work"
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    work = root / "chunk-03"
    (work / "audit").mkdir(parents=True)
    (work / "native").mkdir()
    original = dict(
        通过=False, 各块=[dict(块号=3, 是否通过=False, 清晰度最低值=0.0, 残留=37)]
    )
    (delivery / "report.json").write_text(json.dumps(original))
    (work / "audit/clarity.json").write_text(
        json.dumps({"全部样本": [dict(retention_ratio=0.0, measurable=True)]})
    )
    (work / "native/native-report.json").write_text(json.dumps({"清晰度统计": []}))
    result = enrich(root, delivery)
    assert (
        not result["通过"]
        and result["各块"][0]["残留"] == 37
        and result["各块"][0]["清晰度最低值"] == 0
    )
    assert result["各块"][0]["清晰度原始记录"]["全部样本"][0]["retention_ratio"] == 0
    assert json.loads((delivery / "report-v0-原始.json").read_text()) == original
