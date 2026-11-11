"""清晰度最新策略不改变残留、闪烁与目检的硬门槛。"""

from worker.hls_acceptance import clarity_summary, hard_acceptance
from worker.hls_v7 import load_assets, scale_plan
from worker.process import PipelineError
import hashlib, json
import pytest


def test_low_clarity_is_recorded_without_becoming_rejection():
    samples = [
        dict(measurable=True, retention_ratio=q, frame=i)
        for i, q in enumerate([0.69, 0.7, 0.79, 0.9])
    ]
    summary = clarity_summary(samples)
    assert summary["全部样本"] == samples
    assert summary["低于0.7样本"] == [samples[0]]
    assert summary["最小保留比"] == 0.69
    assert summary["不是验收硬门槛"] is True
    assert hard_acceptance(0, True, True, True)


@pytest.mark.parametrize(
    "args",
    [
        (1, True, True, True),
        (0, False, True, True),
        (0, True, False, True),
        (0, True, True, False),
        (0, True, True, "待目检"),
    ],
)
def test_hard_gates_are_preserved(args):
    assert not hard_acceptance(*args)


def test_empty_or_unmeasurable_clarity_does_not_invent_minimum():
    assert clarity_summary([])["最小保留比"] is None
    assert (
        clarity_summary([dict(measurable=False, retention_ratio=0.1)])["低于0.7样本数"]
        == 0
    )


def test_source_and_assets_must_match_before_reuse(tmp_path):
    prior = tmp_path / "prior"
    prior.write_bytes(b"validated-prior")
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            dict(
                source_sha256="source-a",
                method="v7",
                prior=str(prior),
                sha256=dict(prior=hashlib.sha256(prior.read_bytes()).hexdigest()),
            )
        )
    )
    assert load_assets(manifest, "source-a")["method"] == "v7"
    with pytest.raises(PipelineError):
        load_assets(manifest, "other-source")
    prior.write_bytes(b"changed")
    with pytest.raises(PipelineError):
        load_assets(manifest, "source-a")


def test_native_plan_keeps_same_timestamps_and_context():
    assert scale_plan(dict(start=750, end=1500, left=690, right=1560), 2) == dict(
        start=1500, end=3000, left=1380, right=3120
    )


def test_false_positive_review_must_cover_exact_current_candidates(tmp_path):
    from worker.hls_final_audit import approved_review

    proof = tmp_path / "proof.jpg"
    proof.write_bytes(b"local-evidence")
    path = tmp_path / "review.json"
    data = {
        "视频摘要": "video",
        "候选摘要": "candidates",
        "逐项复核": [{"候选序号": 0, "结论": "误报", "证据文件": str(proof)}],
    }
    path.write_text(json.dumps(data, ensure_ascii=False))
    assert approved_review(path, "video", "candidates", 1)
    assert not approved_review(path, "changed", "candidates", 1)
    assert not approved_review(path, "video", "changed", 1)
    assert not approved_review(path, "video", "candidates", 2)


def test_full_report_accepts_low_clarity_only_after_hard_checks(tmp_path):
    from scripts.hls_full_report import build

    (tmp_path / "audit").mkdir()
    sample = dict(time=0.5, frame=30, measurable=True, retention_ratio=0.62)
    verification = dict(
        block_audits=[dict(index=1, clarity={"全部样本": [sample]})],
        chunks=[dict(left=0)],
        final_encoded_audit={"通过": True},
        hls_mp4_decoded_frames_equal=True,
        audio_copy_verified=True,
        elapsed_ms=1000,
        source_duration_ms=250133,
        result_duration_ms=250133,
        hls_frame_count=15008,
        audio_timeline_max_error_ms=0,
        av_start_offset_ms=0,
        av_end_offset_ms=62,
        seams=[],
        motion_compensated_seams=[],
    )
    (tmp_path / "audit/verification.json").write_text(json.dumps(verification))
    (tmp_path / "browser.json").write_text(
        json.dumps(
            {k: True for k in ["hls_playback", "seek", "mp4_download", "zip_download"]}
        )
    )
    report = build(tmp_path)
    assert report["通过"] is False
    (tmp_path / "目检.json").write_text(json.dumps({"通过": True, "无闪烁": True}))
    report = build(tmp_path)
    assert report["通过"] is True
    assert report["清晰度"]["低于0.7样本数"] == 1
    assert report["清晰度"]["全部样本"][0]["retention_ratio"] == 0.62


def test_clarity_sampling_covers_both_phases_and_late_target_boundaries():
    from scripts.hls_detail_verify import sample_pairs

    indices = [i // 2 for i in range(180)]
    d = dict(
        scenes=[[0, 45], [45, 90]],
        tracks=[dict(kind="moving", hints=[dict(frame=23), dict(frame=37)])],
    )
    selected = sample_pairs(indices, d)
    assert {30, 31, 60, 61, 120, 121} <= selected
    assert set(range(44, 49)) <= selected
    assert set(range(72, 77)) <= selected
    assert set(range(88, 93)) <= selected
    assert sample_pairs(indices, d, 1) == set(range(1, 180))


def test_block_review_covers_low_band_and_all_detectors_without_discarding_raw_items():
    from worker.hls_v7 import block_candidates
    point = dict(correlation=.6, relative_amplitude=.3)
    quality = dict(moving_evidence=[dict(frame=5)], fixed_template_scores=[dict(components=[point])], residual_count=2)
    brand = dict(证据=[dict(frame=7)], 残留证据数=1)
    panel = dict(证据=[dict(frame=8)], 候选数=1)
    low = dict(区域=[dict(region=4, residual=True, components=[point])], 残留区域数=1)
    rows = block_candidates(quality, brand, panel, low)
    assert len(rows) == 5 and rows[-1]["类型"] == "完整图形低频"
    assert rows[-1]["详情"] == low["区域"][0]
    with pytest.raises(PipelineError):
        block_candidates(dict(quality, residual_count=1), brand, panel, low)


def test_same_candidate_count_cannot_reuse_a_different_candidate_review(tmp_path):
    from worker.hls_final_audit import approved_review
    proof = tmp_path / "对照图.jpg"
    proof.write_bytes(b"evidence")
    path = tmp_path / "review.json"
    path.write_text(json.dumps(dict(视频摘要="same-video", 候选摘要="old-candidates", 逐项复核=[dict(候选序号=0, 结论="误报", 证据文件=str(proof))])))
    assert not approved_review(path, "same-video", "new-candidates", 1)
