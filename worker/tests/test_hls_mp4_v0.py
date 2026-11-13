"""明文整集调度保留源规格、失败证据和完整音轨。"""

import json
from pathlib import Path
import pytest
from scripts import hls_mp4_v0 as runner
from worker.process import run
from worker.vision import sha256


def source(root, rate=15):
    (root / 'out').mkdir()
    path = root / 'source.mp4'
    run(['ffmpeg','-v','error','-y','-f','lavfi','-i',f'testsrc2=size=64x96:rate={rate}',
         '-f','lavfi','-i','sine=sample_rate=44100','-t','2.07','-c:v','libx264',
         '-threads','1','-pix_fmt','yuv420p','-c:a','aac',path])
    return path


def test_wrong_digest_stops_before_planning(tmp_path):
    source(tmp_path)
    with pytest.raises(ValueError, match='摘要不匹配'):
        runner.initialize(tmp_path, '0' * 64)
    assert not (tmp_path / 'run.json').exists()


@pytest.mark.parametrize('rate', [15, 24, 30])
def test_plans_derive_portrait_geometry_rate_and_duration(tmp_path, rate):
    path = source(tmp_path, rate)
    data = runner.initialize(tmp_path, sha256(path))
    assert data['源尺寸'] == [64, 96]
    assert data['源帧率'] == rate
    assert data['计划'][-1]['end'] == data['源帧数']
    assert all((p['right'] - p['left']) / data['处理帧率'] <= 30 for p in data['计划'])


def test_partial_model_frames_keep_originals_and_report_missing(tmp_path):
    for name in ['frames', 'composite']:
        (tmp_path / name).mkdir()
    for i in range(3):
        (tmp_path / 'frames' / f'{i:06d}.png').write_bytes(b'original')
    (tmp_path / 'composite/000001.png').write_bytes(b'repaired')
    dest = tmp_path / 'selected'
    assert runner.select_frames(tmp_path, 3, dest) == [0, 2]
    assert [p.read_bytes() for p in sorted(dest.iterdir())] == [b'original', b'repaired', b'original']


def test_prior_cache_rejects_other_source_and_modified_template(tmp_path):
    prior = tmp_path / 'multitrack-prior.npz'
    prior.write_bytes(b'automatic-content-template')
    report = dict(通过=True, 源摘要='source-a', 先验摘要=sha256(prior))
    runner.write(tmp_path / 'prior-report.json', report)
    assert runner.validated_prior(tmp_path, 'source-a') == prior
    with pytest.raises(ValueError, match='内容摘要不一致'):
        runner.validated_prior(tmp_path, 'source-b')
    prior.write_bytes(b'modified-template')
    with pytest.raises(ValueError, match='内容摘要不一致'):
        runner.validated_prior(tmp_path, 'source-a')


def test_single_track_references_trim_context(tmp_path):
    detected = dict(scenes=[[0, 6], [6, 10]], hints=[dict(frame=i, box=[0,0,1,1]) for i in range(10)])
    result = runner.cropped_references(detected, 3, 8)
    assert result['scenes'] == [[0, 3], [3, 5]]
    assert [p['frame'] for p in result['hints']] == list(range(5))
    assert detected['scenes'] == [[0, 6], [6, 10]]


def test_delivery_keeps_full_source_audio_and_does_not_claim_acceptance(tmp_path):
    path = source(tmp_path)
    data = runner.initialize(tmp_path, sha256(path))
    chunk = tmp_path / 'out/chunk-01'
    chunk.mkdir()
    run(['ffmpeg','-v','error','-y','-i',path,'-an','-c:v','copy',chunk/'video.mp4'])
    runner.write(chunk/'report.json',dict(块号=1,核验={},阶段失败=[]))
    report = runner.deliver(tmp_path, data)
    assert report['技术检查通过'] is True
    assert report['音频压缩数据一致'] is True
    assert report['通过'] is False
    assert report['未通过块'] == [1]
    assert report['输出规格']['帧率'] == 15
