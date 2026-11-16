import json
import cv2
import numpy as np
import pytest
from worker.sam31_masks import Config, accepted_tracks, binary, checked_union, run


def test_explicit_enable_and_limit(tmp_path):
    with pytest.raises(ValueError, match='显式启用'):
        run(tmp_path, tmp_path/'out', tmp_path/'weight', Config())
    with pytest.raises(ValueError, match='不得放宽'):
        Config(max_layer_fraction=.21)


def test_persistence_uses_identity_not_position():
    cfg = Config(min_track_frames=3, min_track_fraction=.5)
    assert accepted_tracks({'moving': [0,1,2], 'flash': [0], 'sparse': [0,10,20]}, cfg) == {'moving'}


def test_union_rejects_bad_input():
    with pytest.raises(ValueError):
        binary(np.zeros((2,3)), (3,2))
    with pytest.raises(ValueError):
        binary(np.full((2,2),128), (2,2))
    assert np.all(checked_union([np.eye(2),1-np.eye(2)], (2,2)) == 255)


class Predictor:
    def __init__(self, incomplete=False, large=False):
        self.closed = 0
        self.incomplete = incomplete
        self.large = large
    def handle_request(self, request):
        if request['type'] == 'close_session': self.closed += 1
        return {'session_id':'test'}
    def handle_stream_request(self, request):
        for i in range(2 if self.incomplete else 3):
            masks = np.zeros((2,10,10), bool)
            masks[0, i, i] = True  # 移动对象持续存在，应保留。
            if self.large: masks[0,:6,:6] = True
            masks[1,9,9] = i == 0  # 一帧误报，应剔除。
            yield {'frame_index':i,'outputs':dict(out_obj_ids=[1,2],out_probs=[.9,.9],out_binary_masks=masks)}


def prepare(tmp):
    frames, base = tmp/'frames', tmp/'base'
    frames.mkdir(); base.mkdir()
    for i in range(3):
        cv2.imwrite(str(frames/f'{i:06d}.png'), np.zeros((10,10,3),np.uint8))
        m=np.zeros((10,10),np.uint8); m[8,8]=255
        cv2.imwrite(str(base/f'{i:06d}.png'),m)
    checkpoint=tmp/'checkpoint'; checkpoint.write_bytes(b'test-weight')
    return frames,base,checkpoint


def test_video_source_and_merge_preserve_baseline(tmp_path):
    frames,base,checkpoint=prepare(tmp_path)
    p=Predictor()
    data=run(frames,tmp_path/'out',checkpoint,Config(enabled=True,prompts=('logo',)),base,p)
    assert p.closed == 1
    assert len(data['layers']) == 1
    assert [r['sam_pixels'] for r in data['frames']] == [1,1,1]
    assert [r['added_pixels'] for r in data['frames']] == [1,1,1]
    assert all(r['removed_pixels']==0 for r in data['frames'])
    assert data['layers'][0]['maximum_fraction'] == .01
    assert len(data['layers'][0]['sha256']) == 3


def test_missing_propagation_frame_fails_and_closes(tmp_path):
    frames,base,checkpoint=prepare(tmp_path)
    p=Predictor(incomplete=True)
    with pytest.raises(ValueError, match='全部输入帧'):
        run(frames,tmp_path/'out',checkpoint,Config(enabled=True,prompts=('logo',)),base,p)
    assert p.closed == 1
    assert not (tmp_path/'out/metrics.json').exists()


def test_large_object_is_rejected_without_splitting(tmp_path):
    frames,base,checkpoint=prepare(tmp_path)
    data=run(frames,tmp_path/'out',checkpoint,Config(enabled=True,prompts=('logo',)),base,Predictor(large=True))
    assert data['layers'] == []
    assert all(r['sam_pixels']==0 for r in data['frames'])
    assert '20%' in data['tracks']['w000000-p0-o1']['rejection']


def test_official_session_api_compatibility_rejects_unsupported_offload():
    from types import SimpleNamespace
    from worker.sam31_masks import adapt_session_api
    class Model:
        def init_state(self, resource_path, offload_video_to_cpu=False):
            return resource_path, offload_video_to_cpu
    predictor=SimpleNamespace(model=Model())
    adapt_session_api(predictor)
    assert predictor.model.init_state(resource_path='frames',offload_video_to_cpu=True,offload_state_to_cpu=False)==('frames',True)
    with pytest.raises(ValueError,match='不支持'):
        predictor.model.init_state(resource_path='frames',offload_state_to_cpu=True)


def test_inference_batch_only_changes_resource_batch_sizes():
    from types import SimpleNamespace
    from worker.sam31_masks import configure_inference_batches
    model=SimpleNamespace(batched_grounding_batch_size=16,postprocess_batch_size=16,max_num_objects=16,score_threshold_detection=.4)
    configure_inference_batches(SimpleNamespace(model=model),Config(enabled=True,inference_batch_size=1))
    assert model.batched_grounding_batch_size == model.postprocess_batch_size == 1
    assert model.max_num_objects == 16 and model.score_threshold_detection == .4
    with pytest.raises(ValueError):Config(inference_batch_size=0)
