import json
import numpy as np
from scripts.diagnose_watermark_residue import intervals,bounds,color_blue,write
from scripts.diagnose_mask_color_tracks import overlap


def test_intervals_support_numpy_and_inclusive_endpoints():
    assert intervals(np.array([8,2,3,3,9,12]))==[[2,3],[8,9],[12,12]]
    json.dumps(intervals(np.array([2,3])))


def test_empty_intervals_and_bounds_are_not_missing_lifecycles():
    assert intervals([])==[] and bounds(np.zeros((4,5),bool)) is None


def test_bounds_are_exclusive_at_right_and_bottom():
    m=np.zeros((10,20),bool);m[2:7,4:12]=True
    assert bounds(m)==[4,2,12,7]


def test_blue_candidate_requires_color_and_brightness():
    x=np.array([[[255,0,0],[120,120,120],[20,40,60],[20,0,0]]],np.uint8)
    assert color_blue(x).tolist()==[[True,False,False,False]]


def test_summary_writer_preserves_numpy_counts_and_box_iou(tmp_path):
    p=tmp_path/'audit.json';write(p,dict(pixels=np.int64(17),passed=np.bool_(True)))
    assert json.loads(p.read_text())==dict(pixels=17,passed=True)
    assert overlap([0,0,10,10],[0,0,10,10])==1
    assert overlap([0,0,10,10],[10,10,20,20])==0
