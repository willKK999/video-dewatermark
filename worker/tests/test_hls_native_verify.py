"""原生帧率复检必须保留晚出现及重叠的所有独立轨迹。"""

from scripts.hls_native_verify import remap_references


def test_native_mapping_checks_every_repeated_and_late_observation():
    first = dict(
        id=1, hints=[dict(frame=0, box=[1, 2, 3, 4]), dict(frame=1, box=[2, 3, 4, 5])]
    )
    late = dict(id=2, hints=[dict(frame=1, box=[6, 7, 8, 9])])
    d = dict(tracks=[first, late], prior="先验")
    result = remap_references(d, [0, 0, 1, 1, 1])
    assert [p["frame"] for p in result["tracks"][0]["hints"]] == [0, 1, 2, 3, 4]
    assert [p["frame"] for p in result["tracks"][1]["hints"]] == [2, 3, 4]
    assert d["tracks"][1]["hints"][0]["frame"] == 1
