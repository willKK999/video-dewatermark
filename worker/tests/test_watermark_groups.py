from dataclasses import replace
import pytest
from worker.watermark_groups import GroupSettings,AnimationSettings,relation,connected_groups,animation_interval,planned_windows


def test_icon_text_scale_together_despite_screen_translation():
    a={};b={}
    for i in range(30):
        s=1+i*.04;a[i]=[20+i,30,20+i+20*s,30+20*s];b[i]=[20+i+25*s,30,20+i+60*s,30+20*s]
    result=relation(a,b,30,GroupSettings())
    assert result and result['scale_correlation']>.99


def test_same_screen_region_without_motion_agreement_is_not_group():
    a={i:[20,20,40,40] for i in range(30)};b={i:[60+i*3,20,80+i*3,40] for i in range(30)}
    assert relation(a,b,30,GroupSettings()) is None


def test_scene_content_never_joins_even_with_identical_motion():
    boxes={i:[20,20,40,40] for i in range(30)}
    groups,links=connected_groups({'logo':dict(scene=0,boxes=boxes),'panel':dict(scene=0,boxes=boxes,rejected=True)},30,GroupSettings())
    assert len(groups)==2 and not links


def test_mixed_node_cannot_bridge_independent_groups():
    a={i:[10,10,30,30] for i in range(30)};b={i:[80+i,10,100+i,30] for i in range(30)}
    # 同一混合候选与双方近邻，但双方本身的运动不相容。
    nodes={k:dict(scene=0,boxes=x) for k,x in [('a',a),('b',b),('mixed',{i:[20,10,90+i*.5,30] for i in range(30)})]}
    groups,_=connected_groups(nodes,30,replace(GroupSettings(),maximum_neighbor_size=10.,offset_residual=2.,size_residual=2.))
    assert not any('a' in group and 'b' in group for group in groups)


def test_animation_complete_interval_and_ordinary_coverage():
    cfg=AnimationSettings()
    schedule=planned_windows(0,200,36,8,[[70,130]],cfg)
    assert (62,138) in schedule
    assert set(range(200))=={i for a,b in schedule for i in range(a,b)}
    assert not any(70<a<130 or 70<b<130 for a,b in schedule)
    assert max(b-a for a,b in schedule)<=120


def test_long_animation_has_large_overlap_and_stays_in_scene():
    schedule=planned_windows(30,300,36,8,[[50,280]],AnimationSettings())
    event=[(a,b) for a,b in schedule if b-a>36]
    assert all(30<=a<b<=300 and b-a<=120 for a,b in schedule)
    assert all(min(b,d)-max(a,c)>=90 for (a,b),(c,d) in zip(event,event[1:]))


def test_no_animation_keeps_historical_windows():
    from worker.vision import windows
    assert planned_windows(0,100,36,8,[],AnimationSettings())==list(windows(0,100,36,8))
    assert animation_interval({i:[10,10,30,30] for i in range(90)},AnimationSettings()) is None
    assert animation_interval({i:[10,10,30+i,30+i] for i in range(40)},AnimationSettings())


def test_twenty_percent_cannot_be_relaxed():
    with pytest.raises(ValueError):GroupSettings(mask_fraction_limit=.21)
