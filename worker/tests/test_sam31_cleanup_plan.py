from scripts import sam31_cleanup_plan as cleanup


def test_cleanup_only_numbered_intermediate_pngs_and_preserves_all_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup,'ROOT',tmp_path)
    safe=['v1/chunk/composite/000001.png','v6/chunk/encoded/000002.png','v2/detail/000003.png']
    preserved=['v9/composite/000001.png','v1/composite/masks/000001.png',
               'v1/object-masks/composite/000001.png','v1/evidence/composite/000001.png',
               'v1/composite/mask.png','v1/composite/报告.png','v1/frames/000001.png',
               'v1/composite/000001.mp4','v1/composite/中文报告.md']
    for name in safe+preserved:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
        assert cleanup.eligible(p)==(name in safe)
    link=tmp_path/'v1/composite/000003.png';link.symlink_to(tmp_path/safe[0])
    assert not cleanup.eligible(link)
