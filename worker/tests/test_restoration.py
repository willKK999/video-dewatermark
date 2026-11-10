import cv2
import numpy as np
from worker.restoration import CleanReferences, context_mask, soft_composite


def test_feather_keeps_core_and_outside_exact():
    original = np.full((40, 50, 3), 80, np.uint8)
    repair = np.full_like(original, 160)
    mask = np.zeros((40, 50), np.uint8)
    mask[10:20, 10:30] = 255
    result = soft_composite(original, repair, mask)
    assert np.all(result[13:17, 13:27] == 160)
    assert np.array_equal(result[context_mask(mask) == 0], original[context_mask(mask) == 0])
    assert 80 < result[10, 15, 0] < 160
    assert np.array_equal(result[mask == 0], original[mask == 0])
    assert np.array_equal(soft_composite(original, repair, np.zeros_like(mask)), original)


def background():
    rng = np.random.default_rng(8)
    return cv2.GaussianBlur(rng.integers(20, 230, (160, 180, 3), dtype=np.uint8), (3, 3), 0)


def test_clean_references_recover_background_without_modifying_subtitles():
    clean = background()
    mask = np.zeros(clean.shape[:2], np.uint8)
    mask[60:85, 65:95] = 255
    image = clean.copy()
    image[mask > 0] = 250
    repair = clean.copy()
    repair[mask > 0] = 80
    references = CleanReferences([clean.copy() for _ in range(3)], [np.zeros_like(mask) for _ in range(3)])
    result = references.restore(image, mask, repair)
    assert np.abs(result[65:80, 70:90].astype(float)-clean[65:80, 70:90]).mean() < 1
    assert np.array_equal(result[mask == 0], repair[mask == 0])


def test_masked_or_disagreeing_donors_do_not_fill():
    clean = background()
    mask = np.zeros(clean.shape[:2], np.uint8)
    mask[60:85, 65:95] = 255
    repair = np.full_like(clean, 50)
    refs = CleanReferences([clean]*3, [mask]*3)
    assert np.array_equal(refs.restore(clean, mask, repair), repair)
    donors = [clean.copy() for _ in range(3)]
    donors[2][mask > 0] = 255
    refs = CleanReferences(donors, [np.zeros_like(mask)]*3)
    assert np.array_equal(refs.restore(clean, mask, repair), repair)


def test_textureless_and_insufficient_evidence_fall_back():
    image = np.full((50, 60, 3), 80, np.uint8)
    mask = np.full((50, 60), 255, np.uint8)
    assert np.array_equal(CleanReferences([image]*3, [mask]*3).restore(image, mask, image), image)


def test_affine_reference_alignment():
    clean = background()
    mask = np.zeros(clean.shape[:2], np.uint8)
    mask[65:90, 70:100] = 255
    damaged = clean.copy()
    damaged[mask > 0] = 250
    donors = [cv2.warpAffine(clean, np.float32([[1, 0, shift], [0, 1, 2]]), (180, 160)) for shift in [2, 4, 6]]
    refs = CleanReferences(donors, [np.zeros_like(mask)]*3)
    repair = np.full_like(clean, 50)
    result = refs.restore(damaged, mask, repair)
    assert np.abs(result[70:85, 75:95].astype(float)-clean[70:85, 75:95]).mean() < 2
    assert np.array_equal(result[mask == 0], repair[mask == 0])


def test_unrelated_scene_references_are_rejected():
    clean = background()
    mask = np.zeros(clean.shape[:2], np.uint8)
    mask[65:90, 70:100] = 255
    repair = np.full_like(clean, 50)
    refs = CleanReferences([255-clean]*3, [np.zeros_like(mask)]*3)
    assert np.array_equal(refs.restore(clean, mask, repair), repair)


def test_inpaint_passes_context_masks_and_resets_references_at_cuts(tmp_path, monkeypatch):
    import worker.vision as vision
    monkeypatch.setenv('INPAINT_PYTHON', 'python')
    monkeypatch.setenv('PROPAINTER_ROOT', str(tmp_path))
    (tmp_path/'masks').mkdir()
    (tmp_path/'frames').mkdir()
    mask = np.zeros((40, 50), np.uint8)
    mask[15:25, 20:30] = 255
    paths=[]
    for i in range(5):
        path=tmp_path/'frames'/f'{i:06d}.png'
        cv2.imwrite(str(path), np.full((40,50,3), 30 if i < 3 else 90, np.uint8))
        cv2.imwrite(str(tmp_path/'masks'/path.name), mask)
        paths.append(path)
    reference_scenes=[]
    class References:
        def __init__(self, images, masks):
            self.stats = {}
            reference_scenes.append([int(im[0,0,0]) for im in images])
        def restore(self, original, mask, repaired):
            return repaired
    monkeypatch.setattr(vision, 'CleanReferences', References)
    def run(args, *unused, **kwargs):
        args=list(map(str,args))
        frames=__import__('pathlib').Path(args[args.index('--video')+1])
        masks=__import__('pathlib').Path(args[args.index('--mask')+1])
        output=__import__('pathlib').Path(args[args.index('--output')+1])/'frames'/'frames'
        output.mkdir(parents=True)
        for path in frames.glob('*.png'):
            assert np.array_equal(cv2.imread(str(masks/path.name),0), mask)
            cv2.imwrite(str(output/path.name), np.full((40,50,3),150,np.uint8))
    monkeypatch.setattr(vision,'run',run)
    vision.inpaint(tmp_path,paths,[(0,3),(3,5)],'propainter')
    assert reference_scenes == [[30,30,30],[90,90]]
    for path in paths:
        result=cv2.imread(str(tmp_path/'composite'/path.name))
        original=cv2.imread(str(path))
        assert np.all(result[18:22,23:27] == 150)
        assert np.array_equal(result[context_mask(mask)==0],original[context_mask(mask)==0])


def test_partial_visible_ring_retains_verified_clean_pixels():
    clean = background()
    mask = np.zeros(clean.shape[:2], np.uint8)
    mask[60:85, 65:95] = 255
    damaged = clean.copy()
    damaged[mask > 0] = 250
    donor_mask = np.zeros_like(mask)
    donor_mask[50:95, 40:80] = 255
    refs = CleanReferences([clean.copy() for _ in range(3)], [donor_mask]*3)
    repair = np.full_like(clean, 50)
    result = refs.restore(damaged, mask, repair)
    assert np.abs(result[65:80, 86:92].astype(float)-clean[65:80, 86:92]).mean() < 2
    assert np.array_equal(result[65:80, 65:80],repair[65:80, 65:80])
    assert refs.stats['restored_pixels'] > 0


def test_small_smooth_consensus_holes_are_filled_within_mask():
    from worker.restoration import fill_small_holes
    background = np.full((40, 50, 3), 100, np.uint8)
    mask = np.zeros((40, 50), np.uint8)
    mask[8:32, 8:42] = 255
    confidence = (mask > 0).astype(np.uint8)
    confidence[18:21, 20:23] = 0
    background[18:21, 20:23] = 0
    before = background.copy()
    result, reliable, filled = fill_small_holes(background, confidence, mask)
    assert filled == 9 and np.all(reliable[18:21,20:23] == 1)
    assert result[18:21,20:23].min() >= 95
    assert np.array_equal(result[mask == 0], before[mask == 0])


def test_textured_or_large_consensus_gaps_keep_model_result():
    from worker.restoration import fill_small_holes
    mask = np.full((40,50), 255, np.uint8)
    confidence = np.ones_like(mask)
    confidence[15:18,20:23] = 0
    background = np.indices(mask.shape).sum(axis=0) % 2 * 200
    background = np.repeat(background[...,None],3,axis=2).astype(np.uint8)
    before = background.copy()
    result, reliable, filled = fill_small_holes(background, confidence, mask)
    assert filled == 0 and np.array_equal(result,before)
    confidence[8:32,20:24] = 0
    background = np.full((40,50,3),100,np.uint8)
    _, _, filled = fill_small_holes(background,confidence,mask)
    assert filled == 0


def test_inner_feather_does_not_mix_watermark_back():
    original = np.full((50,60,3),70,np.uint8)
    actual = np.zeros((50,60),np.uint8)
    actual[20:30,20:40] = 255
    original[actual > 0] = 240
    mask = context_mask(actual,3)
    result = soft_composite(original,np.full_like(original,70),mask)
    assert np.all(result[actual > 0] == 70)
    assert np.array_equal(result[mask == 0], original[mask == 0])


def test_sparse_feather_matches_dense_arithmetic_exactly():
    rng = np.random.default_rng(92)
    original = rng.integers(0, 256, (73, 81, 3), dtype=np.uint8)
    repaired = rng.integers(0, 256, original.shape, dtype=np.uint8)
    mask = np.zeros(original.shape[:2], np.uint8)
    mask[0:17, 8:23] = 255
    mask[43:73, 65:81] = 255
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(distance/3, 0, 1)
    expected = np.rint(original*(1-alpha[..., None])+repaired*alpha[..., None]).astype(np.uint8)
    expected[mask == 0] = original[mask == 0]
    assert np.array_equal(soft_composite(original, repaired, mask), expected)


def test_next_model_window_overlaps_ordered_compositing(tmp_path, monkeypatch):
    import threading
    import worker.vision as vision
    monkeypatch.setenv('INPAINT_PYTHON', 'python')
    monkeypatch.setenv('PROPAINTER_ROOT', str(tmp_path))
    (tmp_path/'frames').mkdir()
    (tmp_path/'masks').mkdir()
    mask = np.zeros((40, 50), np.uint8)
    mask[15:25, 20:30] = 255
    paths = []
    for i in range(65):
        path = tmp_path/'frames'/f'{i:06d}.png'
        cv2.imwrite(str(path), np.full((40, 50, 3), 80, np.uint8))
        cv2.imwrite(str(tmp_path/'masks'/path.name), mask)
        paths.append(path)
    next_model = threading.Event()
    writes = []

    class References:
        def __init__(self, *args):
            self.stats = {}
        def restore(self, original, mask, repaired):
            assert next_model.wait(timeout=5)
            writes.append(len(writes))
            return repaired

    calls = []
    def run(args, *unused, **kwargs):
        args = list(map(str, args))
        frames = __import__('pathlib').Path(args[args.index('--video')+1])
        output = __import__('pathlib').Path(args[args.index('--output')+1])/'frames'/'frames'
        output.mkdir(parents=True)
        calls.append(frames)
        if len(calls) == 2:
            next_model.set()
        for path in frames.glob('*.png'):
            cv2.imwrite(str(output/path.name), np.full((40, 50, 3), 150, np.uint8))

    monkeypatch.setattr(vision, 'CleanReferences', References)
    monkeypatch.setattr(vision, 'run', run)
    vision.inpaint(tmp_path, paths, [(0, 65)], 'propainter')
    assert len(calls) == 2 and len(writes) == 77
    assert len(list((tmp_path/'composite').glob('*.png'))) == 65
    assert not list(tmp_path.glob('window-*'))
