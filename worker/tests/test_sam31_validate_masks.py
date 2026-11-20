import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest
from worker.hls_layers import write_manifest
from scripts.sam31_validate_masks import main


@pytest.mark.parametrize('tamper',[False,True])
def test_independent_validation_rejects_merge_pixels_outside_sources(tmp_path,monkeypatch,tamper):
    frames=tmp_path/'frames';frames.mkdir();f=frames/'000000.png'
    cv2.imwrite(str(f),np.zeros((10,10,3),np.uint8))
    root=tmp_path/'sam';root.mkdir();layer=root/'layers/00';layer.mkdir(parents=True)
    sam=np.zeros((10,10),np.uint8);sam[1,1]=255
    base=np.zeros_like(sam);base[8,8]=255
    merged=sam|base
    if tamper:merged[5,5]=255
    for directory,mask in [(layer,sam),(root/'masks',sam),(root/'merged-masks',merged),(tmp_path/'base-layer',base)]:
        directory.mkdir(exist_ok=True);cv2.imwrite(str(directory/f.name),mask)
    cfg={'enabled':True}
    source={'config':cfg,'frames':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()}}
    (root/'source.json').write_text(json.dumps(source))
    metrics={'config':cfg,'input_frame_count':1,'layers':[{'directory':'layers/00','sha256':{f.name:hashlib.sha256((layer/f.name).read_bytes()).hexdigest()}}]}
    (root/'metrics.json').write_text(json.dumps(metrics))
    bm=tmp_path/'base-manifest';bm.mkdir();(bm/'detected.json').write_text('{}')
    write_manifest(bm,bm/'detected.json',[tmp_path/'base-layer'],[f])
    monkeypatch.setattr(sys,'argv',['validate','--result',str(root),'--frames',str(frames),'--baseline',str(tmp_path/'base-layer'),'--baseline-manifest',str(bm/'layers.json'),'--baseline-prefix',str(tmp_path),'--baseline-local',str(tmp_path)])
    if tamper:
        with pytest.raises(ValueError,match='严格并集'):main()
        assert not (root/'独立核验.json').exists()
    else:
        main();data=json.loads((root/'独立核验.json').read_text())
        assert data['merged_is_exact_union'] and data['baseline_removed_pixel_frames']==0
