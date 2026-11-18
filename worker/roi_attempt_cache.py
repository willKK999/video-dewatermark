"""无损保留每次未过ROI的暂存结果，避免重复存储整幅RGB背景。"""
from pathlib import Path
import os,shutil
import cv2
import numpy as np


def _compact_one(item):
    reference_path, staged_path, delta_path, active_packed, total_px = item
    old = cv2.imread(str(reference_path))
    image = cv2.imread(str(staged_path))
    active = np.unpackbits(np.frombuffer(active_packed, dtype=np.uint8), bitorder='little')[:total_px]
    indices = np.flatnonzero(active).astype(np.int32)
    pixels = image.reshape(-1, 3)[indices]
    reconstructed = old.copy()
    reconstructed.reshape(-1, 3)[indices] = pixels
    if not np.array_equal(reconstructed, image):
        raise ValueError('暂存结果不能由基线和施工像素无损重建')
    np.savez_compressed(delta_path, indices=indices, pixels=pixels)


def compact(task,baseline,dest,files,masks,ownership,start,end,pool=None):
    staged=task/'staged'
    if not all((staged/files[j].name).exists() for j in range(start,end)):return False
    baseline.mkdir(exist_ok=True);delta=task/'staged-delta';delta.mkdir(exist_ok=True)
    if pool is not None and end-start>1:
        items=[]
        for j in range(start,end):
            name=files[j].name;reference=baseline/name
            if not reference.exists():os.link(dest/name,reference)
            active=((masks[j]&ownership).ravel()>0)
            items.append((str(reference),str(staged/name),str(delta/(Path(name).stem+'.npz')),np.packbits(active,bitorder='little').tobytes(),active.size))
        for _ in pool.imap(_compact_one,items):
            pass
        shutil.rmtree(staged)
        return True
    for j in range(start,end):
        name=files[j].name;reference=baseline/name
        if not reference.exists():os.link(dest/name,reference)
        old=cv2.imread(str(reference));image=cv2.imread(str(staged/name));indices=np.flatnonzero((masks[j]&ownership).ravel()).astype(np.int32)
        pixels=image.reshape(-1,3)[indices];reconstructed=old.copy();reconstructed.reshape(-1,3)[indices]=pixels
        if not np.array_equal(reconstructed,image):raise ValueError('暂存结果不能由基线和施工像素无损重建')
        np.savez_compressed(delta/(Path(name).stem+'.npz'),indices=indices,pixels=pixels)
    shutil.rmtree(staged)
    return True


def restore(task,baseline,name):
    image=cv2.imread(str(baseline/name))
    with np.load(task/'staged-delta'/(Path(name).stem+'.npz'),allow_pickle=False) as data:image.reshape(-1,3)[data['indices']]=data['pixels']
    return image
