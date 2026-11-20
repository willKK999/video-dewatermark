"""同源、同帧范围的已验证掩膜逐像素等价检查；任何差异均阻止补画。"""
import json
from pathlib import Path
import cv2
import numpy as np
from .process import PipelineError
from .vision import sha256
from .hls_layers import VerifiedLayers


def compare(work, reference, output):
    work, reference, output = map(Path, (work, reference, output))
    binding = json.loads((work / 'frames-binding.json').read_text())
    if binding != json.loads((reference / 'frames-binding.json').read_text()):
        raise PipelineError('SOURCE_CHANGED', '掩膜比较的源片、范围或逐帧摘要不一致')
    folders = [reference, work]
    summaries = [json.loads((p / 'model-mask-summary.json').read_text())['逐帧'] for p in folders]
    names = sorted(binding['sha256'])
    if any([r['mask_key'] for r in rows] != ['masks/' + n for n in names] for rows in summaries):
        raise PipelineError('MASK_INVALID', '掩膜比较清单帧数或顺序不一致')
    validators = [VerifiedLayers(p / 'layers.json', p / 'detections.json') for p in folders]
    records = []
    for i, name in enumerate(names):
        images = []
        for folder, rows, validator in zip(folders, summaries, validators):
            path = folder / rows[i]['mask_key']
            if sha256(path) != rows[i]['sha256']:
                raise PipelineError('MASK_INVALID', '掩膜比较输入摘要不一致')
            image = cv2.imread(str(path), 0)
            if image is None:
                raise PipelineError('MASK_INVALID', '掩膜比较输入不可读')
            validator.validate(i, image)
            images.append(image > 0)
        old, new = images
        if old.shape != new.shape:
            raise PipelineError('MASK_INVALID', '掩膜比较几何不一致')
        total = int(np.count_nonzero(old | new))
        intersection = int(np.count_nonzero(old & new))
        a, b = int(np.count_nonzero(old)), int(np.count_nonzero(new))
        records.append(dict(帧=i, v1像素=a, v4像素=b, v1覆盖率=a/old.size,
                            v4覆盖率=b/new.size, IoU=intersection/total if total else 1.,
                            缺失像素=a-intersection, 新增像素=b-intersection))
    report = dict(源摘要=binding['source']['source_sha256'], 逐帧=records,
                  缺失像素总数=sum(r['缺失像素'] for r in records),
                  新增像素总数=sum(r['新增像素'] for r in records),
                  最低IoU=min((r['IoU'] for r in records), default=1.),
                  通过=bool(records) and all(r['缺失像素']==r['新增像素']==0 for r in records))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['通过']:
        raise PipelineError('MASK_NOT_EQUIVALENT', 'v7掩膜与v1不等价，禁止进入补画', details=report)
    return report
