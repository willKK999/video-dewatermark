"""列出或按预先记录的清单清理可再生成PNG中间帧，保留所有掩膜及成片。"""
import argparse
import json
from pathlib import Path

ROOT=Path('/workspace/dewatermark-data/watermark-test-878ffd51')
VERSIONS=('v1','v2','v3','v4','v5','v6')


def eligible(path):
    relative=path.relative_to(ROOT)
    dirs=relative.parts[1:-1]
    return (relative.parts[0] in VERSIONS and path.suffix=='.png' and path.stem.isdigit()
            and any(x in dirs for x in ('composite','encoded','detail'))
            and not any('mask' in x or x=='evidence' for x in dirs)
            and path.is_file() and not path.is_symlink())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--apply',action='store_true')
    args=p.parse_args();plan=Path(args.plan)
    if not args.apply:
        rows=[]
        for version in VERSIONS:
            for path in (ROOT/version).rglob('*.png'):
                if eligible(path):
                    stat=path.stat()
                    rows.append(dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
        plan.write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        print('清理计划：',len(rows),'个PNG，逻辑字节',sum(r['size'] for r in rows))
        return
    rows=json.loads(plan.read_text())
    for row in rows:
        path=Path(row['path']);stat=path.stat()
        if not eligible(path) or stat.st_size!=row['size'] or stat.st_mtime_ns!=row['mtime_ns']:
            raise ValueError('候选中间帧已变动，停止清理')
    for row in rows:Path(row['path']).unlink()
    print('已按固定清单删除可再生成PNG中间帧：',len(rows))


if __name__=='__main__':main()
