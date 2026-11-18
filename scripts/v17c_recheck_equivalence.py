"""对比两个逐层ROI运行目录（旧/新）的结果等价性与CPU阶段耗时。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

EXCLUDED_TIMING_FIELDS = (
    '总耗时秒',
    '耗时秒',
    '超时秒',
    '复检耗时秒',
    '历史补画耗时秒',
    'CPU耗时秒',
)


def _resolve_run_dir(path: Path) -> Path:
    """支持传入外层输出目录或直接传入 run/ 目录。"""
    if (path / 'roi-report.json').exists():
        return path
    if (path / 'run' / 'roi-report.json').exists():
        return path / 'run'
    raise FileNotFoundError(f'未找到 roi-report.json：{path}')


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


PATH_SUFFIX_FIELDS = {'暂存路径', '基线路径'}


def _normalize_path_suffix(value: str, run_dir: Path) -> str:
    p = Path(value)
    # 兼容跨机器（如 Vast /workspace/dewatermark-data/... 与本地路径）比较相对后缀
    if p.name in {'staged-delta'} and len(p.parts) >= 2:
        return f'{p.parts[-2]}/{p.parts[-1]}'
    if p.name.startswith('roi-') and p.name.endswith('-baseline'):
        return p.name
    try:
        return str(p.relative_to(run_dir))
    except ValueError:
        return p.name


def _compare_values(old: Any, new: Any, path: str, diffs: list[dict], excluded: set[str], old_run: Path, new_run: Path):
    """递归比对 JSON 结构，排除耗时类字段，规范化目录前缀路径字段；数值与容器要求严格相等。"""
    if type(old) is not type(new):
        # JSON 中 int/float 保持类型严格检查；但允许 None 与 None
        diffs.append(dict(路径=path, 类型='类型不一致', 旧值=repr(old)[:200], 新值=repr(new)[:200]))
        return
    if isinstance(old, dict):
        old_keys = {k for k in old.keys() if k not in excluded}
        new_keys = {k for k in new.keys() if k not in excluded}
        for k in sorted(old_keys - new_keys):
            diffs.append(dict(路径=f'{path}.{k}', 类型='新结果缺少字段', 旧值=repr(old[k])[:200], 新值=None))
        for k in sorted(new_keys - old_keys):
            diffs.append(dict(路径=f'{path}.{k}', 类型='新结果多出字段', 旧值=None, 新值=repr(new[k])[:200]))
        for k in sorted(old_keys & new_keys):
            ov, nv = old[k], new[k]
            if k in PATH_SUFFIX_FIELDS and isinstance(ov, str) and isinstance(nv, str):
                ov = _normalize_path_suffix(ov, old_run)
                nv = _normalize_path_suffix(nv, new_run)
            _compare_values(ov, nv, f'{path}.{k}', diffs, excluded, old_run, new_run)
        return
    if isinstance(old, list):
        if len(old) != len(new):
            diffs.append(dict(路径=path, 类型='列表长度不一致', 旧值=len(old), 新值=len(new)))
            return
        for idx, (ov, nv) in enumerate(zip(old, new)):
            _compare_values(ov, nv, f'{path}[{idx}]', diffs, excluded, old_run, new_run)
        return
    if old != new:
        diffs.append(dict(路径=path, 类型='值不一致', 旧值=repr(old)[:200], 新值=repr(new)[:200]))


def _parse_progress_cpu_times(run_dir: Path) -> dict[tuple[int, int], float]:
    """从 progress.log 解析每个 (任务, 尝试) 的 CPU 阶段耗时（秒）。"""
    log_path = run_dir.parent / 'progress.log'
    if not log_path.exists():
        return {}
    cpu_start: dict[tuple[int, int], datetime] = {}
    cpu_dur: dict[tuple[int, int], float] = {}
    re_start = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) 逐层ROI 任务=(\d+) 尝试=(\d+) 模型已结束，开始CPU参考回填与残留复检')
    re_end = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) 逐层ROI 任务=(\d+) 尝试=(\d+) 参数=')
    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        m1 = re_start.match(line)
        if m1:
            dt = datetime.strptime(m1[1], '%Y-%m-%d %H:%M:%S')
            cpu_start[(int(m1[2]), int(m1[3]))] = dt
            continue
        m2 = re_end.match(line)
        if m2:
            key = (int(m2[2]), int(m2[3]))
            if key in cpu_start:
                dt = datetime.strptime(m2[1], '%Y-%m-%d %H:%M:%S')
                cpu_dur[key] = max(0.0, (dt - cpu_start[key]).total_seconds())
    return cpu_dur


def _compare_npz_dirs(old_run: Path, new_run: Path) -> tuple[dict, list[dict]]:
    """逐位比较所有尝试目录中的 staged-delta/*.npz 数组。"""
    diffs: list[dict] = []
    old_dirs = {p.name: p / 'staged-delta' for p in sorted(old_run.glob('roi-*-attempt-*')) if (p / 'staged-delta').is_dir()}
    new_dirs = {p.name: p / 'staged-delta' for p in sorted(new_run.glob('roi-*-attempt-*')) if (p / 'staged-delta').is_dir()}

    for name in sorted(set(old_dirs) - set(new_dirs)):
        diffs.append(dict(路径=f'staged-delta/{name}', 类型='新结果缺少staged-delta'))
    for name in sorted(set(new_dirs) - set(old_dirs)):
        diffs.append(dict(路径=f'staged-delta/{name}', 类型='新结果多出staged-delta'))

    per_attempt = {}
    for name in sorted(set(old_dirs) & set(new_dirs)):
        od, nd = old_dirs[name], new_dirs[name]
        ofiles = {p.name: p for p in sorted(od.glob('*.npz'))}
        nfiles = {p.name: p for p in sorted(nd.glob('*.npz'))}
        ok = True
        if set(ofiles) != set(nfiles):
            ok = False
            diffs.append(
                dict(
                    路径=f'staged-delta/{name}',
                    类型='npz文件列表不一致',
                    缺少=sorted(set(ofiles) - set(nfiles)),
                    多出=sorted(set(nfiles) - set(ofiles)),
                )
            )
        checked = 0
        for fname in sorted(set(ofiles) & set(nfiles)):
            checked += 1
            with np.load(ofiles[fname], allow_pickle=False) as oa, np.load(nfiles[fname], allow_pickle=False) as na:
                if set(oa.files) != set(na.files):
                    ok = False
                    diffs.append(dict(路径=f'staged-delta/{name}/{fname}', 类型='npz键不一致', 旧值=oa.files, 新值=na.files))
                    continue
                for k in oa.files:
                    a1, a2 = oa[k], na[k]
                    if a1.dtype != a2.dtype or a1.shape != a2.shape or not np.array_equal(a1, a2):
                        ok = False
                        diffs.append(
                            dict(
                                路径=f'staged-delta/{name}/{fname}[{k}]',
                                类型='npz数组逐位不一致',
                                旧形状=list(a1.shape),
                                新形状=list(a2.shape),
                                旧类型=str(a1.dtype),
                                新类型=str(a2.dtype),
                            )
                        )
        per_attempt[name] = dict(一致=ok, npz文件数=checked)
    return per_attempt, diffs


def _compare_png_tree(old_run: Path, new_run: Path, subdir: str, active_names: set[str] | None = None) -> tuple[dict, list[dict]]:
    """通过 sha256 逐文件校验 roi-layer-00 或 roi-coverage 的逐位一致性。"""
    diffs: list[dict] = []
    od, nd = old_run / subdir, new_run / subdir
    if not od.is_dir() or not nd.is_dir():
        diffs.append(dict(路径=subdir, 类型='目录缺失', 旧存在=od.is_dir(), 新存在=nd.is_dir()))
        return dict(一致=False, 文件数=0, 任务覆盖帧数=0), diffs
    ofiles = {p.name: p for p in sorted(od.glob('*.png'))}
    nfiles = {p.name: p for p in sorted(nd.glob('*.png'))}
    if set(ofiles) != set(nfiles):
        diffs.append(dict(路径=subdir, 类型='PNG文件列表不一致', 缺少=sorted(set(ofiles) - set(nfiles))[:20], 多出=sorted(set(nfiles) - set(ofiles))[:20]))
    common = sorted(set(ofiles) & set(nfiles))
    mismatch_files = []
    active_checked = 0
    for name in common:
        if active_names is not None and name in active_names:
            active_checked += 1
        if _sha256(ofiles[name]) != _sha256(nfiles[name]):
            mismatch_files.append(name)
            if len(diffs) < 100:
                diffs.append(dict(路径=f'{subdir}/{name}', 类型='PNG sha256 不一致'))
    return dict(一致=(len(diffs) == 0), 文件数=len(common), 任务覆盖帧数=active_checked, 不一致文件数=len(mismatch_files)), diffs


def _compare_seam_pngs(old_run: Path, new_run: Path) -> list[dict]:
    """若存在 seam-difference / seam-shared 目录，同样核对 sha256。"""
    diffs: list[dict] = []
    for sub in ('seam-difference', 'seam-shared'):
        old_dirs = {p.name: p / sub for p in sorted(old_run.glob('roi-*-attempt-*')) if (p / sub).is_dir()}
        new_dirs = {p.name: p / sub for p in sorted(new_run.glob('roi-*-attempt-*')) if (p / sub).is_dir()}
        for name in sorted(set(old_dirs) & set(new_dirs)):
            ofiles = {p.name: p for p in sorted(old_dirs[name].glob('*.png'))}
            nfiles = {p.name: p for p in sorted(new_dirs[name].glob('*.png'))}
            if set(ofiles) != set(nfiles):
                diffs.append(dict(路径=f'{name}/{sub}', 类型='文件列表不一致'))
                continue
            for fn in sorted(ofiles):
                if _sha256(ofiles[fn]) != _sha256(nfiles[fn]):
                    diffs.append(dict(路径=f'{name}/{sub}/{fn}', 类型='PNG sha256 不一致'))
                    if len(diffs) >= 100:
                        return diffs
    return diffs


def compare_runs(old_path: Path, new_path: Path, max_tasks: int = 0) -> dict:
    old_run = _resolve_run_dir(old_path)
    new_run = _resolve_run_dir(new_path)
    old_rep = json.loads((old_run / 'roi-report.json').read_text(encoding='utf-8'))
    new_rep = json.loads((new_run / 'roi-report.json').read_text(encoding='utf-8'))
    if max_tasks > 0:
        old_rep['任务'] = [t for t in old_rep.get('任务', []) if int(t['任务']) <= max_tasks]
        new_rep['任务'] = [t for t in new_rep.get('任务', []) if int(t['任务']) <= max_tasks]
        old_rep['未过块'] = [x for x in old_rep.get('未过块', []) if int(x) <= max_tasks]
        new_rep['未过块'] = [x for x in new_rep.get('未过块', []) if int(x) <= max_tasks]
        old_rep['不可填块'] = [x for x in old_rep.get('不可填块', []) if int(x) <= max_tasks]
        new_rep['不可填块'] = [x for x in new_rep.get('不可填块', []) if int(x) <= max_tasks]
    excluded = set(EXCLUDED_TIMING_FIELDS)

    report_diffs: list[dict] = []
    _compare_values(old_rep, new_rep, 'roi-report', report_diffs, excluded, old_run, new_run)

    npz_summary, npz_diffs = _compare_npz_dirs(old_run, new_run)
    seam_diffs = _compare_seam_pngs(old_run, new_run)
    if max_tasks > 0:
        allowed = {f'roi-{tid:04d}-attempt-' for tid in range(1, max_tasks + 1)}
        npz_diffs = [d for d in npz_diffs if any(p in d['路径'] for p in allowed)]
        seam_diffs = [d for d in seam_diffs if any(p in d['路径'] for p in allowed)]
        npz_summary = {k: v for k, v in npz_summary.items() if any(k.startswith(p) for p in allowed)}

    # 收集任务覆盖帧文件名（例如 000000.png ...）
    active_names: set[str] = set()
    for t in old_rep.get('任务', []):
        for fidx in range(int(t['开始帧']), int(t['结束帧'])):
            active_names.add(f'{fidx:06d}.png')

    layer_summary, layer_diffs = _compare_png_tree(old_run, new_run, 'roi-layer-00', active_names)
    cov_summary, cov_diffs = _compare_png_tree(old_run, new_run, 'roi-coverage', active_names)

    audit_diffs: list[dict] = []
    oa, na = old_run / '层输入自检.json', new_run / '层输入自检.json'
    if oa.exists() and na.exists():
        o_aud = json.loads(oa.read_text(encoding='utf-8'))
        n_aud = json.loads(na.read_text(encoding='utf-8'))
        if max_tasks > 0:
            o_aud['尝试'] = [x for x in o_aud.get('尝试', []) if int(x['任务']) <= max_tasks]
            n_aud['尝试'] = [x for x in n_aud.get('尝试', []) if int(x['任务']) <= max_tasks]
            o_aud['源水印未排除像素'] = sum(f.get('源水印未排除像素', 0) for a in o_aud['尝试'] for f in a['帧'])
            n_aud['源水印未排除像素'] = sum(f.get('源水印未排除像素', 0) for a in n_aud['尝试'] for f in a['帧'])
            o_aud['全部已检查尝试通过'] = all(a['通过'] for a in o_aud['尝试'])
            n_aud['全部已检查尝试通过'] = all(a['通过'] for a in n_aud['尝试'])
        _compare_values(o_aud, n_aud, '层输入自检', audit_diffs, excluded, old_run, new_run)

    old_cpu_log = _parse_progress_cpu_times(old_run)
    new_cpu_log = _parse_progress_cpu_times(new_run)

    task_rows = []
    old_tasks = old_rep.get('任务', [])
    new_tasks = new_rep.get('任务', [])
    for idx in range(max(len(old_tasks), len(new_tasks))):
        ot = old_tasks[idx] if idx < len(old_tasks) else None
        nt = new_tasks[idx] if idx < len(new_tasks) else None
        tid = ot['任务'] if ot else nt['任务']
        t_diffs: list[dict] = []
        if ot is not None and nt is not None:
            _compare_values(ot, nt, f'任务[{tid}]', t_diffs, excluded, old_run, new_run)
        else:
            t_diffs.append(dict(路径=f'任务[{tid}]', 类型='任务缺失'))

        attempts_old = ot.get('尝试', []) if ot else []
        attempts_new = nt.get('尝试', []) if nt else []
        old_attempt_dt = sum(float(a.get('耗时秒') or 0.0) for a in attempts_old)
        new_attempt_dt = sum(float(a.get('耗时秒') or 0.0) for a in attempts_new)
        old_cpu_dt = sum(old_cpu_log.get((tid, int(a['尝试'])), 0.0) for a in attempts_old)
        new_cpu_dt = sum(new_cpu_log.get((tid, int(a['尝试'])), 0.0) for a in attempts_new)
        # 若有精确 CPU耗时秒 字段则优先使用
        if attempts_new and any('CPU耗时秒' in a for a in attempts_new):
            new_cpu_exact = sum(float(a.get('CPU耗时秒') or 0.0) for a in attempts_new)
        else:
            new_cpu_exact = new_cpu_dt
        if attempts_old and any('CPU耗时秒' in a for a in attempts_old):
            old_cpu_exact = sum(float(a.get('CPU耗时秒') or 0.0) for a in attempts_old)
        else:
            old_cpu_exact = old_cpu_dt

        eff_frames = sum(int(a.get('残留汇总', {}).get('有效施工帧数', 0)) for a in attempts_old)
        if not eff_frames and ot is not None:
            eff_frames = int(ot['结束帧']) - int(ot['开始帧'])
        old_per_frame = (old_cpu_exact / eff_frames) if (eff_frames and old_cpu_exact > 0) else (old_attempt_dt / eff_frames if eff_frames else None)
        new_per_frame = (new_cpu_exact / eff_frames) if (eff_frames and new_cpu_exact > 0) else (new_attempt_dt / eff_frames if eff_frames else None)

        task_rows.append(
            dict(
                任务=tid,
                施工区间=[ot['开始帧'], ot['结束帧']] if ot else [nt['开始帧'], nt['结束帧']],
                尝试数=len(attempts_old),
                有效施工帧数=eff_frames,
                report一致=len(t_diffs) == 0,
                旧任务耗时秒=round(float(ot.get('耗时秒', 0.0)), 3) if ot else None,
                新任务耗时秒=round(float(nt.get('耗时秒', 0.0)), 3) if nt else None,
                旧CPU阶段耗时秒=round(old_cpu_exact, 3) if old_cpu_exact > 0 else None,
                新CPU阶段耗时秒=round(new_cpu_exact, 3) if new_cpu_exact > 0 else None,
                旧单帧耗时秒=round(old_per_frame, 4) if old_per_frame is not None else None,
                新单帧耗时秒=round(new_per_frame, 4) if new_per_frame is not None else None,
            )
        )

    all_diffs = report_diffs + audit_diffs + npz_diffs + seam_diffs + layer_diffs + cov_diffs
    return dict(
        一致=len(all_diffs) == 0,
        旧目录=str(old_run),
        新目录=str(new_run),
        排除的耗时字段=list(EXCLUDED_TIMING_FIELDS),
        对比摘要=dict(
            任务数_旧=len(old_tasks),
            任务数_新=len(new_tasks),
            roi_report不一致数=len(report_diffs),
            层输入自检不一致数=len(audit_diffs),
            staged_delta不一致数=len(npz_diffs),
            seam_png不一致数=len(seam_diffs),
            roi_layer_00=layer_summary,
            roi_coverage=cov_summary,
            staged_delta明细=npz_summary,
        ),
        各任务对比=task_rows,
        不一致明细=all_diffs[:200],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True, help='旧运行目录或旧 run 目录')
    p.add_argument('--new', type=Path, required=True, help='新运行目录或新 run 目录')
    p.add_argument('--max-tasks', type=int, default=0, help='仅比对前 N 个任务（0 表示全部）')
    p.add_argument('--output', type=Path, default=None, help='输出 JSON 路径')
    args = p.parse_args()
    res = compare_runs(args.old, args.new, max_tasks=args.max_tasks)
    text = json.dumps(res, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text)
    if not res['一致']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
