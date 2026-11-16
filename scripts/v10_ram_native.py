"""低磁盘诊断用启动器：原生库以内存文件描述符加载，其余依赖复用。"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--view',type=Path,required=True)
    p.add_argument('--cuda-libs',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args(); args.view.mkdir(parents=True,exist_ok=False)
    handles=[]
    for source in args.package.rglob('*'):
        relative=source.relative_to(args.package); dest=args.view/relative
        if source.is_dir(): dest.mkdir(exist_ok=True);continue
        if '.so' in source.name and not source.is_symlink():
            fd=os.memfd_create(source.name,0)
            with source.open('rb') as inp, os.fdopen(os.dup(fd),'wb') as out: shutil.copyfileobj(inp,out,1<<20)
            handles.append(fd);dest.symlink_to(f'/proc/{os.getpid()}/fd/{fd}')
        else: dest.symlink_to(source.resolve())
    env=os.environ.copy();env['PYTHONPATH']=str(args.view)+os.pathsep+env.get('PYTHONPATH','')
    libraries={str(args.view/'paddle/libs'),str(args.view/'paddle/base')}
    libraries.update(str(p.parent) for p in args.cuda_libs.rglob('*.so*'))
    env['LD_LIBRARY_PATH']=os.pathsep.join(sorted(libraries))+os.pathsep+env.get('LD_LIBRARY_PATH','')
    cudnn=args.cuda_libs/'cudnn/lib/libcudnn.so.9'
    if cudnn.exists():
        env['LD_PRELOAD']=str(cudnn)+os.pathsep+env.get('LD_PRELOAD','')
        env['FLAGS_cudnn_dir']=str(cudnn.parent)
    print(f'内存原生库就绪：{len(handles)}个文件，未占用结果磁盘',flush=True)
    command=args.command
    if command and command[0]=='--': command=command[1:]
    try: sys.exit(subprocess.call(command,env=env))
    finally:
        for fd in handles:os.close(fd)


if __name__=='__main__':main()
