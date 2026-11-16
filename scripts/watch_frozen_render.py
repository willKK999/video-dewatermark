"""同机监控已启动的补画；未启动、无结果退出、超时均报错，不无限等待。"""
import argparse,json,os,time
from pathlib import Path


def state(output):
    output=Path(output)
    if (output/'failure.json').exists():return 'failed'
    if (output/'render.json').exists():return 'rendered'
    pidfile=output/'worker-pid.json'
    if not pidfile.exists():raise RuntimeError('补画未启动：没有进程记录和结果文件')
    pid=json.loads(pidfile.read_text())['pid']
    if not isinstance(pid,int) or pid<=1:raise RuntimeError('补画进程记录无效')
    try:os.kill(pid,0)
    except ProcessLookupError:raise RuntimeError('补画进程已退出，但没有结果或失败文件') from None
    return 'running'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--maximum-seconds',type=float,default=10800);p.add_argument('--poll-seconds',type=float,default=10)
    args=p.parse_args()
    if args.maximum_seconds<=0 or not 0<args.poll_seconds<=60:raise ValueError('监控必须有正数总期限和最多60秒轮询')
    end=time.monotonic()+args.maximum_seconds
    while True:
        current=state(args.output)
        if current!='running':
            print('补画失败' if current=='failed' else '补画编码已完成，仍需验收',flush=True)
            return 1 if current=='failed' else 0
        if time.monotonic()>=end:raise RuntimeError('补画监控已到总期限，停止等待')
        time.sleep(min(args.poll_seconds,max(0,end-time.monotonic())))


if __name__=='__main__':
    try:raise SystemExit(main())
    except RuntimeError as exc:print(str(exc),flush=True);raise SystemExit(2)
