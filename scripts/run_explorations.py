"""Explicit GPU lanes; never select a GPU or stop someone else's process.

python scripts/run_explorations.py --gpus 0 1 [--after-gpu 2 --after-pid 12345]
Default: start risk/BC, then prox on whichever of those lanes finishes first.
Optional: prox uses the named third GPU only after the exact old PID exits.
"""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]


def process_identity(pid):
    # Linux /proc starttime prevents waiting forever on a recycled PID.
    try:
        stat=Path(f'/proc/{pid}/stat').read_text()
        fields=stat[stat.rfind(')')+2:].split()
        if fields[0]=='Z':return None
        return fields[19]
    except FileNotFoundError:
        return None


def run(args, popen=subprocess.Popen, sleep=time.sleep, identity=process_identity):
    experiments=['trusted_lr015_risk','trusted_lr015_bc','trusted_lr015_prox']
    old_identity=identity(args.after_pid) if args.after_pid is not None else None
    if args.after_pid is not None and old_identity is None:
        raise ValueError('Old PID is not running. Verify it, or launch prox manually when its GPU is free.')
    if args.dry_run:
        for experiment,gpu in zip(experiments[:2],args.gpus):
            print(f'NOW: bash scripts/run_experiment.sh {experiment} {gpu} {args.seed}')
        if args.after_pid is not None:
            print(f'AFTER PID {args.after_pid} exits: bash scripts/run_experiment.sh {experiments[2]} {args.after_gpu} {args.seed}')
        else:print(f'AFTER first of these two completes: {experiments[2]} on that freed GPU')
        return 0
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder=ROOT/'results'/'exploration_launches'/f'{stamp}_{os.getpid()}'
    folder.mkdir(parents=True,exist_ok=False)
    active=[];failed=[];pending=True
    def launch(name,gpu):
        stream=(folder/f'{name}.log').open('w',encoding='utf-8')
        try:
            child=popen(['bash',str(ROOT/'scripts/run_experiment.sh'),name,str(gpu),str(args.seed)],
                        cwd=str(ROOT),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        finally:stream.close()
        active.append((child,name,gpu))
        print(f'START {name} GPU={gpu} launcher_pid={child.pid} log={folder/name}.log',flush=True)
    launch(experiments[0],args.gpus[0]);launch(experiments[1],args.gpus[1])
    try:
        while active or pending:
            free=[]
            for child,name,gpu in list(active):
                status=child.poll()
                if status is not None:
                    active.remove((child,name,gpu));free.append(gpu)
                    print(f'END {name} status={status}',flush=True)
                    if status:failed.append(name)
            if pending:
                if args.after_pid is not None:
                    if identity(args.after_pid)!=old_identity:
                        launch(experiments[2],args.after_gpu);pending=False
                elif free:
                    launch(experiments[2],free[0]);pending=False
            if active or pending:sleep(10)
    except KeyboardInterrupt:
        print('Queue stopped. Already launched experiments remain running; no process was killed.',flush=True)
        return 130
    print(f'Completed; failed={failed}; logs={folder}',flush=True)
    return 1 if failed else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',nargs=2,type=int,required=True,help='Two currently free GPUs')
    parser.add_argument('--after-gpu',type=int)
    parser.add_argument('--after-pid',type=int,help='Existing training Python PID, not a launcher shell PID')
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if len(set(args.gpus))!=2 or min(args.gpus)<0 or args.seed<0:parser.error('Require two distinct nonnegative GPU IDs and seed')
    if (args.after_gpu is None)!=(args.after_pid is None):parser.error('Supply both --after-gpu and --after-pid')
    if args.after_gpu is not None and (args.after_gpu<0 or args.after_gpu in args.gpus or args.after_pid<=0):parser.error('Third GPU must be distinct and PID positive')
    try:return run(args)
    except (ValueError,OSError) as exc:parser.exit(2,f'{exc}\n')


if __name__=='__main__':raise SystemExit(main())
