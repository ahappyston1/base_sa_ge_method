"""Run three independent experiments, at most one process per specified GPU.

Uses the current Python environment. Explicitly supply GPUs that are free;
this launcher does not inspect or interrupt other training processes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
MODES = ('evidence', 'labelhead', 'separation', 'labelhead_constant', 'labelhead_step', 'labelhead_step_tail', 'labelhead_guard')


def command(mode, gpu, run_id):
    return [sys.executable, '-u', str(ROOT/'ppfpsl.py'), '--config',
            str(ROOT/'configs'/f'experiment_bc_targets_{mode}.yaml'),
            '--dataset', 'CIFAR10', '--alpha', '0.1', '--gpu_id', str(gpu), '--run_id', run_id]


def run_queue(gpus, jobs, launch):
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    failed = threading.Event()
    results = []
    lock = threading.Lock()

    def worker(gpu):
        while not failed.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                code = launch(job, gpu)
            except Exception as exc:
                print(f'[queue] {job} failed: {exc}', flush=True)
                code = 1
            with lock:
                results.append((job, gpu, code))
            if code:
                failed.set()
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    return results, failed.is_set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--experiments', choices=MODES, nargs='+', default=['evidence','labelhead','separation'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if any(g < 0 for g in args.gpus) or len(set(args.gpus)) != len(args.gpus):
        parser.error('Provide unique nonnegative GPU indices')
    if len(set(args.experiments)) != len(args.experiments):
        parser.error('Experiments must be unique')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')+f'_{os.getpid()}'
    run_ids = {name: f'bc_targets_{name}_{stamp}' for name in args.experiments}
    if args.dry_run:
        for i, name in enumerate(args.experiments):
            print(shlex.join(command(name, args.gpus[i % len(args.gpus)], run_ids[name])))
        print(f'At most {len(args.gpus)} simultaneous jobs; queued jobs use the next released GPU.')
        return 0
    import torch
    if not torch.cuda.is_available() or any(g >= torch.cuda.device_count() for g in args.gpus):
        parser.error('Requested GPU is unavailable in the current Python environment')
    logdir = ROOT/'results'/'target_experiment_launches'
    logdir.mkdir(parents=True, exist_ok=True)

    def launch(name, gpu):
        log = logdir/(run_ids[name]+'.log')
        print(f'[start] {name} GPU={gpu} log={log}', flush=True)
        with log.open('x', encoding='utf-8') as handle:
            code = subprocess.run(command(name, gpu, run_ids[name]), cwd=ROOT,
                                  stdout=handle, stderr=subprocess.STDOUT).returncode
        print(f'[done] {name} GPU={gpu} exit={code}', flush=True)
        return code

    results, failed = run_queue(args.gpus, args.experiments, launch)
    for name, gpu, code in results:
        print(f'{name}: GPU={gpu}, exit={code}', flush=True)
    if failed:
        print('A job failed. Remaining unstarted jobs were not launched; inspect its log.', flush=True)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
