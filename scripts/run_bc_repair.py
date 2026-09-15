"""Run the three full BC A-repair experiments on explicitly supplied free GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ('recover', 'correct', 'joint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--experiments', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error('Supply unique nonnegative GPU ids')
    if len(set(args.experiments)) != len(args.experiments):
        parser.error('Duplicate experiments are not allowed')
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + f'_{os.getpid()}'
    folder = ROOT / 'results' / 'bc_repair_launches' / stamp

    def lane(index, gpu):
        for mode in args.experiments[index::len(args.gpus)]:
            run_id = f'rev13_bc_repair_{mode}_{stamp}'
            command = [sys.executable, '-u', str(ROOT / 'ppfpsl.py'), '--config',
                       str(ROOT / 'configs' / f'experiment_bc_repair_{mode}.yaml'),
                       '--gpu_id', str(gpu), '--run_id', run_id]
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                with (folder / f'{mode}.log').open('w', encoding='utf8') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    if not args.dry_run:
        folder.mkdir(parents=True, exist_ok=False)
        print(f'Launch logs: {folder}', flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(lane, i, gpu) for i, gpu in enumerate(args.gpus)]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
