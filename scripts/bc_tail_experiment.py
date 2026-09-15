#!/usr/bin/env python3
"""Inspect a trusted local BC checkpoint or launch an isolated tail fork."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['inspect','control','breliability','aguard','featurehalf'])
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stop-after-round', type=int, default=0,
                        help='For control only: stop at 226-300, e.g. 230 for a five-round replay check')
    opt = parser.parse_args()
    if opt.stop_after_round and (opt.mode != 'control' or not 226 <= opt.stop_after_round <= 300):
        parser.error('--stop-after-round is only for control, between 226 and 300')
    import torch
    from bc_tail import validate_fork
    source = opt.checkpoint.resolve(strict=True)
    # Only load user-owned training checkpoints (torch pickle format).
    ckpt = torch.load(source, map_location='cpu', weights_only=False)
    print(json.dumps(dict(path=str(source), round=ckpt.get('round'), method_rev=ckpt.get('method_rev'),
        geometry_controls=ckpt.get('geometry_controls'), lr_controls=ckpt.get('lr_controls'),
        client_states=len(ckpt.get('client_states', {}))), ensure_ascii=False, indent=2))
    cfg_file = source.parent / 'config.json'
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text(encoding='utf8'))
    else:
        cfg = ckpt.get('run_identity', {})
    keys = ('seed_model','seed_partition','seed_sample','dataset','alpha','num_clients',
            'num_online_clients','num_labeled_per_class','mu','local_epochs','batch_labeled','num_workers')
    missing = [k for k in keys if k not in cfg]
    if missing:
        parser.error('Missing original run metadata: ' + ', '.join(missing))
    run_id = f'rev13_bc_tail_{opt.mode}_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    destination = ROOT / 'results/CIFAR10/runs' / (run_id + '_a0.1')
    try:
        validate_fork(ckpt, ckpt.get('geometry_controls', {}), ckpt.get('lr_controls', {}),
                      str(source), str(destination), {k:cfg[k] for k in keys})
    except ValueError as exc:
        parser.error(str(exc))
    print('Complete BC round225 state: structurally eligible for a fork (GPU replay still required).')
    if opt.mode == 'inspect':
        return
    if cfg['dataset'] != 'CIFAR10' or cfg['alpha'] != .1:
        parser.error('These fixed recipes target CIFAR10 alpha=0.1')
    config = 'experiment_trusted_lr015_bc.yaml' if opt.mode == 'control' else f'experiment_bc_tail_{opt.mode}.yaml'
    command = ['bash', str(ROOT/'scripts/train.sh'), '--config', str(ROOT/'configs'/config),
               '--gpu_id', str(opt.gpu), '--run_id', run_id, '--resume', str(source), '--bc_fork', '1',
               '--seed', str(cfg['seed_model']), '--partition_seed', str(cfg['seed_partition']),
               '--sample_seed', str(cfg['seed_sample']), '--num_workers', str(cfg['num_workers'])]
    if opt.stop_after_round:
        command += ['--stop_after_round', str(opt.stop_after_round)]
    print(shlex.join(command))
    if not opt.dry_run:
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHON=sys.executable), check=True)


if __name__ == '__main__':
    main()
