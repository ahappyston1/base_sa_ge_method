#!/usr/bin/env python3
"""Strict BC prefix/replay verification followed by explicitly assigned GPU lanes."""
import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / 'results/CIFAR10/runs/rev13_trusted_lr015_bc_s7_20260914_181549_541595_a0.1'


def rows(path):
    result = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            try:
                number = Decimal(row['round'])
                r = int(number)
                if number != r or r < 1 or r in result:
                    raise ValueError('duplicate or invalid round')
            except (KeyError, InvalidOperation, ValueError) as exc:
                raise ValueError(f'{path}: malformed round: {row.get("round")}') from exc
            result[r] = row
    if not result or sorted(result) != list(range(min(result), max(result)+1)):
        raise ValueError(f'{path}: empty data or missing rounds')
    return result


def compare(reference, candidate, start, end, fields):
    """Exact numeric CSV equality, never a tolerance or rounded comparison."""
    differences = []
    for r in range(start, end+1):
        if r not in reference or r not in candidate:
            differences.append(dict(round=r, field='round', reason='missing'))
            continue
        for field in fields:
            old, new = reference[r].get(field), candidate[r].get(field)
            try:
                a, b = Decimal(old), Decimal(new)
                equal = a.is_finite() and b.is_finite() and a == b
            except (InvalidOperation, TypeError):
                equal = False
            if not equal:
                differences.append(dict(round=r, field=field, reference=old, candidate=new))
    return dict(passed=not differences, start=start, end=end,
                mismatch_count=len(differences), first_differences=differences[:20])


def write_json(path, data):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(temporary, path)


def run_jobs(jobs, gpus, folder, popen=subprocess.Popen, sleep=time.sleep):
    """At most one job per named lane. A failure cancels pending launches only."""
    waiting = list(jobs); active = []; failed = []
    free = list(gpus)
    while waiting or active:
        # Poll all children before reusing a lane: a simultaneous failure must
        # prevent a new launch even when another child completed successfully.
        for item in list(active):
            child, name, gpu = item
            status = child.poll()
            if status is not None:
                active.remove(item); free.append(gpu)
                print(f'END {name}: exit={status}', flush=True)
                if status:
                    failed.append(name)
        if failed:
            waiting.clear()
        while waiting and free:
            name, build = waiting.pop(0); gpu = free.pop(0)
            command = build(gpu)
            with (folder / f'{name}.log').open('w', encoding='utf8') as stream:
                child = popen(command, cwd=ROOT, env=dict(os.environ, PYTHON=sys.executable),
                              stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            active.append((child, name, gpu))
            print(f'START {name}: GPU={gpu} PID={child.pid} log={folder / (name+".log")}', flush=True)
        if active:
            sleep(5)
    if failed:
        raise RuntimeError('Training failed; pending jobs were not launched: '+', '.join(failed))


def train_command(config, run_id, cfg, gpu, checkpoint=None, stop=0):
    command = ['bash', str(ROOT/'scripts/train.sh'), '--config', str(ROOT/'configs'/config),
               '--dataset', 'CIFAR10', '--alpha', '0.1', '--gpu_id', str(gpu), '--run_id', run_id,
               '--seed', str(cfg['seed_model']), '--partition_seed', str(cfg['seed_partition']),
               '--sample_seed', str(cfg['seed_sample']), '--num_workers', str(cfg['num_workers'])]
    if checkpoint:
        command += ['--resume', str(checkpoint), '--bc_fork', '1']
    if stop:
        command += ['--stop_after_round', str(stop)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', type=int, required=True,
                        help='One to three GPUs reserved by you; script never detects or evicts other jobs')
    parser.add_argument('--reference-run', type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument('--prefix-checkpoint', type=Path,
                        help='Optional existing original BC round225 checkpoint; still checks its CSV prefix')
    opt = parser.parse_args()
    if not 1 <= len(opt.gpus) <= 3 or len(set(opt.gpus)) != len(opt.gpus) or min(opt.gpus) < 0:
        parser.error('Specify 1-3 distinct nonnegative GPU IDs')
    reference = opt.reference_run.resolve(strict=True)
    cfg = json.loads((reference/'config.json').read_text(encoding='utf8'))
    required = ('seed_model','seed_partition','seed_sample','num_workers','geometry_controls','lr_controls')
    if any(k not in cfg for k in required) or cfg.get('dataset') != 'CIFAR10' or cfg.get('alpha') != .1:
        parser.error('Reference must be the completed CIFAR10 alpha=0.1 BC run with full config')
    if cfg['geometry_controls'].get('bc_teacher') != 1 or 'bc_tail' in cfg['geometry_controls']:
        parser.error('Reference must be original BC, not a tail variant')
    references = {file:rows(reference/(file+'.csv')) for file in ('metrics','dynamics')}
    for file, data in references.items():
        if sorted(data) != list(range(1,301)):
            parser.error(f'Reference {file}.csv must contain all rounds 1-300 exactly once')
    stamp = time.strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
    folder = ROOT/'results/bc_tail_launches'/stamp
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(reference=str(reference), gpus=opt.gpus, status='preflight', stages={})
    def status(value):
        record['status'] = value; write_json(folder/'status.json', record)
        print(f'STATUS {value}: {folder / "status.json"}', flush=True)
    def check(stage, directory, start, end):
        reports = {file:compare(references[file], rows(directory/(file+'.csv')), start, end, fields)
                   for file, fields in [('metrics',('acc','phase','lr')),
                                        ('dynamics',('phase','lr','gate','aux_scale','trusted_weight_gate'))]}
        record['stages'][stage] = reports
        write_json(folder/(stage+'_comparison.json'), reports)
        if not all(r['passed'] for r in reports.values()):
            raise ValueError(stage+' mismatch: see comparison report; no automatic tolerance relaxation')
    try:
        # Check metadata against the fixed recipe before spending 225 rounds.
        sys.path.insert(0, str(ROOT))
        from options import args_parser
        from fl_runner import _geometry_controls
        from training_dynamics import lr_controls
        from bc_tail import run_identity, validate_fork
        argv = sys.argv
        try:
            sys.argv = ['auto','--config',str(ROOT/'configs/experiment_trusted_lr015_bc.yaml'),
                        '--seed',str(cfg['seed_model']),'--partition_seed',str(cfg['seed_partition']),
                        '--sample_seed',str(cfg['seed_sample']),'--num_workers',str(cfg['num_workers'])]
            args = args_parser(); args.num_rounds=300; args.num_labeled=500
        finally:
            sys.argv = argv
        expected = run_identity(args)
        if any(cfg.get(k) != v for k,v in expected.items()) or cfg['geometry_controls'] != _geometry_controls(args) or cfg['lr_controls'] != lr_controls(args):
            raise ValueError('Reference settings differ from fixed BC recipe')
        if opt.prefix_checkpoint:
            checkpoint = opt.prefix_checkpoint.resolve(strict=True)
            prefix = checkpoint.parent
        else:
            run_id = 'bc_auto_prefix_'+stamp
            prefix = ROOT/'results/CIFAR10/runs'/(run_id+'_a0.1')
            record['prefix'] = str(prefix); status('training_prefix')
            run_jobs([('prefix',lambda gpu:train_command('experiment_trusted_lr015_bc.yaml',run_id,cfg,gpu,stop=225))],opt.gpus[:1],folder)
            checkpoint = prefix/'round_0225.pt'
        record['checkpoint'] = str(checkpoint)
        import torch
        ckpt = torch.load(checkpoint,map_location='cpu',weights_only=False)
        replay_id = 'bc_auto_replay_'+stamp
        replay = ROOT/'results/CIFAR10/runs'/(replay_id+'_a0.1')
        validate_fork(ckpt,_geometry_controls(args),lr_controls(args),str(checkpoint),str(replay),expected)
        del ckpt
        status('checking_prefix'); check('prefix',prefix,1,225)
        status('training_replay')
        run_jobs([('replay',lambda gpu:train_command('experiment_trusted_lr015_bc.yaml',replay_id,cfg,gpu,checkpoint,230))],opt.gpus[:1],folder)
        record['replay'] = str(replay)
        status('checking_replay'); check('replay',replay,226,230)
        jobs = []; destinations = {}
        for mode in ('breliability','aguard','featurehalf'):
            rid = f'bc_auto_{mode}_{stamp}'
            destinations[mode] = str(ROOT/'results/CIFAR10/runs'/(rid+'_a0.1'))
            jobs.append((mode,lambda gpu,m=mode,r=rid:train_command(f'experiment_bc_tail_{m}.yaml',r,cfg,gpu,checkpoint)))
        record['experiments'] = destinations; status('training_tails')
        run_jobs(jobs,opt.gpus,folder)
        for mode,destination in destinations.items():
            result = rows(Path(destination)/'metrics.csv')
            if sorted(result) != list(range(226,301)):
                raise ValueError(mode+' did not finish all tail rounds')
        status('complete')
        return 0
    except KeyboardInterrupt:
        record['error'] = 'Interrupted. Already launched processes may still run; inspect logs/PIDs before restarting.'
        status('interrupted'); return 130
    except Exception as exc:
        record['error'] = str(exc); status('failed')
        print(str(exc),file=sys.stderr,flush=True); return 1


if __name__ == '__main__':
    raise SystemExit(main())
