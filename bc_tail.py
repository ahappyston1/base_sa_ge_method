"""Late BC objectives and strict communication-round checkpoint forks."""
import math
import os
import torch
import torch.nn.functional as F


def gate(round_idx):
    t = min(1., max(0., (round_idx - 225) / 35.))
    return .5 - .5 * math.cos(math.pi * t)


@torch.no_grad()
def reliability(first, second, temperature=1.):
    q = F.softmax(first / temperature, -1)
    p = F.softmax(second / temperature, -1)
    mix = (q + p) / 2
    js = .5 * ((q * (q.clamp_min(1e-12).log() - mix.clamp_min(1e-12).log())).sum(-1)
               + (p * (p.clamp_min(1e-12).log() - mix.clamp_min(1e-12).log())).sum(-1))
    return (q.max(-1).values * (1 - js / math.log(2))).clamp(0, 1), q, p


@torch.no_grad()
def reweight_b(weights, mask, scores, strength):
    old = weights.detach() * mask
    center = (old * scores).sum() / old.sum().clamp_min(1e-12)
    return old * (1 + .5 * strength * (scores - center))


@torch.no_grad()
def guard_a(weights, mask, pred, q, p, strength):
    cq, yq = q.max(-1); cp, yp = p.max(-1)
    selected = mask & (pred != yq) & (yq == yp) & (cq >= .90) & (cp >= .90)
    return weights.detach() * (1 - .20 * strength * selected), selected


def tail_controls(mode):
    return dict(version=1, mode=mode, start=225, end=260, b_alpha=.5,
                a_cap=.2, a_threshold=.9, feature_final=.05)


def validate_fork(ckpt, geometry, lr, source, destination, expected_run):
    """Only explicit baseline-BC round225 forks may change tail controls."""
    required = ('global_model','p_ref','p_ref_valid','client_states','fedavg_acc',
                'np_random_state','torch_rng_state','python_rng_state',
                'numpy_global_rng_state','cuda_rng_state','schedule')
    missing = [k for k in required if k not in ckpt or ckpt[k] is None]
    if missing:
        raise ValueError('Incomplete BC checkpoint: ' + ', '.join(missing))
    if ckpt.get('method_rev') != 13 or ckpt.get('round') != 225:
        raise ValueError('BC fork requires a complete METHOD_REV=13 checkpoint at round 225; latest/best cannot be relabeled')
    if len(ckpt['fedavg_acc']) != 225:
        raise ValueError('Checkpoint accuracy history must contain exactly 225 rounds')
    original = ckpt.get('geometry_controls', {})
    target = dict(geometry); target.pop('bc_tail', None)
    if original != target or original.get('bc_teacher') != 1:
        raise ValueError('Fork source must be unmodified BC with identical base geometry controls')
    if ckpt.get('lr_controls') != lr or lr['rounds'] != 300:
        raise ValueError('Fork must preserve the original 300-round LR schedule')
    if os.path.realpath(os.path.dirname(source)) == os.path.realpath(destination):
        raise ValueError('Fork output must be separate from source run')
    # Older checkpoints lack run identity. Require the original companion config
    # rather than silently permitting different data splits or local training.
    import json
    cfg_path = os.path.join(os.path.dirname(source), 'config.json')
    saved = ckpt.get('run_identity')
    if saved is None:
        if not os.path.isfile(cfg_path):
            raise ValueError('Older checkpoint requires original companion config.json')
        with open(cfg_path, encoding='utf-8') as f:
            saved = json.load(f)
    for key, value in expected_run.items():
        if key not in saved or saved[key] != value:
            raise ValueError('BC fork run setting differs or is missing: ' + key)
    states = ckpt['client_states']
    if len(states) != expected_run['num_clients']:
        raise ValueError('Incomplete client states')
    iterable = states.values() if isinstance(states, dict) else states
    for state in iterable:
        if not state or any(k not in state for k in ('bc_heads','rho_bar','rho_valid','p_loc','loc_valid','inited')):
            raise ValueError('Missing client BC head/prototype state')


def run_identity(args):
    return dict(seed_model=int(args.seed), seed_partition=int(args.partition_seed),
                seed_sample=int(args.sample_seed), dataset=args.dataset, alpha=float(args.alpha),
                num_clients=int(args.num_clients), num_online_clients=int(args.num_online_clients),
                num_labeled_per_class=int(args.num_labeled), mu=int(args.mu),
                local_epochs=int(args.local_epochs), batch_labeled=int(args.batch_size_local_labeled_fixmatch),
                num_workers=int(args.num_workers))
