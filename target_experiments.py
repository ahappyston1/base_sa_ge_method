"""Independent extensions of bc_targets; diagnostic truth never enters decisions.

The auxiliary head is ridge regression on frozen global features, fitted from
online clients' labeled sufficient statistics. Its validation split holds out
labels from head fitting, not from previous backbone training.
"""
from contextlib import contextmanager
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F


MODES = ('none', 'evidence', 'labelhead', 'separation')


def controls(mode):
    common = dict(version=1, mode=mode, ramp=[60, 120])
    if mode == 'evidence':
        common.update(own_view_conf=.65, own_history_conf=.60,
                      conflict_margin_scale=.5, geometry='missing_is_neutral')
    elif mode == 'labelhead':
        common.update(fit_start=31, ridge=1., temperature=.15, folds=2,
                      min_class_count=8, min_validation_votes=8,
                      wilson_lower=.65, min_margin=.10,
                      alternative_teacher_conf=.60, auxiliary_target_mix=.75,
                      history_conf=.60, require_original_class_present=True,
                      history='per_sample_prior_participations')
    elif mode == 'separation':
        common.update(weight=.03, temperature=.20, max_negative_weight=2.,
                      positives='true_labels_only', unique_ids=True,
                      second_view_bn='batch_stats_without_buffer_update')
    return common


@torch.no_grad()
def revise(result, mode, pred, a, q, p, hist, mature, z, ref, auxiliary=None, guard=False):
    """Modify A only. B/C, original weights and the outer target ramp are intact."""
    result = {k: v.clone() for k, v in result.items()}
    result['before_state'] = result['state'].clone()
    s = result['state']
    own = pred[:, None]
    if mode == 'evidence':
        valid = ref['valid'] & (ref['trust'] > 0)
        sim = z @ ref['centers'].T
        ours = sim.gather(1, own).squeeze(1)
        competing = sim.masked_fill(~valid[None], -1e9).clone()
        competing.scatter_(1, own, -1e9)
        best, rival = competing.max(-1)
        known = valid[pred] & (valid.sum() >= 2)
        conflict = (known & valid[rival] & ((best - ours) > .5*ref['margin_scale'][pred])
                    & ((1-best) <= ref['radius'][rival]))
        # Weak/missing geometry alone is never positive evidence of an error.
        positive_own = ((q.argmax(-1) == pred) & (p.argmax(-1) == pred)
                        & (hist.argmax(-1) == pred)
                        & (q.gather(1, own).squeeze(1) >= .65)
                        & (p.gather(1, own).squeeze(1) >= .65)
                        & (hist.gather(1, own).squeeze(1) >= .60))
        # Conservative protection: absence of evidence does not restore all A.
        protect = a & mature & positive_own & ~conflict
        s[protect] = 0
        result['geometry_unknown'] = a & ~known
        result['geometry_conflict'] = a & conflict
    elif mode == 'labelhead':
        if auxiliary is None:
            raise ValueError('labelhead requires a current-round frozen auxiliary head')
        aq, good, ah, amature, available = auxiliary
        alt = aq.argmax(-1)
        stable = amature & good & available[pred] & (ah.argmax(-1) == alt) & (ah.max(-1).values >= .60)
        teacher_agrees = ((q.argmax(-1) == alt) & (p.argmax(-1) == alt)
                          & (q.max(-1).values >= .60) & (p.max(-1).values >= .60))
        reliable = a & stable & teacher_agrees
        protect = reliable & (alt == pred)
        if guard:
            result['guard_protected'] = protect & (s != 0)
            result['guard_hard_share'] = torch.full_like(q[:, 0], guard_controls()['hard_share'])
        s[protect] = 0
        change = reliable & (alt != pred)
        s[change] = 2
        # One replacement target, not an additional CE against the old label.
        result['target'][change] = (.75*aq + .25*(q+p)/2)[change]
        result['aux_prediction'] = alt
        result['aux_good'] = good
        result['aux_override'] = change
        result['aux_stable'] = stable
        result['aux_teacher_agrees'] = teacher_agrees
    result['a_changed'] = a & (s != 0)
    result['soft_a'] = a & (s == 2)
    if guard:
        # The routing state stays identical to ordinary labelhead. Only protected
        # classification loss and prototype writes change; feature masks do not.
        blocked = a & ((result['before_state'] != 0) | (s != 0))
        result['prototype_blocked'] = blocked
        result['feature_a'] = result['a_changed'].clone()
    return result


def guard_controls():
    return dict(version=2, protection='original_labelhead_set', hard_share=.5,
                correction='unchanged', prototype='union_before_after_nonhard',
                feature='original_labelhead')


GUARD_FIELDS = ('guard_protected','guard_protected_correct','guard_blocked_protected',
                'guard_blocked_correct_mass','guard_blocked_wrong_mass',
                'guard_ce_reduced_correct_mass','guard_ce_reduced_wrong_mass')


@torch.no_grad()
def guard_audit(result, pred, truth, weights, strength):
    protected = result['guard_protected']
    correct = pred == truth
    mass = weights.detach()*protected*strength
    reduced = mass*(1-result['guard_hard_share'])
    return dict(zip(GUARD_FIELDS, (float(protected.sum()),float((protected & correct).sum()),
                float(protected.sum()),float((mass*correct).sum()),float((mass*~correct).sum()),
                float((reduced*correct).sum()),float((reduced*~correct).sum()))))


@contextmanager
def no_bn_updates(model):
    modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    previous = [m.track_running_stats for m in modules]
    try:
        for m in modules:
            m.track_running_stats = False
        yield
    finally:
        for m, old in zip(modules, previous):
            m.track_running_stats = old


def separation_loss(z1, z2, labels, ids, logits, temperature=.20):
    """Two labeled views, deduplicated identities; no pseudo-label positives.

    Bounded, detached confusion weights affect only true inter-class negatives.
    Single-class batches abstain rather than learning a collapse objective.
    """
    order = torch.argsort(ids, stable=True)
    sorted_ids = ids[order]
    keep = torch.ones_like(sorted_ids, dtype=torch.bool)
    keep[1:] = sorted_ids[1:] != sorted_ids[:-1]
    first = order[keep]
    y = labels[first]
    if len(y.unique()) < 2:
        return (z1.sum()+z2.sum())*0
    z = F.normalize(torch.cat((z1[first], z2[first])), dim=-1)
    y2 = y.repeat(2)
    logit = z @ z.T / temperature
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = (y2[:, None] == y2[None, :]) & ~diagonal
    with torch.no_grad():
        probabilities = logits[first].softmax(-1)
        cross = probabilities[:, y]
        confusion = .5*(cross+cross.T)
        negative_weight = (1+confusion).repeat(2, 2)
        negative_weight.masked_fill_(y2[:, None] == y2[None, :], 1.)
    denominator = torch.logsumexp((logit+negative_weight.log()).masked_fill(diagonal, -torch.inf), -1)
    log_prob = logit-denominator[:, None]
    return -((log_prob*positive).sum(-1)/positive.sum(-1).clamp_min(1)).mean()


def sufficient_statistics(z, y, folds, classes):
    x = torch.cat((z.detach().double().cpu(), torch.ones(len(z), 1, dtype=torch.float64)), 1)
    y = y.cpu(); folds = folds.cpu()
    gram = x.new_zeros(2, x.shape[1], x.shape[1])
    rhs = x.new_zeros(2, x.shape[1], classes)
    counts = x.new_zeros(2, classes)
    for fold in range(2):
        selected = folds == fold
        xf = x[selected]; yf = y[selected]
        gram[fold] = xf.T @ xf
        rhs[fold] = xf.T @ F.one_hot(yf, classes).double()
        counts[fold] = torch.bincount(yf, minlength=classes).double()
    return dict(gram=gram, rhs=rhs, counts=counts)


def solve_head(gram, rhs):
    penalty = torch.eye(len(gram), dtype=gram.dtype, device=gram.device)
    penalty[-1, -1] = .01
    return torch.linalg.solve(gram+penalty, rhs)


def raw_scores(z, weight):
    return z @ weight[:-1] + weight[-1]


def calibration_histogram(scores, truth, valid):
    """Client-side OOF score-bin counts; no per-image features sent to server."""
    score = scores.masked_fill(~valid[None], -1e9)
    top, cls = score.topk(2, dim=-1)
    gap = (top[:, 0]-top[:, 1]).clamp(0, .99999)
    bins = (gap*20).long()
    out = scores.new_zeros(scores.shape[1], 20, 2)
    if valid.sum() >= 2:
        key = cls[:, 0]*20+bins
        out.view(-1, 2).index_add_(0, key, torch.stack((torch.ones_like(gap),
                                 (cls[:, 0] == truth).to(gap.dtype)), 1))
    return out


def calibration_thresholds(histogram):
    # Search a fixed grid using out-of-fold head predictions; not test labels.
    n = histogram[:, :, 0].flip(1).cumsum(1).flip(1)
    correct = histogram[:, :, 1].flip(1).cumsum(1).flip(1)
    p = correct/n.clamp_min(1)
    z = 1.96; nn = n.clamp_min(1)
    lower = (p+z*z/(2*nn)-z*torch.sqrt((p*(1-p)/nn+z*z/(4*nn*nn)).clamp_min(0)))/(1+z*z/nn)
    grid = torch.arange(20, device=n.device, dtype=n.dtype)/20
    eligible = (n >= 8) & (lower >= .65) & (grid[None] >= .10)
    thresholds = torch.where(eligible, grid[None], torch.full_like(n, float('inf'))).min(-1).values
    return thresholds


@torch.no_grad()
def fit_online_head(model, dataset, client_indices, classes, transform, device):
    """Simulated two-message federated fit on this round's online labeled data.

    Caches below represent client-local data. Server-facing quantities are only
    sufficient statistics and validation histograms. No old feature bank used.
    """
    was_training = model.training
    model.eval()
    caches = []
    try:
        for indices in client_indices:
            ids = sorted(set(map(int, indices)))
            if not ids:
                continue
            features, labels = [], []
            for start in range(0, len(ids), 128):
                rows = [dataset[i] for i in ids[start:start+128]]
                images = torch.stack([transform(row[0]) for row in rows]).to(device)
                features.append(F.normalize(model(images)[0], dim=-1).cpu().double())
                labels.extend(int(row[1]) for row in rows)
            z = torch.cat(features); y = torch.tensor(labels)
            folds = torch.zeros_like(y)
            for cls in range(classes):
                ix = torch.where(y == cls)[0]
                folds[ix] = torch.arange(len(ix)) % 2
            caches.append((z, y, folds))
    finally:
        model.train(was_training)
    if not caches:
        raise ValueError('No online labeled samples for labelhead')
    stats = [sufficient_statistics(z, y, fold, classes) for z, y, fold in caches]
    totals = {key: sum(s[key] for s in stats) for key in stats[0]}
    fold_heads = [solve_head(totals['gram'][i], totals['rhs'][i]) for i in range(2)]
    histogram = torch.zeros(classes, 20, 2, dtype=torch.float64)
    for z, y, fold in caches:
        for heldout in range(2):
            selected = fold == heldout
            valid = totals['counts'][1-heldout] >= 4
            histogram += calibration_histogram(raw_scores(z[selected], fold_heads[1-heldout]), y[selected], valid)
    weight = solve_head(totals['gram'].sum(0), totals['rhs'].sum(0))
    counts = totals['counts'].sum(0)
    return dict(weight=weight.float(), counts=counts.float(), valid=counts >= 8,
                threshold=calibration_thresholds(histogram).float(), calibration=histogram.float())


@torch.no_grad()
def predict_head(z, head):
    weight = head['weight'].to(z)
    valid = head['valid'].to(z.device)
    score = raw_scores(z, weight).masked_fill(~valid[None], -1e9)
    top, cls = score.topk(2, dim=-1)
    if int(valid.sum()) < 2:
        return torch.full_like(score, 1/score.shape[1]), torch.zeros(len(z), dtype=torch.bool, device=z.device)
    good = (top[:, 0]-top[:, 1]) >= head['threshold'].to(z)[cls[:, 0]]
    return (score/.15).softmax(-1), good


def write_head_audit(path, round_idx, head):
    """Replace replayed/uncommitted rounds after a checkpoint restart."""
    path = Path(path)
    fields = ['round','class','labeled_count','valid','margin_threshold','oof_votes','oof_correct']
    previous = []
    if path.exists():
        with path.open(newline='', encoding='utf-8') as handle:
            previous = [row for row in csv.DictReader(handle) if int(row['round']) < round_idx]
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(previous)
        for cls in range(len(head['counts'])):
            writer.writerow(dict(round=round_idx, **{'class': cls}, labeled_count=float(head['counts'][cls]),
                valid=int(head['valid'][cls]), margin_threshold=float(head['threshold'][cls]),
                oof_votes=float(head['calibration'][cls,:,0].sum()),
                oof_correct=float(head['calibration'][cls,:,1].sum())))
    os.replace(temporary, path)


AUDIT_FIELDS = ('protected', 'protected_correct', 'protected_correct_mass', 'protected_wrong_mass',
                'aux_eligible', 'aux_correct', 'aux_correct_when_student_wrong',
                'aux_override', 'aux_fix', 'aux_harm', 'aux_stable', 'aux_teacher_agrees',
                'geometry_unknown', 'geometry_conflict')


@torch.no_grad()
def audit(result, pred, a, truth, weight):
    """Read-only repeated-visit audit, grouped by original predicted class."""
    zero = torch.zeros_like(a)
    right = pred == truth
    protected = a & (result['before_state'] != 0) & (result['state'] == 0)
    good = a & result.get('aux_good', zero)
    aux_right = result.get('aux_prediction', pred) == truth
    override = a & result.get('aux_override', zero)
    values = (protected, protected & right, weight*protected*right, weight*protected*~right,
              good, good & aux_right, good & aux_right & ~right, override,
              override & ~right & (result['target'].argmax(-1) == truth),
              override & right & (result['target'].argmax(-1) != truth),
              a & result.get('aux_stable', zero), a & result.get('aux_teacher_agrees', zero),
              result.get('geometry_unknown', zero), result.get('geometry_conflict', zero))
    out = torch.zeros(result['target'].shape[1], len(values), device=pred.device, dtype=torch.float64)
    out.index_add_(0, pred, torch.stack([v.double() for v in values], 1))
    return out
