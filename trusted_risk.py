"""A-only risk weighting. Calibration labels come exclusively from labeled queries.

The posterior-like shrinkage below is a heuristic penalty, not an unbiased
probability: queries are excluded from centers, but not supervised training.
No state is accumulated across epochs, so repeated query visits are not treated
as new independent observations. B weighting and prototype fitting stay intact.
"""
import math
import torch
import torch.nn.functional as F

GROUPS = ('neutral', 'distance', 'conflict', 'unknown')
AUDIT_FIELDS = ('a_visits', 'a_correct', 'old_weight', 'new_weight',
                'old_wrong_weight', 'new_wrong_weight', 'calibrated_visits')


def new_audit(num_classes, device):
    return torch.zeros(num_classes*4, len(AUDIT_FIELDS), device=device, dtype=torch.float64)


def audit_rows(buffer):
    return [dict(pred_class=i//4, geometry_group=GROUPS[i%4], **dict(zip(AUDIT_FIELDS, row)))
            for i, row in enumerate(buffer.cpu().tolist()) if row[0]]


@torch.no_grad()
def evidence(z, yhat, ref):
    sim = z @ ref['centers'].T
    own = sim.gather(1, yhat[:, None]).squeeze(1)
    rivals = sim.masked_fill(~ref['valid'][None, :], -1e9).clone()
    rivals.scatter_(1, yhat[:, None], -1e9)
    best, rival = rivals.max(1)
    known = ref['valid'][yhat] & (ref['valid'].sum() >= 2)
    absolute = (ref['radius'][yhat] - (1-own)) / ref['distance_scale'][yhat]
    # A rival must fit its own reference, not just be the least bad available class.
    gap = best-own
    conflict = (known & (gap > .5*ref['margin_scale'][yhat])
                & ((1-best) <= ref['radius'][rival]) & (ref['trust'][rival] > 0))
    group = torch.zeros_like(yhat)
    group[known & (absolute < 0)] = 1
    group[conflict] = 2
    group[~known] = 3
    severity = torch.where(group == 2, 1-torch.exp(-gap.clamp_min(0)/ref['margin_scale'][yhat]),
                           1-torch.exp(absolute.clamp_max(0)))
    trust = ref['trust'][yhat]
    trust = torch.where(group == 2, torch.minimum(trust, ref['trust'][rival]), trust)
    return group, severity, trust * known.float()


@torch.no_grad()
def fit_calibration(z, logits, y, ref, threshold=.95, temperature=1.):
    # Identical stable per-class alternating support/query split to fit_reference.
    query = torch.zeros_like(y, dtype=torch.bool)
    for cls in range(len(ref['valid'])):
        if ref['valid'][cls]:
            query[torch.where(y == cls)[0][1::2]] = True
    confidence, pred = F.softmax(logits/temperature, dim=-1).max(1)
    group, _, _ = evidence(z, pred, ref)
    band = (confidence >= .99).long()
    count = z.new_zeros((len(ref['valid']), 2, 4))
    errors = torch.zeros_like(count)
    selected = query & (confidence >= threshold)
    key = pred[selected]*8 + band[selected]*4 + group[selected]
    count.view(-1).index_add_(0, key, torch.ones_like(confidence[selected]))
    errors.view(-1).index_add_(0, key, (pred[selected] != y[selected]).to(z.dtype))
    return dict(count=count, errors=errors, threshold=float(threshold))


@torch.no_grad()
def score_a(z, yhat, eval_logits, ref, gate, age_fraction=0.,
            distance_cap=.15, conflict_cap=.60, prior_count=8., temperature=1.):
    group, severity, trust = evidence(z, yhat, ref)
    cap = torch.where(group == 2, severity.new_full(severity.shape, conflict_cap),
                      severity.new_full(severity.shape, distance_cap))
    prior = severity*cap
    cal = ref['risk_calibration']
    # Match the eval-mode calibration. Train-mode pseudo-labels can disagree;
    # those cases get geometric fallback, not calibration for another label.
    conf, eval_pred = F.softmax(eval_logits/temperature, dim=-1).max(1)
    band = (conf >= .99).long()
    n = cal['count'][yhat, band, group]
    errors = cal['errors'][yhat, band, group]
    pool_n = cal['count'].sum(0)[band, group]-n
    pool_errors = cal['errors'].sum(0)[band, group]-errors
    # Leave-own-class-out pooling avoids counting the same queries twice.
    pool_penalty = (pool_errors + prior_count*prior)/(pool_n+prior_count)
    calibrated = (errors + prior_count*pool_penalty)/(n+prior_count)
    usable = (eval_pred == yhat) & (conf >= cal['threshold'])
    penalty = torch.where(usable, calibrated, prior).clamp_min(0)
    penalty = torch.minimum(penalty, cap)
    active = (group == 1) | (group == 2)
    authority = float(gate)*math.exp(-max(0., age_fraction))*trust
    weight = 1-authority*penalty*active.float()
    return dict(weight=weight, group=group, calibration_n=torch.where(usable, n+pool_n, torch.zeros_like(n)),
                penalty=penalty*active.float())


@torch.no_grad()
def audit(buffer, result, in_a, pred, diagnostic_gt, old_weight):
    """GT is used only after weights are computed, for read-only reporting."""
    correct = (pred == diagnostic_gt).double()
    a = in_a.double()
    old = old_weight.detach().double()*a
    new = result['weight'].double()*a
    values = torch.stack((a, a*correct, old, new, old*(1-correct), new*(1-correct),
                          a*(result['calibration_n'] > 0)), dim=1)
    buffer.index_add_(0, pred*4+result['group'], values)
