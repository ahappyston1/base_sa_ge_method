"""Labeled-reference geometry with measured authority, not pressure-based authority.

Reference fitting takes ONLY labeled features/labels. Query scoring takes no GT.
Reliability scores are heuristics, not calibrated correctness probabilities.
"""
import math
import torch
import torch.nn.functional as F


@torch.no_grad()
def fit_reference(z, y, num_classes, min_count=4):
    device = z.device
    centers = z.new_zeros((num_classes, z.shape[1]))
    valid = torch.zeros(num_classes, device=device, dtype=torch.bool)
    support_n = z.new_zeros(num_classes)
    query_mask = torch.zeros(len(y), device=device, dtype=torch.bool)
    for c in range(num_classes):
        ids = torch.where(y == c)[0]
        if len(ids) < min_count:
            continue
        support, query = ids[::2], ids[1::2]
        mean = z[support].mean(0)
        if float(mean.norm()) < 1e-6:
            continue
        centers[c] = F.normalize(mean, dim=0)
        valid[c] = True
        support_n[c] = len(support)
        query_mask[query] = True
    qz, qy = z[query_mask], y[query_mask]
    radius = z.new_ones(num_classes)
    margin_floor = z.new_zeros(num_classes)
    distance_scale = z.new_full((num_classes,), .05)
    margin_scale = z.new_full((num_classes,), .05)
    trust = z.new_zeros(num_classes)
    query_n = z.new_zeros(num_classes)
    votes = z.new_zeros(num_classes)
    if int(valid.sum()) >= 2 and len(qy):
        sims = (qz @ centers.T).masked_fill(~valid[None, :], -1e9)
        pred = sims.argmax(1)
        own = sims.gather(1, qy[:, None]).squeeze(1)
        other = sims.clone()
        other.scatter_(1, qy[:, None], -1e9)
        margins = own - other.max(1).values
        distances = 1 - own
        for c in range(num_classes):
            mask = qy == c
            query_n[c] = mask.sum()
            if mask.any():
                radius[c] = torch.quantile(distances[mask], .90)
                margin_floor[c] = torch.quantile(margins[mask], .10)
                distance_scale[c] = distances[mask].std(unbiased=False).clamp_min(.05)
                margin_scale[c] = margins[mask].std(unbiased=False).clamp_min(.05)
            chosen = pred == c
            n = int(chosen.sum())
            votes[c] = n
            if n >= 2 and valid[c]:
                # Wilson lower confidence bound for classwise predicted precision.
                p = float((qy[chosen] == c).float().mean())
                z95 = 1.96
                lower = (p + z95*z95/(2*n) - z95*math.sqrt(p*(1-p)/n + z95*z95/(4*n*n))) / (1 + z95*z95/n)
                # Insufficient reference support further reduces authority.
                trust[c] = max(0., lower) * min(1., float(query_n[c]) / min_count)
    return dict(centers=centers, valid=valid, radius=radius, margin_floor=margin_floor,
                distance_scale=distance_scale, margin_scale=margin_scale,
                trust=trust, support_n=support_n, query_n=query_n, votes=votes)


@torch.no_grad()
def score_queries(z, yhat, reference, gate, age_fraction=0.0):
    p, valid = reference['centers'], reference['valid']
    sims = z @ p.T
    own = sims.gather(1, yhat[:, None]).squeeze(1)
    other = sims.masked_fill(~valid[None, :], -1e9)
    other.scatter_(1, yhat[:, None], -1e9)
    ok = valid[yhat] & (valid.sum() >= 2)
    margin = own - other.max(1).values
    relative = (margin - reference['margin_floor'][yhat]) / reference['margin_scale'][yhat]
    absolute = (reference['radius'][yhat] - (1-own)) / reference['distance_scale'][yhat]
    score = torch.where(ok, torch.minimum(relative, absolute), torch.zeros_like(own))
    # Conforming points keep full weight; only out-of-reference tails are penalized.
    # This is a conformity score, explicitly not a correctness probability.
    reliability = torch.exp(score.clamp(max=0.0))
    authority = gate * reference['trust'][yhat] * math.exp(-max(0., age_fraction)) * ok.float()
    # Unknown/untrusted geometry is a no-op; trusted disagreement reduces weight.
    weight = 1.0 - authority * (1.0 - reliability)
    return dict(score=score, reliability=reliability, authority=authority,
                weight=weight, valid=ok)


@torch.no_grad()
def refresh_reference(model, dataset, transform, num_classes, device, min_count=4, batch_size=128,
                      risk_threshold=None, risk_temperature=1., include_query_audit=False):
    """Fresh eval features; deterministic transforms, no DataLoader/RNG consumption.

    Split by stable dataset ID within each class. Held-out here means excluded
    from centroid fitting, NOT excluded from supervised training.
    """
    n = int(dataset.client_dataset_original_len)
    ids = list(dataset.indices[:n])
    order = sorted({int(sid): j for j, sid in enumerate(ids)}.items())
    rows = [dataset.client_dataset[j] for _, j in order]
    features, labels, logits = [], [], []
    training = model.training
    model.eval()
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start+batch_size]
            x = torch.stack([transform(image) for image, _ in batch]).to(device)
            z, logit = model(x)
            features.append(F.normalize(z, dim=-1))
            if risk_threshold is not None:
                logits.append(logit)
            labels.extend(int(y) for _, y in batch)
    finally:
        model.train(training)
    if not features:
        raise ValueError('Trusted geometry requires at least one labeled example')
    z, y = torch.cat(features), torch.tensor(labels, device=device)
    ref = fit_reference(z, y, num_classes, min_count)
    if include_query_audit:
        from bc_reconstruction import reference_audit
        ref['query_audit'] = reference_audit(z, y, ref)
    if risk_threshold is not None:
        from trusted_risk import fit_calibration
        ref['risk_calibration'] = fit_calibration(z, torch.cat(logits), y, ref, risk_threshold, risk_temperature)
    return ref
