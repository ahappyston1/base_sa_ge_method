"""Read-only FedAvg diagnostics; no forward passes, gradients or RNG use."""
import math
import torch


@torch.no_grad()
def update_rows(global_state, local_states, fused_state, sample_counts,
                client_ids, parameter_names):
    """Separate parameter updates from BN running means/variances.

    Norms are Euclidean, weights match FedAvg, undefined ratios are NaN.
    Processes one state tensor at a time rather than flattening full models.
    """
    if not local_states or len(local_states) != len(sample_counts) or len(local_states) != len(client_ids):
        raise ValueError("Require aligned nonempty clients and sample counts")
    if any(n < 0 for n in sample_counts) or sum(sample_counts) <= 0:
        raise ValueError("Require nonnegative counts with positive sum")
    weights = [n / sum(sample_counts) for n in sample_counts]
    params = set(parameter_names)
    groups = {
        "params": [k for k in global_state if k in params],
        "bn_mean": [k for k in global_state if k.endswith("running_mean")],
        "bn_var": [k for k in global_state if k.endswith("running_var")],
    }
    summary, clients = {}, [dict(client=int(c), fedavg_weight=w) for c, w in zip(client_ids, weights)]
    for group, keys in groups.items():
        if not keys:
            continue
        device = global_state[keys[0]].device
        n = len(local_states)
        # FP64 scalar reductions avoid cancellation in disagreement statistics.
        norms = torch.zeros(n, dtype=torch.float64, device=device)
        dots = torch.zeros_like(norms)
        distances = torch.zeros_like(norms)
        base_sq = torch.zeros((), dtype=torch.float64, device=device)
        fused_sq = torch.zeros_like(base_sq)
        for key in keys:
            base = global_state[key].detach().to(dtype=torch.float64)
            delta = fused_state[key].detach().to(dtype=torch.float64) - base
            base_sq += base.square().sum()
            fused_sq += delta.square().sum()
            for i, state in enumerate(local_states):
                local = state[key].detach().to(dtype=torch.float64) - base
                norms[i] += local.square().sum()
                dots[i] += (local * delta).sum()
                distances[i] += (local - delta).square().sum()
        ns, ds, vs = norms.cpu().tolist(), dots.cpu().tolist(), distances.cpu().tolist()
        base_norm, global_norm = math.sqrt(base_sq.item()), math.sqrt(fused_sq.item())
        local_norms = [math.sqrt(x) for x in ns]
        local_mean = sum(w * x for w, x in zip(weights, local_norms))
        local_rms = math.sqrt(sum(w * x for w, x in zip(weights, ns)))
        disagreement = math.sqrt(sum(w * x for w, x in zip(weights, vs)))
        def ratio(a, b):
            return a / b if b > 0 else float("nan")
        summary.update({
            group + "_global_norm": global_norm,
            group + "_global_relative": ratio(global_norm, base_norm),
            group + "_local_mean_norm": local_mean,
            group + "_local_rms_norm": local_rms,
            group + "_disagreement_rms": disagreement,
            group + "_disagreement_relative": ratio(disagreement, local_rms),
            group + "_retained_ratio": ratio(global_norm, local_mean),
        })
        for i, row in enumerate(clients):
            row[group + "_update_norm"] = local_norms[i]
            row[group + "_update_relative"] = ratio(local_norms[i], base_norm)
            row[group + "_distance_to_fused"] = math.sqrt(vs[i])
            row[group + "_cos_to_fused"] = ratio(ds[i], local_norms[i] * global_norm)
    return summary, clients
