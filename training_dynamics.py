"""Pure helpers for the PPFPSL learning rate and read-only dynamics log."""
import math


def schedule_rounds(args):
    """Zero preserves legacy behavior; an explicit horizon decouples extra tail rounds."""
    return int(getattr(args, 'baseline_schedule_rounds', 0) or args.num_rounds)


def trusted_weight_gate(round_idx, standard_gate, args):
    """Optional early weight ramp; never moves the phase/loss/reference-loss schedule."""
    start = int(getattr(args, 'trusted_weight_start', 0))
    if not start:
        return float(standard_gate)
    end = int(args.trusted_weight_end)
    t = min(1., max(0., (round_idx-start)/float(end-start)))
    return max(float(standard_gate), .5*(1-math.cos(math.pi*t)))


def lr_controls(args):
    return {
        "initial": float(args.lr_local_training),
        "minimum": float(getattr(args, "lr_min", 1e-4)),
        "rounds": schedule_rounds(args),
        "mid_start": int(getattr(args, "lr_mid_start", 60)),
        "mid_end": int(getattr(args, "lr_mid_end", 90)),
        "mid_factor": float(getattr(args, "lr_mid_factor", 1.0)),
    }


def cosine_learning_rate(round_idx, initial, rounds, minimum=1e-4,
                         mid_start=60, mid_end=90, mid_factor=1.0):
    if not (math.isfinite(initial) and math.isfinite(minimum)
            and 0 < minimum <= initial and rounds > 0 and round_idx >= 0):
        raise ValueError("Require finite 0 < minimum <= initial, rounds > 0, round_idx >= 0")
    if not (0 <= mid_start < mid_end and math.isfinite(mid_factor) and 0 < mid_factor <= 1):
        raise ValueError("Require 0 <= mid_start < mid_end and 0 < mid_factor <= 1")
    # Preserve REV11 exactly within its training budget; no rebound after it.
    angle = math.pi * min(round_idx, rounds) / max(1, rounds)
    base = initial * 0.5 * (1.0 + math.cos(angle))
    if mid_factor != 1.0:
        progress = min(1.0, max(0.0, (round_idx - mid_start) / (mid_end - mid_start)))
        base *= 1.0 + (mid_factor - 1.0) * 0.5 * (1.0 - math.cos(math.pi * progress))
    return max(base, minimum)


def dynamics_row(round_idx, phase, logs, args):
    """Ratios use pooled visits; losses use local-step weighting, not client means."""
    def total(key):
        return sum(float(x.get(key, 0)) for x in logs)

    n = total("cnt_u")
    steps = total("n_batches")
    a_scale = float(args.lambda_A)
    b_scale = float(args.lambda_B)
    row = {"round": round_idx, "phase": phase}
    for key in ("lr", "gate", "aux_scale"):
        row[key] = sum(float(x[key]) * float(x["n_batches"]) for x in logs) / max(steps, 1)
    for bucket in ("a", "b"):
        count = total("cnt_" + bucket)
        mass = total("dyn_" + bucket + "_mass")
        row[bucket + "_ratio"] = count / max(n, 1)
        row[bucket + "_mean_weight"] = mass / count if count else 0.0
        row[bucket + "_effective_mass_per_u"] = mass / max(n, 1)
        row[bucket + "_wrong_mass_per_u"] = total("dyn_" + bucket + "_wrong") / max(n, 1)
    row["c_ratio"] = total("cnt_c") / max(n, 1)
    candidates = total("dyn_b_score_only_n")
    row["b_score_only_candidates"] = int(candidates)
    row["b_score_only_accuracy"] = (total("dyn_b_score_only_correct") / candidates
                                    if candidates else 0.0)
    row["b_score_only_admitted"] = int(total("dyn_b_score_only_admitted"))
    for key, coef in (("L_sup", 1.0), ("L_A", a_scale),
                      ("L_B", b_scale), ("L_proto", float(args.lambda_proto))):
        weighted = 0.0
        for x in logs:
            scale = 1.0
            if key == "L_A":
                scale = float(x["lambda_A_scale"])
            elif key == "L_B":
                scale = float(x["lambda_B_scale"])
            if phase == 1 and key in ("L_B", "L_proto"):
                scale *= float(x["aux_scale"])
            weighted += float(x[key]) * float(x["n_batches"]) * coef * scale
        row[key + "_effective"] = weighted / max(steps, 1)
    row["loss_reconstructed"] = sum(row[k + "_effective"] for k in ("L_sup", "L_A", "L_B", "L_proto"))
    for flag,key in [('bc_teacher','L_feature_effective'),('mid_prox_mu','L_prox_effective')]:
        if getattr(args,flag,0):
            row[key] = sum(float(x.get(key,0))*float(x['n_batches']) for x in logs)/max(steps,1)
            row['loss_reconstructed'] += row[key]
    if getattr(args,'bc_teacher',0):
        row['teacher_b_visits'] = int(total('bc_teacher_b_total'))
        row['teacher_b_precision'] = total('bc_teacher_b_correct')/max(total('bc_teacher_b_total'),1)
        row['bc_feature_std'] = sum(float(x.get('bc_feature_std',0))*float(x['n_batches']) for x in logs)/max(steps,1)
    if getattr(args, 'bc_tail', 'none') != 'none':
        from bc_tail import gate
        row['tail_gate'] = gate(round_idx)
        for key in ('tail_b_old_mass','tail_b_new_mass','tail_b_old_wrong','tail_b_new_wrong',
                    'tail_a_guard_visits','tail_a_guard_correct','tail_a_removed_correct','tail_a_removed_wrong',
                    'tail_b_true_prob','tail_b_true_nll'):
            row[key] = total(key)
        for band in range(3):
            for suffix in ('visits', 'correct'):
                key = f'tail_score_{band}_{suffix}'
                row[key] = total(key)
    if getattr(args,'bc_targets',0):
        from bc_targets import FIELDS, gate
        row['target_gate']=gate(round_idx)
        for cls in range(10):
            for name in FIELDS:
                key=f'target_c{cls}_{name}'
                row[key]=total(key)
    row["loss_observed"] = sum(float(x["loss"]) * float(x["n_batches"]) for x in logs) / max(steps, 1)
    if getattr(args, 'target_experiment', 'none') != 'none':
        from target_experiments import AUDIT_FIELDS
        for cls in range(args.num_classes):
            for name in AUDIT_FIELDS:
                key = f'experiment_c{cls}_{name}'
                row[key] = total(key)
            for truth_cls in range(args.num_classes):
                key = f'experiment_a_pred{cls}_true{truth_cls}'
                row[key] = total(key)
        if args.target_experiment == 'separation':
            key = 'L_separation_effective'
            row[key] = sum(float(x.get(key,0))*float(x['n_batches']) for x in logs)/max(steps,1)
            row['loss_reconstructed'] += row[key]
            row['separation_active_batches'] = total('separation_active_batches')
            row['separation_unique_labeled'] = total('separation_unique_labeled')
    row["u_visits"] = int(n)
    row["local_steps"] = int(steps)
    row["trust_authority_mean"] = total("trust_authority_sum") / max(n, 1)
    row["trust_valid_fraction"] = total("trust_valid_visits") / max(n, 1)
    row["trust_refreshes"] = int(total("trust_refreshes"))
    if getattr(args, 'trusted_weight_start', 0):
        row['trusted_weight_gate'] = sum(float(x.get('trusted_weight_gate', 0)) * float(x['n_batches']) for x in logs) / max(steps, 1)
    return row
