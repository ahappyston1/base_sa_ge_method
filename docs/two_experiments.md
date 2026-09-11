# REV13: two experiments in one repository

No separate Git branches are necessary. Both experiments use the same entry
point and select behavior with pp_geom_mode. Configurations are full snapshots,
not partial overlays. The default YAML remains legacy geometry with LR=0.1.

## Server commands

Run in the repository root, after activating the existing PyTorch environment.
The last argument is the model/sampling seed; partition_seed remains 0.

```bash
# Validate before committing GPU time.
python -m pytest tests/test_ppfpsl_routing.py tests/test_trusted_geometry.py tests/test_geometry_audit.py tests/test_training_dynamics.py tests/test_experiment_configs.py tests/test_update_diagnostics.py -q

# New geometry only, LR=0.1, GPU 0, seed 7
bash scripts/run_experiment.sh trusted 0 7

# Higher initial LR only, legacy geometry, LR=0.15, GPU 1, seed 7
bash scripts/run_experiment.sh high_lr 1 7

# Optional same-code baseline, legacy geometry + LR=0.1
bash scripts/run_experiment.sh reference 2 7
```

Launch the two training commands in separate server terminals/sessions for
concurrent runs. The launcher itself does not background or schedule jobs.
It uses experiment + seed + timestamp + process ID for separate run directories.
All recipes use CIFAR10, alpha=0.1, 300 rounds, 20/8 clients, 5 local epochs,
num_workers=16, cap=0, pp_teacher=0, lr_min=0.0001, lr_mid_factor=1,
pp_b_conf_rescue=0. No EMA and no middle-LR reduction are bundled into these runs.

High-LR recipe differs from the reference recipe ONLY in lr_local_training.
Trusted recipe differs ONLY in pp_geom_mode; that mode is a compound geometry
redesign, not a one-line weighting ablation. Compare each to the reference;
the two experimental runs do not isolate geometry if compared only to each other.
Historical REV11 runs are useful context, but lack the same revision and audit.

## Exact trusted mode design

Phase1 is unchanged. Starting in Phase2, at the beginning of each local epoch:

1. Deduplicate the original labeled IDs and extract current-student eval features
   with deterministic normalization, without updating BN or consuming augmentation
   RNG. This avoids the dataset's 2000x length and repeated-access support counts.
2. For each class with >=4 unique labels, alternate stable IDs into support and
   query halves. Fit the normalized centroid only with support features. Query
   features never contribute to that centroid. This is held out from centroid
   fitting only: these labels are still used for supervised model training.
3. On query features, estimate classwise 90th-percentile own distance and
   10th-percentile own-versus-nearest-other margin, with observed standard
   deviations (floor 0.05). Estimate centroid classifier predicted-class precision
   using a Wilson lower bound, attenuated if own-class query support is small.
   With fewer than two valid classes, or insufficient evidence, authority is zero.

For each unlabeled sample, the student training weak view still supplies its
pseudo-label. A separate eval/no-grad forward supplies the geometry feature.
No unlabeled GT is passed to reference fitting or query scoring.

The geometry score is the minimum of standardized margin support and
standardized own-distance support. Conformity is exp(min(score,0)), so points
inside both reference bounds receive 1, not an automatic penalty.

Authority = gate * class trust * exp(-within-epoch step fraction).
Correction weight = 1 - authority * (1 - conformity).
Untrusted/missing geometry has zero authority and correction weight 1. This is
integral to the new mode, not the removed legacy unknown-B-score special case.
Trust and conformity are bounded heuristic controls, not calibrated probabilities.

ABC remains mutually exclusive and hard-partitioned by confidence:
A uses s>=tau_warmup; B is non-A with s>=eta_B; C is the rest. Configured caps
still apply. Geometry does NOT hard-reject samples in this first trusted version.
A CE weight uses the correction weight; B KL weight uses s*correction weight.
All weights are detached. The supervised prototype loss uses the fresh reference
centers. Existing persistent prototype aggregation/writes remain in the code,
but their historical pressure/mixed centers do not control trusted Phase2/3
routing or supervised prototype targets. A write weights follow the selected
mode's A weights as in the existing pipeline.

Pressure min-max, pressure-dependent alpha, pressure thresholds and Phase3 boost
are inactive in trusted mode. Their config values are retained for legacy mode.
The phase schedule remains, but Phase3 only keeps gate at 1 in trusted mode.
This first implementation uses fresh LOCAL references, not a calibrated global
reference pool; under strong Non-IID, unavailable local classes can have zero
geometry authority. Do not interpret that as proof that geometry is unhelpful.

## Diagnostics and limitations

- config.json and checkpoints identify the mode, calibration minimum count and
  LR controls. METHOD_REV=13 prevents resuming old checkpoints. Start both runs
  from scratch; do not switch modes/LR using an existing checkpoint.
- trust_reference.csv records every local epoch's per-class support/query counts,
  predicted-class votes, trust, radius and margin statistics.
- dynamics.csv reports mean authority and valid fraction. Inspect these before
  claiming the new mechanism actually influenced training.
- geometry_audit.csv retains client/class/confidence strata. In legacy mode its
  score is cosine margin with interpolated threshold; in trusted mode it is the
  standardized joint score with zero threshold. Do not pool the two score units.
- geom_drop_rate is zero in trusted mode by construction; it does not measure
  the amount of geometric downweighting. Use authority and effective A/B mass.
- Reference and query use eval mode, but weights and BN statistics can evolve
  within a local epoch. Rebuilding each epoch and age attenuation reduce staleness;
  they do not establish perfect feature-space alignment. Deterministic reference
  views and augmented query views are also not identical distributions.
- Calibration uses few local labels that the classifier has trained on; held-out
  centroid queries avoid direct self-inclusion, not all estimation bias.
- Trusted mode costs an extra labeled extraction per local epoch and an extra
  eval geometry forward per unlabeled step. Equal rounds do not imply equal
  compute. Report wall-clock cost alongside final accuracy.

## Update diagnostics (enabled by the existing diag_geom switch)

Both experiment recipes and reference now write updates.csv and
client_updates.csv. Training rules, learning rates and checkpoint schema are
unchanged. Statistics run before loading aggregated weights into the global
model, use detached tensors, and require no additional model forward, labels,
RNG draw or local training step. There is extra tensor-reduction/I/O overhead;
its wall-clock cost has not yet been measured.

For each selected client define d_k = local parameters - round-start global
parameters, with p_k equal to its normalized FedAvg sample count. Let d be the
actual fused parameters minus the round-start global parameters.

- params_local_mean_norm = sum(p_k * ||d_k||): typical update size, weighted
  exactly as aggregation. params_global_norm = ||d|| and global_relative divides
  by the round-start parameter norm.
- params_disagreement_rms = sqrt(sum(p_k * ||d_k-d||^2)). Its relative version
  divides by sqrt(sum(p_k * ||d_k||^2)); compare this as well as absolute size
  so a uniform increase in step size is not mistaken for worse disagreement.
- params_retained_ratio = ||d|| / sum(p_k * ||d_k||): lower values indicate
  stronger directional cancellation, not necessarily worse generalization.
- client_updates.csv includes client ID, FedAvg weight, local step count, A
  ratio/precision, update norm, distance to the fused update and cosine to it.
  A negative cosine means opposition to the aggregate direction, not proof
  that this client's update is wrong. A precision is diagnostic GT only.
- BN running means and variances have separate bn_mean / bn_var statistics;
  these buffers are excluded from params. BN affine weights remain parameters.
  Integer counters are excluded. BN changes alone do not establish BN-caused
  accuracy loss; that would need an evaluation/control experiment.
- Zero-denominator norms/directions are NaN, not fabricated zeros. Acc and
  delta_acc use fractions (0.03 = 3 percentage points); the first delta is NaN.
  Round, phase, gate and LR allow alignment with ABC in dynamics.csv.

Inspect rounds 61-200 and phase transitions; relate update size, relative
disagreement, cancellation and BN changes to accuracy drops. These are
observational diagnostics, not a causal identification of client drift.
Trusted/LR=0.1 versus legacy/LR=0.15 differs in two mechanisms. A strict LR
comparison needs reference/LR=0.1 with the same code, seed and partition.
Historical accuracy curves remain useful, but their missing update statistics
cannot be reconstructed. No third diagnostic-only training job is launched.

## Verification status

12 lightweight tests (configuration isolation/CLI and LR/dynamics) passed;
Python compilation and Bash syntax checks passed. Tensor tests were added but
NOT run locally because this host lacks PyTorch and pytest. No local or server
training was started by this task. The server test command above is required
before interpreting the implementation as runtime-validated.
