# REV12: geometry, middle-round stability and independent experiments

## What the completed runs establish

Source: geom_cur_s7_a0.1 metrics.csv / diag.csv, rounds 201–270.
The [0.95,0.99) band contains 1,725,226 visits (12.6676% of all unlabeled visits).
Geometry opposes 80,033 visits (0.58765% of all visits; 4.638% of the band).
Band precision before filtering is 66.8846%; support precision is 67.8588%.
The actual precision improvement among retained band samples is about 0.974pp,
not the support–opposition gap of 21pp. Opposition still contains 46.8582%
correct predictions. The full A bucket has visit-weighted precision 84.0754%
in this window; band-support precision is not A precision.
These figures are descriptive, not causal, and count repeated sample visits.

## Changes

- METHOD_REV=12; start fresh. Probability ceiling=0.99 and complete gate retained.
- Removed pp_unknown_b_conf and its B-score override. Invalid prototypes still
  cannot generate artificial margins. Unknown geometry skips the margin test,
  while neutral margin reliability participates in the mixed score as in the
  earlier rule. Missing-geometry C=0 does not prove all other unknown-geometry
  samples were absent; a new baseline is preferable for strict pairing.
- lr_mid_factor=1 is the default. Setting 0.5 leaves rounds 1–60 unchanged,
  cosine-ramps the multiplier over rounds 61–90, and retains 0.5 thereafter.
  The LR floor remains 0.0001, applied after the multiplier. No restart or
  increase of the initial LR. This tests local-update amplitude, not a proven
  remedy for drift or BN problems.
- pp_b_conf_rescue=0 remains default. Setting 1 expands B admission using
  confidence, while retaining the B margin check and mixed reliability weight.
  This tests the score-only rejected C subgroup, not all C samples.
- geometry_audit.csv adds Phase2/3 strata: round, client, predicted class,
  confidence bin, support/opposition/unknown, visit counts, correct counts,
  selected-A counts, selected-A correct counts, A weight and wrong A weight.
  Bins cover [0,.6), [.6,.9), [.9,.95), [.95,.97), [.97,.99),
  [.99,.995), [.995,1] including the old diagnostic blind spot.
  Support/opposition uses the actual per-sample A margin threshold for that
  round; thresholds change through gate, so compare within rounds/strata first.
  The file supports matched-client/class/confidence comparisons and testing
  whether A weighting favors correct samples. It is not a causal proof and
  excludes Phase1, where geometry is not used for routing.
- dynamics.csv continues recording effective coefficient masses and losses
  including aux/lambda. All GT-derived quantities are read-only diagnostics.
  Model behavior can change indirectly in experimental runs. Diagnostics use
  no random sampling and do not backpropagate.

## Server validation before full runs

```bash
python -m pytest tests/test_ppfpsl_routing.py tests/test_geometry_audit.py tests/test_training_dynamics.py -q
```

Local validation: 8 standard-library tests passed; Python compilation and git
whitespace checks passed. PyTorch/pytest unavailable locally, so tensor tests
and full training were not run here. No claim of all tests passing.

## Three paired runs (same code, different single controls)

Use available GPU numbers, new run IDs, no --resume. The prior REV11 results
remain historical references; the new baseline gives the stratified audit.

```bash
# Clean baseline + richer geometry diagnostics
bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu_id 0 --run_id rev12_base_s7 --seed 7 --partition_seed 0 --sample_seed 7 --num_workers 16 --max_rounds 300 --lr_mid_factor 1 --lr_min 0.0001 --pp_b_conf_rescue 0

# Middle-round LR only
bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu_id 1 --run_id rev12_midlr_s7 --seed 7 --partition_seed 0 --sample_seed 7 --num_workers 16 --max_rounds 300 --lr_mid_factor 0.5 --lr_min 0.0001 --pp_b_conf_rescue 0

# B admission only
bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu_id 2 --run_id rev12_brescue_s7 --seed 7 --partition_seed 0 --sample_seed 7 --num_workers 16 --max_rounds 300 --lr_mid_factor 1 --lr_min 0.0001 --pp_b_conf_rescue 1
```

Evaluate middle-round accuracy means and step-to-step changes together with
last-10/last-30 accuracy. A smoother but worse model is not a success. More B
samples alone is not a success. Compare correctly/incorrectly labeled A mean
weights in matched strata (correct mass = total mass - wrong mass), alongside
coverage. Do not count repeated visits as independent observations or use GT
precision to control routing or stages. No existing result supports promising
recovery of the full remaining seven percentage points.
