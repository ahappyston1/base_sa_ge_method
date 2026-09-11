#!/usr/bin/env bash
# One explicit experiment per process; GPU training is launched only by the user.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/run_experiment.sh {trusted|high_lr|reference} GPU [SEED]" >&2
  exit 2
fi
EXPERIMENT="$1"
GPU_ID="$2"
SEED_ID="${3:-7}"
case "$EXPERIMENT" in trusted|high_lr|reference) ;; *) echo "Unknown experiment: $EXPERIMENT" >&2; exit 2 ;; esac
[[ "$GPU_ID" =~ ^[0-9]+$ && "$SEED_ID" =~ ^[0-9]+$ ]] || { echo 'GPU and seed must be nonnegative integers' >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="rev13_${EXPERIMENT}_s${SEED_ID}_$(date +%Y%m%d_%H%M%S)_$$"
bash "$ROOT/scripts/train.sh" --dataset CIFAR10 --alpha 0.1 --gpu_id "$GPU_ID" \
  --config "configs/experiment_${EXPERIMENT}.yaml" --run_id "$RUN_ID" \
  --seed "$SEED_ID" --partition_seed 0 --sample_seed "$SEED_ID"
