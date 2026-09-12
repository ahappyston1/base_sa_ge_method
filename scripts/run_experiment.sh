#!/usr/bin/env bash
# One explicit experiment per process; GPU training is launched only by the user.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: bash scripts/run_experiment.sh {trusted_lr015|legacy_lr015_500|legacy_lr020|trusted|high_lr|reference} GPU [SEED] [--dry-run]" >&2
  exit 2
fi
EXPERIMENT="$1"
GPU_ID="$2"
shift 2
SEED_ID=7
if [[ $# -gt 0 && "$1" != --dry-run ]]; then
  SEED_ID="$1"
  shift
fi
DRY_RUN=0
if [[ $# -gt 0 && "$1" == --dry-run ]]; then
  DRY_RUN=1
  shift
fi
[[ $# -eq 0 ]] || { echo 'Unexpected arguments' >&2; exit 2; }
case "$EXPERIMENT" in trusted_lr015|legacy_lr015_500|legacy_lr020|trusted|high_lr|reference) ;; *) echo "Unknown experiment: $EXPERIMENT" >&2; exit 2 ;; esac
[[ "$GPU_ID" =~ ^[0-9]+$ && "$SEED_ID" =~ ^[0-9]+$ ]] || { echo 'GPU and seed must be nonnegative integers' >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="rev13_${EXPERIMENT}_s${SEED_ID}_$(date +%Y%m%d_%H%M%S)_$$"
CONFIG_PATH="$ROOT/configs/experiment_${EXPERIMENT}.yaml"
[[ -f "$CONFIG_PATH" ]] || { echo "Missing configuration: $CONFIG_PATH" >&2; exit 2; }
COMMAND=(bash "$ROOT/scripts/train.sh" --dataset CIFAR10 --alpha 0.1 --gpu_id "$GPU_ID"
  --config "$CONFIG_PATH" --run_id "$RUN_ID"
  --seed "$SEED_ID" --partition_seed 0 --sample_seed "$SEED_ID")
if [[ "$DRY_RUN" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
"${COMMAND[@]}"
