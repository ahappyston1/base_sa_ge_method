#!/usr/bin/env bash
# CIFAR-10 上连续跑 α=0.1 与 α=0.5。
#   bash scripts/run_cifar10.sh --gpu_id 0
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_id|--gpu) GPU="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

for ALPHA in 0.1 0.5; do
  echo "========== CIFAR10 α=${ALPHA} =========="
  bash scripts/train.sh --dataset CIFAR10 --alpha "${ALPHA}" --gpu_id "${GPU}" "${EXTRA[@]+"${EXTRA[@]}"}"
done
