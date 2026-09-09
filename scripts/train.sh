#!/usr/bin/env bash
# 单次 PPFPSL 实验。额外参数会原样传给 ppfpsl.py。
# 每次启动会带时间戳，不会覆盖正在跑的 α=0.1。
# 示例：
#   bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu_id 1
#   bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu_id 1 --run_id 20260907_170605
#   DATASET=CIFAR100 ALPHA=0.5 GPU=1 bash scripts/train.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${DATASET:-CIFAR10}"
ALPHA="${ALPHA:-0.1}"
GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/default_ppfpsl.yaml}"

# base 环境没有 torch，裸 python 会直接 ModuleNotFoundError。
# 优先用 $PYTHON，其次当前 python，最后找 conda 的 py39。
resolve_python() {
  local cand
  for cand in "${PYTHON:-}" python "$(conda info --base 2>/dev/null)/envs/py39/bin/python"; do
    [[ -z "$cand" ]] && continue
    if "$cand" -c 'import torch' >/dev/null 2>&1; then echo "$cand"; return 0; fi
  done
  echo "[train] 找不到装了 torch 的 python，请先 conda activate py39 或设 PYTHON=<解释器>" >&2
  return 1
}
PYTHON="$(resolve_python)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --gpu_id|--gpu) GPU="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    *) break ;;
  esac
done

echo "[train] dataset=${DATASET} alpha=${ALPHA} gpu=${GPU} config=${CONFIG} python=${PYTHON}"
"${PYTHON}" ppfpsl.py --config "${CONFIG}" --dataset "${DATASET}" --alpha "${ALPHA}" --gpu_id "${GPU}" "$@"
