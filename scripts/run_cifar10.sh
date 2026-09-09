#!/usr/bin/env bash
# CIFAR-10 上并行跑多个 α（默认 0.1 与 0.5），共用一张卡。
#   bash scripts/run_cifar10.sh --gpu_id 0
#   ALPHAS="0.1 0.5 1" WORKERS=6 bash scripts/run_cifar10.sh --gpu_id 0
#
# 单个 run 约占 3GB 显存，16GB 卡上并行 3~4 个没问题。
# 增广是 CPU 瓶颈，所以并行度实际受核数限制：WORKERS × 并行数 ≲ nproc。
# 同一时间戳作为 run_id，每个 α 一个独立目录 results/CIFAR10/runs/<run_id>_a<alpha>/。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU=0
ALPHAS="${ALPHAS:-0.1 0.5}"
NJOBS=$(echo ${ALPHAS} | wc -w)
# 默认按核数分配：每个 run 拿到约一半核数除以并行数，限制在 [4,16]
if [[ -z "${WORKERS:-}" ]]; then
  WORKERS=$(( $(nproc) / (2 * NJOBS) ))
  (( WORKERS < 4 )) && WORKERS=4
  (( WORKERS > 16 )) && WORKERS=16
fi
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_id|--gpu) GPU="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

STAMP="$(date +%Y%m%d_%H%M%S)"

echo "[run] run_id=${STAMP} alphas=${ALPHAS} workers=${WORKERS} gpu=${GPU}"
echo "[run] 产物目录 results/CIFAR10/runs/${STAMP}_a<alpha>/"

pids=()
names=()
outs=()
for ALPHA in ${ALPHAS}; do
  # 与 fl_runner._run_paths 的 %g 归一保持一致，两边指向同一个目录
  ALPHA_N="$(awk -v a="${ALPHA}" 'BEGIN{printf "%g", a}')"
  RUN_DIR="results/CIFAR10/runs/${STAMP}_a${ALPHA_N}"
  mkdir -p "${RUN_DIR}"
  out="${RUN_DIR}/launch.out"
  bash scripts/train.sh \
    --dataset CIFAR10 --alpha "${ALPHA}" --gpu_id "${GPU}" \
    --num_workers "${WORKERS}" --run_id "${STAMP}" \
    "${EXTRA[@]+"${EXTRA[@]}"}" > "${out}" 2>&1 &
  pids+=("$!")
  names+=("α=${ALPHA}")
  outs+=("${out}")
  echo "[run] α=${ALPHA} pid=$! -> ${out}"
  sleep 5  # 错开数据集校验与 CUDA 初始化
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[run] ${names[$i]} 完成"
  else
    echo "[run] ${names[$i]} 失败，见 ${outs[$i]}" >&2
    fail=1
  fi
done
exit "${fail}"
