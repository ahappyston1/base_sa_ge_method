#!/usr/bin/env bash
# REV11 几何控制消融（CIFAR-10, α=0.1, 300 轮）。
#   bash scripts/run_geom_ablation.sh --gpu_id 0
#
# 三个 run 共用同一划分与同一套种子，不改学习率：
#   seed=7  partition_seed=0  sample_seed=7  num_workers 一致
#
#   1. geom_diag_s7     诊断基线：关上本次新控制，保留 diag.csv
#   2. geom_cur_s7      当前改动版：YAML 默认（ceiling=0.99 / 完整 gate / 未知几何按置信度）
#   3. geom_noboost_s7  当前版上去掉 Phase3 boost
#
# 若这 3 个显示当前版有效，再补第 4 个（默认不跑）：
#   ONLY_UNKNOWN=1 bash scripts/run_geom_ablation.sh --gpu_id 0
#   → geom_nounk_s7   只关未知几何回退
#
# 汇总：python scripts/summarize_geom_ablation.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU=0
WORKERS="${WORKERS:-16}"
SEED="${SEED:-7}"
PARTITION="${PARTITION:-0}"
ONLY_UNKNOWN="${ONLY_UNKNOWN:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_id|--gpu) GPU="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --partition_seed) PARTITION="$2"; shift 2 ;;
    --only-unknown) ONLY_UNKNOWN=1; shift ;;
    *) shift ;;
  esac
done

echo "[geom-ablation] gpu=${GPU} workers=${WORKERS} seed=${SEED} partition=${PARTITION} only_unknown=${ONLY_UNKNOWN}"
echo "[geom-ablation] 学习率沿用 YAML，不在本组覆盖"

pids=()
names=()
outs=()

launch() {  # run_id 附加参数...
  local rid="$1"; shift
  local dir="results/CIFAR10/runs/${rid}_a0.1"
  mkdir -p "${dir}"
  bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu_id "${GPU}" \
    --num_workers "${WORKERS}" --run_id "${rid}" \
    --seed "${SEED}" --partition_seed "${PARTITION}" --sample_seed "${SEED}" \
    --diag_geom 1 \
    "$@" > "${dir}/launch.out" 2>&1 &
  pids+=("$!")
  names+=("${rid}")
  outs+=("${dir}/launch.out")
  echo "[geom-ablation] ${rid} pid=$! -> ${dir}/launch.out"
  sleep 5
}

if [[ "${ONLY_UNKNOWN}" == "1" ]]; then
  launch geom_nounk_s7 \
    --pp_tau_ceiling 0.99 --pp_complete_gate 1 --pp_unknown_b_conf 0
else
  launch geom_diag_s7 \
    --pp_tau_ceiling 0 --pp_complete_gate 0 --pp_unknown_b_conf 0
  # 当前改动版：不覆盖 ceiling/gate/unknown/boost，走 default_ppfpsl.yaml
  launch geom_cur_s7
  launch geom_noboost_s7 \
    --pp_phase3_geom_boost 0
fi

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[geom-ablation] ${names[$i]} 完成"
  else
    echo "[geom-ablation] ${names[$i]} 失败，见 ${outs[$i]}" >&2
    fail=1
  fi
done
exit "${fail}"
