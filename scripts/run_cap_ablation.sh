#!/usr/bin/env bash
# 「早期 A 限流并渐进放开」的两种子配对对照（CIFAR-10, α=0.1）。
#   bash scripts/run_cap_ablation.sh --gpu_id 0
#
# 设计：随机区组，每个区组内基线与限流只差 cap，其余（模型种子 / 划分种子 /
# 在线抽样种子 / num_workers）完全一致。num_workers 必须一致，否则增广随机流
# 不同，配对失效。
#
#   区组 1  seed=7  partition=0   基线 = 已有 run 20260908_195152_a0.1（复用，
#                                 该 run 与当前代码同版本：fl_runner.py 改于
#                                 19:42，run 起于 19:52）
#                                 限流 = capA_s7
#   区组 2  seed=17 partition=1   基线 = base_s17，限流 = capA_s17
#
# 限流规则：Phase1 的 A 占比上限 0.25；Phase2 按 gate 从 0.25 线性放开到 1.0
# （_cap_bucket 中 cap=1.0 等价于不限制，且插值连续，不会像 cap=0 那样在边界
# 上翻转语义）；Phase3 不受限。注意 Phase2 前约 23/39 轮 cap 实际生效，本实验
# 不是只改 Phase1。
#
# 全程零代码改动：cap 由命令行覆盖 YAML，基线与限流共用 default_ppfpsl.yaml。
# 这些 run_id 不是时间戳格式，续训时需显式带上 --run_id。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU=0
WORKERS="${WORKERS:-16}"
CAP_P1="${CAP_P1:-0.25}"
CAP_P23="${CAP_P23:-1.0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_id|--gpu) GPU="$2"; shift 2 ;;
    *) shift ;;
  esac
done

echo "[cap-ablation] workers=${WORKERS} cap_p1=${CAP_P1} cap_p2=${CAP_P23} gpu=${GPU}"
echo "[cap-ablation] 区组1 基线复用 results/CIFAR10/runs/20260908_195152_a0.1"

pids=()
names=()
outs=()

launch() {  # run_id seed partition 附加参数...
  local rid="$1" seed="$2" part="$3"; shift 3
  local dir="results/CIFAR10/runs/${rid}_a0.1"
  mkdir -p "${dir}"
  bash scripts/train.sh --dataset CIFAR10 --alpha 0.1 --gpu_id "${GPU}" \
    --num_workers "${WORKERS}" --run_id "${rid}" \
    --seed "${seed}" --partition_seed "${part}" --sample_seed "${seed}" \
    "$@" > "${dir}/launch.out" 2>&1 &
  pids+=("$!")
  names+=("${rid}")
  outs+=("${dir}/launch.out")
  echo "[cap-ablation] ${rid} pid=$! -> ${dir}/launch.out"
  sleep 5
}

launch capA_s7  7  0 --pp_a_ratio_cap_p1 "${CAP_P1}" --pp_a_ratio_cap "${CAP_P23}"
launch base_s17 17 1
launch capA_s17 17 1 --pp_a_ratio_cap_p1 "${CAP_P1}" --pp_a_ratio_cap "${CAP_P23}"

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[cap-ablation] ${names[$i]} 完成"
  else
    echo "[cap-ablation] ${names[$i]} 失败，见 ${outs[$i]}" >&2
    fail=1
  fi
done
exit "${fail}"
