#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/disk1/xiaoyanwang/code/base_sa_ge_bc_rec
cd "$ROOT"
LAUNCH_DIR="$ROOT/results/bc_rec_launches"
LOG="$LAUNCH_DIR/queue.log"
mkdir -p "$LAUNCH_DIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

WAIT_PIDS=(3446435 3609112 3609113 4148397 4148719)
EXPS=(bc_rec_unmasked bc_rec_masked bc_rec_audit)
GPU_ID=0
SEED_ID=7

log "queued ${EXPS[*]} after PIDs ${WAIT_PIDS[*]} gpu=$GPU_ID seed=$SEED_ID queue_pid=$$"

while true; do
  alive=()
  for p in "${WAIT_PIDS[@]}"; do
    if kill -0 "$p" 2>/dev/null; then
      alive+=("$p")
    fi
  done
  if [[ ${#alive[@]} -eq 0 ]]; then
    log "predecessor PIDs finished"
    break
  fi
  log "waiting predecessors: still alive ${alive[*]}"
  sleep 60
done

while true; do
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF' || true)
  if [[ -z "${busy// }" ]]; then
    log "GPU${GPU_ID} compute apps empty; launching queue"
    break
  fi
  log "waiting GPU free; compute apps: ${busy//$'\n'/, }"
  sleep 60
done

export PYTHON=/home/disk1/xiaoyanwang/anaconda3/envs/py39/bin/python
export PATH="/home/disk1/xiaoyanwang/anaconda3/envs/py39/bin:$PATH"

for exp in "${EXPS[@]}"; do
  exp_log="$LAUNCH_DIR/${exp}.log"
  log "START $exp gpu=$GPU_ID seed=$SEED_ID log=$exp_log"
  set +e
  bash "$ROOT/scripts/run_experiment.sh" "$exp" "$GPU_ID" "$SEED_ID" >> "$exp_log" 2>&1
  rc=$?
  set -e
  log "END $exp rc=$rc"
  if [[ $rc -ne 0 ]]; then
    log "STOP queue because $exp failed with rc=$rc"
    exit "$rc"
  fi
done

log "all three experiments finished successfully"
