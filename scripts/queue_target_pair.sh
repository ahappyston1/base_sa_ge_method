#!/usr/bin/env bash
# Wait until GPU0 has enough free VRAM for two ~3.1GiB jobs, then launch
# labelhead + separation in parallel. Does not touch other training PIDs.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-/home/disk1/xiaoyanwang/anaconda3/envs/py39/bin/python}"
STAMP="${1:?stamp required}"
NEED_FREE="${NEED_FREE_MIB:-6500}"
LOGDIR="$ROOT/results/target_experiment_launches"
QLOG="$ROOT/target_experiments_queue.log"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }

log "queue pair waiting for free VRAM>=${NEED_FREE}MiB (stamp=$STAMP)"

while true; do
  free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  nproc="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]' || true)"
  # refuse to launch if either job already running
  if pgrep -af "bc_targets_(labelhead|separation)_${STAMP}" | grep -v grep | grep -q ppfpsl; then
    log "abort: labelhead/separation already present for stamp=$STAMP"
    exit 1
  fi
  log "waiting: free=${free}MiB gpu_procs=${nproc}"
  if [[ "$free" -ge "$NEED_FREE" ]]; then
    # require free stable for one more check (avoid race during alloc)
    sleep 15
    free2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
    if [[ "$free2" -ge "$NEED_FREE" ]]; then
      log "capacity OK (free=${free2}MiB); launching labelhead + separation"
      break
    fi
    log "free dipped to ${free2}MiB; keep waiting"
  fi
  sleep 60
done

for mode in labelhead separation; do
  RID="bc_targets_${mode}_${STAMP}"
  nohup "$PY" -u ppfpsl.py \
    --config "configs/experiment_bc_targets_${mode}.yaml" \
    --dataset CIFAR10 --alpha 0.1 --gpu_id 0 \
    --run_id "$RID" \
    > "$LOGDIR/${RID}.log" 2>&1 &
  log "started ${mode} PID=$! run_id=$RID"
done
log "queued pair launched"
