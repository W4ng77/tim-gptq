#!/usr/bin/env bash
# E2 work-queue worker: atomic claims (mkdir) + locked per-GPU admission gate (MAXPROC arms/GPU).
# usage: bash E2_launch_wq.sh <gpu-id> <worker-tag> map:density:seed [...]   (density bf16-reference ignores seed)
set -uo pipefail
GPU="$1"; TAG="$2"; shift 2
MAXPROC=${MAXPROC:-2}
HOLD=${HOLD:-60}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
SRC=$ROOT/e2_mt_gptq.py
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate timgptq
export CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p "$ROOT/runs" "$ROOT/logs" "$ROOT/claims"
LOCK="$ROOT/lock.gpu$GPU"
GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
gpu_procs () {
  local n=0 pid
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$GPU_UUID" '$1==u{print $2}'); do
    if ps -p "$pid" -o args= 2>/dev/null | grep -q "e2_mt_gptq.py"; then n=$((n+1)); fi
  done
  echo "$n"
}
for spec in "$@"; do
  IFS=: read -r map density seed <<< "$spec"
  if [ "$density" = "bf16-reference" ]; then name="E2-bf16-reference"; else name="E2-${map}-${density}-c${seed}"; fi
  if [ -d "$ROOT/runs/$name" ]; then continue; fi
  if ! mkdir "$ROOT/claims/$name" 2>/dev/null; then continue; fi
  while true; do
    until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
    if [ "$(gpu_procs)" -lt "$MAXPROC" ]; then break; fi
    rmdir "$LOCK"; sleep 20
  done
  echo "START gpu=$GPU worker=$TAG $name $(date -u +%FT%TZ)" | tee -a "$ROOT/logs/timeline.log"
  t0=$(date +%s)
  python "$SRC" --map "$map" --density "$density" --seed "$seed" --quantization-seed 20260729 --nsamples 128 --wbits 3 --groupsize 128 --eval-samples -1 --gen-batch-size 16 --output-dir "$ROOT/runs" --run-name "$name" > "$ROOT/logs/$name.log" 2>&1 &
  PY=$!
  sleep "$HOLD"; rmdir "$LOCK"
  wait "$PY"; rc=$?
  t1=$(date +%s)
  echo "END   gpu=$GPU worker=$TAG $name rc=$rc wall_s=$((t1-t0)) $(date -u +%FT%TZ)" | tee -a "$ROOT/logs/timeline.log"
done
echo "WORKER DONE gpu=$GPU worker=$TAG $(date -u +%FT%TZ)" | tee -a "$ROOT/logs/timeline.log"
