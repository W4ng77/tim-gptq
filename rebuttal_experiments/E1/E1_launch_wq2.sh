#!/usr/bin/env bash
# E1-mini work-queue worker v2: atomic claims (mkdir) + a per-GPU LOCKED admission gate.
# A worker admits a new arm only while holding the GPU lock and only if fewer than MAXPROC E1
# python processes are on the GPU; it keeps the lock for HOLD seconds after launching so the
# new process is visible to nvidia-smi before the next admission check.
# usage: bash E1_launch_wq2.sh <gpu-id> <worker-tag> arm:seed [...]
set -uo pipefail
GPU="$1"; TAG="$2"; shift 2
MAXPROC=${MAXPROC:-2}
HOLD=${HOLD:-75}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
SRC=$REPO/implementations/whisper_qwen/src/asr_experiment.py
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate timgptq
export CUDA_VISIBLE_DEVICES="$GPU"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
mkdir -p "$ROOT/runs" "$ROOT/logs" "$ROOT/claims"
LOCK="$ROOT/lock.gpu$GPU"
GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
COMMON="--model qwen-0.6b --quant-scope text-backbone --sequence-support full \
 --wbits 3 --groupsize 128 --nsamples 128 --calib-batch-size 16 \
 --quantization-seed 20260729 \
 --eval --eval-samples 3000 --datasets librispeech-other,voxpopuli,gigaspeech --eval-batch-size 8 \
 --output-dir $ROOT/runs"
gpu_procs () {
  local n=0 pid
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F', ' -v u="$GPU_UUID" '$1==u{print $2}'); do
    if ps -p "$pid" -o args= 2>/dev/null | grep -q "asr_experiment.py"; then n=$((n+1)); fi
  done
  echo "$n"
}
for spec in "$@"; do
  arm="${spec%%:*}"; seed="${spec##*:}"
  case "$arm" in
    uniform)    MODE="--mode gptq+seqcal" ;;
    tim)        MODE="--mode gptq+seqhess --propagated-clip-max 2" ;;
    rsq-native) MODE="--mode gptq+rsqattn --rsq-score-normalization native --rsq-min-value 0.1 --rsq-max-value 1.0" ;;
    rsq-kl)     MODE="--mode gptq+rsqattn --rsq-score-normalization bounded-kl --propagated-clip-max 2" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  name="E1-${arm}-qwen06-w3-c${seed}"
  if [ -d "$ROOT/runs/$name" ]; then continue; fi
  if ! mkdir "$ROOT/claims/$name" 2>/dev/null; then continue; fi
  # locked admission gate
  while true; do
    until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
    if [ "$(gpu_procs)" -lt "$MAXPROC" ]; then break; fi
    rmdir "$LOCK"; sleep 20
  done
  echo "START gpu=$GPU worker=$TAG $name $(date -u +%FT%TZ)" | tee -a "$ROOT/logs/timeline.log"
  t0=$(date +%s)
  python "$SRC" $COMMON $MODE --seed "$seed" --run-name "$name" > "$ROOT/logs/$name.log" 2>&1 &
  PY=$!
  sleep "$HOLD"; rmdir "$LOCK"
  wait "$PY"; rc=$?
  t1=$(date +%s)
  echo "END   gpu=$GPU worker=$TAG $name rc=$rc wall_s=$((t1-t0)) $(date -u +%FT%TZ)" | tee -a "$ROOT/logs/timeline.log"
done
echo "WORKER DONE gpu=$GPU worker=$TAG $(date -u +%FT%TZ)" | tee -a "$ROOT/logs/timeline.log"
