#!/usr/bin/env bash
# E3 work-queue worker: atomic claims + locked per-GPU admission gate (MAXPROC arms per GPU).
# usage: bash E3_launch_wq.sh <gpu-id> <worker-tag> fmt:density:seed [...]   (fmt = mxfp4|nvfp4|bf16)
set -uo pipefail
GPU="$1"; TAG="$2"; shift 2
MAXPROC=${MAXPROC:-2}
HOLD=${HOLD:-75}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
SRC=$REPO/implementations/whisper_qwen/src/asr_experiment.py
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate timgptq
export CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p "$ROOT/runs" "$ROOT/logs" "$ROOT/claims"
LOCK="$ROOT/lock.gpu$GPU"
GPU_UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
COMMON="--model qwen-0.6b --quant-scope text-backbone --sequence-support full \
 --nsamples 128 --calib-batch-size 16 --quantization-seed 20260729 \
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
  IFS=: read -r fmt density seed <<< "$spec"
  if [ "$fmt" = "bf16" ]; then
    name="E3-bf16-reference"; MODE="--mode fp16 --wbits 4 --groupsize 128"; SEEDARG="--seed 20260729"
  else
    case "$density" in
      uniform) MODE="--mode gptq+seqcal --wbits 4 --groupsize 128 --fp4-format $fmt" ;;
      tim)     MODE="--mode gptq+seqhess --propagated-clip-max 2 --wbits 4 --groupsize 128 --fp4-format $fmt" ;;
      *) echo "unknown density $density"; exit 2 ;;
    esac
    name="E3-${fmt}-${density}-qwen06-c${seed}"; SEEDARG="--seed $seed"
  fi
  if [ -d "$ROOT/runs/$name" ]; then continue; fi
  if ! mkdir "$ROOT/claims/$name" 2>/dev/null; then continue; fi
  while true; do
    until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
    if [ "$(gpu_procs)" -lt "$MAXPROC" ]; then break; fi
    rmdir "$LOCK"; sleep 20
  done
  echo "START gpu=$GPU worker=$TAG $name $(date -u +%FT%TZ)" | tee -a "$ROOT/logs/timeline.log"
  t0=$(date +%s)
  python "$SRC" $COMMON $MODE $SEEDARG --run-name "$name" > "$ROOT/logs/$name.log" 2>&1 &
  PY=$!
  sleep "$HOLD"; rmdir "$LOCK"
  wait "$PY"; rc=$?
  t1=$(date +%s)
  echo "END   gpu=$GPU worker=$TAG $name rc=$rc wall_s=$((t1-t0)) $(date -u +%FT%TZ)" | tee -a "$ROOT/logs/timeline.log"
done
echo "WORKER DONE gpu=$GPU worker=$TAG $(date -u +%FT%TZ)" | tee -a "$ROOT/logs/timeline.log"
