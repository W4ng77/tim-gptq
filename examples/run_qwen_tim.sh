#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
OUTPUT_DIR=${OUTPUT_DIR:-runs}
CALIBRATION_SEED=${CALIBRATION_SEED:-20260729}

CUDA_VISIBLE_DEVICES="$GPU_ID" python implementations/whisper_qwen/src/asr_experiment.py \
  --model qwen-0.6b \
  --mode gptq \
  --quant-scope text-backbone \
  --sequence-support full \
  --wbits 3 --groupsize 128 \
  --nsamples 128 --calib-batch-size 16 \
  --seed "$CALIBRATION_SEED" --quantization-seed 20260729 \
  --frame-weighting task-fisher --propagated-clip-max 2 \
  --eval --eval-samples -1 \
  --output-dir "$OUTPUT_DIR" \
  --run-name "tim-qwen06-w3-c${CALIBRATION_SEED}"

