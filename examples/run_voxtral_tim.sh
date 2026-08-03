#!/usr/bin/env bash
set -euo pipefail

GPU_ID=${GPU_ID:-0}
OUTPUT_DIR=${OUTPUT_DIR:-runs}
CALIBRATION_SEED=${CALIBRATION_SEED:-20260729}

CUDA_VISIBLE_DEVICES="$GPU_ID" python implementations/voxtral/src/asr_experiment.py \
  --model voxtral-mini \
  --mode gptq \
  --quant-scope encoder \
  --wbits 2 --groupsize 128 \
  --nsamples 128 --calib-batch-size 16 \
  --seed "$CALIBRATION_SEED" --quantization-seed 20260729 \
  --frame-weighting task-fisher --propagated-clip-max 2 \
  --eval --eval-samples -1 \
  --output-dir "$OUTPUT_DIR" \
  --run-name "tim-voxtral-mini-w2-c${CALIBRATION_SEED}"

