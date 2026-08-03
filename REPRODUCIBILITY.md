# Reproducibility map

## Method components

| Paper object | Code entry point |
|---|---|
| Deployment Map and Qwen sequence Support | `implementations/whisper_qwen/src/asr_experiment.py` |
| Task-loss pullback collection | `implementations/*/src/frame_weighting.py` |
| Bounded mean-one I-projection | `normalize_task_fisher_weights` in `frame_weighting.py` |
| Weighted Gram accumulation | `implementations/*/src/gptq.py` and `quantization_runtime.py` |
| Marginal-preserving permutation | `permute_calibration_token_weights` in `frame_weighting.py` |
| Prompt/response-stratified permutation | Whisper/Qwen `frame_weighting.py` |
| Fixed grouped-GPTQ solver | `implementations/*/src/gptq.py` |

The paper's TIM-GPTQ estimator uses `--mode gptq --frame-weighting
task-fisher --propagated-clip-max 2`. The option name
`--propagated-clip-max` is retained for artifact compatibility; in task-Fisher
mode it is the final upper bound of the ratio-preserving I-projection, not a
clip-then-renormalize operation.

## Identification analyses

| Claim or contrast | Analysis program |
|---|---|
| Paired multi-draw efficacy | `analysis/task_fisher/paired_multidraw_bootstrap.py` |
| Crossed multi-draw efficacy | `analysis/task_fisher/crossed_multidraw_bootstrap.py` |
| Map × Support × Density | `analysis/task_fisher/paired_multidraw_factorial.py` and `crossed_multidraw_factorial.py` |
| Weight-state correspondence | `analysis/revision/nested_permutation_bootstrap.py` |
| Bit × target × Density | `analysis/overnight/crossed_bit_target_interaction_bootstrap.py` |
| External paired targets | `analysis/overnight/paired_external_bootstrap.py` |
| Context/entity contrasts | `analysis/overnight/paired_entity_bootstrap.py` and crossed variants |

All bootstrap scripts accept run directories as arguments and write JSON plus
a human-readable Markdown summary. Paths are supplied at invocation time; no
machine-specific run root is embedded in the anonymous package.

## Determinism and fairness

- `--seed` controls calibration-example sampling.
- `--quantization-seed` controls model loading and quantization RNG.
- A draw is an independently calibrated artifact, not a rerun of evaluation.
- Uniform and TIM comparisons must match model, scope, bit width, group size,
  calibration IDs, quantization seed, and evaluation rows.
- Do not combine partial outputs. Require terminal `status.json` and exact
  example-ID alignment before inference.
- WER is primary. Capped WER is reserved for declared collapse contrasts.

## Data

Calibration uses LibriSpeech `train.clean.100`. Internal evaluation uses
LibriSpeech test-other, VoxPopuli English, and GigaSpeech. External evaluation
uses ContextASR-Bench, a voice-qualified ProfASR-Bench manifest, and five
FLEURS languages. The repository includes a manifest schema example but not
audio or benchmark text. Materialize source datasets under their own licenses
and replace each manifest's `audio_path` with a local or relative path.

