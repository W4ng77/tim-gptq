# E2 — out-of-domain existence check of the TIM Map x Density decomposition on MT (frozen protocol)

Frozen 2026-09-15 before any E2 artifact was evaluated. Claim scope: an out-of-domain
existence check of the Map/Density decomposition on one non-ASR sequence-to-sequence
task; not a general MT quantization claim. No languages, models, bit widths, or
follow-up arms will be added after results are seen.

## 1. Fixed configuration

| Field | Value |
|---|---|
| Model | Qwen/Qwen3-0.6B (snapshot c1899de), bf16. NOTE: the requested "Qwen3-0.6B-Instruct" does not exist on the Hub; Qwen3-0.6B is the post-trained instruction/chat model of that size (thinking mode switchable), used here with `enable_thinking=False`. |
| Task / data | FLORES-200 eng_Latn -> deu_Latn from the public release tarball (dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz). Calibration pool = dev (997 sentence pairs). Evaluation = devtest (1,012 pairs), all rows, disjoint from dev by construction. File sha256 recorded in every run's config.json. |
| Quantized component | all 28 decoder layers' linear modules (q/k/v, o, gate/up, down); embeddings, lm_head and norms untouched |
| Bits / group / solver | W3, G128, the paper's grouped GPTQ (`gptq.Helper`, `run_gptq`), percdamp 0.01, blocksize 128, no act-order, asymmetric per-group with integer zero point (`quant.py`), sequential quantized-so-far stream, batch 1 per sample; imported unchanged |
| Support (all arms) | prompt + teacher-forced target rows (the full sequence); loss labels mask the prompt |
| Calibration draws | 128 pairs sampled from dev with `random.Random(seed).sample`, seeds 20260729 / 20260730 / 20260731 (draw 1/2/3); indices serialised in calibration.json |
| Quantization seed | 20260729 for all arms |
| Evaluation Map (all arms) | deployment chat template: user turn "Translate the following English text into German. Output only the German translation.\n\n{src}", `add_generation_prompt=True`, `enable_thinking=False`; greedy decoding (`do_sample=False`, `num_beams=1`), max 256 new tokens, batch 16 with left padding, EOS = <\|im_end\|> or <\|endoftext\|>; any `</think>` block stripped; hypothesis otherwise unmodified |
| Metrics | sacrebleu 2.6.0: chrF++ (`corpus_chrf`, word_order=2) and BLEU (`corpus_bleu`, 13a), corpus level; per draw and draw mean |
| Reference | one BF16 native pass (no calibration), reported as context only, not one of the four conditions |
| Determinism | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `HF_HUB_OFFLINE=1`; two arms per GPU (results are per-process deterministic; wall time not single-tenant) |

## 2. Arms (2 x 2 factorial, 3 draws each = 12 artifacts)

| Map (calibration prompt) | Density | Label |
|---|---|---|
| minimal/raw: `English: {src}\nGerman: {tgt}<\|im_end\|>` (no chat template) | Uniform GPTQ (w = 1) | minimal/uniform |
| minimal/raw | TIM-GPTQ | minimal/tim |
| deployment: the same chat template as evaluation, with the reference as the assistant turn + <\|im_end\|> | Uniform GPTQ | deployment/uniform |
| deployment | TIM-GPTQ | deployment/tim |

TIM density = the paper's Qwen text-backbone estimator: one teacher-forced forward/backward
per calibration pair; g_t = mean_h (dL/dh_t)^2 at each layer's `mlp.down_proj` output;
ratio-preserving KL/I-projection to unit mean with final bounds [1e-4, 2] (code-identical
`normalize_task_fisher_weights`); weights applied to all linear groups of that layer via
`Helper.add_batch(token_weights=...)`. Uniform = same rows, w = 1. Within a draw all four
arms share the 128 dev indices; the two Maps differ only in the prompt text around the same
source/target pair; the two Densities differ only in w.

## 3. Pre-specified reporting

- chrF++ and BLEU per arm, per draw, and draw mean; BF16 reference.
- Map contrast under Uniform: deployment/uniform - minimal/uniform, per draw and mean; sign
  consistency across the three draws.
- Density contrast under each Map: tim - uniform, per draw and mean; sign consistency.
- Also reported: Map contrast under TIM, and a crossed draw x sentence bootstrap (1,000
  reps, seed 20260731) of the draw-mean corpus chrF++ and BLEU differences for those
  contrasts, as descriptive intervals.
- Integrity: identical dev indices across the four arms of a draw; identical devtest rows;
  status.json completed for all 12; empty-hypothesis counts; config diffs limited to
  map/density/seed/run-name.

## 4. Execution

- Script `e2_mt_gptq.py`; launcher `E2_launch_wq.sh` (atomic claims, locked admission gate,
  2 arms per GPU); run directories `E2/runs/E2-{map}-{density}-c{seed}` and
  `E2/runs/E2-bf16-reference`.
- GPU 0: minimal/uniform x3, minimal/tim x3; GPU 1: deployment/uniform x3, deployment/tim x3;
  BF16 reference on whichever GPU frees first.
- A smoke test (4 calibration pairs, 16 devtest rows, scratch directory) verifies only that
  the pipeline runs; it is not an artifact.
