# E1-mini — matched RSQ attention-concentration Density control (frozen protocol)

Frozen: 2026-09-15, before any E1 artifact was evaluated. Nothing below is changed
after results are seen. Purpose: isolate the same-axis overlap raised by Reviewer JkfK
(token-importance weighting of the GPTQ Gram) while holding TIM's solver geometry, Map,
Support, calibration sample, bit width and evaluation fixed. This arm is NOT "full RSQ":
no rotation, no RSQ calibration corpus, no weight-clipping search. In every log and
table it is called the **matched RSQ attention-concentration Density control**.

## 1. Fixed configuration (all four arms)

| Field | Value | Source |
|---|---|---|
| Model | Qwen/Qwen3-ASR-0.6B (`qwen-0.6b`), snapshot 5eb1441 | paper Table 1 / App. E.1 |
| Quantized component | `--quant-scope text-backbone` (thinker text layers, groups q/k/v, o, gate/up, down) | paper Sec. 5 |
| Bits / group | W3, G128 (`--wbits 3 --groupsize 128`) | paper Sec. 5, examples/run_qwen_tim.sh |
| Solver | grouped GPTQ, percdamp 0.01, blocksize 128, no act-order, asymmetric int zero-point | code defaults, unchanged |
| Calibration Map | deployment inference template (`model._build_text_prompt("", "English")`), instantiated by `include_labels=True` | quantization_runtime._qwen_make_calibration_data |
| Support | `--sequence-support full`: prompt + teacher-forced transcript + eos rows | paper App. D.1 "deployment template / full" |
| Calibration corpus | LibriSpeech clean/train.100 (openslr/librispeech_asr, snapshot 71cacbf), 128 utterances, deterministic shuffle by `--seed` | paper App. C.1 |
| Calibration draws | `--seed` 20260729, 20260730, 20260731 (draw 1, 2, 3) | paper App. C "frozen seeds 20260729–20260731" |
| Quantization seed | `--quantization-seed 20260729` | examples/run_qwen_tim.sh |
| Evaluation | internal-3: librispeech-other, voxpopuli, gigaspeech from hf-audio/open-asr-leaderboard (snapshot b6bdcd0), `--eval-samples 3000` (first 3,000 rows per split in parquet order via `islice`; gives 2,939 / 1,842 / 3,000 rows, matching the paper's stated counts), `--eval-batch-size 8`; raw corpus WER; macro = equal-domain mean | paper App. E.4 ("2,939/1,842/3,000"), Sec. 5 (13,401 = five splits capped at 3,000) |
| Determinism | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `HF_HUB_OFFLINE=1`, one GPU per arm, arms run sequentially | launcher |
| Code | tim-gptq @ 12d30a5 + the E1 patch (`gptq+rsqattn` mode); every run records `environment.json` (source-tree sha256) | git diff in this directory |

Calibration IDs: the artifact serializes them per run (`calibration.json`). The paper's
original run directories are not on this machine, so ID identity with the original D.3
artifacts cannot be verified byte-for-byte; identity across the four E1 arms of the
same draw IS verified after the runs (same seed -> same deterministic shuffle -> same
128 IDs). Uniform and TIM are therefore **re-run here as reconstructed baselines**, which
the task spec permits when exact-match artifacts are unusable (absent).

## 2. Arms

| Arm | `--mode` | Gram row weight w_t | Notes |
|---|---|---|---|
| Uniform (reconstructed) | `gptq+seqcal` | 1 | deployment template + full rows; uniform Gram. `gptq+seqcal` is the code's only path that instantiates the deployment template with uniform weights (plain `gptq` uses the minimal audio-token prompt). |
| TIM (reconstructed) | `gptq+seqhess --propagated-clip-max 2` | KL/I-projection of g_t = mean_h (dL_TF/dh_t)^2 at each layer's `mlp.down_proj` output, FP model, one backward per utterance; final bounds **[1e-4, 2]**, unit mean | This is the code path used for the paper's Qwen text-backbone TIM arm (`sequence_hessian.hessian = 2 X^T diag(w) X`). NOTE: the paper text (App. B) states a = 1e-3; the Qwen code path uses 1e-4 (`_normalize_qwen_sequence_token_weights`). The bounded-KL RSQ arm uses the same function so that the projection is code-identical to TIM. |
| RSQ-score/native | `gptq+rsqattn --rsq-score-normalization native --rsq-min-value 0.1 --rsq-max-value 1.0` | r = clamp(minmax(s) * (1 - 0.1) + 0.1, 0.1, 1); w = r / mean(r) | RSQ's own scaling (`normalize_weight`, then the unit-mean renormalization in the official `add_batch`). |
| RSQ-score/bounded-KL | `gptq+rsqattn --rsq-score-normalization bounded-kl --propagated-clip-max 2` | KL/I-projection of the raw score s to unit mean, bounds [1e-4, 2] (`_normalize_qwen_sequence_token_weights`) | identical raw score as the native arm; identical projection code as TIM. |

Shared rows for both RSQ arms: same Map, Support, calibration IDs, seeds, GPTQ, and
evaluation as Uniform/TIM. The only change is the Gram row weight.

## 3. RSQ attention-concentration score: definition and implementation choice

Source: Sung, Yadav, Li, Yoon, Bansal, "RSQ: Learning from Important Tokens Leads to
Better Quantized LLMs", arXiv:2503.01820, Sec. 4.3, and the official repository
github.com/ylsung/rsq (`fake_quant/input_weighting_module.py::OriginalAttentionWeighting`,
`fake_quant/gptq_utils.py::GPTQ.add_batch`, `fake_quant/configs/input_weighting/attncon.yaml`,
`scripts/run_rsq.sh`).

- Paper: "To calculate the attention concentration of the j-th token, we sum over the
  second dimension of A, and further sum the scores of every head together...
  R = {sum_{m,i} A_{mij} : 1 <= j <= T}" and "we linearly transform the importance values
  into a bounded range [r_min, r_max]" with r_max = 1 (Sec. 4.3, Eq. 4).
- Code: for the layer being quantized, `input_layernorm(x)` -> `layer.self_attn(...,
  output_attentions=True)` -> `attn.sum(dim=1)` (heads) `.sum(dim=1)` (queries)
  `.mean(dim=0)` (batch of one); `normalize == "default"` -> min-max to
  [min_value, max_value] + clamp; `add_batch`: `weighting = weighting / weighting.sum()
  * T`, `inp = inp * weighting ** 0.5`, i.e. H accumulates sum_t w_t x_t x_t^T with
  w = r / mean(r). Computed on the sequential (already partially quantized) stream
  `inps` before any module of the layer is quantized; applied to all linear modules of
  the layer (`weighting_apply_module all`).
- r_min: the official scripts sweep {0.1, 0.05, 0.02, 0.01, 0.005}; the paper reports
  r_min = 0.01 optimal WITH rotation (Sec. 5.2) and "the best perplexity is achieved at
  r_min = 0.1" for scaling WITHOUT rotation (App. C.5, Fig. 9). Our control has no
  rotation, so the frozen native setting is **r_min = 0.1, r_max = 1**. r_min = 0.01 is
  recorded as the alternative not run.

Implementation in this codebase (`_qwen_rsq_attention_concentration`):
- Per text layer l, before any of its groups is quantized, for each of the 128 cached
  samples: run layer l on its cached sequential input (`cache_q`, the quantized-so-far
  stream that also feeds the Gram) with `_attn_implementation` forced to `eager` and an
  explicit additive causal mask (batch 1, unpadded), capture the post-softmax
  probabilities A (1, heads, T, T) from `self_attn`, and set
  s_j = sum_{m} sum_{i} A[0, m, i, j]. Row-sum deviation from 1 is logged.
- Native: r = clamp((s - min s)/(max s - min s) * (1 - r_min) + r_min, r_min, 1);
  w = r / mean(r). If max s == min s (never expected), w = 1 and the sample is counted
  as degenerate.
- Bounded-KL: w = `_normalize_qwen_sequence_token_weights(s, clip_max=2)` (the TIM
  Qwen path: floor, ratio normalization, bisection multiplier, final bounds [1e-4, 2],
  unit mean).
- Weights are keyed by `thinker.model.layers.{l}` and consumed by the unchanged
  `_qwen_collect_module_stats_on_cached_layer -> Helper.add_batch(token_weights=...)`,
  exactly as the TIM seqhess weights are.
- Metadata per run (`quantization.json -> rsq_attention_control`): definition, citation,
  r_min/r_max, per-layer raw-score range, weight min/max/mean, ESS fraction, saturation
  fraction, attention row-sum deviation, degenerate-sample count.

Known departures from RSQ that are intrinsic to the matched design (recorded, not
tuned): no rotation; ASR audio+prompt+transcript sequences instead of WikiText-2
4096-token chunks; 128 instead of 256 calibration sequences; per-sample score as in
RSQ but sequence lengths are variable; GPTQ damping/clip settings are TIM's, not RSQ's
(`--w_clip` not reproduced).

## 4. Execution plan

- GPU 0: RSQ-score/native draws 1 -> 3, then Uniform draws 1 -> 3.
- GPU 1: RSQ-score/bounded-KL draws 1 -> 3, then TIM draws 1 -> 3.
- Launcher: `E1_launch.sh` (sequential per GPU, refuses to overwrite, logs wall time).
- Run directories: `E1/runs/E1-{uniform|tim|rsq-native|rsq-kl}-qwen06-w3-c{seed}`.
- An artifact is terminal only when `status.json` is `{"state": "completed"}`.
- Smoke test (4 calibration samples, no evaluation, scratch directory) was run only to
  verify the pipeline executes and the metadata is populated; it is not an E1 artifact.

## 5. Analysis (pre-specified)

- Primary: internal-3 macro WER per arm (draw mean), and candidate - Uniform with the
  paper's crossed calibration-draw x utterance bootstrap
  (`analysis/task_fisher/crossed_multidraw_bootstrap.py`, 20,000 reps, seed 20260731,
  `--datasets librispeech-other voxpopuli gigaspeech`), pairing draw k of the candidate
  with draw k of Uniform.
- Also reported with the same convention: RSQ-score/bounded-KL - TIM, RSQ-score/native -
  TIM, RSQ-score/bounded-KL - RSQ-score/native.
- Integrity checks: identical calibration IDs across the four arms of each draw
  (`calibration.json`), identical example-ID order and references across all 12 runs'
  JSONL files (the bootstrap script raises otherwise), identical config fields except
  mode / weighting flags, `status.json` completed for all 12, empty-output counts.
- Wall time: per-arm wall seconds from the launcher timeline and
  `metrics.json.quantization_seconds`; no separate timing runs.
- No adaptive follow-up. No additional arms, models, bit widths, or r_min values.

## 6. Execution amendment (2026-09-15 ~20:50 UTC, recorded before any E1 WER was read)

Operational change only; no design, arm, seed, calibration, solver, or evaluation setting
changed. The two sequential per-GPU launchers were stopped after their first arms had
started (those two arms kept running to completion untouched), and the remaining ten arms
are executed by `E1_launch_wq.sh`: an atomic-claim (mkdir) work queue with two workers per
GPU, each worker waiting while its GPU already hosts two E1 processes. Reason: single-arm
GPU utilisation was 20-50% with ~6 GiB used of 16 GiB. Consequences: (i) numerical results
are unaffected (independent processes, deterministic per-process computation); (ii) wall
times are no longer single-tenant and must not be compared across arms or with the paper's
Appendix C.3 numbers; (iii) the two first arms have no launcher END line, their timing comes
from `metrics.json`. GPU assignment stays as planned (GPU 0: RSQ-native then Uniform; GPU 1:
RSQ-bounded-KL then TIM).

## 7. Evaluation-row correction (2026-09-15 21:0x UTC, recorded before any E1 WER was read)

The first launch used `--eval-samples -1` (the artifact's example scripts), which evaluates
the complete GigaSpeech test split (19,931 rows). The paper's internal-3 GigaSpeech has
3,000 rows ("2,939/1,842/3,000", App. E.4) and its "13,401 internal utterances" equals the
five registered splits each capped at 3,000 rows (2,620 + 2,939 + 3,000 + 1,842 + 3,000), so
the paper's runs used a per-split cap of 3,000 (`--eval-samples 3000`; the code takes the
first N rows in parquet order, deterministically). All 12 arms were stopped before any had
completed, their partial outputs were discarded (logs kept under `logs/restart1/`), and all
12 were relaunched with `--eval-samples 3000`. Every other setting is unchanged. Row identity
with the paper's runs additionally assumes the same dataset snapshot file order (snapshot
b6bdcd0 here); the paper's snapshot hash is not recorded in the artifact.
