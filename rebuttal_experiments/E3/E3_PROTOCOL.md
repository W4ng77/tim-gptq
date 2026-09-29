# E3 — numerical transfer of the TIM calibration measure through MXFP4 / NVFP4 quantizer geometry (frozen protocol)

Frozen 2026-09-16 (UTC) before any E3 artifact was evaluated. Rebuttal-only numerical
fake-quantization experiment. No hardware-speed claim: the RTX 4080 executes dequantized
BF16 GEMMs; the experiment tests whether the calibration-measure effect transfers across
quantizer geometry only. No additional settings are searched after outcomes are seen.

## 1. Fixed configuration (identical to E1 except the quantizer)

| Field | Value |
|---|---|
| Model / component | Qwen3-ASR-0.6B (snapshot 5eb1441), `--quant-scope text-backbone` (28 layers x q/k/v, o, gate/up, down) |
| Map / Support | deployment inference template; full Support (prompt + teacher-forced transcript rows) |
| Calibration | LibriSpeech clean/train.100, 128 utterances, draws = `--seed` 20260729 / 20260730 / 20260731 (same IDs as E1, verified per draw), `--quantization-seed 20260729` |
| Density | Uniform (`gptq+seqcal`, w = 1) vs TIM (`gptq+seqhess --propagated-clip-max 2`, KL floor 1e-4 as in the paper's Qwen code path) |
| Solver | the paper's grouped GPTQ (`gptq.py`): same Gram, damping 0.01, blocksize 128, no act-order, same column ordering and error compensation. ONLY the quantization operator is replaced: `corrected weight -> format block fake-quant -> dequantized q -> unchanged error compensation` (`run_gptq(..., fp4_format=...)`). Group size = format block size along the contracting dimension. |
| Evaluation | internal-3 (librispeech-other 2,939 / voxpopuli 1,842 / gigaspeech first 3,000 rows), `--eval-samples 3000`, batch 8, raw corpus WER; macro = equal-domain mean; same rows as E1 |
| Reference | one BF16 native pass (`--mode fp16`, no calibration), context only |
| Determinism / execution | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `HF_HUB_OFFLINE=1`; two arms per GPU (locked admission gate); wall times not single-tenant |

## 2. Formats (software fake quantization, `fp4_formats.py`, float32 arithmetic)

**MXFP4** (OCP Microscaling Formats v1.0): elements E2M1 = {0, +-0.5, +-1, +-1.5, +-2, +-3,
+-4, +-6}; block size 32 along the input/contracting dimension (32 consecutive weight
columns per output row); one E8M0 shared scale per block, X = 2^(floor(log2(max|v|)) - 2)
(emax of E2M1 = 2); elements = RNE_E2M1(v / X), saturating at +-6.

**NVFP4** (1-D numerical emulation of NVIDIA's Blackwell NVFP4; not native execution):
elements E2M1; block size 16 along the contracting dimension; FP8 E4M3 block scale and FP32
per-tensor scale following NVIDIA's scaling equations:
s_tensor = amax(W) / (6 x 448) (FP32, computed once from the module's weight before GPTQ),
s_block = E4M3(amax(block) / 6 / s_tensor) (RNE via torch.float8_e4m3fn, clamped to 448),
s_decoded = float(s_block) x s_tensor, q = RNE_E2M1(w / s_decoded) saturating at +-6,
w_hat = q x s_decoded. Block scales are recomputed at every block boundary from the
error-compensated weights, exactly where the integer quantizer's `find_params` runs.

Dead columns, damping, Cholesky factor, block-wise error propagation and the tail update in
`run_gptq` are byte-identical for both Densities and both formats.

## 3. Pre-run verification (E3/test_fp4_formats.py, pytest)

- E2M1 codebook values; round-to-nearest-even at all seven midpoints; saturation.
- RNE agreement with an independent brute-force nearest/even implementation on 250k
  random values plus all ties.
- E2M1 rounding agreement (bitwise) with the third-party reference
  `compressed_tensors.quantization.utils.cast_to_fp4` (compressed-tensors 0.18.0).
- MXFP4: block = 32, scale is a power of two, exponent equals floor(log2(amax)) - 2,
  dequantized values are codebook multiples of the scale, Q(Q(W)) = Q(W).
- NVFP4: block = 16, block scale E4M3-representable and <= 448, tensor scale =
  amax/(6 x 448), decoded scale = E4M3 scale x tensor scale, values are codebook
  multiples of the decoded scale.
- Error bounds: |w - w_hat| <= scale inside the codebook range; MXFP4 saturation error < 2 x
  scale (amax/X in [4, 8)); NVFP4 amax/scale <= 6(1 + 1/8).
- In-solver instrumentation: inside `run_gptq` the FP4 operator is called once per
  column, block scales once per block of the format's size, the output equals
  RNE_E2M1(col/scale) x scale, the group size requested by the CLI (128) is overridden by the
  block size, the same operator/code path serves Uniform and TIM (the operator source
  contains no token-weight logic; its sha256 is recorded in every run's quantization.json),
  and different Grams give different weights.
- Reference comparison of MX scales and NVFP4 QDQ against compressed-tensors: see
  section 6 (filled in before launch).

## 4. Arms (12 artifacts + BF16 reference)

| Format | Density | Draws |
|---|---|---|
| MXFP4 | Uniform | 3 |
| MXFP4 | TIM | 3 |
| NVFP4 | Uniform | 3 |
| NVFP4 | TIM | 3 |

Run directories `E3/runs/E3-{mxfp4|nvfp4}-{uniform|tim}-qwen06-c{seed}` and
`E3/runs/E3-bf16-reference`. GPU 0: MXFP4 arms; GPU 1: NVFP4 arms; the BF16 reference on
whichever GPU frees first.

## 5. Pre-specified reporting

- WER per internal-3 dataset per draw; macro per draw and draw mean; BF16 reference.
- TIM - Uniform for each format with the paper's crossed calibration-draw x utterance
  bootstrap (20,000 reps, seed 20260731); also NVFP4 - MXFP4 within each Density as a
  descriptive geometry contrast.
- Numerical reconstruction error sum_modules ||W - Q(W)||_F^2 (and relative to ||W||^2) per
  run, from the GPTQ output weights.
- Existing INT results only as context: E1's INT3/G128 Uniform/TIM (same draws, same rows).
  No INT4/G128 Qwen-0.6B artifact exists on this machine and none is run.
- Integrity: status completed x 13; identical calibration IDs across the four arms of a
  draw and with E1; identical evaluation rows; config diffs limited to
  mode/density/format/seed/run-name; identical quantizer source hash across runs;
  empty-output counts.
- No adaptive follow-up.

## 6. Reference-implementation comparison (recorded before launch; compressed-tensors 0.18.0)

- E2M1 element rounding: bitwise identical to `compressed_tensors.quantization.utils.cast_to_fp4`
  on 150k random values plus all seven midpoint ties (both round half to the even encoding).
- MXFP4 E8M0 block scale: identical E8M0 exponent to `generate_mx_scales` (the vLLM/compressed-tensors
  reference) for every block whose amax has a binary mantissa < 1.75 (80.1% of uniformly random
  block maxima); for mantissa >= 1.75 the reference rounds the exponent UP by one (avoids saturation,
  coarser step), whereas the OCP MX v1.0 text formula X = 2^(floor(log2(amax)) - 2) used here keeps
  the lower exponent and saturates elements in (6, 8) x X to 6. This is a known implementation
  choice; the OCP formula is what the protocol specifies and what all E3 runs use.
- NVFP4: E4M3 block scales identical to `calculate_qparams(..., strategy=tensor_group, group_size=16,
  scale_dtype=float8_e4m3fn, global_scale=6*448/amax)`; dequantized tensors agree to within
  floating-point operation order (max |diff| <= 1e-6 x max|W|, exact on a subset), since the
  reference applies the global scale as a multiplier (x / (s / G)) and this implementation as a
  divisor (x / (s * s_tensor)) with s_tensor = 1/G.
- All 12 unit tests pass (E3/logs/pytest_fp4_formats.log): codebook, RNE ties, saturation, MXFP4
  block/E8M0/idempotence, NVFP4 block/E4M3/tensor scale, error bounds, in-solver instrumentation
  (operator called once per column, scales once per format block, identical code path for Uniform
  and TIM, no token-weight logic in the operator), block-size override of the CLI group size, and
  the three reference comparisons above.

## 7. Post-launch deviation record (written 2026-09-16T01:35Z, after all 12 quantized runs completed; no setting or result was changed)

- The artifact pins `torch==2.7.0` (whisper_qwen requirements). E1 and E2 ran under torch 2.7.0+cu126
  (recorded in their `environment.json`). At 2026-09-16T00:32Z, `pip install compressed-tensors==0.18.0`
  (the reference implementation used for §6) pulled its dependency `torch>=2.10.0` and upgraded the shared
  `timgptq` env to torch 2.14.0+cu130 (with CUDA 13 runtime wheels and triton 3.8). The E3 unit tests
  (00:42Z), all 12 E3 quantized runs (00:42:54Z onward), and the E3 BF16 reference therefore ran under
  torch 2.14.0+cu130, as recorded in every E3 `environment.json`.
- Consequence: Uniform and TIM are environment-matched within E3 (identical torch, CUDA, source tree hash,
  quantizer source hash). The E1 INT3/G128 context rows and the paper's FP16 number were produced under
  torch 2.7.0+cu126 and are not environment-matched to the E3 numbers; the protocol already labels them
  context only. The E3-vs-paper BF16 comparison is likewise cross-environment.
- No re-run was launched on this finding. Restoring `torch==2.7.0+cu126` and re-running the 13 E3 jobs
  (~1 h on two GPUs) is a decision left to the authors; it would be a protocol-fidelity re-run, not an
  adaptive follow-up, and if done both versions must be reported.
- Check under the upgraded build (2026-09-16T01:33Z, working tree 12d30a5 + E1/E3 patches, torch 2.14.0+cu130):
  the artifact's protocol tests pass, whisper_qwen 77/77 and voxtral 66/66
  (`E3/logs/protocol_tests_torch214_whisper_qwen.log`, `E3/logs/protocol_tests_torch214_voxtral.log`).
