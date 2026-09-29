# Author-response experiments

Code for three experiments run for the ARR author response, after submission. Each folder holds the protocol
frozen before evaluation (`*_PROTOCOL.md`), a work-queue launcher (two runs per GPU on two GPUs), and the
analysis script that writes the result report. Run outputs are not included.

| Folder | Experiment |
|---|---|
| `E1/` | Matched RSQ attention-concentration Density control, Qwen3-ASR-0.6B text backbone, W3/G128, internal-3 |
| `E2/` | Machine-translation existence check of the Map/Density decomposition, Qwen3-0.6B, FLORES-200 en-de, W3/G128 |
| `E3/` | MXFP4 and NVFP4 fake quantization inside the GPTQ solver, Qwen3-ASR-0.6B text backbone |

## Source changes

- **E1** adds `--mode gptq+rsqattn` with `--rsq-score-normalization {native,bounded-kl}`, `--rsq-min-value`, and
  `--rsq-max-value` to `implementations/whisper_qwen`. The score is RSQ's attention concentration on the current
  layer's sequential input; this is a Density control, not a reimplementation of full RSQ (no rotation).
- **E3** adds `--fp4-format {none,mxfp4,nvfp4}` and `implementations/whisper_qwen/src/fp4_formats.py`.
  The FP4 operator replaces only the per-column quantize step inside `run_gptq`; GPTQ error compensation is
  unchanged. MXFP4 is E2M1 with block 32 and an E8M0 scale. NVFP4 is E2M1 with block 16, an E4M3 block scale, and an
  FP32 tensor scale; it is a 1-D numerical emulation, not native Blackwell execution. All GEMMs run dequantized, so
  no speed is measured. Each quantized run records `||W - Q(W)||^2` in `quantization.json`.
- **E2** uses the standalone driver `E2/e2_mt_gptq.py`, which imports this artifact's GPTQ solver and KL projection.

Tests for the FP4 module: `python -m pytest -q implementations/whisper_qwen/tests/test_fp4_formats.py`.

## Running

Each launcher writes `runs/`, `logs/`, and `claims/` next to itself. It expects a conda env named `timgptq`
built from `implementations/whisper_qwen/requirements.txt`; set `CONDA_BASE` if conda is not in `~/miniconda3`.
Internal-3 evaluation uses `--eval-samples 3000`, which reproduces the paper's 2,939 / 1,842 / 3,000 rows.
E3's analysis also reads E1's Uniform runs for context rows, so run E1 first. E2 expects the public
FLORES-200 release unpacked at `E2/data/flores200_dataset` (checksums in `E2/E2_PROTOCOL.md`).

The reference comparisons in the FP4 tests need `compressed-tensors`, which requires torch>=2.10 and will upgrade
the pinned torch 2.7.0; install it in a separate environment.
