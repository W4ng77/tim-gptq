# TIM-GPTQ

Anonymous implementation for *Quantize What the Decoder Consumes: Task-Induced
Measures for Low-Bit ASR*.

TIM-GPTQ changes the calibration measure used to construct GPTQ Gram matrices.
It collects deployment-format states, scores them with transcription-loss
pullbacks, applies a bounded ratio-preserving I-projection, and then calls the
unchanged grouped-GPTQ solver. It does not add inference-time operators or
change the saved weight layout.

## Repository layout

- `implementations/whisper_qwen`: Whisper, Distil-Whisper, Moonshine, and
  Qwen3-ASR implementation, including prompt/support interventions and
  prompt/response-stratified permutation.
- `implementations/voxtral`: Voxtral implementation. This is kept as a
  separate source tree because the vendor model API and cached-sequential
  execution path differ from the other families.
- `analysis`: paired, crossed, factorial, bit-by-target, and nested-permutation
  bootstrap programs used by the paper.
- `examples`: reproducible command templates for the principal method and its
  identification controls.
- `rebuttal_experiments`: code for the author-response experiments added after
  submission (RSQ attention-concentration Density control, a machine-translation
  existence check, and MXFP4/NVFP4 fake quantization inside the GPTQ solver).
- `manifests`: schema example and instructions for materializing evaluation
  audio locally. Audio, model weights, caches, and predictions are not bundled.

The two implementation trees intentionally share some files. They are the
tested architecture-specific snapshots used for the reported experiments; an
untested source-level merge would make the artifact less faithful.

## Environment

Python 3.12 and CUDA were used for the reported experiments. Install one model
tree at a time in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r implementations/whisper_qwen/requirements.txt
pip install -r requirements-analysis.txt
```

For Voxtral, additionally install the pinned interface dependency:

```bash
pip install mistral-common==1.9.1
```

Model and dataset licenses are controlled by their original providers. This
artifact downloads neither automatically during tests nor redistributes them.

## Quick start

List registered models:

```bash
python implementations/whisper_qwen/src/asr_experiment.py --list-models
python implementations/voxtral/src/asr_experiment.py --list-models
```

The principal Whisper encoder-W2 command is:

```bash
python implementations/whisper_qwen/src/asr_experiment.py \
  --model whisper-large-v3 \
  --mode gptq \
  --quant-scope encoder \
  --wbits 2 --groupsize 128 \
  --nsamples 128 --calib-batch-size 16 \
  --seed 20260729 --quantization-seed 20260729 \
  --frame-weighting task-fisher \
  --propagated-clip-max 2 \
  --eval --eval-samples -1 \
  --output-dir runs --run-name tim-whisper-large-v3-w2
```

Uniform GPTQ uses the same command with `--frame-weighting none`. The
identification control adds `--task-weight-permutation within-sample` and a
frozen `--task-weight-permutation-seed`. Qwen Map and Support are controlled by
the inference-template and `--sequence-support` arguments exposed in
`--help`. Ready-to-edit commands for all three families are under `examples/`.

Every run writes `config.json`, `environment.json`, `calibration.json`,
`quantization.json`, per-dataset JSONL predictions, `metrics.json`, and
`status.json`. A run is terminal only when `status.json` contains
`"state": "completed"`; the metrics file is written incrementally and is not a
completion signal.

## Tests

The protocol tests use small mocks and do not download checkpoints or data:

```bash
python -m pytest -q implementations/whisper_qwen/tests/test_protocol.py
python -m pytest -q implementations/voxtral/tests/test_protocol.py
```

The packaged snapshots pass 77 and 66 tests, respectively. The FP4 format
tests added for the author response run with

```bash
python -m pytest -q implementations/whisper_qwen/tests/test_fp4_formats.py
```

(12 tests; the three reference comparisons are skipped unless
`compressed-tensors` is installed). See
`REPRODUCIBILITY.md` for claim-to-code and analysis mappings.

## Scope of the anonymous artifact

The package contains TIM-GPTQ, matched GPTQ/RTN/AWQ and diagnostic controls,
and the statistical analysis code required by the paper. It deliberately
excludes checkpoints, audio, generated transcripts, scheduler logs, machine
paths, credentials, Git history, and unrelated prior-project artifacts.

