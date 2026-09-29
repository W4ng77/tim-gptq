#!/usr/bin/env python3
"""Assemble E3_FP4_REBUTTAL_RESULT.md from the E3 run directories (raw results + integrity only)."""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

E3 = Path(__file__).resolve().parent
E1 = Path(__file__).resolve().parents[1] / "E1"
RUNS = E3 / "runs"
ANALYSIS = (Path(__file__).resolve().parents[2] / "analysis/task_fisher/crossed_multidraw_bootstrap.py")
DATASETS = ["librispeech-other", "voxpopuli", "gigaspeech"]
SEEDS = [20260729, 20260730, 20260731]
FMTS = ["mxfp4", "nvfp4"]
DENS = ["uniform", "tim"]
ARMS = [(f, d) for f in FMTS for d in DENS]
PP = 100.0


def rd(f, d, s):
    return RUNS / f"E3-{f}-{d}-qwen06-c{s}"


def load(p):
    return json.loads(Path(p).read_text())


def status(p):
    q = Path(p) / "status.json"
    return load(q).get("state") if q.is_file() else "missing"


def wers(p):
    m = load(Path(p) / "metrics.json")
    return {e["dataset"]: PP * float(e["wer"]) for e in m["evaluations"]}, m


def macro(w):
    return sum(w[d] for d in DATASETS) / len(DATASETS)


def bootstrap(cand_dirs, base_dirs, cand_label, base_label, outname):
    outdir = E3 / "bootstrap" / outname
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ANALYSIS), "--output-dir", str(outdir), "--datasets", *DATASETS,
           "--reps", "20000", "--seed", "20260731", "--baseline-label", base_label, "--candidate-label", cand_label]
    for b, c in zip(base_dirs, cand_dirs):
        cmd += ["--simple-pair", str(b), str(c)]
    subprocess.run(cmd, check=True, cwd=str(ANALYSIS.parent))
    return load(sorted(outdir.glob("*.json"))[0])


def fmt_ci(b):
    m = b["macro"]
    est = PP * m.get("delta_draw_mean", m.get("estimate"))
    return f"{est:+.3f}", f"[{PP * m['ci95_lower']:+.3f}, {PP * m['ci95_upper']:+.3f}]", ", ".join(f"{PP * v:+.3f}" for v in b["macro_delta_by_draw"]), "; ".join(
        f"{d}: {PP * b['bootstrap'][d]['estimate']:+.3f} [{PP * b['bootstrap'][d]['ci95_lower']:+.3f}, {PP * b['bootstrap'][d]['ci95_upper']:+.3f}]" for d in DATASETS)


def main():
    L, problems = [], []
    L.append("# E3 result: TIM calibration measure through MXFP4 / NVFP4 quantizer geometry (Qwen3-ASR-0.6B text backbone)\n")
    L.append("Raw results and integrity checks only; no interpretation. Protocol: E3_PROTOCOL.md (frozen before evaluation). "
             "Software fake quantization inside the paper's GPTQ solver (corrected weight -> format block fake-quant -> dequantized q -> unchanged error compensation); "
             "MXFP4 = E2M1 / block 32 / E8M0 scale (OCP MX v1.0); NVFP4 = E2M1 / block 16 / E4M3 block scale x FP32 tensor scale, a 1-D numerical emulation and NOT native Blackwell execution. "
             "All GEMMs run dequantized in BF16 on RTX 4080; no hardware speed is measured or claimed. Deployment Map, full Support, calibration seeds 20260729/20260730/20260731 "
             "(same IDs as E1), quantization seed 20260729, internal-3 rows as E1/paper (2,939/1,842/3,000). Contrasts: candidate minus baseline in WER percentage points; "
             "a 95% interval excluding zero is resolved.\n")
    L.append("## Terminal status\n")
    L.append("| Arm | Draw 1 | Draw 2 | Draw 3 |")
    L.append("|---|---|---|---|")
    complete = True
    for f, d in ARMS:
        row = [status(rd(f, d, s)) for s in SEEDS]
        complete &= all(r == "completed" for r in row)
        L.append(f"| {f.upper()} / {d} | " + " | ".join(row) + " |")
    ref_dir = RUNS / "E3-bf16-reference"
    L.append(f"| BF16 reference (context) | {status(ref_dir)} | | |")
    L.append("")
    if not complete:
        L.append("**Not all 12 artifacts are completed; no bootstrap is run.**\n")

    W, M = {}, {}
    for f, d in ARMS:
        for s in SEEDS:
            if status(rd(f, d, s)) == "completed":
                W[(f, d, s)], M[(f, d, s)] = wers(rd(f, d, s))
    ref = wers(ref_dir)[0] if status(ref_dir) == "completed" else None

    boot = {}
    if complete:
        for f in FMTS:
            boot[(f, "tim-uniform")] = bootstrap([rd(f, "tim", s) for s in SEEDS], [rd(f, "uniform", s) for s in SEEDS], f"{f}-tim", f"{f}-uniform", f"{f}_tim_minus_uniform")
        for d in DENS:
            boot[("nvfp4-mxfp4", d)] = bootstrap([rd("nvfp4", d, s) for s in SEEDS], [rd("mxfp4", d, s) for s in SEEDS], f"nvfp4-{d}", f"mxfp4-{d}", f"nvfp4_minus_mxfp4_{d}")

    L.append("## Internal-3 macro WER (%) per draw, draw mean, and TIM - Uniform per format\n")
    L.append("| Arm | Draw 1 | Draw 2 | Draw 3 | Macro (draw mean) | TIM - Uniform (pp) | 95% CI |")
    L.append("|---|---|---|---|---|---|---|")
    if ref:
        L.append(f"| BF16 reference (native weights loaded in bfloat16, no calibration; artifact mode name `fp16`) | {macro(ref):.3f} | | | {macro(ref):.3f} | — | — |")
    for f, d in ARMS:
        cells = [f"{macro(W[(f, d, s)]):.3f}" if (f, d, s) in W else "n/a" for s in SEEDS]
        mean = f"{sum(macro(W[(f, d, s)]) for s in SEEDS) / 3:.3f}" if all((f, d, s) in W for s in SEEDS) else "n/a"
        if d == "tim" and (f, "tim-uniform") in boot:
            est, ci, _, _ = fmt_ci(boot[(f, "tim-uniform")])
        else:
            est, ci = ("0", "—") if d == "uniform" else ("n/a", "n/a")
        L.append(f"| {f.upper()} / {d} | " + " | ".join(cells) + f" | {mean} | {est} | {ci} |")
    L.append("")

    L.append("## Contrasts (crossed calibration-draw x utterance bootstrap, 20,000 reps, seed 20260731)\n")
    L.append("| Contrast | Δ macro (pp) | 95% CI | Δ by draw | Δ per dataset [95% CI] |")
    L.append("|---|---|---|---|---|")
    for f in FMTS:
        if (f, "tim-uniform") in boot:
            est, ci, by, per = fmt_ci(boot[(f, "tim-uniform")])
            L.append(f"| {f.upper()}: TIM - Uniform | {est} | {ci} | {by} | {per} |")
    for d in DENS:
        if ("nvfp4-mxfp4", d) in boot:
            est, ci, by, per = fmt_ci(boot[("nvfp4-mxfp4", d)])
            L.append(f"| NVFP4 - MXFP4 under {d} (geometry, descriptive) | {est} | {ci} | {by} | {per} |")
    L.append("")

    L.append("## Per-dataset WER (%) per draw\n")
    L.append("| Arm | Draw | " + " | ".join(DATASETS) + " | macro |")
    L.append("|---|---|" + "---|" * (len(DATASETS) + 1))
    if ref:
        L.append("| BF16 reference | — | " + " | ".join(f"{ref[x]:.3f}" for x in DATASETS) + f" | {macro(ref):.3f} |")
    for f, d in ARMS:
        for i, s in enumerate(SEEDS, 1):
            if (f, d, s) in W:
                w = W[(f, d, s)]
                L.append(f"| {f.upper()} / {d} | {i} (c{s}) | " + " | ".join(f"{w[x]:.3f}" for x in DATASETS) + f" | {macro(w):.3f} |")
    L.append("")

    # context: E1 INT3/G128
    L.append("## Context only: E1 INT3/G128 (same draws and rows; not a matched-format comparison; no INT4/G128 Qwen-0.6B artifact exists on this machine)\n")
    e1 = {}
    for d in DENS:
        for s in SEEDS:
            p = E1 / "runs" / f"E1-{d}-qwen06-w3-c{s}"
            if (p / "metrics.json").is_file():
                e1[(d, s)] = wers(p)[0]
    if e1:
        L.append("| Arm | Draw 1 | Draw 2 | Draw 3 | Mean |")
        L.append("|---|---|---|---|---|")
        for d in DENS:
            vals = [macro(e1[(d, s)]) for s in SEEDS if (d, s) in e1]
            L.append(f"| INT3/G128 {d} (E1) | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {sum(vals) / len(vals):.3f} |")
        L.append("E1 TIM - Uniform (INT3/G128): see E1_RSQ_REBUTTAL_RESULT.md (-0.100 pp [-0.417, +0.162]).")
    L.append("")

    # reconstruction
    L.append("## Numerical reconstruction error of the GPTQ output, sum over the 196 quantized modules of ||W - Q(W)||_F^2\n")
    L.append("| Arm | Draw | sum ||W-Q(W)||^2 | relative to sum ||W||^2 | mean sq error per weight |")
    L.append("|---|---|---|---|---|")
    recon = {}
    for f, d in ARMS:
        for i, s in enumerate(SEEDS, 1):
            p = rd(f, d, s) / "quantization.json"
            if p.is_file():
                r = load(p).get("reconstruction")
                if r:
                    recon[(f, d, s)] = r
                    L.append(f"| {f.upper()} / {d} | {i} | {r['sum_sq_error']:.2f} | {r['relative_sq_error']:.5f} | {r['mean_sq_error_per_weight']:.3e} |")
    for f, d in ARMS:
        vals = [recon[(f, d, s)]["sum_sq_error"] for s in SEEDS if (f, d, s) in recon]
        if len(vals) == 3:
            L.append(f"| {f.upper()} / {d} | mean | {sum(vals) / 3:.2f} | {sum(recon[(f, d, s)]['relative_sq_error'] for s in SEEDS) / 3:.5f} | |")
    L.append("")

    # integrity
    L.append("## Integrity checks\n")
    for s in SEEDS:
        ids = {}
        for f, d in ARMS:
            p = rd(f, d, s) / "calibration.json"
            if p.is_file():
                ids[f"{f}/{d}"] = load(p)["example_ids"]
        e1p = E1 / "runs" / f"E1-uniform-qwen06-w3-c{s}" / "calibration.json"
        if e1p.is_file():
            ids["E1-uniform"] = load(e1p)["example_ids"]
        same = len({json.dumps(v) for v in ids.values()}) == 1 if ids else False
        L.append(f"- Draw c{s}: calibration IDs identical across {sorted(ids)}: **{same}** (128 each: {all(len(v) == 128 for v in ids.values())})")
        if not same:
            problems.append(f"calibration IDs differ in draw {s}")
    cfgs = {(f, d, s): load(rd(f, d, s) / "config.json") for f, d in ARMS for s in SEEDS if (rd(f, d, s) / "config.json").is_file()}
    if cfgs:
        allk = set().union(*[c.keys() for c in cfgs.values()])
        diff = sorted(k for k in allk if len({json.dumps(c.get(k), sort_keys=True) for c in cfgs.values()}) > 1)
        expected = {"run_name", "seed", "mode", "effective_model_state", "sequence_calibration", "sequence_hessian", "propagated_clip_max", "fp4_format", "groupsize"}
        unexpected = sorted(set(diff) - expected)
        L.append(f"- config.json fields differing across the 12 runs: {diff}; unexpected: **{unexpected if unexpected else 'none'}** (groupsize differs only by format block size: {sorted({(c['fp4_format'], c['groupsize']) for c in cfgs.values()})})")
        if unexpected:
            problems.append(f"unexpected config differences: {unexpected}")
        L.append(f"- wbits recorded: {sorted({c['wbits'] for c in cfgs.values()})}")
    qh = set()
    for f, d in ARMS:
        for s in SEEDS:
            p = rd(f, d, s) / "quantization.json"
            if p.is_file():
                q = load(p)
                qh.add(q["fp4_format"]["quantizer_source_sha256"])
                if q["fp4_format"]["block_size_contracting_dim"] != (32 if f == "mxfp4" else 16):
                    problems.append(f"block size mismatch in {f}/{d}/{s}")
    L.append(f"- fp4_formats.py source sha256 identical across all runs (same quantizer code path for Uniform and TIM): **{len(qh) == 1}** {qh}")
    align = True
    for x in DATASETS:
        ref_rows = None
        for f, d in ARMS:
            for s in SEEDS:
                p = rd(f, d, s) / f"{x}.jsonl"
                if p.is_file():
                    rows = [(r["example_id"], r["reference"]) for r in map(json.loads, p.read_text().splitlines()) if r]
                    if ref_rows is None:
                        ref_rows = rows
                        L.append(f"- {x}: {len(rows)} rows (reference run E3-{f}-{d}-c{s})")
                    elif rows != ref_rows:
                        align = False
                        problems.append(f"rows differ for {x} in {f}/{d}/{s}")
        e1p = E1 / "runs" / f"E1-uniform-qwen06-w3-c{SEEDS[0]}" / f"{x}.jsonl"
        if e1p.is_file() and ref_rows is not None:
            e1rows = [(r["example_id"], r["reference"]) for r in map(json.loads, e1p.read_text().splitlines()) if r]
            L.append(f"  - identical to E1 rows: {e1rows == ref_rows}")
    L.append(f"- Evaluation example IDs and references identical across all E3 runs: **{align}**")
    L.append("- Empty predictions per run:")
    for f, d in ARMS:
        for s in SEEDS:
            parts = []
            for x in DATASETS:
                p = rd(f, d, s) / f"{x}.jsonl"
                if p.is_file():
                    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                    parts.append(f"{x} {sum(1 for r in rows if not r['prediction'].strip())}/{len(rows)}")
            if parts:
                L.append(f"  - E3-{f}-{d}-c{s}: " + "; ".join(parts))
    envs = {(load(rd(f, d, s) / 'environment.json').get('source_tree_sha256'), load(rd(f, d, s) / 'environment.json').get('torch')) for f, d in ARMS for s in SEEDS if (rd(f, d, s) / 'environment.json').is_file()}
    L.append(f"- environment (source_tree_sha256, torch) identical across runs: **{len(envs) == 1}** {envs}")
    # Environment deviation check (added 2026-09-16T01:35Z after the 12 quantized runs completed, before results were read in full):
    # compare the torch/CUDA build recorded by the E3 runs with the E1 records and the artifact's pinned requirement.
    e3_env = [load(rd(f, d, s) / "environment.json") for f, d in ARMS for s in SEEDS if (rd(f, d, s) / "environment.json").is_file()]
    ref_env = load(ref_dir / "environment.json") if (ref_dir / "environment.json").is_file() else {}
    e1_env = [load(p) for p in sorted((E1 / "runs").glob("E1-*/environment.json"))]
    pinned = None
    for req in sorted(Path(__file__).resolve().parents[2] / "implementations/whisper_qwen".glob("requirements*.txt")):
        m = re.search(r"^torch==(\S+)", req.read_text(), re.M)
        if m:
            pinned = (req.name, m.group(1))
    e3_torch = {(e.get("torch"), e.get("cuda")) for e in e3_env}
    e1_torch = {(e.get("torch"), e.get("cuda")) for e in e1_env}
    L.append(f"- torch/CUDA recorded by the 12 E3 runs: {sorted(e3_torch)}; by the BF16 reference: {(ref_env.get('torch'), ref_env.get('cuda'))}; by the E1 runs: {sorted(e1_torch)}; artifact pin: {pinned}")
    dev_lines = []
    if pinned and any(t is None or not str(t).startswith(pinned[1]) for t, _ in e3_torch):
        dev_lines.append(f"E3 ran under torch {sorted(e3_torch)} while the artifact pins torch=={pinned[1]} ({pinned[0]}); E1 (INT3 context rows) ran under {sorted(e1_torch)}.")
    if e3_torch and e1_torch and e3_torch != e1_torch:
        dev_lines.append("Cause: `pip install compressed-tensors==0.18.0` (reference implementation for the pre-run FP4 comparison, installed 2026-09-16T00:32Z) requires torch>=2.10 and upgraded torch in the shared `timgptq` env before the E3 launch (00:42:54Z); E1 and E2 had already completed under the pinned build. Discovered 2026-09-16T01:35Z from the environment records, after the 12 quantized runs had finished. All 12 E3 arms, the E3 unit tests, and the BF16 reference share the same upgraded build, so Uniform and TIM remain environment-matched within E3; the E1 INT3 context rows and the paper's FP16 number were produced under torch 2.7.0+cu126 and are not environment-matched to E3.")
    for suite in ("whisper_qwen", "voxtral"):
        lp = E3 / "logs" / f"protocol_tests_torch214_{suite}.log"
        if lp.is_file() and dev_lines:
            m = re.search(r"(\d+) passed", lp.read_text())
            dev_lines.append(f"Artifact protocol tests under the upgraded build ({suite}, working tree with E1/E3 patches): {m.group(0) if m else 'no pass line'} ({lp.name}).")
    if dev_lines:
        L.append("\n## DEVIATION from the pinned artifact environment (not a run failure; recorded for the rebuttal)\n")
        L += [f"- {x}" for x in dev_lines]
        L.append("")
    L.append("## Wall time (two arms per GPU; not comparable across arms or with the paper)\n")
    tl = (E3 / "logs" / "timeline.log").read_text() if (E3 / "logs" / "timeline.log").is_file() else ""
    walls = {m.group(2): (m.group(1), m.group(3), m.group(4)) for m in re.finditer(r"END\s+gpu=(\d)\s+worker=\S+\s+(E3-\S+)\s+rc=(\d+)\s+wall_s=(\d+)", tl)}
    L.append("| Run | GPU | rc | wall (s) | quantization_seconds | total_seconds |")
    L.append("|---|---|---|---|---|---|")
    for f, d in ARMS:
        for s in SEEDS:
            name = f"E3-{f}-{d}-qwen06-c{s}"
            w = walls.get(name)
            mm = M.get((f, d, s))
            L.append(f"| {name} | {w[0] if w else 'n/a'} | {w[1] if w else 'n/a'} | {w[2] if w else 'n/a'} | {mm['quantization_seconds']:.1f} | {mm['total_seconds']:.1f} |" if mm else f"| {name} | n/a | n/a | n/a | n/a | n/a |")
    L.append("")
    L.append("## Run directories\n")
    for f, d in ARMS:
        for s in SEEDS:
            L.append(f"- {rd(f, d, s)}")
    L.append(f"- {ref_dir}")
    L.append(f"\nBootstrap outputs: {E3 / 'bootstrap'}. Code: tim-gptq @ 12d30a5 + E1/e1_rsqattn.patch + E3/e3_fp4.patch (fp4_formats.py, gptq.py fp4_format hook, --fp4-format flag, reconstruction stats).")
    if problems:
        L.append("\n## PROBLEMS DETECTED\n")
        L += [f"- {p}" for p in problems]
    out = E3 / "E3_FP4_REBUTTAL_RESULT.md"
    out.write_text("\n".join(L) + "\n")
    print(out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
