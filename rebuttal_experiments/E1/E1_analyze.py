#!/usr/bin/env python3
"""Assemble E1_RSQ_REBUTTAL_RESULT.md from the 12 E1 run directories.

Reports raw matched results and integrity checks only. No interpretation.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

E1 = Path(__file__).resolve().parent
RUNS = E1 / "runs"
ANALYSIS = (Path(__file__).resolve().parents[2] / "analysis/task_fisher/crossed_multidraw_bootstrap.py")
DATASETS = ["librispeech-other", "voxpopuli", "gigaspeech"]
SEEDS = [20260729, 20260730, 20260731]
ARMS = [
    ("uniform", "Uniform GPTQ (reconstructed here)"),
    ("tim", "TIM-GPTQ task-Fisher + bounded KL (reconstructed here)"),
    ("rsq-native", "matched RSQ attention-concentration Density control / native scaling"),
    ("rsq-kl", "matched RSQ attention-concentration Density control / bounded-KL"),
]
CONTRASTS = [
    ("rsq-native", "uniform"),
    ("rsq-kl", "uniform"),
    ("tim", "uniform"),
    ("rsq-kl", "tim"),
    ("rsq-native", "tim"),
    ("rsq-kl", "rsq-native"),
]


def run_dir(arm: str, seed: int) -> Path:
    return RUNS / f"E1-{arm}-qwen06-w3-c{seed}"


def load(path: Path):
    return json.loads(path.read_text())


def status(arm, seed):
    p = run_dir(arm, seed) / "status.json"
    return load(p).get("state") if p.is_file() else "missing"


def per_dataset_wer(arm, seed):
    m = load(run_dir(arm, seed) / "metrics.json")
    out = {}
    for ev in m["evaluations"]:
        out[ev["dataset"]] = 100.0 * float(ev["wer"])
    return out, m


def macro(wers):
    return sum(wers[d] for d in DATASETS) / len(DATASETS)


def bootstrap(candidate, baseline):
    outdir = E1 / "bootstrap" / f"{candidate}_minus_{baseline}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(ANALYSIS), "--output-dir", str(outdir),
           "--datasets", *DATASETS, "--reps", "20000", "--seed", "20260731",
           "--baseline-label", baseline, "--candidate-label", candidate]
    for seed in SEEDS:
        cmd += ["--simple-pair", str(run_dir(baseline, seed)), str(run_dir(candidate, seed))]
    subprocess.run(cmd, check=True, cwd=str(ANALYSIS.parent))
    files = sorted(outdir.glob("*.json"))
    if not files:
        raise RuntimeError(f"bootstrap wrote no json in {outdir}")
    return load(files[0]), files[0]


def main():
    lines = []
    problems = []
    lines.append("# E1-mini result: matched RSQ attention-concentration Density control\n")
    lines.append("Raw matched results and integrity checks only (no interpretation). "
                 "Protocol: E1_PROTOCOL.md (frozen before any evaluation). Model Qwen3-ASR-0.6B, "
                 "text backbone W3/G128, deployment Map, full Support, calibration LibriSpeech "
                 "train.clean.100 (128 utterances), draws = calibration seeds 20260729/20260730/20260731, "
                 "quantization seed 20260729, evaluation internal-3 (librispeech-other, voxpopuli, "
                 "gigaspeech, first 3,000 rows per split as in the paper), raw corpus WER in %, macro = equal-domain mean. "
                 "Uniform and TIM are RECONSTRUCTED on this machine (no original artifacts exist here; protocol section 1). "
                 "Execution amendments recorded before any WER was read: protocol sections 6 (two arms per GPU) and 7 (--eval-samples 3000 relaunch). "
                 "Contrast columns are candidate minus baseline in percentage points; a 95% interval excluding zero is resolved.\n")

    # --- terminal status ---
    lines.append("## Terminal status (status.json)\n")
    lines.append("| Arm | Draw 1 (c20260729) | Draw 2 (c20260730) | Draw 3 (c20260731) |")
    lines.append("|---|---|---|---|")
    complete = True
    for arm, label in ARMS:
        row = [status(arm, s) for s in SEEDS]
        if any(r != "completed" for r in row):
            complete = False
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    lines.append("")
    if not complete:
        lines.append("**Not all 12 artifacts are terminal-completed; the tables below cover only completed cells and no bootstrap is run.**\n")

    # --- WER table ---
    wer = {}
    meta = {}
    for arm, _ in ARMS:
        for s in SEEDS:
            if status(arm, s) == "completed":
                wer[(arm, s)], meta[(arm, s)] = per_dataset_wer(arm, s)

    def cell(arm, s):
        return f"{macro(wer[(arm, s)]):.3f}" if (arm, s) in wer else "n/a"

    boot = {}
    if complete:
        for cand, base in CONTRASTS:
            boot[(cand, base)], _ = bootstrap(cand, base)

    PP = 100.0  # bootstrap JSON stores WER ratios; multiply by 100 exactly once (paper App. G)

    def delta_ci(cand, base):
        if (cand, base) not in boot:
            return "n/a", "n/a"
        b = boot[(cand, base)]["macro"]
        est = b['delta_draw_mean'] if 'delta_draw_mean' in b else b['estimate']
        return f"{PP * est:+.3f}", f"[{PP * b['ci95_lower']:+.3f}, {PP * b['ci95_upper']:+.3f}]"

    lines.append("## Internal-3 macro WER (%) by draw, draw mean, and crossed-bootstrap contrast vs Uniform\n")
    lines.append("| Arm | Draw 1 | Draw 2 | Draw 3 | Macro WER (draw mean) | Δ vs Uniform (pp) | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm, label in ARMS:
        cells = [cell(arm, s) for s in SEEDS]
        mean = (sum(macro(wer[(arm, s)]) for s in SEEDS) / 3) if all((arm, s) in wer for s in SEEDS) else None
        mean_s = f"{mean:.3f}" if mean is not None else "n/a"
        if arm == "uniform":
            d, ci = "0", "—"
        else:
            d, ci = delta_ci(arm, "uniform")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {mean_s} | {d} | {ci} |")
    lines.append("")

    # --- additional contrasts ---
    lines.append("## Additional pre-specified contrasts (same crossed draw × utterance bootstrap, 20,000 reps, seed 20260731)\n")
    lines.append("| Contrast | Δ macro (pp) | 95% CI | Δ by draw (pp) | Δ per dataset (pp) [95% CI] |")
    lines.append("|---|---|---|---|---|")
    for cand, base in CONTRASTS:
        if (cand, base) not in boot:
            lines.append(f"| {cand} − {base} | n/a | n/a | n/a | n/a |")
            continue
        b = boot[(cand, base)]
        m = b["macro"]
        est = PP * m.get("delta_draw_mean", m.get("estimate"))
        by_draw = ", ".join(f"{PP * v:+.3f}" for v in b["macro_delta_by_draw"])
        per_ds = "; ".join(
            f"{d}: {PP * b['bootstrap'][d]['estimate']:+.3f} [{PP * b['bootstrap'][d]['ci95_lower']:+.3f}, {PP * b['bootstrap'][d]['ci95_upper']:+.3f}]"
            for d in DATASETS)
        lines.append(f"| {cand} − {base} | {est:+.3f} | [{m['ci95_lower'] * PP:+.3f}, {m['ci95_upper'] * PP:+.3f}] | {by_draw} | {per_ds} |")
    lines.append("")

    # --- per-dataset draw-level values ---
    lines.append("## Draw-level per-dataset WER (%)\n")
    lines.append("| Arm | Draw | " + " | ".join(DATASETS) + " | macro |")
    lines.append("|---|---|" + "---|" * (len(DATASETS) + 1))
    for arm, _ in ARMS:
        for i, s in enumerate(SEEDS, 1):
            if (arm, s) in wer:
                w = wer[(arm, s)]
                lines.append(f"| {arm} | {i} (c{s}) | " + " | ".join(f"{w[d]:.3f}" for d in DATASETS) + f" | {macro(w):.3f} |")
    lines.append("")

    # --- integrity checks ---
    lines.append("## Integrity checks\n")
    # calibration IDs per draw across arms
    for s in SEEDS:
        ids = {}
        for arm, _ in ARMS:
            p = run_dir(arm, s) / "calibration.json"
            if p.is_file():
                ids[arm] = load(p)["example_ids"]
        same = len({json.dumps(v) for v in ids.values()}) == 1 if ids else False
        n = {a: len(v) for a, v in ids.items()}
        lines.append(f"- Draw c{s}: calibration IDs identical across {sorted(ids)}: **{same}**; counts {n}; first IDs {list(ids.values())[0][:3] if ids else 'n/a'}")
        if not same:
            problems.append(f"calibration IDs differ in draw {s}")
    # IDs differ across draws?
    first = {}
    for s in SEEDS:
        p = run_dir("uniform", s) / "calibration.json"
        if p.is_file():
            first[s] = load(p)["example_ids"]
    if len(first) == 3:
        overlap = len(set(first[SEEDS[0]]) & set(first[SEEDS[1]]) & set(first[SEEDS[2]]))
        lines.append(f"- Calibration IDs across the three draws: pairwise overlaps "
                     f"{len(set(first[SEEDS[0]]) & set(first[SEEDS[1]]))}/"
                     f"{len(set(first[SEEDS[0]]) & set(first[SEEDS[2]]))}/"
                     f"{len(set(first[SEEDS[1]]) & set(first[SEEDS[2]]))}, three-way {overlap}.")
    # config diffs
    diff_fields = set()
    cfgs = {}
    for arm, _ in ARMS:
        for s in SEEDS:
            p = run_dir(arm, s) / "config.json"
            if p.is_file():
                cfgs[(arm, s)] = load(p)
    if cfgs:
        allk = set().union(*[c.keys() for c in cfgs.values()])
        ignore = {"run_name", "seed", "protocol", "mode", "effective_model_state", "sequence_calibration",
                  "sequence_hessian", "rsq_attention", "rsq_score_normalization", "propagated_clip_max"}
        for k in sorted(allk):
            if len({json.dumps(c.get(k), sort_keys=True) for c in cfgs.values()}) > 1:
                diff_fields.add(k)
        unexpected = sorted(diff_fields - ignore)
        lines.append(f"- config.json fields differing across the 12 runs: {sorted(diff_fields)}; "
                     f"unexpected (outside mode/weighting/seed/run-name): **{unexpected if unexpected else 'none'}**")
        if unexpected:
            problems.append(f"unexpected config differences: {unexpected}")
    # evaluation rows alignment
    align_ok = True
    for d in DATASETS:
        ref = None
        for arm, _ in ARMS:
            for s in SEEDS:
                p = run_dir(arm, s) / f"{d}.jsonl"
                if not p.is_file():
                    continue
                rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                key = [(r["example_id"], r["reference"]) for r in rows]
                if ref is None:
                    ref = key
                    lines.append(f"- {d}: {len(rows)} evaluation rows (reference run E1-{arm}-c{s})")
                elif key != ref:
                    align_ok = False
                    problems.append(f"evaluation rows differ for {d} in E1-{arm}-c{s}")
    lines.append(f"- Evaluation example IDs and references identical across all runs for all datasets: **{align_ok}**")
    # empty outputs
    lines.append("- Empty predictions per run (count / rows):")
    for arm, _ in ARMS:
        for s in SEEDS:
            parts = []
            for d in DATASETS:
                p = run_dir(arm, s) / f"{d}.jsonl"
                if p.is_file():
                    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                    empty = sum(1 for r in rows if not r["prediction"].strip())
                    parts.append(f"{d} {empty}/{len(rows)}")
            if parts:
                lines.append(f"  - E1-{arm}-c{s}: " + "; ".join(parts))
    # environment
    envs = {}
    for arm, _ in ARMS:
        for s in SEEDS:
            p = run_dir(arm, s) / "environment.json"
            if p.is_file():
                e = load(p)
                envs[(arm, s)] = (e.get("git_commit"), e.get("source_tree_sha256"), e.get("torch"), e.get("transformers"))
    if envs:
        uniq = {v for v in envs.values()}
        lines.append(f"- environment.json (git_commit, source_tree_sha256, torch, transformers) identical across runs: **{len(uniq) == 1}**; value {next(iter(uniq))} (git_commit is 'unknown' because the launcher's working directory is not the repository; the source-tree sha256 is the code identity, and E1/e1_rsqattn.patch is the diff against tim-gptq commit 12d30a5)")
    # weighting metadata summaries
    lines.append("- Weighting metadata (quantization.json), per run, averaged over the 28 text layers:")
    for arm, _ in ARMS:
        for s in SEEDS:
            p = run_dir(arm, s) / "quantization.json"
            if not p.is_file():
                continue
            q = load(p)
            if "rsq_attention_control" in q:
                L = q["rsq_attention_control"]["layer_weight_summaries"]
                n = len(L)
                wmax = max(v["weight_max"] for v in L.values())
                wmin = min(v["weight_min"] for v in L.values())
                ess = sum(v["ess_fraction_mean"] for v in L.values()) / n
                sat = sum(v["saturated_upper_fraction"] for v in L.values()) / n
                dev = max(v["max_attention_row_sum_deviation"] for v in L.values())
                deg = sum(v["degenerate_samples"] for v in L.values())
                lines.append(f"  - E1-{arm}-c{s}: layers {n}, w in [{wmin:.4f}, {wmax:.4f}], mean ESS/T {ess:.3f}, upper-saturation {sat:.4f}, max attn row-sum dev {dev:.4f}, degenerate samples {deg}")
            elif "sequence_hessian" in q:
                L = q["sequence_hessian"]["layer_weight_summaries"]
                n = len(L)
                wmax = max(v["maximum"] for v in L.values())
                wmin = min(v["minimum"] for v in L.values())
                lines.append(f"  - E1-{arm}-c{s}: seqhess layers {n}, w in [{wmin:.4f}, {wmax:.4f}], clipping {q['sequence_hessian'].get('clipping')}")
            else:
                lines.append(f"  - E1-{arm}-c{s}: uniform weights (no weighting block); keys {sorted(q.keys())}")
    lines.append("")

    # --- wall time ---
    lines.append("## Wall time (from launcher timeline and metrics.json; no separate timing runs)\n")
    lines.append("All 12 arms ran two-per-GPU under the locked work queue (protocol sections 6-7); wall times are not single-tenant and are not comparable across arms or with the paper's Appendix C.3.\n")
    timeline = (E1 / "logs" / "timeline.log").read_text() if (E1 / "logs" / "timeline.log").is_file() else ""
    walls = {}
    for m in re.finditer(r"END\s+gpu=(\d)\s+(?:worker=\S+\s+)?(E1-\S+)\s+rc=(\d+)\s+wall_s=(\d+)", timeline):
        walls[m.group(2)] = (int(m.group(4)), m.group(1), m.group(3))
    lines.append("| Run | GPU | rc | launcher wall (s) | quantization_seconds (calibration+GPTQ) | total_seconds |")
    lines.append("|---|---|---|---|---|---|")
    for arm, _ in ARMS:
        for s in SEEDS:
            name = f"E1-{arm}-qwen06-w3-c{s}"
            w = walls.get(name)
            mm = meta.get((arm, s))
            qs = f"{mm['quantization_seconds']:.1f}" if mm else "n/a"
            ts = f"{mm['total_seconds']:.1f}" if mm and "total_seconds" in mm else "n/a"
            lines.append(f"| {name} | {w[1] if w else 'n/a'} | {w[2] if w else 'n/a'} | {w[0] if w else 'n/a'} | {qs} | {ts} |")
    lines.append("")

    # --- run directories ---
    lines.append("## Run directories\n")
    for arm, _ in ARMS:
        for s in SEEDS:
            lines.append(f"- {run_dir(arm, s)}")
    lines.append("")
    lines.append("Bootstrap outputs: " + str(E1 / "bootstrap") + " (one JSON + Markdown per contrast).")
    lines.append("Code: tim-gptq @ 12d30a5 + E1/e1_rsqattn.patch (sha256 in E1_PROTOCOL.md section 1 / patch file).")
    if problems:
        lines.append("\n## PROBLEMS DETECTED\n")
        for p in problems:
            lines.append(f"- {p}")
    out = E1 / "E1_RSQ_REBUTTAL_RESULT.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
