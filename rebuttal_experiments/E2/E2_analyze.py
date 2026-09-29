#!/usr/bin/env python3
"""Assemble E2_MT_REBUTTAL_RESULT.md from the E2 run directories. Raw results + integrity only."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import sacrebleu

E2 = Path(__file__).resolve().parent
RUNS = E2 / "runs"
SEEDS = [20260729, 20260730, 20260731]
MAPS = ["minimal", "deployment"]
DENS = ["uniform", "tim"]
ARMS = [(m, d) for m in MAPS for d in DENS]
BOOT_REPS = 1000
BOOT_SEED = 20260731


def rd(m, d, s):
    return RUNS / f"E2-{m}-{d}-c{s}"


def load(p):
    return json.loads(Path(p).read_text())


def status(p):
    q = Path(p) / "status.json"
    return load(q)["state"] if q.is_file() else "missing"


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


from sacrebleu.metrics import CHRF, BLEU
_CHRF = CHRF(word_order=2)
_BLEU = BLEU()


def corpus_scores(hyps, refs):
    return (sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score,
            sacrebleu.corpus_bleu(hyps, [refs]).score)


def sentence_stats(hyps, refs):
    """Per-sentence sufficient statistics (sacrebleu internals, as used by its own paired bootstrap)."""
    c = np.asarray(_CHRF._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)
    b = np.asarray(_BLEU._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)
    return c, b


def scores_from_stats(c_stats_sum, b_stats_sum):
    chrf = _CHRF._compute_score_from_stats(list(c_stats_sum)).score
    bleu = _BLEU._compute_score_from_stats(list(b_stats_sum)).score
    return chrf, bleu


def main():
    L = []
    problems = []
    L.append("# E2 result: TIM Map x Density existence check on FLORES-200 eng->deu (Qwen3-0.6B, W3/G128)\n")
    L.append("Raw results and integrity checks only; no interpretation. Protocol: E2_PROTOCOL.md (frozen before evaluation). "
             "Four conditions x three calibration draws (seeds 20260729/20260730/20260731) = 12 artifacts; quantization seed 20260729; "
             "Support = prompt + teacher-forced target for all arms; evaluation = FLORES-200 devtest (1,012 rows, disjoint from the dev calibration pool), "
             "deployment chat template, greedy decoding; corpus chrF++ and BLEU (sacrebleu). Contrasts are candidate minus baseline in metric points. "
             "The requested 'Qwen3-0.6B-Instruct' does not exist on the Hub; Qwen/Qwen3-0.6B (the post-trained chat model) is used with enable_thinking=False.\n")

    # status
    L.append("## Terminal status\n")
    L.append("| Arm | Draw 1 | Draw 2 | Draw 3 |")
    L.append("|---|---|---|---|")
    complete = True
    for m, d in ARMS:
        row = [status(rd(m, d, s)) for s in SEEDS]
        complete &= all(r == "completed" for r in row)
        L.append(f"| {m}/{d} | " + " | ".join(row) + " |")
    ref_dir = RUNS / "E2-bf16-reference"
    ref_status = status(ref_dir)
    L.append(f"| BF16 reference (context only) | {ref_status} | | |")
    L.append("")
    if not complete:
        L.append("**Not all 12 artifacts are completed; tables cover completed cells only and no bootstrap is run.**\n")

    # metrics
    M = {}
    for m, d in ARMS:
        for s in SEEDS:
            if status(rd(m, d, s)) == "completed":
                M[(m, d, s)] = load(rd(m, d, s) / "metrics.json")
    ref = load(ref_dir / "metrics.json") if ref_status == "completed" else None

    def cell(m, d, s, k):
        return f"{M[(m, d, s)][k]:.2f}" if (m, d, s) in M else "n/a"

    def mean(m, d, k):
        v = [M[(m, d, s)][k] for s in SEEDS if (m, d, s) in M]
        return (sum(v) / len(v)) if len(v) == 3 else None

    for k, label in [("chrf++", "chrF++"), ("bleu", "BLEU")]:
        L.append(f"## {label} (devtest, corpus level) per draw and draw mean\n")
        L.append("| Arm | Draw 1 | Draw 2 | Draw 3 | Mean |")
        L.append("|---|---|---|---|---|")
        if ref:
            L.append(f"| BF16 reference (no calibration) | {ref[k]:.2f} | | | {ref[k]:.2f} |")
        for m, d in ARMS:
            mu = mean(m, d, k)
            L.append(f"| {m}/{d} | " + " | ".join(cell(m, d, s, k) for s in SEEDS) + f" | {mu:.2f} |" if mu is not None else f"| {m}/{d} | " + " | ".join(cell(m, d, s, k) for s in SEEDS) + " | n/a |")
        L.append("")

    # contrasts
    CONTRASTS = [
        ("Map under Uniform: deployment/uniform - minimal/uniform", ("deployment", "uniform"), ("minimal", "uniform")),
        ("Density under minimal Map: minimal/tim - minimal/uniform", ("minimal", "tim"), ("minimal", "uniform")),
        ("Density under deployment Map: deployment/tim - deployment/uniform", ("deployment", "tim"), ("deployment", "uniform")),
        ("Map under TIM: deployment/tim - minimal/tim", ("deployment", "tim"), ("minimal", "tim")),
    ]
    boot = {}
    if complete:
        rng = np.random.default_rng(BOOT_SEED)
        # per-sentence sufficient statistics, computed once per run; verified against corpus scores
        ST = {}
        for m, d in ARMS:
            for s in SEEDS:
                rows = read_jsonl(rd(m, d, s) / "devtest.jsonl")
                hy = [r["hypothesis"] for r in rows]
                rf = [r["reference"] for r in rows]
                c, b = sentence_stats(hy, rf)
                chk = scores_from_stats(c.sum(axis=0), b.sum(axis=0))
                full = corpus_scores(hy, rf)
                if abs(chk[0] - full[0]) > 1e-6 or abs(chk[1] - full[1]) > 1e-6:
                    raise RuntimeError(f"stat aggregation mismatch for {m}/{d}/{s}: {chk} vs {full}")
                ST[(m, d, s)] = (c, b)
        n = ST[ARMS[0] + (SEEDS[0],)][0].shape[0]
        for label, cand, base in CONTRASTS:
            samples = {"chrf++": [], "bleu": []}
            for _ in range(BOOT_REPS):
                idx = rng.integers(0, n, size=n)
                draws = rng.integers(0, 3, size=3)
                dc = {"chrf++": 0.0, "bleu": 0.0}
                for di in draws:
                    s = SEEDS[di]
                    cc, cb = ST[cand + (s,)]
                    bc, bb = ST[base + (s,)]
                    c_chrf, c_bleu = scores_from_stats(cc[idx].sum(axis=0), cb[idx].sum(axis=0))
                    b_chrf, b_bleu = scores_from_stats(bc[idx].sum(axis=0), bb[idx].sum(axis=0))
                    dc["chrf++"] += (c_chrf - b_chrf) / 3
                    dc["bleu"] += (c_bleu - b_bleu) / 3
                for k in samples:
                    samples[k].append(dc[k])
            boot[label] = {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in samples.items()}

    for k, label in [("chrf++", "chrF++"), ("bleu", "BLEU")]:
        L.append(f"## Pre-specified contrasts, {label} (candidate - baseline, points)\n")
        L.append("| Contrast | Draw 1 | Draw 2 | Draw 3 | Mean | Signs consistent | Crossed draw x sentence bootstrap 95% (1,000 reps) |")
        L.append("|---|---|---|---|---|---|---|")
        for clabel, cand, base in CONTRASTS:
            ds = []
            for s in SEEDS:
                if (cand + (s,)) in M and (base + (s,)) in M:
                    ds.append(M[cand + (s,)][k] - M[base + (s,)][k])
            if len(ds) == 3:
                signs = {np.sign(x) for x in ds}
                cons = "yes" if len(signs) == 1 and 0 not in signs else "no"
                ci = boot.get(clabel, {}).get(k)
                ci_s = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "n/a"
                L.append(f"| {clabel} | " + " | ".join(f"{x:+.2f}" for x in ds) + f" | {sum(ds)/3:+.2f} | {cons} | {ci_s} |")
            else:
                L.append(f"| {clabel} | n/a | n/a | n/a | n/a | n/a | n/a |")
        L.append("")

    # integrity
    L.append("## Integrity checks\n")
    for s in SEEDS:
        idxs = {}
        for m, d in ARMS:
            p = rd(m, d, s) / "calibration.json"
            if p.is_file():
                idxs[f"{m}/{d}"] = load(p)["dev_indices"]
        same = len({json.dumps(v) for v in idxs.values()}) == 1 if idxs else False
        L.append(f"- Draw c{s}: identical 128 dev indices across {sorted(idxs)}: **{same}**; first indices {list(idxs.values())[0][:5] if idxs else 'n/a'}")
        if not same:
            problems.append(f"dev indices differ in draw {s}")
    sets = []
    for s in SEEDS:
        p = rd("minimal", "uniform", s) / "calibration.json"
        if p.is_file():
            sets.append(set(load(p)["dev_indices"]))
    if len(sets) == 3:
        L.append(f"- Dev-index overlap between draws: {len(sets[0]&sets[1])}/{len(sets[0]&sets[2])}/{len(sets[1]&sets[2])} (pairwise), {len(sets[0]&sets[1]&sets[2])} (all three).")
    # devtest identity
    ref_rows = None
    ok = True
    for m, d in ARMS:
        for s in SEEDS:
            p = rd(m, d, s) / "devtest.jsonl"
            if p.is_file():
                rows = [(r["index"], r["source"], r["reference"]) for r in read_jsonl(p)]
                if ref_rows is None:
                    ref_rows = rows
                    L.append(f"- devtest rows per run: {len(rows)} (reference run E2-{m}-{d}-c{s})")
                elif rows != ref_rows:
                    ok = False
                    problems.append(f"devtest rows differ in E2-{m}-{d}-c{s}")
    L.append(f"- devtest sources/references identical across all runs: **{ok}**")
    # config diffs
    cfgs = {}
    for m, d in ARMS:
        for s in SEEDS:
            p = rd(m, d, s) / "config.json"
            if p.is_file():
                cfgs[(m, d, s)] = load(p)
    if cfgs:
        allk = set().union(*[c.keys() for c in cfgs.values()])
        diff = sorted(k for k in allk if len({json.dumps(c.get(k), sort_keys=True) for c in cfgs.values()}) > 1)
        unexpected = sorted(set(diff) - {"map", "density", "seed", "run_name"})
        L.append(f"- config.json fields differing across runs: {diff}; unexpected: **{unexpected if unexpected else 'none'}**")
        if unexpected:
            problems.append(f"unexpected config differences: {unexpected}")
        c0 = next(iter(cfgs.values()))
        L.append(f"- data file sha256: {c0['data_files_sha256']}")
    envs = {json.dumps(load(rd(m, d, s) / "environment.json"), sort_keys=True) for m, d in ARMS for s in SEEDS if (rd(m, d, s) / "environment.json").is_file()}
    L.append(f"- environment.json identical across runs: **{len(envs) == 1}**" + (f"; {next(iter(envs))}" if len(envs) == 1 else ""))
    L.append("- Empty hypotheses / quantization seconds / total seconds per run:")
    for m, d in ARMS:
        for s in SEEDS:
            if (m, d, s) in M:
                mm = M[(m, d, s)]
                L.append(f"  - E2-{m}-{d}-c{s}: empty {mm['empty_hypotheses']}/{mm['num_examples']}, quantization {mm['quantization_seconds']:.1f}s, total {mm['total_seconds']:.1f}s")
    L.append("- TIM weight summaries (quantization.json), min/max weight and mean ESS/T over the 28 layers:")
    for m in MAPS:
        for s in SEEDS:
            p = rd(m, "tim", s) / "quantization.json"
            if p.is_file():
                q = load(p)
                S = q["layer_weight_summaries"]
                L.append(f"  - E2-{m}-tim-c{s}: w in [{min(v['minimum'] for v in S.values()):.4f}, {max(v['maximum'] for v in S.values()):.4f}], mean ESS/T {sum(v['ess_fraction_mean'] for v in S.values())/len(S):.3f}, mean teacher-forced loss {q['mean_teacher_forced_loss']:.4f}")
    L.append("")
    L.append("## Run directories\n")
    for m, d in ARMS:
        for s in SEEDS:
            L.append(f"- {rd(m, d, s)}")
    L.append(f"- {ref_dir}")
    if problems:
        L.append("\n## PROBLEMS DETECTED\n")
        L += [f"- {p}" for p in problems]
    out = E2 / "E2_MT_REBUTTAL_RESULT.md"
    out.write_text("\n".join(L) + "\n")
    print(out)
    print("\n".join(L))


if __name__ == "__main__":
    main()
