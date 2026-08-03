#!/usr/bin/env python
"""Audio-native error mode analyzer for TIM-GPTQ ASR quantization runs.

For every run x dataset under runs/<suite>/<run>/ this script computes, from the
per-utterance jsonl transcripts ({"index","example_id","reference","prediction"}):

  1. S/D/I decomposition   - substitution / deletion / insertion rates, each
                             normalized by total reference words (jiwer alignment).
  2. Generation-collapse    - per-utterance labels for autoregressive-decoding
     detection                failure modes that aggregate WER hides:
                               * halluc_loop : an n-gram (n>=3) repeated >=3 times
                                 consecutively in the prediction (and not in the
                                 reference - guard against natural repetition),
                                 OR len(pred)/len(ref) > 2
                               * truncation  : len(pred)/len(ref) < 0.5 AND
                                 per-utt deletion rate > 0.5
                               * empty       : empty prediction
                               * normal      : everything else (incl. correct)
                             plus the length-ratio distribution (p10/p50/p90).
  3. Rare-word error rate  - global reference word frequencies are counted once
                             over all fp16 runs (cached to ref_word_freq.json);
                             reference words are bucketed into high-freq
                             (top 5000) vs tail; per-bucket error rate = fraction
                             of bucket ref words aligned as substitute/delete.
                             (Insertions are not attributable to a ref word.)
  4. Flip analysis         - each quantized run is paired per-utterance (by
                             example_id) against the fp16 run of the same model;
                             a "flip" is wer_quant - wer_fp16 > 0.5 (fp16 fine,
                             quantized collapsed). Reports flip rate and the
                             error-mode distribution of flipped utterances.

Outputs: analysis/error_modes_summary.csv and analysis/ERROR_MODES.md.

The script is idempotent and incremental: runs without metrics.json are skipped,
only datasets listed in metrics.json (i.e. completed evaluations) are processed,
and per-(run,dataset) results are cached in analysis/cache/ keyed by jsonl
mtime/size + frequency-table hash, so re-running after more runs finish (e.g.
w3-matrix) only computes the new cells.

Usage:
  python analysis/error_modes.py
  ... [--suites w4-headroom-repro w3-matrix] [--rebuild-freq] [--no-cache]
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import jiwer

WORKBENCH = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = WORKBENCH / "runs"
DEFAULT_OUT_DIR = WORKBENCH / "analysis"

TOP_K = 5000
NGRAM_MIN = 3
NGRAM_MAX = 20
REPEAT_MIN = 3
HALLUC_LEN_RATIO = 2.0
TRUNC_LEN_RATIO = 0.5
TRUNC_DEL_RATE = 0.5
FLIP_DELTA = 0.5

MODES = ["normal", "halluc_loop", "truncation", "empty"]

CACHE_VERSION = 4  # bump to invalidate all caches when analysis logic changes
FLIP_MIN_REF_LEN = 5  # supplementary flip rate restricted to refs >= this many words


# ---------------------------------------------------------------- discovery

def discover_runs(runs_root, suites):
    """Yield dicts describing runs that have a metrics.json (completed or
    partially completed)."""
    runs = []
    if suites:
        suite_dirs = [runs_root / s for s in suites]
    else:
        suite_dirs = sorted(
            d for d in runs_root.iterdir() if d.is_dir() and d.name != "smoke"
        )
    for suite_dir in suite_dirs:
        if not suite_dir.is_dir():
            print(f"[warn] suite dir missing, skipping: {suite_dir}")
            continue
        for run_dir in sorted(suite_dir.iterdir()):
            metrics_path = run_dir / "metrics.json"
            config_path = run_dir / "config.json"
            if not run_dir.is_dir() or not metrics_path.exists():
                continue
            if not config_path.exists():
                print(f"[warn] no config.json, skipping: {run_dir}")
                continue
            try:
                config = json.loads(config_path.read_text())
                metrics = json.loads(metrics_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"[warn] unreadable config/metrics ({e}), skipping: {run_dir}")
                continue
            datasets = {}
            for ev in metrics.get("evaluations", []):
                ds = ev["dataset"]
                jsonl = run_dir / f"{ds}.jsonl"
                if jsonl.exists():
                    datasets[ds] = {"jsonl": jsonl, "wer_reported": ev.get("wer"),
                                    "num_examples": ev.get("num_examples")}
            runs.append({
                "suite": suite_dir.name,
                "run": run_dir.name,
                "run_dir": run_dir,
                "model": config.get("model", ""),
                "model_alias": config.get("model_alias", run_dir.name),
                "method": config.get("method", config.get("mode", "")),
                "scope": config.get("quant_scope", ""),
                "wbits": config.get("wbits", ""),
                "datasets": datasets,
            })
    return runs


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate a torn final line of a file being written right now
                print(f"[warn] bad jsonl line in {path}, ignoring rest")
                break
    return records


# ---------------------------------------------------------- word frequencies

def build_freq_table(runs, cache_path, rebuild=False):
    """Count reference word frequencies once over all fp16 runs, deduplicated
    by (dataset, example_id). Cached; auto-rebuilds when the set of available
    fp16 (run, dataset) sources changes."""
    sources = []
    for r in runs:
        if r["method"] != "fp16":
            continue
        for ds, info in sorted(r["datasets"].items()):
            st = info["jsonl"].stat()
            sources.append(f"{r['suite']}/{r['run']}/{ds}:{st.st_size}")
    sources = sorted(sources)

    if cache_path.exists() and not rebuild:
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("sources") == sources and cached.get("top_k") == TOP_K:
                print(f"[freq] using cached table: {cache_path}")
                return Counter(cached["counts"]), cached["fingerprint"]
        except (json.JSONDecodeError, OSError):
            pass

    print(f"[freq] building reference word-frequency table from "
          f"{len(sources)} fp16 run-datasets ...")
    counts = Counter()
    seen = set()
    for r in runs:
        if r["method"] != "fp16":
            continue
        for ds, info in sorted(r["datasets"].items()):
            for rec in load_jsonl(info["jsonl"]):
                key = (ds, rec.get("example_id", rec.get("id", rec.get("index"))))
                if key in seen:
                    continue
                seen.add(key)
                counts.update(rec["reference"].split())
    fingerprint = hashlib.md5(
        json.dumps(sources, sort_keys=True).encode()).hexdigest()[:12]
    cache_path.write_text(json.dumps({
        "sources": sources, "top_k": TOP_K, "fingerprint": fingerprint,
        "n_unique_refs": len(seen), "vocab_size": len(counts),
        "counts": dict(counts),
    }))
    print(f"[freq] {len(seen)} unique references, vocab {len(counts)}, "
          f"cached to {cache_path}")
    return counts, fingerprint


def high_freq_set(counts, top_k=TOP_K):
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return set(w for w, _ in ranked[:top_k])


# --------------------------------------------------------- per-utt analysis

def has_repeat_loop(tokens, nmin=NGRAM_MIN, nmax=NGRAM_MAX, repeat=REPEAT_MIN):
    """True if some n-gram (nmin<=n<=nmax) occurs >= `repeat` times
    consecutively."""
    L = len(tokens)
    max_n = min(nmax, L // repeat)
    for n in range(nmin, max_n + 1):
        limit = L - n * repeat + 1
        for i in range(limit):
            seg = tokens[i:i + n]
            if all(tokens[i + k * n:i + (k + 1) * n] == seg
                   for k in range(1, repeat)):
                return True
    return False


def classify_mode(ref_toks, pred_toks, del_rate):
    if len(pred_toks) == 0:
        return "empty"
    ratio = len(pred_toks) / len(ref_toks)
    if ratio > HALLUC_LEN_RATIO:
        return "halluc_loop"
    if has_repeat_loop(pred_toks) and not has_repeat_loop(ref_toks):
        return "halluc_loop"
    if ratio < TRUNC_LEN_RATIO and del_rate > TRUNC_DEL_RATE:
        return "truncation"
    return "normal"


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)


def analyze_dataset(records, hi_words):
    """Full per-utterance analysis for one run x dataset.

    Returns (summary_dict, per_utt) where per_utt maps example_id ->
    (per-utt WER, mode) for later flip pairing."""
    refs, preds, ids = [], [], []
    for rec in records:
        ref = rec.get("reference", "").strip()
        if not ref:
            continue  # cannot score an empty reference
        refs.append(ref)
        preds.append(rec.get("prediction", "").strip())
        ids.append(str(rec.get("example_id", rec.get("id", rec.get("index")))))

    n = len(refs)
    tot = {"ref_words": 0, "S": 0, "D": 0, "I": 0}
    mode_counts = Counter()
    len_ratios = []
    bucket = {"hi": [0, 0], "tail": [0, 0]}  # [errors, total ref words]
    per_utt = {}

    # jiwer chokes on empty hypothesis strings inside a batch on some
    # versions; substitute a placeholder and treat those utterances as
    # all-deletions manually.
    empty_idx = set(i for i, p in enumerate(preds) if not p)
    safe_preds = [p if p else "<empty>" for p in preds]
    out = jiwer.process_words(refs, safe_preds)

    for i in range(n):
        ref_toks = refs[i].split()
        pred_toks = preds[i].split()
        R = len(ref_toks)
        if i in empty_idx:
            S, D, I = 0, R, 0
            err_words = set(range(R))
        else:
            S = D = I = 0
            err_words = set()
            for ch in out.alignments[i]:
                span = range(ch.ref_start_idx, ch.ref_end_idx)
                if ch.type == "substitute":
                    S += ch.ref_end_idx - ch.ref_start_idx
                    err_words.update(span)
                elif ch.type == "delete":
                    D += ch.ref_end_idx - ch.ref_start_idx
                    err_words.update(span)
                elif ch.type == "insert":
                    I += ch.hyp_end_idx - ch.hyp_start_idx
        del_rate = D / R
        wer_utt = (S + D + I) / R
        mode = classify_mode(ref_toks, pred_toks, del_rate)
        mode_counts[mode] += 1
        len_ratios.append(len(pred_toks) / R)
        tot["ref_words"] += R
        tot["S"] += S
        tot["D"] += D
        tot["I"] += I
        for j, w in enumerate(ref_toks):
            b = "hi" if w in hi_words else "tail"
            bucket[b][1] += 1
            if j in err_words:
                bucket[b][0] += 1
        per_utt[ids[i]] = (round(wer_utt, 6), mode, R)

    len_ratios.sort()
    RW = max(tot["ref_words"], 1)
    hi_err, hi_tot = bucket["hi"]
    tail_err, tail_tot = bucket["tail"]
    summary = {
        "n_utts": n,
        "ref_words": tot["ref_words"],
        "wer": (tot["S"] + tot["D"] + tot["I"]) / RW,
        "sub_rate": tot["S"] / RW,
        "del_rate": tot["D"] / RW,
        "ins_rate": tot["I"] / RW,
        "frac_normal": mode_counts["normal"] / n if n else float("nan"),
        "frac_halluc_loop": mode_counts["halluc_loop"] / n if n else float("nan"),
        "frac_truncation": mode_counts["truncation"] / n if n else float("nan"),
        "frac_empty": mode_counts["empty"] / n if n else float("nan"),
        "len_ratio_p10": percentile(len_ratios, 0.10),
        "len_ratio_p50": percentile(len_ratios, 0.50),
        "len_ratio_p90": percentile(len_ratios, 0.90),
        "hi_freq_words": hi_tot,
        "tail_words": tail_tot,
        "hi_freq_err_rate": hi_err / hi_tot if hi_tot else float("nan"),
        "tail_err_rate": tail_err / tail_tot if tail_tot else float("nan"),
    }
    summary["tail_to_hi_ratio"] = (
        summary["tail_err_rate"] / summary["hi_freq_err_rate"]
        if hi_tot and tail_tot and summary["hi_freq_err_rate"] > 0
        else float("nan"))
    return summary, per_utt


# ----------------------------------------------------------------- caching

def cache_key(jsonl_path, freq_fp):
    st = jsonl_path.stat()
    return f"v{CACHE_VERSION}|{st.st_size}|{int(st.st_mtime)}|{freq_fp}|{TOP_K}"


def analyze_run_dataset(run, ds, info, hi_words, freq_fp, cache_dir, use_cache):
    cache_file = cache_dir / f"{run['suite']}__{run['run']}__{ds}.json"
    key = cache_key(info["jsonl"], freq_fp)
    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("key") == key:
                return cached["summary"], dict(cached["per_utt"])
        except (json.JSONDecodeError, OSError):
            pass
    records = load_jsonl(info["jsonl"])
    if info.get("num_examples") and len(records) != info["num_examples"]:
        print(f"[warn] {run['run']}/{ds}: jsonl has {len(records)} lines, "
              f"metrics.json says {info['num_examples']}")
    summary, per_utt = analyze_dataset(records, hi_words)
    if info.get("wer_reported") is not None:
        summary["wer_reported"] = info["wer_reported"]
    if use_cache:
        cache_file.write_text(json.dumps(
            {"key": key, "summary": summary,
             "per_utt": {k: list(v) for k, v in per_utt.items()}}))
    return summary, per_utt


# ------------------------------------------------------------------- flips

def flip_stats(per_utt_q, per_utt_fp, delta=FLIP_DELTA):
    common = set(per_utt_q) & set(per_utt_fp)
    n = len(common)
    if n == 0:
        return {}
    flips, rev_flips = [], 0
    n_long = n_long_flips = 0
    for eid in common:
        wq, mq, rl = per_utt_q[eid]
        wf = per_utt_fp[eid][0]
        is_flip = wq - wf > delta
        if is_flip:
            flips.append(mq)
        elif wf - wq > delta:
            rev_flips += 1
        if rl >= FLIP_MIN_REF_LEN:
            n_long += 1
            n_long_flips += is_flip
    mode_c = Counter(flips)
    nf = len(flips)
    out = {
        "n_paired": n,
        "flip_rate": nf / n,
        "flip_rate_reverse": rev_flips / n,
        "n_flips": nf,
        "n_paired_len5": n_long,
        "flip_rate_len5": n_long_flips / n_long if n_long else float("nan"),
    }
    for m in MODES:
        out[f"flip_mode_{m}"] = mode_c[m] / nf if nf else float("nan")
    return out


# ------------------------------------------------------------------ report

def fmt(x, pct=False, digits=2):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    if pct:
        return f"{100 * x:.{digits}f}"
    return f"{x:.{digits + 2}f}" if isinstance(x, float) else str(x)


def write_report(rows, out_path, freq_meta):
    """rows: list of flat summary dicts (one per run x dataset)."""
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model_alias"], []).append(r)

    L = []
    L.append("# ASR Quantization Error Modes")
    L.append("")
    L.append(f"Generated {datetime.now():%Y-%m-%d %H:%M} by `analysis/error_modes.py`. "
             f"Incremental: re-run after more runs finish to refresh.")
    n_runs = len(set((r['suite'], r['run']) for r in rows))
    L.append(f"Runs analyzed: **{n_runs}** ({', '.join(sorted(set(r['suite'] for r in rows)))}); "
             f"run x dataset cells: **{len(rows)}**.")
    L.append("")
    L.append("Definitions: S/D/I rates are substitutions/deletions/insertions over "
             "reference words (S+D+I = WER). Collapse modes per utterance: "
             "`halluc_loop` = n-gram (n>=3) repeated >=3x consecutively or "
             "len(pred)/len(ref) > 2; `truncation` = length ratio < 0.5 with "
             "deletion rate > 0.5; `empty` = empty prediction; `normal` = rest. "
             f"Rare words: reference words outside the top {TOP_K} of the global "
             "fp16-reference frequency table "
             f"(vocab {freq_meta.get('vocab_size', '?')}, "
             f"{freq_meta.get('n_unique_refs', '?')} unique refs). "
             f"Flip: per-utt WER(quant) - WER(fp16) > {FLIP_DELTA}.")
    L.append("")

    for model in sorted(by_model):
        rs = by_model[model]
        L.append(f"## {model} ({rs[0]['model']}, w{rs[0]['wbits']})")
        L.append("")
        L.append("| dataset | method | scope | WER% | S% | D% | I% | "
                 "halluc% | trunc% | empty% | lenR p10/p50/p90 | "
                 "hiWER% | tailWER% | tail/hi | flip% | flip%(ref>=5) | rev-flip% |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = sorted(rs, key=lambda r: (r["dataset"], r["method"] != "fp16",
                                          r["method"], r["scope"]))
        for r in order:
            lenr = (f"{fmt(r['len_ratio_p10'])}/{fmt(r['len_ratio_p50'])}/"
                    f"{fmt(r['len_ratio_p90'])}")
            L.append("| {ds} | {m} | {sc} | {wer} | {s} | {d} | {i} | {h} | {t} | "
                     "{e} | {lr} | {hi} | {tl} | {ratio} | {fl} | {fl5} | {rfl} |".format(
                ds=r["dataset"], m=r["method"], sc=r["scope"],
                wer=fmt(r["wer"], pct=True), s=fmt(r["sub_rate"], pct=True),
                d=fmt(r["del_rate"], pct=True), i=fmt(r["ins_rate"], pct=True),
                h=fmt(r["frac_halluc_loop"], pct=True),
                t=fmt(r["frac_truncation"], pct=True),
                e=fmt(r["frac_empty"], pct=True), lr=lenr,
                hi=fmt(r["hi_freq_err_rate"], pct=True),
                tl=fmt(r["tail_err_rate"], pct=True),
                ratio=fmt(r.get("tail_to_hi_ratio")),
                fl=fmt(r.get("flip_rate"), pct=True),
                fl5=fmt(r.get("flip_rate_len5"), pct=True),
                rfl=fmt(r.get("flip_rate_reverse"), pct=True)))
        L.append("")

        # (a) damage decomposition vs fp16, aggregated over datasets
        fp16 = {r["dataset"]: r for r in rs if r["method"] == "fp16"}
        quant_keys = sorted(set((r["method"], r["scope"]) for r in rs
                                if r["method"] != "fp16"))
        if fp16 and quant_keys:
            L.append(f"### {model}: damage decomposition vs fp16 "
                     "(macro-avg over shared datasets)")
            L.append("")
            L.append("| method | scope | dWER% | dS% | dD% | dI% | S-share | "
                     "flip% | n_flips | flips halluc/trunc/empty/normal |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for meth, sc in quant_keys:
                qs = [r for r in rs if r["method"] == meth and r["scope"] == sc
                      and r["dataset"] in fp16]
                if not qs:
                    continue
                dw = ds_ = dd = di = fl = 0.0
                fm = Counter()
                nfl = 0
                for q in qs:
                    f = fp16[q["dataset"]]
                    dw += q["wer"] - f["wer"]
                    ds_ += q["sub_rate"] - f["sub_rate"]
                    dd += q["del_rate"] - f["del_rate"]
                    di += q["ins_rate"] - f["ins_rate"]
                    fl += q.get("flip_rate") or 0.0
                    nfq = q.get("n_flips") or 0
                    nfl += nfq
                    for m in MODES:
                        v = q.get(f"flip_mode_{m}")
                        if v is not None and v == v:
                            fm[m] += v * nfq
                k = len(qs)
                dw, ds_, dd, di, fl = dw / k, ds_ / k, dd / k, di / k, fl / k
                sshare = ds_ / dw if abs(dw) > 1e-9 else float("nan")
                fdist = "/".join(
                    fmt(fm[m] / nfl if nfl else float("nan"), pct=True, digits=0)
                    for m in ["halluc_loop", "truncation", "empty", "normal"])
                L.append(f"| {meth} | {sc} | {fmt(dw, pct=True)} | "
                         f"{fmt(ds_, pct=True)} | {fmt(dd, pct=True)} | "
                         f"{fmt(di, pct=True)} | {fmt(sshare)} | "
                         f"{fmt(fl, pct=True)} | {nfl} | {fdist} |")
            L.append("")

            # (c) rare-word fragility vs fp16
            L.append(f"### {model}: rare-word fragility vs fp16 "
                     "(macro-avg over shared datasets)")
            L.append("")
            L.append("relative increase = (quant - fp16) / fp16 of the bucket "
                     "error rate; amplification = tail relative increase minus "
                     "high-freq relative increase (positive => tail words "
                     "disproportionately damaged).")
            L.append("")
            L.append("| method | scope | hi err fp16->quant | tail err fp16->quant | "
                     "hi rel-incr | tail rel-incr | amplification |")
            L.append("|---|---|---|---|---|---|---|")
            for meth, sc in quant_keys:
                qs = [r for r in rs if r["method"] == meth and r["scope"] == sc
                      and r["dataset"] in fp16]
                if not qs:
                    continue
                hi_f = hi_q = tl_f = tl_q = ri_h = ri_t = 0.0
                k = len(qs)
                for q in qs:
                    f = fp16[q["dataset"]]
                    hi_f += f["hi_freq_err_rate"]; hi_q += q["hi_freq_err_rate"]
                    tl_f += f["tail_err_rate"]; tl_q += q["tail_err_rate"]
                    if f["hi_freq_err_rate"] > 0:
                        ri_h += (q["hi_freq_err_rate"] - f["hi_freq_err_rate"]) / f["hi_freq_err_rate"]
                    if f["tail_err_rate"] > 0:
                        ri_t += (q["tail_err_rate"] - f["tail_err_rate"]) / f["tail_err_rate"]
                hi_f, hi_q, tl_f, tl_q = hi_f / k, hi_q / k, tl_f / k, tl_q / k
                ri_h, ri_t = ri_h / k, ri_t / k
                L.append(f"| {meth} | {sc} | "
                         f"{fmt(hi_f, pct=True)} -> {fmt(hi_q, pct=True)} | "
                         f"{fmt(tl_f, pct=True)} -> {fmt(tl_q, pct=True)} | "
                         f"{fmt(ri_h, pct=True, digits=1)}% | "
                         f"{fmt(ri_t, pct=True, digits=1)}% | "
                         f"{fmt(ri_t - ri_h, pct=True, digits=1)}pp |")
            L.append("")

    out_path.write_text("\n".join(L) + "\n")
    print(f"[out] report -> {out_path}")


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--suites", nargs="*", default=None,
                    help="suite dirs under runs/ (default: all except smoke)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--rebuild-freq", action="store_true",
                    help="force rebuild of the reference word-frequency table")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore and do not write per-run caches")
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    runs = discover_runs(args.runs_root, args.suites)
    n_cells = sum(len(r["datasets"]) for r in runs)
    print(f"[discover] {len(runs)} runs with metrics.json, {n_cells} "
          f"run x dataset cells")
    if not runs:
        sys.exit("no runs found")

    freq_cache = out_dir / "ref_word_freq.json"
    counts, freq_fp = build_freq_table(runs, freq_cache, rebuild=args.rebuild_freq)
    hi_words = high_freq_set(counts)
    freq_meta = {"vocab_size": len(counts),
                 "n_unique_refs": json.loads(freq_cache.read_text()).get("n_unique_refs")}

    use_cache = not args.no_cache
    # first pass: analyze everything, keep per-utt tables of fp16 runs
    results = {}   # (suite, run, ds) -> summary
    per_utts = {}  # (suite, run, ds) -> per_utt
    for r in runs:
        for ds, info in sorted(r["datasets"].items()):
            summary, per_utt = analyze_run_dataset(
                r, ds, info, hi_words, freq_fp, cache_dir, use_cache)
            results[(r["suite"], r["run"], ds)] = summary
            per_utts[(r["suite"], r["run"], ds)] = per_utt
            print(f"[run] {r['run']:45s} {ds:20s} wer={summary['wer']:.4f} "
                  f"S={summary['sub_rate']:.4f} D={summary['del_rate']:.4f} "
                  f"I={summary['ins_rate']:.4f}")

    # fp16 lookup: model_alias -> {dataset: (suite, run)}
    fp16_index = {}
    for r in runs:
        if r["method"] != "fp16":
            continue
        for ds in r["datasets"]:
            # prefer a same-suite baseline; keep first otherwise
            fp16_index.setdefault((r["model_alias"], ds), []).append(
                (r["suite"], r["run"]))

    rows = []
    for r in runs:
        for ds in sorted(r["datasets"]):
            key = (r["suite"], r["run"], ds)
            row = {
                "suite": r["suite"], "run": r["run"], "model": r["model"],
                "model_alias": r["model_alias"], "method": r["method"],
                "scope": r["scope"], "wbits": r["wbits"], "dataset": ds,
            }
            row.update(results[key])
            if r["method"] != "fp16":
                cands = fp16_index.get((r["model_alias"], ds), [])
                same_suite = [c for c in cands if c[0] == r["suite"]]
                pick = (same_suite or cands or [None])[0]
                if pick:
                    fp_key = (pick[0], pick[1], ds)
                    row["fp16_run"] = pick[1]
                    row["delta_wer_vs_fp16"] = (
                        row["wer"] - results[fp_key]["wer"])
                    row.update(flip_stats(per_utts[key], per_utts[fp_key]))
            rows.append(row)

    # ---- CSV
    cols = ["suite", "run", "model", "model_alias", "method", "scope", "wbits",
            "dataset", "n_utts", "ref_words", "wer", "wer_reported",
            "sub_rate", "del_rate", "ins_rate",
            "frac_normal", "frac_halluc_loop", "frac_truncation", "frac_empty",
            "len_ratio_p10", "len_ratio_p50", "len_ratio_p90",
            "hi_freq_words", "tail_words", "hi_freq_err_rate", "tail_err_rate",
            "tail_to_hi_ratio", "fp16_run", "delta_wer_vs_fp16",
            "n_paired", "n_flips", "flip_rate", "flip_rate_reverse",
            "n_paired_len5", "flip_rate_len5",
            "flip_mode_halluc_loop", "flip_mode_truncation",
            "flip_mode_empty", "flip_mode_normal"]
    csv_path = out_dir / "error_modes_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"[out] summary -> {csv_path} ({len(rows)} rows)")

    write_report(rows, out_dir / "ERROR_MODES.md", freq_meta)


if __name__ == "__main__":
    main()
