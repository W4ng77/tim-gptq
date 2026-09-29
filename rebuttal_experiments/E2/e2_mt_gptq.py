#!/usr/bin/env python3
"""E2: out-of-domain existence check of the TIM Map x Density decomposition on
machine translation (FLORES-200 eng_Latn -> deu_Latn) with Qwen3-0.6B, W3/G128.

The GPTQ solver (gptq.Helper / run_gptq), the quantizer (quant.py) and the bounded
ratio-preserving KL/I-projection (frame_weighting.normalize_task_fisher_weights) are
imported UNCHANGED from the paper's implementation tree. Only the data path, the two
calibration prompt Maps, and the causal-LM sequential driver are new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

SRC = str(Path(__file__).resolve().parents[2] / "implementations/whisper_qwen/src")
sys.path.insert(0, SRC)
from gptq import Helper  # noqa: E402
from frame_weighting import normalize_task_fisher_weights  # noqa: E402

import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import sacrebleu  # noqa: E402

MODEL_ID = "Qwen/Qwen3-0.6B"
GROUPS = [
    ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    ["self_attn.o_proj"],
    ["mlp.gate_proj", "mlp.up_proj"],
    ["mlp.down_proj"],
]
USER_INSTRUCTION = (
    "Translate the following English text into German. "
    "Output only the German translation.\n\n{src}"
)
KL_CLIP_MIN = 1e-4  # identical to the paper's Qwen text path (_normalize_qwen_sequence_token_weights)
KL_CLIP_MAX = 2.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", choices=("minimal", "deployment"), required=True)
    p.add_argument("--density", choices=("uniform", "tim", "bf16-reference"), required=True)
    p.add_argument("--seed", type=int, required=True, help="calibration draw seed")
    p.add_argument("--quantization-seed", type=int, default=20260729)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--wbits", type=int, default=3)
    p.add_argument("--groupsize", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data/flores200_dataset"))
    p.add_argument("--eval-samples", type=int, default=-1)
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", required=True)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def load_flores(data_dir: Path):
    dev_src = read_lines(data_dir / "dev" / "eng_Latn.dev")
    dev_tgt = read_lines(data_dir / "dev" / "deu_Latn.dev")
    test_src = read_lines(data_dir / "devtest" / "eng_Latn.devtest")
    test_tgt = read_lines(data_dir / "devtest" / "deu_Latn.devtest")
    if len(dev_src) != len(dev_tgt) or len(test_src) != len(test_tgt):
        raise ValueError("FLORES source/target line counts differ.")
    return list(zip(dev_src, dev_tgt)), list(zip(test_src, test_tgt))


def deployment_prompt(tok, src: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": USER_INSTRUCTION.format(src=src)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_sample(tok, src: str, tgt: str, map_mode: str, device) -> dict:
    if map_mode == "deployment":
        prompt = deployment_prompt(tok, src)
        full = prompt + tgt + tok.eos_token
    elif map_mode == "minimal":
        prompt = f"English: {src}\nGerman:"
        full = prompt + " " + tgt + tok.eos_token
    else:
        raise ValueError(map_mode)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full_ids = tok(full, return_tensors="pt", add_special_tokens=False).input_ids
    n = prompt_ids.shape[1]
    if full_ids.shape[1] <= n or not torch.equal(prompt_ids, full_ids[:, :n]):
        raise ValueError("Teacher-forced input does not preserve the prompt prefix.")
    labels = full_ids.clone()
    labels[:, :n] = -100
    return {
        "input_ids": full_ids.to(device),
        "labels": labels.to(device),
        "prompt_tokens": int(n),
        "total_tokens": int(full_ids.shape[1]),
    }


@torch.no_grad()
def _catch_layer0(model, samples, device):
    layers = model.model.layers
    orig = layers[0]
    store = {}

    class Catcher(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.attention_type = getattr(inner, "attention_type", "full_attention")

        def forward(self, hidden_states, **kwargs):
            store["hidden_states"] = hidden_states.detach()
            store["kwargs"] = {
                k: (v.detach() if torch.is_tensor(v) else v) for k, v in kwargs.items()
            }
            raise RuntimeError("catcher stop")

    cache = []
    layers[0] = Catcher(orig)
    try:
        for s in samples:
            store.clear()
            try:
                model(input_ids=s["input_ids"], use_cache=False)
            except RuntimeError as e:
                if "catcher stop" not in str(e):
                    raise
            kw = dict(store["kwargs"])
            pe = kw.get("position_embeddings")
            if pe is not None:
                kw["position_embeddings"] = tuple(t.detach().cpu() for t in pe)
            kw = {
                k: (v.cpu() if torch.is_tensor(v) else v) for k, v in kw.items()
            }
            kw["past_key_values"] = None
            kw["use_cache"] = False
            cache.append({"hidden_states": store["hidden_states"].cpu(), "kwargs": kw})
    finally:
        layers[0] = orig
    return cache


def _kw_to(kw: dict, device):
    out = {}
    for k, v in kw.items():
        if k == "position_embeddings" and v is not None:
            out[k] = tuple(t.to(device) for t in v)
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def collect_tim_weights(model, samples, device):
    """Paper Qwen text path: g_t = mean_h (dL_TF / d h_t)^2 at each layer's mlp.down_proj
    output, one teacher-forced forward/backward per sample, then the bounded
    ratio-preserving KL/I-projection to unit mean with final weights in [1e-4, 2]."""
    layers = model.model.layers
    by_layer = [[] for _ in layers]
    summaries = [
        {"minimum": float("inf"), "maximum": 0.0, "sum": 0.0, "count": 0, "ess_sum": 0.0, "samples": 0}
        for _ in layers
    ]
    losses = []
    for s in samples:
        captured = {}
        handles = []

        def make_hook(i):
            def hook(_m, _i, out):
                out.retain_grad()
                captured[i] = out

            return hook

        for i, layer in enumerate(layers):
            handles.append(layer.mlp.down_proj.register_forward_hook(make_hook(i)))
        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out = model(input_ids=s["input_ids"], labels=s["labels"], use_cache=False)
                out.loss.backward()
            losses.append(float(out.loss.detach()))
            for i in range(len(layers)):
                g = captured[i].grad
                if g is None:
                    raise RuntimeError(f"missing gradient at layer {i}")
                score = g.detach().float().square().mean(dim=-1).reshape(-1)
                w = normalize_task_fisher_weights(score, clip_min=KL_CLIP_MIN, clip_max=KL_CLIP_MAX).cpu()
                by_layer[i].append(w)
                sm = summaries[i]
                sm["minimum"] = min(sm["minimum"], float(w.min()))
                sm["maximum"] = max(sm["maximum"], float(w.max()))
                sm["sum"] += float(w.sum())
                sm["count"] += int(w.numel())
                sm["ess_sum"] += float(1.0 / (1.0 + w.sub(1.0).square().mean()))
                sm["samples"] += 1
        finally:
            for h in handles:
                h.remove()
            model.zero_grad(set_to_none=True)
    for sm in summaries:
        sm["mean"] = sm.pop("sum") / max(sm["count"], 1)
        sm["ess_fraction_mean"] = sm.pop("ess_sum") / max(sm["samples"], 1)
    torch.cuda.empty_cache()
    return by_layer, summaries, sum(losses) / max(len(losses), 1)


@torch.no_grad()
def quantize_sequential(model, samples, weights, args, device, log):
    layers = model.model.layers
    cache = _catch_layer0(model, samples, device)
    n_modules = 0
    for lid, layer in enumerate(layers):
        layer.to(device)
        mods = dict(layer.named_modules())
        for group in GROUPS:
            present = [n for n in group if n in mods]
            if not present:
                continue
            primary = mods[present[0]]
            helper = Helper(primary)
            latest = {"x": None}

            def hook(_m, inp, _o):
                latest["x"] = inp[0].detach()

            h = primary.register_forward_hook(hook)
            try:
                for i, c in enumerate(cache):
                    latest["x"] = None
                    layer(c["hidden_states"].to(device), **_kw_to(c["kwargs"], device))
                    x = latest["x"]
                    if x is None:
                        raise RuntimeError("no input captured")
                    tw = weights[lid][i].to(device) if weights is not None else None
                    if tw is not None and tw.numel() != x.shape[1]:
                        raise ValueError(f"token weights {tw.numel()} != rows {x.shape[1]}")
                    helper.add_batch(x, token_weights=tw)
            finally:
                h.remove()
            for name in present:
                module = mods[name]
                w_q = helper.run_gptq(
                    module,
                    percdamp=args.percdamp,
                    wbits=args.wbits,
                    groupsize=args.groupsize,
                    actorder=False,
                    return_W=True,
                )
                module.weight.data.copy_(w_q.to(module.weight.dtype))
                n_modules += 1
            helper.free()
        # propagate the quantized-so-far stream
        for c in cache:
            c["hidden_states"] = layer(c["hidden_states"].to(device), **_kw_to(c["kwargs"], device)).detach().cpu()
        log(f"layer {lid + 1}/{len(layers)} quantized")
    return n_modules


def strip_think(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


@torch.no_grad()
def evaluate(model, tok, pairs, args, device, log):
    tok.padding_side = "left"
    hyps = []
    prompts = [deployment_prompt(tok, src) for src, _ in pairs]
    for start in range(0, len(prompts), args.gen_batch_size):
        batch = prompts[start : start + args.gen_batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=[151645, 151643],
        )
        new = out[:, enc["input_ids"].shape[1] :]
        for row in new:
            hyps.append(strip_think(tok.decode(row, skip_special_tokens=True)))
        if (start // args.gen_batch_size) % 10 == 0:
            log(f"generated {min(start + args.gen_batch_size, len(prompts))}/{len(prompts)}")
    refs = [tgt for _, tgt in pairs]
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return hyps, refs, chrf, bleu


def main():
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "log.txt"

    def log(msg):
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    device = torch.device("cuda:0")
    data_dir = Path(args.data_dir)
    dev, devtest = load_flores(data_dir)
    config = vars(args).copy()
    config.update(
        {
            "model_id": MODEL_ID,
            "support": "prompt + teacher-forced target rows (full)",
            "evaluation_map": "deployment chat template (enable_thinking=False), greedy",
            "kl_clip": [KL_CLIP_MIN, KL_CLIP_MAX],
            "groups": GROUPS,
            "user_instruction": USER_INSTRUCTION,
            "minimal_prompt": "English: {src}\\nGerman: {tgt}<|im_end|>",
            "data_files_sha256": {
                str(p.relative_to(data_dir)): sha256_file(p)
                for p in [
                    data_dir / "dev" / "eng_Latn.dev",
                    data_dir / "dev" / "deu_Latn.dev",
                    data_dir / "devtest" / "eng_Latn.devtest",
                    data_dir / "devtest" / "deu_Latn.devtest",
                ]
            },
            "dev_rows": len(dev),
            "devtest_rows": len(devtest),
        }
    )
    write_json(out_dir / "config.json", config)
    write_json(
        out_dir / "environment.json",
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sacrebleu": sacrebleu.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "script_sha256": sha256_file(Path(__file__)),
            "gptq_py_sha256": sha256_file(Path(SRC) / "gptq.py"),
            "quant_py_sha256": sha256_file(Path(SRC) / "quant.py"),
            "frame_weighting_py_sha256": sha256_file(Path(SRC) / "frame_weighting.py"),
        },
    )
    write_json(out_dir / "status.json", {"state": "running"})
    started = time.perf_counter()
    try:
        set_seed(args.quantization_seed)
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
        log(f"loaded {MODEL_ID} attn={model.config._attn_implementation}")

        quant_seconds = 0.0
        if args.density != "bf16-reference":
            rng = random.Random(args.seed)
            idx = sorted(rng.sample(range(len(dev)), args.nsamples))
            samples = [build_sample(tok, dev[i][0], dev[i][1], args.map, device) for i in idx]
            write_json(
                out_dir / "calibration.json",
                {
                    "dev_indices": idx,
                    "num_examples": len(idx),
                    "map": args.map,
                    "prompt_tokens": [s["prompt_tokens"] for s in samples],
                    "total_tokens": [s["total_tokens"] for s in samples],
                    "first_source": dev[idx[0]][0],
                },
            )
            t0 = time.perf_counter()
            weights = None
            qmeta = {"density": args.density, "map": args.map}
            if args.density == "tim":
                weights, summaries, mean_loss = collect_tim_weights(model, samples, device)
                qmeta.update(
                    {
                        "hessian": "2 * X.T * diag(w) * X",
                        "token_sensitivity": "mean squared gradient of the teacher-forced translation loss w.r.t. each layer's mlp.down_proj output",
                        "normalization": f"ratio-preserving KL/I-projection to unit mean, final weights in [{KL_CLIP_MIN:g}, {KL_CLIP_MAX:g}]",
                        "mean_teacher_forced_loss": mean_loss,
                        "layer_weight_summaries": {f"layer{i}": s for i, s in enumerate(summaries)},
                    }
                )
                log("task-Fisher weights collected")
            n_mod = quantize_sequential(model, samples, weights, args, device, log)
            quant_seconds = time.perf_counter() - t0
            qmeta.update({"quantized_modules": n_mod, "wbits": args.wbits, "groupsize": args.groupsize, "percdamp": args.percdamp})
            write_json(out_dir / "quantization.json", qmeta)
            log(f"quantized {n_mod} modules in {quant_seconds:.1f}s")
            del samples
            torch.cuda.empty_cache()

        pairs = devtest if args.eval_samples < 0 else devtest[: args.eval_samples]
        hyps, refs, chrf, bleu = evaluate(model, tok, pairs, args, device, log)
        with (out_dir / "devtest.jsonl").open("w", encoding="utf-8") as fh:
            for i, ((src, ref), hyp) in enumerate(zip(pairs, hyps)):
                fh.write(json.dumps({"index": i, "source": src, "reference": ref, "hypothesis": hyp}, ensure_ascii=False) + "\n")
        metrics = {
            "chrf++": chrf.score,
            "bleu": bleu.score,
            "chrf_signature": str(chrf.format(signature=True)) if hasattr(chrf, "format") else "",
            "bleu_signature": str(bleu.format(signature=True)) if hasattr(bleu, "format") else "",
            "num_examples": len(pairs),
            "empty_hypotheses": sum(1 for h in hyps if not h.strip()),
            "quantization_seconds": quant_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "status.json", {"state": "completed"})
        log(f"chrF++ {chrf.score:.3f} BLEU {bleu.score:.3f}")
    except Exception as exc:
        write_json(out_dir / "status.json", {"state": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
