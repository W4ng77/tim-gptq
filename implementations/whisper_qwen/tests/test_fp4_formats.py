"""Unit tests for fp4_formats (E2M1 codebook, MXFP4 E8M0 block-32, NVFP4 E4M3 block-16 + FP32
tensor scale) and for the FP4 path inside the paper's GPTQ solver. Run with pytest."""
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fp4_formats as F  # noqa: E402
from gptq import Helper  # noqa: E402

torch.manual_seed(0)
CODEBOOK = sorted({s * v for v in [0, 0.5, 1, 1.5, 2, 3, 4, 6] for s in (1, -1)})


def brute_force_rne(x):
    """Independent E2M1 rounding: nearest codebook value, ties to the even encoding."""
    out = []
    for v in x.tolist():
        v = max(-6.0, min(6.0, v))
        cands = sorted(CODEBOOK, key=lambda c: (abs(c - v), 0))
        best = [c for c in CODEBOOK if abs(abs(c - v) - abs(cands[0] - v)) < 1e-12]
        if len(best) == 1:
            out.append(best[0])
        else:
            # even mantissa encodings: 0, +-1, +-2, +-4 (mantissa bit 0)
            even = [c for c in best if abs(c) in (0.0, 1.0, 2.0, 4.0)]
            out.append(even[0])
    return torch.tensor(out)


def test_codebook_values():
    assert F.E2M1_VALUES.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    x = torch.tensor(CODEBOOK)
    assert torch.equal(F.round_e2m1(x), x)


def test_e2m1_rne_ties_and_random():
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    expect = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    assert torch.equal(F.round_e2m1(ties), expect)
    assert torch.equal(F.round_e2m1(-ties), -expect)
    x = torch.cat([torch.randn(200_000) * 3, torch.rand(50_000) * 14 - 7, ties, -ties])
    assert torch.equal(F.round_e2m1(x), brute_force_rne(x))


def test_saturation():
    x = torch.tensor([6.01, 100.0, -7.0, float("inf")])
    assert F.round_e2m1(x).tolist() == [6.0, 6.0, -6.0, 6.0]


def test_mxfp4_block_and_e8m0_scale():
    assert F.block_size("mxfp4") == 32
    w = torch.randn(64, 32 * 7)
    q = F.fake_quant_tensor(w, "mxfp4")
    for start in range(0, w.shape[1], 32):
        blk = w[:, start : start + 32]
        amax = blk.abs().amax(dim=1)
        scale = F.block_scales(blk, "mxfp4")
        # power of two, spec exponent formula, and elements within +-6*scale
        assert torch.all(torch.log2(scale) == torch.floor(torch.log2(scale)))
        assert torch.equal(scale, torch.exp2(torch.floor(torch.log2(amax)) - 2))
        assert torch.all((q[:, start : start + 32] / scale[:, None]).abs() <= 6)
        # dequantized values are codebook multiples of the scale
        codes = q[:, start : start + 32] / scale[:, None]
        assert torch.equal(F.round_e2m1(codes), codes)


def test_nvfp4_block_e4m3_and_tensor_scale():
    assert F.block_size("nvfp4") == 16
    w = torch.randn(48, 16 * 9) * 0.05
    ts = F.nvfp4_tensor_scale(w)
    assert math.isclose(ts, float(w.abs().max()) / (6 * 448))
    q = F.fake_quant_tensor(w, "nvfp4")
    e4m3_repr = lambda t: torch.equal(t, t.to(torch.float8_e4m3fn).float())  # noqa: E731
    for start in range(0, w.shape[1], 16):
        blk = w[:, start : start + 16]
        amax = blk.abs().amax(dim=1)
        local = F.e4m3_cast(amax / 6.0 / ts)            # the FP8 E4M3 block scale
        assert e4m3_repr(local), "block scale must be E4M3-representable"
        assert torch.all(local <= 448.0)
        scale = F.block_scales(blk, "nvfp4", ts)
        assert torch.equal(scale, torch.where(local * ts > 0, local * ts, torch.ones_like(local)))
        qb = q[:, start : start + 16]
        codes = qb / scale[:, None]
        # dequantized values are exactly (E2M1 code) x (decoded scale); compare in the value domain
        assert torch.equal(F.round_e2m1(codes) * scale[:, None], qb)
        assert torch.all(codes.abs() <= 6 + 1e-6)


def test_error_bounds_and_mxfp4_idempotence():
    for fmt in ("mxfp4", "nvfp4"):
        w = torch.randn(32, 128)
        q = F.fake_quant_tensor(w, fmt)
        bs = F.block_size(fmt)
        ts = F.nvfp4_tensor_scale(w) if fmt == "nvfp4" else None
        for start in range(0, 128, bs):
            blk = w[:, start : start + bs]
            scale = F.block_scales(blk, fmt, ts)
            codes_in = blk / scale[:, None]
            err = (blk - q[:, start : start + bs]).abs() / scale[:, None]
            inside = codes_in.abs() <= 6.0
            # nearest-value rounding inside the codebook range: error at most half the largest gap (4 -> 6)
            assert torch.all(err[inside] <= 1.0 + 1e-6)
            if fmt == "mxfp4":
                # OCP floor(log2) scale: amax/X in [4, 8); values in (6, 8) saturate to 6 (error < 2)
                assert torch.all(codes_in.abs() < 8.0)
                assert torch.all(err <= 2.0)
            else:
                # E4M3 rounding of the block scale is within 2^-3 relative, so amax/scale <= 6 * (1 + 1/8)
                assert torch.all(codes_in.abs() <= 6.0 * (1 + 1 / 8) + 1e-6)
    # MXFP4 (power-of-two scales) is a projection: Q(Q(W)) = Q(W)
    w = torch.randn(32, 128)
    q = F.fake_quant_tensor(w, "mxfp4")
    assert torch.equal(F.fake_quant_tensor(q, "mxfp4"), q)


def test_zero_block():
    w = torch.zeros(4, 32)
    assert torch.equal(F.fake_quant_tensor(w, "mxfp4"), w)
    w2 = torch.zeros(4, 16)
    assert torch.equal(F.fake_quant_block(w2, "nvfp4", 1.0), w2)


def _gptq_fp4(fmt, token_weights):
    torch.manual_seed(1)
    layer = torch.nn.Linear(256, 64, bias=False).cuda()
    x = torch.randn(1, 40, 256).cuda()
    h = Helper(layer)
    h.add_batch(x, token_weights=token_weights)
    return h.run_gptq(layer, percdamp=0.01, wbits=4, groupsize=128, actorder=False, return_W=True, fp4_format=fmt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_gptq_fp4_operator_inside_solver_and_shared_code_path(monkeypatch):
    """Instrument the FP4 operator: inside run_gptq every column is quantized with the
    format quantizer against the block scale computed at the block boundary from the
    error-compensated weights, and the operator receives no token weights."""
    import fp4_formats as FF
    calls = {"scales": [], "cols": []}
    orig_scales, orig_col = FF.block_scales, FF.fake_quant_column

    def rec_scales(block, fmt, ts=None):
        out = orig_scales(block, fmt, ts)
        calls["scales"].append((block.shape[1], fmt, out.clone()))
        return out

    def rec_col(col, scale):
        out = orig_col(col, scale)
        calls["cols"].append((col.clone(), scale.clone(), out.clone()))
        return out

    monkeypatch.setattr(FF, "block_scales", rec_scales)
    monkeypatch.setattr(FF, "fake_quant_column", rec_col)
    for fmt in ("mxfp4", "nvfp4"):
        calls["scales"].clear(); calls["cols"].clear()
        w_uni = _gptq_fp4(fmt, None)
        n_scales_uni, n_cols_uni = len(calls["scales"]), len(calls["cols"])
        bs = F.block_size(fmt)
        assert n_scales_uni == 256 // bs and n_cols_uni == 256           # one scale per block, one call per column
        assert all(shape == bs and f == fmt for shape, f, _ in calls["scales"])
        for col, scale, out in calls["cols"]:
            assert torch.equal(out, F.round_e2m1(col.float() / scale) * scale)
            if fmt == "mxfp4":
                assert torch.all(torch.log2(scale) == torch.floor(torch.log2(scale)))
        calls["scales"].clear(); calls["cols"].clear()
        w_tim = _gptq_fp4(fmt, torch.rand(40).cuda() + 0.5)
        assert (len(calls["scales"]), len(calls["cols"])) == (n_scales_uni, n_cols_uni)
        assert not torch.equal(w_uni, w_tim)                              # different Gram, same operator
    import inspect
    assert "token_weights" not in inspect.getsource(FF)


def test_reference_e2m1_rounding_matches_compressed_tensors():
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.quantization.utils import cast_to_fp4
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    x = torch.cat([torch.randn(100_000) * 3, torch.rand(50_000) * 14 - 7, ties, -ties])
    assert torch.equal(F.round_e2m1(x), cast_to_fp4(x.clone()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_gptq_fp4_uses_block_size_not_requested_groupsize():
    # groupsize=128 requested, but the format block size must govern scales
    torch.manual_seed(2)
    layer = torch.nn.Linear(64, 8, bias=False).cuda()
    x = torch.randn(1, 20, 64).cuda()
    h = Helper(layer)
    h.add_batch(x)
    w = h.run_gptq(layer, percdamp=0.01, wbits=4, groupsize=128, actorder=False, return_W=True, fp4_format="mxfp4").float().cpu()
    # each 32-column block must be codebook multiples of a single power-of-two scale per row
    for start in (0, 32):
        blk = w[:, start : start + 32]
        nz = blk.abs()[blk != 0]
        assert nz.numel() > 0
        scale = F.block_scales(blk, "mxfp4")
        codes = blk / scale[:, None]
        assert torch.equal(F.round_e2m1(codes), codes)


def test_reference_mx_scales_vs_compressed_tensors():
    """OCP text formula (ours) vs the compressed-tensors/vLLM reference: identical E8M0
    exponents whenever the block amax mantissa is < 1.75; the reference rounds the exponent
    up (no saturation) when the mantissa is >= 1.75, ours keeps floor(log2) and saturates."""
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales
    amax = torch.rand(200_000) * 10 + 1e-3
    mine = torch.log2(F.e8m0_scale(amax))
    ref = generate_mx_scales(amax.clone(), 4).float() - 127
    mant = amax / torch.exp2(torch.floor(torch.log2(amax)))
    assert torch.equal(mine[mant < 1.75], ref[mant < 1.75])
    assert torch.all((ref - mine)[mant >= 1.75] == 1.0)


def test_reference_nvfp4_vs_compressed_tensors():
    """NVFP4 block scales (E4M3, block 16, global scale 6*448/amax) and dequantized weights
    against compressed-tensors' calculate_qparams + fake_quantize (tensor_group strategy)."""
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.quantization import QuantizationArgs
    from compressed_tensors.quantization.utils.helpers import calculate_qparams
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    torch.manual_seed(3)
    for rows, cols, sc in [(64, 256, 0.05), (128, 512, 1.0), (16, 64, 3.0)]:
        w = torch.randn(rows, cols) * sc
        args = QuantizationArgs(num_bits=4, type="float", strategy="tensor_group", group_size=16,
                                symmetric=True, dynamic=False, scale_dtype=torch.float8_e4m3fn)
        G = torch.tensor(448.0 * 6.0 / float(w.abs().max()), dtype=torch.float32)  # reference uses a multiplier
        blocks = w.reshape(rows, cols // 16, 16)
        mn, mx = blocks.amin(-1), blocks.amax(-1)
        ref_scales, zp = calculate_qparams(mn, mx, args, global_scale=G)
        ref = fake_quantize(w, ref_scales, zp, args, global_scale=G).float()
        mine = F.fake_quant_tensor(w, "nvfp4")
        ts = F.nvfp4_tensor_scale(w)
        my_local = F.e4m3_cast(blocks.abs().amax(-1) / 6.0 / ts)
        assert torch.equal(my_local, ref_scales.float().reshape(my_local.shape)), "E4M3 block scales differ"
        # same codes and scales; only floating-point operation order differs (x/(s/G) vs x/(s*ts))
        assert torch.allclose(mine, ref, rtol=0.0, atol=1e-6 * float(w.abs().max()))
