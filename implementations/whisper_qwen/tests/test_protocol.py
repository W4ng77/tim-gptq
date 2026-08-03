from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from asr_eval import DATASET_SPECS, EVALUATION_DATASET_ID
from asr_eval import (
    _normalize_whisper_inputs,
    score_calibration_nll,
    score_qwen_calibration_wer,
)
from asr_experiment import (
    _encoder_state_sidecar_path,
    _export_whisper_encoder_state,
    _import_whisper_encoder_state,
    _record_new_baseline_metadata,
    _select_boundaryguard_candidate,
    _state_dict_manifest,
    _state_dict_sha256,
    _source_tree_sha256,
    _validate_calibration_augment_args,
    _validate_encoder_state_args,
    _validate_frame_weighting_args,
    _validate_rotation_args,
    build_parser,
    parse_datasets,
    resolve_mode,
    resolve_run_seeds,
    run,
)
from asr_models import MODEL_SPECS, resolve_model_spec
from asr_quantization import (
    _cross_attention_channel_importance,
    _decoder_sequential,
    _encoder_sequential,
    _hutchinson_token_sensitivity,
)
from gptq import Helper
from interface_bridge import fit_affine_encoder_bridge
from quantization_pipeline import _awq_auto_clip_max, _group_wbits
from quantization_runtime import (
    _normalize_qwen_sequence_token_weights,
    _qwen_mask_transcript_labels,
    _qwen_select_sequence_rows,
    _qwen_sequence_support_masks,
    _qwen_scope_includes_group,
    _qwen_scope_includes_stack,
    iter_rtn_named_modules,
)
from quantization_utils import (
    _is_ffn_output_name,
    _record_rotation_module,
    _should_apply_qep,
    _should_protect_ffn_output,
    pseudo_quantize_tensor,
)
from rotation import (
    build_rotation,
    hadamard_matrix,
    largest_power_of_two_divisor,
)
from sequence_objective import (
    paired_example_bootstrap_ci,
    paired_nll_recovery,
    paired_transcript_bootstrap_ci,
    paired_transcript_drift_recovery,
    select_best_sequence_candidate,
)
from calibration_augment import augment_waveform
from dataset_sources import LocalParquetDataset
from frame_weighting import (
    DEFAULT_ATTENTION_FLOOR,
    DEFAULT_PROPAGATED_PROBES,
    PropagatedFrameWeights,
    permute_calibration_token_weights,
    permute_calibration_token_weights_stratified,
    PROPAGATED_CLIP_MIN,
    PropagatedFrameWeights,
    aggregate_cross_attention_mass,
    attention_frame_weights,
    compute_calibration_frame_weights,
    compute_sample_frame_weights,
    downsample_feature_mask,
    energy_frame_weights,
    finalize_frame_weight_statistics,
    frame_weights_apply_to_group,
    hutchinson_frame_sensitivities,
    mask_frame_weights,
    normalize_propagated_weights,
    normalize_task_fisher_weights,
    record_frame_weight_statistics,
    resolve_group_frame_weights,
)
from hessian_energy_probe import (
    finalize_stats,
    merge_stats,
    padding_energy_stats,
)


def _tiny_random_whisper(attn_implementation: str):
    """Random-init tiny Whisper (8 mel bins, 16 mel frames -> 8 positions)."""
    from transformers import WhisperConfig, WhisperForConditionalGeneration

    config = WhisperConfig(
        vocab_size=64,
        num_mel_bins=8,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=8,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        suppress_tokens=None,
        begin_suppress_tokens=None,
    )
    model = WhisperForConditionalGeneration._from_config(
        config, attn_implementation=attn_implementation
    )
    return model.eval()


def _tiny_random_moonshine(attn_implementation: str):
    """Random-init tiny Moonshine (variable-length raw-audio encoder)."""
    from transformers import MoonshineConfig, MoonshineForConditionalGeneration

    config = MoonshineConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        encoder_num_hidden_layers=2,
        decoder_num_hidden_layers=2,
        encoder_num_attention_heads=2,
        decoder_num_attention_heads=2,
        encoder_num_key_value_heads=2,
        decoder_num_key_value_heads=2,
    )
    model = MoonshineForConditionalGeneration._from_config(
        config, attn_implementation=attn_implementation
    )
    return model.eval()


class ProtocolTests(unittest.TestCase):
    def test_target_models_are_registered(self):
        self.assertEqual(
            set(MODEL_SPECS),
            {
                "qwen-0.6b",
                "qwen-1.7b",
                "moonshine-tiny",
                "moonshine-base",
                "whisper-tiny",
                "whisper-base",
                "whisper-small",
                "whisper-medium",
                "whisper-large-v3",
                "distil-whisper-large-v3",
            },
        )

    def test_evaluation_protocol(self):
        self.assertEqual(EVALUATION_DATASET_ID, "hf-audio/open-asr-leaderboard")
        self.assertEqual(
            DATASET_SPECS,
            {
                "librispeech-clean": ("librispeech", "test.clean"),
                "librispeech-other": ("librispeech", "test.other"),
                "spgispeech": ("spgispeech", "test"),
                "voxpopuli": ("voxpopuli", "test"),
                "gigaspeech": ("gigaspeech", "test"),
            },
        )
        self.assertNotIn("tedlium", DATASET_SPECS)

    def test_all_dataset_alias(self):
        self.assertEqual(parse_datasets("all"), list(DATASET_SPECS))

    def test_model_alias_resolution(self):
        spec = resolve_model_spec("qwen-0.6b")
        self.assertEqual(spec.model_id, "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(spec.group_size, 128)

    def test_large_whisper_alias_resolution(self):
        for alias, model_id in (
            ("whisper-medium", "openai/whisper-medium"),
            ("whisper-large-v3", "openai/whisper-large-v3"),
            (
                "distil-whisper-large-v3",
                "distil-whisper/distil-large-v3",
            ),
        ):
            spec = resolve_model_spec(alias)
            self.assertEqual(spec.model_id, model_id)
            self.assertEqual(spec.family, "whisper")
            self.assertEqual(spec.group_size, 128)
            self.assertIs(resolve_model_spec(model_id), spec)

    def test_mode_resolution(self):
        args = build_parser().parse_args(["--mode", "gptq+dynamic-alpha"])
        resolve_mode(args)
        self.assertEqual(args.method, "gptq")
        self.assertTrue(args.qep)
        self.assertTrue(args.dynamic_alpha)

        boundary = build_parser().parse_args(["--mode", "gptq+boundaryguard"])
        resolve_mode(boundary)
        self.assertEqual(boundary.method, "gptq")
        self.assertFalse(boundary.qep)
        self.assertTrue(boundary.boundaryguard)
        self.assertEqual(boundary.boundaryguard_min_calibration_wer_improvement, 0.02)
        self.assertEqual(boundary.boundaryguard_min_relative_nll_degradation, 0.10)

        ffncomp = build_parser().parse_args(["--mode", "gptq+ffncomp"])
        resolve_mode(ffncomp)
        self.assertEqual(ffncomp.method, "gptq")
        self.assertTrue(ffncomp.qep)
        self.assertFalse(ffncomp.dynamic_alpha)
        self.assertEqual(ffncomp.qep_target, "ffn-output")

    def test_qep_target_selection(self):
        legacy = build_parser().parse_args(["--mode", "gptq+qep"])
        resolve_mode(legacy)
        self.assertTrue(_should_apply_qep(legacy, "self_attn.q_proj"))
        self.assertTrue(_should_apply_qep(legacy, "mlp.down_proj"))
        self.assertFalse(_should_apply_qep(legacy, "fc2"))

        targeted = build_parser().parse_args(["--mode", "gptq+ffncomp"])
        resolve_mode(targeted)
        self.assertTrue(_is_ffn_output_name("fc2"))
        self.assertTrue(_is_ffn_output_name("mlp.down_proj"))
        self.assertTrue(_should_apply_qep(targeted, "fc2"))
        self.assertTrue(_should_apply_qep(targeted, "mlp.down_proj"))
        self.assertFalse(_should_apply_qep(targeted, "fc1"))
        self.assertFalse(_should_apply_qep(targeted, "self_attn.q_proj"))

        gated = build_parser().parse_args(["--mode", "gptq+ffngate"])
        resolve_mode(gated)
        self.assertTrue(gated.qep)
        self.assertTrue(gated.qep_gate)
        self.assertEqual(gated.qep_target, "ffn-output")
        self.assertEqual(gated.qep_gate_candidates, (0.0, 0.1, 0.25, 0.5))

        protected = build_parser().parse_args(["--mode", "gptq+ffnprotect"])
        resolve_mode(protected)
        self.assertEqual(protected.method, "gptq")
        self.assertFalse(protected.qep)
        self.assertTrue(protected.ffn_output_fp16)
        self.assertTrue(_should_protect_ffn_output(protected, "fc2"))
        self.assertTrue(
            _should_protect_ffn_output(protected, "mlp.down_proj")
        )
        self.assertFalse(
            _should_protect_ffn_output(protected, "mlp.up_proj")
        )

        tail = build_parser().parse_args(["--mode", "gptq+tailffnprotect"])
        resolve_mode(tail)
        self.assertTrue(tail.tail_ffn_output_fp16)
        self.assertFalse(
            _should_protect_ffn_output(
                tail,
                "fc2",
                layer_idx=3,
                total_layers=6,
            )
        )
        self.assertTrue(
            _should_protect_ffn_output(
                tail,
                "fc2",
                layer_idx=4,
                total_layers=6,
            )
        )
        self.assertTrue(
            _should_protect_ffn_output(
                tail,
                "mlp.down_proj",
                layer_idx=21,
                total_layers=28,
            )
        )

        sequence = build_parser().parse_args(["--mode", "gptq+seqprotect"])
        resolve_mode(sequence)
        self.assertEqual(sequence.method, "gptq")
        self.assertTrue(sequence.sequence_protect)
        self.assertFalse(sequence.tail_ffn_output_fp16)

        sequence_hessian = build_parser().parse_args(
            ["--mode", "gptq+seqhess"]
        )
        resolve_mode(sequence_hessian)
        self.assertEqual(sequence_hessian.method, "gptq")
        self.assertTrue(sequence_hessian.sequence_hessian)
        self.assertFalse(sequence_hessian.ffn_output_fp16)

        propagation_hessian = build_parser().parse_args(
            ["--mode", "gptq+prophess"]
        )
        resolve_mode(propagation_hessian)
        self.assertTrue(propagation_hessian.propagation_hessian)
        self.assertFalse(propagation_hessian.sequence_hessian)

        sequence_calibration = build_parser().parse_args(
            ["--mode", "gptq+seqcal"]
        )
        resolve_mode(sequence_calibration)
        self.assertTrue(sequence_calibration.sequence_calibration)
        self.assertFalse(sequence_calibration.sequence_hessian)

    def test_quantization_seed_is_resolved_independently(self):
        args = build_parser().parse_args(["--seed", "17"])
        resolve_run_seeds(args)
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.quantization_seed, 17)
        args = build_parser().parse_args(
            ["--seed", "17", "--quantization-seed", "23"]
        )
        resolve_run_seeds(args)
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.quantization_seed, 23)

    def test_sequence_weighted_hessian(self):
        layer = torch.nn.Linear(2, 1, bias=False)
        helper = Helper(layer)
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        weights = torch.tensor([0.5, 1.5])
        helper.add_batch(inputs, token_weights=weights)
        matrix = inputs.reshape(-1, 2)
        expected = 2.0 * matrix.t() @ torch.diag(weights) @ matrix
        self.assertTrue(torch.allclose(helper.H_q, expected))

    def test_qwen_sequence_task_weights_use_frozen_clip(self):
        uniform = _normalize_qwen_sequence_token_weights(
            torch.ones(5), clip_max=2.0
        )
        self.assertTrue(torch.equal(uniform, torch.ones(5)))

        values = torch.tensor([1.0, 1.0, 100.0])
        actual = _normalize_qwen_sequence_token_weights(values, clip_max=2.0)
        self.assertAlmostEqual(float(actual.mean()), 1.0, places=6)
        self.assertLessEqual(float(actual.max()), 2.0)
        self.assertGreaterEqual(float(actual.min()), 1e-4)
        self.assertTrue(torch.allclose(actual, torch.tensor([0.5, 0.5, 2.0])))

        sparse = _normalize_qwen_sequence_token_weights(
            torch.tensor([0.0] * 99 + [1.0]), clip_max=2.0
        )
        self.assertAlmostEqual(float(sparse.mean()), 1.0, places=6)
        self.assertLessEqual(float(sparse.max()), 2.0)
        self.assertGreaterEqual(float(sparse.min()), 1e-4)

        with self.assertRaisesRegex(ValueError, "clip_max"):
            _normalize_qwen_sequence_token_weights(values, clip_max=0.0)

    def test_qwen_prompt_support_selects_only_masked_prefix_rows(self):
        labels = torch.tensor([[-100, -100, 7, 8]])
        masks = _qwen_sequence_support_masks(
            [{"labels": labels}], support="prompt"
        )
        inputs = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
        weights = torch.tensor([[0.5, 1.5, 0.8, 1.2]])
        selected_inputs, selected_weights = _qwen_select_sequence_rows(
            inputs, weights, masks[0]
        )
        self.assertEqual(tuple(selected_inputs.shape), (1, 2, 3))
        self.assertTrue(torch.equal(selected_inputs, inputs[:, :2]))
        self.assertTrue(torch.equal(selected_weights, weights[:, :2]))

    def test_qwen_prompt_support_requires_full_label_boundary(self):
        with self.assertRaisesRegex(ValueError, "both prompt and transcript"):
            _qwen_sequence_support_masks(
                [{"labels": torch.full((1, 4), -100)}], support="prompt"
            )

    def test_qwen_full_matched_support_matches_prompt_row_count(self):
        samples = [
            {"labels": torch.tensor([[-100, -100, 7, 8, 9]])},
            {"labels": torch.tensor([[-100, 4, 5, 6]])},
        ]
        first = _qwen_sequence_support_masks(
            samples, support="full-matched", seed=23
        )
        repeat = _qwen_sequence_support_masks(
            samples, support="full-matched", seed=23
        )
        changed = _qwen_sequence_support_masks(
            samples, support="full-matched", seed=24
        )
        for sample, mask, mask_repeat in zip(samples, first, repeat):
            self.assertEqual(
                int(mask.sum()), int(sample["labels"].eq(-100).sum())
            )
            self.assertTrue(torch.equal(mask, mask_repeat))
        self.assertTrue(
            any(not torch.equal(left, right) for left, right in zip(first, changed))
        )

    def test_task_weight_permutation_preserves_each_layer_marginal(self):
        original = PropagatedFrameWeights(
            [[torch.tensor([0.2, 0.8, 2.0])]],
            [torch.tensor([0.5, 1.0, 1.5])],
        )
        first = permute_calibration_token_weights(original, seed=17)
        second = permute_calibration_token_weights(original, seed=17)
        for before, after, repeat in (
            (
                original.encoder_layer_weights[0][0],
                first.encoder_layer_weights[0][0],
                second.encoder_layer_weights[0][0],
            ),
            (
                original.cross_attention_weights[0],
                first.cross_attention_weights[0],
                second.cross_attention_weights[0],
            ),
        ):
            self.assertTrue(torch.equal(after, repeat))
            self.assertTrue(
                torch.equal(torch.sort(before).values, torch.sort(after).values)
            )

    def test_stratified_permutation_preserves_each_stratum(self):
        weights = {
            "layer.0": [torch.tensor([[0.1, 0.3, 0.8, 1.2, 1.6, 2.0]])],
            "layer.1": [torch.tensor([[0.2, 0.4, 0.6, 1.4, 1.7, 1.9]])],
        }
        masks = [torch.tensor([[True, True, True, False, False, False]])]
        first = permute_calibration_token_weights_stratified(
            weights, masks, seed=17
        )
        repeat = permute_calibration_token_weights_stratified(
            weights, masks, seed=17
        )
        for key in sorted(weights):
            before = weights[key][0].reshape(-1)
            after = first[key][0].reshape(-1)
            self.assertTrue(torch.equal(after, repeat[key][0].reshape(-1)))
            for stratum in (masks[0].reshape(-1), ~masks[0].reshape(-1)):
                self.assertTrue(
                    torch.equal(
                        torch.sort(before[stratum]).values,
                        torch.sort(after[stratum]).values,
                    )
                )

    def test_task_fisher_kl_projection_preserves_unsaturated_ratios(self):
        values = torch.tensor([1.0, 2.0, 3.0, 100.0])
        actual = normalize_task_fisher_weights(
            values,
            clip_min=1e-4,
            clip_max=2.0,
        )
        self.assertAlmostEqual(float(actual.mean()), 1.0, places=6)
        self.assertGreaterEqual(float(actual.min()), 1e-4)
        self.assertLessEqual(float(actual.max()), 2.0)
        ratios = actual[:3] / actual[0]
        self.assertTrue(
            torch.allclose(ratios, torch.tensor([1.0, 2.0, 3.0]))
        )

    def test_hutchinson_sensitivity_for_scaled_identity(self):
        torch.manual_seed(3)
        source = torch.randn(1, 4, 3, requires_grad=True)
        target = 2.0 * source
        sensitivity = _hutchinson_token_sensitivity(
            source,
            target,
            probes=3,
        )
        expected = torch.full_like(sensitivity, 4.0 / 3.0)
        self.assertTrue(torch.allclose(sensitivity, expected))

    def test_qwen_sequence_labels_mask_prompt(self):
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1
            eos_token = "<eos>"

        class Processor:
            audio_token = "<audio>"
            tokenizer = Tokenizer()

            def __call__(self, text, audio, sampling_rate, return_tensors):
                del audio, sampling_rate, return_tensors
                length = len(text)
                return {
                    "input_ids": torch.arange(length).unsqueeze(0),
                    "attention_mask": torch.ones(1, length),
                }

        class Wrapper:
            processor = Processor()

            @staticmethod
            def _build_text_prompt(context, language):
                del context, language
                return "prompt:"

        # The dataset reader is integration-tested elsewhere; this unit checks
        # the exact masking operation through a minimal encoded pair.
        processor = Wrapper.processor
        prompt = Wrapper._build_text_prompt("", "English")
        encoded_prompt = processor(
            text=prompt,
            audio=[],
            sampling_rate=16000,
            return_tensors="pt",
        )
        encoded_full = processor(
            text=prompt + "words" + processor.tokenizer.eos_token,
            audio=[],
            sampling_rate=16000,
            return_tensors="pt",
        )
        labels = _qwen_mask_transcript_labels(
            encoded_prompt["input_ids"],
            encoded_full["input_ids"],
        )
        self.assertTrue((labels[:, : len(prompt)] == -100).all())
        self.assertTrue((labels[:, len(prompt) :] >= 0).all())

    def test_sequence_nll_recovery_and_confirmation(self):
        plain = {
            "per_example": [
                {
                    "example_id": "a",
                    "num_tokens": 2,
                    "mean_nll": 3.0,
                },
                {
                    "example_id": "b",
                    "num_tokens": 4,
                    "mean_nll": 2.0,
                },
            ]
        }
        restored = {
            "per_example": [
                {
                    "example_id": "a",
                    "num_tokens": 2,
                    "mean_nll": 2.0,
                },
                {
                    "example_id": "b",
                    "num_tokens": 4,
                    "mean_nll": 1.5,
                },
            ]
        }
        recovery = paired_nll_recovery(plain, restored)
        self.assertAlmostEqual(recovery["mean_token_nll_recovery"], 4.0 / 6.0)
        interval = paired_example_bootstrap_ci(
            recovery["per_example"],
            replicates=1_000,
            seed=7,
        )
        self.assertGreater(interval["lower"], 0.0)
        selected = select_best_sequence_candidate(
            {
                "dec.layer0.fc2": {"mean_token_nll_recovery": -0.1},
                "dec.layer1.fc2": {"mean_token_nll_recovery": 0.2},
            }
        )
        self.assertEqual(selected["candidate"], "dec.layer1.fc2")
        self.assertTrue(selected["eligible_for_confirmation"])

    def test_fp16_transcript_drift_recovery(self):
        fp16 = {
            "example_ids": ["a", "b"],
            "predictions": ["one two three", "four five"],
        }
        plain = {
            "example_ids": ["a", "b"],
            "predictions": ["one too", "four six"],
        }
        restored = {
            "example_ids": ["a", "b"],
            "predictions": ["one two three", "four five"],
        }
        recovery = paired_transcript_drift_recovery(fp16, plain, restored)
        self.assertEqual(recovery["num_reference_words"], 5)
        self.assertAlmostEqual(recovery["sequence_drift_recovery"], 3.0 / 5.0)
        interval = paired_transcript_bootstrap_ci(
            recovery["per_example"],
            replicates=1_000,
            seed=7,
        )
        self.assertGreater(interval["lower"], 0.0)
        selected = select_best_sequence_candidate(
            {
                "dec.layer0.fc2": {"sequence_drift_recovery": 0.1},
                "dec.layer1.fc2": {"sequence_drift_recovery": 0.3},
            },
            metric="sequence_drift_recovery",
        )
        self.assertEqual(selected["candidate"], "dec.layer1.fc2")
        self.assertEqual(selected["selection_metric"], "sequence_drift_recovery")

    def test_qep_output_surrogate_matches_direct_error_differences(self):
        torch.manual_seed(7)
        layer = torch.nn.Linear(3, 2, bias=False)
        helper = Helper(layer)
        x_q = torch.randn(16, 3)
        x_fp = x_q + 0.2 * torch.randn(16, 3)
        helper.add_batch_qep(x_q, x_fp)

        reference = layer.weight.detach().clone()
        candidates = [
            reference + 0.05 * torch.randn_like(reference),
            reference + 0.20 * torch.randn_like(reference),
        ]
        direct = [
            ((x_q @ candidate.t()) - (x_fp @ reference.t())).pow(2).sum().item()
            for candidate in candidates
        ]
        surrogate = [
            helper.qep_output_reconstruction_surrogate(
                layer,
                candidate,
                reference,
            )
            for candidate in candidates
        ]
        expected_difference = (2.0 / x_q.shape[0]) * (direct[1] - direct[0])
        self.assertAlmostEqual(
            surrogate[1] - surrogate[0],
            expected_difference,
            places=5,
        )

    def test_boundaryguard_dual_failure_detector(self):
        common = {
            "min_wer_improvement": 0.02,
            "min_relative_nll_degradation": 0.10,
        }
        by_wer = _select_boundaryguard_candidate(
            fp_nll=9.7,
            uniform_nll=8.3,
            protected_nll=9.3,
            uniform_wer=3.13,
            protected_wer=0.06,
            **common,
        )
        self.assertEqual(by_wer["selected"], "boundary-protected-w3")
        self.assertTrue(by_wer["wer_trigger"])
        self.assertFalse(by_wer["nll_trigger"])

        by_nll = _select_boundaryguard_candidate(
            fp_nll=3.5,
            uniform_nll=5.2,
            protected_nll=4.7,
            uniform_wer=0.116,
            protected_wer=0.121,
            **common,
        )
        self.assertEqual(by_nll["selected"], "boundary-protected-w3")
        self.assertFalse(by_nll["wer_trigger"])
        self.assertTrue(by_nll["nll_trigger"])

        stable = _select_boundaryguard_candidate(
            fp_nll=1.89,
            uniform_nll=2.01,
            protected_nll=1.95,
            uniform_wer=0.048,
            protected_wer=0.048,
            **common,
        )
        self.assertEqual(stable["selected"], "uniform-w3")
        self.assertFalse(stable["wer_trigger"])
        self.assertFalse(stable["nll_trigger"])

    def test_source_tree_hash_is_stable_sha256(self):
        first = _source_tree_sha256()
        second = _source_tree_sha256()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_whisper_encoder_state_round_trip_and_strict_validation(self):
        torch.manual_seed(31)
        source = _tiny_random_whisper("eager")
        with torch.no_grad():
            next(source.model.encoder.parameters()).add_(0.125)

        args = build_parser().parse_args(
            [
                "--model",
                "whisper-tiny",
                "--mode",
                "fp16",
                "--save-encoder-state",
                "unused.pt",
            ]
        )
        args.groupsize = MODEL_SPECS["whisper-tiny"].group_size
        resolve_mode(args)
        resolve_run_seeds(args)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "encoder.pt"
            exported = _export_whisper_encoder_state(
                source,
                artifact,
                MODEL_SPECS["whisper-tiny"],
                args,
            )
            self.assertTrue(artifact.is_file())
            self.assertTrue(_encoder_state_sidecar_path(artifact).is_file())
            self.assertEqual(len(exported["artifact_sha256"]), 64)
            self.assertEqual(len(exported["state_dict_sha256"]), 64)
            self.assertEqual(
                exported["source"]["model_alias"],
                "whisper-tiny",
            )

            torch.manual_seed(32)
            target = _tiny_random_whisper("eager")
            # Decoder depth is intentionally outside the encoder signature:
            # Large and Distil share an encoder but have different consumers.
            target.config.decoder_layers = 1
            imported = _import_whisper_encoder_state(
                target,
                artifact,
                MODEL_SPECS["whisper-tiny"],
            )
            self.assertEqual(
                imported["effective_model_state"],
                "fp16-consumer-with-imported-encoder",
            )
            for name, source_tensor in source.model.encoder.state_dict().items():
                self.assertTrue(
                    torch.equal(
                        source_tensor,
                        target.model.encoder.state_dict()[name],
                    ),
                    msg=name,
                )

            incompatible = _tiny_random_whisper("eager")
            incompatible.config.activation_function = "relu"
            with self.assertRaisesRegex(ValueError, "encoder config mismatch"):
                _import_whisper_encoder_state(
                    incompatible,
                    artifact,
                    MODEL_SPECS["whisper-tiny"],
                )

            with self.assertRaises(FileExistsError):
                _export_whisper_encoder_state(
                    source,
                    artifact,
                    MODEL_SPECS["whisper-tiny"],
                    args,
                )

    def test_whisper_encoder_state_rejects_missing_tensor(self):
        source = _tiny_random_whisper("eager")
        args = build_parser().parse_args(["--model", "whisper-tiny"])
        args.groupsize = MODEL_SPECS["whisper-tiny"].group_size
        resolve_mode(args)
        resolve_run_seeds(args)

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "encoder.pt"
            _export_whisper_encoder_state(
                source,
                artifact,
                MODEL_SPECS["whisper-tiny"],
                args,
            )
            _encoder_state_sidecar_path(artifact).unlink()
            payload = torch.load(
                artifact,
                map_location="cpu",
                weights_only=True,
            )
            removed_name = sorted(payload["state_dict"])[0]
            del payload["state_dict"][removed_name]
            payload["state_dict_manifest"] = _state_dict_manifest(
                payload["state_dict"]
            )
            payload["state_dict_sha256"] = _state_dict_sha256(
                payload["state_dict"]
            )
            torch.save(payload, artifact)

            with self.assertRaisesRegex(ValueError, "state_dict keys"):
                _import_whisper_encoder_state(
                    _tiny_random_whisper("eager"),
                    artifact,
                    MODEL_SPECS["whisper-tiny"],
                )

    def test_encoder_state_cli_prevents_double_quantization(self):
        bad = build_parser().parse_args(
            [
                "--model",
                "whisper-tiny",
                "--mode",
                "gptq",
                "--load-encoder-state",
                "encoder.pt",
            ]
        )
        resolve_mode(bad)
        resolve_run_seeds(bad)
        with self.assertRaisesRegex(ValueError, "requires --mode fp16"):
            _validate_encoder_state_args(
                bad,
                MODEL_SPECS["whisper-tiny"],
            )

        both = build_parser().parse_args(
            [
                "--save-encoder-state",
                "one.pt",
                "--load-encoder-state",
                "two.pt",
            ]
        )
        resolve_mode(both)
        resolve_run_seeds(both)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _validate_encoder_state_args(
                both,
                MODEL_SPECS["whisper-tiny"],
            )

    def test_encoder_state_run_artifacts_label_imported_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "shared-encoder.pt"
            export_args = build_parser().parse_args(
                [
                    "--model",
                    "whisper-tiny",
                    "--mode",
                    "fp16",
                    "--output-dir",
                    str(root / "runs"),
                    "--run-name",
                    "export",
                    "--save-encoder-state",
                    str(artifact),
                ]
            )
            with (
                mock.patch(
                    "asr_experiment.load_model",
                    return_value=(_tiny_random_whisper("eager"), None),
                ),
                mock.patch(
                    "asr_experiment._move_model_to_cuda_if_possible",
                    side_effect=lambda model: model,
                ),
            ):
                export_dir = run(export_args)
            self.assertEqual(
                json.loads((export_dir / "status.json").read_text())["state"],
                "completed",
            )

            import_args = build_parser().parse_args(
                [
                    "--model",
                    "whisper-tiny",
                    "--mode",
                    "fp16",
                    "--output-dir",
                    str(root / "runs"),
                    "--run-name",
                    "import",
                    "--load-encoder-state",
                    str(artifact),
                ]
            )
            with (
                mock.patch(
                    "asr_experiment.load_model",
                    return_value=(_tiny_random_whisper("eager"), None),
                ),
                mock.patch(
                    "asr_experiment._move_model_to_cuda_if_possible",
                    side_effect=lambda model: model,
                ),
            ):
                import_dir = run(import_args)

            status = json.loads((import_dir / "status.json").read_text())
            config = json.loads((import_dir / "config.json").read_text())
            metrics = json.loads((import_dir / "metrics.json").read_text())
            encoder_record = json.loads(
                (import_dir / "encoder_state.json").read_text()
            )
            self.assertEqual(status["state"], "completed")
            self.assertEqual(
                config["effective_model_state"],
                "fp16-consumer-with-imported-encoder",
            )
            self.assertEqual(
                metrics["effective_model_state"],
                "fp16-consumer-with-imported-encoder",
            )
            self.assertEqual(metrics["encoder_state"]["operation"], "import")
            self.assertEqual(encoder_record["operation"], "import")

    def test_component_scope_groups(self):
        self.assertEqual(len(_encoder_sequential("encoder")), 4)
        self.assertEqual(_encoder_sequential("decoder"), [])
        self.assertEqual(len(_decoder_sequential("decoder-self-attn")), 2)
        cross_attention = _decoder_sequential("decoder-cross-attn")
        self.assertEqual(
            cross_attention,
            [
                ["encoder_attn.k_proj", "encoder_attn.v_proj"],
                ["encoder_attn.q_proj"],
                ["encoder_attn.out_proj"],
            ],
        )
        self.assertEqual(len(_decoder_sequential("decoder-ffn")), 2)
        groups = _decoder_sequential("full-except-cross-attn")
        self.assertEqual(len(groups), 4)
        self.assertFalse(any(group[0].startswith("encoder_attn.") for group in groups))

    def test_qwen_text_backbone_scope_keeps_audio_tower_fp(self):
        self.assertFalse(_qwen_scope_includes_stack("text-backbone", "audio"))
        self.assertTrue(_qwen_scope_includes_stack("text-backbone", "text"))
        self.assertTrue(_qwen_scope_includes_stack("encoder", "audio"))
        self.assertFalse(_qwen_scope_includes_stack("encoder", "text"))
        self.assertTrue(_qwen_scope_includes_stack("full", "audio"))
        self.assertTrue(_qwen_scope_includes_stack("full", "text"))
        attention = ["self_attn.q_proj"]
        ffn = ["mlp.gate_proj", "mlp.up_proj"]
        self.assertTrue(
            _qwen_scope_includes_group("text-attention", "text", attention)
        )
        self.assertFalse(
            _qwen_scope_includes_group("text-attention", "text", ffn)
        )
        self.assertTrue(_qwen_scope_includes_group("text-ffn", "text", ffn))
        self.assertFalse(
            _qwen_scope_includes_group("text-ffn", "text", attention)
        )

    def test_interface_aware_group_bit_overrides(self):
        args = build_parser().parse_args(
            [
                "--wbits",
                "3",
                "--cross-attn-wbits",
                "4",
                "--encoder-tail-wbits",
                "4",
                "--encoder-tail-layers",
                "2",
            ]
        )
        resolve_mode(args)
        self.assertEqual(
            _group_wbits(args, "dec", 0, 4, ["encoder_attn.q_proj"]),
            4,
        )
        self.assertEqual(
            _group_wbits(args, "dec", 0, 4, ["self_attn.q_proj"]),
            3,
        )
        self.assertEqual(_group_wbits(args, "enc", 1, 4, ["fc1"]), 3)
        self.assertEqual(_group_wbits(args, "enc", 2, 4, ["fc1"]), 4)

        selected = build_parser().parse_args(
            [
                "--wbits",
                "3",
                "--encoder-tail-wbits",
                "4",
                "--encoder-promote-layers",
                "1,3",
            ]
        )
        resolve_mode(selected)
        self.assertEqual(_group_wbits(selected, "enc", 1, 4, ["fc1"]), 4)
        self.assertEqual(_group_wbits(selected, "enc", 2, 4, ["fc1"]), 3)

    def test_parser_accepts_two_bit_stress_tests(self):
        args = build_parser().parse_args(["--wbits", "2"])
        self.assertEqual(args.wbits, 2)

        torch.manual_seed(7)
        layer = torch.nn.Linear(16, 8, bias=False)
        helper = Helper(layer)
        helper.add_batch(torch.randn(32, 16))
        quantized = helper.run_gptq(
            layer,
            wbits=2,
            groupsize=8,
            return_W=True,
        )
        self.assertEqual(quantized.shape, layer.weight.shape)
        self.assertTrue(torch.isfinite(quantized).all())

    def test_parser_accepts_score_only_acoustic_stress(self):
        args = build_parser().parse_args(
            [
                "--score-calibration-wer",
                "--calibration-score-samples",
                "128",
                "--calibration-score-augment",
                "acoustic",
                "--calibration-score-augment-ratio",
                "1.0",
            ]
        )
        self.assertEqual(args.calibration_score_augment, "acoustic")
        self.assertEqual(args.calibration_score_augment_ratio, 1.0)

    def test_qep_percdamp_changes_correction(self):
        initial_weight = torch.tensor([[1.0, -0.5]])
        corrected = []
        for percdamp in (0.01, 1.0):
            layer = torch.nn.Linear(2, 1, bias=False)
            layer.weight.data.copy_(initial_weight)
            helper = Helper(layer)
            helper.H_q.copy_(torch.tensor([[2.0, 0.5], [0.5, 1.0]]))
            helper.H_delta.copy_(torch.tensor([[0.2, 0.0], [0.0, -0.1]]))
            helper.run_weight_correct(layer, percdamp=percdamp, perccorr=0.5)
            corrected.append(layer.weight.detach().clone())
        self.assertFalse(torch.allclose(corrected[0], corrected[1]))

    def test_whisper_attention_mask_is_preserved(self):
        class Config:
            max_source_positions = 3

        class Conv:
            stride = (1,)

        class Encoder:
            conv1 = Conv()
            conv2 = Conv()

        class Inner:
            encoder = Encoder()

        class Model:
            config = Config()
            model = Inner()

        inputs = {
            "input_features": torch.ones(1, 80, 2),
            "attention_mask": torch.ones(1, 2, dtype=torch.int32),
        }
        normalized = _normalize_whisper_inputs(inputs, Model())
        self.assertEqual(normalized["input_features"].shape[-1], 3)
        self.assertEqual(normalized["attention_mask"].shape[-1], 3)
        self.assertEqual(normalized["attention_mask"].tolist(), [[1, 1, 0]])

    def test_group_quantization_accepts_remainder(self):
        weight = torch.arange(20, dtype=torch.float32).reshape(2, 10)
        quantized, scales, zeros = pseudo_quantize_tensor(
            weight,
            n_bit=3,
            q_group_size=4,
            get_scale_zp=True,
        )
        self.assertEqual(quantized.shape, weight.shape)
        self.assertEqual(scales.shape, (2, 3))
        self.assertEqual(zeros.shape, (2, 3))

    def test_awq_clip_search_flattens_group_views_for_quantization(self):
        weight = torch.randn(8, 10)
        input_feat = torch.randn(4, 10)
        clip_max = _awq_auto_clip_max(
            weight,
            input_feat,
            n_bit=3,
            q_group_size=4,
            zero_point=True,
            n_grid=2,
            max_shrink=0.5,
        )
        self.assertEqual(clip_max.shape, (8, 1, 1))
        self.assertTrue(torch.isfinite(clip_max).all())

    def test_local_parquet_fixed_index_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            pq.write_table(
                pa.table({"id": [f"id-{i}" for i in range(7)], "value": range(7)}),
                path,
                row_group_size=3,
            )
            dataset = LocalParquetDataset((path,)).select_indices([1, 3, 6])
            self.assertEqual(dataset.num_rows, 7)
            self.assertEqual(
                list(dataset.iter_columns(["id"])),
                [{"id": "id-1"}, {"id": "id-3"}, {"id": "id-6"}],
            )
            self.assertEqual(
                [row["value"] for row in dataset],
                [1, 3, 6],
            )

    def test_rtn_component_scope_filters_decoder_groups(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = torch.nn.Linear(2, 2)
                self.v_proj = torch.nn.Linear(2, 2)
                self.q_proj = torch.nn.Linear(2, 2)
                self.out_proj = torch.nn.Linear(2, 2)

        class Layer(torch.nn.Module):
            def __init__(self, decoder=False):
                super().__init__()
                self.self_attn = Attention()
                if decoder:
                    self.encoder_attn = Attention()
                self.fc1 = torch.nn.Linear(2, 2)
                self.fc2 = torch.nn.Linear(2, 2)

        class Stack(torch.nn.Module):
            def __init__(self, decoder=False):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(decoder=decoder)])

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Stack()
                self.decoder = Stack(decoder=True)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Core()

        names = [
            name
            for name, _ in iter_rtn_named_modules(
                Model(),
                quant_scope="decoder-cross-attn",
            )
        ]
        self.assertEqual(len(names), 4)
        self.assertTrue(all(".encoder_attn." in name for name in names))

    def test_affine_interface_bridge_recovers_channelwise_mapping(self):
        class Encoder(torch.nn.Module):
            def __init__(self, scale, bias):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.scale = float(scale)
                self.bias = float(bias)

            def forward(self, values):
                return {"last_hidden_state": values * self.scale + self.bias}

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder(1.0, 0.0)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Core()

        model = Model()
        reference = Encoder(2.0, 1.0)
        calibration = [
            {"input_values": torch.tensor([[[0.0, 1.0], [2.0, 3.0]]])},
            {"input_values": torch.tensor([[[4.0, 5.0], [6.0, 7.0]]])},
        ]
        handle, metadata = fit_affine_encoder_bridge(
            model,
            reference,
            calibration,
            torch.device("cpu"),
        )
        try:
            output = model.model.encoder(torch.tensor([[[1.5, 2.5]]]))
            self.assertTrue(
                torch.allclose(
                    output["last_hidden_state"],
                    torch.tensor([[[4.0, 6.0]]]),
                    atol=1e-5,
                )
            )
            self.assertLess(metadata["mse_after"], 1e-10)
            self.assertGreater(metadata["mse_before"], metadata["mse_after"])
        finally:
            handle.remove()

    def test_cross_attention_channel_importance_is_normalized(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = torch.nn.Linear(2, 2, bias=False)
                self.v_proj = torch.nn.Linear(2, 2, bias=False)
                self.k_proj.weight.data.copy_(torch.tensor([[2.0, 0.0], [2.0, 0.0]]))
                self.v_proj.weight.data.copy_(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_attn = Attention()

        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer()])

        class Core(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = Decoder()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Core()

        importance = _cross_attention_channel_importance(Model())
        self.assertTrue(torch.allclose(importance.mean(), torch.tensor(1.0)))
        self.assertGreater(importance[0], importance[1])

    def test_calibration_nll_is_token_weighted(self):
        class Output:
            def __init__(self, loss):
                self.loss = loss

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.calls = 0

            def forward(self, input_values, labels):
                del input_values, labels
                self.calls += 1
                return Output(torch.tensor(float(self.calls)))

        calibration = [
            {
                "input_values": torch.zeros(1, 2),
                "decoder_input_ids": torch.zeros(1, 2, dtype=torch.long),
                "dataset_id": "a",
            },
            {
                "input_values": torch.zeros(1, 2),
                "decoder_input_ids": torch.zeros(1, 4, dtype=torch.long),
                "dataset_id": "b",
            },
        ]
        result = score_calibration_nll(Model(), calibration, torch.device("cpu"))
        self.assertEqual(result["num_tokens"], 6)
        self.assertAlmostEqual(result["mean_token_nll"], 10 / 6)

    def test_qwen_calibration_wer_uses_retained_references(self):
        class Result:
            def __init__(self, text):
                self.text = text

        class Model:
            def transcribe(self, audio, language, return_time_stamps):
                self.audio = audio
                self.language = language
                self.return_time_stamps = return_time_stamps
                return [Result("HELLO WORLD"), Result("test")]

        calibration = [
            {
                "__audio__": torch.zeros(16).numpy(),
                "__text__": "Hello, world!",
                "__dataset_id__": "a",
            },
            {
                "__audio__": torch.ones(8).numpy(),
                "__text__": "test",
                "__dataset_id__": "b",
            },
        ]
        model = Model()
        score = score_qwen_calibration_wer(
            model,
            calibration,
            torch.device("cpu"),
        )
        self.assertEqual(score["num_examples"], 2)
        self.assertEqual(score["wer"], 0.0)
        self.assertEqual(score["example_ids"], ["a", "b"])
        self.assertEqual(len(model.audio), 2)

    def test_gptaq_mode_resolution_and_defaults(self):
        args = build_parser().parse_args(["--mode", "gptq+gptaq"])
        resolve_mode(args)
        resolve_run_seeds(args)
        self.assertEqual(args.method, "gptq")
        self.assertTrue(args.gptaq)
        self.assertFalse(args.qep)
        self.assertFalse(args.qep_gate)
        self.assertFalse(args.dynamic_alpha)
        self.assertEqual(args.gptaq_alpha, 0.25)
        self.assertEqual(args.rotate, "none")

        plain = build_parser().parse_args(["--mode", "gptq"])
        resolve_mode(plain)
        self.assertFalse(plain.gptaq)

    def test_rotation_seed_defaults_to_quantization_seed(self):
        args = build_parser().parse_args(["--seed", "17"])
        resolve_run_seeds(args)
        self.assertEqual(args.rotation_seed, 17)
        args = build_parser().parse_args(
            ["--seed", "17", "--quantization-seed", "23"]
        )
        resolve_run_seeds(args)
        self.assertEqual(args.rotation_seed, 23)
        args = build_parser().parse_args(
            ["--seed", "17", "--rotation-seed", "5"]
        )
        resolve_run_seeds(args)
        self.assertEqual(args.rotation_seed, 5)

    def test_rotation_cli_validation(self):
        good = build_parser().parse_args(["--mode", "gptq", "--rotate", "hadamard"])
        resolve_mode(good)
        _validate_rotation_args(good)

        for mode in ("rtn", "awq", "fp16"):
            bad = build_parser().parse_args(["--mode", mode, "--rotate", "hadamard"])
            resolve_mode(bad)
            with self.assertRaises(ValueError):
                _validate_rotation_args(bad)

        gated = build_parser().parse_args(
            ["--mode", "gptq+ffngate", "--rotate", "hadamard"]
        )
        resolve_mode(gated)
        with self.assertRaises(ValueError):
            _validate_rotation_args(gated)

    def test_hadamard_rotation_orthogonality_and_block_support(self):
        expected_blocks = {384: 128, 512: 512, 288: 32, 416: 32, 52: 4, 13: 1}
        for dim, block in expected_blocks.items():
            self.assertEqual(largest_power_of_two_divisor(dim), block)
            rotation = build_rotation(dim, seed=3, tag="enc.layer0.fc1")
            identity = torch.eye(dim, dtype=rotation.dtype)
            self.assertTrue(
                torch.allclose(rotation.t() @ rotation, identity, atol=1e-5)
            )
            if block < dim:
                # Broad-support block-diagonal: no mass outside the first block.
                self.assertTrue(torch.all(rotation[:block, block:] == 0))
                self.assertTrue(torch.all(rotation[block:, :block] == 0))

        with self.assertRaises(ValueError):
            hadamard_matrix(12)
        with self.assertRaises(ValueError):
            largest_power_of_two_divisor(0)

    def test_hadamard_rotation_is_seed_and_tag_deterministic(self):
        base = build_rotation(64, seed=3, tag="enc.layer0.fc1")
        again = build_rotation(64, seed=3, tag="enc.layer0.fc1")
        other_tag = build_rotation(64, seed=3, tag="enc.layer1.fc1")
        other_seed = build_rotation(64, seed=4, tag="enc.layer0.fc1")
        self.assertTrue(torch.equal(base, again))
        self.assertFalse(torch.equal(base, other_tag))
        self.assertFalse(torch.equal(base, other_seed))

    def test_rotated_gptq_foldback_is_equivalent_at_high_precision(self):
        # in_features=288 rotates with block 32 while the moonshine-tiny
        # groupsize 52 deliberately crosses Hadamard block boundaries and
        # leaves a remainder group (288 = 5*52 + 28).
        torch.manual_seed(1)
        layer = torch.nn.Linear(288, 24, bias=False)
        original = layer.weight.detach().clone()
        helper = Helper(layer)
        helper.add_batch(torch.randn(128, 288))
        folded = helper.run_gptq(
            layer,
            wbits=16,
            groupsize=52,
            return_W=True,
            rotate="hadamard",
            rotation_seed=7,
            rotation_tag="enc.layer0.fc1",
        )
        self.assertEqual(folded.shape, original.shape)
        relative_error = (
            (folded.float() - original.float()).norm() / original.float().norm()
        )
        self.assertLess(relative_error.item(), 1e-3)

    def test_rotated_gptq_changes_low_bit_solution(self):
        torch.manual_seed(1)
        layer = torch.nn.Linear(288, 24, bias=False)
        helper = Helper(layer)
        helper.add_batch(torch.randn(128, 288))
        rotated = helper.run_gptq(
            layer,
            wbits=3,
            groupsize=52,
            return_W=True,
            rotate="hadamard",
            rotation_seed=7,
            rotation_tag="enc.layer0.fc1",
        )
        plain = helper.run_gptq(layer, wbits=3, groupsize=52, return_W=True)
        self.assertFalse(torch.allclose(rotated, plain))

    def test_rotation_rejects_non_linear_layers_and_unknown_schemes(self):
        conv = torch.nn.Conv2d(2, 4, kernel_size=3)
        helper = Helper(conv)
        # Both guards fire before any Hessian statistics are consumed.
        with self.assertRaises(TypeError):
            helper.run_gptq(conv, wbits=4, return_W=True, rotate="hadamard")
        with self.assertRaises(ValueError):
            helper.run_gptq(conv, wbits=4, return_W=True, rotate="givens")

    def test_gptaq_improves_asymmetric_objective_on_small_matrices(self):
        gptq_objectives = []
        gptaq_objectives = []
        for seed in range(8):
            torch.manual_seed(seed)
            layer = torch.nn.Linear(16, 8, bias=False)
            reference = layer.weight.detach().clone()
            x_q = torch.randn(64, 16)
            x_fp = x_q + 0.25 * torch.randn(64, 16) + 0.1 * x_q.roll(1, dims=1)

            helper = Helper(layer)
            helper.add_batch_qep(x_q, x_fp)
            w_gptq = helper.run_gptq(layer, wbits=3, groupsize=-1, return_W=True)
            w_gptaq = helper.run_gptq(
                layer,
                wbits=3,
                groupsize=-1,
                return_W=True,
                gptaq_alpha=0.25,
            )
            self.assertFalse(torch.allclose(w_gptq, w_gptaq))

            def asymmetric_objective(candidate):
                return (
                    (x_q @ candidate.float().t() - x_fp @ reference.t())
                    .pow(2)
                    .sum()
                    .item()
                )

            gptq_objectives.append(asymmetric_objective(w_gptq))
            gptaq_objectives.append(asymmetric_objective(w_gptaq))

        # The asymmetric-calibration correction should lower the objective it
        # targets on average (and on the majority of deterministic draws).
        self.assertLess(sum(gptaq_objectives), sum(gptq_objectives))
        wins = sum(
            gptaq < gptq
            for gptaq, gptq in zip(gptaq_objectives, gptq_objectives)
        )
        self.assertGreaterEqual(wins, 5)

    def test_gptaq_with_zero_delta_matches_plain_gptq(self):
        torch.manual_seed(0)
        layer = torch.nn.Linear(16, 8, bias=False)
        helper = Helper(layer)
        x = torch.randn(64, 16)
        helper.add_batch_qep(x, x)
        plain = helper.run_gptq(layer, wbits=3, groupsize=-1, return_W=True)
        corrected = helper.run_gptq(
            layer,
            wbits=3,
            groupsize=-1,
            return_W=True,
            gptaq_alpha=0.25,
        )
        self.assertTrue(torch.allclose(plain, corrected))

    def test_gptaq_composes_with_rotation(self):
        # The strictly-triangular GPTAQ correction is basis-dependent (like
        # actorder), so composing with rotation gives a distinct, valid
        # solution rather than an exact fold-back identity; check that the
        # composition runs and actually differs from each single lever.
        torch.manual_seed(2)
        layer = torch.nn.Linear(64, 8, bias=False)
        helper = Helper(layer)
        x_q = torch.randn(96, 64)
        helper.add_batch_qep(x_q, x_q + 0.2 * torch.randn(96, 64))
        composed = helper.run_gptq(
            layer,
            wbits=3,
            groupsize=32,
            return_W=True,
            gptaq_alpha=0.25,
            rotate="hadamard",
            rotation_seed=11,
            rotation_tag="dec.layer0.fc1",
        )
        rotation_only = helper.run_gptq(
            layer,
            wbits=3,
            groupsize=32,
            return_W=True,
            rotate="hadamard",
            rotation_seed=11,
            rotation_tag="dec.layer0.fc1",
        )
        gptaq_only = helper.run_gptq(
            layer,
            wbits=3,
            groupsize=32,
            return_W=True,
            gptaq_alpha=0.25,
        )
        self.assertEqual(composed.shape, layer.weight.shape)
        self.assertTrue(torch.isfinite(composed).all())
        self.assertFalse(torch.allclose(composed, rotation_only))
        self.assertFalse(torch.allclose(composed, gptaq_only))

    def test_new_baseline_metadata_records(self):
        args = build_parser().parse_args(
            ["--mode", "gptq+gptaq", "--rotate", "hadamard", "--rotation-seed", "9"]
        )
        resolve_mode(args)
        resolve_run_seeds(args)
        _record_new_baseline_metadata(args)
        metadata = args._quantization_metadata
        self.assertEqual(metadata["gptaq"]["alpha"], 0.25)
        self.assertIn("2504.02692", metadata["gptaq"]["reference"])
        self.assertEqual(metadata["rotation_config"]["rotation_seed"], 9)

        _record_rotation_module(args, "enc.layer0.fc1", 288)
        rotation = args._quantization_metadata["rotation"]
        self.assertEqual(rotation["seed"], 9)
        self.assertEqual(
            rotation["modules"]["enc.layer0.fc1"]["hadamard_block_size"],
            32,
        )

    def test_audio_capture_cli_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.frame_weighting, "none")
        self.assertEqual(args.calib_augment, "none")
        self.assertEqual(args.calib_augment_ratio, 0.5)
        resolve_mode(args)
        _validate_frame_weighting_args(args)
        _validate_calibration_augment_args(args)

    def test_feature_mask_downsampling_matches_conv_arithmetic(self):
        # Whisper front-end: kernel 3, padding 1, strides (1, 2);
        # output length is floor((L - 1) / s) + 1 == len(range(0, L, s)).
        for length, strides in ((3000, (1, 2)), (2999, (1, 2)), (7, (2,)), (10, (2, 2))):
            mask = torch.ones(1, length)
            downsampled = downsample_feature_mask(mask, strides)
            expected_length = length
            for stride in strides:
                conv = torch.nn.Conv1d(1, 1, kernel_size=3, stride=stride, padding=1)
                with torch.no_grad():
                    expected_length = conv(
                        torch.zeros(1, 1, expected_length)
                    ).shape[-1]
            self.assertEqual(downsampled.shape[-1], expected_length)

        # 3000 mel frames -> 1500 encoder positions; speech prefix stays aligned.
        mask = torch.zeros(1, 3000)
        mask[:, :1234] = 1
        downsampled = downsample_feature_mask(mask, (1, 2))
        self.assertEqual(downsampled.shape[-1], 1500)
        self.assertEqual(int(downsampled.sum()), 617)  # ceil(1234 / 2)
        self.assertTrue(bool(downsampled[0, 616] == 1))
        self.assertTrue(bool(downsampled[0, 617] == 0))

    def test_mask_weighted_hessian_matches_manual_masked_gram(self):
        torch.manual_seed(11)
        inputs = torch.randn(1, 6, 3)
        weights = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        weighted_helper = Helper(torch.nn.Linear(3, 2, bias=False))
        weighted_helper.add_batch(inputs, token_weights=weights)

        manual_helper = Helper(torch.nn.Linear(3, 2, bias=False))
        manual_helper.add_batch(inputs[:, :3, :])

        self.assertTrue(
            torch.allclose(weighted_helper.H_q, manual_helper.H_q, atol=1e-6)
        )

    def test_energy_frame_weights_are_unit_mean_and_suppress_padding(self):
        # Speech frames at normalized log-mel 0.5, padding at -0.5:
        # linear power 1e-2 vs 1e-6, so padding weights collapse toward zero.
        features = torch.full((1, 2, 8), -0.5)
        features[:, :, :4] = 0.5
        weights = energy_frame_weights(features, strides=(1, 2))
        self.assertEqual(weights.shape, (4,))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=5)
        self.assertGreater(float(weights[:2].min()), 1.9)
        self.assertLess(float(weights[2:].max()), 1e-3)

        # Clipping bounds a dominant frame before the final renormalization.
        spiky = torch.full((1, 2, 8), -1.0)
        spiky[:, :, 0] = 1.0
        clipped = energy_frame_weights(spiky, strides=(1, 2), clip_max=2.0)
        self.assertAlmostEqual(float(clipped.mean()), 1.0, places=5)
        self.assertLess(float(clipped.max()), 4.01)

    def test_frame_weight_computation_per_family(self):
        class Conv1:
            stride = (1,)

        class Conv2:
            stride = (2,)

        class Encoder:
            conv1 = Conv1()
            conv2 = Conv2()

        class Inner:
            encoder = Encoder()

        class FakeModel:
            model = Inner()

        whisper_sample = {
            "input_features": torch.full((1, 2, 8), -0.5),
            "feature_attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]),
        }
        moonshine_sample = {"input_values": torch.randn(1, 100)}
        weights, info = compute_calibration_frame_weights(
            FakeModel(),
            [whisper_sample, moonshine_sample],
            "mask",
        )
        self.assertEqual(weights[0].tolist(), [1.0, 1.0, 1.0, 0.0])
        self.assertIsNone(weights[1])
        self.assertEqual(info["noop_samples"], 1)
        self.assertEqual(info["conv_strides"], [1, 2])

        # mask mode without the stored mask must fail loudly, not silently.
        with self.assertRaises(ValueError):
            compute_sample_frame_weights(
                {"input_features": whisper_sample["input_features"]},
                "mask",
                (1, 2),
            )
        # An all-padding mask must not zero the whole sample.
        with self.assertRaises(ValueError):
            mask_frame_weights(torch.zeros(1, 8), (1, 2))
        # mode none is always a no-op.
        self.assertIsNone(
            compute_sample_frame_weights(whisper_sample, "none", (1, 2))
        )

    def test_frame_weight_group_predicate(self):
        self.assertTrue(frame_weights_apply_to_group("enc", ["fc1"]))
        self.assertTrue(
            frame_weights_apply_to_group("enc", ["self_attn.k_proj"])
        )
        self.assertTrue(
            frame_weights_apply_to_group(
                "dec",
                ["encoder_attn.k_proj", "encoder_attn.v_proj", "encoder_attn.q_proj"],
            )
        )
        self.assertTrue(
            frame_weights_apply_to_group("dec", ["cross_attn.k_proj"])
        )
        self.assertFalse(
            frame_weights_apply_to_group("dec", ["encoder_attn.out_proj"])
        )
        self.assertFalse(
            frame_weights_apply_to_group("dec", ["self_attn.k_proj"])
        )
        self.assertFalse(frame_weights_apply_to_group("dec", ["fc1"]))
        self.assertFalse(frame_weights_apply_to_group("dec", []))

    def test_frame_weighting_rejects_dual_stream_and_non_gptq_modes(self):
        for mode in ("gptq+gptaq", "gptq+qep", "gptq+dynamic-alpha", "gptq+ffngate"):
            args = build_parser().parse_args(
                ["--mode", mode, "--frame-weighting", "mask"]
            )
            resolve_mode(args)
            with self.assertRaises(ValueError):
                _validate_frame_weighting_args(args)

        for mode in ("rtn", "awq", "fp16"):
            args = build_parser().parse_args(
                ["--mode", mode, "--frame-weighting", "energy"]
            )
            resolve_mode(args)
            with self.assertRaises(ValueError):
                _validate_frame_weighting_args(args)

        for mode in ("gptq", "gptq+seqhess"):
            args = build_parser().parse_args(
                ["--mode", mode, "--frame-weighting", "mask"]
            )
            resolve_mode(args)
            _validate_frame_weighting_args(args)

    def test_attention_frame_weights_unit_mean_and_floor(self):
        # No sub-floor mass: exact unit mean, floor inactive, scale invariant.
        mass = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weights = attention_frame_weights(mass, floor=0.05)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreater(float(weights.min()), 0.05)
        rescaled = attention_frame_weights(mass * 7.5, floor=0.05)
        self.assertTrue(torch.allclose(weights, rescaled, atol=1e-6))

        # The soft identity mixture has exact unit mean and floor lower bound.
        spiky = torch.tensor([100.0, 1e-6, 1e-6, 1e-6])
        floored = attention_frame_weights(spiky, floor=0.05)
        self.assertAlmostEqual(float(floored.min()), 0.05, places=6)
        self.assertAlmostEqual(float(floored.mean()), 1.0, places=6)

        # floor=0 keeps the exact unit mean.
        zero_floor = attention_frame_weights(spiky, floor=0.0)
        self.assertAlmostEqual(float(zero_floor.mean()), 1.0, places=6)

        with self.assertRaises(ValueError):
            attention_frame_weights(mass, floor=-0.1)
        with self.assertRaises(ValueError):
            attention_frame_weights(mass, floor=1.0)
        with self.assertRaises(ValueError):
            attention_frame_weights(torch.zeros(4))

    def test_cross_attention_aggregation_over_steps_layers_heads(self):
        torch.manual_seed(7)
        layers = [torch.rand(1, 2, 3, 4) for _ in range(2)]
        # Rows are attention distributions over four encoder positions.
        layers = [layer / layer.sum(-1, keepdim=True) for layer in layers]
        mass = aggregate_cross_attention_mass(tuple(layers))
        expected = torch.zeros(4)
        count = 0
        for layer in layers:
            for head in range(2):
                for step in range(3):
                    expected += layer[0, head, step]
                    count += 1
        expected /= count
        self.assertTrue(torch.allclose(mass, expected, atol=1e-6))
        # Distributions stay normalized after averaging.
        self.assertAlmostEqual(float(mass.sum()), 1.0, places=5)

        # Missing weights (fused kernels) must fail loudly.
        with self.assertRaises(ValueError):
            aggregate_cross_attention_mass((layers[0], None))
        with self.assertRaises(ValueError):
            aggregate_cross_attention_mass(None)

    def test_attention_mode_differs_from_mask_and_softens_padding(self):
        torch.manual_seed(3)
        model = _tiny_random_whisper("sdpa")
        mask = torch.zeros(1, 16, dtype=torch.long)
        mask[:, :10] = 1  # 5 speech / 3 padding encoder positions
        sample = {
            "input_features": torch.randn(1, 8, 16),
            "feature_attention_mask": mask,
            "decoder_input_ids": torch.tensor([[1, 5, 9, 2]]),
        }
        attn_weights, info = compute_calibration_frame_weights(
            model, [sample], "attention", floor=0.05
        )
        mask_weights, _ = compute_calibration_frame_weights(
            model, [sample], "mask"
        )
        weights = attn_weights[0]
        self.assertEqual(weights.shape, (8,))
        self.assertGreaterEqual(float(weights.min()), 0.05)
        # Soft consumption weighting: padding rows keep nonzero weight,
        # the binary mask zeroes them; the two modes must differ.
        self.assertGreater(float(weights[5:].min()), 0.0)
        self.assertEqual(float(mask_weights[0][5:].max()), 0.0)
        self.assertFalse(torch.allclose(weights, mask_weights[0]))

        # SDPA cannot emit weights: the pass falls back to eager and restores.
        self.assertTrue(info["attention_pass"]["eager_fallback"])
        self.assertEqual(
            info["attention_pass"]["requested_attn_implementation"], "sdpa"
        )
        self.assertEqual(model.config._attn_implementation, "sdpa")
        self.assertEqual(info["attention_pass"]["num_decoder_layers"], 2)
        self.assertEqual(info["attention_pass"]["num_attention_heads"], 2)

        summary = info["weight_summary"]
        self.assertGreaterEqual(summary["mean_weight"], 1.0 - 1e-6)
        self.assertLessEqual(summary["mean_weight"], 1.05 + 1e-6)
        padding_vs_speech = summary["padding_vs_speech"]
        self.assertEqual(padding_vs_speech["samples_with_feature_mask"], 1)
        self.assertEqual(padding_vs_speech["padding_frames"], 3)
        self.assertEqual(padding_vs_speech["speech_frames"], 5)
        self.assertGreater(padding_vs_speech["padding_mean_weight"], 0.0)
        self.assertGreater(padding_vs_speech["speech_mean_weight"], 0.0)

        # An eager model takes the direct path with identical weights.
        eager_model = _tiny_random_whisper("eager")
        eager_model.load_state_dict(model.state_dict())
        eager_weights, eager_info = compute_calibration_frame_weights(
            eager_model, [sample], "attention", floor=0.05
        )
        self.assertFalse(eager_info["attention_pass"]["eager_fallback"])
        self.assertTrue(torch.allclose(weights, eager_weights[0], atol=1e-5))

    def test_attention_mode_weights_moonshine_variable_length(self):
        torch.manual_seed(4)
        model = _tiny_random_moonshine("eager")
        sample = {
            "input_values": torch.randn(1, 16_000),
            "decoder_input_ids": torch.tensor([[1, 4, 6]]),
        }
        weights, info = compute_calibration_frame_weights(
            model, [sample], "attention"
        )
        # mask mode is a no-op for Moonshine; attention mode is not.
        self.assertIsNotNone(weights[0])
        self.assertEqual(info["noop_samples"], 0)
        self.assertGreaterEqual(
            float(weights[0].min()), DEFAULT_ATTENTION_FLOOR
        )
        mask_weights, mask_info = compute_calibration_frame_weights(
            model, [sample], "mask"
        )
        self.assertIsNone(mask_weights[0])
        self.assertEqual(mask_info["noop_samples"], 1)

    def test_attention_frame_weighting_cli_validation(self):
        args = build_parser().parse_args(
            ["--mode", "gptq", "--frame-weighting", "attention"]
        )
        self.assertEqual(args.frame_weighting_floor, DEFAULT_ATTENTION_FLOOR)
        resolve_mode(args)
        _validate_frame_weighting_args(args)

        for bad_floor in ("1.5", "-0.2"):
            bad = build_parser().parse_args(
                [
                    "--mode",
                    "gptq",
                    "--frame-weighting",
                    "attention",
                    "--frame-weighting-floor",
                    bad_floor,
                ]
            )
            resolve_mode(bad)
            with self.assertRaises(ValueError):
                _validate_frame_weighting_args(bad)

        # attention mode has no per-sample entry point without a model pass.
        with self.assertRaises(ValueError):
            compute_sample_frame_weights(
                {"input_features": torch.zeros(1, 2, 8)}, "attention", (1, 2)
            )

    def test_task_fisher_weights_are_layerwise_unit_mean(self):
        torch.manual_seed(13)
        model = _tiny_random_whisper("eager")
        mask = torch.zeros(1, 16, dtype=torch.long)
        mask[:, :10] = 1
        sample = {
            "input_features": torch.randn(1, 8, 16),
            "feature_attention_mask": mask,
            "decoder_input_ids": torch.tensor([[1, 5, 9, 2]]),
        }
        weights, info = compute_calibration_frame_weights(
            model,
            [sample],
            "task-fisher",
            propagated_clip_max=5.0,
        )
        self.assertIsInstance(weights, PropagatedFrameWeights)
        self.assertEqual(info["mode"], "task-fisher")
        self.assertEqual(info["num_encoder_layers"], 2)
        self.assertEqual(info["noop_samples"], 0)
        self.assertTrue(np.isfinite(info["mean_teacher_forced_loss"]))
        self.assertIn("final weights", info["normalization"])
        self.assertIn("with unit mean", info["normalization"])
        for layer_index in range(2):
            layer_weights = weights.encoder_layer_weights[layer_index][0]
            self.assertEqual(layer_weights.shape, (8,))
            self.assertAlmostEqual(float(layer_weights.mean()), 1.0, places=5)
            self.assertGreaterEqual(
                float(layer_weights.min()), PROPAGATED_CLIP_MIN - 1e-7
            )
            self.assertLessEqual(float(layer_weights.max()), 5.0 + 1e-6)
        final_weights = weights.cross_attention_weights[0]
        self.assertEqual(final_weights.shape, (8,))
        self.assertAlmostEqual(float(final_weights.mean()), 1.0, places=5)
        self.assertGreaterEqual(
            float(final_weights.min()), PROPAGATED_CLIP_MIN - 1e-7
        )
        self.assertLessEqual(float(final_weights.max()), 5.0 + 1e-6)

        args = build_parser().parse_args(
            [
                "--mode",
                "gptq",
                "--frame-weighting",
                "task-fisher",
                "--propagated-clip-max",
                "5",
            ]
        )
        resolve_mode(args)
        _validate_frame_weighting_args(args)

    def test_hutchinson_frame_sensitivities_match_analytic_diagonal_stack(self):
        # Channel-diagonal two-layer linear stack: with per-channel scalings
        # the Rademacher signs square away, so a single probe is *exactly*
        # the analytic sensitivity s_t = (mass_t / d) * mean_h(scale_h^2).
        torch.manual_seed(9)
        frames, hidden = 5, 3
        x = torch.randn(1, frames, hidden, requires_grad=True)
        a_diag = torch.tensor([0.5, -1.0, 2.0])
        b_diag = torch.tensor([1.5, 0.25, -2.0])
        h1 = x * a_diag
        h2 = h1 * b_diag
        mass = torch.tensor([0.2, 0.4, 1.4, 2.0, 1.0])
        s_x, s_h1 = hutchinson_frame_sensitivities([x, h1], h2, mass, probes=1)
        expected_x = mass / hidden * float((a_diag * b_diag).square().mean())
        expected_h1 = mass / hidden * float(b_diag.square().mean())
        self.assertTrue(torch.allclose(s_x.reshape(-1), expected_x, atol=1e-5))
        self.assertTrue(torch.allclose(s_h1.reshape(-1), expected_h1, atol=1e-5))

        with self.assertRaises(ValueError):
            hutchinson_frame_sensitivities([x], h2, mass, probes=0)
        with self.assertRaises(ValueError):
            hutchinson_frame_sensitivities([x], h2, mass[:-1], probes=1)
        with self.assertRaises(ValueError):
            hutchinson_frame_sensitivities([x], h2.detach(), mass, probes=1)
        with self.assertRaises(ValueError):
            hutchinson_frame_sensitivities([], h2, mass, probes=1)

    def test_normalize_propagated_weights_unit_mean_and_clip(self):
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weights = normalize_propagated_weights(values)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        rescaled = normalize_propagated_weights(values * 12.5)
        self.assertTrue(torch.allclose(weights, rescaled, atol=1e-6))

        # A dominant spike is clipped at 100x mean before renormalization,
        # and tiny values at 1e-3: dynamic range is bounded by 1e5.
        spiky = torch.tensor([1e9, 1.0, 1.0, 1e-12])
        clipped = normalize_propagated_weights(spiky)
        self.assertAlmostEqual(float(clipped.mean()), 1.0, places=5)
        ratio = float(clipped.max() / clipped.min())
        self.assertLessEqual(ratio, 100.0 / 1e-3 + 1.0)
        self.assertGreater(float(clipped.min()), 0.0)

        robust = normalize_propagated_weights(spiky, clip_max=2.0)
        self.assertAlmostEqual(float(robust.mean()), 1.0, places=5)
        self.assertLessEqual(
            float(robust.max() / robust.min()),
            2.0 / 1e-3 + 1.0,
        )
        self.assertLess(float(robust.max()), float(clipped.max()))

        with self.assertRaises(ValueError):
            normalize_propagated_weights(torch.zeros(4))
        with self.assertRaises(ValueError):
            normalize_propagated_weights(torch.tensor([1.0, float("nan")]))

    def test_task_fisher_ess_shrinkage_has_analytic_floor(self):
        values = torch.tensor([1000.0, 1.0, 1.0, 1.0, 1.0])
        weights = normalize_task_fisher_weights(
            values,
            clip_max=100.0,
            min_ess_fraction=0.5,
        )
        ess_fraction = float(
            weights.sum().square()
            / (weights.numel() * weights.square().sum())
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertGreaterEqual(ess_fraction, 0.5 - 1e-6)
        with self.assertRaises(ValueError):
            normalize_task_fisher_weights(values, min_ess_fraction=1.1)

    def test_propagated_shallow_sensitivity_is_flatter_under_mixing(self):
        # The Level-1 mechanism in a controlled form: two uniform-mixing
        # position layers between the shallow anchor and the consumed
        # output. Backpropagating a peaked consumption mass through each
        # mixing step must flatten it, so CV falls monotonically with
        # distance from the output (weights are unit mean, CV == std).
        torch.manual_seed(2)
        frames, hidden = 8, 4
        mix = 0.5 * torch.eye(frames) + 0.5 / frames
        x = torch.randn(1, frames, hidden, requires_grad=True)
        h1 = torch.einsum("ts,bsd->btd", mix, x)
        h2 = torch.einsum("ts,bsd->btd", mix, h1)
        mass = torch.full((frames,), 0.1)
        mass[-1] = frames - 0.7  # peaked, unit mean
        s_x, s_h1 = hutchinson_frame_sensitivities([x, h1], h2, mass, probes=8)

        def cv(values):
            return float(normalize_propagated_weights(values).std(unbiased=False))

        self.assertLess(cv(s_x), cv(s_h1))
        self.assertLess(cv(s_h1), cv(mass))
        # No manual floor, yet mixing keeps every position's weight nonzero.
        self.assertGreater(float(normalize_propagated_weights(s_x).min()), 0.0)

    def test_propagated_mode_differs_per_layer_from_attention_on_whisper(self):
        torch.manual_seed(3)
        model = _tiny_random_whisper("sdpa")
        mask = torch.zeros(1, 16, dtype=torch.long)
        mask[:, :10] = 1  # 5 speech / 3 padding encoder positions
        sample = {
            "input_features": torch.randn(1, 8, 16),
            "feature_attention_mask": mask,
            "decoder_input_ids": torch.tensor([[1, 5, 9, 2]]),
        }
        weights, info = compute_calibration_frame_weights(
            model, [sample], "propagated", probes=2
        )
        self.assertIsInstance(weights, PropagatedFrameWeights)
        attn_weights, _ = compute_calibration_frame_weights(
            model, [sample], "attention", floor=DEFAULT_ATTENTION_FLOOR
        )

        num_layers = len(model.model.encoder.layers)
        self.assertEqual(num_layers, 2)
        for layer_index in range(num_layers):
            layer_weights = weights.weights_for_group("enc", layer_index)[0]
            self.assertEqual(layer_weights.shape, (8,))
            self.assertAlmostEqual(float(layer_weights.mean()), 1.0, places=4)
            # Per-layer sensitivities are not the final attention mass.
            self.assertFalse(
                torch.allclose(layer_weights, attn_weights[0], atol=1e-4)
            )
            # No manual floor, yet self-attention mixing keeps rows nonzero.
            self.assertGreater(float(layer_weights.min()), 0.0)

        # The cross-attention K/V group consumes the final encoder states,
        # so it uses the (clip-processed) attention mass itself.
        cross_weights = weights.weights_for_group("dec", 0)[0]
        self.assertTrue(torch.allclose(cross_weights, attn_weights[0], atol=1e-4))

        with self.assertRaises(IndexError):
            weights.weights_for_group("enc", num_layers)
        # List-valued modes pass through the group resolver unchanged.
        self.assertIs(
            resolve_group_frame_weights(attn_weights, "enc", 1), attn_weights
        )
        self.assertIs(
            resolve_group_frame_weights(weights, "dec", 3),
            weights.cross_attention_weights,
        )

        self.assertEqual(info["mode"], "propagated")
        self.assertEqual(info["probes"], 2)
        self.assertEqual(info["num_encoder_layers"], num_layers)
        self.assertTrue(info["attention_pass"]["eager_fallback"])
        self.assertEqual(model.config._attn_implementation, "sdpa")
        flatness = info["flatness"]
        self.assertEqual(len(flatness["cv_by_layer"]), num_layers)
        self.assertEqual(
            len(flatness["padding_to_speech_ratio_by_layer"]), num_layers
        )
        for ratio in flatness["padding_to_speech_ratio_by_layer"]:
            self.assertIsNotNone(ratio)
        summaries = info["layer_weight_summaries"]
        self.assertIn("cross_attention_kv", summaries)
        for layer_index in range(num_layers):
            summary = summaries[f"enc.layer{layer_index}"]
            self.assertAlmostEqual(summary["mean_weight"], 1.0, places=4)
            self.assertEqual(
                summary["padding_vs_speech"]["padding_frames"], 3
            )
            self.assertEqual(
                summary["padding_vs_speech"]["speech_frames"], 5
            )

        # Identity terminal metric + pullback is the missing factorial arm:
        # K/V stays uniform while encoder layers reflect Jacobian geometry.
        uniform_pullback, uniform_info = compute_calibration_frame_weights(
            model,
            [sample],
            "propagated-uniform",
            probes=2,
        )
        self.assertEqual(uniform_info["mode"], "propagated-uniform")
        self.assertEqual(uniform_info["terminal_metric"], "uniform")
        self.assertTrue(
            torch.equal(
                uniform_pullback.weights_for_group("dec", 0)[0],
                torch.ones(8),
            )
        )
        for layer_index in range(num_layers):
            layer_weights = uniform_pullback.weights_for_group(
                "enc", layer_index
            )[0]
            self.assertAlmostEqual(float(layer_weights.mean()), 1.0, places=4)

    def test_propagated_mode_weights_moonshine_variable_length(self):
        torch.manual_seed(4)
        model = _tiny_random_moonshine("eager")
        sample = {
            "input_values": torch.randn(1, 16_000),
            "decoder_input_ids": torch.tensor([[1, 4, 6]]),
        }
        weights, info = compute_calibration_frame_weights(
            model, [sample], "propagated", probes=2
        )
        self.assertEqual(info["noop_samples"], 0)
        num_layers = len(model.model.encoder.layers)
        lengths = set()
        for layer_index in range(num_layers):
            layer_weights = weights.weights_for_group("enc", layer_index)[0]
            self.assertIsNotNone(layer_weights)
            lengths.add(int(layer_weights.numel()))
        cross_weights = weights.weights_for_group("dec", 0)[0]
        lengths.add(int(cross_weights.numel()))
        self.assertEqual(len(lengths), 1)  # all anchored on encoder positions
        # No feature mask exists, so the padding contrast is unavailable.
        self.assertEqual(
            info["flatness"]["padding_to_speech_ratio_by_layer"],
            [None] * num_layers,
        )

    def test_propagated_frame_weighting_cli_validation(self):
        for frame_mode in ("propagated", "propagated-uniform"):
            args = build_parser().parse_args(
                ["--mode", "gptq", "--frame-weighting", frame_mode]
            )
            self.assertEqual(args.propagated_probes, DEFAULT_PROPAGATED_PROBES)
            self.assertEqual(args.propagated_clip_max, 100.0)
            resolve_mode(args)
            _validate_frame_weighting_args(args)

        bad = build_parser().parse_args(
            [
                "--mode",
                "gptq",
                "--frame-weighting",
                "propagated",
                "--propagated-probes",
                "0",
            ]
        )
        resolve_mode(bad)
        with self.assertRaises(ValueError):
            _validate_frame_weighting_args(bad)

        bad_clip = build_parser().parse_args(
            [
                "--mode",
                "gptq",
                "--frame-weighting",
                "propagated",
                "--propagated-clip-max",
                "0.0001",
            ]
        )
        resolve_mode(bad_clip)
        with self.assertRaises(ValueError):
            _validate_frame_weighting_args(bad_clip)

        # Dual-stream statistics (QEP/GPTAQ) still refuse frame weighting.
        for mode in ("gptq+gptaq", "gptq+qep"):
            dual = build_parser().parse_args(
                ["--mode", mode, "--frame-weighting", "propagated"]
            )
            resolve_mode(dual)
            with self.assertRaises(ValueError):
                _validate_frame_weighting_args(dual)

        # Pullback modes have no per-sample entry point without model passes.
        for frame_mode in ("propagated", "propagated-uniform"):
            with self.assertRaises(ValueError):
                compute_sample_frame_weights(
                    {"input_features": torch.zeros(1, 2, 8)},
                    frame_mode,
                    (1, 2),
                )

    def test_frame_weight_statistics_recording(self):
        args = argparse.Namespace()
        record_frame_weight_statistics(
            args,
            "enc.layer0.fc1",
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        )
        record_frame_weight_statistics(
            args,
            "enc.layer0.fc1",
            torch.tensor([1.0, 0.0]),
        )
        finalized = finalize_frame_weight_statistics(args)
        stats = finalized["enc.layer0.fc1"]
        self.assertEqual(stats["rows"], 6)
        self.assertAlmostEqual(stats["mean_weight"], 0.5)
        self.assertAlmostEqual(stats["zero_weight_fraction"], 0.5)

    def test_calibration_augmentation_is_deterministic_and_varied(self):
        rng = np.random.default_rng(123)
        audio = (0.1 * rng.standard_normal(16_000)).astype(np.float32)

        record_keys = set()
        types = set()
        for index in range(12):
            first, first_record = augment_waveform(
                audio, seed=5, index=index, ratio=1.0
            )
            second, second_record = augment_waveform(
                audio, seed=5, index=index, ratio=1.0
            )
            self.assertTrue(np.array_equal(first, second))
            self.assertEqual(first_record, second_record)
            record_keys.add(json.dumps(first_record, sort_keys=True))
            types.add(first_record.get("type"))
        # Different indices draw different augmentations.
        self.assertGreater(len(record_keys), 1)
        self.assertGreaterEqual(len(types - {None}), 2)

        # ratio 0 keeps the waveform bit-identical and records the skip.
        unchanged, record = augment_waveform(audio, seed=5, index=0, ratio=0.0)
        self.assertTrue(np.array_equal(unchanged, audio))
        self.assertFalse(record["applied"])

        with self.assertRaises(ValueError):
            augment_waveform(audio, seed=5, index=0, ratio=1.5)

        bad = build_parser().parse_args(
            ["--calib-augment", "acoustic", "--calib-augment-ratio", "1.2"]
        )
        with self.assertRaises(ValueError):
            _validate_calibration_augment_args(bad)

    def test_padding_energy_probe_statistics(self):
        hidden = torch.tensor([[3.0, 4.0], [0.6, 0.8]])
        position_mask = torch.tensor([1.0, 0.0])
        stats = padding_energy_stats(hidden, position_mask)
        stats = merge_stats({}, stats)
        finalized = finalize_stats(stats)
        self.assertEqual(finalized["positions"], 2)
        self.assertEqual(finalized["padding_positions"], 1)
        self.assertAlmostEqual(finalized["padding_position_fraction"], 0.5)
        self.assertAlmostEqual(finalized["gram_trace_total"], 26.0, places=5)
        self.assertAlmostEqual(
            finalized["padding_gram_trace_fraction"], 1.0 / 26.0, places=6
        )
        self.assertAlmostEqual(finalized["mean_norm_speech"], 5.0, places=5)
        self.assertAlmostEqual(finalized["mean_norm_padding"], 1.0, places=5)
        self.assertAlmostEqual(
            finalized["speech_to_padding_norm_ratio"], 5.0, places=5
        )

        with self.assertRaises(ValueError):
            padding_energy_stats(hidden, torch.ones(3))


if __name__ == "__main__":
    unittest.main()
