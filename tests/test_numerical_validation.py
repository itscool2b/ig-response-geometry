"""Check that engineering comparisons isolate the declared arithmetic factor."""

import json

import pytest
import torch

from scripts.validate_rdt_numerics import comparison_metrics, group_values, same_gradient_stream


def test_one_gradient_stream_is_shared_by_both_accumulators():
    coefficients = torch.arange(128, 256).float() / 128
    calls = []

    def forward(z):
        calls.append(z.detach().clone())
        return (z.float() * coefficients).sum()

    actual = torch.ones(128, dtype=torch.bfloat16)
    result = same_gradient_stream(forward, actual, torch.zeros_like(actual), m=64)
    assert len(calls) == 65
    maps = result["maps"]
    torch.testing.assert_close(maps["same_stream_fp32_sum_fp32_finish"], coefficients, atol=0, rtol=0)
    assert not torch.equal(maps["same_stream_low_sum_fp32_finish"], coefficients)
    # Old end-to-end error is independently known from the preserved audit.
    max_relative_error = ((maps["historical_arithmetic_diagnostic"] - coefficients).abs() / coefficients).max()
    assert max_relative_error.item() == 0.0625
    assert result["metadata"]["forward_evaluations"] == 65


def test_historical_endpoint_reconstruction_error_is_not_hidden():
    actual = torch.tensor([1.0])
    baseline = torch.tensor([1e8])
    result = same_gradient_stream(lambda z: z.sum(), actual, baseline, m=2)
    assert result["metadata"]["max_endpoint_reconstruction_error"] == 1
    assert result["metadata"]["reconstructed_input_score"] == 0


def test_comparison_detects_canceling_errors_despite_matching_signed_sum():
    actual, corrupted = torch.tensor([1.0, -1.0, 1.0]), torch.tensor([0.0, 0.0, 1.0])
    metrics = comparison_metrics(actual, corrupted, reference_groups=actual, candidate_groups=corrupted)
    assert metrics["reference_signed_sum"] == metrics["candidate_signed_sum"] == 1
    assert metrics["absolute_l1_difference"] == 2
    assert metrics["relative_l1_difference"] == pytest.approx(2 / 3)
    assert metrics["max_absolute_coordinate_difference"] == 1
    json.dumps(metrics, allow_nan=False)


def test_constant_rank_and_zero_map_norm_are_explicitly_undefined():
    actual, other = torch.zeros(20), torch.ones(20)
    metrics = comparison_metrics(actual, other, reference_groups=actual, candidate_groups=other)
    assert metrics["relative_l1_difference"] is None
    assert metrics["absolute_group_spearman"] is None
    assert metrics["rank_status"] == "constant_group_scores"
    assert metrics["top_five_percent_count"] == 1
    json.dumps(metrics, allow_nan=False)


def test_group_rankings_use_eligible_observed_modality_only():
    vision = torch.arange(24).reshape(1, 12, 2).float()
    assert torch.equal(group_values(vision, "vision", {}), vision.sum(-1)[0, 6:8])
    language = torch.arange(8).reshape(1, 4, 2).float()
    ctx = {"lang_attn_mask": torch.tensor([[True, True, False, False]])}
    assert torch.equal(group_values(language, "language", ctx), language.sum(-1)[0, :2])
    state = torch.arange(128).reshape(1, 1, 128).float()
    assert group_values(state, "state", {}).tolist() == [0, 1, 2, 3, 4, 5, 6, 10]


def test_group_ranking_matches_evaluator_despite_signed_cancellation():
    language = torch.tensor([[[4.0, -4.0], [1.0, 1.0]]])
    ctx = {"lang_attn_mask": torch.tensor([[True, True]])}
    assert group_values(language, 'language', ctx).tolist() == [8.0, 2.0]
