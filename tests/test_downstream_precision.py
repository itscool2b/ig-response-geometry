"""Numerical isolation checks, including failures that Module.float misses."""
import json

import pytest
import torch

from integrated_gradients import integrated_gradients
from scripts.validate_downstream_precision import (
    FP32OperationAudit, cache_context, cache_identity, common_response_curves,
    convert_downstream_to_fp32, fp32_math_settings, make_cached_functions,
    normalize_curve, ranked_group_indices, repeat_endpoints, response_scores,
)


class DtypeEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.bfloat16
        self.weight = torch.nn.Parameter(torch.tensor([1.125], dtype=self.dtype), requires_grad=False)

    def forward(self, x):
        return (x.to(self.dtype) * self.weight).float()


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = torch.bfloat16
        self.t_embedder = DtypeEmbedding()
        self.freq_embedder = DtypeEmbedding()

    def forward(self, state_action, freqs, timestep, language, image, lang_mask):
        # All tensors remain in the active model dtype; embeddings are exercised
        # separately by the hidden-dtype test below.
        return state_action[:, 1:] * .25 + state_action[:, :1] * .125 + (
            (language * lang_mask.unsqueeze(-1)).mean(1, keepdim=True) * .5
            + image.mean(1, keepdim=True) * .25)


class Scheduler:
    config = {"algorithm_type": "dpmsolver++"}

    @classmethod
    def from_config(cls, config):
        return cls()

    def set_timesteps(self, count):
        self.timesteps = torch.arange(count)

    def step(self, prediction, timestep, sample):
        from types import SimpleNamespace
        return SimpleNamespace(prev_sample=sample * .5 + prediction)


class Runner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyModel()
        self.state_adaptor = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.state_adaptor.weight.zero_()
            self.state_adaptor.weight[:, :128].copy_(torch.eye(128, dtype=torch.bfloat16))
        self.pred_horizon, self.action_dim, self.num_inference_timesteps = 2, 128, 2
        self.noise_scheduler_sample = Scheduler()
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def img_adaptor(self, *args):
        raise AssertionError("Must not readapt cached image features")

    def lang_adaptor(self, *args):
        raise AssertionError("Must not readapt cached language features")


def cached_fixture():
    from per_step_attribution import MANISKILL_INDICES
    dtype = torch.bfloat16
    state = torch.zeros(1, 1, 128, dtype=dtype)
    state[..., MANISKILL_INDICES] = .25
    mask = torch.zeros_like(state)
    mask[..., MANISKILL_INDICES] = 1
    language = torch.arange(4 * 128).reshape(1, 4, 128).to(dtype) / 512
    image = torch.arange(12 * 128).reshape(1, 12, 128).to(dtype) / 2048
    return {
        "lang_adapted": language, "lang_adapted_bl": torch.zeros_like(language),
        "img_adapted": image, "img_adapted_bl": torch.zeros_like(image),
        "state_input_actual": state, "state_input_baseline": torch.zeros_like(state),
        "state_traj_actual": state.clone(), "action_mask": mask,
        "ctrl_freqs": torch.tensor([25], dtype=dtype),
        "lang_attn_mask": torch.tensor([[True, True, False, False]]),
        "initial_noise": torch.full((1, 2, 128), .0625),
        "ref_action": torch.zeros(1, 2, 128, dtype=dtype),
    }


def test_float_alone_misses_dtype_attribute_and_operation_audit_catches_it():
    runner = Runner().float()
    assert runner.model.t_embedder.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="Non-fp32 output"):
        with FP32OperationAudit():
            runner.model.t_embedder(torch.tensor([1.003]))


def test_controlled_conversion_preserves_numeric_weights_and_fixes_hidden_casts():
    runner = Runner()
    expected = {name: value.float().clone() for name, value in runner.state_dict().items()}
    result = convert_downstream_to_fp32(runner)
    assert result["value_preservation"] == "exact"
    assert len(result["dtype_attributes"]) == 3
    for name, value in runner.state_dict().items():
        assert value.dtype == torch.float32
        assert torch.equal(value, expected[name])
    with FP32OperationAudit() as audit:
        result = runner.model.t_embedder(torch.tensor([1.003]))
    torch.testing.assert_close(result, torch.tensor([1.003 * 1.125]))
    assert audit.operations


def test_unknown_dtype_attribute_fails_before_mutation():
    runner = Runner()
    runner.surprise_dtype = torch.bfloat16
    with pytest.raises(ValueError, match="contract changed"):
        convert_downstream_to_fp32(runner)
    assert all(parameter.dtype == torch.bfloat16 for parameter in runner.parameters())


def test_unregistered_low_precision_tensor_is_not_silently_accepted():
    runner = Runner()
    runner.model.hidden = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Unconverted plain tensor"):
        convert_downstream_to_fp32(runner)


def test_nested_dtype_contract_fails_before_precision_conversion():
    runner = Runner()
    runner.model.options = {"dtype": torch.bfloat16}
    with pytest.raises(ValueError, match="nested dtype"):
        convert_downstream_to_fp32(runner)
    assert all(parameter.dtype == torch.bfloat16 for parameter in runner.parameters())


def test_cached_functions_use_exact_values_noise_masks_and_frozen_reference():
    runner, source = Runner(), cached_fixture()
    cache = cache_context(source)
    before = cache_identity(cache)
    lowctx, lowactions, _, lowpairs = make_cached_functions(runner, cache, torch.bfloat16, "logpi")
    with torch.no_grad():
        stored = lowactions["vision"](lowpairs["vision"][0]).cpu()
    cache["ref_action"] = stored.clone()
    before = cache_identity(cache)
    convert_downstream_to_fp32(runner)
    ctx, actions, scalars, pairs = make_cached_functions(runner, cache, torch.float32, "logpi")
    assert torch.equal(ctx["ref_action"], stored.float())
    for key in ("lang_adapted", "img_adapted", "state_input_actual", "state_traj_actual", "initial_noise", "action_mask"):
        assert torch.equal(ctx[key], cache[key].float())
    with FP32OperationAudit(), torch.no_grad():
        pred = actions["vision"](pairs["vision"][0])
    expected = response_scores(pred, stored)["quadratic"]
    assert torch.equal(scalars["vision"](pairs["vision"][0]), expected)
    assert cache_identity(cache) == before
    # Source/cache are independent, so a caller's later mutation cannot rewrite
    # the comparison context behind the identity recorded above.
    source["img_adapted"].zero_()
    assert cache_identity(cache) == before


def test_raw_state_path_is_distinct_from_cached_frozen_state_token():
    runner, cache = Runner(), cached_fixture()
    convert_downstream_to_fp32(runner)
    ctx, actions, _, pairs = make_cached_functions(runner, cache, torch.float32, "logpi")
    with torch.no_grad():
        runner.state_adaptor.weight[:, :128].mul_(2)
    # Action-token adaptor also changes both paths; their *difference* isolates
    # state preprocessing because one holds the old cached state token fixed.
    vision = actions["vision"](pairs["vision"][0])
    state = actions["state"](pairs["state"][0])
    assert not torch.equal(vision, state)
    assert torch.equal(ctx["state_traj_actual"], cache["state_traj_actual"].float())


def test_repeat_endpoint_probe_records_nondeterminism_and_zero_gap_range():
    calls = 0
    def changing(value):
        nonlocal calls
        calls += 1
        return torch.full((1, 1, 128), float(calls))
    result, samples = repeat_endpoints(changing, torch.ones(1), torch.zeros(1), torch.zeros(1, 1, 128), 3)
    assert not result["actual"]["bitwise_repeatable"]
    assert result["actual"]["max_action_repeat_difference"] == 2
    assert len(set(result["baseline"]["actions_sha256"])) == 3
    json.dumps(result, allow_nan=False)


def test_zero_and_unbounded_response_normalization_do_not_hide_failure():
    zero = normalize_curve([3, 4], 3, 3, [0, 1])
    assert zero["auc"] is None and zero["status"] == "zero_gap"
    curve = normalize_curve([-2, 0, 3], 0, 1, [0, .5, 1])
    assert curve["normalized"] == [-2, 0, 3]
    assert curve["auc"] == .25
    with pytest.raises(ValueError, match="finite"):
        normalize_curve([0, float("nan")], 0, 1, [0, 1])


def test_common_response_is_independent_of_rank_origin_and_saves_raw_predictions():
    actual, baseline = torch.ones(1, 20, 1), torch.zeros(1, 20, 1)
    coefficients = torch.arange(1, 21).float().reshape(1, 20, 1)
    calls = []
    def action(value):
        assert value.dtype == torch.float32
        calls.append(value.clone())
        output = torch.zeros(1, 1, 128)
        output[..., 0] = (value * coefficients).sum()
        return output
    reference = action(actual)
    ascending, descending = torch.arange(20), torch.arange(19, -1, -1)
    report, outputs = common_response_curves(action, actual, baseline, reference,
        {"bf16": ascending, "fp32": descending}, "language", [0, .5, 1])
    for name, deleted_value in (("bf16", 1), ("fp32", 20)):
        row = report["rankings"][name]
        assert row["replacement_counts"] == [0, 1, 10, 20]
        # One changed entry among 8 active action dimensions gives Q=-d^2/16.
        assert row["curves"]["deletion"]["raw_scores"]["quadratic"][1] == -(deleted_value**2) / 16
        assert outputs[name + "_deletion"].shape == (4, 1, 1, 128)
        assert torch.equal(outputs[name + "_deletion"][0], reference)
    assert report["endpoints"]["actual"]["quadratic"] == 0
    assert torch.equal(actual, torch.ones_like(actual))
    json.dumps(report, allow_nan=False)


def test_rank_population_is_eligible_and_signed_cancellation_cannot_erase_token():
    actual = torch.ones(1, 4, 2)
    attribution = torch.tensor([[[4., -4.], [1., 1.], [99., 99.], [99., 99.]]])
    ctx = {"lang_attn_mask": torch.tensor([[True, True, False, False]])}
    assert ranked_group_indices(attribution, "language", ctx, actual).tolist() == [0, 1]
    with pytest.raises(ValueError, match="same eligible"):
        common_response_curves(lambda x: torch.zeros(1, 1, 128), actual, torch.zeros_like(actual),
            torch.zeros(1, 1, 128), {"one": torch.tensor([0, 1]), "other": torch.tensor([0, 2])}, "language", [0, 1])


def test_fp32_settings_restore_callers_configuration():
    before = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32)
    with fp32_math_settings() as settings:
        assert settings["float32_matmul_precision"] == "highest"
        assert settings["cuda_matmul_allow_tf32"] is False
        assert settings["cudnn_allow_tf32"] is False
    after = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32)
    assert before == after


def test_real_diffusers_solver_and_backward_pass_fp32_operation_audit():
    from diffusers import DPMSolverMultistepScheduler
    runner, cache = Runner(), cached_fixture()
    runner.noise_scheduler_sample = DPMSolverMultistepScheduler(
        num_train_timesteps=20, beta_schedule="squaredcos_cap_v2",
        prediction_type="sample", algorithm_type="dpmsolver++", solver_order=2)
    convert_downstream_to_fp32(runner)
    ctx, actions, scalars, pairs = make_cached_functions(runner, cache, torch.float32, "logpi")
    actual, baseline = pairs["language"]
    with fp32_math_settings(), FP32OperationAudit() as audit:
        leaf = ((actual + baseline) / 2).detach().requires_grad_(True)
        score = scalars["language"](leaf)
        gradient = torch.autograd.grad(score, leaf)[0]
    assert gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
    assert torch.equal(gradient[:, 2:], torch.zeros_like(gradient[:, 2:]))
    assert audit.operations
    # This fixture is linear in the language inputs, followed by a quadratic
    # score. Trapezoid integrates its linear gradient exactly up to fp32 error.
    result = integrated_gradients(scalars["language"], actual, baseline, m=2, return_result=True)
    assert result.relative_residual < 1e-5
    torch.testing.assert_close(result.attributions, (actual - baseline) * gradient, rtol=1e-5, atol=1e-7)
