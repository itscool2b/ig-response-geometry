"""Analytical CPU validation, independent of RDT/checkpoints or GPU hardware."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from integrated_gradients import (
    IntegratedGradientsResult,
    NonFiniteAttributionError,
    QUADRATURE_RULES,
    integrated_gradients,
)


@pytest.mark.parametrize("rule", QUADRATURE_RULES)
@pytest.mark.parametrize("m", [16, 64, 128, 300, 1000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_historical_linear_coordinate_failure_is_repaired(dtype, m, rule):
    # Original audit fixture: bf16 errors reached 6.25% at m=64 despite
    # reported completeness error of only 0.26%. Every coordinate is known.
    coefficients = torch.arange(128, 256, dtype=torch.float32) / 128
    actual = torch.ones(128, dtype=dtype)
    result = integrated_gradients(
        lambda z: (coefficients * z.float()).sum(), actual,
        torch.zeros_like(actual), m=m, quadrature=rule, return_result=True,
    )
    torch.testing.assert_close(result.attributions, coefficients, atol=1e-6, rtol=1e-6)
    assert result.attributions.dtype == torch.float32
    assert result.forward_dtype == str(dtype)
    assert result.absolute_residual == 0


@pytest.mark.parametrize("rule", QUADRATURE_RULES)
def test_bf16_displacement_product_and_signed_coordinates_use_master_precision(rule):
    actual = torch.tensor([1.0, 1.0078125, -0.5, 0.125], dtype=torch.bfloat16)
    baseline = torch.tensor([-256.0, 0.0, 0.25, -0.125], dtype=torch.bfloat16)
    coefficients = torch.tensor([1.0, 1.0078125, -2.0, -0.25])
    result = integrated_gradients(
        lambda z: (z.float() * coefficients).sum(), actual, baseline,
        m=64, quadrature=rule, return_result=True,
    )
    expected = (actual.float() - baseline.float()) * coefficients
    torch.testing.assert_close(result.attributions, expected, atol=1e-6, rtol=1e-6)
    assert result.attributions[0] == 257  # bf16 displacement would round to 256
    assert result.attributions[1] != expected[1].to(torch.bfloat16).float()
    assert result.attribution_sum == expected.sum().item()


@pytest.mark.parametrize("rule", QUADRATURE_RULES)
def test_quadratic_signed_coordinates_have_exact_integral(rule):
    actual = torch.tensor([2.0, -3.0, 0.25], dtype=torch.float64)
    baseline = torch.tensor([-1.0, 1.0, -0.5], dtype=torch.float64)
    coefficients = torch.tensor([2.0, -0.5, 4.0], dtype=torch.float64)
    result = integrated_gradients(
        lambda z: (coefficients * z.square()).sum(), actual, baseline,
        m=8, quadrature=rule, arithmetic_dtype=torch.float64, return_result=True,
    )
    expected = coefficients * (actual.square() - baseline.square())
    torch.testing.assert_close(result.attributions, expected, atol=1e-10, rtol=1e-10)
    assert result.attributions.dtype == torch.float64
    assert result.absolute_residual == 0


@pytest.mark.parametrize("rule", QUADRATURE_RULES)
def test_cubic_distinguishes_quadrature_weights_and_matches_analytic_error(rule):
    m = 8
    result = integrated_gradients(
        lambda z: z.pow(3).sum(), torch.ones(1, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64), m=m, quadrature=rule,
        arithmetic_dtype=torch.float64, return_result=True,
    )
    # For integral_0^1 3a^2 da, trapezoid error is 1/(2m^2).
    # The historical equally weighted endpoints instead give error 1/(2m).
    error = 1 / (2 * m * m) if rule == "trapezoid" else 1 / (2 * m)
    assert result.attributions.item() == pytest.approx(1 + error, abs=1e-10)
    assert result.absolute_residual == pytest.approx(error, abs=1e-10)
    assert result.diagnostics()["forward_evaluations"] == m + 1


def test_relu_crossing_has_bounded_coordinate_quadrature_error():
    actual = torch.tensor([1.0, 1.0], dtype=torch.float64)
    baseline = torch.tensor([-1.0, -2.0], dtype=torch.float64)
    coefficients = torch.tensor([3.0, -2.0], dtype=torch.float64)
    m = 128
    result = integrated_gradients(
        lambda z: (coefficients * torch.relu(z)).sum(), actual, baseline,
        m=m, arithmetic_dtype=torch.float64, return_result=True,
    )
    expected = coefficients * (torch.relu(actual) - torch.relu(baseline))
    # The one discontinuity in each derivative can affect at most one cell.
    error_bound = coefficients.abs() * (actual - baseline).abs() / m
    assert bool(((result.attributions - expected).abs() <= error_bound).all())


def test_exact_endpoints_are_reused_without_reconstruction_or_extra_forwards():
    points = []

    def forward(z):
        points.append(z.detach().clone())
        return (z * 1e-8).sum()

    actual = torch.tensor([1.0])
    baseline = torch.tensor([1e8])
    result = integrated_gradients(forward, actual, baseline, m=2, return_result=True)
    assert len(points) == 3
    assert torch.equal(points[0], baseline)
    assert torch.equal(points[-1], actual)
    assert result.input_score == pytest.approx(1e-8)


def test_zero_gap_retains_signed_coordinates_and_json_safe_undefined_residual():
    result = integrated_gradients(
        lambda z: z[0] - z[1], torch.ones(2), torch.zeros(2), m=8, return_result=True,
    )
    torch.testing.assert_close(result.attributions, torch.tensor([1.0, -1.0]))
    assert result.expected_gap == 0
    assert result.relative_residual is None
    assert result.absolute_residual == 0
    assert result.diagnostics()["gap_status"] == "zero_gap"
    json.dumps(result.diagnostics(), allow_nan=False)


def test_zero_displacement_has_zero_attribution_even_with_nonzero_derivative():
    actual = torch.tensor([2.0, -3.0])
    result = integrated_gradients(lambda z: z.sum(), actual, actual, m=2, return_result=True)
    assert torch.equal(result.attributions, torch.zeros_like(actual))
    assert result.relative_residual is None


def test_forward_dtype_can_be_changed_independently_of_arithmetic_dtype():
    seen_dtypes = []

    def forward(z):
        seen_dtypes.append(z.dtype)
        return (z * 1.00001).sum()

    actual = torch.ones(2, dtype=torch.bfloat16)
    result = integrated_gradients(
        forward, actual, torch.zeros_like(actual), m=2,
        forward_dtype=torch.float64, arithmetic_dtype=torch.float64, return_result=True,
    )
    assert seen_dtypes == [torch.float64] * 3
    torch.testing.assert_close(
        result.attributions, torch.full((2,), 1.00001, dtype=torch.float64),
        atol=1e-10, rtol=1e-10,
    )


def test_tensor_api_and_enclosing_no_grad_remain_supported():
    with torch.no_grad():
        result = integrated_gradients(lambda z: z.sum(), torch.ones(2), torch.zeros(2), 2)
    assert isinstance(result, torch.Tensor)
    assert not result.requires_grad
    assert torch.equal(result, torch.ones(2))


class CorruptedBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x[0] - x[1] + x[2] + x[3]

    @staticmethod
    def backward(ctx, upstream):
        return upstream * torch.tensor([float("nan"), float("inf"), -float("inf"), 1.0])


def test_nonfinite_backward_cannot_be_hidden_by_scalar_completeness():
    with pytest.raises(NonFiniteAttributionError) as error:
        integrated_gradients(
            CorruptedBackward.apply, torch.ones(4), torch.zeros(4), m=64,
            diagnostic_context={"modality": "vision", "run_id": "analytical-test"},
        )
    diagnostic = error.value.diagnostics
    assert diagnostic["stage"] == "gradient"
    assert diagnostic["alpha"] == 0
    assert diagnostic["nonfinite_count"] == 3
    assert (diagnostic["nan_count"], diagnostic["posinf_count"], diagnostic["neginf_count"]) == (1, 1, 1)
    assert diagnostic["sample_indices"] == [[0], [1], [2]]
    assert diagnostic["context"]["modality"] == "vision"
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("field", ["input", "baseline"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_inputs_fail_before_forward(field, value):
    actual, baseline = torch.ones(2), torch.zeros(2)
    (actual if field == "input" else baseline)[0] = value

    def unexpected_forward(z):
        pytest.fail("Nonfinite input must be rejected before calling the model")

    with pytest.raises(NonFiniteAttributionError) as error:
        integrated_gradients(unexpected_forward, actual, baseline, m=2)
    assert error.value.diagnostics["stage"] == field


@pytest.mark.parametrize("bad_alpha", [0.0, 0.5, 1.0])
def test_nonfinite_endpoint_or_interior_score_fails(bad_alpha):
    def forward(z):
        return z.sum() * (float("nan") if z.item() == bad_alpha else 1)

    with pytest.raises(NonFiniteAttributionError) as error:
        integrated_gradients(forward, torch.ones(1), torch.zeros(1), m=2)
    assert error.value.diagnostics["stage"] == "score"
    assert error.value.diagnostics["alpha"] == bad_alpha


class FiniteScoreHugeGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.shape = z.shape
        return z.sum() * 0

    @staticmethod
    def backward(ctx, upstream):
        return upstream * torch.full(ctx.shape, torch.finfo(torch.float32).max)


def test_overflow_in_accumulation_fails_instead_of_returning_a_map():
    with pytest.raises(NonFiniteAttributionError) as error:
        integrated_gradients(FiniteScoreHugeGradient.apply, torch.ones(1), torch.zeros(1), m=2)
    assert error.value.diagnostics["stage"] == "gradient_sum"


def test_overflow_in_attribution_product_fails():
    with pytest.raises(NonFiniteAttributionError) as error:
        integrated_gradients(
            FiniteScoreHugeGradient.apply, torch.tensor([2.0]), torch.zeros(1), m=1,
        )
    assert error.value.diagnostics["stage"] == "attribution"


def test_finite_master_values_that_overflow_model_dtype_fail():
    with pytest.raises(NonFiniteAttributionError) as error:
        integrated_gradients(
            lambda z: z.sum(), torch.tensor([1e10]), torch.zeros(1), m=2,
            forward_dtype=torch.float16,
        )
    assert error.value.diagnostics["stage"] == "model_input"


def test_displacement_overflow_fails():
    huge = torch.finfo(torch.float32).max
    with pytest.raises(NonFiniteAttributionError) as error:
        integrated_gradients(lambda z: z.sum(), torch.tensor([huge]), torch.tensor([-huge]), m=2)
    assert error.value.diagnostics["stage"] == "displacement"


@pytest.mark.parametrize("m", [0, -1, 1.5, True, "3"])
def test_invalid_integration_budgets_fail(m):
    with pytest.raises(ValueError, match="positive integer"):
        integrated_gradients(lambda z: z.sum(), torch.ones(2), torch.zeros(2), m=m)


@pytest.mark.parametrize("kwargs", [{"quadrature": "riemann"}, {"arithmetic_dtype": torch.bfloat16}, {"forward_dtype": torch.int32}])
def test_invalid_numerical_modes_fail(kwargs):
    with pytest.raises(ValueError):
        integrated_gradients(lambda z: z.sum(), torch.ones(2), torch.zeros(2), **kwargs)


def test_shape_mismatch_cannot_silently_broadcast():
    with pytest.raises(ValueError, match="identical shapes"):
        integrated_gradients(lambda z: z.sum(), torch.ones(2), torch.zeros(1))


@pytest.mark.parametrize("forward", [lambda z: z, lambda z: 1.0, lambda z: z.sum().detach()])
def test_bad_forward_contract_fails(forward):
    with pytest.raises(ValueError):
        integrated_gradients(forward, torch.ones(2), torch.zeros(2), m=2)


@pytest.fixture
def per_step_module(monkeypatch):
    # The wrapper needs only the public state-index mapping at import time.
    # Provide that contract without downloading the GPU upstream or a model.
    state_config = ModuleType("configs.state_vec")
    state_config.STATE_VEC_IDX_MAPPING = {
        **{f"right_arm_joint_{i}_pos": i for i in range(7)},
        "right_gripper_open": 10,
    }
    monkeypatch.setitem(sys.modules, "configs.state_vec", state_config)
    saved_path = sys.path[:]
    spec = importlib.util.spec_from_file_location(
        "numerical_test_per_step", Path(__file__).resolve().parents[1] / "per_step_attribution.py",
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = saved_path


def test_per_modality_wrapper_reuses_endpoints_and_keeps_fp32_results(per_step_module, monkeypatch):
    mod = per_step_module
    zero = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
    actual = zero.clone()
    actual[..., mod.MANISKILL_INDICES] = 1
    ctx = {
        "img_adapted": actual, "img_adapted_bl": zero,
        "lang_adapted": actual, "lang_adapted_bl": zero,
        "state_input_actual": actual, "state_input_baseline": zero,
        "ref_action": zero,
        "initial_noise": zero,
        "initial_noise_sha256": "fixture-noise",
        "sampler_metadata": {"adapter_version": "fixture"},
    }
    calls = []

    def forward(z):
        calls.append(z)
        return z.sum()

    monkeypatch.setattr(mod, "prepare_ig_context", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(mod, "build_forward_fns", lambda _ctx: (forward,) * 3)
    result = mod.compute_ig_for_step(*([None] * 12), m=4, quadrature="legacy_endpoint_average")
    assert len(calls) == 3 * (4 + 1)
    for name in ["vision", "language", "state"]:
        stats = result[name]
        assert stats["attribution"].dtype == torch.float32
        assert stats["ig_sum"] == stats["expected_gap"] == 8
        assert stats["completeness_err"] == 0
        assert stats["numerics"]["quadrature"] == "legacy_endpoint_average"
        assert stats["numerics"]["nonfinite_count"] == 0
    assert result["state"]["per_joint"].tolist() == [1] * 8


def test_per_modality_failure_has_run_and_modality_identity(per_step_module, monkeypatch):
    mod = per_step_module
    ctx = {"img_adapted": torch.ones(4), "img_adapted_bl": torch.zeros(4)}
    monkeypatch.setattr(mod, "prepare_ig_context", lambda *args, **kwargs: ctx)
    monkeypatch.setattr(mod, "build_forward_fns", lambda _ctx: (CorruptedBackward.apply,) * 3)
    with pytest.raises(NonFiniteAttributionError) as error:
        mod.compute_ig_for_step(*([None] * 12), m=2, diagnostic_context={"run_id": "run-test"})
    assert error.value.diagnostics["context"] == {"run_id": "run-test", "modality": "vision"}
