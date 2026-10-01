"""Independent calculus, response and production-kernel checks for the toy."""
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.integrate import quad
import torch

from analysis.revision.nested_grid_aliasing import (
    action, exact_ig, exact_trapezoidal_ig, intervention_curves,
    path_gradient, quadratic_target, ranking, run_example,
)
from integrated_gradients import integrated_gradients


@pytest.mark.parametrize("frequency", [1, 3, 8, 128])
def test_independent_integral_and_directional_cancellation(frequency):
    # Integrate each oscillatory half-period separately, using the independent
    # analytic derivative rather than the production quadrature or exact helper.
    edges = np.arange(2 * frequency + 1) / (2 * frequency)
    values = [math.fsum(quad(lambda t: path_gradient(t, frequency=frequency)[i], lo, hi,
                            epsabs=1e-13, epsrel=1e-13)[0]
                        for lo, hi in zip(edges, edges[1:])) for i in range(2)]
    np.testing.assert_allclose(values, [.3, .2], rtol=0, atol=2e-14)
    assert sum(exact_ig()) == Fraction(1, 2)
    for t in [.031, .197, .529, .873]:
        gradient = path_gradient(t, frequency=frequency)
        assert math.isclose(sum(gradient), 1 - t, rel_tol=0, abs_tol=2e-16)
        assert min(gradient) > 0


def test_autograd_matches_gradient_off_grid_and_transverse_term_is_nontrivial():
    for t in [.037, .218, .529, .899]:
        point = torch.tensor([t, t], dtype=torch.float64, requires_grad=True)
        q = quadratic_target(point, frequency=8)
        gradient = torch.autograd.grad(q, point)[0]
        torch.testing.assert_close(gradient, torch.tensor(path_gradient(t, frequency=8), dtype=torch.float64), atol=5e-15, rtol=0)
        assert math.isclose(float(q.detach()), -(t - 1) ** 2 / 2, rel_tol=0, abs_tol=1e-16)
    off_path = torch.tensor([.3, .45], dtype=torch.float64)
    # Its midpoint .375 happens to be a node for M=8, so choose M=3.
    assert not math.isclose(float(action(off_path, frequency=3)), .4 * .3 + .6 * .45, abs_tol=1e-3)


def test_exact_coarse_maps_have_nonzero_gap_and_opposite_true_rank():
    true = (Fraction(3, 10), Fraction(1, 5))
    for intervals in [1, 2, 4, 8, 16, 32, 64, 128]:
        estimate = exact_trapezoidal_ig(intervals, frequency=128)
        assert estimate == (Fraction(1, 5), Fraction(3, 10))
        assert sum(estimate) == sum(true) == Fraction(1, 2)
        assert ranking(estimate) == [1, 0] != ranking(true)
        assert sum(abs(a - b) for a, b in zip(estimate, true)) == Fraction(1, 5)
    assert exact_trapezoidal_ig(256, frequency=128) == true


def test_arbitrary_finite_uniform_grid_list_can_share_the_aliasing_frequency():
    budgets = [3, 5, 8, 12]
    frequency = math.lcm(*budgets)
    assert all(exact_trapezoidal_ig(m, frequency=frequency) == (Fraction(1, 5), Fraction(3, 10)) for m in budgets)
    # This is a family tailored to a fixed finite grid list, not a claim that a
    # particular adaptive algorithm can never resolve a particular function.
    with pytest.raises(ValueError, match="only"):
        exact_trapezoidal_ig(7, frequency=frequency)


def test_same_order_means_identical_masks_and_responses_not_accurate_coordinates():
    curves = [intervention_curves(ranking(exact_trapezoidal_ig(m))) for m in [32, 64, 128]]
    assert curves[0] == curves[1] == curves[2]
    np.testing.assert_allclose(curves[0]["insertion"]["normalized_quadratic"], [0, .84, 1], atol=1e-14)
    np.testing.assert_allclose(curves[0]["deletion"]["normalized_quadratic"], [1, .64, 0], atol=1e-14)
    assert intervention_curves(ranking(exact_ig())) != curves[0]
    assert intervention_curves((1, 0)) == curves[0]
    with pytest.raises(ValueError, match="permutation"):
        intervention_curves([True, False])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_production_trapezoid_is_repeatable_complete_and_wrong_on_three_grids(dtype):
    actual = torch.ones(2, dtype=dtype)
    baseline = torch.zeros_like(actual)
    forward = lambda z: quadratic_target(z, frequency=128)
    for m in [32, 64, 128, 256]:
        kwargs = dict(m=m, quadrature="trapezoid", arithmetic_dtype=dtype,
                      forward_dtype=dtype, return_result=True)
        first = integrated_gradients(forward, actual, baseline, **kwargs)
        second = integrated_gradients(forward, actual, baseline, **kwargs)
        assert torch.equal(first.attributions, second.attributions)
        expected = torch.tensor([float(v) for v in exact_trapezoidal_ig(m)], dtype=dtype)
        torch.testing.assert_close(first.attributions, expected, rtol=0, atol=1e-6)
        assert first.absolute_residual < 1e-6
        assert first.expected_gap == .5
        assert ranking(first.attributions) == ([1, 0] if m <= 128 else [0, 1])
        if m <= 128:
            assert sum(abs(a - float(b)) for a, b in zip(first.attributions.tolist(), exact_ig())) > .19999


def test_finer_grid_resolves_only_the_fixed_frequency_example():
    actual = torch.ones(2, dtype=torch.float64)
    orders = []
    for frequency in [128, 256]:
        result = integrated_gradients(lambda z: quadratic_target(z, frequency=frequency),
            actual, torch.zeros_like(actual), m=256, quadrature="trapezoid", arithmetic_dtype=torch.float64,
            forward_dtype=torch.float64, return_result=True)
        orders.append(ranking(result.attributions))
    assert orders == [[0, 1], [1, 0]]


def test_report_preserves_measured_residuals_and_common_response_scope():
    report = run_example(frequency=8)
    assert report["exact_ig"] == ["3/10", "1/5"]
    assert report["budgets"] == [2, 4, 8, 16]
    assert len(report["observations"]) == 8
    assert all(row["repeat_bitwise_equal"] for row in report["observations"])
    assert all(row["measured_absolute_completeness_residual"] >= 0 for row in report["observations"])
    assert any(row["measured_absolute_completeness_residual"] > 0 for row in report["observations"])
    assert report["environment"]["device"] == "cpu"
    assert "PickSingleYCB" in report["limits"][-1]
    json.dumps(report, allow_nan=False)


def test_cli_writes_new_report_and_refuses_replacement(tmp_path):
    output = tmp_path / "example.json"
    command = [sys.executable, "-m", "analysis.revision.nested_grid_aliasing", "--frequency", "4", "--output", str(output)]
    cwd = Path(__file__).resolve().parents[1]
    subprocess.run(command, cwd=cwd, check=True, capture_output=True)
    original = output.read_bytes()
    assert json.loads(original)["frequency"] == 4
    assert b"\r\n" not in original
    repeated = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    assert repeated.returncode != 0 and "fresh output" in repeated.stderr
    assert output.read_bytes() == original


@pytest.mark.parametrize("value", [0, -1, True, 2.5])
def test_invalid_frequency_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        exact_trapezoidal_ig(4, frequency=value)
