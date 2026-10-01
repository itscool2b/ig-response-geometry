"""Independent analytical and differential checks for response geometry."""
from fractions import Fraction
import math

import numpy as np
import pytest
from scipy.integrate import quad, trapezoid
import torch

from analysis.revision.response_geometry_theory import (
    gradient_scale, normalized_responses, polynomial_rank_reversal,
)


def test_gradient_weight_identity_on_nonlinear_vector_action_and_at_self_reference():
    reference = torch.tensor([1.8, math.sin(1.) + 1.], dtype=torch.float64)
    for alpha in (0., .2, .8, .999, 1.):
        x = torch.tensor([alpha, alpha], dtype=torch.float64, requires_grad=True)
        action = torch.stack([x[0] ** 2 + .8 * x[1], torch.sin(x[0]) + x[1] ** 3])
        r = ((action - reference) ** 2).sum()
        q = -r / (2 * 2 * 3.)
        l2 = -torch.sqrt(r + 1e-12)
        q_gradient = torch.autograd.grad(q, x, retain_graph=True)[0]
        l_gradient = torch.autograd.grad(l2, x)[0]
        scale = gradient_scale(r.item(), active_entries=2, sigma_squared=3.)
        torch.testing.assert_close(q_gradient, scale * l_gradient, rtol=2e-14, atol=1e-15)
        if alpha == 1:
            assert torch.equal(q_gradient, torch.zeros_like(x))
            assert torch.equal(l_gradient, torch.zeros_like(x))


@pytest.mark.parametrize("epsilon", [0., 1e-12, .2, 100.])
def test_normalization_identity_endpoint_equalities_and_overshoot(epsilon):
    R = 2.
    r = np.array([0., .01, .5, 1.9, R, 3., 10.])
    values = normalized_responses(r, R, epsilon=epsilon)
    q = 1 - r / R
    l = (np.sqrt(R + epsilon) - np.sqrt(r + epsilon)) / (np.sqrt(R + epsilon) - np.sqrt(epsilon))
    np.testing.assert_allclose(values["quadratic"], q, rtol=0, atol=0)
    np.testing.assert_allclose(values["l2"], l, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(q - l, values["difference_identity"], rtol=1e-10, atol=1e-13)
    assert values["quadratic"][0] == values["l2"][0] == 1.
    assert values["quadratic"][4] == values["l2"][4] == 0.
    assert np.all(values["difference_identity"][1:4] > 0)
    assert np.all(values["difference_identity"][5:] < 0)


def test_positive_grid_weights_preserve_in_range_auc_order_but_not_overshoot_order():
    grid = np.array([0., .01, .3, .9, 1.])
    # Nonmonotone residuals still satisfy the theorem when they stay in range.
    in_range = normalized_responses([0., .8, .1, .9, 1.], 1.)
    assert trapezoid(in_range["quadratic"], grid) > trapezoid(in_range["l2"], grid)
    overshoot = normalized_responses([0., 10., 10., 10., 1.], 1.)
    assert trapezoid(overshoot["quadratic"], grid) < trapezoid(overshoot["l2"], grid)


def test_unstabilized_gap_has_u_one_minus_u_form():
    u = np.array([0., .1, .5, 1., 2.])
    result = normalized_responses((3 * u) ** 2, 9., epsilon=0.)
    np.testing.assert_allclose(result["difference_identity"], u * (1 - u), atol=1e-15)


def test_constructive_rank_reversal_is_exact_and_survives_production_stabilizer():
    example = polynomial_rank_reversal()
    assert example["quadratic_ig_exact"] == ["23/30", "64/75"]
    assert example["unstabilized_l2_ig_limit_exact"] == ["1", "4/5"]
    assert Fraction(example["quadratic_feature_2_minus_1_exact"]) == Fraction(13, 150)
    assert example["stabilized_l2_feature_1_minus_2_lower_bound"] > .19999
    smooth = np.array(example["stabilized_l2_ig"])
    deficit = np.array([1., .8]) - smooth
    assert np.all(deficit >= 0)
    assert np.all(deficit <= np.array(example["stabilized_l2_coordinate_deficit_upper_bounds"]))
    assert smooth[0] > smooth[1]
    assert example["stabilized_l2_completeness_absolute_residual"] < 1e-11
    # Independently integrate the polynomial gradients, not the closed form.
    d = lambda t: 1.8 - t * t - .8 * t
    measured = [quad(lambda t: 2 * t * d(t), 0, 1)[0], quad(lambda t: .8 * d(t), 0, 1)[0]]
    np.testing.assert_allclose(measured, [23 / 30, 64 / 75], rtol=1e-14)
    assert sum(measured) == pytest.approx(1.8 ** 2 / 2)


def test_a_target_change_does_not_necessarily_reverse_the_same_example_family():
    example = polynomial_rank_reversal(c=Fraction(1, 2))
    q = list(map(Fraction, example["quadratic_ig_exact"]))
    assert q[0] > q[1] and example["stabilized_l2_ig"][0] > example["stabilized_l2_ig"][1]


@pytest.mark.parametrize("r,R,epsilon", [(-1., 1., 1e-12), (0., 0., 1e-12), (0., -1., 1e-12),
                                       (math.nan, 1., 1e-12), (1., 1., -1.)])
def test_undefined_gaps_and_invalid_residuals_are_not_sanitized(r, R, epsilon):
    with pytest.raises(ValueError):
        normalized_responses(r, R, epsilon=epsilon)
