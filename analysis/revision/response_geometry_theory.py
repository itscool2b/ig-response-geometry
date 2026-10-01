"""Elementary response-transform identities and a constructive IG rank reversal.

This CPU-only analysis is independent of the frozen RDT runtime. It proves no
new general metric principle and authenticates no historical model context.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad


def _positive(value, name, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _residual(value):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Squared residual must be finite and nonnegative")
    return result


def gradient_scale(squared_residual, *, active_entries, sigma_squared=1., epsilon=1e-12):
    """Multiplier satisfying grad Q = multiplier * grad stabilized L2.

    A fixed detached reference, the same differentiable action map, the same
    active entries and strictly positive epsilon are required.
    """
    if type(active_entries) is not int or active_entries < 1:
        raise ValueError("active_entries must be a positive integer")
    sigma_squared = _positive(sigma_squared, "sigma_squared")
    epsilon = _positive(epsilon, "epsilon")
    return np.sqrt(_residual(squared_residual) + epsilon) / (active_entries * sigma_squared)


def normalized_responses(squared_residual, baseline_squared_residual, *, epsilon=1e-12):
    """Exact-self-reference responses with Q(0), L(0) as actual endpoints.

    R>0 is mandatory. The epsilon in L is not a substitute for an endpoint
    normalization gap. epsilon=0 exposes the unstabilized limiting identity.
    """
    r = _residual(squared_residual)
    R = _positive(baseline_squared_residual, "baseline_squared_residual")
    epsilon = _positive(epsilon, "epsilon", zero=True)
    a, b, z = math.sqrt(epsilon), math.sqrt(R + epsilon), np.sqrt(r + epsilon)
    quadratic = 1 - r / R
    # Rationalization avoids subtracting nearly equal stabilized endpoints.
    l2 = quadratic * (b + a) / (b + z)
    difference_identity = (z - a) * (b - z) / R
    return dict(quadratic=quadratic, l2=l2, difference_identity=difference_identity)


def polynomial_rank_reversal(*, c=Fraction(4, 5), epsilon=1e-12):
    """mu(x)=x1**2+c*x2, baseline=(0,0), actual=(1,1), D=sigma²=1.

    Exact rational Q and limiting L values are kept separate from adaptive
    quadrature of the smooth epsilon-positive L target.
    """
    c = Fraction(c)
    if not 0 < c < 1:
        raise ValueError("The constructive example uses 0<c<1")
    epsilon = _positive(epsilon, "epsilon")
    q = (Fraction(1, 2) + c / 3, 2 * c / 3 + c * c / 2)
    limit = (Fraction(1), c)
    cf, root = float(c), math.sqrt(epsilon)
    bounds = (2 * root / cf, root)

    def integrand(t, feature):
        d = (1 - t) * (1 + t + cf)
        derivative = 2 * t if feature == 0 else cf
        return derivative * d / math.hypot(d, root)

    # Resolve the narrow regularization boundary layer adjacent to t=1.
    points = sorted({0., max(0., 1 - 100 * root / cf), max(0., 1 - root / cf), 1.})
    smooth, errors = [], []
    for feature in range(2):
        pieces = [quad(integrand, lo, hi, args=(feature,), epsabs=1e-13, epsrel=1e-13, limit=200)
                  for lo, hi in zip(points, points[1:])]
        smooth.append(math.fsum(value for value, _ in pieces))
        errors.append(math.fsum(error for _, error in pieces))
    gap = math.hypot(1 + cf, root) - root
    return dict(
        kind="analytic_constructive_example_not_RDT_evidence", c=str(c), epsilon=epsilon,
        policy="mu(x1,x2)=x1^2+c*x2", baseline=[0, 0], actual=[1, 1],
        reference_action=1 + cf, active_entries=1, sigma_squared=1,
        quadratic_ig_exact=[str(v) for v in q],
        unstabilized_l2_ig_limit_exact=[str(v) for v in limit],
        stabilized_l2_ig=smooth, quadrature_error_estimates=errors,
        stabilized_l2_coordinate_deficit_upper_bounds=list(bounds),
        stabilized_l2_completeness_absolute_residual=abs(math.fsum(smooth) - gap),
        quadratic_feature_2_minus_1_exact=str(q[1] - q[0]),
        stabilized_l2_feature_1_minus_2_lower_bound=1 - bounds[0] - cf,
        interpretation="An output-transform-dependent path weight can reverse IG ranks; reversal is not universal.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = polynomial_rank_reversal()
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
