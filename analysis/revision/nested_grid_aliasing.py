"""A smooth self-reference IG counterexample to finite nested-grid agreement.

This standalone CPU example does not modify any frozen analysis output or RDT
runtime. Exact identities concern real arithmetic. Measured kernel residuals
remain in the report and are not replaced by their exact mathematical values.

Run: python -m analysis.revision.nested_grid_aliasing --output runs/aliasing.json
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import torch

from integrated_gradients import integrated_gradients


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def action(z, *, frequency=128):
    """Smooth scalar policy; no special handling or snapping at grid nodes."""
    _positive_integer(frequency, "frequency")
    if not isinstance(z, torch.Tensor) or z.shape != (2,) or not z.is_floating_point():
        raise ValueError("Expected two floating-point coordinates")
    middle = (z[0] + z[1]) / 2
    h = (2 / 5) * torch.sin(math.pi * frequency * middle).square()
    return (2 / 5) * z[0] + (3 / 5) * z[1] + (z[0] - z[1]) * h


def quadratic_target(z, *, frequency=128):
    """Q=-(mu-1)^2/2, with detached own reference 1 and D=sigma_squared=1."""
    return -(action(z, frequency=frequency) - 1).square() / 2


def path_gradient(t, *, frequency=128):
    """Analytic coordinate gradient on (t,t), independent of torch autograd."""
    _positive_integer(frequency, "frequency")
    t = float(t)
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("Path location must lie in [0,1]")
    h = (2 / 5) * math.sin(math.pi * frequency * t) ** 2
    return ((1 - t) * (2 / 5 + h), (1 - t) * (3 / 5 - h))


def exact_ig():
    return (Fraction(3, 10), Fraction(1, 5))


def exact_trapezoidal_ig(intervals, *, frequency=128):
    """Exact values only for the two grid classes proved in the appendix.

    For m dividing M all sampled h values vanish. For m=2M they alternate
    between zero and 2/5. Other grids are deliberately not inferred here.
    """
    _positive_integer(intervals, "intervals")
    _positive_integer(frequency, "frequency")
    if frequency % intervals == 0:
        return (Fraction(1, 5), Fraction(3, 10))
    if intervals == 2 * frequency:
        return exact_ig()
    raise ValueError("Exact formula is stated only for m dividing M or m=2M")


def ranking(values):
    """Absolute-coordinate order, with the production stable-index tie rule."""
    values = tuple(float(v) for v in values)
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected two finite attribution coordinates")
    return sorted(range(2), key=lambda i: (-abs(values[i]), i))


def intervention_curves(order, *, frequency=128):
    """Evaluate the same toy policy at all whole-feature prefix masks.

    These are deterministic saved-order responses, not additional IG estimates.
    Q is normalized using its actual/baseline gap 1/2. Raw RMS is |mu-1| for
    one active action. Response arithmetic is common float64 for every order.
    """
    if len(order) != 2 or any(type(i) is not int for i in order) or sorted(order) != [0, 1]:
        raise ValueError("Expected a permutation of the two feature indices")
    order = list(order)
    result = {}
    for direction in ("insertion", "deletion"):
        points, actions = [], []
        for count in range(3):
            point = torch.zeros(2, dtype=torch.float64) if direction == "insertion" else torch.ones(2, dtype=torch.float64)
            point[order[:count]] = 1 if direction == "insertion" else 0
            points.append(point.tolist())
            actions.append(float(action(point, frequency=frequency)))
        q = [-(value - 1) ** 2 / 2 for value in actions]
        result[direction] = dict(fractions=[0., .5, 1.], points=points,
                                 action=actions, quadratic=q,
                                 normalized_quadratic=[1 + 2 * value for value in q],
                                 raw_rms=[abs(value - 1) for value in actions])
    return result


def run_example(*, frequency=128):
    """Run the unchanged production IG kernel on an explicit CPU toy only."""
    _positive_integer(frequency, "frequency")
    if frequency % 4:
        raise ValueError("The displayed three nested coarse grids require M divisible by 4")
    budgets = (frequency // 4, frequency // 2, frequency, 2 * frequency)
    observations = []
    for dtype in (torch.float32, torch.float64):
        actual = torch.ones(2, dtype=dtype, device="cpu")
        baseline = torch.zeros_like(actual)
        forward = lambda z: quadratic_target(z, frequency=frequency)
        for intervals in budgets:
            kwargs = dict(m=intervals, quadrature="trapezoid", arithmetic_dtype=dtype,
                          forward_dtype=dtype, return_result=True)
            first = integrated_gradients(forward, actual, baseline, **kwargs)
            repeated = integrated_gradients(forward, actual, baseline, **kwargs)
            values = first.attributions.tolist()
            order = ranking(values)
            error = math.fsum(abs(value - float(truth)) for value, truth in zip(values, exact_ig()))
            observations.append(dict(
                dtype=str(dtype), intervals=intervals, measured_ig=values,
                exact_grid_ig=[str(v) for v in exact_trapezoidal_ig(intervals, frequency=frequency)],
                measured_gap=first.expected_gap, measured_attribution_sum=first.attribution_sum,
                measured_absolute_completeness_residual=first.absolute_residual,
                measured_relative_completeness_residual=first.relative_residual,
                measured_coordinate_l1_error=error, measured_relative_coordinate_l1_error=error / .5,
                ranking=order, repeat_bitwise_equal=torch.equal(first.attributions, repeated.attributions),
                response_curves=intervention_curves(order, frequency=frequency)))
    root = Path(__file__).resolve().parents[2]
    return dict(
        kind="smooth_nested_grid_aliasing_counterexample", version=1,
        scope="Analytic construction with CPU kernel checks; not an RDT failure diagnosis or a numerical-setting approval.",
        frequency=frequency, budgets=list(budgets), baseline=[0, 0], actual=[1, 1],
        fixed_own_reference=1, active_entries=1, sigma_squared=1,
        exact_ig=[str(v) for v in exact_ig()], exact_endpoint_gap="1/2",
        aliased_grid_ig=["1/5", "3/10"], aliased_signed_coordinate_error=["-1/10", "1/10"],
        aliased_coordinate_l1_error="1/5", aliased_relative_coordinate_l1_error="2/5",
        exact_ranking=[0, 1], aliased_ranking=[1, 0], observations=observations,
        environment=dict(torch=torch.__version__, device="cpu"),
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in ("analysis/revision/nested_grid_aliasing.py", "integrated_gradients.py")},
        limits=["Perfect exact-grid equalities do not imply bitwise equal FP32 or FP64 quadrature outputs; their measured residuals are retained.",
                "The 2M grid resolves this particular sinusoidal construction, not every smooth target.",
                "Common response curves agree because the coarse maps select the same ordered masks on the same policy.",
                "No claim is made that this mechanism caused the observed PickSingleYCB failures."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("Choose a fresh output path; frozen results are not overwritten")
    report = run_example(frequency=args.frequency)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        print(json.dumps(dict(output=str(args.output), observations=len(report["observations"]))))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
