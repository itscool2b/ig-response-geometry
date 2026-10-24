"""The reviewer's one-dimensional endpoint-weighting example, four quadrature rules.

g(u) = u, baseline 0, input 1, reference action g(1) = 1, so r(u) = (1-u)^2.
N target F = -sqrt(r + eps); the path derivative is f(a) = (1-a)/sqrt((1-a)^2+eps)
and the exact attribution is sqrt(1+eps) - sqrt(eps). A Q target with d_a = 1,
F = -r/2, f(a) = 1 - a and exact value 1/2, is a contrast.

Float64 values use correctly rounded operations and math.fsum, so they are
platform-stable. Gauss-Legendre nodes for the stable float64 column and for
the 50-digit check come from mpmath Newton iteration on the Legendre
recurrence; numpy.polynomial.legendre.leggauss is evaluated separately because
its LAPACK eigenvalue step can move nodes by an ulp between platforms.
"""
from __future__ import annotations

import math
import platform

import mpmath as mp
import numpy as np

RULES = ("endpoint_inclusive", "trapezoid", "midpoint", "gauss_legendre")
DIGITS = 50
NODE_DIGITS = 70


def f64(target, a, eps):
    if target == "N":
        d = 1.0 - a
        return d / math.sqrt(d * d + eps)
    return 1.0 - a


def fmp(target, a, eps):
    if target == "N":
        d = 1 - a
        return d / mp.sqrt(d * d + eps)
    return 1 - a


def exact64(target, eps):
    return math.sqrt(1.0 + eps) - math.sqrt(eps) if target == "N" else 0.5


def exactmp(target, eps):
    return mp.sqrt(1 + eps) - mp.sqrt(eps) if target == "N" else mp.mpf(1) / 2


def legendre_nodes(m):
    """m-point Gauss-Legendre nodes and weights on [-1, 1] at NODE_DIGITS digits."""
    with mp.workdps(NODE_DIGITS):
        tolerance = mp.mpf(10) ** (-(NODE_DIGITS - 5))
        nodes, weights = [], []
        for k in range(1, m // 2 + 1):
            x = mp.cos(mp.pi * (k - mp.mpf(1) / 4) / (m + mp.mpf(1) / 2))
            for _ in range(200):
                p0, p1 = mp.mpf(1), x
                for n in range(2, m + 1):
                    p0, p1 = p1, ((2 * n - 1) * x * p1 - (n - 1) * p0) / n
                derivative = m * (x * p1 - p0) / (x * x - 1)
                step = p1 / derivative
                x -= step
                if abs(step) < tolerance:
                    break
            else:
                raise ArithmeticError("Legendre Newton iteration did not converge")
            p0, p1 = mp.mpf(1), x
            for n in range(2, m + 1):
                p0, p1 = p1, ((2 * n - 1) * x * p1 - (n - 1) * p0) / n
            derivative = m * (x * p1 - p0) / (x * x - 1)
            weight = 2 / ((1 - x * x) * derivative * derivative)
            nodes.extend((x, -x))
            weights.extend((weight, weight))
        if m % 2:
            p0, p1 = mp.mpf(1), mp.mpf(0)
            for n in range(2, m + 1):
                p0, p1 = p1, (-(n - 1) * p0) / n
            derivative = m * (-p0) / (-1)
            nodes.append(mp.mpf(0))
            weights.append(2 / (derivative * derivative))
        order = sorted(range(m), key=lambda i: nodes[i])
        return [nodes[i] for i in order], [weights[i] for i in order]


def rule64(rule, target, m, eps, nodes=None):
    if rule == "endpoint_inclusive":
        return math.fsum(f64(target, j / m, eps) for j in range(m + 1)) / (m + 1)
    if rule == "trapezoid":
        ends = [f64(target, 0.0, eps) / 2, f64(target, 1.0, eps) / 2]
        return math.fsum(ends + [f64(target, j / m, eps) for j in range(1, m)]) / m
    if rule == "midpoint":
        return math.fsum(f64(target, (j + 0.5) / m, eps) for j in range(m)) / m
    x, w = nodes
    return math.fsum((wi / 2.0) * f64(target, (xi + 1.0) / 2.0, eps) for xi, wi in zip(x, w))


def rulemp(rule, target, m, eps, nodes=None):
    if rule == "endpoint_inclusive":
        return mp.fsum(fmp(target, mp.mpf(j) / m, eps) for j in range(m + 1)) / (m + 1)
    if rule == "trapezoid":
        inner = mp.fsum(fmp(target, mp.mpf(j) / m, eps) for j in range(1, m))
        return (inner + (fmp(target, mp.mpf(0), eps) + fmp(target, mp.mpf(1), eps)) / 2) / m
    if rule == "midpoint":
        return mp.fsum(fmp(target, (j + mp.mpf(1) / 2) / m, eps) for j in range(m)) / m
    x, w = nodes
    return mp.fsum(wi / 2 * fmp(target, (xi + 1) / 2, eps) for xi, wi in zip(x, w))


def text(value, digits=40):
    return mp.nstr(value, digits, strip_zeros=False, min_fixed=-5, max_fixed=5)


def compute(protocol):
    spec = protocol["endpoint_example"]
    eps = float(spec["epsilon"])
    rows, identity, numpy_rows = [], [], []
    with mp.workdps(DIGITS):
        eps_mp = mp.mpf(eps)
        for m in spec["intervals"]:
            mp_nodes = legendre_nodes(m)
            with mp.workdps(DIGITS):
                nodes50 = ([+x for x in mp_nodes[0]], [+w for w in mp_nodes[1]])
            nodes64 = ([float(x) for x in mp_nodes[0]], [float(w) for w in mp_nodes[1]])
            numpy_x, numpy_w = np.polynomial.legendre.leggauss(m)
            for target in spec["targets"]:
                exact = exact64(target, eps)
                exact_mp = exactmp(target, eps_mp)
                for rule in RULES:
                    value = rule64(rule, target, m, eps, nodes64)
                    value_mp = rulemp(rule, target, m, eps_mp, nodes50)
                    rows.append(dict(
                        target=target, m=m, rule=rule, evaluations=m + 1 if rule in RULES[:2] else m,
                        uses_input_endpoint=rule in RULES[:2], value_float64=value,
                        relative_error_float64=(value - exact) / exact,
                        value_mp50=text(value_mp), relative_error_mp50=text((value_mp - exact_mp) / exact_mp),
                        float64_minus_mp50=float(value - value_mp), exact_float64=exact, exact_mp50=text(exact_mp),
                        rounded_6=round(value, 6)))
                    if rule == "gauss_legendre":
                        legacy = rule64(rule, target, m, eps, (numpy_x.tolist(), numpy_w.tolist()))
                        numpy_rows.append(dict(target=target, m=m, value_numpy_leggauss=legacy,
                                               minus_stable_float64=legacy - value,
                                               minus_mp50=float(legacy - value_mp)))
                average, trapezoid = rule64(RULES[0], target, m, eps), rule64(RULES[1], target, m, eps)
                ends = (f64(target, 0.0, eps) + f64(target, 1.0, eps)) / 2
                rhs = trapezoid + (ends - trapezoid) / (m + 1)
                average_mp, trapezoid_mp = rulemp(RULES[0], target, m, eps_mp), rulemp(RULES[1], target, m, eps_mp)
                ends_mp = (fmp(target, mp.mpf(0), eps_mp) + fmp(target, mp.mpf(1), eps_mp)) / 2
                rhs_mp = trapezoid_mp + (ends_mp - trapezoid_mp) / (m + 1)
                identity.append(dict(target=target, m=m, endpoint_inclusive=average, trapezoid=trapezoid,
                                     f_at_baseline=f64(target, 0.0, eps), f_at_input=f64(target, 1.0, eps),
                                     identity_rhs=rhs, float64_difference=average - rhs,
                                     mp50_difference=text(average_mp - rhs_mp)))
    environment = dict(numpy=np.__version__, system=platform.system(), machine=platform.machine())
    return rows, identity, dict(rows=numpy_rows, environment=environment,
                                note="numpy.polynomial.legendre.leggauss nodes come from a LAPACK eigenvalue "
                                     "solve plus one Newton step; their last bits, and so these sums, can differ "
                                     "by a few ulps between platforms")
