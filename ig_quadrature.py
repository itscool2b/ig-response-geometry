"""Quadrature rules and path-node evaluation for the NHSJS revision numerics.

integrated_gradients.py stays byte-identical because the manuscript lineage
pins it. This module adds what the revision needs beside it:

- master-precision node evaluation that counts nonfinite values instead of
  raising, so a failing cell is recorded with every node still evaluated;
- one pass over the nested dyadic grid j/N that yields the composite
  trapezoid rule T_m, the endpoint-inclusive average E_m (paper Equation 3)
  and the midpoint rule M_n for every power-of-two budget at once, with the
  Simpson and Romberg combinations derived from them;
- Gauss-Legendre rules and an adaptive vector Gauss-Kronrod (21-point)
  reference, neither of which evaluates a path endpoint;
- a replica of the historical bf16 kernel, archived at
  docs/tmlr-readiness/archives/active-originals/2026-10-01-stable-base/
  integrated_gradients.py.historical, that counts nonfinite gradients before
  replacing them with zero as the historical code did without recording it.

Every map is accumulated on the eligible feature slice. Coordinates outside it
have zero displacement, so their attribution is zero under every rule.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np
import torch

from integrated_gradients import NonFiniteAttributionError

# QUADPACK qk21 abscissae (nonnegative half, decreasing) and weights.
GK21_NODES = (
    0.995657163025808080735527280689003, 0.973906528517171720077964012084452,
    0.930157491355708226001207180059508, 0.865063366688984510732096688423493,
    0.780817726586416897063717578345042, 0.679409568299024406234327365114874,
    0.562757134668604683339000099272694, 0.433395394129247190799265943165784,
    0.294392862701460198131126603103866, 0.148874338981631210884826001129720,
    0.000000000000000000000000000000000,
)
GK21_KRONROD_WEIGHTS = (
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390,
    0.054755896574351996031381300244580, 0.075039674810919952767043140916190,
    0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077958109831074, 0.134709217311473325928054001771707,
    0.142775938577060080797094273138717, 0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
)
# Gauss 10-point weights for GK21_NODES[1], [3], [5], [7], [9].
GK21_GAUSS_WEIGHTS = (
    0.066671344308688137593568809893332, 0.149451349150580593145776339657697,
    0.219086362515982043995534934228163, 0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
)


def nonfinite_counts(tensor):
    """Counts of NaN, +inf and -inf values; zero dict for a finite tensor."""
    if bool(torch.isfinite(tensor).all()):
        return dict(nan=0, posinf=0, neginf=0)
    return dict(nan=int(torch.isnan(tensor).sum()), posinf=int(torch.isposinf(tensor).sum()),
                neginf=int(torch.isneginf(tensor).sum()))


def total_nonfinite(counts):
    return counts["nan"] + counts["posinf"] + counts["neginf"]


class Eligible:
    """Eligible feature groups of one modality tensor (batch one, rank three)."""

    def __init__(self, indices, axis):
        if axis not in (1, 2) or not indices or sorted(set(indices)) != list(indices):
            raise ValueError("Eligible indices must be sorted, distinct and on axis 1 or 2")
        self.indices = list(indices)
        self.axis = axis
        self._index = None

    def take(self, tensor):
        if self._index is None or self._index.device != tensor.device:
            self._index = torch.tensor(self.indices, device=tensor.device)
        return tensor.index_select(self.axis, self._index)

    def group_scores(self, eligible_map):
        """Sum of absolute coordinate values within each eligible group."""
        reduce_dim = 2 if self.axis == 1 else 1
        return eligible_map.abs().sum(dim=reduce_dim)[0]

    def group_counts(self, eligible_tensor, predicate):
        reduce_dim = 2 if self.axis == 1 else 1
        return predicate(eligible_tensor).sum(dim=reduce_dim)[0]


def _point(baseline, actual, displacement, alpha):
    # Exact endpoints, as in integrated_gradients: no subtract/add cancellation.
    if alpha == 0.0:
        return baseline
    if alpha == 1.0:
        return actual
    return baseline + alpha * displacement


def evaluate_node(score_fn, baseline, actual, displacement, alpha, *, model_dtype, eligible,
                  want_gradient=True):
    """Evaluate at baseline + alpha * displacement, with exact endpoints."""
    score, gradient, record = evaluate_point(score_fn, _point(baseline, actual, displacement, alpha),
                                             model_dtype=model_dtype, eligible=eligible,
                                             want_gradient=want_gradient)
    record["alpha"] = alpha
    return score, gradient, record


def evaluate_point(score_fn, point, *, model_dtype, eligible, want_gradient=True):
    """Evaluate the target and, optionally, its master-precision path gradient.

    The leaf has the master dtype and is cast to the model dtype without
    detaching, so autograd returns the gradient in master precision, exactly
    as integrated_gradients does. Nonfinite values are counted, never replaced.
    Returns (score, eligible_gradient, record); score and gradient are None when
    the forward itself fails.
    """
    record = {}
    with torch.enable_grad():
        leaf = point.detach().requires_grad_(want_gradient)
        model_input = leaf.to(dtype=model_dtype)
        try:
            score = score_fn(model_input)
        except NonFiniteAttributionError as exc:
            record.update(status="forward_nonfinite", stage=exc.diagnostics.get("stage"),
                          inner_step=exc.diagnostics.get("step"),
                          forward_nonfinite_count=exc.diagnostics.get("nonfinite_count"))
            return None, None, record
        if not isinstance(score, torch.Tensor) or score.numel() != 1:
            raise ValueError("score_fn must return one scalar Tensor")
        record["score_nonfinite"] = nonfinite_counts(score.detach())
        value = float(score.detach())
        if not want_gradient:
            record["status"] = "score_only"
            return value, None, record
        gradient = torch.autograd.grad(score, leaf)[0]
    record["gradient_nonfinite_full"] = nonfinite_counts(gradient)
    selected = eligible.take(gradient)
    record["gradient_nonfinite_eligible"] = nonfinite_counts(selected)
    if total_nonfinite(record["gradient_nonfinite_eligible"]):
        bad = eligible.group_counts(selected, lambda t: ~torch.isfinite(t))
        record["nonfinite_groups"] = {str(eligible.indices[i]): int(c) for i, c in enumerate(bad.tolist()) if c}
    record["status"] = "finite" if not (total_nonfinite(record["gradient_nonfinite_full"])
                                        or total_nonfinite(record["score_nonfinite"])) else "nonfinite"
    return value, selected, record


def dyadic_pass(score_fn, actual, baseline, levels, *, model_dtype, eligible,
                master_dtype=torch.float32, keep_gradients=False, on_node=None):
    """Evaluate the 2**levels + 1 nodes j/N once and accumulate every nested rule.

    T_m (composite trapezoid) and E_m (endpoint-inclusive average, paper Eq. 3)
    use the nodes k/m for every m = 2**l, l = 0..levels; M_n (midpoint) uses the
    nodes (k + 1/2)/n for every n = 2**l, l = 0..levels-1. Accumulators add in
    node order with the same weights as integrated_gradients, so T_m and E_m on
    this grid equal the pinned kernel's maps bit for bit at matching m.
    """
    if type(levels) is not int or levels < 1:
        raise ValueError("levels must be a positive integer")
    started = time.perf_counter()
    N = 2 ** levels
    actual_m = actual.detach().to(dtype=master_dtype)
    baseline_m = baseline.detach().to(dtype=master_dtype)
    displacement = actual_m - baseline_m
    disp_eligible = eligible.take(displacement)
    zeros = lambda: torch.zeros_like(disp_eligible)
    trap = {2 ** l: zeros() for l in range(levels + 1)}
    end = {2 ** l: zeros() for l in range(levels + 1)}
    mid = {2 ** l: zeros() for l in range(levels)}
    scores, records, kept = {}, [], {}
    defined = True
    for i in range(N + 1):
        alpha = i / N
        score, gradient, record = evaluate_node(score_fn, baseline_m, actual_m, displacement, alpha,
                                                model_dtype=model_dtype, eligible=eligible)
        record["node"] = i
        records.append(record)
        if on_node is not None:
            on_node(record)
        if gradient is None or record["status"] != "finite":
            defined = False
        if score is not None:
            scores[i] = score
        if gradient is None:
            continue
        if keep_gradients:
            kept[i] = gradient.detach().clone()
        for l in range(levels + 1):
            step = N // (2 ** l)
            if i % step == 0:
                m = 2 ** l
                k = i // step
                trap[m].add_(gradient, alpha=0.5 if k in (0, m) else 1.0)
                end[m].add_(gradient)
        for l in range(levels):
            step = N // (2 ** l)
            if i % step == step // 2:
                mid[2 ** l].add_(gradient)
        del gradient
    rules = {}
    path_gradients = {}
    if defined:
        for m, total in trap.items():
            average = total / m
            path_gradients[f"T{m}"] = average
            rules[f"T{m}"] = disp_eligible * average
        for m, total in end.items():
            average = total / (m + 1)
            path_gradients[f"E{m}"] = average
            rules[f"E{m}"] = disp_eligible * average
        for n, total in mid.items():
            average = total / n
            path_gradients[f"M{n}"] = average
            rules[f"M{n}"] = disp_eligible * average
        rules.update(simpson_and_romberg(rules, levels))
    gap = scores[N] - scores[0] if 0 in scores and N in scores else None
    return dict(levels=levels, intervals=N, defined=defined, rules=rules, path_gradients=path_gradients,
                displacement=disp_eligible, endpoint_scores=dict(baseline=scores.get(0), input=scores.get(N)),
                expected_gap=gap, node_records=records, node_gradients=kept,
                master_dtype=str(master_dtype), model_dtype=str(model_dtype),
                elapsed_seconds=time.perf_counter() - started)


def simpson_and_romberg(rules, levels):
    """Simpson S_2n = (T_n + 2 M_n)/3 and Richardson tables on T and on M (float64)."""
    derived = {}
    for l in range(levels):
        n = 2 ** l
        derived[f"S{2 * n}"] = ((rules[f"T{n}"].double() + 2 * rules[f"M{n}"].double()) / 3).to(rules[f"T{n}"].dtype)
    for prefix, top in (("T", levels), ("M", levels - 1)):
        previous = None
        for k in range(top + 1):
            row = [rules[f"{prefix}{2 ** k}"].double()]
            for j in range(1, k + 1):
                row.append(row[j - 1] + (row[j - 1] - previous[j - 1]) / (4 ** j - 1))
            previous = row
            derived[f"R{prefix}{2 ** k}"] = row[-1].to(rules[f"{prefix}{2 ** k}"].dtype)
    return derived


def attribution_summary(eligible_map, expected_gap):
    """Completeness accounting for one eligible-slice attribution map."""
    total = float(eligible_map.double().sum())
    if expected_gap is None:
        return dict(ig_sum=total, absolute_residual=None, relative_residual=None)
    residual = abs(total - expected_gap)
    return dict(ig_sum=total, absolute_residual=residual,
                relative_residual=residual / abs(expected_gap) if expected_gap != 0 else None)


def gauss_legendre_pass(score_fn, actual, baseline, n, *, model_dtype, eligible,
                        master_dtype=torch.float32, accumulate_dtype=torch.float64):
    """n-point Gauss-Legendre on [0, 1]; nodes never touch the endpoints.

    Node positions are formed in float64 and rounded once to the master dtype.
    Endpoint scores come from two extra score-only evaluations for completeness.
    """
    started = time.perf_counter()
    x, w = np.polynomial.legendre.leggauss(n)
    actual_m = actual.detach().to(dtype=master_dtype)
    baseline_m = baseline.detach().to(dtype=master_dtype)
    displacement = actual_m - baseline_m
    disp_eligible = eligible.take(displacement)
    accumulator = torch.zeros_like(disp_eligible, dtype=accumulate_dtype)
    records, defined = [], True
    for node, (xi, wi) in enumerate(sorted(zip(x.tolist(), w.tolist()))):
        alpha = (xi + 1.0) / 2.0
        point = (baseline_m.double() + alpha * displacement.double()).to(master_dtype)
        score, gradient, record = evaluate_point(score_fn, point, model_dtype=model_dtype, eligible=eligible)
        record.update(node=node, alpha=alpha, weight=wi / 2.0)
        records.append(record)
        if gradient is None or record["status"] != "finite":
            defined = False
            continue
        accumulator.add_(gradient.to(accumulate_dtype), alpha=wi / 2.0)
    endpoint = {}
    for name, alpha in (("baseline", 0.0), ("input", 1.0)):
        score, _, record = evaluate_node(score_fn, baseline_m, actual_m, displacement, alpha,
                                         model_dtype=model_dtype, eligible=eligible, want_gradient=False)
        endpoint[name] = score
    gap = None if None in endpoint.values() else endpoint["input"] - endpoint["baseline"]
    attribution = (disp_eligible.to(accumulate_dtype) * accumulator) if defined else None
    return dict(n=n, defined=defined, attribution=attribution,
                path_gradient=accumulator if defined else None,
                endpoint_scores=endpoint, expected_gap=gap, node_records=records,
                elapsed_seconds=time.perf_counter() - started)


def _gk21(f, a, b):
    center, half = (a + b) / 2.0, (b - a) / 2.0
    values = {}
    kronrod = gauss = None
    for index, node in enumerate(GK21_NODES):
        for sign in ((1.0, -1.0) if node else (1.0,)):
            alpha = center + sign * half * node
            value = f(alpha)
            values[alpha] = value
            term = GK21_KRONROD_WEIGHTS[index] * value
            kronrod = term if kronrod is None else kronrod + term
            if index % 2 == 1:
                g_term = GK21_GAUSS_WEIGHTS[index // 2] * value
                gauss = g_term if gauss is None else gauss + g_term
    kronrod = kronrod * half
    gauss = gauss * half
    return kronrod, float((kronrod - gauss).abs().sum()), len(values)


def adaptive_gk21(f, *, rtol, atol=0.0, max_intervals=49, snapshot_evaluations=()):
    """Adaptive vector Gauss-Kronrod (21-point) on [0, 1] with an L1 error norm.

    f(alpha) returns a float64 tensor. The worst interval is bisected until the
    summed error estimate is at most max(atol, rtol * ||integral||_1) or the
    interval cap is reached. The returned log records every refinement, and
    snapshots keep the integral estimate after the first refinement whose
    evaluation count reaches each requested level, giving several budgets
    from one run.
    """
    if rtol <= 0 or max_intervals < 1:
        raise ValueError("rtol and max_intervals must be positive")
    integral, error, evaluations = _gk21(f, 0.0, 1.0)
    intervals = {(0.0, 1.0): (integral, error)}
    heap = [(-error, 0.0, 1.0)]
    log, snapshots = [], {}
    pending = sorted(set(snapshot_evaluations))

    def state():
        total = sum(value for value, _ in intervals.values())
        err = math.fsum(e for _, e in intervals.values())
        return total, err

    total, total_error = state()
    log.append(dict(evaluations=evaluations, intervals=len(intervals), error_l1=total_error,
                    integral_l1=float(total.abs().sum())))
    while total_error > max(atol, rtol * float(total.abs().sum())) and len(intervals) < max_intervals:
        _, a, b = heapq.heappop(heap)
        del intervals[(a, b)]
        middle = (a + b) / 2.0
        for left, right in ((a, middle), (middle, b)):
            value, err, count = _gk21(f, left, right)
            evaluations += count
            intervals[(left, right)] = (value, err)
            heapq.heappush(heap, (-err, left, right))
        total, total_error = state()
        log.append(dict(evaluations=evaluations, intervals=len(intervals), error_l1=total_error,
                        integral_l1=float(total.abs().sum())))
        while pending and evaluations >= pending[0]:
            snapshots[pending.pop(0)] = dict(evaluations=evaluations, estimate=total.clone(),
                                             error_l1=total_error)
    converged = total_error <= max(atol, rtol * float(total.abs().sum()))
    return dict(integral=total, error_l1=total_error, evaluations=evaluations, intervals=len(intervals),
                converged=converged, log=log, snapshots=snapshots,
                partition=sorted([a, b] for a, b in intervals))


def adaptive_pass(score_fn, actual, baseline, *, model_dtype, eligible, rtol, max_intervals=49,
                  master_dtype=torch.float32, snapshot_evaluations=(105, 210, 420, 840)):
    """Adaptive GK21 attribution; any nonfinite node makes the cell undefined."""
    started = time.perf_counter()
    actual_m = actual.detach().to(dtype=master_dtype)
    baseline_m = baseline.detach().to(dtype=master_dtype)
    displacement = actual_m - baseline_m
    disp_eligible = eligible.take(displacement)
    records = []
    failure = {}

    def integrand(alpha):
        point = (baseline_m.double() + alpha * displacement.double()).to(master_dtype)
        _, gradient, record = evaluate_point(score_fn, point, model_dtype=model_dtype, eligible=eligible)
        record["alpha"] = alpha
        records.append(record)
        if gradient is None or record["status"] != "finite":
            failure.setdefault("first", record)
            return torch.zeros_like(disp_eligible, dtype=torch.float64)
        return gradient.to(torch.float64)

    result = adaptive_gk21(integrand, rtol=rtol, max_intervals=max_intervals,
                           snapshot_evaluations=snapshot_evaluations)
    endpoint = {}
    for name, alpha in (("baseline", 0.0), ("input", 1.0)):
        endpoint[name], _, _ = evaluate_node(score_fn, baseline_m, actual_m, displacement, alpha,
                                             model_dtype=model_dtype, eligible=eligible, want_gradient=False)
    gap = None if None in endpoint.values() else endpoint["input"] - endpoint["baseline"]
    defined = not failure
    attribution = disp_eligible.double() * result["integral"] if defined else None
    snapshots = {k: dict(evaluations=v["evaluations"], error_l1=v["error_l1"],
                         attribution=disp_eligible.double() * v["estimate"]) for k, v in result["snapshots"].items()} if defined else {}
    return dict(defined=defined, attribution=attribution, endpoint_scores=endpoint, expected_gap=gap,
                error_l1=result["error_l1"], evaluations=result["evaluations"], intervals=result["intervals"],
                converged=result["converged"], log=result["log"], partition=result["partition"],
                snapshots=snapshots, node_records=records, first_failure=failure.get("first"),
                elapsed_seconds=time.perf_counter() - started)


def historical_pass(score_fn, actual, baseline, m, *, eligible):
    """Replica of the archived bf16 kernel, with nonfinite counting added.

    Arithmetic follows the archived code: the path is rebuilt in the input
    dtype as baseline + alpha * diff (so the last node need not equal the
    input), gradients are summed in the input dtype after nan_to_num, the sum
    is divided by m + 1, and completeness uses two separate no-grad forwards.
    A node whose forward raises (the revised sampler refuses nonfinite values)
    contributes a zero gradient, which is what the historical NaN gradient
    became after nan_to_num; it is counted as a failed node.
    """
    started = time.perf_counter()
    diff = actual - baseline
    sum_gradients = torch.zeros_like(actual)
    records = []
    totals = dict(nan=0, posinf=0, neginf=0, failed_forwards=0, eligible_nan=0, eligible_posinf=0,
                  eligible_neginf=0)
    for k in range(m + 1):
        alpha = k / m
        record = dict(node=k, alpha=alpha)
        interpolated = (baseline + alpha * diff).detach().requires_grad_(True)
        with torch.enable_grad():
            try:
                output = score_fn(interpolated)
            except NonFiniteAttributionError as exc:
                record.update(status="forward_nonfinite", stage=exc.diagnostics.get("stage"))
                totals["failed_forwards"] += 1
                records.append(record)
                continue
            grad = torch.autograd.grad(output, interpolated)[0]
        record["score"] = float(output.detach())
        full = nonfinite_counts(grad)
        selected = nonfinite_counts(eligible.take(grad))
        record["gradient_nonfinite_full"] = full
        record["gradient_nonfinite_eligible"] = selected
        for key in ("nan", "posinf", "neginf"):
            totals[key] += full[key]
            totals["eligible_" + key] += selected[key]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        sum_gradients += grad
        record["status"] = "finite" if not total_nonfinite(full) else "nonfinite_zeroed"
        records.append(record)
        del interpolated, output, grad
    avg_gradients = sum_gradients / (m + 1)
    attributions = diff * avg_gradients
    rebuilt = baseline + 1.0 * diff
    rebuild_error = float((rebuilt.float() - actual.float()).abs().max())
    endpoint = {}
    for name, tensor in (("input", actual), ("baseline", baseline)):
        try:
            with torch.no_grad():
                endpoint[name] = float(score_fn(tensor))
        except NonFiniteAttributionError:
            endpoint[name] = None
    gap = None if None in endpoint.values() else endpoint["input"] - endpoint["baseline"]
    stored_sum = float(attributions.sum())  # input-dtype reduction, as the historical .sum().item()
    fp32_sum = float(attributions.float().sum())
    residual = None if gap is None else abs(stored_sum - gap)
    return dict(m=m, attribution=eligible.take(attributions), ig_sum_input_dtype=stored_sum,
                ig_sum_fp32=fp32_sum, endpoint_scores=endpoint, expected_gap=gap,
                absolute_residual=residual,
                relative_residual=None if gap in (None, 0) else residual / abs(gap),
                endpoint_rebuild_max_abs_error=rebuild_error, nonfinite_totals=totals,
                evaluations=m + 1, node_records=records, input_dtype=str(actual.dtype),
                elapsed_seconds=time.perf_counter() - started)


def rank_order(scores, indices):
    """Decreasing score, then increasing eligible index: the repository tie rule."""
    order = torch.argsort(scores, descending=True, stable=True).cpu().tolist()
    return [indices[i] for i in order]


def average_ranks(values):
    """1-based average ranks with ties sharing their mean rank (float64)."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    i = 0
    while i < len(array):
        j = i
        while j + 1 < len(array) and array[order[j + 1]] == array[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman correlation with average ranks; None when either side is constant."""
    ra, rb = average_ranks(a), average_ranks(b)
    if np.all(ra == ra[0]) or np.all(rb == rb[0]):
        return None
    ra -= ra.mean()
    rb -= rb.mean()
    return float(np.dot(ra, rb) / math.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))


def compare_maps(reference, candidate, eligible, *, counts):
    """Coordinate- and ranking-level differences of candidate against reference."""
    ref, cand = reference.double(), candidate.double()
    l1_ref = float(ref.abs().sum())
    difference = cand - ref
    scores_ref = eligible.group_scores(ref)
    scores_cand = eligible.group_scores(cand)
    order_ref = rank_order(scores_ref, eligible.indices)
    order_cand = rank_order(scores_cand, eligible.indices)
    top = max(1, math.ceil(0.05 * len(eligible.indices)))
    overlap = len(set(order_ref[:top]) & set(order_cand[:top])) / top
    jaccard = []
    for count in counts:
        if count in (0, len(order_ref)):
            continue
        a, b = set(order_ref[:count]), set(order_cand[:count])
        jaccard.append(dict(count=count, jaccard=len(a & b) / len(a | b)))
    return dict(relative_l1=float(difference.abs().sum()) / l1_ref if l1_ref else None,
                l1=float(difference.abs().sum()), max_abs=float(difference.abs().max()),
                signed_sum_difference=float(cand.sum() - ref.sum()),
                group_spearman=spearman(scores_ref.cpu().numpy(), scores_cand.cpu().numpy()),
                top1_equal=order_ref[0] == order_cand[0], top5pct_overlap=overlap,
                top5pct_size=top, deleted_set_jaccard=jaccard)
