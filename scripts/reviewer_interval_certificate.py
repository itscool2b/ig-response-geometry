"""Narrow, post hoc replay contract for optional BCa and t sensitivity endpoints.

This is not a registered inference criterion or a universal SciPy guarantee.
Both historical and replay endpoints must fit an independently propagated
operation envelope. Current special-function calls are checked against an
outward-rounded reference with explicit relative accuracy contracts: 16u for
normal CDF, 8u for inverse CDFs and floating powers. These budgets are post hoc.
Unpreserved historical intermediates cannot be retrospectively certified.
Primary estimates, percentile intervals, populations and decisions stay exact.
"""
from collections import defaultdict
from decimal import Decimal
from functools import lru_cache
import hashlib
import math

import mpmath as mp
import numpy as np
from scipy.stats import norm, t as student_t


mp.mp.dps = 90
mp.iv.dps = 90
PRECISION = 90
ACCURACY_UNITS = 8
CDF_ACCURACY_UNITS = 16


def iv(a, b=None):
    return mp.iv.mpf([a, a if b is None else b])


def edges(x):
    return mp.mpf(x.a), mp.mpf(x.b)


def magnitude(x):
    return max(map(abs, edges(x)))


def gamma(n):
    nu = iv(n) / 2**53
    return nu / (1 - nu)


def widen(x, error):
    # Error construction AND endpoint widening use directed interval arithmetic.
    upper = edges(error if hasattr(error, "a") else iv(error))[1]
    return x + iv(-upper, upper)


def rounded(x, operations=1):
    return widen(x, gamma(operations) * iv(magnitude(x)))


def reduction(values):
    return widen(sum(values, iv(0)), gamma(len(values)) * sum((iv(magnitude(x)) for x in values), iv(0)))


def contains(record, value):
    lo, hi = map(mp.mpf, record["enclosure"])
    return math.isfinite(float(value)) and lo <= mp.mpf(value) <= hi


def exact_decimal(value):
    """Serialize a pinned mpmath1.3 binary endpoint without inward rounding.

    The _mpf_ tuple is (sign, integer mantissa, binary exponent, bit count).
    Decimal tuple construction is exact and ignores the arithmetic context.
    """
    value = mp.mpf(value)
    if not mp.isfinite(value):
        raise ValueError("nonfinite certificate endpoint")
    sign, mantissa, exponent, _ = value._mpf_
    if exponent < 0:
        digits = str(mantissa * 5**(-exponent))
        decimal_exponent = exponent
    else:
        digits = str(mantissa << exponent)
        decimal_exponent = 0
    return str(Decimal((sign, tuple(map(int, digits)), decimal_exponent)))


def packed(x, **extra):
    return dict(enclosure=[exact_decimal(v) for v in edges(x)], **extra)


@lru_cache(maxsize=16384)
def phi_point(x):
    """Normal CDF via its integral series and alternating next-term remainder."""
    x = mp.mpf(x)
    if x < 0:
        return 1 - phi_point(-x)
    if x == 0:
        return iv(.5)
    if x > 8:
        raise ValueError("normal argument outside certified series domain")
    xx, total = iv(x), iv(0)
    term = xx
    for k in range(1000):
        total = total + term if k % 2 == 0 else total - term
        following = term * xx * xx * (2*k+1) / (2*(k+1)*(2*k+3))
        if magnitude(following) < mp.mpf("1e-70") and edges(xx*xx/(2*(k+1)))[1] < 1:
            remainder = iv(0, edges(following)[1])
            total = total - remainder if k % 2 == 0 else total + remainder
            return iv(.5) + total / mp.iv.sqrt(2*mp.iv.pi)
        term = following
    raise ValueError("normal series did not converge")


def monotone(function, x):
    lo, hi = edges(x)
    return iv(edges(function(lo))[0], edges(function(hi))[1])


def inverse(function, p, lo, hi):
    p = mp.mpf(p)
    if not edges(function(lo))[1] < p < edges(function(hi))[0]:
        raise ValueError("quantile outside certified inversion domain")
    for _ in range(220):
        # Dyadic bracket endpoints and midpoint are exact at this precision.
        mid = (lo + hi) / 2
        a, b = edges(function(mid))
        if b < p:
            lo = mid
        elif a > p:
            hi = mid
        elif hi-lo < mp.mpf("1e-40"):
            break
        else:
            raise ValueError("quantile inversion comparison unresolved")
        if hi-lo < mp.mpf("1e-40"):
            return iv(lo, hi)
    else:
        raise ValueError("quantile inversion did not converge")
    return iv(lo, hi)


@lru_cache(maxsize=512)
def normal_quantile(p):
    if p == .5:
        return iv(0)
    return inverse(phi_point, p, mp.mpf(-8), mp.mpf(8))


def t_cdf_point(x, degrees):
    """Integer-df CDF from the finite integral recurrence for cos^(df-1)."""
    if not isinstance(degrees, int) or not 1 <= degrees <= 1000:
        raise ValueError("unsupported t degrees of freedom")
    x = iv(x)
    root = mp.iv.sqrt(degrees + x*x)
    sine, cosine = x/root, mp.iv.sqrt(iv(degrees))/root
    integrals = [mp.iv.atan2(x, mp.iv.sqrt(iv(degrees))), sine]
    for k in range(2, degrees):
        integrals.append(sine * cosine**(k-1)/k + iv(k-1)/k * integrals[k-2])
    if degrees % 2:
        m = (degrees-1)//2
        factor = iv(4**m * math.factorial(m)**2) / (mp.iv.pi * math.factorial(2*m))
    else:
        m = degrees//2
        factor = iv(math.factorial(2*m)) / (4**m * math.factorial(m)*math.factorial(m-1))
    return iv(.5) + factor*integrals[degrees-1]


@lru_cache(maxsize=256)
def t_quantile(p, degrees):
    if p == .5:
        return iv(0)
    return inverse(lambda x: t_cdf_point(x, degrees), p, mp.mpf(0), mp.mpf(16384))


def contract(reference, actual, label, checks, units=ACCURACY_UNITS):
    """Checked reproduction contract, not an assumed library-wide error theorem."""
    enclosure = widen(reference, iv(units)/2**53 * iv(magnitude(reference)))
    if not contains(packed(enclosure), float(actual)):
        raise ValueError(label + " violates its checked accuracy contract")
    lo, hi = edges(reference)
    error = abs(iv(float(actual))-reference)
    relative = error/abs(reference)*2**53 if not lo <= 0 <= hi else iv(0) if float(actual) == lo == hi == 0 else None
    checks.append(dict(operation=label, actual=float(actual), reference=packed(reference),
                       accuracy_units=units, relative_error_units_upper=None if relative is None else exact_decimal(edges(relative)[1])))
    return enclosure


def envelope(reference, units=ACCURACY_UNITS):
    return widen(reference, iv(units)/2**53 * iv(magnitude(reference)))


def quantile_enclosure(samples, probability):
    ordered = np.sort(samples)
    n = len(ordered)
    lo, hi = edges(probability)
    if lo < 0 or hi > 1:
        raise ValueError("adjusted probability outside [0,1]")
    index = rounded(probability*(n-1))
    left, right = edges(index)
    left, right = max(mp.mpf(0), left), min(mp.mpf(n-1), right)
    # Include all crossed interpolation bins. Monotonic interpolation's extrema
    # are at the endpoints, but arithmetic scale can peak at an interior bin.
    first, last = min(n-2, int(mp.floor(left))), min(n-2, int(mp.floor(right)))
    if last-first > 4:
        raise ValueError("unstable quantile interpolation region")
    bounds = []
    for j in range(first, last+1):
        fraction = iv(max(left, mp.mpf(j))-j, min(right, mp.mpf(j+1))-j)
        a, b = float(ordered[j]), float(ordered[j+1])
        result = iv(a) + fraction*(iv(b)-iv(a))
        # NumPy's two-sided lerp uses subtraction, product, addition, plus
        # (1-fraction) on the upper branch; five operations cover both branches.
        bounds.append(widen(result, gamma(5)*(iv(abs(a))+iv(abs(b)))))
    return iv(min(edges(x)[0] for x in bounds), max(edges(x)[1] for x in bounds)), [first, last]


def normal_domain(values):
    """Conservative supported range keeps cubes and reductions away from underflow.

    This verifier rejects exceptional arithmetic rather than extending a
    relative-error model into subnormal or overflowing intermediate results.
    """
    values = np.asarray(values, dtype=float)
    nonzero = np.abs(values[values != 0])
    if not np.isfinite(values).all() or nonzero.size and (nonzero.min() < 1e-90 or nonzero.max() > 1e90):
        raise ValueError("sensitivity arithmetic outside supported normal range")
    return dict(nonzero_values=int(nonzero.size), minimum_absolute_nonzero=float(nonzero.min()) if nonzero.size else None,
                maximum_absolute=float(nonzero.max()) if nonzero.size else 0.)


def enclose(values, samples, alpha, kinds=("bca", "t"), authenticated_mean=None):
    """Derive envelopes from the exact binary64 episode and bootstrap vectors."""
    if mp.__version__ != "1.3.0" or mp.mp.dps != PRECISION or mp.iv.dps != PRECISION:
        raise ValueError("certificate runtime or interval precision changed")
    values, samples = np.asarray(values, dtype=float), np.asarray(samples, dtype=float)
    if values.ndim != 1 or samples.ndim != 1 or len(values) < 2 or len(samples) < 2 or \
            not np.isfinite(values).all() or not np.isfinite(samples).all() or not 0 < alpha < 1:
        raise ValueError("nonfinite or insufficient sensitivity input")
    n, count = len(values), len(samples)
    census = dict(values=normal_domain(values), bootstrap_samples=normal_domain(samples))
    v = [iv(float(x)) for x in values]
    total = reduction(v)
    mean = rounded(total/n)
    checked = []
    result = dict(episodes=n, draws=count, alpha=alpha,
                  values_sha256=hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
                  samples_sha256=hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
                  runtime=dict(mpmath=mp.__version__, decimal_precision=PRECISION), checked_operations=checked,
                  normal_arithmetic_census=census)
    if "bca" in kinds:
        lo, hi = edges(mean)
        if authenticated_mean is not None:
            if float(values.mean()) != authenticated_mean:
                raise ValueError("authenticated historical mean differs from reconstruction")
            # The historical theta was preserved as the primary gap mean or
            # contrast estimate. Equality of that field is an exact fallback
            # for bootstrap ties, not an assumed cross-platform reduction.
            lo = hi = mp.mpf(authenticated_mean)
        low_count = sum(mp.mpf(float(x)) < lo for x in samples)
        high_count = sum(mp.mpf(float(x)) < hi for x in samples)
        if low_count != high_count:
            raise ValueError("bootstrap ties make bias count sensitive to reduction rounding")
        probability = min(max(low_count/count, 1/(2*count)), 1-1/(2*count))
        if low_count in (0, count):
            raise ValueError("degenerate BCa bias probability")
        z0 = contract(normal_quantile(probability), norm.ppf(probability), "normal_ppf_bias", checked)
        jack = [rounded((total-x)/(n-1), 2) for x in v]
        mean_j = rounded(reduction(jack)/n)
        centred = [rounded(mean_j-x) for x in jack]
        squares = reduction([rounded(x*x) for x in centred])
        if edges(squares)[0] <= 0:
            raise ValueError("jackknife variance not separated from zero")
        actual_jack = (values.sum()-values)/(n-1)
        actual_c = actual_jack.mean()-actual_jack
        census["jackknife_centred"] = normal_domain(actual_c)
        actual_cubes = actual_c**3
        for value, cube in zip(actual_c, actual_cubes):
            contract(iv(float(value))**3, cube, "numpy_cube_power", checked)
        cubes = reduction([envelope(x*x*x) for x in centred])
        actual_squares = np.sum(actual_c**2)
        actual_power = actual_squares**1.5
        power_ref = iv(float(actual_squares))*mp.iv.sqrt(iv(float(actual_squares)))
        contract(power_ref, actual_power, "scalar_power_3_over_2", checked)
        power = envelope(squares*mp.iv.sqrt(squares))
        acceleration = rounded(cubes/rounded(6*power))
        actual_a = float(np.sum(actual_cubes)/(6*actual_power))
        actual_z0 = float(norm.ppf(probability))
        endpoints = []
        for tail in (float(alpha/2), float(1-alpha/2)):
            actual_z = float(norm.ppf(tail))
            z = contract(normal_quantile(tail), actual_z, "normal_ppf_tail", checked)
            sum_z = rounded(z0+z)
            divisor = rounded(1-rounded(acceleration*sum_z))
            if edges(divisor)[0] <= 0 <= edges(divisor)[1]:
                raise ValueError("BCa denominator crosses zero")
            argument = rounded(z0+rounded(sum_z/divisor))
            actual_arg = actual_z0+(actual_z0+actual_z)/(1-actual_a*(actual_z0+actual_z))
            contract(phi_point(mp.mpf(actual_arg)), norm.cdf(actual_arg), "normal_cdf_adjustment", checked, CDF_ACCURACY_UNITS)
            probability_box = envelope(monotone(phi_point, argument), CDF_ACCURACY_UNITS)
            endpoint, bins = quantile_enclosure(samples, probability_box)
            endpoints.append(packed(endpoint, quantile_bins=bins, probability=packed(probability_box)))
        result.update(bca=endpoints, bias_count=low_count, acceleration=packed(acceleration),
                      bias_mean_bound_exactly=authenticated_mean is not None,
                      samples_equal_mean=int(np.count_nonzero(samples == values.mean())))
    if "t" in kinds:
        census["t_centred"] = normal_domain(values-values.mean())
        centred = [rounded(x-mean) for x in v]
        squares = reduction([rounded(x*x) for x in centred])
        if edges(squares)[0] <= 0:
            raise ValueError("sample variance not separated from zero")
        sd = rounded(mp.iv.sqrt(rounded(squares/(n-1))))
        standard_error = rounded(sd/rounded(mp.iv.sqrt(iv(n))))
        p = float(1-alpha/2)
        critical = contract(t_quantile(p, n-1), student_t.ppf(p, n-1), "student_t_ppf", checked)
        half = rounded(critical*standard_error)
        result["t"] = [packed(rounded(mean-half)), packed(rounded(mean+half))]
    return result


def _area(source, name, direction, response, metric):
    item = source["rankings"].get(name)
    if item is None or item["status"] != "defined":
        return None
    outcome = item["curves"][direction]["responses"][response]
    if metric != "normalized_auc_round":
        return outcome.get(metric)
    values = outcome["values"]
    if source["count_rule"] != "max1_round" or any(x is None for x in values):
        return None
    n = len(source["eligible_indices"])
    counts = [round(n*k/100) for k in source["nominal_grid_percent"]]
    curve = []
    for count, stored, value in zip(counts, source["selected_counts"], values):
        if count != stored and count != 0:
            raise ValueError("unsupported count-rule change")
        curve.append(value if count == stored else values[0])
    table = source["response_table"]
    initial, baseline = (table[source[k]][response] for k in ("input_response_id", "baseline_response_id"))
    if initial is None or baseline is None or abs(initial-baseline) <= source["denominator_min"].get(response, 0):
        return None
    axis = np.asarray([100*c/n for c in counts], dtype=np.float64)/100.
    normalized = (np.asarray(curve, dtype=np.float64)-baseline)/(initial-baseline)
    return float(np.trapezoid(normalized, axis))


def _advantage(source, name, direction, response, metric):
    value = _area(source, name, direction, response, metric)
    randoms = ["random_exact"] if "random_exact" in source["rankings"] else \
              sorted(n for n in source["rankings"] if n.startswith("random_"))
    controls = [_area(source, n, direction, response, metric) for n in randoms]
    if value is None or not controls or any(x is None for x in controls):
        return None
    reference = math.fsum(controls)/len(controls)
    return reference-value if direction == "deletion" else value-reference


def _resample(matrix, draws, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    pieces = []
    for begin in range(0, draws, 1000):
        indices = rng.integers(0, len(matrix), size=(min(begin+1000, draws)-begin, len(matrix)))
        pieces.append(matrix[indices].mean(axis=1))
    return np.concatenate(pieces)


class Certificates:
    """Reconstruct changed optional intervals from manifest-authenticated raw scores.

    read(name) is the caller's strict, manifest-bound byte reader. Only P, B1
    and S's registered decision/contrast records have a contract; other output
    categories remain exact. Caches contain immutable authenticated JSON data.
    """
    def __init__(self, read, parse_json, old_manifest):
        self.read, self.json = read, parse_json
        self.config = self.json(read("reproduction.json"))
        self.plan = self.json(read(self.config["plan"]))
        if hashlib.sha256(read(self.config["plan"])).hexdigest() != old_manifest["plan_sha256"]:
            raise ValueError("certificate plan differs from frozen analysis")
        self.specs = {b["id"]: b for b in old_manifest["inputs"]["banks"]}
        self.banks = {}
        self.records = []

    def contexts(self, bank):
        if bank in self.banks:
            return self.banks[bank]
        if bank not in ("P", "B1", "S"):
            raise ValueError("unregistered sensitivity bank")
        spec = self.specs[bank]
        bank_blob = self.read(self.config["input_paths"][spec["bank"]])
        protocol_blob = self.read(self.config["input_paths"][spec["protocol"]])
        bank_hash = hashlib.sha256(bank_blob).hexdigest()
        protocol_hash = hashlib.sha256(protocol_blob).hexdigest()
        if self.json(protocol_blob)["banks"][bank]["bank_sha256"] != bank_hash:
            raise ValueError("sensitivity bank/protocol mismatch")
        output = []
        for entry in self.json(bank_blob)["contexts"]:
            if entry["status"] != "available":
                output.append((entry, None))
                continue
            key = f'e{entry["episode"]:04d}-c{entry["policy_call_idx"]:03d}'
            prefix = f"runs/{bank}/{key}/scoring/"
            completion = self.json(self.read(prefix+"completion.json"))
            raw = self.read(prefix+"record.json")
            binding = completion["binding"]
            if completion["status"] != "complete" or completion["files_sha256"]["record.json"] != hashlib.sha256(raw).hexdigest() or \
                    binding["bank_id"] != bank or binding["bank_sha256"] != bank_hash or \
                    binding["protocol_sha256"] != protocol_hash or binding["context_id"] != entry["context_id"]:
                raise ValueError("scoring completion binding mismatch")
            output.append((entry, self.json(raw)))
        self.banks[bank] = output
        return output

    def _vectors(self, name, path, document):
        parts = name.split("/")
        if len(parts) != 3 or parts[0] != "methods" or parts[1] not in ("P", "B1", "S"):
            raise ValueError("unsupported interval location")
        contexts = self.contexts(parts[1])
        boot = self.plan["methods_study"]["bootstrap"]
        if parts[2] in ("decisions.json", "decisions_round_rule.json", "decisions_nominal_axis.json"):
            modality, _, cell, _, _, _ = path
            if path[1] != "cells" or path[3:] != ("advantage", "best_minus_second", "adjusted"):
                raise ValueError("unsupported decision sensitivity")
            summary = document[modality]["cells"][cell]["advantage"]
            metric = document[modality]["metric"]
            response, direction = cell.split("/")
            names, episodes = summary["names"], summary["episode_ids"]
            if summary["best"] is None or summary["tie"] is not None:
                raise ValueError("tied best method has no interval exception")
            rows = defaultdict(list)
            for entry, score in contexts:
                if entry["episode_id"] in episodes:
                    if score is None:
                        raise ValueError("retained episode has unavailable call")
                    rows[entry["episode_id"]].append([_advantage(score["modalities"][modality], n, direction, response, metric) for n in names])
            matrix = np.array([[math.fsum(call[j] for call in rows[e])/len(rows[e]) for j in range(len(names))] for e in episodes])
            if not np.isfinite(matrix).all() or {n: float(matrix.mean(axis=0)[j]) for j,n in enumerate(names)} != summary["mean"]:
                raise ValueError("reconstructed method means differ from primary record")
            means = matrix.mean(axis=0)
            order = sorted(range(len(names)), key=lambda j: -means[j])
            if names[order[0]] != summary["best"] or names[order[1]] != summary["second"] or means[order[0]] == means[order[1]]:
                raise ValueError("reconstructed best/second method changed")
            samples = _resample(matrix, boot["draws"], boot["seed"])
            i, j = order[:2]
            values, draws = matrix[:, i]-matrix[:, j], samples[:, i]-samples[:, j]
            target = summary["best_minus_second"]["adjusted"]
            estimate = summary["best_minus_second"]["mean"]
        elif parts[2] == "contrasts.json":
            metric, index = path
            target = document[metric][index]
            contrast = self.plan["methods_study"]["primary_family"][index]
            if metric not in ("normalized_auc", "normalized_auc_round") or target["metric"] != metric or \
                    any(target[k] != v for k,v in contrast.items() if k != "metric"):
                raise ValueError("contrast identity differs from registered plan")
            planned = list(dict.fromkeys(e["episode_id"] for e,s in contexts))
            episodes = [e for e in planned if e not in target["excluded_episodes"]]
            if contrast["contrast"] == "matched_calibrated":
                terms = [(contrast["method"], contrast["method_response"], 1), (contrast["control"], contrast["control_response"], -1)]
            elif contrast["contrast"] == "method_minus_method":
                terms = [(contrast["method"], contrast["response"], 1), (contrast["control"], contrast["response"], -1)]
            else:
                terms = [(contrast["method"], contrast["response"], 1)]
            rows = defaultdict(list)
            for entry, score in contexts:
                if entry["episode_id"] in episodes:
                    if score is None:
                        raise ValueError("retained contrast call missing")
                    total = 0.
                    for method, response, sign in terms:
                        value = _advantage(score["modalities"]["vision"], method, contrast["direction"], response, metric)
                        if value is None:
                            raise ValueError("undefined retained contrast")
                        total += sign*value
                    rows[entry["episode_id"]].append(total)
            values = np.array([math.fsum(rows[e])/len(rows[e]) for e in episodes])
            matrix = values[:, None]
            samples = _resample(matrix, boot["draws"], boot["seed"])
            draws = samples[:, 0]
            estimate = target["estimate"]
        else:
            raise ValueError("unsupported sensitivity output")
        if len(values) != target["episodes"] or float(values.mean()) != estimate or \
                np.quantile(draws, [target["alpha"]/2, 1-target["alpha"]/2]).tolist() != target["interval"]:
            raise ValueError("reconstruction differs from exact primary estimate or percentile interval")
        provenance = dict(matrix_sha256=hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest(),
                          full_bootstrap_sha256=hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest(),
                          episode_ids=episodes, bootstrap=dict(boot), quantile_method="linear")
        return values, draws, target, estimate, provenance

    def for_file(self, name, expected, actual):
        """Return only exact JSON paths to independently enclosed changed endpoints."""
        found = []
        def visit(a, b, path=()):
            if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
                for k in a:
                    visit(a[k], b[k], path+(k,))
            elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
                for k, (x,y) in enumerate(zip(a,b)):
                    visit(x,y,path+(k,))
            elif type(a) is float and type(b) is float and a != b and len(path) >= 4 and \
                    path[-4] == "sensitivity" and path[-3] in ("bca", "t") and path[-2] == "interval" and path[-1] in (0,1):
                found.append(path)
        visit(expected, actual)
        grouped = defaultdict(set)
        for path in found:
            grouped[path[:-4]].add(path[-3])
        allowed = {}
        for base, kinds in grouped.items():
            values, draws, target, estimate, provenance = self._vectors(name, base, expected)
            proof = enclose(values, draws, target["alpha"], kinds=tuple(sorted(kinds)), authenticated_mean=estimate)
            proof.update(file=name, location=list(base), provenance=provenance)
            other = actual
            for key in base:
                other = other[key]
            for kind in kinds:
                for endpoint in (0,1):
                    if not contains(proof[kind][endpoint], target["sensitivity"][kind]["interval"][endpoint]) or \
                            not contains(proof[kind][endpoint], other["sensitivity"][kind]["interval"][endpoint]):
                        raise ValueError("optional endpoint outside independent operation envelope")
            for kind in kinds:
                for endpoint in (0,1):
                    path = base+("sensitivity",kind,"interval",endpoint)
                    if path in found:
                        allowed["/"+"/".join(map(str,path))] = proof[kind][endpoint]
            self.records.append(proof)
        return allowed
