import importlib.machinery
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import ig_quadrature as q
from integrated_gradients import NonFiniteAttributionError, integrated_gradients

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_KERNEL = ROOT / "docs/tmlr-readiness/archives/active-originals/2026-10-01-stable-base/integrated_gradients.py.historical"


def toy_problem(dtype=torch.float32, groups=6, hidden=4, eligible=(1, 2, 4), seed=0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(hidden, 3, generator=generator, dtype=torch.float64)
    vector = torch.randn(3, generator=generator, dtype=torch.float64)
    actual = torch.randn(1, groups, hidden, generator=generator, dtype=torch.float64)
    baseline = actual.clone()
    for g in eligible:
        baseline[0, g] = torch.randn(hidden, generator=generator, dtype=torch.float64) * 0.3

    def score(x):
        out = torch.tanh(x.to(torch.float64) @ weight) * vector
        return out.sum().to(torch.float32)

    return actual.to(dtype), baseline.to(dtype), q.Eligible(list(eligible), 1), score


def test_gk21_constants_have_the_documented_exactness():
    nodes = np.array(q.GK21_NODES)
    full_nodes = np.concatenate([-nodes[:-1], nodes[::-1]])
    kronrod = np.array(q.GK21_KRONROD_WEIGHTS)
    full_kronrod = np.concatenate([kronrod[:-1], kronrod[::-1]])
    gauss_nodes = nodes[1::2]
    gauss = np.array(q.GK21_GAUSS_WEIGHTS)
    full_gauss_nodes = np.concatenate([-gauss_nodes, gauss_nodes[::-1]])
    full_gauss = np.concatenate([gauss, gauss[::-1]])
    for degree in range(32):
        exact = 0.0 if degree % 2 else 2.0 / (degree + 1)
        assert np.dot(full_kronrod, full_nodes ** degree) == pytest.approx(exact, abs=1e-13)
        if degree <= 19:
            assert np.dot(full_gauss, full_gauss_nodes ** degree) == pytest.approx(exact, abs=1e-13)
    assert np.dot(full_kronrod, full_nodes ** 32) != pytest.approx(2.0 / 33, abs=1e-12)


def test_adaptive_gk21_matches_scipy_quad_vec_on_a_vector_integrand():
    from scipy.integrate import quad_vec

    def f(alpha):
        return torch.tensor([math.sin(30 * alpha), math.exp(alpha), 1 / (1 + 400 * (alpha - 0.3) ** 2)],
                            dtype=torch.float64)

    result = q.adaptive_gk21(f, rtol=1e-11, max_intervals=200)
    reference, _ = quad_vec(lambda a: f(a).numpy(), 0, 1, epsrel=1e-12, norm="max")
    assert result["converged"]
    np.testing.assert_allclose(result["integral"].numpy(), reference, rtol=1e-10)
    assert result["log"][-1]["evaluations"] == result["evaluations"]


@pytest.mark.parametrize("m", [2, 4, 8, 16])
@pytest.mark.parametrize("rule,prefix", [("trapezoid", "T"), ("legacy_endpoint_average", "E")])
def test_dyadic_rules_equal_the_pinned_kernel_bit_for_bit(m, rule, prefix):
    actual, baseline, eligible, score = toy_problem(dtype=torch.bfloat16)
    levels = int(math.log2(16))
    nested = q.dyadic_pass(score, actual, baseline, levels, model_dtype=torch.bfloat16, eligible=eligible)
    kernel = integrated_gradients(score, actual, baseline, m=m, quadrature=rule, return_result=True)
    assert nested["defined"]
    assert torch.equal(nested["rules"][f"{prefix}{m}"], eligible.take(kernel.attributions))
    outside = [i for i in range(actual.shape[1]) if i not in eligible.indices]
    assert torch.count_nonzero(kernel.attributions[:, outside]) == 0
    assert nested["expected_gap"] == pytest.approx(kernel.expected_gap, abs=0)


def test_nested_rule_identities_hold_to_rounding():
    actual, baseline, eligible, score = toy_problem(dtype=torch.float64)
    out = q.dyadic_pass(score, actual, baseline, 5, model_dtype=torch.float64, eligible=eligible,
                        master_dtype=torch.float64, keep_gradients=True)
    r, N = out["rules"], 32
    f0, f1 = out["node_gradients"][0], out["node_gradients"][N]
    disp = out["displacement"]
    close = dict(rtol=1e-12, atol=1e-14)
    for n in (1, 2, 4, 8, 16):
        torch.testing.assert_close(r[f"T{2 * n}"], (r[f"T{n}"] + r[f"M{n}"]) / 2, **close)
        torch.testing.assert_close(r[f"S{2 * n}"], (r[f"T{n}"] + 2 * r[f"M{n}"]) / 3, **close)
    for m in (1, 2, 4, 8, 16, 32):
        identity = r[f"T{m}"] + (disp * (f0 + f1) / 2 - r[f"T{m}"]) / (m + 1)
        torch.testing.assert_close(r[f"E{m}"], identity, **close)
    # The second Romberg column on trapezoids is Simpson's rule.
    torch.testing.assert_close(r["RT2"], r["S2"], **close)
    torch.testing.assert_close(r["RM2"], (4 * r["M2"] - r["M1"]) / 3, **close)


def one_dimensional():
    actual = torch.ones(1, 1, 1, dtype=torch.float64)
    baseline = torch.zeros(1, 1, 1, dtype=torch.float64)
    return actual, baseline, q.Eligible([0], 1)


@pytest.mark.parametrize("degree,simpson_exact,trap_exact", [(1, True, True), (3, True, False), (5, False, False)])
def test_polynomial_exactness_of_trapezoid_simpson_and_gauss(degree, simpson_exact, trap_exact):
    score = lambda x: (x.to(torch.float64) ** (degree + 1)).sum() / (degree + 1)
    actual, baseline, eligible = one_dimensional()
    exact = 1.0 / (degree + 1)  # integral of alpha**degree
    out = q.dyadic_pass(score, actual, baseline, 3, model_dtype=torch.float64, eligible=eligible,
                        master_dtype=torch.float64)
    assert (abs(float(out["rules"]["T8"]) - exact) < 1e-14) is trap_exact
    assert (abs(float(out["rules"]["S8"]) - exact) < 1e-14) is simpson_exact
    gl = q.gauss_legendre_pass(score, actual, baseline, 3, model_dtype=torch.float64, eligible=eligible,
                               master_dtype=torch.float64)
    assert float(gl["attribution"]) == pytest.approx(exact, abs=1e-14)  # 3 nodes: exact through degree 5


def test_reviewer_endpoint_example_reproduces_and_open_rules_avoid_the_jump():
    eps = 1e-12
    score = lambda x: -torch.sqrt((1 - x.to(torch.float64)) ** 2 + eps).sum()
    actual, baseline, eligible = one_dimensional()
    out = q.dyadic_pass(score, actual, baseline, 9, model_dtype=torch.float64, eligible=eligible,
                        master_dtype=torch.float64)
    exact = math.sqrt(1 + eps) - math.sqrt(eps)
    assert float(out["rules"]["E64"]) == pytest.approx(0.984615, abs=5e-7)
    assert float(out["rules"]["T64"]) == pytest.approx(0.992187, abs=5e-7)
    for n in (16, 64, 256):
        assert float(out["rules"][f"M{n}"]) == pytest.approx(exact, abs=2e-6)
    for m in (16, 64, 256):
        relative = (exact - float(out["rules"][f"T{m}"])) / exact
        assert relative == pytest.approx(1 / (2 * m), rel=1e-3)
    gl = q.gauss_legendre_pass(score, actual, baseline, 64, model_dtype=torch.float64, eligible=eligible,
                               master_dtype=torch.float64)
    assert float(gl["attribution"]) == pytest.approx(exact, abs=1e-6)


def aliasing_score(K=128):
    def score(z):
        z = z.to(torch.float64)
        z1, z2 = z[0, 0, 0], z[0, 1, 0]
        a = 0.4 * z1 + 0.6 * z2 + (z1 - z2) * 0.4 * torch.sin(math.pi * K * (z1 + z2) / 2) ** 2
        return -(a - 1) ** 2 / 2
    return score


def test_nested_grid_aliasing_fools_dyadic_rules_but_not_gauss_legendre():
    K = 128
    score = aliasing_score(K)
    actual = torch.ones(1, 2, 1, dtype=torch.float64)
    baseline = torch.zeros(1, 2, 1, dtype=torch.float64)
    eligible = q.Eligible([0, 1], 1)
    out = q.dyadic_pass(score, actual, baseline, 8, model_dtype=torch.float64, eligible=eligible,
                        master_dtype=torch.float64)
    for m in (32, 64, 128):  # every grid whose size divides K sees h = 0
        np.testing.assert_allclose(out["rules"][f"T{m}"].flatten().numpy(), [0.2, 0.3], atol=1e-12)
        np.testing.assert_allclose(out["rules"][f"RT{m}"].flatten().numpy(), [0.2, 0.3], atol=1e-9)
    np.testing.assert_allclose(out["rules"]["T256"].flatten().numpy(), [0.3, 0.2], atol=1e-12)
    shift = K / (5 * (2 * K + 1))  # appendix equation for the endpoint-inclusive average on the 2K grid
    np.testing.assert_allclose(out["rules"]["E256"].flatten().numpy(), [0.2 + shift, 0.3 - shift], atol=1e-12)
    for n in (32, 64, 128):  # off-grid nodes keep the exact order but have not converged
        gl = q.gauss_legendre_pass(score, actual, baseline, n, model_dtype=torch.float64, eligible=eligible,
                                   master_dtype=torch.float64)
        values = gl["attribution"].flatten().numpy()
        assert values[0] > values[1] and abs(values[0] - 0.3) > 1e-3
    gl = q.gauss_legendre_pass(score, actual, baseline, 256, model_dtype=torch.float64, eligible=eligible,
                               master_dtype=torch.float64)
    np.testing.assert_allclose(gl["attribution"].flatten().numpy(), [0.3, 0.2], atol=1e-12)


def test_nonfinite_gradients_are_counted_and_make_the_cell_undefined():
    def score(x):
        x = x.to(torch.float64)
        return torch.sqrt((x - 0.5).abs()).sum()  # infinite slope at alpha = 0.5

    actual = torch.ones(1, 1, 1, dtype=torch.float64)
    baseline = torch.zeros(1, 1, 1, dtype=torch.float64)
    eligible = q.Eligible([0], 1)
    out = q.dyadic_pass(score, actual, baseline, 2, model_dtype=torch.float64, eligible=eligible,
                        master_dtype=torch.float64)
    assert not out["defined"] and out["rules"] == {}
    bad = [r for r in out["node_records"] if r["status"] != "finite"]
    assert [r["node"] for r in bad] == [2]
    assert q.total_nonfinite(bad[0]["gradient_nonfinite_full"]) == 1
    hist = q.historical_pass(score, actual, baseline, 4, eligible=eligible)
    assert hist["nonfinite_totals"]["nan"] + hist["nonfinite_totals"]["posinf"] + hist["nonfinite_totals"]["neginf"] == 1
    assert torch.isfinite(hist["attribution"]).all()


def test_forward_failures_are_recorded_not_raised():
    calls = {"n": 0}

    def score(x):
        calls["n"] += 1
        if calls["n"] == 2:
            raise NonFiniteAttributionError(dict(stage="sampler_step", step=3, alpha=None, nonfinite_count=1,
                                                 numel=1, context={}))
        return (x.to(torch.float64) ** 2).sum()

    actual = torch.ones(1, 1, 1, dtype=torch.float64)
    baseline = torch.zeros(1, 1, 1, dtype=torch.float64)
    out = q.dyadic_pass(score, actual, baseline, 2, model_dtype=torch.float64, eligible=q.Eligible([0], 1),
                        master_dtype=torch.float64)
    assert not out["defined"]
    assert out["node_records"][1]["status"] == "forward_nonfinite"
    assert len(out["node_records"]) == 5


def load_archived_kernel():
    loader = importlib.machinery.SourceFileLoader("archived_ig_kernel", str(ARCHIVED_KERNEL))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


@pytest.mark.parametrize("m", [4, 8, 64])
def test_historical_replica_equals_the_archived_bf16_kernel(m, capsys):
    actual, baseline, eligible, score = toy_problem(dtype=torch.bfloat16, seed=3)
    archived = load_archived_kernel().integrated_gradients(score, actual, baseline, m=m)
    capsys.readouterr()
    replica = q.historical_pass(score, actual, baseline, m, eligible=eligible)
    assert archived.dtype == torch.bfloat16
    assert torch.equal(replica["attribution"], eligible.take(archived))
    assert replica["ig_sum_input_dtype"] == float(archived.sum())
    assert replica["nonfinite_totals"]["nan"] == 0


def test_compare_maps_reports_identity_and_reversal():
    eligible = q.Eligible([0, 1, 2, 3], 1)
    a = torch.tensor([[[4.0], [3.0], [2.0], [1.0]]])
    same = q.compare_maps(a, a.clone(), eligible, counts=[0, 1, 2, 4])
    assert same["relative_l1"] == 0 and same["group_spearman"] == pytest.approx(1.0) and same["top1_equal"]
    reversed_map = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    flipped = q.compare_maps(a, reversed_map, eligible, counts=[0, 1, 2, 4])
    assert flipped["group_spearman"] == pytest.approx(-1.0) and not flipped["top1_equal"]
    assert [j["count"] for j in flipped["deleted_set_jaccard"]] == [1, 2]
