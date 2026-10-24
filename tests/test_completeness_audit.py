import csv
import gzip
import json
import math
import shutil
import zlib

import numpy as np
import pytest

from analysis import completeness_audit as ca
from analysis.completeness_audit import endpoint_example as ee
from analysis.revision.core import IntegrityError, Record


def test_bf16_rounding_ties_to_even_and_detects_stored_sums():
    assert ca.bf16_round(1.0 + 2.0 ** -8) == 1.0
    assert ca.bf16_round(1.0 + 3 * 2.0 ** -8) == 1.0 + 2.0 ** -6
    assert ca.bf16_round(-1.0 - 2.0 ** -9) == -1.0
    assert ca.bf16_round(0.00063323974609375) == 0.00063323974609375
    assert ca.bf16_round(8.75) == 8.75 and ca.bf16_round(8.7557) == 8.75
    assert ca.bf16_round(0.0) == 0.0
    with pytest.raises(IntegrityError):
        ca.bf16_round(float("nan"))


def test_bf16_half_spacings_and_relative_bound():
    assert ca.bf16_half_spacings(1.0) == (2.0 ** -9, 2.0 ** -8)
    assert ca.bf16_half_spacings(-1.5) == (2.0 ** -8, 2.0 ** -8)
    assert ca.rounding_interval(-1.0) == (-1.0 - 2.0 ** -8, -1.0 + 2.0 ** -9)
    rng = np.random.default_rng(0)
    for value in (rng.standard_normal(2000) * 10.0 ** rng.integers(-6, 4, 2000)).tolist():
        rounded = ca.bf16_round(value)
        assert abs(rounded - value) <= ca.BF16_RELATIVE_BOUND * abs(value)
        low, high = ca.rounding_interval(rounded)
        assert low <= value <= high


def step(line, gap, total, **extra):
    row = dict(event="step", task="T", model="M", seed=1, episode=0, policy_call_idx=line,
               vision_gap=gap, vision_ig_sum=total, vision_err=abs(total - gap) / (abs(gap) + 1e-12),
               lang_gap=gap, lang_ig_sum=total, lang_err=0.0, state_gap=gap, state_ig_sum=total, state_err=0.0)
    row.update(extra)
    return Record("s", "0" * 64, line + 1, f"{line:064d}", row)


def test_call_rows_bf16_explanation_and_definite_shortfall():
    exact = ca.bf16_round(0.5 + 2.0 ** -12)
    rows = ca.call_rows("c", "s", [step(0, 0.5 + 2.0 ** -12, exact), step(1, 1.0, 0.9375), step(2, 0.0, 0.25)])
    vision = [r for r in rows if r["modality"] == "vision"]
    assert vision[0]["rounding_explains"] and vision[0]["adjusted_residual"] == 0.0
    assert not vision[1]["rounding_explains"] and vision[1]["definite_shortfall"]
    assert vision[1]["adjusted_residual"] == pytest.approx((1.0 - (0.9375 + 2.0 ** -9)) / 1.0)
    assert vision[2]["residual"] is None
    summary = ca.residual_summary(vision)
    assert summary["n_undefined_zero_gap"] == 1 and summary["shortfall_share"] == 1.0
    bf16 = ca.bf16_summary(vision)
    assert bf16["n_analysed"] == 2 and bf16["share_rounding_explains"] == 0.5
    assert bf16["definite_shortfall_share"] == 0.5


def faith(line, gap):
    row = dict(event="faithfulness", task="T", model="M", seed=1, episode=0, policy_call_idx=line,
               vision_f_input=0.0, vision_f_baseline=-gap, lang_f_input=0.0, lang_f_baseline=-gap)
    return Record("f", "1" * 64, line + 1, f"{line:064d}", row)


def test_step_selection_requires_a_unique_full_fingerprint_match():
    loaded = {"f": dict(last=[faith(0, 0.5), faith(1, 0.25)]),
              "own": dict(last=[step(0, 0.5, 0.5), step(1, 0.25, 0.25)]),
              "partial": dict(last=[step(0, 0.5, 0.5)]),
              "other": dict(last=[step(0, 0.6, 0.5), step(1, 0.25, 0.25)])}
    cohort = dict(faithfulness=[dict(path="f")], candidate_steps=[dict(path=p) for p in ("partial", "other", "own")])
    selection = ca.select_steps(cohort, loaded)
    assert selection[0]["selected"] == "own" and selection[0]["qualified"] == ["own"]
    assert selection[0]["selection_status"] == "fingerprint_unique" and selection[0]["alternatives"] == []
    loaded["copy"] = dict(last=loaded["own"]["last"])
    cohort["candidate_steps"].append(dict(path="copy"))
    with pytest.raises(IntegrityError, match="not determined"):
        ca.select_steps(cohort, loaded)
    cohort["declared_primary_when_ambiguous"] = {"f": dict(steps="copy", basis="chronology")}
    selection = ca.select_steps(cohort, loaded)
    assert selection[0]["selected"] == "copy" and selection[0]["alternatives"] == ["own"]
    assert selection[0]["selection_status"] == "fingerprint_ambiguous_declared_primary"
    cohort["declared_primary_when_ambiguous"] = {"f": dict(steps="other", basis="unqualified")}
    with pytest.raises(IntegrityError, match="not determined"):
        ca.select_steps(cohort, loaded)


def test_endpoint_example_reproduces_reviewer_value_and_identity():
    rows, identity, numpy_part = ee.compute(dict(endpoint_example=dict(epsilon="1e-12", intervals=[16, 64], targets=["N", "Q"])))
    values = {(r["target"], r["m"], r["rule"]): r for r in rows}
    assert values[("N", 64, "endpoint_inclusive")]["rounded_6"] == 0.984615
    assert round(values[("N", 64, "trapezoid")]["value_float64"], 6) == 0.992187
    for (target, m, rule), row in values.items():
        assert abs(row["float64_minus_mp50"]) < 1e-15
        if target == "Q":
            assert row["value_float64"] == pytest.approx(0.5, abs=1e-15)
    for row in identity:
        assert abs(row["float64_difference"]) <= 4 * math.ulp(row["endpoint_inclusive"])
        assert abs(float(row["mp50_difference"])) < 1e-45
    assert all(abs(r["minus_stable_float64"]) < 1e-14 for r in numpy_part["rows"])
    x, w = ee.legendre_nodes(16)
    nx, nw = np.polynomial.legendre.leggauss(16)
    assert np.allclose([float(v) for v in x], nx, atol=1e-15) and np.allclose([float(v) for v in w], nw, atol=1e-14)


def test_real_data_reproduction_selection_and_reviewer_value(tmp_path):
    import platform
    fresh = tmp_path / "fresh"
    ca.run(fresh)
    recorded = json.loads((ca.RESULTS / "provenance.json").read_bytes())
    produced = json.loads((fresh / "provenance.json").read_bytes())
    same_zlib = recorded["environment"]["zlib_runtime"] == zlib.ZLIB_RUNTIME_VERSION
    for name, entry in recorded["artifacts"].items():
        if "uncompressed_sha256" in entry:
            assert produced["artifacts"][name]["uncompressed_sha256"] == entry["uncompressed_sha256"], name
            if same_zlib:
                assert produced["artifacts"][name]["sha256"] == entry["sha256"], name
        else:
            assert (fresh / name).read_bytes() == (ca.RESULTS / name).read_bytes(), name
    numpy_part = json.loads((ca.RESULTS / "endpoint_numpy_leggauss.json").read_bytes())
    fresh_numpy = json.loads((fresh / "endpoint_numpy_leggauss.json").read_bytes())
    if numpy_part["environment"] == dict(numpy=np.__version__, system=platform.system(), machine=platform.machine()):
        assert fresh_numpy == numpy_part
    for old, new in zip(numpy_part["rows"], fresh_numpy["rows"]):
        assert new["value_numpy_leggauss"] == pytest.approx(old["value_numpy_leggauss"], rel=1e-14, abs=1e-15)
    for key in ("protocol_sha256", "source_sha256", "inputs", "analysis_version"):
        assert produced[key] == recorded[key], key
    assert ca.verify(ca.RESULTS)["status"] == "verified"
    selection = json.loads((ca.RESULTS / "step_selection.json").read_bytes())
    picked = {path_name(s["faithfulness"]): (path_name(s["selected"]), s["selection_status"])
              for rows in selection["selection"].values() for s in rows}
    assert picked["metrics_faithfulness_metrics_PickCube-v1_170m_seed42_logpi_20260615_004816.jsonl"] == (
        "metrics_PickCube-v1_170m_seed42_logpi.jsonl", "fingerprint_unique")
    assert picked["metrics_faithfulness_metrics_PickCube-v1_170m_seed142_l2_20260530_082947.jsonl"] == (
        "metrics_PickCube-v1_170m_seed142_l2.jsonl", "fingerprint_unique")
    assert picked["metrics_faithfulness_metrics_PegInsertionSide-v1_170m_seed142_m128_20260529_160739.jsonl"][1] == \
        "fingerprint_ambiguous_declared_primary"
    base = selection["comparison_populations"][0]
    assert all(m["vision_general_agree"] == 0 and m["lang_general_agree"] == 0 for m in base["fingerprints"])
    assert sum(m["n_joined"] for m in base["fingerprints"]) == 750
    with (ca.RESULTS / "endpoint_example.csv").open(encoding="utf-8", newline="") as stream:
        rows = {(r["target"], r["m"], r["rule"]): r for r in csv.DictReader(stream)}
    assert rows[("N", "64", "endpoint_inclusive")]["rounded_6"] == "0.984615"
    with (ca.RESULTS / "endpoint_identity.csv").open(encoding="utf-8", newline="") as stream:
        identity = list(csv.DictReader(stream))
    assert sorted({(r["target"], int(r["m"])) for r in identity}) == sorted(
        (t, m) for t in ("N", "Q") for m in (16, 32, 64, 128, 256, 512))
    for row in identity:
        e, t = float(row["endpoint_inclusive"]), float(row["trapezoid"])
        rhs = t + ((float(row["f_at_baseline"]) + float(row["f_at_input"])) / 2 - t) / (int(row["m"]) + 1)
        assert abs(e - rhs) <= 4 * math.ulp(e) and abs(float(row["mp50_difference"])) < 1e-45
    with (ca.RESULTS / "residuals.csv").open(encoding="utf-8", newline="") as stream:
        residuals = {(r["cohort"], r["grouping"], r["modality"], r["occurrence_view"]): r for r in csv.DictReader(stream)}
    assert residuals[("Q_ranked_headline", "pooled", "vision", "last")]["n_rows"] == "750"
    assert residuals[("N_ranked_headline", "pooled", "vision", "conflict_excluded")]["n_rows"] == "727"
    copy = tmp_path / "copy"
    shutil.copytree(ca.RESULTS, copy)
    (copy / "bf16.csv").write_bytes(b"x\n")
    with pytest.raises(IntegrityError, match="Artifact mismatch"):
        ca.verify(copy)


def path_name(path):
    return path.rsplit("/", 1)[-1]
