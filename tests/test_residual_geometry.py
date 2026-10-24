import csv
import gzip
import json
import shutil
import zlib

import pytest

from analysis.paired_rescoring import residual_geometry as rg
from analysis.revision.core import IntegrityError


def call(record, case, delta, baseline, maximum, index=1, high=0):
    return dict(case=case, figure="toy", cohort="c", record_id=record, episode_key=["s", record],
                episode_id=record, policy_call_idx=index, first_call=index == 0, paired_status="defined",
                delta=delta, auc_q=0.0, auc_l2=0.0, residual_actual=0.0, residual_baseline=baseline,
                max_interior_residual=maximum, ratio_max_interior_to_baseline=maximum / baseline,
                overshoot=high > 0, fully_in_range=high == 0, n_high_distance_overshoot=high,
                n_low_distance_undershoot=0)


def test_bins_use_declared_closure_and_order():
    edges = ["0.5", "1", "2"]
    assert rg.bin_label(1.0, edges, "right") == "(0.5,1]"
    assert rg.bin_label(0.5, edges, "right") == "<=0.5"
    assert rg.bin_label(2.5, edges, "right") == ">2"
    assert rg.bin_label(1.0, ["1e-1", "1", "10"], "left") == "[1,10)"
    assert rg.bin_label(0.01, ["1e-1", "1", "10"], "left") == "<1e-1"
    assert rg.bin_label(10.0, ["1e-1", "1", "10"], "left") == ">=10"
    assert rg.bin_order(edges, "right") == ["<=0.5", "(0.5,1]", "(1,2]", ">2"]


def test_geometry_row_ratio_and_flags():
    source = dict(record_id="r", episode_key=["s"], episode_id="e", policy_call_idx=0, paired_status="defined",
                  delta=-3.0, auc_q=-4.0, auc_l2=-1.0, residual_actual=0.0, residual_baseline=0.5,
                  max_interior_residual=200.0, n_high_distance_overshoot=2, n_low_distance_undershoot=0)
    row = rg.geometry_row("c", "f", "k", source)
    assert row["ratio_max_interior_to_baseline"] == 400.0 and row["first_call"] and row["overshoot"]
    assert not row["fully_in_range"]
    assert rg.geometry_row("c", "f", "k", dict(source, residual_baseline=0.0))["ratio_max_interior_to_baseline"] is None


def test_correlations_and_bin_counts_by_subset():
    rows = [call("a", "T", 0.1, 1.0, 0.9), call("b", "T", 0.05, 2.0, 1.5), call("c", "T", 0.02, 3.0, 2.0),
            call("d", "T", -50.0, 0.1, 100.0, index=0, high=1), call("e", "T", -5.0, 0.3, 9.0, high=1),
            call("f", "T", -500.0, 0.05, 400.0, index=0, high=1)]
    result = {(r["case"], r["subset"]): r for r in rg.correlations(rows, ["T"])}
    assert result[("T", "all")]["n_calls"] == 6 and result[("T", "fully_in_range")]["n_calls"] == 3
    assert result[("T", "overshoot")]["spearman_delta_vs_ratio"] == -1.0
    assert result[("T", "fully_in_range")]["spearman_delta_vs_B"] == -1.0
    protocol = dict(bins=dict(B_edges=["1e-1", "1", "10"], ratio_edges=["0.5", "1", "2", "10", "100"]))
    counts = rg.bins(rows, ["T"], protocol)
    b = {r["B_bin"]: r for r in counts if r["binning"] == "B"}
    assert b["<1e-1"]["n_calls"] == 1 and b["[1e-1,1)"]["n_calls"] == 2 and b["[1,10)"]["n_calls"] == 3
    ratio = {r["ratio_bin"]: r for r in counts if r["binning"] == "ratio"}
    assert ratio[">100"]["n_delta_below_minus_100"] == 1 and ratio["(0.5,1]"]["n_calls"] == 3
    joint = [r for r in counts if r["binning"] == "joint"]
    assert sum(r["n_calls"] for r in joint) == 6 and all(r["n_calls"] for r in joint)


def test_figures_are_byte_reproducible(tmp_path):
    from analysis.paired_rescoring.residual_geometry import figures
    rows = []
    for k in range(8):
        for i in range(12):
            rows.append(call(f"{k}-{i}", f"case{k}", (-1) ** i * 10.0 ** (i % 4 - 2), 10.0 ** (i % 5 - 2),
                             10.0 ** (i % 5 - 2) * 10.0 ** (i % 6 - 2), index=i % 3, high=int(i % 6 > 2)))
    _, protocol = rg.load_protocol()
    spec = dict(protocol["figures"], figures=[dict(name="toy", panels=[dict(case=f"case{k}", label=f"case {k}") for k in range(8)])])
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        info = figures.render(rows, dict(figures=spec), tmp_path / name)
    for suffix in ("pdf", "png"):
        assert (tmp_path / "a" / f"toy.{suffix}").read_bytes() == (tmp_path / "b" / f"toy.{suffix}").read_bytes()
    assert b"CreationDate" not in (tmp_path / "a" / "toy.pdf").read_bytes()
    assert info["png_dpi"] == 300


def assert_reproduces(fresh, committed):
    """Data artifacts and PDFs byte for byte; gzip by payload (and bytes when the
    zlib build matches); PNG by bytes when Pillow's zlib matches, else by pixels."""
    from PIL import features
    recorded = json.loads((committed / "provenance.json").read_bytes())
    produced = json.loads((fresh / "provenance.json").read_bytes())
    same_zlib = recorded["environment"]["zlib_runtime"] == zlib.ZLIB_RUNTIME_VERSION
    same_png_zlib = recorded["figure_environment"]["pillow_zlib"] == features.version("zlib")
    for name, entry in recorded["artifacts"].items():
        mine = produced["artifacts"][name]
        if "uncompressed_sha256" in entry:
            assert mine["uncompressed_sha256"] == entry["uncompressed_sha256"], name
            if same_zlib:
                assert mine["sha256"] == entry["sha256"], name
        elif name.endswith(".png"):
            assert produced["png_pixels_sha256"][name] == recorded["png_pixels_sha256"][name], name
            if same_png_zlib:
                assert mine["sha256"] == entry["sha256"], name
        else:
            assert (fresh / name).read_bytes() == (committed / name).read_bytes(), name
    for key in ("protocol_sha256", "source_sha256", "inputs", "analysis_version"):
        assert produced[key] == recorded[key], key


def test_real_data_reproduction_and_frozen_equality(tmp_path):
    fresh = tmp_path / "fresh"
    rg.run(fresh)
    assert_reproduces(fresh, rg.RESULTS)
    assert rg.verify(rg.RESULTS)["status"] == "verified"
    checks = json.loads((rg.RESULTS / "checks.json").read_bytes())
    head = checks["headline"]
    assert head["n_frozen_rows"] == head["n_recomputed"] == head["n_exact_delta"] == 6000
    assert head["n_exact_residual_baseline"] == head["n_exact_residual_points"] == 6000
    assert checks["oneb"]["n_recomputed"] == checks["oneb"]["n_exact_delta"] == 4796
    assert checks["n_zero_baseline_calls"] == 0 and checks["n_undefined_calls"] == 0
    with (rg.RESULTS / "correlations.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16 * 3
    calls = list(csv.DictReader(gzip.open(rg.RESULTS / "calls.csv.gz", "rt", encoding="utf-8", newline="")))
    q_tail = min((r for r in calls if r["case"] == "Q:vision:insertion"), key=lambda r: float(r["delta"]))
    assert float(q_tail["delta"]) < -4400 and q_tail["first_call"] == "True"
    assert float(q_tail["ratio_max_interior_to_baseline"]) > 30000
    for name in ("headline.pdf", "oneb.pdf"):
        assert b"CreationDate" not in (rg.RESULTS / name).read_bytes()
    copy = tmp_path / "copy"
    shutil.copytree(rg.RESULTS, copy)
    (copy / "bins.csv").write_bytes(b"x\n")
    with pytest.raises(IntegrityError, match="Artifact mismatch"):
        rg.verify(copy)
