"""Regression checks for denominators, provenance and saved-tensor calculations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1] / "analysis/numerical_case/2026-10-01-v1"
spec = importlib.util.spec_from_file_location("saved_numerical_summary", ROOT / "summarize.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.fixture(scope="module")
def data():
    raw = (ROOT / "inputs/sealed_v6.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == summary.INPUT_SHA256
    return json.loads(raw)


def test_complete_roster_and_missing_statuses(data):
    value = summary.summarize(data)
    assert (value["planned_contexts"], value["available_contexts"], value["completed_contexts"], value["completed_episodes"]) == (24,21,12,6)
    assert value["cell_status_counts"] == {"complete_diagnostics":72,"call_not_reached":18,"unsealed_at_audit_start":54}
    assert len(value["roster"]) == 24
    assert all(r["complete_cells"] == 0 for r in value["roster"] if r["stratum"].endswith("-1b"))
    assert sum(r["context_id"] is None for r in value["roster"]) == 3
    # Three unavailable calls with null context IDs remain three separate planned units.
    assert len({(r["stratum"],r["episode"],r["call"]) for r in value["roster"] if r["context_id"] is None}) == 3


def test_dropped_or_duplicate_arm_is_rejected(data):
    for operation in ("drop", "duplicate"):
        altered = copy.deepcopy(data)
        if operation == "drop":
            altered["rows"].pop()
        else:
            altered["rows"][1] = copy.deepcopy(altered["rows"][0])
        with pytest.raises(ValueError,match="all six arms"):
            summary.summarize(altered)


def test_available_context_identity_corruption_is_rejected(data):
    altered = copy.deepcopy(data)
    altered["rows"][0]["context_id"] = "0"*64
    with pytest.raises(ValueError,match="identity"):
        summary.summarize(altered)


def test_prior_attempts_deduplicate_across_six_arms(data):
    value = summary.summarize(data)
    assert len(value["attempts"]) == 48
    assert value["distinct_attempt_status_counts"] == {
        "completed":13,"infrastructure_or_contract_failure":2,"unsealed_at_audit_start":33}
    # Completed jobs include one unavailable context; no missing attempt is called a failure.
    assert value["completed_contexts"] == 12


def test_equality_and_coverage_counts_are_not_conflated(data):
    value = summary.summarize(data)
    assert value["repeatability_equality_check_counts"] == {"satisfies":1800}
    assert value["repeated_condition_and_coverage_check_counts"] == {"satisfies":2880}
    assert value["criterion_status_counts"] == {"satisfies":11146,"violates":302,"not_applicable":144}
    assert len(value["per_cell"]) == 72


def test_saved_criteria_not_silently_reclassified(data):
    altered = copy.deepcopy(data)
    altered["rows"][0]["checks"][0]["status"] = "violates"
    with pytest.raises(ValueError,match="Criterion counts"):
        summary.summarize(altered)


def test_finest_pair_denominator_and_full_gap_are_distinct(data):
    value = summary.summarize(data)
    row = next(r for r in value["per_cell"] if r["stratum"]=="picksingleycb-170m" and r["call"]==12 and r["modality"]=="state" and r["target"]=="Q")
    assert row["coordinate_absolute_l1_IG"] == pytest.approx(.25489155943796504)
    assert row["reference_coordinate_l1_IG"] == pytest.approx(.7935805439337855)
    assert row["coordinate_relative_l1_IG"] == pytest.approx(.3211917950690492)
    assert row["finest_signed_gap"] == pytest.approx(.6821948885917664)
    assert row["finest_relative_residual"] == pytest.approx(.1143573741991385)
    assert row["max_rms_curve_absolute_difference"] == 0


def test_finest_pair_summary_retains_heterogeneity(data):
    value = summary.summarize(data)
    rows = {(r["modality"],r["target"]):r for r in value["modality_target_extrema"]}
    assert {k:r["contexts_with_finest_pair_threshold_violation"] for k,r in rows.items()} == {
        ("language","Q"):0,("language","L2"):0,("vision","Q"):1,("vision","L2"):4,("state","Q"):2,("state","L2"):2}
    assert sum(r["finest_completeness_over_recorded_threshold"] for r in value["per_cell"]) == 4


def test_saved_tensor_corruption_is_rejected(data):
    record = copy.deepcopy(data["ycb_saved_tensors"][0]["tensors"][0]["tensors"]["gradient"])
    record["values"][0] += .01
    with pytest.raises(ValueError,match="tensor hash"):
        summary.tensor_values(record)


def test_saved_gradient_geometry_and_directional_values(data):
    value = summary.diagnose_ycb(data)
    assert len(value["budget_accounting"]) == 12
    assert len(value["fixed_probes"]) == 28
    nonzero = [r["gradient_identity_relative_l1"] for r in value["shared_distance_gradient_identity"] if r["gradient_identity_relative_l1"] is not None]
    assert max(nonzero) == pytest.approx(4.1367372648599594e-7)
    assert sum(r["zero_gradient_endpoint"] for r in value["shared_distance_gradient_identity"]) == 2
    probe = next(r for r in value["fixed_probes"] if (r["call"],r["target"],r["alpha"]) == (0,"Q",.25))
    assert probe["directional_derivative"] == pytest.approx(-1.4317692213953706)
    assert value["model_evaluations"] == 0
    assert all(r["active_entries"] == 512 and r["sigma_squared"] == 1 for r in value["distance_normalizations"])


def test_geometry_rejects_unbound_dimension(data):
    altered = copy.deepcopy(data)
    altered["ycb_saved_tensors"][0]["distance_normalization"]["active_entries"] = 256
    with pytest.raises(ValueError,match="normalization provenance"):
        summary.diagnose_ycb(altered)


def test_source_collection_is_fixed_index_not_uniform_call_sampling(data):
    for row in data["source_collection_protocol"]["strata"]:
        assert row["max_policy_calls"] == 13
        assert row["max_episode_steps"] == 400
        assert row["evaluate_calls"] == [0,12]
    assert len(data["queues"]) == 3
    assert all(len(q["jobs"]) == 24 for q in data["queues"])
    assert all(s["earlier_complete_report_hashes_unchanged"] for s in data["earlier_snapshots"] if s["same_estimand"])


def test_reproducer_is_byte_identical_and_tex_cites_csv(tmp_path):
    for name in ("one","two"):
        subprocess.run([sys.executable,str(ROOT/"summarize.py"),"--output",str(tmp_path/name)],check=True,capture_output=True)
    for path in (tmp_path/"one").iterdir():
        assert path.read_bytes() == (tmp_path/"two"/path.name).read_bytes()
        if path.suffix in (".json", ".md", ".tex"):
            assert b"\r" not in path.read_bytes()
        elif path.suffix == ".csv":
            assert path.read_bytes().count(b"\n") == path.read_bytes().count(b"\r\n")
    assert b"\r" not in (ROOT/"inputs/sealed_v6.json").read_bytes()
    csv_sha = hashlib.sha256((tmp_path/"one/strata.csv").read_bytes()).hexdigest()
    assert csv_sha in (tmp_path/"one/numerical_roster.tex").read_text()


def test_changed_input_is_rejected_before_output(data,tmp_path):
    input_path = tmp_path/"changed.json"
    input_path.write_text(json.dumps(data))
    run = subprocess.run([sys.executable,str(ROOT/"summarize.py"),"--input",str(input_path),"--output",str(tmp_path/"out")],capture_output=True,text=True)
    assert run.returncode != 0 and "Input projection changed" in run.stderr
    assert not (tmp_path/"out").exists()
