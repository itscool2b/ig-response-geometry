"""RDT-1B PickCube success split of the paired Q-minus-N rescoring (CPU only).

The saved RDT-1B curves are Q-scored (historical logpi target). Each call is
rescored with the frozen ``paired_curve``; Delta = AUC_Q - AUC_N within the
same saved curve and endpoints. Success labels come from each run's own
``episode_end`` rows. Nothing is clipped, trimmed or re-run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import zlib

import numpy as np

from analysis.paired_rescoring import curve_stats as cs
from analysis.revision.core import IntegrityError, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROTOCOL = HERE / "protocol.json"
RESULTS = HERE / "results" / "2026-10-23-v1"
OUTPUTS = ("protocol.json", "inputs.json", "calls.csv.gz", "episodes.csv", "summary.csv", "call_index.csv")
SOURCES = ("analysis/paired_rescoring/oneb_success/__init__.py",
           "analysis/paired_rescoring/oneb_success/__main__.py",
           "analysis/paired_rescoring/curve_stats/__init__.py",
           "analysis/paired_rescoring/analyze.py",
           "analysis/revision/core.py", "analysis/revision/analyze.py", "analysis/revision/verify.py")
CALL_FIELDS = ("run", "role", "success", "record_id", "source", "line", "episode_key", "episode_id",
               "policy_call_idx", "modality", "direction", "paired_status", "delta", "auc_q", "auc_l2",
               "gap_q", "gap_l2", "residual_actual", "residual_baseline", "max_interior_residual",
               "residual_points", "n_grid_nodes", "n_in_range", "n_high_distance_overshoot",
               "n_low_distance_undershoot", "contribution_in_range", "contribution_high_distance_overshoot",
               "contribution_low_distance_undershoot", "decomposition_residual",
               "generalized_identity_max_error")
SUMMARY_FIELDS = (
    "population", "case", "modality", "direction", "outcome_group", "n_raw_calls", "n_defined_calls",
    "n_undefined_calls", "n_raw_episodes", "n_defined_episodes", "calls_per_episode_min",
    "calls_per_episode_max", "paired_median", "paired_median_ci_lo", "paired_median_ci_hi",
    "positive_calls", "negative_calls", "zero_calls", "positive_share", "overshoot_calls",
    "overshoot_share", "undershoot_calls", "fully_in_range_calls", "fully_in_range_share",
    "fully_in_range_positive_calls", "overshoot_positive_calls", "overshoot_positive_share",
    "min_delta", "max_delta", "equal_call_mean", "equal_episode_mean", "equal_episode_mean_ci_lo",
    "equal_episode_mean_ci_hi", "equal_episode_contribution_in_range",
    "equal_episode_contribution_high_distance_overshoot",
    "equal_episode_contribution_low_distance_undershoot", "decomposition_check",
    "median_episode_mean", "positive_episode_means", "positive_episode_mean_share",
    "episodes_any_overshoot", "episodes_any_overshoot_share", "episodes_fully_in_range",
    "episodes_fully_in_range_share", "min_episode_mean", "max_episode_mean", "interval_status",
    "bootstrap_draws", "bootstrap_seed", "population_sha256")
INDEX_FIELDS = (
    "population", "case", "modality", "direction", "outcome_group", "n_raw_calls", "n_defined_calls",
    "n_defined_episodes", *(f"{part}_{key}" for part in ("first", "later") for key in (
        "raw_calls", "defined_calls", "median", "equal_call_mean", "min_delta", "max_delta",
        "positive_share", "overshoot_calls", "overshoot_share", "fully_in_range_share",
        "mean_contribution", "in_range_contribution", "high_distance_overshoot_contribution")),
    "first_plus_later_contribution")
EPISODE_FIELDS = ("run", "role", "case", "modality", "direction", "success", "episode_id", "episode_key",
                  "n_calls", "n_defined_calls", "mean_delta", "median_delta", "mean_contribution_in_range",
                  "mean_contribution_high_distance_overshoot", "mean_contribution_low_distance_undershoot",
                  "overshoot_calls", "fully_in_range_calls", "first_call_delta")


def load_protocol(root=ROOT):
    blob = (Path(root) / "analysis/paired_rescoring/oneb_success/protocol.json").read_bytes()
    return blob, json.loads(blob)


def authenticate(root, protocol):
    """Authenticate every declared file, then prove each run's own provenance."""
    runs = protocol["runs"]
    expected = {run[kind]["path"]: run[kind]["sha256"] for run in runs for kind in ("faithfulness", "steps")}
    loaded, files, anchor = cs.authenticate_named(root, list(expected), expected=expected)
    if anchor["input_manifest_sha256"] != protocol["anchors"]["input_manifest_sha256"]:
        raise IntegrityError("Input manifest differs from the protocol anchor")
    selected, run_audit = {}, []
    for run in runs:
        faith_file, step_file = loaded[run["faithfulness"]["path"]], loaded[run["steps"]["path"]]
        faith = cs.faithfulness_rows(faith_file["last"])
        clean_ids = {r.id for r in faith_file["clean"]}
        steps = step_file["last"]
        mark = cs.fingerprint(faith, steps)
        if not (mark["all_agree_original"] and mark["all_agree_general"]):
            raise IntegrityError(f"Own-sidecar fingerprint fails for run {run['run']}")
        structure, labels = cs.episode_structure(faith, cs.episode_ends(steps))
        if structure["call_count_mismatches"]:
            raise IntegrityError(f"episode_end call counts differ from saved curves: {run['run']}")
        roster = run["expected_roster"]
        observed = {k: structure[k] for k in roster}
        if observed != roster:
            raise IntegrityError(f"Run roster differs from the protocol: {run['run']}: {observed}")
        native = cs.native_scale_check(faith, protocol["native_scale"])
        selected[run["run"]] = dict(faith=faith, labels=labels)
        run_audit.append(dict(
            run=run["run"], role=run["role"], faithfulness=run["faithfulness"]["path"],
            steps=run["steps"]["path"], fingerprint=mark, structure=structure, native=native,
            conflict_excluded_membership_identical=all(r.id in clean_ids for r in faith),
            documented_language_input=run["documented_language_input"]))
    return selected, dict(anchor=anchor, files=files, runs=run_audit)


def compute(root, protocol):
    selected, audit = authenticate(root, protocol)
    roles = {run["run"]: run["role"] for run in protocol["runs"]}
    calls = []
    for run in protocol["runs"]:
        label = run["run"]
        for record in selected[label]["faith"]:
            success = selected[label]["labels"][tuple(record.row[k] for k in cs.EPISODE_FIELDS)]
            for modality in protocol["modalities"]:
                for direction in protocol["directions"]:
                    row = cs.rescore(record, modality, direction, native=protocol["native_scale"],
                                     dimension=protocol["dimension"], epsilon=protocol["stabilizer_epsilon"])
                    calls.append(dict(run=label, role=roles[label], success=success, **row))
    draws, seed = protocol["bootstrap"]["draws"], protocol["bootstrap"]["seed"]
    summaries, indices = [], []
    for population in protocol["populations"]:
        for modality in population["modalities"]:
            for direction in protocol["directions"]:
                case = [c for c in calls if c["run"] in population["runs"]
                        and c["modality"] == modality and c["direction"] == direction]
                for group in protocol["outcome_groups"]:
                    members = [c for c in case if group == "all" or c["success"] == (group == "success")]
                    identity = dict(population=population["id"], case=f"{modality}:{direction}",
                                    modality=modality, direction=direction, outcome_group=group)
                    summaries.append(dict(identity, **cs.summarize(members, draws=draws, seed=seed)))
                    indices.append(dict(identity, **cs.call_index_summary(members)))
    episodes = []
    for run in protocol["runs"]:
        for modality in protocol["modalities"]:
            for direction in protocol["directions"]:
                case = [c for c in calls if c["run"] == run["run"] and c["modality"] == modality
                        and c["direction"] == direction]
                labels = {c["episode_id"]: c["success"] for c in case}
                for row in cs.episode_rows(case):
                    episodes.append(dict(run=run["run"], role=run["role"], case=f"{modality}:{direction}",
                                         modality=modality, direction=direction,
                                         success=labels[row["episode_id"]], **row))
    audit["n_calls_by_run"] = {run["run"]: sum(c["run"] == run["run"] for c in calls) // (
        len(protocol["modalities"]) * len(protocol["directions"])) for run in protocol["runs"]}
    audit["max_generalized_identity_error"] = max(c["generalized_identity_max_error"] or 0 for c in calls)
    audit["max_abs_decomposition_residual"] = max(abs(c["decomposition_residual"] or 0) for c in calls)
    audit["n_undefined_case_calls"] = sum(c["paired_status"] != "defined" for c in calls)
    return dict(audit=audit, calls=calls, summaries=summaries, indices=indices, episodes=episodes)


def source_hashes(root):
    return {name: sha256((Path(root) / name).read_bytes()) for name in SOURCES}


def environment():
    return dict(python=platform.python_version(), numpy=np.__version__, system=platform.system(),
                machine=platform.machine(), zlib_runtime=zlib.ZLIB_RUNTIME_VERSION)


def run(output, *, root=ROOT):
    output, root = Path(output), Path(root)
    if output.exists():
        raise IntegrityError("Refusing an existing output directory")
    protocol_bytes, protocol = load_protocol(root)
    result = compute(root, protocol)
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    cs.write_json(output / "inputs.json", result["audit"])
    (output / "calls.csv.gz").write_bytes(cs.gzip_bytes(cs.csv_bytes(result["calls"], CALL_FIELDS)))
    (output / "episodes.csv").write_bytes(cs.csv_bytes(result["episodes"], EPISODE_FIELDS))
    (output / "summary.csv").write_bytes(cs.csv_bytes(result["summaries"], SUMMARY_FIELDS))
    (output / "call_index.csv").write_bytes(cs.csv_bytes(result["indices"], INDEX_FIELDS))
    provenance = dict(
        schema_version=1, analysis=protocol["analysis"], analysis_version=protocol["analysis_version"],
        protocol_sha256=sha256(protocol_bytes), source_sha256=source_hashes(root),
        inputs=[dict(path=f["path"], sha256=f["sha256"], checkout_sha256=f["checkout_sha256"])
                for f in result["audit"]["files"]],
        artifacts={name: cs.artifact_record(output / name) for name in OUTPUTS},
        environment=environment(),
        command="python -m analysis.paired_rescoring.oneb_success --output <new-empty-directory>",
        verify_command="python -m analysis.paired_rescoring.oneb_success --verify <result-directory>",
        interpretation="retrospective_descriptive_response_only_rescoring_no_efficacy_claim")
    cs.write_json(output / "provenance.json", provenance)
    return provenance


def verify(output=RESULTS, *, root=ROOT):
    output, root = Path(output), Path(root)
    provenance = json.loads((output / "provenance.json").read_bytes())
    if set(provenance["artifacts"]) != set(OUTPUTS):
        raise IntegrityError("Artifact roster mismatch")
    for name, record in provenance["artifacts"].items():
        if cs.artifact_record(output / name) != record:
            raise IntegrityError(f"Artifact mismatch: {name}")
    protocol_bytes, protocol = load_protocol(root)
    if sha256(protocol_bytes) != provenance["protocol_sha256"] or (output / "protocol.json").read_bytes() != protocol_bytes:
        raise IntegrityError("Protocol mismatch")
    if source_hashes(root) != provenance["source_sha256"]:
        raise IntegrityError("Source mismatch")
    _, audit = authenticate(root, protocol)
    recorded = json.loads((output / "inputs.json").read_bytes())
    if {k: recorded[k] for k in ("anchor", "files", "runs")} != json.loads(cs.json_text(audit)):
        raise IntegrityError("Input provenance mismatch")
    return dict(status="verified", artifacts=len(OUTPUTS), analysis_version=provenance["analysis_version"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--verify", type=Path)
    args = parser.parse_args()
    print(cs.json_text(run(args.output) if args.output else verify(args.verify)), end="")
