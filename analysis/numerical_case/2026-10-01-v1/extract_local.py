"""Project preserved local evidence into a portable, CPU-only summary input.

No network, model imports, or inference. Original evidence is never modified.
Torch is used only to decode authenticated saved CPU tensors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = ""

AUDIT = "numerical_audits_20261001T0336Z/v6-sealed-20261001T0336Z/audit.json"
AUDIT_SHA = "2b24d6c62be16adf6a6db1e5f4cec02b495262b44fad93a973bbdc8241a25ac7"
EARLIER = {
    "v6_0222": ("numerical_audits_20261001T0222Z/v6-partial-20261001T0222Z/audit.json", "51e5ade48521c619b45ff33c82f70aceb3b6e03a888dc30704848ac77b6f19a2"),
    "v6_0307": ("numerical_audits_20261001T0307Z/v6-sealed-20261001T0307Z/audit.json", "ca6eae2dfb1feec32affcf277a0009f101f0e06d91326755c0a443b8b7359c42"),
    "v4_0222": ("numerical_audits_20261001T0222Z/v4-complete-20261001T0222Z/audit.json", "f2f4ac599b4035e45b8dd8417bed36473087cdef823527248c905d1ad4ec0d3e"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, expected=None):
    if expected is not None and digest(path) != expected:
        raise ValueError(f"Input hash mismatch: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value):
    if isinstance(value, dict):
        return {clean(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        return value.replace("/workspace/tmlr/", "artifact-root/")
    return value


def tensor_record(tensor, expected=None):
    import torch
    value = tensor.detach().cpu().contiguous()
    if value.dtype != torch.float32:
        raise ValueError("Expected the recorded FP32 probe tensor")
    header = {"shape": list(value.shape), "dtype": str(value.dtype)}
    sha = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    sha.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    if expected and sha.hexdigest() != expected:
        raise ValueError("Tensor hash mismatch")
    return {**header, "sha256": sha.hexdigest(), "values": value.reshape(-1).tolist()}


def project(evidence, source_repo):
    import torch
    audit = read(evidence / AUDIT, AUDIT_SHA)
    result = {k: audit[k] for k in ("criteria", "protocol_sha256", "protocol_version", "read_policy",
        "planned_contexts", "planned_context_target_modality_cells", "context_status_counts", "cell_status_counts",
        "attempt_status_counts", "criterion_status_counts", "auditor_sha256", "sealed_membership", "repair_lineage")}
    result.update(schema_version=1, source_audit_sha256=AUDIT_SHA,
        cutoff_label="2026-10-01 03:36 UTC sealed audit snapshot",
        cutoff_qualification="A recorded snapshot label, not a final stop-time census or a per-job completion timestamp.",
        final_stop_completion_state="later_completion_unknown_and_scientifically_unassessed_here",
        rows=[{k: r[k] for k in r if k != "criterion_audit"} for r in audit["rows"]])
    collection_path = evidence.parent / "E01_coverage_bank_v4.json"
    collection = read(collection_path, "882d6303e7e7f0f224e3ae324382d9c461e9ba8a9d0089f5c2b9493c9de3092c")
    result["source_collection_protocol"] = dict(sha256=digest(collection_path),
        **{k: collection[k] for k in ("selection", "sampling_precision", "strata", "collection", "solver_steps", "failure_rule")})
    protocol_path = evidence.parent / "E01_exact_fp32_probe_v6.json"
    result["numerical_protocol"] = read(protocol_path, audit["protocol_sha256"])
    # Earlier snapshots are overlapping observations, not additional replicates.
    result["earlier_snapshots"] = []
    current = {(r["stratum"], r["episode"], r["call"], r["modality"], r["target"]): r for r in audit["rows"]}
    for label, (name, sha) in EARLIER.items():
        old = read(evidence / name, sha)
        comparisons = []
        if label.startswith("v6"):
            for row in old["rows"]:
                if row["status"] == "complete_diagnostics":
                    now = current[tuple(row[k] for k in ("stratum", "episode", "call", "modality", "target"))]
                    if now["report_sha256"] != row["report_sha256"]:
                        raise ValueError("Previously complete report changed")
                    comparisons.append(row["report_sha256"])
        result["earlier_snapshots"].append(dict(label=label, audit_sha256=sha,
            context_status_counts=old["context_status_counts"], cell_status_counts=old["cell_status_counts"],
            earlier_complete_report_hashes_unchanged=comparisons,
            same_estimand=label.startswith("v6")))
    snapshot = evidence / "numerical-audit-snapshot-20261001T011638Z"
    result["queues"] = []
    bank_population = {}
    for group in ("numerics-probe-v6", "numerics-probe-v6-pathfix", "numerics-probe-v6-numpyfix"):
        path = snapshot / group / "queue.json"
        queue = read(path)
        if queue["protocol_sha256"] != audit["protocol_sha256"]:
            raise ValueError("Queue protocol mismatch")
        queue_record = dict(label=group, sha256=digest(path), source_sha256=queue["source_sha256"], jobs=[])
        for job in queue["jobs"]:
            bank_path = snapshot / group / "banks" / Path(job["bank_file"]).name
            bank = read(bank_path, job["bank_sha256"])
            stratum = Path(job["bank_file"]).stem
            context = next(c for c in bank["contexts"] if
                f"{stratum}-e{c['episode']:03d}-c{c['policy_call_idx']:03d}" == job["job_id"])
            if context["context_id"] != job["context_id"]:
                raise ValueError("Queue job and bank context identity mismatch")
            key = (stratum, context["episode"], context["policy_call_idx"])
            population = dict(stratum=stratum, bank_sha256=job["bank_sha256"],
                source_metrics_sha256=bank["source_metrics_sha256"], source_run_id=bank["source_run_id"],
                selection=bank["collector_call_selection"], **context)
            # Relocation changes some bank paths/digests, not the planned context identity.
            if key in bank_population and bank_population[key]["context_id"] != context["context_id"]:
                raise ValueError("Bank context changed across queue relocation")
            bank_population[key] = population
            queue_record["jobs"].append({k: job[k] for k in
                ("job_id", "context_id", "planned_status", "bank_sha256", "decision_sha256")})
        result["queues"].append(queue_record)
    result["bank_population"] = [bank_population[k] for k in sorted(bank_population)]
    score_source = subprocess.check_output(["git", "show", "d9e8901:paired_comparison.py"], cwd=source_repo)
    score_sha = hashlib.sha256(score_source).hexdigest()
    if score_sha != result["queues"][-1]["source_sha256"]["paired_comparison.py"]:
        raise ValueError("Saved numerical source differs from inspected Git blob")
    text = score_source.decode("utf-8")
    start = text.index("def common_scores(")
    snippet = text[start:text.index("\ndef ", start + 1)].strip()
    if '"Q": -squared / (2 * difference.numel())' not in snippet:
        raise ValueError("Score normalization definition changed")
    result["score_definition"] = dict(source_git_revision="d9e8901", source_blob_sha256=score_sha,
        function=snippet, sigma_squared=1.0, qualification="Unit sigma squared is the exact pinned source definition, not a fitted parameter.")
    ycb = evidence / "ycb_path_diagnosis_v1"
    download = read(ycb / "download_manifest.json")
    if download["audit_sha256"] != AUDIT_SHA:
        raise ValueError("YCB provenance refers to a different audit")
    for name, item in download["files"].items():
        read(ycb / name, item["sha256"])
        if audit["source_files_sha256"].get(item["source"]) != item["sha256"]:
            raise ValueError("YCB report is not authenticated by sealed audit")
    raw_manifest = read(ycb / "raw_download_manifest.json")
    projection = read(ycb / "state_tensor_projection.json")
    result["ycb_saved_tensors"] = []
    for call in (0, 12):
        directory = ycb / f"picksingleycb-170m-e001-c{call:03d}"
        top = read(directory / "report.json")
        runtime_contract = {k: top[k] for k in ("forward_precision", "settings_transition", "startup", "math_settings")}
        if "runtime_contract" in result and result["runtime_contract"] != runtime_contract:
            raise ValueError("Representative saved contexts have different runtime contracts")
        result["runtime_contract"] = runtime_contract
        context = top["contexts"][0]
        active_reference = torch.tensor(context["probe_identity"]["probe_reference_active_values"], dtype=torch.float32)
        active_indices = context["probe_identity"]["active_action_indices"]
        if list(active_reference.shape) != [1,64,8] or len(active_indices) != 8:
            raise ValueError("Unexpected saved active action support")
        normalization = dict(active_reference_shape=list(active_reference.shape), active_indices=active_indices,
            active_entries=active_reference.numel(), sigma_squared=result["score_definition"]["sigma_squared"],
            reference_sha256=context["probe_identity"]["probe_reference_sha256"],
            shape_source="authenticated report probe_reference_active_values",
            score_source_sha256=score_sha)
        source = context["probe_context_artifact"]
        projected = projection["projection"][str(call)]
        if projected["source_file_sha256"] != source["sha256"]:
            raise ValueError("Projected state has wrong source artifact")
        inputs = {k: tensor_record(torch.tensor(v, dtype=torch.float32), source["tensor_sha256"][k])
                  for k, v in projected["tensors"].items()}
        for target in ("Q", "L2"):
            parent = directory / f"ep000001-call{call:04d}" / f"state-{target}"
            report = read(parent / "report.json")
            tensors = []
            for item in report["gradient_probes"] + report["budgets"]:
                path = parent / item["file"]
                if digest(path) != item["sha256"]:
                    raise ValueError("Saved raw artifact hash mismatch")
                matches = [v for v in raw_manifest["artifacts"].values()
                    if (ycb / v["local"]).resolve() == path.resolve()]
                if len(matches) != 1 or matches[0]["sha256"] != item["sha256"]:
                    raise ValueError("Raw download receipt mismatch")
                saved = torch.load(path, map_location="cpu", weights_only=True)
                tensors.append({k: item[k] for k in ("file", "sha256", "repeat", "alpha", "score", "gradient_l1", "m", "diagnostics") if k in item} |
                    {"tensors": {k: tensor_record(v, item["tensor_sha256"][k]) for k, v in saved.items()}})
            result["ycb_saved_tensors"].append(dict(call=call, target=target,
                report_sha256=digest(parent / "report.json"), source_context_sha256=source["sha256"],
                state_inputs=inputs, tensors=tensors, distance_normalization=normalization))
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU extraction unexpectedly initialized CUDA")
    result["local_verification"] = dict(verified_ycb_reports=len(download["files"]),
        verified_raw_ycb_tensor_files=sum(len(r["tensors"]) for r in result["ycb_saved_tensors"]),
        full_cohort_raw_artifacts="verified by original sealed auditor; not all raw files are locally present",
        ycb_context_projection="state tensor hashes match sealed full-context tensor metadata; full44MB context files are not copied")
    return clean(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    args = parser.parse_args()
    value = project(args.evidence, args.source_repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output_sha256": digest(args.output), "rows": len(value["rows"]), **value["local_verification"]}))
