"""Descriptive RMS sensitivity to authenticated executed source-command positions.

Reads completed paired artifacts only. No model execution, primary replacement,
confidence intervals, pilot efficacy disclosure, or guessed temporal membership.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import math

import numpy as np
from scipy.integrate import trapezoid
import torch

from experiment_io import canonical_json, file_hash, object_hash, strict_json, tensor_hash

COLLECTOR_SHA256 = "9bd7a95b3fd48b624f2a79a35c89eef46fddeeab023e0514674efc575fad9bf7"
ACTIVE = [0, 1, 2, 3, 4, 5, 6, 10]
SUPPORTS = ("full_chunk", "executed_source_positions")


def validate_protocol(protocol):
    canonical_json(protocol)
    if protocol.get("kind") != "paired_execution_support_sensitivity_protocol" or protocol.get("schema_version") != 1:
        raise ValueError("Unknown sensitivity protocol")
    if protocol.get("recorded_before_comparative_inspection") is not True:
        raise ValueError("A prospective calculation protocol is required")
    if protocol.get("stage") != "descriptive_saved_action_analysis_only":
        raise ValueError("Sensitivity cannot change the inferential stage")
    if protocol.get("supported_collector_sha256") != [COLLECTOR_SHA256]:
        raise ValueError("Collector mapping has not been independently reviewed")
    expected = dict(mode="authenticated_executed_source_positions_v1", predicted_chunk_length=64,
                    action_subsampling=4, active_indices=ACTIVE)
    if any(protocol["membership"].get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported execution membership definition")
    if protocol["drift"].get("action_ratio_minimum") != 1e-8 or protocol["drift"].get("state_ratio_minimum") != 0.:
        raise ValueError("Drift denominator rules differ from the reviewed calculation")


def executed_membership(rows, terminals, configuration):
    """Infer commands submitted to env.step, never physical robot motion."""
    if configuration.get("collector_type") != "context_only" or configuration.get("control_mode") != "pd_joint_pos":
        raise ValueError("Unsupported source collector/controller")
    if configuration.get("action_subsampling") != 4 or configuration.get("source_sha256", {}).get("collect_contexts.py") != COLLECTOR_SHA256:
        raise ValueError("Source command mapping is not the reviewed collector")
    if any(type(configuration.get(k)) is not int or configuration[k] < 1
           for k in ("episodes", "max_policy_calls", "max_episode_steps")):
        raise ValueError("Invalid declared collector limits")
    if set(terminals) != set(range(configuration["episodes"])):
        raise ValueError("Incomplete committed episode roster")
    result = {}
    for episode, terminal in sorted(terminals.items()):
        if terminal.get("episode") != episode or type(terminal.get("policy_calls")) is not int:
            raise ValueError("Terminal episode or call identity differs")
        if any(type(terminal.get(k)) is not bool for k in ("terminated", "truncated", "collector_step_cap")):
            raise ValueError("Terminal stop flags must be explicit booleans")
        calls = sorted((r for r in rows if r["episode"] == episode), key=lambda r: r["policy_call_idx"])
        if not calls or any(type(r["policy_call_idx"]) is not int for r in calls) or [r["policy_call_idx"] for r in calls] != list(range(len(calls))):
            raise ValueError("Every consecutive executed source call is required")
        starts = [r["env_step_at_call"] for r in calls]
        total = terminal["env_steps"]
        if any(type(x) is not int or x < 0 for x in starts + [total]) or starts[0] != 0:
            raise ValueError("Invalid source step counters")
        if terminal["policy_calls"] != len(calls) or not 1 <= len(calls) <= configuration["max_policy_calls"]:
            raise ValueError("Terminal call count differs")
        if not 1 <= total <= configuration["max_episode_steps"]:
            raise ValueError("Terminal exceeds the declared horizon")
        if terminal["collector_step_cap"] != (total == configuration["max_episode_steps"]):
            raise ValueError("Terminal step-cap flag contradicts the executed horizon")
        reason = terminal.get("stop_reason")
        expected_reason = ("terminated" if terminal.get("terminated") else "truncated" if terminal.get("truncated")
                           else "collector_environment_step_cap" if terminal.get("collector_step_cap") else "protocol_call_limit")
        if reason != expected_reason:
            raise ValueError("Terminal stop reason contradicts its flags")
        if reason == "collector_environment_step_cap" and total != configuration["max_episode_steps"]:
            raise ValueError("Collector step cap was not reached")
        if reason == "protocol_call_limit" and len(calls) != configuration["max_policy_calls"]:
            raise ValueError("Protocol call cap was not reached")
        for index, row in enumerate(calls):
            end = starts[index + 1] if index + 1 < len(calls) else total
            count = end - starts[index]
            if not 1 <= count <= 16 or (index + 1 < len(calls) and count != 16):
                raise ValueError("Source counters contradict the stride-four execution loop")
            if index + 1 == len(calls) and reason == "protocol_call_limit" and count != 16:
                raise ValueError("A call-limit stop cannot interrupt its action chunk")
            key = (episode, row["policy_call_idx"])
            result[key] = dict(mode="authenticated_executed_source_positions_v1",
                predicted_step_positions=list(range(0, 4 * count, 4)), executed_commands=count,
                start_env_step=starts[index], end_env_step=end, source_row_sha256=object_hash(row),
                terminal_record_sha256=object_hash(terminal),
                interpretation="Source command positions submitted to env.step; not actual motion or counterfactual probe execution.")
    if len(result) != len(rows):
        raise ValueError("Source contains duplicate or foreign episode rows")
    return result


def rms_on_support(action, reference, positions):
    a, b = np.asarray(action, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != (1, 64, 8) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Saved active actions must be finite [1,64,8] tensors")
    if not positions or positions != sorted(set(positions)) or any(type(i) is not int or not 0 <= i < 64 for i in positions):
        raise ValueError("Invalid predicted-step support")
    return float(np.sqrt(np.mean((a[:, positions, :] - b[:, positions, :]) ** 2)))


def guarded_ratio(numerator, denominator, minimum):
    if numerator is None or denominator is None:
        return dict(status="unavailable", value=None, numerator=numerator, denominator=denominator, minimum=minimum)
    if any(not math.isfinite(x) or x < 0 for x in (numerator, denominator, minimum)):
        raise ValueError("Drift ratio arguments must be finite and nonnegative")
    status = "zero_denominator" if denominator == 0 else "nearzero_denominator" if denominator <= minimum else "defined"
    return dict(status=status, value=numerator / denominator if status == "defined" else None,
                numerator=numerator, denominator=denominator, minimum=minimum)


def reproject_results(results, positions):
    """Reuse exact mask membership; average responses after taking each RMS."""
    reference = results["active_reference_action"]
    table = results["response_table"]
    if any(item.get("status") not in {"finite", "numerical_failure"} for item in table.values()):
        raise ValueError("Unknown saved response outcome")
    supports = dict(full_chunk=list(range(64)), executed_source_positions=positions)
    output = {}
    for support, indices in supports.items():
        values = {key: None if item["status"] == "numerical_failure" else
                  rms_on_support(item["active_action"], reference, indices) for key, item in table.items()}
        curves = {}
        for label, ranking in results["rankings"].items():
            if ranking["status"] != "defined":
                curves[label] = dict(status=ranking["status"], failure_kind=ranking.get("failure_kind"))
                continue
            exact = ranking.get("control_kind") == "exact_uniform_subset_expectation"
            directions = {}
            for direction in ("deletion", "insertion"):
                original = ranking["curves"][direction]
                groups = original["response_ids"] if exact else [[key] for key in original["response_ids"]]
                curve = [None if any(values[key] is None for key in group) else
                         math.fsum(values[key] for key in group) / len(group) for group in groups]
                area = None if any(v is None for v in curve) else float(trapezoid(curve, results["realized_fractions"]))
                directions[direction] = dict(values=curve, raw_auc=area,
                    status="nonfinite_response" if area is None else "defined",
                    primary_saved_full_chunk_fp32_raw_auc=original["responses"]["RMS"]["raw_auc"])
            curves[label] = dict(status="defined", curves=directions)
        output[support] = dict(predicted_step_positions=indices, active_entries=8 * len(indices),
            response_rms=values, rankings=curves, baseline_rms=values[results["baseline_response_id"]],
            actual_rms=values[results["input_response_id"]], realized_fractions=results["realized_fractions"])
    return output


def drift_report(row, payload, projected, positions, *, source_state_norm=None):
    provenance = row.get("probe_identity")
    if not provenance:
        raise ValueError("The declared offline probe needs source-to-probe identity")
    reference = payload["ref_action"]
    if tuple(reference.shape) != (1, 64, 128) or tensor_hash(reference) != provenance["source_reference_sha256"]:
        raise ValueError("Source reference differs from the probe provenance")
    actual = row["results"]["active_reference_action"]
    if actual != provenance["probe_reference_active_values"] or provenance["active_action_indices"] != ACTIVE:
        raise ValueError("Probe reference or active coordinates differ")
    source = reference[..., ACTIVE].float().tolist()
    by_support = {}
    for name, indices in (("full_chunk", list(range(64))), ("executed_source_positions", positions)):
        drift = rms_on_support(actual, source, indices)
        by_support[name] = dict(action_drift_rms_float64=drift,
            drift_over_all_baseline_rms=guarded_ratio(drift, projected[name]["baseline_rms"], 1e-8))
    return dict(by_support=by_support,
        producer_reported={k: provenance[k] for k in ("reference_drift_l2", "reference_drift_rms", "reference_drift_max_abs", "state_token_drift_l2")},
        state_drift_over_source_norm=guarded_ratio(provenance["state_token_drift_l2"], source_state_norm, 0.),
        state_ratio_arithmetic="Producer-reported FP32 L2 numerator / authenticated source-token float64 L2 denominator.",
        interpretation="Descriptive endpoint/state drift; no derivative, rank or behavior equivalence follows.")


def descriptive_summaries(original_rows, derived_rows, bank):
    import paired_study_analysis as study
    primary = study.stratum_data(original_rows, bank)
    membership = primary["membership"]
    retained = {item["episode_id"] for item in membership if item["joint_complete"]}
    lookup = {(r["episode_id"], r["policy_call_idx"], r["modality"]): r for r in derived_rows}
    summaries = []
    for contrast in study.LOCAL_CONTRASTS:
        episodes = []
        for member in membership:
            if member["episode_id"] not in retained:
                continue
            calls = []
            for entry in member["contexts"]:
                row = lookup[(member["episode_id"], entry["call"], contrast["modality"])]
                effect = {}
                for support in SUPPORTS:
                    rankings = row["supports"][support]["rankings"]
                    random = ["random_exact"] if "random_exact" in rankings else sorted(k for k in rankings if k.startswith("random_"))
                    controls = random if contrast["control"] == "random" else [contrast["control"]]
                    area = lambda label: rankings[label]["curves"][contrast["direction"]]["raw_auc"]
                    sign = 1 if contrast["direction"] == "deletion" else -1
                    effect[support] = sign * (area(contrast["method"]) - math.fsum(area(c) for c in controls) / len(controls))
                calls.append(effect)
            values = {support: math.fsum(c[support] for c in calls) / len(calls) for support in SUPPORTS}
            episodes.append(dict(episode_id=member["episode_id"], selected_calls=len(calls), **values,
                support_effect_difference=values[SUPPORTS[1]] - values[SUPPORTS[0]]))
        means = {support: math.fsum(e[support] for e in episodes) / len(episodes) if episodes else None for support in SUPPORTS}
        summaries.append(dict(contrast=contrast, conditional_episodes=len(episodes), means=means, episodes=episodes,
            mean_support_effect_difference=means[SUPPORTS[1]] - means[SUPPORTS[0]] if episodes else None))
    return dict(planned_episodes=primary["planned_episodes"], planned_contexts=primary["planned_contexts"],
        conditioning=primary["conditioning"], membership=membership, failure_counts=primary["failure_counts"],
        contrasts=summaries, inference="Descriptive stratum summaries only; unchanged primary joint-complete membership, no intervals or tests.")


def analyze(study_path, source_path, protocol_path, *, study_sha256, study_completion_sha256,
            source_sha256, protocol_sha256, prepared=None):
    import paired_comparison as paired
    import paired_study_analysis as study
    import faithfulness
    import fp32_probe_cache
    paths = {Path(study_path): study_sha256, Path(source_path): source_sha256, Path(protocol_path): protocol_sha256}
    paths[Path(str(study_path) + ".completion.json")] = study_completion_sha256
    for path, digest in paths.items():
        if file_hash(path) != digest:
            raise ValueError("Trusted input hash differs: " + str(path))
    helpers = {Path(module.__file__): file_hash(module.__file__) for module in (paired, study, faithfulness, fp32_probe_cache)}
    helpers[Path(__file__)] = file_hash(__file__)
    protocol = strict_json(protocol_path)
    validate_protocol(protocol)
    rows, manifest, completion = paired.load_completed_study(study_path)
    config = manifest["configuration"]
    if config["protocol"].get("stage") != "confirmatory_locked":
        raise ValueError("Do not disclose signed sensitivity outcomes from a variance-only pilot")
    if config["protocol"].get("forward_precision") != paired.FP32_PROBE:
        raise ValueError("This protocol concerns the declared offline FP32 probe")
    bank = config["bank"]
    source_rows, source_manifest = faithfulness.authenticated_source(source_path)
    if (manifest["source_metrics_sha256"] != source_sha256 or
            manifest["source_run_id"] != source_manifest["run_id"] or
            manifest["source_configuration_sha256"] != source_manifest["configuration_sha256"]):
        raise ValueError("Paired artifact names another source export or identity")
    selected = paired.bank_rows(bank, source_rows, source_manifest, source_sha256)
    rebuilt = paired.make_bank(source_path, bank["selection"])
    if rebuilt != bank:
        raise ValueError("Frozen bank differs from the complete authenticated collector")
    root = Path(str(source_path) + ".run")
    terminals = {d["episode"]: d["records"][-1] for d in (strict_json(p) for p in (root / "episodes").glob("ep*.json"))}
    positions = executed_membership(source_rows, terminals, source_manifest["configuration"])
    preparation = fp32_probe_cache.verify_preparation(prepared, config["prepared_completion_sha256"]) if prepared else None
    entries = {(e["episode"], e["policy_call_idx"]): e for e in bank["contexts"]}
    caches = {}
    derived = []
    for row in rows:
        common = {k: row[k] for k in ("episode_id", "policy_call_idx", "modality", "source_context_id", "status")}
        common["paired_row_sha256"] = object_hash(row)
        key = (row["episode"], row["policy_call_idx"])
        entry = entries[key]
        if (row["episode_id"], row["source_context_id"]) != (entry["episode_id"], entry["context_id"]):
            raise ValueError("Paired row and executed source context identity differ")
        member = positions[key]
        if member["terminal_record_sha256"] != entry["terminal_record_sha256"] or member["source_row_sha256"] != entry["row_sha256"]:
            raise ValueError("Execution membership differs from the frozen source context")
        if key not in caches:
            payload, _ = faithfulness.load_sidecar(selected[key], source_path, source_manifest)
            if tuple(payload["ref_action"].shape) != (1, 64, 128):
                raise ValueError("Source reference shape cannot authenticate the declared command positions")
            state_norm = None
            cache_file = state_hash = None
            if preparation:
                cache, record = fp32_probe_cache.load_prepared_context(prepared, preparation, entry, source_manifest, selected[key], payload)
                cache_file = record["cache_file"]
                state_hash = tensor_hash(cache["state_traj_actual"])
                state_norm = float(cache["state_traj_actual"].double().norm().item())
                del cache
            # Images and other cached inputs are unnecessary after authentication.
            caches[key] = {"ref_action": payload["ref_action"]}, state_norm, cache_file, state_hash
            del payload
        payload, state_norm, cache_file, state_hash = caches[key]
        if row["status"] != "evaluated":
            derived.append(dict(**common, membership=member, failure_kind=row.get("failure_kind"),
                reason=row.get("reason"), supports=None))
            continue
        if preparation:
            if row["probe_identity"]["prepared_cache"] != cache_file:
                raise ValueError("Probe uses another prepared cache")
            if row["probe_identity"]["cached_source_state_token_sha256"] != state_hash:
                raise ValueError("State denominator differs from the source token")
        supports = reproject_results(row["results"], member["predicted_step_positions"])
        derived.append(dict(**common, membership=member, supports=supports,
            drift=drift_report(row, payload, supports, member["predicted_step_positions"], source_state_norm=state_norm)))
    summary = descriptive_summaries(rows, derived, bank)
    # Re-authenticate immutable stores, not merely their convenient export names.
    paired.load_completed_study(study_path)
    faithfulness.authenticated_source(source_path)
    if prepared:
        fp32_probe_cache.verify_preparation(prepared, config["prepared_completion_sha256"])
    for path, digest in {**paths, **helpers}.items():
        if file_hash(path) != digest:
            raise ValueError("Input or analysis source changed during calculation")
    return dict(schema_version=1, kind="descriptive_execution_support_sensitivity", protocol_sha256=protocol_sha256,
        study_sha256=study_sha256, source_metrics_sha256=source_sha256,
        study_completion_sha256=study_completion_sha256,
        study_manifest_sha256=file_hash(str(study_path) + ".manifest.json"),
        collector_sha256=COLLECTOR_SHA256, helper_sha256={str(p): h for p, h in helpers.items()},
        prepared_completion_sha256=config["prepared_completion_sha256"] if prepared else None,
        planned_context_modality_rows=len(rows), row_status_counts=dict(Counter(r["status"] for r in derived)),
        rows=derived, summary=summary,
        interpretation="Fixed-ranking saved-action support sensitivity; source command positions, not closed-loop behavioral effects. Primary inference is unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("study", "source", "protocol"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--study-completion-sha256", required=True)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    result = analyze(args.study, args.source, args.protocol, study_sha256=args.study_sha256,
        study_completion_sha256=args.study_completion_sha256,
        source_sha256=args.source_sha256, protocol_sha256=args.protocol_sha256, prepared=args.prepared)
    with args.out.open("xb") as stream:
        stream.write(canonical_json(result) + b"\n")


if __name__ == "__main__":
    main()
