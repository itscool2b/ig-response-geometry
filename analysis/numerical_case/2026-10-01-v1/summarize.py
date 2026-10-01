"""Reproduce descriptive numerical tables from saved data, using stdlib only.

This script does not evaluate a model, select a budget, or approve a method.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

INPUT_SHA256 = "c5c265b3cfaf38d76d3de99c4db1b57b6177f74cfbaa5131b8bf9e49ed117098"
ARMS = {(m, t) for m in ("vision", "language", "state") for t in ("Q", "L2")}


def tensor_values(record):
    values = record["values"]
    if record["dtype"] != "torch.float32" or math.prod(record["shape"]) != len(values):
        raise ValueError("Invalid tensor projection")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite value in finite diagnostic projection")
    header = {k: record[k] for k in ("dtype", "shape")}
    sha = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    sha.update(struct.pack("<" + "f" * len(values), *values))
    if sha.hexdigest() != record["sha256"]:
        raise ValueError("Tensor projection does not reproduce original tensor hash")
    return values


def key(row):
    return row["stratum"], row["episode"], row["call"]


def max_or_none(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def validate_roster(data):
    groups = defaultdict(list)
    for row in data["rows"]:
        groups[key(row)].append(row)
    bank = {(r["stratum"], r["episode"], r["policy_call_idx"]): r for r in data["bank_population"]}
    if len(bank) != len(data["bank_population"]) or set(bank) != set(groups):
        raise ValueError("Audit and planned bank membership differ")
    for identity, rows in groups.items():
        if len(rows) != 6 or {(r["modality"], r["target"]) for r in rows} != ARMS:
            raise ValueError("Every planned context must retain exactly all six arms")
        if any(("context_id" in r and r["context_id"] != bank[identity]["context_id"]) or r["planned_status"] != bank[identity]["status"] for r in rows):
            raise ValueError("Context identity or planned availability mismatch")
        if len({r["status"] for r in rows}) != 1:
            raise ValueError("This sealed snapshot requires uniform six-arm context status")
    if len(groups) != data["planned_contexts"] or len(data["rows"]) != data["planned_context_target_modality_cells"]:
        raise ValueError("Recorded denominator does not match full membership")
    return groups


def summarize(data):
    groups = validate_roster(data)
    bank = {(r["stratum"], r["episode"], r["policy_call_idx"]): r for r in data["bank_population"]}
    roster, strata, per_cell = [], [], []
    completed = [r for r in data["rows"] if r["status"] == "complete_diagnostics"]
    for identity, rows in sorted(groups.items()):
        row = rows[0]
        roster.append(dict(stratum=identity[0], episode=identity[1], call=identity[2],
            context_id=bank[identity]["context_id"], source_status=row["planned_status"], snapshot_status=row["status"],
            planned_cells=6, complete_cells=sum(r["status"] == "complete_diagnostics" for r in rows),
            stop_time_status="not_assessed_by_this_snapshot"))
    for stratum in sorted({r["stratum"] for r in roster}):
        records = [r for r in roster if r["stratum"] == stratum]
        counts = Counter(r["snapshot_status"] for r in records)
        strata.append(dict(stratum=stratum, planned_contexts=len(records),
            completed_contexts=counts["complete_diagnostics"], source_unavailable_contexts=counts["call_not_reached"],
            unsealed_contexts=counts["unsealed_at_audit_start"], planned_cells=len(records)*6,
            completed_cells=counts["complete_diagnostics"]*6, source_unavailable_cells=counts["call_not_reached"]*6,
            unsealed_cells=counts["unsealed_at_audit_start"]*6))
    all_checks = [c for r in completed for c in r["checks"]]
    counts = Counter(c["status"] for c in all_checks)
    if dict(counts) != data["criterion_status_counts"]:
        raise ValueError("Criterion counts do not reproduce the sealed audit")
    equality = [c for c in all_checks if "repeatability" in c["criterion"]]
    repeated_conditions = [c for c in all_checks if "repeat" in c["criterion"]]
    for row in completed:
        budgets = sorted({o["m"] for o in row["observations"]})
        if len(budgets) != 3:
            raise ValueError("Unexpected budget ladder")
        coarse, fine = budgets[-2:]
        comparisons = [c for c in row["checks"] if c.get("candidate_m") == coarse and c.get("reference_m") == fine]
        finest = [o for o in row["observations"] if o["m"] == fine and o["repeat"] == 0]
        if len(finest) != 1:
            raise ValueError("Ambiguous finest diagnostic")
        finest = finest[0]
        coord = {c["ranking"]: c for c in comparisons if c["criterion"] == "coordinate_relative_l1"}
        if set(coord) != {"IG", "path_gradient"}:
            raise ValueError("Missing finer-pair coordinate comparison")
        rms = [c for c in comparisons if c["criterion"] == "rms_curve_difference"]
        residual = finest["relative_residual"]
        violations = [c for c in comparisons if c["status"] == "violates"]
        per_cell.append(dict(stratum=row["stratum"], episode=row["episode"], call=row["call"],
            context_id=row["context_id"], modality=row["modality"], target=row["target"],
            candidate_m=coarse, reference_m=fine, report_sha256=row["report_sha256"],
            coordinate_relative_l1_IG=coord["IG"]["value"], coordinate_absolute_l1_IG=coord["IG"]["absolute_l1_difference"],
            reference_coordinate_l1_IG=coord["IG"]["reference_l1_inferred_from_reported_ratio"],
            coordinate_relative_l1_path_gradient=coord["path_gradient"]["value"],
            finest_signed_gap=finest["expected_gap"], finest_signed_sum=finest["ig_sum"],
            finest_absolute_residual=finest["absolute_residual"], finest_relative_residual=residual,
            finest_completeness_over_recorded_threshold=None if residual is None else residual > data["criteria"]["residual"],
            max_rms_curve_absolute_difference=max_or_none(c["value"] for c in rms),
            max_rms_curve_difference_over_baseline=max_or_none(c["value"] / c["baseline_rms"] for c in rms if c["baseline_rms"] > 0),
            finest_pair_has_threshold_violation=bool(violations), finest_pair_violation_checks=len(violations),
            full_ladder_violation_checks=sum(c["status"] == "violates" for c in row["checks"])))
    aggregate = []
    for modality, target in sorted(ARMS):
        rows = [r for r in per_cell if r["modality"] == modality and r["target"] == target]
        aggregate.append(dict(modality=modality, target=target, completed_contexts=len(rows),
            contexts_with_finest_pair_threshold_violation=sum(r["finest_pair_has_threshold_violation"] for r in rows),
            contexts_with_finest_completeness_over_threshold=sum(r["finest_completeness_over_recorded_threshold"] is True for r in rows),
            max_finer_pair_coordinate_relative_l1_IG=max_or_none(r["coordinate_relative_l1_IG"] for r in rows),
            max_finest_relative_residual=max_or_none(r["finest_relative_residual"] for r in rows),
            max_rms_curve_difference_over_baseline=max_or_none(r["max_rms_curve_difference_over_baseline"] for r in rows)))
    attempts = {}
    for row in data["rows"]:
        for attempt in row["attempts"]:
            identity = attempt["queue_sha256"], attempt["job_id"]
            if identity in attempts and attempts[identity] != attempt:
                raise ValueError("Inconsistent repeated attempt metadata")
            attempts[identity] = attempt
    attempt_counts = dict(Counter(a["status"] for a in attempts.values()))
    if attempt_counts != data["attempt_status_counts"]:
        raise ValueError("Attempt deduplication changed the recorded counts")
    return dict(kind="descriptive_saved_snapshot_not_numerical_approval", cutoff_label=data["cutoff_label"],
        source_audit_sha256=data["source_audit_sha256"], final_stop_completion_state=data["final_stop_completion_state"],
        planned_contexts=len(roster), available_contexts=sum(r["source_status"] == "available" for r in roster),
        completed_contexts=sum(r["snapshot_status"] == "complete_diagnostics" for r in roster),
        completed_episodes=len({(r["stratum"], r["episode"]) for r in completed}),
        planned_cells=len(data["rows"]), completed_cells=len(completed),
        cell_status_counts=dict(Counter(r["status"] for r in data["rows"])),
        criterion_status_counts=dict(counts), repeatability_equality_check_counts=dict(Counter(c["status"] for c in equality)),
        repeated_condition_and_coverage_check_counts=dict(Counter(c["status"] for c in repeated_conditions)),
        repeat_design=dict(completed_arms=len(completed), fixed_alpha_points_per_arm=7, fixed_alpha_repeats=3,
            finest_map_repeats=2, endpoint_repeats=3,
            qualification="These repeat and coverage checks share contexts, tensors, masks and criteria; they are not independent samples."),
        distinct_attempt_status_counts=attempt_counts, roster=roster, strata=strata, per_cell=per_cell,
        modality_target_extrema=aggregate,
        attempts=[attempts[k] for k in sorted(attempts)],
        qualification="All original checks remain in the input. A conditional completeness check only applies if that budget is selected. No budget is selected here; the finer map is a comparator, not truth.")


def diagnose_ycb(data):
    rows, probes, geometry = [], [], []
    gradients = {}
    normalizations = {}
    for record in data["ycb_saved_tensors"]:
        call, target = record["call"], record["target"]
        normalization = record["distance_normalization"]
        if (math.prod(normalization["active_reference_shape"]) != normalization["active_entries"] or
            normalization["active_reference_shape"][-1] != len(normalization["active_indices"]) or
            normalization["score_source_sha256"] != data["score_definition"]["source_blob_sha256"] or
            normalization["sigma_squared"] != data["score_definition"]["sigma_squared"]):
            raise ValueError("Saved score normalization provenance mismatch")
        normalizations[call] = normalization
        actual = tensor_values(record["state_inputs"]["state_input_actual"])
        baseline = tensor_values(record["state_inputs"]["state_input_baseline"])
        delta = [a-b for a,b in zip(actual,baseline)]
        for saved in record["tensors"]:
            values = {k: tensor_values(v) for k,v in saved["tensors"].items()}
            if "alpha" in saved:
                g = values["gradient"]
                if not math.isclose(math.fsum(abs(v) for v in g), saved["gradient_l1"], rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError("Saved gradient norm does not reproduce report")
                if saved["repeat"] == 0:
                    gradients[(call,target,saved["alpha"])] = (g, saved["score"])
                    probes.append(dict(call=call,target=target,alpha=saved["alpha"],score=saved["score"],
                        gradient_l1=saved["gradient_l1"], directional_derivative=math.fsum(d*v for d,v in zip(delta,g)),
                        gradient_tensor_sha256=saved["tensors"]["gradient"]["sha256"]))
            elif saved["repeat"] == 0:
                diag = saved["diagnostics"]
                product_error = max(abs(a-d*g) for a,d,g in zip(values["attribution"],delta,values["path_gradient"]))
                rows.append(dict(call=call,target=target,m=saved["m"],signed_gap=diag["expected_gap"],
                    signed_ig_sum=diag["ig_sum"],absolute_residual=diag["absolute_residual"],relative_residual=diag["relative_residual"],
                    coordinate_l1=math.fsum(abs(v) for v in values["attribution"]),
                    path_gradient_l1=math.fsum(abs(v) for v in values["path_gradient"]),
                    max_product_discrepancy_fp64_reconstruction=product_error,
                    report_sha256=record["report_sha256"],raw_artifact_sha256=saved["sha256"]))
    for (call,target,alpha), (q,score) in sorted(gradients.items()):
        if target != "Q":
            continue
        l, lscore = gradients[(call,"L2",alpha)]
        normalization = normalizations[call]
        factor = -lscore / (normalization["active_entries"] * normalization["sigma_squared"])
        absolute = math.fsum(abs(qi-factor*li) for qi,li in zip(q,l))
        denom = math.fsum(abs(qi) for qi in q)
        geometry.append(dict(call=call,alpha=alpha,expected_positive_gradient_multiplier=factor,
            gradient_identity_absolute_l1=absolute,gradient_identity_relative_l1=None if denom==0 else absolute/denom,
            zero_gradient_endpoint=denom==0))
    return dict(scope="Two saved contexts from one YCB episode, not independent episode replicates.",
        model_evaluations=0, quadrature_nodes_recomputed=0,
        budget_accounting=rows,fixed_probes=probes,shared_distance_gradient_identity=geometry,
        distance_normalizations=[dict(call=k, **normalizations[k]) for k in sorted(normalizations)],
        qualification="Sparse saved probes and budget differences diagnose unresolved numerical sensitivity; they do not locate all unsaved path features or identify a unique cause.")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(summary, diagnosis):
    lines = ["# Preserved FP32 probe diagnostics", "",
        f"Source snapshot: {summary['cutoff_label']}. This is the last fully audited local snapshot used here, not a final census at the later compute stop.", "",
        f"The full roster has {summary['planned_contexts']} contexts and {summary['planned_cells']} target/modality cells. Of 21 available contexts, 12 have all six diagnostics complete, representing six episodes from four 170M tasks. Three planned late calls were never reached. Nine available contexts remain unsealed at this cutoff; their later completion status is unknown here.", "",
        "| Stratum | Planned contexts | Complete | Source unavailable | Unsealed | Complete/planned cells |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in summary["strata"]:
        lines.append(f"| {r['stratum']} | {r['planned_contexts']} | {r['completed_contexts']} | {r['source_unavailable_contexts']} | {r['unsealed_contexts']} | {r['completed_cells']}/{r['planned_cells']} |")
    lines += ["", "All 2,880 recorded repeat-related equality and coverage checks satisfy their recorded conditions, including 1,800 equality checks. The 72 completed arms contain seven fixed-alpha probes repeated three times and finest maps repeated twice. These are dependent diagnostics on the same contexts, not 2,880 independent replicates. Repeatability does not establish quadrature accuracy.", "",
        "Source collection was prospectively capped at 13 policy calls and 400 environment steps, with calls 0 and 12 selected before collection and unavailable calls retained without replacement. This is not uniform sampling from a completed 400-step trajectory. The input binds the source collection protocol, all six banks, initial queue and both repaired queues. Earlier overlapping snapshots are not added to the sample count.", "",
        "The full ladder contains 302 threshold violations, 11,146 satisfied checks and 144 inapplicable checks. These counts combine dependent comparisons and conditional candidate-completeness checks; they are not numbers of failed experiments. Inapplicable checks are the recorded zero-realized-count top-5-percent comparisons for small groups. Every original check is retained in the input.", "",
        "The following extrema and counts use only the two finest recorded budgets: 256 versus 512 for vision/language, and 512 versus 1024 for state. The IG coordinate column uses IG only. RMS extrema include both IG and path-gradient rankings and both deletion/insertion; extrema in different columns may arise in different contexts. The finer map is a comparator, not ground truth. No budget or subset is approved here.", "",
        "| Modality | Target | Complete arms | Any finer-pair violation | Finest completeness >1% | Max IG relative L1 | Max finest residual | Max RMS curve change / baseline RMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["modality_target_extrema"]:
        lines.append(f"| {r['modality']} | {r['target']} | {r['completed_contexts']} | {r['contexts_with_finest_pair_threshold_violation']} | {r['contexts_with_finest_completeness_over_threshold']} | {r['max_finer_pair_coordinate_relative_l1_IG']:.6g} | {r['max_finest_relative_residual']:.6g} | {r['max_rms_curve_difference_over_baseline']:.6g} |")
    lines += ["", "All four finest-budget completeness violations occur in the two YCB episode-1 state contexts, for both targets. The table below preserves every budget for those four arms; the apparent improvement with increasing budget does not make the finest result accurate.", "",
        "| Call | Target | m | Scalar gap | IG sum | Absolute residual | Relative residual |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    for r in diagnosis["budget_accounting"]:
        lines.append(f"| {r['call']} | {r['target']} | {r['m']} | {r['signed_gap']:.8g} | {r['signed_ig_sum']:.8g} | {r['absolute_residual']:.8g} | {r['relative_residual']:.6g} |")
    lines += ["", "These state discrepancies are not merely division by an almost-zero scalar gap: the Q gaps are about 0.47 and 0.68, and the L2 gaps about 22.0 and 26.4. The corresponding coordinate L1 denominators and absolute map changes are retained in per_cell.csv. YCB call 12 Q has a 32.1% IG coordinate difference between 512 and 1024 despite identical transferred deletion/insertion response curves at those budgets. Stable ranks or response curves therefore do not repair its unresolved coordinate integral.", "",
        "Saved probes show large interior changes in action distance and gradients, including nonmonotonic distance along the call-0 state path. This is consistent with difficult path integration. Only seven fixed-alpha gradients and aggregate maps are available for diagnosis, so neither a narrow peak nor an exclusive numerical cause is established. Both Q and L2 fail, which rules out attributing all failures to L2 smoothing at the self-reference endpoint. The common-distance gradient identity is checked descriptively from saved tensors; it is not a new model evaluation.", "",
        "The estimand is an offline FP32 downstream probe on contexts visited by a BF16 behavior policy. BF16-valued weights and cached adapted image/language values are lifted to FP32, raw state is readapted in FP32, recorded noise values are held fixed, and a new FP32 self-reference is used. This is neither a native FP32-weight/encoder rollout nor the exact derivative of the original BF16 policy. The strict recorded backend bundle gives repeatability on tested points; its individual switches were not causally isolated.", "",
        "The narrow paper can report the probe contract, observed repeatability, heterogeneous budget sensitivity and explicit finite-grid failures with the full roster. These diagnostics cannot support attribution superiority, learned-weight dependence, scale generalization, policy efficacy, native-FP32-policy equivalence, or cohort-wide numerical certification. Earlier v3/v4 precision controls use different comparison contracts and are provenance context, not pooled replication of this probe.", "",
        "Attempt accounting preserves two sealed historical infrastructure/contract failures and their retries. Thirteen sealed successful jobs include one source-unavailable call, leaving twelve numerical contexts. Unsealed historical or current jobs are not labeled numerical failures. Budget-stopped work after this snapshot remains outside its assessed evidence.", ""]
    return "\n".join(lines)


def render_tex(summary, output):
    labels = {"pickcube":"PickCube", "stackcube":"StackCube", "peginsertionside":"PegInsertionSide", "picksingleycb":"PickSingleYCB"}
    sha = lambda name: hashlib.sha256((output/name).read_bytes()).hexdigest()
    roster = [f"% Generated from strata.csv SHA256 {sha('strata.csv')}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Stratum & Planned & Complete & Unreached & Unsealed & Cells \\", r"\midrule"]
    for row in summary["strata"]:
        task, model = row["stratum"].rsplit("-",1)
        roster.append(f"{labels[task]} {model.upper()} & {row['planned_contexts']} & {row['completed_contexts']} & {row['source_unavailable_contexts']} & {row['unsealed_contexts']} & {row['completed_cells']}/{row['planned_cells']} " + r"\\")
    totals = {k:sum(r[k] for r in summary["strata"]) for k in ("planned_contexts","completed_contexts","source_unavailable_contexts","unsealed_contexts","completed_cells","planned_cells")}
    roster += [r"\midrule", f"Total & {totals['planned_contexts']} & {totals['completed_contexts']} & {totals['source_unavailable_contexts']} & {totals['unsealed_contexts']} & {totals['completed_cells']}/{totals['planned_cells']} " + r"\\", r"\bottomrule", r"\end{tabular}", ""]
    diagnostics = [f"% Generated from modality_target.csv SHA256 {sha('modality_target.csv')}",
        r"\begin{tabular}{llrrrrr}",r"\toprule",
        r"Modality & Target & $n$ & Viol. & IG $L_1$ (\%) & Residual (\%) & RMS (\%) \\",r"\midrule"]
    for row in summary["modality_target_extrema"]:
        target = "$Q$" if row["target"] == "Q" else "$L$"
        diagnostics.append(f"{row['modality'].capitalize()} & {target} & {row['completed_contexts']} & {row['contexts_with_finest_pair_threshold_violation']} & {100*row['max_finer_pair_coordinate_relative_l1_IG']:.3f} & {100*row['max_finest_relative_residual']:.3f} & {100*row['max_rms_curve_difference_over_baseline']:.3f} " + r"\\")
    diagnostics += [r"\bottomrule",r"\end{tabular}", ""]
    (output/"numerical_roster.tex").write_text("\n".join(roster),encoding="utf-8",newline="\n")
    (output/"numerical_diagnostics.tex").write_text("\n".join(diagnostics),encoding="utf-8",newline="\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "inputs/sealed_v6.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA256:
        raise ValueError("Input projection changed")
    data = json.loads(raw)
    summary, diagnosis = summarize(data), diagnose_ycb(data)
    args.output.mkdir(parents=True, exist_ok=True)
    for name,value in (("summary.json",summary),("ycb_diagnosis.json",diagnosis)):
        (args.output/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
    for name, rows in (("roster.csv",summary["roster"]),("strata.csv",summary["strata"]),
        ("per_cell.csv",summary["per_cell"]),("modality_target.csv",summary["modality_target_extrema"]),
        ("ycb_budgets.csv",diagnosis["budget_accounting"]),("ycb_probes.csv",diagnosis["fixed_probes"])):
        write_csv(args.output/name,rows)
    (args.output/"review.md").write_text(render(summary,diagnosis),encoding="utf-8",newline="\n")
    render_tex(summary,args.output)
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    (args.output/"manifest.json").write_text(json.dumps(dict(input_sha256=INPUT_SHA256,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outputs=manifest),sort_keys=True,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:summary[k] for k in ("planned_contexts","available_contexts","completed_contexts","completed_episodes","planned_cells","completed_cells","criterion_status_counts")}))


if __name__ == "__main__":
    main()
