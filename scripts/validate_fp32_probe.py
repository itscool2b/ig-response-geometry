"""Validate the exact prospective FP32_PROBE without collecting new contexts.

This is separate from the historical precision-factor control. All production
probe, scalar, gradient-integration and perturbation closures are imported from
paired_comparison. A recorded protocol selects bank contexts and settings.
Outputs are exclusive, with raw tensors and durable progress/failure records.
The report never grants a production approval automatically.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

# Keep torch unimported until deterministic startup is configured in main().
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate_decision(decision, bank, bank_hash):
    from experiment_io import canonical_json, object_hash
    from paired_comparison import FP32_PROBE, validate_grid
    canonical_json(decision)
    if not decision.get("decision_id") or decision.get("recorded_before_execution") is not True:
        raise ValueError("A recorded prospective numerical decision is required")
    if decision.get("forward_precision") != FP32_PROBE or decision.get("bank_sha256") != bank_hash:
        raise ValueError("Decision probe contract or context-bank hash mismatch")
    if decision.get("phase") not in {"budget_selection", "heldout_validation"}:
        raise ValueError("Explicit numerical selection or held-out phase required")
    if decision.get("runtime_mode") not in {"baseline", "deterministic"}:
        raise ValueError("Runtime mode must be explicit")
    contexts = decision.get("contexts")
    if not contexts or any(set(c) != {"episode", "policy_call_idx"} for c in contexts):
        raise ValueError("Select explicit episode/call pairs before observing results")
    keys = [(c["episode"], c["policy_call_idx"]) for c in contexts]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate selected context")
    by_key = {(c["episode"], c["policy_call_idx"]): c for c in bank["contexts"]}
    if any(key not in by_key for key in keys):
        raise ValueError("Selection is outside the prespecified collector bank")
    selected = [by_key[key] for key in keys]
    calibration = decision.get("calibration_episode_ids", [])
    if decision["phase"] == "heldout_validation":
        if not calibration or not decision.get("selection_artifact_sha256") or not decision.get("candidate_numerics"):
            raise ValueError("Held-out validation must bind the completed budget-selection decision")
        if set(calibration) & {c["episode_id"] for c in selected}:
            raise ValueError("Held-out contexts share a calibration episode")
    specs = decision.get("numerics", {})
    if set(specs) != {"vision", "language", "state"}:
        raise ValueError("The production probe validation must cover all three modalities")
    for modality, targets in specs.items():
        if set(targets) != {"Q", "L2"}:
            raise ValueError("Both Q and L2 targets must be validated")
        for target, spec in targets.items():
            budgets = spec.get("budgets", [])
            if len(budgets) < 3 or any(type(m) is not int or m < 1 for m in budgets) or any(b != 2*a for a,b in zip(budgets, budgets[1:])):
                raise ValueError("Use at least three explicit consecutively doubled budgets")
            repeated = spec.get("repeat_budgets", [])
            if not repeated or len(set(repeated)) != len(repeated) or not set(repeated) <= set(budgets) or budgets[-1] not in repeated:
                raise ValueError("Repeat budgets must include the finest tested reference")
            if spec.get("arithmetic_dtype") != "float32" or spec.get("quadrature") != "trapezoid":
                raise ValueError("This exact production probe uses fp32 master arithmetic and trapezoidal quadrature")
            if decision["phase"] == "heldout_validation":
                candidate = decision["candidate_numerics"][modality][target]
                if candidate != dict(m=budgets[0], quadrature="trapezoid", arithmetic_dtype="float32") or budgets[0] not in repeated:
                    raise ValueError("Held-out ladder/repeats must start at the locked production candidate")
    for name in ("gradient_repeats", "map_repeats", "endpoint_repeats"):
        if type(decision.get(name)) is not int or decision[name] < 2:
            raise ValueError(f"{name} must permit a repeatability comparison")
    alphas = decision.get("alphas", [])
    if len(alphas) < 3 or alphas[0] != 0 or alphas[-1] != 1 or sorted(set(alphas)) != alphas or any(not math.isfinite(a) or not 0 <= a <= 1 for a in alphas):
        raise ValueError("Fixed-alpha probes must include both endpoints and interior points")
    if not any(.9 <= a < 1 for a in alphas):
        raise ValueError("L2 validation requires a prespecified near-endpoint interior probe")
    validate_grid(decision["grid_percent"])
    if 5 not in decision["grid_percent"]:
        raise ValueError("The response grid must include the realized top-five-percent intervention")
    for target in ("Q", "L2", "RMS"):
        threshold = decision["denominator_min"][target]
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid numerical endpoint-gap threshold")
    for field in ("criteria", "failure_rule", "stopping_rule", "selection_rationale"):
        if not decision.get(field):
            raise ValueError(f"Missing prospective {field}")
    return selected


def save_tensors(path, values):
    import torch
    from experiment_io import file_hash, tensor_hash
    cpu = {key: value.detach().cpu().contiguous() for key, value in values.items()}
    with Path(path).open("xb") as handle:
        torch.save(cpu, handle)
    return dict(file=Path(path).name, sha256=file_hash(path),
                tensor_sha256={key:tensor_hash(value) for key,value in cpu.items()},
                bytes=Path(path).stat().st_size)


def compare_maps(reference, candidate, indices, axis):
    """Use the exact production grouping and count rule, including top5=zero."""
    import torch
    from scipy.stats import spearmanr
    import paired_comparison as paired
    from scripts.validate_rdt_numerics import comparison_metrics
    result = comparison_metrics(reference, candidate)
    a, b = paired.grouped_scores(reference, indices, axis), paired.grouped_scores(candidate, indices, axis)
    order_a = torch.argsort(a, descending=True, stable=True).cpu().tolist()
    order_b = torch.argsort(b, descending=True, stable=True).cpu().tolist()
    constant = bool((a == a[0]).all() or (b == b[0]).all())
    count = round(.05 * len(indices))
    result.update(group_spearman=None if constant else float(spearmanr(a.double().cpu().numpy(), b.double().cpu().numpy()).statistic),
        rank_status="constant_group_scores" if constant else "defined",
        identical_group_scores=torch.equal(a,b), reference_zero_map=bool((reference == 0).all()),
        candidate_zero_map=bool((candidate == 0).all()), eligible_count=len(indices),
        top5_count=count, top5_realized_fraction=count/len(indices),
        top5_overlap=len(set(order_a[:count]) & set(order_b[:count]))/count if count else None,
        top5_status="defined" if count else "zero_realized_count",
        top1_match=order_a[0] == order_b[0], identical_order=order_a == order_b,
        comparison_denominator="L1 norm of finer-budget or lower-repeat-index reference, undefined at zero")
    return result


def compare_repeats(values, indices, axis):
    import torch
    if len(values) < 2:
        raise ValueError("At least two saved repeats are required")
    pairs = [dict(left=i, right=j, bitwise_equal=torch.equal(values[i],values[j]),
                  **compare_maps(values[i],values[j],indices,axis))
             for i,j in itertools.combinations(range(len(values)),2)]
    return dict(pairs=pairs, all_bitwise_equal=all(p["bitwise_equal"] for p in pairs),
                max_absolute_coordinate_difference=max(p["max_absolute_coordinate_difference"] for p in pairs),
                max_absolute_l1_difference=max(p["absolute_l1_difference"] for p in pairs))


def compare_responses(finer, candidate, reference_label, candidate_label):
    """Compare common responses from production evaluate_rankings outputs."""
    if finer["realized_fractions"] != candidate["realized_fractions"] or finer["eligible_indices"] != candidate["eligible_indices"]:
        raise ValueError("Common-response comparison populations or grids differ")
    comparisons = []
    for direction in ("deletion", "insertion"):
        for response in ("Q", "L2", "RMS"):
            a = finer["rankings"][reference_label]["curves"][direction]["responses"][response]
            b = candidate["rankings"][candidate_label]["curves"][direction]["responses"][response]
            differences = [None if x is None or y is None else y-x for x,y in zip(a["values"], b["values"])]
            comparisons.append(dict(direction=direction,response=response,reference_status=a["status"],candidate_status=b["status"],
                raw_curve_candidate_minus_reference=differences,
                max_absolute_raw_curve_difference=None if any(x is None for x in differences) else max(map(abs,differences)),
                raw_auc_candidate_minus_reference=None if a["raw_auc"] is None or b["raw_auc"] is None else b["raw_auc"]-a["raw_auc"],
                normalized_auc_candidate_minus_reference=None if a["normalized_auc"] is None or b["normalized_auc"] is None else b["normalized_auc"]-a["normalized_auc"]))
    return comparisons


def endpoint_repeats(action, actual, baseline, reference, repeats, output):
    import torch
    import paired_comparison as paired
    from experiment_io import tensor_hash
    records, saved = {}, {}
    for name, value in (("actual",actual),("baseline",baseline)):
        with torch.no_grad():
            device_actions = [action(value).detach() for _ in range(repeats)]
            # Score on the production device before copying raw tensors for
            # storage; CPU and CUDA reductions need not round identically.
            scores = [{k:float(v.item()) for k,v in paired.common_scores(a,reference).items()} for a in device_actions]
            actions = [a.cpu() for a in device_actions]
        # Preserve the raw endpoint response even if the subsequent finite or
        # exact-self-reference check fails. JSON never encodes NaN/Inf.
        raw_artifact=save_tensors(output/f"endpoint-{name}.pt",{f"repeat_{i}":a for i,a in enumerate(actions)})
        if any(not torch.isfinite(a).all() for a in actions) or any(not math.isfinite(s) for row in scores for s in row.values()):
            raise FloatingPointError("Nonfinite production endpoint response")
        if name == "actual" and any(tensor_hash(a) != tensor_hash(reference) for a in actions):
            raise FloatingPointError("Production modality actual endpoint differs from its new self-reference")
        records[name] = dict(action_sha256=[tensor_hash(a) for a in actions],scores=scores,
                             artifact=raw_artifact,
                             bitwise_equal=all(torch.equal(actions[0],a) for a in actions),
                             max_absolute_action_difference=max(float((actions[0]-a).abs().max()) for a in actions))
        for repeat,value in enumerate(actions):
            saved[f"{name}_repeat_{repeat}"] = value
    records["artifact"] = save_tensors(output/"endpoints.pt",saved)
    records["gap_ranges"] = {target:[min(x[target] for x in records["actual"]["scores"])-max(x[target] for x in records["baseline"]["scores"]),
                                      max(x[target] for x in records["actual"]["scores"])-min(x[target] for x in records["baseline"]["scores"])] for target in ("Q","L2","RMS")}
    return records


def validate_target(ctx, modality, target, spec, decision, output, context_id):
    """Execute every locked budget/repeat and retain intermediate artifacts."""
    import torch
    import paired_comparison as paired
    from experiment_io import atomic_bytes, canonical_json, tensor_hash
    from scripts.validate_downstream_precision import FP32OperationAudit
    from scripts.validate_gradient_repeatability import path_gradient, rng_identity
    actual,baseline,indices,axis,action = paired.modality_inputs(ctx,modality)
    forward = lambda x: paired.common_scores(action(x),ctx["ref_action"])[target]
    output.mkdir(parents=True,exist_ok=False)
    report = dict(modality=modality,target=target,context_id=context_id,status="running",
                  input_sha256=tensor_hash(actual),baseline_sha256=tensor_hash(baseline),
                  reference_sha256=tensor_hash(ctx["ref_action"]),eligible_indices=indices,
                  specification=spec,gradient_probes=[],operation_audits=[],budgets=[],comparisons=[])
    def checkpoint():
        atomic_bytes(output/"report.json",canonical_json(report)+b"\n")
    checkpoint()
    try:
        report["endpoints"] = endpoint_repeats(action,actual,baseline,ctx["ref_action"],decision["endpoint_repeats"],output)
        gradient_values = defaultdict(list)
        for repeat in range(decision["gradient_repeats"]):
            for alpha_index,alpha in enumerate(decision["alphas"]):
                before = rng_identity()
                if repeat == 0:
                    # Audit every declared alpha, not just one central point.
                    audit = FP32OperationAudit()
                    with audit:
                        score,gradient,point = path_gradient(forward,actual,baseline,alpha,torch.float32)
                    report["operation_audits"].append(dict(alpha=alpha,operations=dict(audit.operations),status="passed"))
                else:
                    score,gradient,point = path_gradient(forward,actual,baseline,alpha,torch.float32)
                after = rng_identity()
                artifact = save_tensors(output/f"gradient-r{repeat}-a{alpha_index}.pt",dict(gradient=gradient))
                gradient_values[alpha].append(gradient.cpu())
                report["gradient_probes"].append(dict(repeat=repeat,alpha=alpha,score=score,
                    gradient_l1=float(gradient.double().abs().sum()),point_sha256=tensor_hash(point),
                    rng_before=before,rng_after=after,rng_unchanged=before==after,**artifact))
                checkpoint()
        report["gradient_repeatability"] = {str(alpha):compare_repeats(values,indices,axis) for alpha,values in gradient_values.items()}
        del gradient_values
        by_budget = {}
        for budget in spec["budgets"]:
            settings = dict(m=budget,quadrature=spec["quadrature"],arithmetic_dtype=spec["arithmetic_dtype"])
            repeats = decision["map_repeats"] if budget in spec["repeat_budgets"] else 1
            maps, gradients, response_sets, budget_rows = [],[],[],[]
            for repeat in range(repeats):
                print(f"fp32 probe {context_id[:10]} {modality} {target}: m={budget}, repeat={repeat}",flush=True)
                before = rng_identity()
                result,path_gradient_map = paired.integrate_with_path_gradient(forward,actual,baseline,settings,
                    context=dict(context_id=context_id,modality=modality,target=target,budget=budget,repeat=repeat))
                after = rng_identity()
                artifact = save_tensors(output/f"m{budget}-r{repeat}.pt",dict(attribution=result.attributions,path_gradient=path_gradient_map))
                rankings = {label:paired.ranked(paired.grouped_scores(tensor,indices,axis),indices) for label,tensor in
                            (("IG",result.attributions),("path_gradient",path_gradient_map))}
                responses = paired.evaluate_rankings(action,ctx["ref_action"],actual,baseline,indices,axis,rankings,
                                                     decision["grid_percent"],decision["denominator_min"])
                responses_path = output/f"m{budget}-r{repeat}-responses.json"
                from experiment_io import file_hash
                with responses_path.open("xb") as stream:
                    stream.write(canonical_json(responses)+b"\n")
                row = dict(repeat=repeat,diagnostics=result.diagnostics(),rng_before=before,rng_after=after,
                           rng_unchanged=before==after,**artifact,response_file=responses_path.name,response_sha256=file_hash(responses_path))
                maps.append(result.attributions.detach().cpu())
                gradients.append(path_gradient_map.detach().cpu())
                response_sets.append(responses)
                budget_rows.append(row)
                report["budgets"].append(dict(m=budget,**row))
                checkpoint()
            repeated = None
            if repeats > 1:
                repeated = dict(IG=compare_repeats(maps,indices,axis),path_gradient=compare_repeats(gradients,indices,axis),
                    common_responses=[dict(left=i,right=j,ranking=label,comparisons=compare_responses(response_sets[i],response_sets[j],label,label))
                        for i,j in itertools.combinations(range(repeats),2) for label in ("IG","path_gradient")])
            by_budget[budget] = dict(IG=maps[0],path_gradient=gradients[0],responses=response_sets[0],repeatability=repeated)
            report.setdefault("map_repeatability",{})[str(budget)] = repeated
            checkpoint()
        # All budget pairs, directed from finer reference to coarser candidate,
        # include adjacent doublings and every candidate versus the finest map.
        for candidate,finer in itertools.combinations(spec["budgets"],2):
            for label in ("IG","path_gradient"):
                report["comparisons"].append(dict(candidate_m=candidate,reference_m=finer,ranking=label,
                    adjacent=finer==2*candidate,finest_reference=finer==spec["budgets"][-1],
                    **compare_maps(by_budget[finer][label],by_budget[candidate][label],indices,axis),
                    common_responses=compare_responses(by_budget[finer]["responses"],by_budget[candidate]["responses"],label,label)))
        report["status"] = "complete_diagnostics_not_approval"
        report["reference_qualification"] = "Finest tested quadrature map is a numerical comparator, not known truth."
        checkpoint()
        return report
    except Exception as error:
        report.update(status="failed",failure=dict(type=type(error).__name__,reason=str(error),diagnostics=getattr(error,"diagnostics",{})))
        checkpoint()
        raise


from fp32_probe_cache import process_startup, runtime_settings, apply_settings, strict_candidate_settings


def load_bank_inputs(args):
    import paired_comparison as paired
    from experiment_io import file_hash,strict_json
    from faithfulness import replay_inputs
    decision,bank=strict_json(args.decision_file),strict_json(args.bank)
    if decision.get("runtime_mode")!="deterministic":
        raise ValueError("The candidate production validator requires strict deterministic execution")
    selected=validate_decision(decision,bank,file_hash(args.bank))
    replay_args=SimpleNamespace(metrics=args.metrics,task=bank["task"],model=bank["model"],target=None,
        lang_dir=args.lang_dir,checkpoint_path=args.checkpoint_path,checkpoint_mode=None,model_revision=None,
        vision_revision=None,no_checkpoint=False,legacy_input=False,limit=None)
    rows,manifest,digest=replay_inputs(replay_args,apply_limit=False)
    selection=bank["selection"]
    rebuilt=paired.make_bank(args.metrics,None if selection["rule"]=="exact_collector_prespecified_call_indices" else selection)
    if rebuilt!=bank:
        raise ValueError("Bank population differs from authenticated collector")
    selected_rows=paired.bank_rows(bank,rows,manifest,digest)
    return decision,bank,selected,replay_args,manifest,digest,selected_rows


def _complete(output,status):
    from experiment_io import atomic_bytes,canonical_json,file_hash
    artifacts={str(p.relative_to(output)):file_hash(p) for p in output.rglob("*") if p.is_file()}
    atomic_bytes(output/"completion.json",canonical_json(dict(status=status,artifacts_sha256=artifacts))+b"\n")


def _prepare(args,output,startup):
    import torch
    from experiment_io import atomic_bytes,canonical_json,file_hash
    from faithfulness import load_sidecar,replay_context,replay_pipeline
    from scripts.validate_downstream_precision import cache_context,cache_identity
    decision,bank,selected,replay_args,manifest,digest,selected_rows=load_bank_inputs(args)
    applied=apply_settings(decision["source_replay_settings"])
    stored_settings=manifest["configuration"].get("runtime_settings")
    if stored_settings is not None and stored_settings!=applied:
        raise ValueError("Source manifest runtime settings differ from the replay declaration")
    report=dict(schema_version=1,status="preparing",kind="fp32_probe_source_preparation",
        decision_sha256=file_hash(args.decision_file),bank_sha256=file_hash(args.bank),source_metrics_sha256=digest,
        source_run_id=manifest["run_id"],source_configuration_sha256=manifest["configuration_sha256"],
        source_pipeline=manifest["configuration"]["pipeline"],source_language=manifest["configuration"]["language"],
        source_runtime=manifest["configuration"]["environment"],
        source_forward_sha256=manifest["configuration"]["source_sha256"],startup=startup,replay_settings=applied,
        source_settings_authentication="manifest" if stored_settings is not None else "protocol_declared_exact_reference_replay_only",
        qualification="Exact replay is verified. If the source did not record runtime flags, historical flag equality is not asserted.",
        contexts=[])
    def checkpoint():
        atomic_bytes(output/"report.json",canonical_json(report)+b"\n")
    checkpoint()
    pipe,lang=replay_pipeline(replay_args,manifest)
    for entry in selected:
        item={**entry}
        report["contexts"].append(item)
        if entry["status"]!="available":
            checkpoint()
            continue
        source_row=selected_rows[(entry["episode"],entry["policy_call_idx"])]
        payload,payload_hash=load_sidecar(source_row,args.metrics,manifest)
        source=replay_context(replay_args,source_row,payload,pipe,lang,verify_reference=True,strict=True)
        cache=cache_context(source)
        item.update(source_replay_verified=True,sidecar_sha256=payload_hash,
                    cached_tensors=cache_identity(cache),
                    cache_file=save_tensors(output/f"ep{entry['episode']:06d}-call{entry['policy_call_idx']:04d}.pt",cache))
        checkpoint()
        del source,cache,payload
    if file_hash(args.metrics)!=digest:
        raise ValueError("Source changed during cache preparation")
    if runtime_settings()!=applied:
        raise ValueError("Source replay mutated declared runtime settings")
    report["status"]="complete_source_preparation"
    checkpoint()
    _complete(output,report["status"])


from fp32_probe_cache import verify_preparation, load_prepared_context


def _run(args,output,startup):
    import torch
    import paired_comparison as paired
    from experiment_io import atomic_bytes,canonical_json,file_hash
    from faithfulness import load_sidecar,replay_pipeline
    from scripts.validate_downstream_precision import convert_downstream_to_fp32,fp32_math_settings
    from integrated_gradients import NonFiniteAttributionError
    decision,bank,selected,replay_args,manifest,digest,selected_rows=load_bank_inputs(args)
    applied=apply_settings(strict_candidate_settings())
    preparation=verify_preparation(args.prepared,args.prepared_sha256)
    if preparation["decision_sha256"]!=file_hash(args.decision_file) or preparation["bank_sha256"]!=file_hash(args.bank) or preparation["source_metrics_sha256"]!=digest:
        raise ValueError("Prepared source bank or decision differs")
    if preparation["replay_settings"]!=decision["source_replay_settings"]:
        raise ValueError("Prepared replay settings differ from the declared transition")
    from scripts import validate_downstream_precision as precision
    from scripts import validate_gradient_repeatability as repeatability
    import integrated_gradients as kernel
    import faithfulness as replay_helper
    import fp32_probe_cache as cache_helper
    report=dict(schema_version=1,status="running",decision=decision,decision_sha256=file_hash(args.decision_file),
        bank_sha256=file_hash(args.bank),source_metrics_sha256=digest,source_manifest=manifest,startup=startup,
        forward_precision=paired.FP32_PROBE,script_sha256=file_hash(__file__),
        helper_sha256={"paired_comparison":file_hash(paired.__file__),"integrated_gradients":file_hash(kernel.__file__),
                       "precision":file_hash(precision.__file__),"repeatability":file_hash(repeatability.__file__),
                       "replay":file_hash(replay_helper.__file__),"cache_runtime":file_hash(cache_helper.__file__)},contexts=[],
        preparation_completion_sha256=args.prepared_sha256,
        settings_transition=dict(source_replay=preparation["replay_settings"],candidate=applied,
                                 source_settings_authentication=preparation["source_settings_authentication"],
                                 process_boundary="CUBLAS workspace configured before Torch in each fresh process"),
        runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,
                     device=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
                     settings=runtime_settings()))
    def checkpoint():
        atomic_bytes(output/"report.json",canonical_json(report)+b"\n")
    checkpoint()
    # This runner is loaded independently of the original preparation process.
    # It is authenticated before conversion; no bf16 replay is relabeled here.
    probe_pipe,_=replay_pipeline(replay_args,manifest)
    report["conversion"]=convert_downstream_to_fp32(probe_pipe["runner"])
    device=next(probe_pipe["runner"].parameters()).device
    with fp32_math_settings() as settings:
        report["math_settings"]=settings
        checkpoint()
        for entry in selected:
            row=dict(episode=entry["episode"],policy_call_idx=entry["policy_call_idx"],episode_id=entry["episode_id"],status=entry["status"])
            report["contexts"].append(row)
            if entry["status"]!="available":
                checkpoint()
                continue
            source_row=selected_rows[(entry["episode"],entry["policy_call_idx"])]
            context_dir=output/f"ep{entry['episode']:06d}-call{entry['policy_call_idx']:04d}"
            context_dir.mkdir(exist_ok=False)
            payload,payload_hash=load_sidecar(source_row,args.metrics,manifest)
            cache,cache_provenance=load_prepared_context(args.prepared,preparation,entry,manifest,source_row,payload)
            source={key:value.to(device) for key,value in cache.items()}
            row.update(source_replay_verified=True,source_context_id=entry["context_id"],source_sidecar_sha256=payload_hash,
                       source_prepared_cache=cache_provenance["cache_file"])
            try:
                probe,provenance=paired.make_fp32_probe_context(source,probe_pipe["runner"])
            except (NonFiniteAttributionError,FloatingPointError) as error:
                # Authentication has already succeeded. An invalid NEW probe
                # is a numerical outcome for every planned arm, not an absent
                # source or permission to replace this context.
                raw_failure=getattr(error,"raw_tensors",{})
                artifact=save_tensors(context_dir/"probe-failure.pt",raw_failure) if raw_failure else None
                if artifact is not None:
                    artifact["file"]=str((context_dir/"probe-failure.pt").relative_to(output))
                row.update(status="probe_numerical_failure",failure_kind="probe_self_reference",
                    failure_artifact=artifact,
                    reason=str(error),diagnostics=getattr(error,"diagnostics",{}),
                    numerics=[dict(modality=modality,target=target,status="numerical_failure",
                                   failure_kind="probe_self_reference",reason=str(error),
                                   diagnostics=getattr(error,"diagnostics",{}))
                              for modality in ("vision","language","state") for target in ("Q","L2")])
                checkpoint()
                del source,payload,cache
                continue
            row["probe_identity"]=provenance
            row["probe_context_artifact"]=save_tensors(context_dir/"probe-context.pt",{k:v for k,v in probe.items() if isinstance(v,torch.Tensor)})
            row["numerics"]=[]
            checkpoint()
            for modality in ("vision","language","state"):
                for target in ("Q","L2"):
                    location=context_dir/f"{modality}-{target}"
                    try:
                        result=validate_target(probe,modality,target,decision["numerics"][modality][target],decision,location,entry["context_id"])
                        outcome=dict(modality=modality,target=target,status=result["status"])
                    except (NonFiniteAttributionError,FloatingPointError) as error:
                        outcome=dict(modality=modality,target=target,status="numerical_failure",reason=str(error),diagnostics=getattr(error,"diagnostics",{}))
                    outcome.update(report_file=str((location/"report.json").relative_to(output)),report_sha256=file_hash(location/"report.json"))
                    row["numerics"].append(outcome)
                    checkpoint()
            row["status"]="evaluated"
            del source,probe,payload,cache
            checkpoint()
    if file_hash(args.metrics)!=digest or file_hash(args.bank)!=report["bank_sha256"] or file_hash(args.decision_file)!=report["decision_sha256"]:
        raise ValueError("Source, bank or locked decision changed during validation")
    if runtime_settings()!=applied:
        raise ValueError("Candidate validation mutated declared runtime settings")
    report["status"]="complete_diagnostics_not_approval"
    report["planned_contexts"]=len(selected)
    report["available_contexts"]=sum(c["status"] in {"evaluated","probe_numerical_failure"} for c in report["contexts"])
    report["numerical_failures"]=sum(n["status"]=="numerical_failure" for c in report["contexts"] for n in c.get("numerics",[]))
    checkpoint()
    _complete(output,report["status"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","run"])
    parser.add_argument("--metrics",type=Path,required=True)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--decision-file",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--lang-dir",required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--prepared",type=Path)
    parser.add_argument("--prepared-sha256")
    args=parser.parse_args()
    if args.stage=="run" and (not args.prepared or not args.prepared_sha256):
        parser.error("run requires --prepared and its pinned --prepared-sha256 completion hash")
    from experiment_io import strict_json
    decision=strict_json(args.decision_file)
    startup=process_startup(args.stage,decision.get("source_replay_settings"))
    output=args.out.resolve()
    output.mkdir(parents=True,exist_ok=False)
    try:
        (_prepare if args.stage=="prepare" else _run)(args,output,startup)
    except Exception as error:
        from experiment_io import atomic_bytes,canonical_json,file_hash
        atomic_bytes(output/"failure.json",canonical_json(dict(status="failed",exception_type=type(error).__name__,
            reason=str(error),diagnostics=getattr(error,"diagnostics",{}),startup=startup,script_sha256=file_hash(__file__)))+b"\n")
        raise


if __name__ == "__main__":
    main()
