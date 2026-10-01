"""Fixed-ladder numerical validation of the E05 two-function control.

Prepare authenticates the behavior-policy cache in a fresh source-runtime
process. Run differentiates each permuted function against its own reference,
restores the trained tensors, and evaluates saved rankings on the trained probe.
Reports preserve the planned roster and never grant a production approval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fp32_probe_cache import process_startup

THRESHOLDS = dict(coordinate_l1=.01, spearman=.99, top5=.95, residual=.01,
                  rms_curve_fraction=.01, rms_area_fraction=.005, auc=.01)


def implementation_hashes():
    import weight_arrangement_control as control
    from experiment_io import file_hash
    extra = ("scripts/validate_weight_arrangement.py", "scripts/validate_fp32_probe.py",
             "scripts/validate_gradient_repeatability.py", "scripts/audit_numerical_cohort.py",
             "scripts/validate_rdt_numerics.py", "experiment_io.py")
    return {**control.implementation_hashes(), **{name: file_hash(ROOT/name) for name in extra}}


def validate_decision(decision, bank, bank_hash, native_hash):
    import torch
    import weight_arrangement_control as control
    from scripts.validate_fp32_probe import validate_decision as validate_base
    selected = validate_base(decision, bank, bank_hash)
    if decision.get("runtime_mode") != "deterministic":
        raise ValueError("E05 validation requires the strict deterministic runtime")
    if any(decision.get(key) != value for key, value in control.SCOPE.items()):
        raise ValueError("E05 null, reference, response, tie, seed or population scope differs")
    if decision.get("native_registry_sha256") != native_hash:
        raise ValueError("E05 native structural registry differs")
    if decision.get("implementation_sha256") != implementation_hashes():
        raise ValueError("E05 numerical implementation differs from the locked sources")
    if decision.get("torch_version") != torch.__version__:
        raise ValueError("Parameter permutation Torch version differs from the decision")
    if decision.get("criterion_thresholds") != THRESHOLDS:
        raise ValueError("E05 numerical thresholds must preserve the exact locked criteria")
    if type(decision.get("parameter_draws")) is not int or decision["parameter_draws"] < 2:
        raise ValueError("Explicit engineering parameter_draws >= 2 required")
    if decision.get("failure_policy") != "retain_every_planned_draw_no_replacement":
        raise ValueError("Every planned parameter draw must be retained without replacement")
    control.tensor_stream(dict(namespace=decision["parameter_seed_namespace"],
        master_seed=decision["parameter_master_seed"], stratum_id=decision["stratum_id"],
        episode_id="validation-placeholder", draw_index=0), "validation-placeholder")
    if not decision.get("checkpoint_identity") or decision.get("checkpoint_mode") not in {"pretrained", "authors", "lora"}:
        raise ValueError("An explicit source checkpoint identity is required")
    return selected


def checkpoint(path, report):
    from experiment_io import atomic_bytes, canonical_json
    atomic_bytes(Path(path), canonical_json(report)+b"\n")


def input_hashes(args, digest):
    from experiment_io import file_hash
    inputs={args.metrics:digest,args.bank:file_hash(args.bank),args.native_registry:file_hash(args.native_registry),args.decision_file:args.decision_sha256}
    if getattr(args,"checkpoint_path",None):
        path=Path(args.checkpoint_path)
        inputs[path]=file_hash(path)
    return inputs


def failure_record(error, directory, name, kind):
    from scripts.validate_fp32_probe import save_tensors
    raw = getattr(error, "raw_tensors", {})
    artifact = save_tensors(directory/name, raw) if raw else None
    return dict(status="numerical_failure", failure_kind=kind,
                exception_type=type(error).__name__, reason=str(error),
                diagnostics=getattr(error, "diagnostics", {}), failure_artifact=artifact)


def collect_own_target(ctx, modality, target, spec, decision, output, context_id):
    """Only randomized-function evaluations are allowed in this phase."""
    import torch
    import paired_comparison as paired
    from experiment_io import tensor_hash
    from integrated_gradients import NonFiniteAttributionError
    from scripts.validate_downstream_precision import FP32OperationAudit
    from scripts.validate_gradient_repeatability import path_gradient, rng_identity
    from scripts.validate_fp32_probe import save_tensors, endpoint_repeats, compare_maps, compare_repeats
    actual, baseline, indices, axis, native_action = paired.modality_inputs(ctx, modality)
    def action(value):
        result=native_action(value)
        if result.dtype != torch.float32 or result.shape != ctx["ref_action"].shape:
            raise ValueError("Randomized endpoint dtype or shape violates the probe contract")
        return result
    failed_raw = {}
    def forward(value):
        def observe(gradient):
            if not torch.isfinite(gradient).all():
                failed_raw.update(path_point=value.detach(), gradient=gradient.detach())
        if value.requires_grad:
            value.register_hook(observe)
        result=action(value)
        score=paired.common_scores(result,ctx["ref_action"])[target]
        if not torch.isfinite(result).all() or not torch.isfinite(score):
            failed_raw.update(path_point=value.detach(), action=result.detach(), score=score.detach())
        return score
    output.mkdir(parents=True, exist_ok=False)
    report = dict(modality=modality, target=target, context_id=context_id, status="collecting_own_maps",
        input_sha256=tensor_hash(actual), baseline_sha256=tensor_hash(baseline),
        input_shape=list(actual.shape), axis=axis, eligible_indices=indices,
        reference_sha256=tensor_hash(ctx["ref_action"]), noise_sha256=tensor_hash(ctx["initial_noise"]),
        reference_scope="randomized_function_own_reference", response_scope="restored_trained_probe_reference",
        specification=spec, planned_budget_repeats=[dict(m=m, repeat=r) for m in spec["budgets"]
            for r in range(decision["map_repeats"] if m in spec["repeat_budgets"] else 1)],
        gradient_probes=[], operation_audits=[], budgets=[], comparisons=[])
    save = lambda: checkpoint(output/"report.json", report)
    save()
    try:
        endpoint_dir = output/"own-endpoints"; endpoint_dir.mkdir()
        report["own_function_endpoints"] = endpoint_repeats(action, actual, baseline, ctx["ref_action"],
                                                            decision["endpoint_repeats"], endpoint_dir)
        gradient_values = defaultdict(list)
        for repeat in range(decision["gradient_repeats"]):
            for alpha_index, alpha in enumerate(decision["alphas"]):
                before = rng_identity()
                if repeat == 0:
                    audit = FP32OperationAudit()
                    with audit:
                        score, gradient, point = path_gradient(forward, actual, baseline, alpha, torch.float32)
                    report["operation_audits"].append(dict(alpha=alpha, operations=dict(audit.operations), status="passed"))
                else:
                    score, gradient, point = path_gradient(forward, actual, baseline, alpha, torch.float32)
                after = rng_identity()
                artifact = save_tensors(output/f"gradient-r{repeat}-a{alpha_index}.pt", dict(gradient=gradient))
                gradient_values[alpha].append(gradient.cpu())
                report["gradient_probes"].append(dict(repeat=repeat, alpha=alpha, score=score,
                    gradient_l1=float(gradient.double().abs().sum()), point_sha256=tensor_hash(point),
                    rng_before=before, rng_after=after, rng_unchanged=before==after, **artifact))
                save()
        report["gradient_repeatability"] = {str(alpha): compare_repeats(values, indices, axis)
                                              for alpha, values in gradient_values.items()}
        del gradient_values
        by_budget = {}
        for m in spec["budgets"]:
            maps, gradients = [], []
            settings = dict(m=m, quadrature=spec["quadrature"], arithmetic_dtype=spec["arithmetic_dtype"])
            repeats = decision["map_repeats"] if m in spec["repeat_budgets"] else 1
            for repeat in range(repeats):
                print(f"E05 own maps {context_id[:10]} {modality}/{target} m={m} repeat={repeat}", flush=True)
                before = rng_identity()
                result, average = paired.integrate_with_path_gradient(forward, actual, baseline, settings,
                    context=dict(context_id=context_id, modality=modality, target=target, budget=m, repeat=repeat))
                after = rng_identity()
                artifact = save_tensors(output/f"m{m}-r{repeat}.pt", dict(attribution=result.attributions, path_gradient=average))
                diagnostics={**result.diagnostics(),"signed_residual":result.attribution_sum-result.expected_gap,
                             "signed_residual_definition":"sum_IG_minus_own_endpoint_gap"}
                report["budgets"].append(dict(m=m, repeat=repeat, diagnostics=diagnostics,
                    rng_before=before, rng_after=after, rng_unchanged=before==after, **artifact))
                maps.append(result.attributions.detach().cpu()); gradients.append(average.detach().cpu())
                save()
            repeated = None if repeats == 1 else dict(IG=compare_repeats(maps, indices, axis),
                                                       path_gradient=compare_repeats(gradients, indices, axis))
            report.setdefault("map_repeatability", {})[str(m)] = repeated
            by_budget[m] = dict(IG=maps[0], path_gradient=gradients[0])
        for lower, higher in itertools.combinations(spec["budgets"], 2):
            for label in ("IG", "path_gradient"):
                report["comparisons"].append(dict(candidate_m=lower, reference_m=higher, ranking=label,
                    adjacent=higher==2*lower, finest_reference=higher==spec["budgets"][-1],
                    **compare_maps(by_budget[higher][label], by_budget[lower][label], indices, axis)))
        report["status"] = "own_maps_complete_pending_trained_responses"
    except (NonFiniteAttributionError, FloatingPointError) as error:
        if failed_raw:
            error.raw_tensors={**getattr(error,"raw_tensors",{}),**failed_raw}
        report.update(failure_record(error, output, "own-failure.pt", "randomized_numerical_failure"))
    except Exception as error:
        report.update(status="contract_failure", exception_type=type(error).__name__, reason=str(error))
        save()
        raise
    save()
    return report


def complete_trained_responses(ctx, report, decision, output):
    """Called only after verified restoration. Raw maps remain randomized."""
    import torch
    import paired_comparison as paired
    from experiment_io import file_hash, tensor_hash, canonical_json
    from integrated_gradients import NonFiniteAttributionError
    from scripts.validate_downstream_precision import FP32OperationAudit
    from scripts.validate_fp32_probe import endpoint_repeats, compare_responses, save_tensors
    if report["status"] != "own_maps_complete_pending_trained_responses":
        return report
    actual, baseline, indices, axis, action = paired.modality_inputs(ctx, report["modality"])
    if (tensor_hash(actual) != report["input_sha256"] or tensor_hash(baseline) != report["baseline_sha256"] or
            tensor_hash(ctx["initial_noise"]) != report["noise_sha256"] or indices != report["eligible_indices"] or axis != report["axis"]):
        raise ValueError("Trained response inputs differ from the randomized attribution path")
    report["trained_reference_sha256"] = tensor_hash(ctx["ref_action"])
    save = lambda: checkpoint(output/"report.json", report)
    failed_actions = {}
    def checked_action(value):
        result = action(value)
        if result.dtype != torch.float32 or result.shape != ctx["ref_action"].shape:
            raise ValueError("Trained response action dtype or shape violates the probe contract")
        if not torch.isfinite(result).all() or any(not torch.isfinite(v) for v in paired.common_scores(result,ctx["ref_action"]).values()):
            failed_actions[tensor_hash(value)] = result.detach()
        return result
    try:
        endpoint_dir = output/"trained-endpoints"; endpoint_dir.mkdir()
        report["trained_response_endpoints"] = endpoint_repeats(checked_action, actual, baseline, ctx["ref_action"],
                                                                decision["endpoint_repeats"], endpoint_dir)
        responses_by_budget = defaultdict(list)
        for row in report["budgets"]:
            path = output/row["file"]
            if file_hash(path) != row["sha256"]:
                raise ValueError("Saved randomized map bytes changed before transfer")
            raw = torch.load(path, map_location=actual.device, weights_only=True)
            if set(raw) != {"attribution", "path_gradient"} or any(tensor_hash(value) != row["tensor_sha256"][key] for key,value in raw.items()):
                raise ValueError("Saved randomized map tensor identity differs")
            rankings = {label: paired.ranked(paired.grouped_scores(raw[key], indices, axis), indices)
                        for label,key in (("IG","attribution"),("path_gradient","path_gradient"))}
            failed_actions.clear()
            audit = FP32OperationAudit()
            with audit:
                responses = paired.evaluate_rankings(checked_action, ctx["ref_action"], actual, baseline, indices, axis,
                    rankings, decision["grid_percent"], decision["denominator_min"])
            name = f"m{row['m']}-r{row['repeat']}-trained-responses.json"
            with (output/name).open("xb") as stream:
                stream.write(canonical_json(responses)+b"\n")
            row.update(response_file=name, response_sha256=file_hash(output/name),
                       response_operation_audit=dict(status="passed", operations=dict(audit.operations)))
            if failed_actions:
                row["failure_actions"] = save_tensors(output/f"m{row['m']}-r{row['repeat']}-failed-actions.pt", failed_actions)
            row["response_numerical_failures"] = sum(value["status"] != "finite" for value in responses["response_table"].values())
            responses_by_budget[row["m"]].append(responses)
            save()
        for m, values in responses_by_budget.items():
            repeatability = report["map_repeatability"][str(m)]
            if repeatability is not None:
                repeatability["common_responses"] = [dict(left=i, right=j, ranking=label,
                    comparisons=compare_responses(values[i],values[j],label,label))
                    for i,j in itertools.combinations(range(len(values)),2) for label in ("IG","path_gradient")]
                repeatability["trained_action_repeatability"] = [dict(left=i,right=j,
                    bitwise_equal=values[i]["response_table"]==values[j]["response_table"])
                    for i,j in itertools.combinations(range(len(values)),2)]
        for comparison in report["comparisons"]:
            label = comparison["ranking"]
            comparison["common_responses"] = compare_responses(responses_by_budget[comparison["reference_m"]][0],
                responses_by_budget[comparison["candidate_m"]][0], label, label)
        report["response_numerical_failures"] = sum(row["response_numerical_failures"] for row in report["budgets"])
        report["status"] = "numerical_failure" if report["response_numerical_failures"] else "complete_diagnostics_not_approval"
        if report["response_numerical_failures"]:
            report["failure_kind"] = "trained_intervention_response"
        report["reference_qualification"] = "Finest tested randomized map is a comparator, not known truth. Completeness uses the own-function gap; response criteria use the restored trained gap."
        report["criterion_audit"] = criterion_audit(report, decision)
    except (NonFiniteAttributionError, FloatingPointError) as error:
        report.update(failure_record(error, output, "trained-failure.pt", "trained_response_numerical_failure"))
    except Exception as error:
        report.update(status="contract_failure", exception_type=type(error).__name__, reason=str(error))
        save()
        raise
    save()
    return report


def criterion_audit(report, decision):
    """Reuse threshold calculations with an explicit two-function endpoint view."""
    from scripts.audit_numerical_cohort import v6_target, check, summarize_checks
    # v6_target uses endpoints for response denominators and actual replay;
    # completeness instead comes from each integration's own diagnostics.
    view = {**report, "endpoints":report["trained_response_endpoints"],
            "reference_sha256":report["trained_reference_sha256"]}
    checks, observations = v6_target(view, decision["criterion_thresholds"], decision)
    own = report["own_function_endpoints"]
    for endpoint in ("actual", "baseline"):
        value = own[endpoint]; hashes = value["action_sha256"]
        checks.extend([check("own_endpoint_bitwise_repeatability", value["bitwise_equal"], "==", True, endpoint=endpoint),
            check("own_endpoint_hash_repeatability", len(set(hashes))==1, "==", True, endpoint=endpoint),
            check("own_endpoint_repeat_count", len(hashes), "==", decision["endpoint_repeats"], endpoint=endpoint)])
        if endpoint == "actual":
            checks.append(check("exact_randomized_self_reference", all(h==report["reference_sha256"] for h in hashes), "==", True))
    for m,item in report["map_repeatability"].items():
        for pair in (item or {}).get("trained_action_repeatability", []):
            checks.append(check("trained_action_bitwise_repeatability", pair["bitwise_equal"], "==", True, m=int(m), left=pair["left"],right=pair["right"]))
    return dict(checks=checks, observations=observations, summary=summarize_checks(checks),
                qualification="Diagnostic audit only; zero/tied/poorly conditioned cases require explicit raw review and no automatic gate approval.")


def run_episode(session, entries, decision, context_loader, output, progress):
    """Execute coherent draws across calls, restoring before every transfer phase."""
    import torch
    import weight_arrangement_control as control
    from experiment_io import object_hash, tensor_hash, file_hash
    from integrated_gradients import NonFiniteAttributionError
    from scripts.validate_fp32_probe import save_tensors
    trained = {}; failures = {}
    for entry in entries:
        if entry["status"] != "available":
            continue
        directory = output/f"context-{entry['context_id']}"; directory.mkdir(exist_ok=True)
        try:
            ctx, provenance = context_loader(entry)
            artifact=save_tensors(directory/"trained-probe.pt", {key:value for key,value in ctx.items() if isinstance(value,torch.Tensor)})
            artifact["file"]=str((directory/"trained-probe.pt").relative_to(output))
            trained[entry["context_id"]] = dict(reference_sha256=tensor_hash(ctx["ref_action"]), provenance=provenance, artifact=artifact)
            del ctx
        except (NonFiniteAttributionError, FloatingPointError) as error:
            failures[entry["context_id"]] = failure_record(error, directory, "trained-probe-failure.pt", "trained_probe_self_reference")
            if failures[entry["context_id"]]["failure_artifact"]:
                failures[entry["context_id"]]["failure_artifact"]["file"]=str((directory/"trained-probe-failure.pt").relative_to(output))
    for draw_index in range(decision["parameter_draws"]):
        started = time.perf_counter()
        identity = dict(namespace=decision["parameter_seed_namespace"], master_seed=decision["parameter_master_seed"],
            stratum_id=decision["stratum_id"], episode_id=entries[0]["episode_id"], draw_index=draw_index)
        draw_dir = output/("draw-"+object_hash(identity)); draw_dir.mkdir()
        items = []
        for entry in entries:
            item = {**entry, "draw_index":draw_index, "numerics":[], "status":"planned", "source_status":entry["status"]}
            for modality in control.MODALITIES:
                for target in control.TARGETS:
                    item["numerics"].append(dict(modality=modality,target=target,status="not_yet_attempted"))
            items.append(item); progress["contexts"].append(item)
        checkpoint(output/"report.json", progress)
        with session.draw(identity) as record:
            checkpoint(draw_dir/"parameter-draw.json", record)
            for entry,item in zip(entries,items):
                item.update(parameter_draw_sha256=record["parameter_draw_sha256"],
                    parameter_draw_file=str((draw_dir/"parameter-draw.json").relative_to(output)),
                    parameter_draw_file_sha256=file_hash(draw_dir/"parameter-draw.json"))
                if entry["status"] != "available":
                    item["status"] = entry["status"]
                    for arm in item["numerics"]: arm.update(status="source_unavailable", reason=entry["status"])
                    continue
                directory = draw_dir/entry["context_id"]; directory.mkdir()
                if entry["context_id"] in failures:
                    item.update(status="numerical_failure", failure=failures[entry["context_id"]])
                    for arm in item["numerics"]: arm.update(status="numerical_failure", failure_kind="trained_probe_self_reference")
                    continue
                item["trained_probe_identity"] = trained[entry["context_id"]]
                try:
                    ctx, provenance = context_loader(entry)
                    item["own_probe_identity"] = provenance
                    item["own_probe_artifact"] = save_tensors(directory/"own-probe.pt",{key:value for key,value in ctx.items() if isinstance(value,torch.Tensor)})
                    item["own_probe_artifact"]["file"]=str((directory/"own-probe.pt").relative_to(output))
                except (NonFiniteAttributionError, FloatingPointError) as error:
                    item.update(status="numerical_failure", failure=failure_record(error,directory,"own-probe-failure.pt","randomized_self_reference"))
                    if item["failure"]["failure_artifact"]:
                        item["failure"]["failure_artifact"]["file"]=str((directory/"own-probe-failure.pt").relative_to(output))
                    for arm in item["numerics"]: arm.update(status="numerical_failure", failure_kind="randomized_self_reference")
                    checkpoint(output/"report.json", progress)
                    continue
                for arm in item["numerics"]:
                    location = directory/f"{arm['modality']}-{arm['target']}"
                    result = collect_own_target(ctx, arm["modality"], arm["target"], decision["numerics"][arm["modality"]][arm["target"]], decision, location, entry["context_id"])
                    arm.update(status=result["status"], report_file=str((location/"report.json").relative_to(output)))
                    checkpoint(output/"report.json", progress)
                del ctx
        session.verify(session.hashes)
        restoration = dict(parameter_draw_sha256=record["parameter_draw_sha256"],
            registry_sha256=object_hash(session.registry), all_trained_tensor_hashes_verified=True,
            restoration_before_any_trained_response=True)
        checkpoint(draw_dir/"restoration.json", restoration)
        for entry,item in zip(entries,items):
            item["restoration_file"] = str((draw_dir/"restoration.json").relative_to(output))
            item["restoration_sha256"] = file_hash(draw_dir/"restoration.json")
            pending = [arm for arm in item["numerics"] if arm["status"]=="own_maps_complete_pending_trained_responses"]
            if not pending:
                continue
            try:
                ctx, provenance = context_loader(entry)
                if tensor_hash(ctx["ref_action"]) != trained[entry["context_id"]]["reference_sha256"]:
                    error=FloatingPointError("Restored trained probe differs from the pre-intervention reference")
                    error.raw_tensors={"restored_reference":ctx["ref_action"]}
                    raise error
            except (NonFiniteAttributionError,FloatingPointError) as error:
                from experiment_io import strict_json
                for arm in pending:
                    path=output/arm["report_file"]
                    failed=strict_json(path)
                    failed.update(failure_record(error,path.parent,"restored-reference-failure.pt","trained_probe_repeatability"))
                    checkpoint(path,failed)
                    arm.update(status="numerical_failure",report_sha256=file_hash(path))
                item["status"]="numerical_failure"
                checkpoint(output/"report.json",progress)
                continue
            for arm in pending:
                from experiment_io import strict_json
                path = output/arm["report_file"]
                result = complete_trained_responses(ctx, strict_json(path), decision, path.parent)
                arm.update(status=result["status"], report_sha256=file_hash(path))
                checkpoint(output/"report.json", progress)
            item["status"] = "numerical_failure" if any(arm["status"]=="numerical_failure" for arm in item["numerics"]) else "evaluated"
            item["restored_trained_reference_sha256"] = provenance["probe_reference_sha256"]
            del ctx
        for item in items:
            for arm in item["numerics"]:
                if arm.get("report_file"):
                    arm["report_sha256"] = file_hash(output/arm["report_file"])
            if item["status"] == "planned":
                item["status"] = "numerical_failure" if any(arm["status"]=="numerical_failure" for arm in item["numerics"]) else "evaluated"
        progress["draws"].append(dict(**restoration, draw=identity, elapsed_seconds=time.perf_counter()-started))
        checkpoint(output/"report.json", progress)


def load_inputs(args):
    from experiment_io import file_hash, strict_json
    from scripts.validate_fp32_probe import load_bank_inputs
    result = load_bank_inputs(args)
    decision, bank, selected, replay_args, manifest, digest, selected_rows = result
    native = strict_json(args.native_registry)
    validate_decision(decision, bank, file_hash(args.bank), file_hash(args.native_registry))
    pipeline=manifest["configuration"]["pipeline"]
    if (decision["checkpoint_identity"] != pipeline["checkpoint"] or
            decision["checkpoint_mode"] != pipeline["checkpoint_mode"]):
        raise ValueError("Declared checkpoint differs from authenticated source checkpoint")
    if file_hash(args.decision_file) != args.decision_sha256:
        raise ValueError("E05 decision differs from its trusted digest")
    return (*result, native)


def prepare(args, output, startup):
    """Same shared cache schema, sealed only after all E05 source checks."""
    import fp32_probe_cache as cache_helper
    from experiment_io import file_hash
    from faithfulness import replay_pipeline, replay_context, load_sidecar
    from scripts.validate_downstream_precision import cache_context, cache_identity
    from scripts.validate_fp32_probe import save_tensors, _complete
    decision,bank,selected,replay_args,manifest,digest,selected_rows,_ = load_inputs(args)
    applied=cache_helper.apply_settings(decision["source_replay_settings"])
    stored=manifest["configuration"].get("runtime_settings")
    if stored is not None and stored != applied:
        raise ValueError("Source manifest runtime differs from the E05 replay declaration")
    hashes=implementation_hashes()
    inputs=input_hashes(args,digest)
    report=dict(schema_version=1,status="preparing",kind="fp32_probe_source_preparation",
        decision_sha256=args.decision_sha256,bank_sha256=file_hash(args.bank),source_metrics_sha256=digest,
        source_run_id=manifest["run_id"],source_configuration_sha256=manifest["configuration_sha256"],
        source_pipeline=manifest["configuration"]["pipeline"],source_language=manifest["configuration"]["language"],
        source_runtime=manifest["configuration"]["environment"],source_forward_sha256=manifest["configuration"]["source_sha256"],
        startup=startup,replay_settings=applied,implementation_sha256=hashes,
        native_registry_sha256=file_hash(args.native_registry),
        source_settings_authentication="manifest" if stored is not None else "protocol_declared_exact_reference_replay_only",
        qualification="Exact source replay verified. Historical flag equality is not asserted when the source did not record flags.",contexts=[])
    checkpoint(output/"report.json",report)
    pipe,language=replay_pipeline(replay_args,manifest)
    for entry in selected:
        item=dict(entry);report["contexts"].append(item)
        if entry["status"] == "available":
            row=selected_rows[entry["episode"],entry["policy_call_idx"]]
            payload,payload_hash=load_sidecar(row,args.metrics,manifest)
            ctx=replay_context(replay_args,row,payload,pipe,language,verify_reference=True,strict=True)
            cache=cache_context(ctx)
            item.update(source_replay_verified=True,sidecar_sha256=payload_hash,cached_tensors=cache_identity(cache),
                cache_file=save_tensors(output/f"ep{entry['episode']:06d}-call{entry['policy_call_idx']:04d}.pt",cache))
            del ctx,cache,payload
        checkpoint(output/"report.json",report)
    for path,expected in {**inputs,**{ROOT/name:value for name,value in hashes.items()}}.items():
        if file_hash(path) != expected:
            raise ValueError("E05 preparation source or input changed during replay")
    if cache_helper.runtime_settings() != applied:
        raise ValueError("E05 preparation changed source runtime settings")
    report["status"]="complete_source_preparation"
    checkpoint(output/"report.json",report)
    _complete(output,report["status"])


def run(args, output, startup):
    import torch
    import paired_comparison as paired
    import weight_arrangement_control as control
    import fp32_probe_cache as cache_helper
    from experiment_io import file_hash, object_hash
    from faithfulness import replay_pipeline, load_sidecar
    from scripts.validate_downstream_precision import convert_downstream_to_fp32, fp32_math_settings
    from scripts.validate_fp32_probe import _complete
    if not startup.get("torch_not_preimported"):
        raise RuntimeError("E05 numerical validation requires a fresh candidate process")
    decision, bank, selected, replay_args, manifest, digest, selected_rows, native = load_inputs(args)
    source_hashes = implementation_hashes()
    inputs = input_hashes(args,digest)
    applied = cache_helper.apply_settings(cache_helper.strict_candidate_settings())
    preparation = cache_helper.verify_preparation(args.prepared, args.prepared_sha256)
    if (preparation["decision_sha256"] != args.decision_sha256 or preparation["bank_sha256"] != file_hash(args.bank) or
            preparation["source_metrics_sha256"] != digest or preparation["replay_settings"] != decision["source_replay_settings"]):
        raise ValueError("Prepared E05 cache decision, bank, source or runtime differs")
    if (preparation.get("implementation_sha256") != source_hashes or
            preparation.get("native_registry_sha256") != file_hash(args.native_registry)):
        raise ValueError("Prepared E05 cache wrapper sources or native registry differ")
    pipe, _ = replay_pipeline(replay_args, manifest)
    conversion = convert_downstream_to_fp32(pipe["runner"])
    structure = control.structure_from_native_registry(native, bank["model"])
    session = control.ArrangementSession(pipe["runner"], structure)
    device = next(pipe["runner"].parameters()).device
    report = dict(kind="weight_arrangement_numerical_validation", schema_version=1, status="running",
        decision=decision, decision_sha256=args.decision_sha256, bank=bank, bank_sha256=file_hash(args.bank),
        source_metrics_sha256=digest, source_manifest=manifest, implementation_sha256=source_hashes,
        production_implementation_sha256=control.implementation_hashes(), parameter_registry=session.registry,
        native_registry_sha256=file_hash(args.native_registry), structure_sha256=object_hash(structure),
        native_registry_root_qualification="Only the inspected torch.nn.Module shell root maps to pinned RDTRunner; every other class, name and shape must match.",
        preparation_completion_sha256=args.prepared_sha256, startup=startup, conversion=conversion,
        settings_transition=dict(source_replay=preparation["replay_settings"], candidate=applied,
            source_settings_authentication=preparation["source_settings_authentication"]),
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
            device_name=torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
            device_capability=list(torch.cuda.get_device_capability(device)) if device.type=="cuda" else None,
            cudnn=torch.backends.cudnn.version(), settings=applied),
        planned_contexts=len(selected), parameter_draws=decision["parameter_draws"],
        planned_context_draw_target_modality_cells=len(selected)*decision["parameter_draws"]*6,
        planned_roster=selected, contexts=[], draws=[])
    checkpoint(output/"report.json", report)
    def context_loader(entry):
        source_row = selected_rows[entry["episode"], entry["policy_call_idx"]]
        payload, payload_hash = load_sidecar(source_row, args.metrics, manifest)
        cache, record = cache_helper.load_prepared_context(args.prepared, preparation, entry, manifest, source_row, payload)
        source = {key:value.to(device) for key,value in cache.items()}
        ctx, identity = paired.make_fp32_probe_context(source, pipe["runner"])
        identity.update(prepared_cache=record["cache_file"], source_sidecar_sha256=payload_hash)
        return ctx, identity
    episodes = defaultdict(list)
    for entry in selected: episodes[entry["episode_id"]].append(entry)
    with fp32_math_settings():
        for entries in episodes.values():
            run_episode(session, sorted(entries,key=lambda e:e["policy_call_idx"]), decision, context_loader, output, report)
    session.verify(session.hashes)
    cache_helper.verify_preparation(args.prepared, args.prepared_sha256)
    for path, expected in {**inputs, **{ROOT/name:value for name,value in source_hashes.items()}}.items():
        if file_hash(path) != expected:
            raise ValueError("E05 numerical source or input changed during execution")
    if cache_helper.runtime_settings() != applied:
        raise ValueError("E05 numerical runtime changed during execution")
    cells = [arm for item in report["contexts"] for arm in item["numerics"]]
    if len(cells) != report["planned_context_draw_target_modality_cells"] or any(arm["status"] in {"not_yet_attempted","own_maps_complete_pending_trained_responses"} for arm in cells):
        raise ValueError("E05 planned numerical roster was not fully accounted")
    report.update(status="complete_diagnostics_not_approval", numerical_failure_cells=sum(arm["status"]=="numerical_failure" for arm in cells),
                  unavailable_cells=sum(arm["status"]=="source_unavailable" for arm in cells))
    checkpoint(output/"report.json", report)
    _complete(output, report["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    for name in ("metrics", "bank", "decision-file", "native-registry", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--lang-dir", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-sha256")
    args = parser.parse_args()
    if args.stage == "run" and (not args.prepared or not args.prepared_sha256):
        parser.error("run requires the authenticated prepared cache and its completion digest")
    from experiment_io import strict_json
    decision = strict_json(args.decision_file)
    startup = process_startup(args.stage, decision.get("source_replay_settings"))
    output = args.out.resolve(); output.mkdir(parents=True, exist_ok=False)
    try:
        if args.stage == "prepare":
            prepare(args, output, startup)
        else:
            run(args, output, startup)
    except Exception as error:
        from experiment_io import file_hash
        checkpoint(output/"failure.json",dict(status="failed", exception_type=type(error).__name__,
            reason=str(error), diagnostics=getattr(error,"diagnostics",{}), startup=startup, script_sha256=file_hash(__file__)))
        raise


if __name__ == "__main__":
    main()
