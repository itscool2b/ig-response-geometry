"""NHSJS revision GPU jobs: integration arms, method rankings and common responses.

A worker process loads the policy once and runs its assigned phases for every
bank context in a static partition:

- ``bf16``: exact source replay; the historical bf16 replica (arm A0) and the
  nested bf16-forward grid with fp32 accumulation (arm A1) for every modality
  and target; the method and fixed-order rankings; the source cache that the
  fp32 phase starts from.
- ``fp32``: the fp32 forward on that cache (A2 nested grid, A3 Gauss-Legendre,
  A4 adaptive Gauss-Kronrod) under strict fp32 settings.
- ``scoring``: every ranking of the context, from both map phases plus the
  random-order and fixed-order controls, scored by one common bf16 forward
  under Q, N and RMS. Every mask and active action is stored.

Each phase writes one directory per context. It becomes visible only when a
completion record binds every file in it to the protocol, bank and source
hashes; an interrupted attempt is moved aside and kept, never reused.

The process must start through ``main`` (or ``python revision_numerics.py``):
CUBLAS is configured from the protocol before torch is imported.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_SCHEMA = "nhsjs_revision_gpu_protocol_v1"
PHASES = ("verify", "bf16", "fp32", "scoring")  # verify is smoke-only
TARGETS = {"Q": "Q", "N": "L2"}  # paper name -> common_scores key
SOURCE_FILES = ("revision_numerics.py", "ig_quadrature.py", "paired_comparison.py", "faithfulness.py",
                "integrated_gradients.py", "experiment_io.py", "pipeline.py", "per_step_attribution.py",
                "rdt_sampling.py", "checkpoint_contract.py", "fp32_probe_cache.py",
                "scripts/validate_downstream_precision.py")
HISTORICAL_KERNEL = ("docs/tmlr-readiness/archives/active-originals/2026-10-01-stable-base/"
                     "integrated_gradients.py.historical")


def read_json(path):
    """strict_json without importing torch-dependent modules."""
    from experiment_io import strict_json
    return strict_json(path)


def startup(protocol, bank_id, phases):
    """Configure CUBLAS for this worker's settings before torch is imported."""
    if "torch" in sys.modules:
        raise RuntimeError("Start the worker in a fresh process before torch import")
    settings = phase_settings(protocol, bank_id, phases)
    inherited = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    value = settings["cublas_workspace_config"]
    if value is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
    return dict(torch_not_preimported=True, inherited_cublas_workspace_config=inherited,
                effective_cublas_workspace_config=value, settings=settings)


def phase_settings(protocol, bank_id, phases):
    """One process holds one settings bundle; fp32 always uses the strict bundle."""
    bank = protocol["banks"][bank_id]
    wanted = []
    for phase in phases:
        if phase not in PHASES:
            raise ValueError(f"Unknown phase {phase}")
        wanted.append(protocol["strict_settings"] if phase == "fp32" else bank["source_replay_settings"])
    if any(item != wanted[0] for item in wanted):
        raise ValueError("Requested phases need different runtime settings; run them in separate processes")
    return wanted[0]


def jsonable(value):
    """JSON-ready copy: nonfinite floats become the strings NaN, Infinity, -Infinity."""
    if isinstance(value, float):
        return value if math.isfinite(value) else ("NaN" if value != value else "Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return jsonable(value.item())
    raise TypeError(f"Unsupported record value {type(value).__name__}")


# Everything below may import torch.

def count_schedule(n, grid, rule):
    """Groups removed at each nominal percentage.

    ``round`` is paired_comparison.evaluate_rankings. ``max1_round`` is the rule
    of the saved headline runs (faithfulness.py at afd1249, b887a96 and e4ad8bf):
    every interior percentage removes at least one group and at most all of them.
    """
    if rule == "round":
        return [round(n * k / 100) for k in grid]
    if rule == "max1_round":
        return [0 if k == 0 else n if k == 100 else min(n, max(1, int(round(n * k / 100)))) for k in grid]
    raise ValueError(f"Unknown count rule {rule}")


def fixed_orders(modality, indices):
    """Data-independent orders: the fixed-order controls of the methods study."""
    indices = list(indices)
    if modality == "vision":
        side = math.isqrt(len(indices))
        if side * side != len(indices):
            raise ValueError("Vision population is not a square patch grid")
        centre = (side - 1) / 2
        def distance(position):
            row, col = divmod(position, side)
            return (row - centre) ** 2 + (col - centre) ** 2
        centre_out = sorted(range(len(indices)), key=lambda p: (distance(p), p))
        return {"fixed_raster": indices, "fixed_reverse_raster": indices[::-1],
                "fixed_centre_out": [indices[p] for p in centre_out]}
    if modality == "language":
        return {"fixed_token_position": indices, "fixed_reverse_token_position": indices[::-1]}
    if modality == "state":
        return {"fixed_joint_order": indices, "fixed_reverse_joint_order": indices[::-1]}
    raise ValueError(f"Unknown modality {modality}")


def order_ranking(order, kind):
    return dict(status="defined", order=list(order), tie_rule=kind, control_kind=kind)


def evaluate_orders(action_forward, reference, actual, baseline, indices, axis, rankings, grid,
                    denominator_min, *, count_rule="round"):
    """paired_comparison.evaluate_rankings with the count rule as a parameter.

    With count_rule="round" the output equals evaluate_rankings exactly (tested).
    Nominal-axis normalized areas are added beside the realized-fraction ones.
    """
    from itertools import combinations
    import torch
    from scipy.integrate import trapezoid
    from experiment_io import object_hash, tensor_hash
    from faithfulness import auc_normalized
    from integrated_gradients import NonFiniteAttributionError
    from paired_comparison import common_scores, replace_positions, validate_grid, validate_population
    from per_step_attribution import MANISKILL_INDICES
    validate_population(actual, baseline, indices, axis)
    validate_grid(grid)
    counts = count_schedule(len(indices), grid, count_rule)
    fractions = [count / len(indices) for count in counts]
    responses = {}
    active_reference = reference[..., MANISKILL_INDICES].detach().float()

    def evaluate(baseline_positions):
        key = object_hash(sorted(baseline_positions))
        if key not in responses:
            value = replace_positions(actual, baseline, baseline_positions, axis)
            try:
                with torch.no_grad():
                    action = action_forward(value)
                    scores = common_scores(action, reference)
                if not torch.isfinite(action).all() or any(not torch.isfinite(x) for x in scores.values()):
                    raise FloatingPointError("Nonfinite common action response")
                if not baseline_positions and tensor_hash(action) != tensor_hash(reference):
                    raise FloatingPointError("Probe self-reference replay differs during common-response evaluation")
                active_action = action[..., MANISKILL_INDICES].detach().float()
                responses[key] = dict(status="finite", baseline_positions=sorted(baseline_positions),
                    input_sha256=tensor_hash(value), action_sha256=tensor_hash(action),
                    active_action=active_action.cpu().tolist(), active_action_sha256=tensor_hash(active_action),
                    **{target: float(score.item()) for target, score in scores.items()})
            except (NonFiniteAttributionError, FloatingPointError) as exc:
                responses[key] = dict(status="numerical_failure", baseline_positions=sorted(baseline_positions),
                    failure_kind="probe_self_reference" if "self-reference" in str(exc) else "nonfinite_intervention_response",
                    reason=str(exc), diagnostics=getattr(exc, "diagnostics", {}), Q=None, L2=None, RMS=None)
        return key

    input_id, baseline_id = evaluate([]), evaluate(indices)
    for ranking in rankings.values():
        if ranking["status"] != "defined":
            continue
        exact = ranking.get("control_kind") == "exact_uniform_subset_expectation"
        if exact:
            if len(indices) > 8:
                raise ValueError("Exact random subset policy is restricted to at most eight groups")
            masks = [[list(group) for group in combinations(indices, count)] for count in counts]
        else:
            order = ranking["order"]
            if sorted(order) != indices:
                raise ValueError("Ranking does not permute the exact eligible population")
            ranking["ranking_sha256"] = object_hash(order)
            masks = [[sorted(order[:count])] for count in counts]
        curves = {}
        for direction in ("deletion", "insertion"):
            response_ids = [[evaluate(mask if direction == "deletion" else sorted(set(indices) - set(mask)))
                             for mask in groups] for groups in masks]
            by_target = {}
            for target in ("Q", "L2", "RMS"):
                groups = [[responses[key][target] for key in keys] for keys in response_ids]
                values = [None if any(v is None for v in group) else math.fsum(group)/len(group) for group in groups]
                endpoints = [responses[input_id][target], responses[baseline_id][target]]
                if any(x is None for x in values + endpoints):
                    by_target[target] = dict(values=values, raw_auc=None, normalized_auc=None,
                                             endpoint_gap=None, status="nonfinite_response",
                                             nominal_normalized_auc=None)
                else:
                    gap = endpoints[0] - endpoints[1]
                    cutoff = denominator_min.get(target, 0)
                    normalized = auc_normalized([100*f for f in fractions], values, *endpoints,
                                                denominator_min=cutoff)
                    status = "zero_endpoint_gap" if gap == 0 else "nearzero_endpoint_gap" if abs(gap) <= cutoff else "defined"
                    by_target[target] = dict(values=values, raw_auc=float(trapezoid(values, fractions)),
                                             normalized_auc=normalized, endpoint_gap=gap, status=status,
                                             nominal_normalized_auc=auc_normalized(list(grid), values, *endpoints,
                                                                                   denominator_min=cutoff))
            curves[direction] = dict(selected_positions=masks if exact else [group[0] for group in masks],
                mask_sha256=[[object_hash(mask) for mask in group] for group in masks] if exact else [object_hash(group[0]) for group in masks],
                response_ids=response_ids if exact else [group[0] for group in response_ids], responses=by_target)
            if exact:
                curves[direction]["subset_counts"] = [len(group) for group in masks]
                curves[direction]["expectation"] = "equal mean over all distinct subsets at each realized count"
        ranking["curves"] = curves
    return dict(rankings=rankings, response_table=responses, eligible_indices=indices,
        nominal_grid_percent=grid, selected_counts=counts, realized_fractions=fractions, count_rule=count_rule,
        input_response_id=input_id, baseline_response_id=baseline_id, denominator_min=denominator_min,
        active_action_indices=list(MANISKILL_INDICES), active_reference_action=active_reference.cpu().tolist(),
        active_reference_sha256=tensor_hash(active_reference),
        response_definition=dict(Q="-sum_FP32((action-reference)^2)/(2D)",
            L2="-sqrt(sum_FP32((action-reference)^2)+1e-12)",
            RMS="sqrt(sum_FP32((action-reference)^2)/D)",
            RMS_units="normalized active-action coordinates; positive displacement without stabilizer"))


# ----------------------------------------------------------------------------
# Phase storage

class PhaseOutput:
    """One context's phase directory, published by rename after its completion."""

    def __init__(self, root, bank_id, context_key, phase, binding):
        self.final = Path(root) / bank_id / context_key / phase
        self.attempts = Path(root) / bank_id / context_key / "attempts"
        self.binding = binding
        self.files = {}

    def complete(self):
        return (self.final / "completion.json").is_file()

    def __enter__(self):
        if self.final.exists():
            raise FileExistsError(f"Phase output exists: {self.final}")
        self.attempts.mkdir(parents=True, exist_ok=True)
        self.work = self.attempts / f"{self.final.name}-{uuid.uuid4().hex}"
        self.work.mkdir()
        self.started = time.time()
        return self

    def write_json(self, name, value):
        from experiment_io import canonical_json, file_hash
        path = self.work / name
        with path.open("xb") as handle:
            handle.write(canonical_json(jsonable(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.files[name] = file_hash(path)
        return self.files[name]

    def write_tensors(self, name, values):
        import torch
        from experiment_io import file_hash
        path = self.work / name
        with path.open("xb") as handle:
            torch.save({key: value.detach().cpu().contiguous() for key, value in values.items()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        self.files[name] = file_hash(path)
        return dict(file=name, sha256=self.files[name])

    def __exit__(self, kind, exception, traceback):
        from experiment_io import canonical_json, file_hash
        if exception is not None:
            failure = dict(status="failed", error_type=kind.__name__, reason=str(exception),
                           binding=self.binding, files_sha256=self.files, started_unix=self.started,
                           ended_unix=time.time())
            try:
                failure["diagnostics"] = jsonable(getattr(exception, "diagnostics", {}))
            except TypeError:
                failure["diagnostics"] = repr(getattr(exception, "diagnostics", {}))[:4000]
            (self.work / "failure.json").write_bytes(canonical_json(failure) + b"\n")
            return False
        for name, digest in self.files.items():
            if file_hash(self.work / name) != digest:
                raise ValueError(f"Phase file changed before completion: {name}")
        completion = dict(status="complete", binding=self.binding, files_sha256=self.files,
                          started_unix=self.started, ended_unix=time.time(), host=platform.node())
        path = self.work / "completion.json"
        with path.open("xb") as handle:
            handle.write(canonical_json(completion) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.final.parent.mkdir(parents=True, exist_ok=True)
        os.rename(self.work, self.final)
        return False


def load_phase(root, bank_id, context_key, phase, binding):
    """Authenticated read of a completed phase directory."""
    from experiment_io import file_hash
    directory = Path(root) / bank_id / context_key / phase
    if not (directory / "completion.json").is_file():
        raise ValueError(f"Phase {phase} is not complete for {context_key}; it must run first")
    completion = read_json(directory / "completion.json")
    if completion.get("status") != "complete":
        raise ValueError(f"Incomplete phase {phase} for {context_key}")
    for key in ("protocol_sha256", "bank_sha256", "context_id"):
        if completion["binding"].get(key) != binding[key]:
            raise ValueError(f"Phase {phase} binds a different {key}")
    present = {p.name for p in directory.iterdir()} - {"completion.json"}
    if present != set(completion["files_sha256"]):
        raise ValueError(f"Phase {phase} directory holds unlisted or missing files")
    for name, digest in completion["files_sha256"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"Phase {phase} file hash differs: {name}")
    return directory, completion


# ----------------------------------------------------------------------------
# Arms

def _target_fn(forward, reference, target):
    from paired_comparison import common_scores
    key = TARGETS[target]
    return lambda x: common_scores(forward(x), reference)[key]


def _residuals(rules, expected_gap):
    import ig_quadrature as q
    return {name: q.attribution_summary(value, expected_gap) for name, value in rules.items()}


def _group_scores(eligible, tensor):
    """Group |attribution| sums in the map's own dtype, as paired_comparison.grouped_scores
    and the historical bf16 ranking reduce them."""
    return eligible.group_scores(tensor)


def _comparisons(reference, candidates, eligible, counts):
    import ig_quadrature as q
    return {name: q.compare_maps(reference, value, eligible, counts=counts) for name, value in candidates.items()}


def _ranking_from_scores(scores, indices):
    import torch
    from paired_comparison import ranked
    if not torch.isfinite(scores).all():
        return dict(status="undefined", reason="nonfinite group scores")
    return ranked(scores.float().cpu(), indices)


def _keep_rule(name, keep):
    return keep == "all" or name in keep


def bf16_phase(ctx, protocol, out, *, modalities, store_nodes):
    """A0 and A1 for every modality and target, plus the bf16-phase rankings."""
    import torch
    import ig_quadrature as q
    from experiment_io import tensor_hash
    from paired_comparison import grouped_scores, modality_inputs
    numerics, rank_cfg, evaluation = protocol["numerics"], protocol["rankings"], protocol["evaluation"]
    record, tensors, rankings = dict(modalities={}), {}, {}
    for modality in modalities:
        actual, baseline, indices, axis, forward = modality_inputs(ctx, modality)
        eligible = q.Eligible(indices, axis)
        counts = count_schedule(len(indices), evaluation["grid_percent"], evaluation["count_rule"])
        entry = dict(eligible_count=len(indices), axis=axis, input_sha256=tensor_hash(actual),
                     baseline_sha256=tensor_hash(baseline), input_dtype=str(actual.dtype), targets={})
        modality_rankings = {"input_difference": _ranking_from_scores(
            grouped_scores(actual.float() - baseline.float(), indices, axis).cpu(), indices)}
        modality_rankings.update({name: order_ranking(order, name) for name, order in fixed_orders(modality, indices).items()})
        keep = numerics["store_rules"][modality]
        for target in TARGETS:
            score_fn = _target_fn(forward, ctx["ref_action"], target)
            target_record = dict(historical={}, nested=None)
            for m in numerics["historical_budgets"]:
                started = time.perf_counter()
                hist = q.historical_pass(score_fn, actual, baseline, m, eligible=eligible)
                name = f"A0m{m}"
                tensors[f"{modality}/{target}/{name}"] = hist["attribution"]
                tensors[f"{modality}/{target}/{name}/group_scores"] = _group_scores(eligible, hist["attribution"])
                modality_rankings[f"IG_{target}_{name}"] = _ranking_from_scores(
                    _group_scores(eligible, hist["attribution"]), indices)
                target_record["historical"][name] = {key: value for key, value in hist.items()
                                                     if key not in ("attribution",)}
                target_record["historical"][name]["wall_seconds"] = time.perf_counter() - started
            keep_nodes = modality in store_nodes
            nested = q.dyadic_pass(score_fn, actual, baseline, numerics["levels"], model_dtype=actual.dtype,
                                   eligible=eligible, master_dtype=torch.float32, keep_gradients=keep_nodes)
            summary = {key: value for key, value in nested.items()
                       if key not in ("rules", "path_gradients", "displacement", "node_gradients")}
            if nested["defined"]:
                rules = nested["rules"]
                comparator = rules[numerics["comparator_rule"]]
                summary["residuals"] = _residuals(rules, nested["expected_gap"])
                hist_maps = {f"A0m{m}": tensors[f"{modality}/{target}/A0m{m}"] for m in numerics["historical_budgets"]}
                summary["versus_comparator"] = _comparisons(comparator, {**rules, **hist_maps}, eligible, counts)
                for name, value in rules.items():
                    tensors[f"{modality}/{target}/A1{name}/group_scores"] = _group_scores(eligible, value)
                    if _keep_rule(name, keep):
                        tensors[f"{modality}/{target}/A1{name}"] = value
                for name in rank_cfg["ig_rules"]:
                    modality_rankings[f"IG_{target}_A1{name}"] = _ranking_from_scores(
                        _group_scores(eligible, rules[name]), indices)
                for name in rank_cfg["path_gradient_rules"]:
                    gradient = nested["path_gradients"][name]
                    tensors[f"{modality}/{target}/A1PG{name}"] = gradient
                    modality_rankings[f"PG_{target}_A1{name}"] = _ranking_from_scores(
                        _group_scores(eligible, gradient), indices)
            else:
                for name in rank_cfg["ig_rules"]:
                    modality_rankings[f"IG_{target}_A1{name}"] = dict(status="undefined", reason="nonfinite node")
                for name in rank_cfg["path_gradient_rules"]:
                    modality_rankings[f"PG_{target}_A1{name}"] = dict(status="undefined", reason="nonfinite node")
            if keep_nodes and nested["node_gradients"]:
                stacked = torch.stack([nested["node_gradients"][i] for i in sorted(nested["node_gradients"])])
                info = out.write_tensors(f"nodes-{modality}-{target}.pt", {"node_gradients": stacked,
                    "nodes": torch.tensor(sorted(nested["node_gradients"])), "displacement": nested["displacement"]})
                summary["node_file"] = info
            target_record["nested"] = summary
            entry["targets"][target] = target_record
            del nested
        rankings[modality] = modality_rankings
        record["modalities"][modality] = entry
    return record, tensors, rankings


def fp32_phase(ctx32, protocol, out, *, modalities, adaptive, store_nodes):
    """A2, A3 and (on its subset) A4 on the fp32 probe context."""
    import torch
    import ig_quadrature as q
    from experiment_io import tensor_hash
    from paired_comparison import modality_inputs
    numerics, rank_cfg, evaluation = protocol["numerics"], protocol["rankings"], protocol["evaluation"]
    record, tensors, rankings = dict(modalities={}), {}, {}
    for modality in modalities:
        actual, baseline, indices, axis, forward = modality_inputs(ctx32, modality)
        if actual.dtype != torch.float32:
            raise ValueError("fp32 phase received a non-fp32 modality input")
        eligible = q.Eligible(indices, axis)
        counts = count_schedule(len(indices), evaluation["grid_percent"], evaluation["count_rule"])
        entry = dict(eligible_count=len(indices), axis=axis, input_sha256=tensor_hash(actual),
                     baseline_sha256=tensor_hash(baseline), targets={})
        modality_rankings = {}
        keep = numerics["store_rules"][modality]
        for target in TARGETS:
            score_fn = _target_fn(forward, ctx32["ref_action"], target)
            target_record = {}
            nested = q.dyadic_pass(score_fn, actual, baseline, numerics["levels"], model_dtype=torch.float32,
                                   eligible=eligible, master_dtype=torch.float32,
                                   keep_gradients=modality in store_nodes)
            summary = {key: value for key, value in nested.items()
                       if key not in ("rules", "path_gradients", "displacement", "node_gradients")}
            maps = {}
            if nested["defined"]:
                maps.update({f"A2{name}": value for name, value in nested["rules"].items()})
                summary["residuals"] = _residuals(nested["rules"], nested["expected_gap"])
            if modality in store_nodes and nested["node_gradients"]:
                stacked = torch.stack([nested["node_gradients"][i] for i in sorted(nested["node_gradients"])])
                summary["node_file"] = out.write_tensors(f"nodes-{modality}-{target}.pt", {"node_gradients": stacked,
                    "nodes": torch.tensor(sorted(nested["node_gradients"])), "displacement": nested["displacement"]})
            target_record["nested"] = summary
            del nested
            target_record["gauss_legendre"] = {}
            for n in numerics["gauss_legendre"]:
                gl = q.gauss_legendre_pass(score_fn, actual, baseline, n, model_dtype=torch.float32,
                                           eligible=eligible, master_dtype=torch.float32)
                target_record["gauss_legendre"][f"GL{n}"] = {key: value for key, value in gl.items()
                                                             if key not in ("attribution", "path_gradient")}
                if gl["defined"]:
                    maps[f"A3GL{n}"] = gl["attribution"].float()
                    target_record["gauss_legendre"][f"GL{n}"]["summary"] = q.attribution_summary(
                        gl["attribution"], gl["expected_gap"])
            if adaptive:
                settings_a4 = numerics["adaptive"]
                result = q.adaptive_pass(score_fn, actual, baseline, model_dtype=torch.float32, eligible=eligible,
                                         rtol=settings_a4["rtol"], max_intervals=settings_a4["max_intervals"],
                                         snapshot_evaluations=tuple(settings_a4["snapshot_evaluations"]))
                target_record["adaptive"] = {key: value for key, value in result.items()
                                             if key not in ("attribution", "snapshots")}
                target_record["adaptive"]["snapshots"] = {str(k): dict(evaluations=v["evaluations"], error_l1=v["error_l1"])
                                                          for k, v in result["snapshots"].items()}
                if result["defined"]:
                    maps["A4"] = result["attribution"].float()
                    target_record["adaptive"]["summary"] = q.attribution_summary(result["attribution"],
                                                                                 result["expected_gap"])
                    for k, v in result["snapshots"].items():
                        maps[f"A4snap{k}"] = v["attribution"].float()
            reference_name = "A4" if adaptive and maps.get("A4") is not None and target_record["adaptive"]["converged"] \
                else f"A2{numerics['comparator_rule']}"
            if reference_name in maps:
                target_record["reference"] = reference_name
                target_record["versus_reference"] = _comparisons(maps[reference_name], maps, eligible, counts)
                pairs = numerics["qualification_triple"]
                if all(name in maps for name in pairs):
                    target_record["qualification"] = {f"{a}|{b}": q.compare_maps(maps[a], maps[b], eligible, counts=counts)
                                                      for i, a in enumerate(pairs) for b in pairs[i + 1:]}
            for name, value in maps.items():
                tensors[f"{modality}/{target}/{name}/group_scores"] = _group_scores(eligible, value)
                if not name.startswith("A2") or _keep_rule(name[2:], keep):
                    tensors[f"{modality}/{target}/{name}"] = value
            for name in rank_cfg["fp32_rules"]:
                modality_rankings[f"IG_{target}_{name}"] = (_ranking_from_scores(_group_scores(eligible, maps[name]), indices)
                                                            if name in maps else dict(status="undefined", reason="not computed or nonfinite"))
            entry["targets"][target] = target_record
        rankings[modality] = modality_rankings
        record["modalities"][modality] = entry
    return record, tensors, rankings


def random_rankings(modality, indices, context_id, protocol):
    from paired_comparison import random_order
    evaluation = protocol["evaluation"]
    if len(indices) <= 8:
        return {"random_exact": dict(status="defined", control_kind="exact_uniform_subset_expectation",
                                     eligible_count=len(indices), monte_carlo_variance=0.0)}
    count = evaluation["random_permutations"][modality]
    return {f"random_{replicate:04d}": random_order(indices, evaluation["random_seed"], context_id, modality, replicate)
            for replicate in range(count)}


def scoring_phase(ctx, protocol, phase_rankings, context_id, *, modalities):
    """Every ranking of the context under one common bf16 forward."""
    from paired_comparison import modality_inputs
    evaluation = protocol["evaluation"]
    record = dict(modalities={})
    for modality in modalities:
        actual, baseline, indices, axis, forward = modality_inputs(ctx, modality)
        rankings = {}
        for phase, by_modality in phase_rankings.items():
            for name, ranking in by_modality.get(modality, {}).items():
                key = name if phase == "bf16" else f"{phase}:{name}"
                if key in rankings:
                    raise ValueError(f"Duplicate ranking {key}")
                rankings[key] = json.loads(json.dumps(ranking))
        rankings.update(random_rankings(modality, indices, context_id, protocol))
        started = time.perf_counter()
        result = evaluate_orders(forward, ctx["ref_action"], actual, baseline, indices, axis, rankings,
                                 evaluation["grid_percent"], evaluation["denominator_min"],
                                 count_rule=evaluation["count_rule"])
        result["wall_seconds"] = time.perf_counter() - started
        record["modalities"][modality] = result
    return record


# ----------------------------------------------------------------------------
# Worker

class ReplayArgs:
    """The attribute set faithfulness replay helpers read from argparse."""

    def __init__(self, bank_cfg):
        self.metrics = bank_cfg["metrics"]
        self.task = bank_cfg["task"]
        self.model = bank_cfg["model"]
        self.target = None
        self.no_checkpoint = False
        self.out = None
        self.limit = None
        self.legacy_input = False
        self.checkpoint_mode = None
        self.checkpoint_path = bank_cfg.get("checkpoint_path")
        self.model_revision = None
        self.vision_revision = None
        self.lang_dir = bank_cfg["lang_dir"]


def source_hashes():
    from experiment_io import file_hash
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def context_key(entry):
    return f"e{entry['episode']:04d}-c{entry['policy_call_idx']:03d}"


def partition(contexts, worker, workers):
    available = [entry for entry in contexts if entry["status"] == "available"]
    return [entry for i, entry in enumerate(available) if i % workers == worker]


def resolve_members(rule, bank):
    """Subset rules are fixed before collection and never read outcomes."""
    if rule == "all":
        return {entry["context_id"] for entry in bank["contexts"] if entry["status"] == "available"}
    if rule == "none":
        return set()
    kind, _, value = rule.partition(":")
    if kind == "first_contexts":
        available = [entry for entry in bank["contexts"] if entry["status"] == "available"]
        return {entry["context_id"] for entry in available[:int(value)]}
    if kind == "first_episodes":
        limit = int(value)
        return {entry["context_id"] for entry in bank["contexts"]
                if entry["status"] == "available" and entry["episode"] < limit}
    if kind == "contexts":
        wanted = set(value.split(","))
        return {entry["context_id"] for entry in bank["contexts"]
                if entry["status"] == "available" and context_key(entry) in wanted}
    raise ValueError(f"Unknown subset rule {rule}")


OVERRIDABLE_NUMERICS = frozenset({"historical_budgets", "store_rules", "gauss_legendre", "adaptive"})


def bank_protocol(protocol, bank_id):
    """The protocol as one bank sees it: its numerics_override merged over the shared numerics.

    Only budgets and storage may differ between banks; the nested grid, its
    comparator and every ranking and evaluation setting stay common.
    """
    override = protocol["banks"][bank_id].get("numerics_override", {})
    unknown = set(override) - OVERRIDABLE_NUMERICS
    if unknown:
        raise ValueError(f"Bank {bank_id} overrides fixed numerics: {sorted(unknown)}")
    effective = dict(protocol)
    effective["numerics"] = {**protocol["numerics"], **override}
    return effective


def validate_protocol(protocol):
    if protocol.get("schema") != PROTOCOL_SCHEMA or protocol.get("recorded_before_execution") is not True:
        raise ValueError("Unknown or unrecorded revision protocol")
    for key in ("decision_id", "banks", "strict_settings", "numerics", "rankings", "evaluation"):
        if key not in protocol:
            raise ValueError(f"Protocol lacks {key}")
    numerics = protocol["numerics"]
    if type(numerics["levels"]) is not int or not 1 <= numerics["levels"] <= 12:
        raise ValueError("Nested levels must be an integer in [1, 12]")
    if numerics["comparator_rule"] not in {f"T{2 ** l}" for l in range(numerics["levels"] + 1)}:
        raise ValueError("Comparator must be a trapezoid on the nested grid")
    if protocol["evaluation"]["count_rule"] not in ("round", "max1_round"):
        raise ValueError("Unknown count rule")
    return protocol


def load_archived_kernel():
    """The archived historical kernel (it prints its completeness line)."""
    import importlib.machinery
    import importlib.util
    loader = importlib.machinery.SourceFileLoader("archived_historical_ig", str(ROOT / HISTORICAL_KERNEL))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


def verify_phase(ctx, protocol, *, modality="vision"):
    """Smoke gates: repeatability, the pinned kernel and the archived kernel, bit for bit."""
    import contextlib
    import io
    import torch
    import ig_quadrature as q
    from integrated_gradients import NonFiniteAttributionError, integrated_gradients
    from paired_comparison import modality_inputs
    actual, baseline, indices, axis, forward = modality_inputs(ctx, modality)
    eligible = q.Eligible(indices, axis)
    outside = [i for i in range(actual.shape[axis]) if i not in set(indices)]
    archived = load_archived_kernel()
    checks = dict(modality=modality, targets={})
    levels = protocol["verification"]["levels"]
    m = 2 ** levels
    for target in TARGETS:
        score_fn = _target_fn(forward, ctx["ref_action"], target)
        item = {}
        runs = [q.dyadic_pass(score_fn, actual, baseline, levels, model_dtype=actual.dtype, eligible=eligible)
                for _ in range(protocol["verification"]["repeats"])]
        item["repeat_seconds"] = [run["elapsed_seconds"] for run in runs]
        item["repeats_defined"] = [run["defined"] for run in runs]
        item["repeats_bit_identical"] = all(run["defined"] for run in runs) and all(
            torch.equal(runs[0]["rules"][name], run["rules"][name]) for run in runs[1:] for name in runs[0]["rules"])
        try:
            started = time.perf_counter()
            kernel = integrated_gradients(score_fn, actual, baseline, m=m, quadrature="trapezoid", return_result=True)
            selection = [slice(None)] * 3
            selection[axis] = outside
            item["pinned_kernel"] = dict(
                seconds=time.perf_counter() - started,
                trapezoid_bit_identical=bool(runs[0]["defined"]) and torch.equal(
                    eligible.take(kernel.attributions), runs[0]["rules"][f"T{m}"]),
                outside_population_nonzero=int(torch.count_nonzero(kernel.attributions[tuple(selection)])))
        except NonFiniteAttributionError as error:
            item["pinned_kernel"] = dict(status="nonfinite", reason=str(error))
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            archived_map = archived.integrated_gradients(score_fn, actual, baseline, m=m)
        replica = q.historical_pass(score_fn, actual, baseline, m, eligible=eligible)
        item["archived_kernel"] = dict(seconds=time.perf_counter() - started, printed=printed.getvalue().strip(),
                                       bit_identical=torch.equal(eligible.take(archived_map), replica["attribution"]),
                                       replica_nonfinite=replica["nonfinite_totals"])
        checks["targets"][target] = item
        del runs
    return checks


def run_worker(args, protocol, startup_record):
    import torch
    from experiment_io import file_hash, tensor_hash
    from faithfulness import load_sidecar, replay_context, replay_inputs, replay_pipeline
    from fp32_probe_cache import apply_settings, runtime_settings
    from paired_comparison import bank_rows, make_bank, make_fp32_probe_context
    from scripts.validate_downstream_precision import (CACHE_KEYS, cache_context, cache_identity,
                                                       convert_downstream_to_fp32, fp32_math_settings)
    bank_cfg = protocol["banks"][args.bank]
    protocol = bank_protocol(protocol, args.bank)
    protocol_sha = file_hash(args.protocol)
    applied = apply_settings(startup_record["settings"])
    sources = source_hashes()
    if bank_cfg.get("source_sha256") is not None and sources != bank_cfg["source_sha256"]:
        raise ValueError("Worker source differs from the bank's frozen source")
    bank_path = Path(bank_cfg["bank"])
    if file_hash(bank_path) != bank_cfg["bank_sha256"]:
        raise ValueError("Bank file differs from the protocol")
    bank = read_json(bank_path)
    replay = ReplayArgs(bank_cfg)
    rows, manifest, digest = replay_inputs(replay, apply_limit=False)
    selection = bank["selection"]
    rebuilt = make_bank(replay.metrics, None if selection["rule"] == "exact_collector_prespecified_call_indices" else selection)
    if rebuilt != bank:
        raise ValueError("Bank differs from the authenticated collector population")
    selected = bank_rows(bank, rows, manifest, digest)
    modalities = bank_cfg["modalities"]
    fp32_members = resolve_members(bank_cfg["fp32_contexts"], bank)
    adaptive_members = resolve_members(bank_cfg["adaptive_contexts"], bank)
    node_members = resolve_members(bank_cfg["full_node_contexts"], bank)
    verify_members = resolve_members(bank_cfg.get("verify_contexts", "none"), bank)
    if not adaptive_members <= fp32_members:
        raise ValueError("Adaptive contexts must be fp32 contexts")
    phases = args.phases
    pipe = lang = runner32 = conversion = None
    if set(phases) & {"verify", "bf16", "scoring"}:
        pipe, lang = replay_pipeline(replay, manifest)
    if "fp32" in phases:
        pipe32, _ = replay_pipeline(replay, manifest)
        runner32 = pipe32["runner"]
        conversion = convert_downstream_to_fp32(runner32)
        del pipe32
    log_path = Path(args.out) / args.bank / "workers" / (
        f"worker-{args.worker:02d}-of-{args.workers:02d}-{'-'.join(phases)}-{uuid.uuid4().hex[:8]}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("x")

    def note(**event):
        event.update(unix=time.time())
        line = json.dumps(jsonable(event), sort_keys=True)
        log.write(line + "\n")
        log.flush()
        os.fsync(log.fileno())
        print(line, flush=True)

    note(event="worker_start", bank=args.bank, phases=phases, worker=args.worker, workers=args.workers,
         protocol_sha256=protocol_sha, settings=applied, startup=startup_record, sources=sources,
         device=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
         fp32_conversion=conversion, pid=os.getpid())
    done = 0
    for entry in partition(bank["contexts"], args.worker, args.workers):
        key = context_key(entry)
        binding = dict(protocol_sha256=protocol_sha, bank_sha256=bank_cfg["bank_sha256"], context_id=entry["context_id"],
                       bank_id=args.bank, source_sha256=sources, runtime_settings=applied)
        row = selected[(entry["episode"], entry["policy_call_idx"])]
        for phase in phases:
            if phase == "fp32" and entry["context_id"] not in fp32_members:
                continue
            if phase == "verify" and entry["context_id"] not in verify_members:
                continue
            if source_hashes() != sources or file_hash(args.protocol) != protocol_sha:
                raise ValueError("Source or protocol changed during the worker run")
            output = PhaseOutput(args.out, args.bank, key, phase, binding)
            if output.complete():
                continue
            started = time.perf_counter()
            note(event="phase_start", context=key, phase=phase)
            try:
                with output:
                    sidecar, sidecar_digest = load_sidecar(row, replay.metrics, manifest)
                    context_record = dict(context_id=entry["context_id"], episode=entry["episode"],
                                          policy_call_idx=entry["policy_call_idx"], reset_seed=entry["reset_seed"],
                                          sidecar_sha256=sidecar_digest, phase=phase)
                    ctx = None
                    if phase in ("verify", "bf16", "scoring"):
                        ctx = replay_context(replay, row, sidecar, pipe, lang, verify_reference=True, strict=True)
                        context_record.update(reference_sha256=tensor_hash(ctx["ref_action"]),
                                              noise_sha256=tensor_hash(ctx["initial_noise"]), source_replay_verified=True)
                    if phase == "verify":
                        record = dict(checks=verify_phase(ctx, protocol))
                    elif phase == "bf16":
                        store_nodes = set(bank_cfg["full_node_modalities"])
                        if entry["context_id"] in node_members:
                            store_nodes.update(modalities)
                        record, tensors, rankings = bf16_phase(ctx, protocol, output, modalities=modalities,
                                                               store_nodes=store_nodes)
                        if entry["context_id"] in fp32_members:
                            cache = cache_context(ctx)
                            output.write_tensors("source-cache.pt", cache)
                            context_record["source_cache_identity"] = cache_identity(cache)
                        output.write_tensors("maps.pt", tensors)
                        output.write_json("rankings.json", rankings)
                    elif phase == "fp32":
                        source_dir, _ = load_phase(args.out, args.bank, key, "bf16", binding_for_load(binding))
                        cache = torch.load(source_dir / "source-cache.pt", weights_only=True, map_location="cpu")
                        if set(cache) != set(CACHE_KEYS) or cache_identity(cache) != read_json(
                                source_dir / "record.json")["context"]["source_cache_identity"]:
                            raise ValueError("Source cache differs from its bf16-phase record")
                        for name in ("initial_noise", "ref_action", "lang_attn_mask", "action_mask", "ctrl_freqs"):
                            if tensor_hash(cache[name]) != tensor_hash(sidecar[name]):
                                raise ValueError("Source cache differs from the sidecar: " + name)
                        device = next(runner32.parameters()).device
                        with fp32_math_settings() as math_settings:
                            context_record["math_settings"] = math_settings
                            try:
                                ctx32, probe = make_fp32_probe_context({k: v.to(device) for k, v in cache.items()}, runner32)
                            except FloatingPointError as error:
                                # A nonfinite fp32 self-reference is a recorded outcome of this context.
                                record = dict(status="probe_numerical_failure", reason=str(error),
                                              diagnostics=getattr(error, "diagnostics", {}))
                                tensors, rankings = {}, {}
                            else:
                                context_record["probe_identity"] = probe
                                record, tensors, rankings = fp32_phase(
                                    ctx32, protocol, output, modalities=modalities,
                                    adaptive=entry["context_id"] in adaptive_members,
                                    store_nodes=set(bank_cfg["full_node_modalities"]) | (
                                        set(modalities) if entry["context_id"] in node_members else set()))
                                record["status"] = "evaluated"
                                del ctx32
                        output.write_tensors("maps.pt", tensors)
                        output.write_json("rankings.json", rankings)
                    else:
                        phase_rankings = {}
                        for source_phase in ("bf16", "fp32"):
                            if source_phase == "fp32" and entry["context_id"] not in fp32_members:
                                continue
                            directory, _ = load_phase(args.out, args.bank, key, source_phase, binding_for_load(binding))
                            phase_rankings[source_phase] = read_json(directory / "rankings.json")
                        record = scoring_phase(ctx, protocol, phase_rankings, entry["context_id"], modalities=modalities)
                    del ctx
                    record["context"] = context_record
                    record["wall_seconds"] = time.perf_counter() - started
                    if runtime_settings() != applied:
                        raise ValueError("Runtime settings changed during the phase")
                    output.write_json("record.json", record)
            except BaseException as error:
                # The attempt directory keeps failure.json. Stop for diagnosis; no retry here.
                note(event="phase_failed", context=key, phase=phase, error_type=type(error).__name__,
                     reason=str(error)[:4000], wall_seconds=time.perf_counter() - started)
                log.close()
                raise
            done += 1
            note(event="phase_complete", context=key, phase=phase, wall_seconds=time.perf_counter() - started)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    note(event="worker_end", completed_phases=done)
    log.close()
    return 0


def binding_for_load(binding):
    return {key: binding[key] for key in ("protocol_sha256", "bank_sha256", "context_id")}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--phases", required=True, help="comma-separated subset of bf16,fp32,scoring")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    args.phases = [phase for phase in args.phases.split(",") if phase]
    if not args.phases or len(set(args.phases)) != len(args.phases):
        raise ValueError("Phases must be distinct")
    if any(phase not in PHASES for phase in args.phases) or sorted(args.phases, key=PHASES.index) != args.phases:
        raise ValueError("Phases must come from verify, bf16, fp32, scoring, in that order")
    if not 0 <= args.worker < args.workers:
        raise ValueError("Worker index outside the partition")
    return args


def main(argv=None):
    args = parse_args(argv)
    protocol = validate_protocol(read_json(args.protocol))
    if args.bank not in protocol["banks"]:
        raise ValueError("Unknown bank")
    record = startup(protocol, args.bank, args.phases)
    return run_worker(args, protocol, record)


if __name__ == "__main__":
    sys.exit(main())
