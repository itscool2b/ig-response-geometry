"""Prospective paired ranking study on authenticated fixed-noise contexts.

No production numerical defaults or automatic experiment launch. The run
command requires a frozen bank, explicit protocol and hashed E01 gate. Context
collectors need no attribution maps or dummy integration budget. One action
forward produces both common responses for every saved intervention mask.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import math
from pathlib import Path
import sys
import time

FP32_PROBE = "fp32_downstream_lifted_bf16_weights_cached_image_language_readapted_state_self_reference_v1"


def configure_cli_startup():
    """The production CLI configures CUBLAS before any torch import."""
    if __name__ != "__main__" or len(sys.argv) < 2 or sys.argv[1] != "run" or "--help" in sys.argv:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--protocol", required=True)
    args, _ = parser.parse_known_args(sys.argv[2:])
    from experiment_io import strict_json
    if strict_json(args.protocol).get("forward_precision") == FP32_PROBE:
        from fp32_probe_cache import process_startup
        return process_startup("run")
    return None


_PROCESS_STARTUP = configure_cli_startup()

import numpy as np
import torch
import integrated_gradients as ig_kernel

from experiment_io import canonical_json, file_hash, object_hash, strict_json, tensor_hash
from faithfulness import (EvaluationWriter, EXT_CAM_START, EXT_CAM_END, add_replay_arguments,
                          authenticated_source, auc_normalized, load_sidecar,
                          replay_context, replay_inputs, replay_pipeline)
from integrated_gradients import integrated_gradients, NonFiniteAttributionError
from per_step_attribution import MANISKILL_INDICES

def write_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")


def validate_grid(grid):
    if not isinstance(grid, list) or len(grid) < 2 or grid[0] != 0 or grid[-1] != 100:
        raise ValueError("Grid must run from zero to 100 percent")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in grid):
        raise ValueError("Grid must be finite numeric percentages")
    if any(a >= b for a, b in zip(grid, grid[1:])):
        raise ValueError("Nominal grid must be strictly increasing")
    return grid


def uniform_executed_selection(config, episodes, rows, selection):
    """Derive a locked simple random sample without reading outcomes or maps."""
    expected = {"rule", "locked_before_collection", "algorithm", "seed_namespace", "seed",
                "calls_per_episode", "max_episode_steps"}
    if set(selection) != expected or selection["rule"] != "uniform_executed_calls_v1":
        raise ValueError("Uniform call selection needs the exact versioned contract")
    collection_protocol = config.get("protocol", {})
    declared = collection_protocol.get("uniform_call_selection")
    for stratum in collection_protocol.get("strata", []):
        if stratum.get("id") == config.get("stratum_id") and "uniform_call_selection" in stratum:
            declared = stratum["uniform_call_selection"]
    if selection["locked_before_collection"] is not True or declared != selection:
        raise ValueError("Uniform selection must be bound in the original collection protocol")
    if selection["algorithm"] != "numpy_pcg64_permutation_v1":
        raise ValueError("Unknown uniform call permutation algorithm")
    if not isinstance(selection["seed_namespace"], str) or not selection["seed_namespace"].strip():
        raise ValueError("A predeclared selection seed namespace is required")
    if type(selection["seed"]) is not int or selection["seed"] < 0:
        raise ValueError("Selection seed must be a nonnegative integer")
    if type(selection["calls_per_episode"]) is not int or selection["calls_per_episode"] < 1:
        raise ValueError("Explicit positive calls per episode required")
    cap = selection["max_episode_steps"]
    if type(cap) is not int or not 1 <= cap <= 400 or config.get("max_episode_steps") != cap:
        raise ValueError("Collection and selection must share an explicit horizon at most 400")
    planned, accounting = [], []
    for ep, terminal in sorted(episodes.items()):
        explicit_cap = terminal.get("stop_reason") == "collector_environment_step_cap"
        if terminal.get("stop_reason") not in {"terminated", "truncated", "collector_environment_step_cap"}:
            raise ValueError("Uniform call sampling requires complete trajectories, not a protocol call cap")
        if explicit_cap and (terminal.get("env_steps") != cap or terminal.get("collector_step_cap") is not True):
            raise ValueError("An explicit collector cap must reach the complete declared horizon")
        if not (terminal.get("terminated") or terminal.get("truncated") or explicit_cap) or not 1 <= terminal.get("env_steps", 0) <= cap:
            raise ValueError("Invalid full-trajectory terminal accounting")
        eligible = sorted(r["policy_call_idx"] for r in rows if r["episode"] == ep)
        count = terminal["policy_calls"]
        if type(count) is not int or count < 1 or eligible != list(range(count)):
            raise ValueError("Uniform selection requires every executed call exactly once")
        stream = dict(namespace=selection["seed_namespace"], master_seed=selection["seed"],
                      task=config["task"], pipeline=config["pipeline"], reset_seed=terminal["seed"])
        digest = object_hash(stream)
        rng = np.random.Generator(np.random.PCG64(int(digest, 16)))
        permutation = rng.permutation(eligible).tolist()
        selected = permutation[:min(selection["calls_per_episode"], count)]
        planned.extend(dict(episode=ep, policy_call_idx=call) for call in selected)
        accounting.append(dict(episode=ep, reset_seed=terminal["seed"], eligible_calls=eligible,
            permutation=permutation, selected_calls=selected, stream_sha256=digest,
            stream=stream, inclusion_probability=len(selected)/count,
            within_episode_weight=1/len(selected), episode_weight=1/len(episodes),
            terminal_record_sha256=object_hash(terminal)))
    return planned, accounting


def make_bank(metrics, selection=None):
    """Freeze an explicit prospective roster; absent calls remain in the bank.

    Selection contains a text rule and planned episode/call pairs. Episode
    reset seeds come from committed episode accounting, not legacy key guesses.
    Creating a bank after collection authenticates its contents but does not
    prove the selection was prospective; the selection artifact must be locked
    before collection by the study protocol.
    """
    before = file_hash(metrics)
    rows, manifest = authenticated_source(metrics)
    config = manifest["configuration"]
    collector_pairs = None
    uniform = selection is not None and selection.get("rule") == "uniform_executed_calls_v1"
    if config.get("collector_type") == "context_only":
        calls = config["call_selection"]["evaluate_calls"]
        collector_pairs = {(ep, call) for ep in range(config["episodes"]) for call in calls}
        for row in rows:
            if row.get("selected_for_analysis") is not (row["policy_call_idx"] in calls):
                raise ValueError("Collector selection flag differs from its locked configuration")
        if selection is None:
            selection = dict(rule="exact_collector_prespecified_call_indices", locked_before_collection=True,
                             collector_protocol_sha256=config["protocol_sha256"],
                             contexts=[dict(episode=ep, policy_call_idx=call) for ep, call in sorted(collector_pairs)])
        elif not uniform and not selection.get("derivative_protocol_sha256"):
            raise ValueError("A subset of collector selection requires a recorded derivative protocol")
    if selection is None:
        raise ValueError("An attribution rollout requires an explicit prospective selection artifact")
    if not selection.get("rule") or not selection.get("locked_before_collection"):
        raise ValueError("An explicitly locked prospective selection rule is required")
    ends = [strict_json(p)["records"][-1] for p in
            sorted(Path(str(metrics) + ".run").joinpath("episodes").glob("ep*.json"))]
    episodes = {r["episode"]: r for r in ends}
    if set(episodes) != set(range(config["episodes"])):
        raise ValueError("Bank requires complete accounting of every declared episode")
    if uniform:
        if config.get("collector_type") != "context_only":
            raise ValueError("Uniform selection requires the all-executed-call context collector")
        planned, selection_accounting = uniform_executed_selection(config, episodes, rows, selection)
        collector_pairs = {(r["episode"], r["policy_call_idx"]) for r in rows}
    else:
        planned, selection_accounting = selection.get("contexts", []), None
    if not planned:
        raise ValueError("Empty planned context population")
    lookup = {(r["episode"], r["policy_call_idx"]): r for r in rows}
    contexts, seen, seen_ids = [], set(), set()
    for planned_context in planned:
        ep, call = planned_context["episode"], planned_context["policy_call_idx"]
        if type(ep) is not int or type(call) is not int or min(ep, call) < 0 or (ep, call) in seen:
            raise ValueError("Invalid or duplicate planned context")
        seen.add((ep, call))
        if collector_pairs is not None and (ep, call) not in collector_pairs:
            raise ValueError("Requested call was not prespecified by the collector")
        if ep not in episodes:
            raise ValueError("Planned episode has no committed terminal accounting")
        terminal = episodes[ep]
        # The revised rollout seed is the simulator reset seed. Its identity is
        # authenticated by both the configuration and episode transaction.
        seed = terminal["seed"]
        if seed != config["seed_base"] + ep:
            raise ValueError("Unsupported or inconsistent source reset-seed schedule")
        episode_id = object_hash(dict(task=config["task"], pipeline=config["pipeline"], reset_seed=seed))
        entry = dict(episode=ep, reset_seed=seed, policy_call_idx=call, episode_id=episode_id,
                     terminal_record_sha256=object_hash(terminal))
        row = lookup.get((ep, call))
        if row is None:
            if call < terminal["policy_calls"]:
                raise ValueError("Executed call is absent from committed source")
            entry.update(status="call_not_reached", context_id=None)
        else:
            _, digest = load_sidecar(row, metrics, manifest)
            if row["seed"] != seed or row["context_id"] in seen_ids:
                raise ValueError("Duplicate context or reset-seed mismatch")
            seen_ids.add(row["context_id"])
            entry.update(status="available", context_id=row["context_id"],
                         row_sha256=object_hash(row), sidecar_sha256=digest)
        contexts.append(entry)
    if file_hash(metrics) != before:
        raise ValueError("Source changed while building bank")
    result = dict(schema_version=1, kind="paired_context_bank", source_metrics_sha256=before,
                source_run_id=manifest["run_id"], source_configuration_sha256=manifest["configuration_sha256"],
                task=config["task"], model=config["model"], pipeline=config["pipeline"],
                language=config["language"], collector_type=config.get("collector_type", "attribution_rollout"),
                collector_call_selection=config.get("call_selection"),
                collector_planned_contexts=len(collector_pairs) if collector_pairs is not None else None,
                selection=selection, contexts=contexts)
    if uniform:
        result["selection_accounting"] = selection_accounting
        result["estimand"] = "equal_episode_mean_of_uniform_executed_calls_under_capped_behavior_policy"
    return result


def bank_rows(bank, rows, manifest, source_digest):
    if bank.get("kind") != "paired_context_bank" or bank["source_metrics_sha256"] != source_digest:
        raise ValueError("Bank source export mismatch")
    if bank["source_run_id"] != manifest["run_id"] or bank["source_configuration_sha256"] != manifest["configuration_sha256"]:
        raise ValueError("Bank source identity mismatch")
    by_id = {r["context_id"]: r for r in rows}
    selected = {}
    seen = set()
    for entry in bank["contexts"]:
        key = (entry["episode"], entry["policy_call_idx"])
        if key in seen:
            raise ValueError("Duplicate planned bank decision")
        seen.add(key)
        if entry["status"] == "available":
            row = by_id[entry["context_id"]]
            if object_hash(row) != entry["row_sha256"] or row["attr_sha256"] != entry["sidecar_sha256"]:
                raise ValueError("Bank context row or sidecar mismatch")
            if (row["episode"], row["policy_call_idx"]) != key or row["seed"] != entry["reset_seed"]:
                raise ValueError("Bank context episode/call identity mismatch")
            selected[key] = row
        elif entry["status"] != "call_not_reached":
            raise ValueError("Unknown bank population status")
    return selected


def validate_protocol(protocol, bank_hash, gate, gate_hash):
    # json.loads can parse an overflowing exponent as inf. Canonical encoding
    # recursively rejects it even in a nested planning field.
    canonical_json(protocol)
    canonical_json(gate)
    if protocol.get("stage") not in {"variance_only_pilot", "confirmatory_locked"}:
        raise ValueError("Protocol must declare pilot or locked confirmatory stage")
    if protocol.get("bank_sha256") != bank_hash or protocol.get("e01_gate_sha256") != gate_hash:
        raise ValueError("Protocol bank or numerical gate hash mismatch")
    if gate.get("status") != "approved_for_paired_study":
        raise ValueError("E01 has not approved these production numerical settings")
    settings = protocol.get("numerics")
    if not isinstance(settings, dict) or not settings or not set(settings) <= {"vision", "language", "state"}:
        raise ValueError("Explicit numerical settings are required for every planned modality")
    for modality, spec in settings.items():
        if set(spec) != {"Q", "L2"}:
            raise ValueError("Both ranking targets require a numerical gate")
        for target, detail in spec.items():
            if type(detail.get("m")) is not int or detail["m"] < 1:
                raise ValueError("No implicit or nonpositive integration budget is supported")
            if detail.get("quadrature") != "trapezoid" or detail.get("arithmetic_dtype") not in {"float32", "float64"}:
                raise ValueError("Explicit revised trapezoidal arithmetic is required")
            if detail != gate.get("approved_numerics", {}).get(modality, {}).get(target):
                raise ValueError("Protocol numerical settings differ from E01 approval")
    validate_grid(protocol["grid_percent"])
    if type(protocol.get("random_permutations")) is not int or protocol["random_permutations"] < 2:
        raise ValueError("At least two permutations are needed to estimate Monte Carlo variance")
    if type(protocol.get("random_seed")) is not int:
        raise ValueError("Explicit randomization seed required")
    if protocol.get("selection_filter") != "none" or protocol.get("cluster_unit") != "reset_episode":
        raise ValueError("The primary bank is unfiltered and reset episodes are the sampling unit")
    if protocol.get("failure_policy") != "retain_planned_denominators_no_replacement":
        raise ValueError("Explicit nonreplacement failure policy required")
    for response in ("Q", "L2", "RMS"):
        cutoff = protocol["denominator_min"][response]
        if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or cutoff < 0:
            raise ValueError("Invalid declared endpoint-gap threshold")
    for key in ("precision_plan", "contrast_family", "sampling_plan", "monte_carlo_plan"):
        if not protocol.get(key):
            raise ValueError(f"Missing prospective {key}")
    family = protocol["contrast_family"]
    if not isinstance(family, list):
        raise ValueError("Contrast family must be an explicit list")
    seen = set()
    labels = {"Q_IG", "L2_IG", "input_difference", "Q_path_gradient", "L2_path_gradient"}
    for contrast in family:
        if set(contrast) != {"modality", "method", "control", "response", "direction", "metric"}:
            raise ValueError("Each contrast must specify modality, method, control, response, direction and metric")
        if contrast["modality"] not in settings or contrast["method"] not in labels or contrast["control"] not in labels | {"random"}:
            raise ValueError("Contrast requests an unavailable ranking or modality")
        if contrast["response"] not in {"Q", "L2", "RMS"} or contrast["direction"] not in {"deletion", "insertion"} or contrast["metric"] not in {"raw_auc", "normalized_auc"}:
            raise ValueError("Unknown contrast estimand")
        if contrast["method"] == contrast["control"] or object_hash(contrast) in seen:
            raise ValueError("Self-contrasts and duplicate contrasts are not permitted")
        seen.add(object_hash(contrast))
    analysis = protocol.get("analysis", {})
    if analysis.get("method") != "episode_percentile_bootstrap" or not 0 < analysis.get("family_alpha", 0) < 1:
        raise ValueError("Explicit prospective interval method and family alpha required")
    if type(analysis.get("draws")) is not int or analysis["draws"] < 1 or type(analysis.get("seed")) is not int:
        raise ValueError("Bootstrap draws and seed must be locked in the protocol")
    if protocol.get("forward_precision") not in {"authenticated_source", FP32_PROBE} or gate.get("forward_precision") != protocol["forward_precision"]:
        raise ValueError("The exact forward precision and reference policy must be E01-approved")
    if gate.get("integrated_gradients_sha256") != file_hash(ig_kernel.__file__):
        raise ValueError("E01 gate does not authenticate the loaded IG kernel")
    if gate.get("paired_comparison_sha256") != file_hash(__file__):
        raise ValueError("E01 gate does not authenticate this paired implementation")
    if protocol["forward_precision"] == FP32_PROBE:
        from scripts import validate_downstream_precision as precision
        import fp32_probe_cache as cache_helper
        if gate.get("precision_helper_sha256") != file_hash(precision.__file__):
            raise ValueError("E01 gate does not authenticate the fp32 conversion helper")
        if gate.get("cache_helper_sha256") != file_hash(cache_helper.__file__):
            raise ValueError("E01 gate does not authenticate prepared-cache and runtime helpers")
        if protocol.get("runtime_settings") != cache_helper.strict_candidate_settings() or gate.get("runtime_settings") != protocol["runtime_settings"]:
            raise ValueError("Production must use the exact E01-approved deterministic runtime settings")
        for name in ("prepared_completion_sha256", "preparation_decision_sha256"):
            if not isinstance(protocol.get(name), str) or len(protocol[name]) != 64:
                raise ValueError("Production must bind the prepared source cache: " + name)
    if protocol.get("random_control") != "exact_uniform_subsets_n_le_8_else_seeded_permutations":
        raise ValueError("The small-population exact random-control policy must be locked")
    return protocol


def validate_gate_population(gate, bank):
    identity = dict(task=bank["task"], model=bank["model"], pipeline_sha256=object_hash(bank["pipeline"]),
                    language_sha256=object_hash(bank["language"]))
    if identity not in gate.get("approved_strata", []):
        raise ValueError("The E01 gate does not cover this task/checkpoint/language stratum")


def integrate_with_path_gradient(forward, actual, baseline, settings, *, context=None):
    """Capture the validated IG kernel's same-stream mean path gradient.

    Hooks observe the gradient before the cast back to the master leaf. They
    do not modify it. This retains coordinates with zero input displacement,
    unlike dividing IG by displacement, and makes no endpoint-only control.
    """
    dtype = getattr(torch, settings["arithmetic_dtype"])
    m, rule = settings["m"], settings["quadrature"]
    gradient_sum = torch.zeros_like(actual, dtype=dtype)
    calls = 0
    hooks_called = 0

    def wrapped(point):
        nonlocal calls
        k = calls
        calls += 1
        weight = 0.5 if rule == "trapezoid" and k in (0, m) else 1.0

        def capture(gradient):
            nonlocal hooks_called
            hooks_called += 1
            gradient_sum.add_(gradient.detach().to(dtype), alpha=weight)
        point.register_hook(capture)
        return forward(point)

    result = integrated_gradients(wrapped, actual, baseline, m=m, quadrature=rule,
                                  arithmetic_dtype=dtype, return_result=True, diagnostic_context=context)
    if calls != m + 1 or hooks_called != calls:
        raise RuntimeError("IG kernel gradient-capture contract changed")
    average = gradient_sum / (m if rule == "trapezoid" else m + 1)
    reconstructed = (actual.to(dtype) - baseline.to(dtype)) * average
    if not torch.isfinite(average).all() or not torch.equal(reconstructed, result.attributions):
        raise RuntimeError("Captured path gradient differs from the validated IG kernel")
    return result, average


def common_scores(action, reference, *, active_indices=MANISKILL_INDICES):
    if action.shape != reference.shape or action.numel() == 0:
        raise ValueError("Action/reference shape mismatch")
    # Cast both operands before subtraction. Q and L2 share this exact array.
    difference = action[..., active_indices].float() - reference[..., active_indices].float()
    squared = (difference * difference).sum()
    return {"Q": -squared / (2 * difference.numel()), "L2": -torch.sqrt(squared + 1e-12),
            "RMS": torch.sqrt(squared / difference.numel())}


def modality_inputs(ctx, modality):
    if modality == "vision":
        actual, baseline = ctx["img_adapted"], ctx["img_adapted_bl"]
        indices, axis = list(range(EXT_CAM_START, EXT_CAM_END)), 1
        forward = lambda x: ctx["seeded_conditional_sample"](ctx["lang_adapted"], x, ctx["state_traj_actual"])
    elif modality == "language":
        actual, baseline = ctx["lang_adapted"], ctx["lang_adapted_bl"]
        indices, axis = ctx["lang_attn_mask"][0].nonzero().flatten().tolist(), 1
        forward = lambda x: ctx["seeded_conditional_sample"](x, ctx["img_adapted"], ctx["state_traj_actual"])
    elif modality == "state":
        actual, baseline = ctx["state_input_actual"], ctx["state_input_baseline"]
        indices, axis = list(MANISKILL_INDICES), 2
        forward = lambda x: ctx["seeded_conditional_sample"](ctx["lang_adapted"], ctx["img_adapted"],
            ctx["runner"].state_adaptor(torch.cat([x, ctx["action_mask"]], dim=2)))
    else:
        raise ValueError("Unknown modality")
    validate_population(actual, baseline, indices, axis)
    return actual, baseline, indices, axis, forward


def attach_probe_failure(error, ctx, stage):
    """Keep numerical failure tensors outside JSON, with finite JSON counts."""
    raw = {key: ctx[key].detach().cpu() for key in
           ("state_input_actual", "state_traj_actual", "ref_action", "action_mask", "initial_noise")
           if isinstance(ctx.get(key), torch.Tensor)}
    error.raw_tensors = raw
    error.diagnostics = {**getattr(error, "diagnostics", {}), "probe_failure_stage": stage,
        "tensors": {key: dict(shape=list(value.shape), dtype=str(value.dtype), numel=value.numel(),
            finite_count=int(torch.isfinite(value).sum()), nan_count=int(torch.isnan(value).sum()),
            infinity_count=int(torch.isinf(value).sum())) for key,value in raw.items()}}
    return error


def make_fp32_probe_context(source, runner, *, sample_fn=None):
    """Construct the prospective offline probe after exact source replay.

    The separate runner must already have undergone the audited one-way fp32
    conversion. Frozen bf16-adapted image/language numeric values are lifted.
    Actual raw state is readapted once in fp32 and shared across every modality,
    so each modality's actual endpoint has one coherent new self-reference.
    """
    if sample_fn is None:
        from rdt_sampling import conditional_sample_with_noise
        sample_fn = conditional_sample_with_noise
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in runner.parameters()):
        raise ValueError("Probe runner has unconverted floating-point parameters")
    names = ("lang_adapted", "lang_adapted_bl", "img_adapted", "img_adapted_bl",
             "state_input_actual", "state_input_baseline", "action_mask", "ctrl_freqs", "initial_noise")
    ctx = {name: source[name].detach().float().clone() for name in names}
    for name in names:
        if not torch.equal(ctx[name].to(source[name].dtype), source[name]):
            raise ValueError(f"Probe changed cached numeric values: {name}")
    ctx["lang_attn_mask"] = source["lang_attn_mask"].clone()
    ctx["runner"] = runner
    with torch.no_grad():
        ctx["state_traj_actual"] = runner.state_adaptor(torch.cat([ctx["state_input_actual"], ctx["action_mask"]], dim=2))
    if ctx["state_traj_actual"].dtype != torch.float32:
        raise ValueError("Probe state-adaptor dtype violates the fp32 contract")
    if not torch.isfinite(ctx["state_traj_actual"]).all():
        raise attach_probe_failure(FloatingPointError("Nonfinite fp32 state-adaptor output"), ctx, "state_adaptor")

    def sample(language, image, state):
        return sample_fn(runner, language, ctx["lang_attn_mask"], image, state,
                         ctx["action_mask"], ctx["ctrl_freqs"], ctx["initial_noise"])
    ctx["seeded_conditional_sample"] = sample
    try:
        with torch.no_grad():
            ctx["ref_action"] = sample(ctx["lang_adapted"], ctx["img_adapted"], ctx["state_traj_actual"]).detach()
    except (NonFiniteAttributionError, FloatingPointError) as error:
        raise attach_probe_failure(error, ctx, "reference_sampler")
    if ctx["ref_action"].dtype != torch.float32:
        raise ValueError("Probe self-reference dtype violates the fp32 contract")
    if ctx["ref_action"].shape != source["ref_action"].shape:
        raise ValueError("Probe self-reference shape violates the source action contract")
    if not torch.isfinite(ctx["ref_action"]).all():
        raise attach_probe_failure(FloatingPointError("Nonfinite fp32 probe self-reference"), ctx, "self_reference")
    difference = ctx["ref_action"][..., MANISKILL_INDICES] - source["ref_action"][..., MANISKILL_INDICES].float()
    state_difference = ctx["state_traj_actual"] - source["state_traj_actual"].float()
    provenance = dict(contract=FP32_PROBE, policy="offline_probe_on_bf16_policy_visited_contexts",
        source_reference_sha256=tensor_hash(source["ref_action"]),
        probe_reference_sha256=tensor_hash(ctx["ref_action"]),
        probe_reference_active_values=ctx["ref_action"][..., MANISKILL_INDICES].cpu().tolist(),
        active_action_indices=list(MANISKILL_INDICES),
        reference_drift_l2=float(difference.norm().item()),
        reference_drift_rms=float(difference.square().mean().sqrt().item()),
        reference_drift_max_abs=float(difference.abs().max().item()),
        cached_source_state_token_sha256=tensor_hash(source["state_traj_actual"]),
        readapted_fp32_state_token_sha256=tensor_hash(ctx["state_traj_actual"]),
        state_token_drift_l2=float(state_difference.norm().item()),
        source_noise_sha256=tensor_hash(source["initial_noise"]),
        probe_noise_sha256=tensor_hash(ctx["initial_noise"]),
        noise_value_policy="same_recorded_values_lifted_to_fp32")
    return ctx, provenance


def validate_population(actual, baseline, indices, axis):
    if actual.shape != baseline.shape or actual.ndim != 3 or actual.shape[0] != 1:
        raise ValueError("Modality tensors must have matching batch-one shapes")
    if not torch.isfinite(actual).all() or not torch.isfinite(baseline).all():
        raise ValueError("Nonfinite modality input or baseline")
    if axis not in (1, 2) or not indices or sorted(set(indices)) != indices or min(indices) < 0 or max(indices) >= actual.shape[axis]:
        raise ValueError("Invalid eligible population")
    outside = [i for i in range(actual.shape[axis]) if i not in indices]
    selection = [slice(None)] * 3
    selection[axis] = outside
    if not torch.equal(actual[tuple(selection)], baseline[tuple(selection)]):
        raise ValueError("Baseline differs outside the declared eligible population")


def grouped_scores(values, indices, axis):
    scores = values.abs().sum(dim=2 if axis == 1 else 1)[0][indices]
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite ranking reduction")
    return scores


def ranked(scores, indices):
    order = torch.argsort(scores, descending=True, stable=True).cpu().tolist()
    return dict(status="defined", order=[indices[i] for i in order], scores=scores.cpu().tolist(),
                scores_sha256=tensor_hash(scores), zero_map=bool((scores == 0).all()),
                tie_rule="decreasing_score_then_increasing_eligible_index",
                unique_scores=int(torch.unique(scores).numel()),
                tied_positions=int(sum(count.item() for count in torch.unique(scores, return_counts=True)[1] if count > 1)))


def random_order(indices, master_seed, context_id, modality, replicate):
    seed = int(object_hash([master_seed, context_id, modality, replicate])[:16], 16)
    order = np.random.Generator(np.random.PCG64(seed)).permutation(indices).tolist()
    return dict(status="defined", order=order, replicate=replicate, seed=seed,
                generator="numpy.PCG64", tie_rule="random_permutation")


def build_rankings(action_forward, reference, actual, baseline, indices, axis, settings,
                   *, context_id, modality, random_permutations, random_seed):
    rankings = {"input_difference": ranked(grouped_scores(actual.float() - baseline.float(), indices, axis), indices)}
    for target in ("Q", "L2"):
        try:
            result, average = integrate_with_path_gradient(
                lambda x: common_scores(action_forward(x), reference)[target], actual, baseline,
                settings[target], context=dict(context_id=context_id, modality=modality, target=target))
            for label, tensor in ((f"{target}_IG", result.attributions), (f"{target}_path_gradient", average)):
                rankings[label] = ranked(grouped_scores(tensor, indices, axis), indices)
                rankings[label]["numerics"] = result.diagnostics()
        except (NonFiniteAttributionError, FloatingPointError) as exc:
            for label in (f"{target}_IG", f"{target}_path_gradient"):
                rankings[label] = dict(status="numerical_failure", reason=str(exc),
                                       failure_kind="integration_numerical_failure",
                                       diagnostics=getattr(exc, "diagnostics", {}))
    if len(indices) <= 8:
        rankings["random_exact"] = dict(status="defined", control_kind="exact_uniform_subset_expectation",
                                        eligible_count=len(indices), monte_carlo_variance=0.0)
    else:
        for replicate in range(random_permutations):
            rankings[f"random_{replicate:04d}"] = random_order(indices, random_seed, context_id, modality, replicate)
    return rankings


def replace_positions(actual, baseline, positions, axis):
    result = actual.clone()
    selection = [slice(None)] * 3
    selection[axis] = positions
    result[tuple(selection)] = baseline[tuple(selection)]
    return result


def evaluate_rankings(action_forward, reference, actual, baseline, indices, axis,
                      rankings, grid, denominator_min):
    """Save active actions and three responses once per unique intervention."""
    from itertools import combinations
    validate_population(actual, baseline, indices, axis)
    validate_grid(grid)
    counts = [round(len(indices) * k / 100) for k in grid]
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
                                             endpoint_gap=None, status="nonfinite_response")
                else:
                    gap = endpoints[0] - endpoints[1]
                    cutoff = denominator_min.get(target, 0)
                    normalized = auc_normalized([100*f for f in fractions], values, *endpoints,
                                                denominator_min=cutoff)
                    status = "zero_endpoint_gap" if gap == 0 else "nearzero_endpoint_gap" if abs(gap) <= cutoff else "defined"
                    by_target[target] = dict(values=values, raw_auc=float(np.trapezoid(values, fractions)),
                                             normalized_auc=normalized, endpoint_gap=gap, status=status)
            curves[direction] = dict(selected_positions=masks if exact else [group[0] for group in masks],
                mask_sha256=[[object_hash(mask) for mask in group] for group in masks] if exact else [object_hash(group[0]) for group in masks],
                response_ids=response_ids if exact else [group[0] for group in response_ids], responses=by_target)
            if exact:
                curves[direction]["subset_counts"] = [len(group) for group in masks]
                curves[direction]["expectation"] = "equal mean over all distinct subsets at each realized count"
        ranking["curves"] = curves
    return dict(rankings=rankings, response_table=responses, eligible_indices=indices,
        nominal_grid_percent=grid, selected_counts=counts, realized_fractions=fractions,
        input_response_id=input_id, baseline_response_id=baseline_id, denominator_min=denominator_min,
        active_action_indices=list(MANISKILL_INDICES), active_reference_action=active_reference.cpu().tolist(),
        active_reference_sha256=tensor_hash(active_reference),
        response_definition=dict(Q="-sum_FP32((action-reference)^2)/(2D)",
            L2="-sqrt(sum_FP32((action-reference)^2)+1e-12)",
            RMS="sqrt(sum_FP32((action-reference)^2)/D)",
            RMS_units="normalized active-action coordinates; positive displacement without stabilizer"))


def paired_episode_summary(records, *, method, control, response, direction, metric,
                           alpha, draws, seed, require_complete_episode=True):
    """Pair contexts first, average permutations/calls, then resample episodes.

    A missing or failed planned call makes that episode incomplete. The primary
    complete-episode contrast does not silently average over surviving calls.
    Its conditional population and all excluded planned episodes are explicit.
    No action-norm eligibility filter exists here.
    """
    if response not in {"Q", "L2", "RMS"} or direction not in {"deletion", "insertion"} or metric not in {"raw_auc", "normalized_auc"}:
        raise ValueError("Unknown paired estimand")
    if not 0 < alpha < 1 or type(draws) is not int or draws < 1:
        raise ValueError("Explicit valid interval configuration is required")
    episodes, seen, permutations = defaultdict(list), set(), []
    numerical_counts = defaultdict(int)
    method_outcomes, control_outcomes, discordance = defaultdict(int), defaultdict(int), defaultdict(int)
    membership = []
    for record in records:
        identity = (record["episode_id"], record["policy_call_idx"], record["modality"])
        if identity in seen:
            raise ValueError("Duplicate planned decision would inflate the paired population")
        seen.add(identity)
        member = dict(episode_id=record["episode_id"], policy_call_idx=record["policy_call_idx"],
                      modality=record["modality"], source_context_id=record.get("source_context_id"), valid=False)
        membership.append(member)
        episode = episodes[record["episode_id"]]
        if record["status"] != "evaluated":
            cause = record.get("failure_kind", record["status"])
            numerical_counts[cause] += 1
            method_outcomes[cause] += 1
            control_outcomes[cause] += 1
            discordance["both_undefined"] += 1
            episode.append(None)
            continue
        rankings = record["results"]["rankings"]

        def value(label):
            item = rankings[label]
            if item["status"] != "defined":
                return None
            return item["curves"][direction]["responses"][response][metric]

        def causes(label, counts):
            item = rankings[label]
            counts[item.get("failure_kind", item["status"])] += 1
            if item["status"] == "defined":
                outcome = item["curves"][direction]["responses"][response]
                counts[outcome.get("status", "defined")] += 1
                if item.get("zero_map"):
                    counts["all_zero_map"] += 1
                if item.get("tied_positions", 0):
                    counts["tied_ranking"] += 1
        treatment = value(method)
        control_labels = [label for label in rankings if label.startswith("random_")] if control == "random" else [control]
        controls = [value(label) for label in control_labels]
        causes(method, method_outcomes)
        for label in control_labels:
            causes(label, control_outcomes)
        if not controls:
            raise ValueError("Random control has no permutations")
        method_missing, control_missing = treatment is None, any(item is None for item in controls)
        discordance["both_undefined" if method_missing and control_missing else "method_only_undefined" if method_missing
                    else "control_only_undefined" if control_missing else "both_defined"] += 1
        if method_missing or control_missing:
            numerical_counts["undefined_paired_outcome"] += 1
            episode.append(None)
            continue
        values = [treatment, *controls]
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Nonfinite value escaped explicit numerical outcome handling")
        sign = -1 if direction == "deletion" else 1
        if response == "RMS" and metric == "raw_auc":
            sign *= -1
        differences = sign * (treatment - np.asarray(controls, dtype=float))
        episode.append(float(differences.mean()))
        member["valid"] = True
        if control == "random":
            if len(controls) < 2 and "random_exact" not in rankings:
                raise ValueError("Random variance needs at least two permutations")
            permutations.append(dict(episode_id=record["episode_id"],
                                     context_mean_variance=0.0 if "random_exact" in rankings else float(differences.var(ddof=1) / len(controls))))
    retained, incomplete = {}, []
    for episode_id, values in episodes.items():
        valid = [x for x in values if x is not None]
        if len(valid) != len(values):
            incomplete.append(episode_id)
        if valid and (not require_complete_episode or len(valid) == len(values)):
            retained[episode_id] = float(np.mean(valid))
    values = np.asarray(list(retained.values()), dtype=float)
    n = len(values)
    estimate = float(values.mean()) if n else None
    interval = None
    if n > 1:
        rng = np.random.Generator(np.random.PCG64(seed))
        bootstrap = np.empty(draws, dtype=float)
        # Bounded memory even for a large precision-selected episode budget.
        for begin in range(0, draws, 1000):
            end = min(begin + 1000, draws)
            bootstrap[begin:end] = values[rng.integers(0, n, size=(end-begin, n))].mean(axis=1)
        interval = np.quantile(bootstrap, [alpha/2, 1-alpha/2]).tolist()
    episode_variance = float(values.var(ddof=1)) if n > 1 else None
    mc_variance = 0.0
    if control == "random" and n:
        for item in permutations:
            if item["episode_id"] in retained:
                # Permutation streams are independently keyed by context. Calls
                # still share an episode for the biological/simulator variance.
                calls = sum(x is not None for x in episodes[item["episode_id"]])
                mc_variance += item["context_mean_variance"] / (calls*calls*n*n)
    else:
        mc_variance = None
    analyzed_membership = []
    for item in membership:
        if item["valid"] and item["episode_id"] in retained:
            calls = sum(value is not None for value in episodes[item["episode_id"]])
            analyzed_membership.append({**item, "within_episode_weight": 1/calls, "episode_weight": 1/n})
    population_hash = object_hash(sorted(analyzed_membership, key=lambda item:
        (item["episode_id"], item["policy_call_idx"], item["modality"])))
    return dict(method=method, control=control, response=response, direction=direction, metric=metric,
                sign="positive_favors_method", estimate=estimate, confidence_interval=interval,
                alpha=alpha, draws=draws, seed=seed, planned_contexts=len(seen),
                valid_contexts=sum(x is not None for group in episodes.values() for x in group),
                planned_episodes=len(episodes), analysis_episodes=n,
                incomplete_episode_ids=incomplete, episode_effects=retained,
                point_population_sha256=population_hash, ci_population_sha256=population_hash,
                planned_membership=membership, analyzed_membership=analyzed_membership,
                conditioning="all_planned_calls_defined" if require_complete_episode else "available_defined_calls",
                episode_variance=episode_variance,
                episode_standard_error=math.sqrt(episode_variance/n) if episode_variance is not None else None,
                random_monte_carlo_standard_error=math.sqrt(mc_variance) if mc_variance is not None else None,
                outcome_counts=dict(numerical_counts),
                method_outcome_counts=dict(method_outcomes), control_outcome_counts=dict(control_outcomes),
                paired_validity_discordance=dict(discordance),
                interval_status="episode_bootstrap" if interval else "unavailable_fewer_than_two_episodes")


class PairedEvaluationWriter(EvaluationWriter):
    """Bind completion to immutable manifest, protocol, bank and auxiliaries."""
    def __enter__(self):
        result = super().__enter__()
        self.bound_manifest_hash = file_hash(self.path.with_name(self.path.name + ".manifest.json"))
        self.auxiliary_hashes = {}
        return result

    def write_auxiliary(self, name, value):
        digest = super().write_auxiliary(name, value)
        self.auxiliary_hashes[name] = digest
        return digest

    def write_tensor_auxiliary(self, name, values):
        if Path(name).name != name or not name.endswith(".pt"):
            raise ValueError("Tensor auxiliary must be a local .pt basename")
        path = self.path.with_name(self.path.name + "." + name)
        with path.open("xb") as stream:
            torch.save({key:value.detach().cpu().contiguous() for key,value in values.items()}, stream)
        digest = file_hash(path)
        self.auxiliary_hashes[name] = digest
        return dict(file=path.name, sha256=digest)

    def __exit__(self, kind, exception, traceback):
        if exception is not None:
            return super().__exit__(kind, exception, traceback)
        try:
            if file_hash(self.args.metrics) != self.source_digest or self.completed != self.expected_rows:
                raise ValueError("Paired source changed or planned records are missing")
            if file_hash(self.path.with_name(self.path.name + ".manifest.json")) != self.bound_manifest_hash:
                raise ValueError("Paired manifest changed during execution")
            for name, digest in self.auxiliary_hashes.items():
                if file_hash(self.path.with_name(self.path.name + "." + name)) != digest:
                    raise ValueError("Paired auxiliary changed during execution")
        except Exception as error:
            super().__exit__(type(error), error, error.__traceback__)
            raise
        self.handle.close()
        config = self.manifest["configuration"]
        write_exclusive(self.path.with_name(self.path.name + ".completion.json"), dict(
            evaluation_id=self.manifest["evaluation_id"], completed_rows=self.completed,
            expected_rows=self.expected_rows, status="complete", output_sha256=file_hash(self.path),
            manifest_sha256=self.bound_manifest_hash, protocol_sha256=config["protocol_sha256"],
            bank_sha256=config["bank_sha256"], auxiliary_sha256=self.auxiliary_hashes))
        return False


def run_study(args):
    if args.legacy_input or args.limit is not None:
        raise ValueError("Paired studies require the entire authenticated planned bank, with no legacy mode or implicit limit")
    protocol, gate, bank = strict_json(args.protocol), strict_json(args.e01_gate), strict_json(args.bank)
    validate_protocol(protocol, file_hash(args.bank), gate, file_hash(args.e01_gate))
    validate_gate_population(gate, bank)
    rows, manifest, digest = replay_inputs(args, apply_limit=False)
    selection = bank["selection"]
    rebuilt = make_bank(args.metrics, None if selection["rule"] == "exact_collector_prespecified_call_indices" else selection)
    if rebuilt != bank:
        raise ValueError("Bank differs from the authenticated collector population and sidecars")
    selected = bank_rows(bank, rows, manifest, digest)
    modalities = list(protocol["numerics"])
    preparation = applied = None
    if protocol["forward_precision"] == FP32_PROBE:
        import fp32_probe_cache as cache_helper
        if not _PROCESS_STARTUP or not _PROCESS_STARTUP.get("torch_not_preimported"):
            raise RuntimeError("Run the production CLI in a fresh process before torch import")
        if not args.prepared or args.prepared_sha256 != protocol["prepared_completion_sha256"]:
            raise ValueError("Production requires the protocol-bound prepared-cache completion")
        applied = cache_helper.apply_settings(protocol["runtime_settings"])
        preparation = cache_helper.verify_preparation(args.prepared, args.prepared_sha256)
        expected = dict(bank_sha256=file_hash(args.bank), source_metrics_sha256=digest,
                        decision_sha256=protocol["preparation_decision_sha256"])
        if any(preparation.get(key) != value for key, value in expected.items()):
            raise ValueError("Prepared bank, source or decision differs from production protocol")
        expected_ids = {entry["context_id"] for entry in bank["contexts"] if entry["status"] == "available"}
        actual_ids = [entry["context_id"] for entry in preparation["contexts"] if entry["status"] == "available"]
        if set(actual_ids) != expected_ids or len(actual_ids) != len(expected_ids):
            raise ValueError("Production preparation must cover each available planned context exactly once")
    configuration = dict(protocol=protocol, protocol_sha256=file_hash(args.protocol),
        bank=bank, bank_sha256=file_hash(args.bank), e01_gate=gate,
        e01_gate_sha256=file_hash(args.e01_gate), script_sha256=file_hash(__file__),
        reference_policy=protocol["forward_precision"], prepared_completion_sha256=getattr(args,"prepared_sha256",None),
        sampling_unit="reset_episode", response_storage="active_actions_and_exact_response_arrays_and_masks")
    with ExitStack() as stack:
        writer = stack.enter_context(PairedEvaluationWriter(args, manifest, digest, "paired_comparison", configuration,
                                                            len(bank["contexts"]) * len(modalities)))
        pipe, lang = replay_pipeline(args, manifest)
        if preparation is not None:
            from scripts.validate_downstream_precision import convert_downstream_to_fp32, fp32_math_settings
            conversion = convert_downstream_to_fp32(pipe["runner"])
            math_settings = stack.enter_context(fp32_math_settings())
            writer.write_auxiliary("fp32_conversion.json", dict(conversion=conversion, math_settings=math_settings,
                runtime_settings=applied, startup=_PROCESS_STARTUP, contract=FP32_PROBE,
                source_replay_settings=preparation["replay_settings"],
                source_settings_authentication=preparation["source_settings_authentication"]))
        for entry in bank["contexts"]:
            common = dict(event="paired_comparison", episode_id=entry["episode_id"],
                          task=bank["task"], model=bank["model"],
                          reset_seed=entry["reset_seed"], planned_context_status=entry["status"])
            if entry["status"] != "available":
                writer.set_context(dict(episode=entry["episode"], seed=entry["reset_seed"],
                                        policy_call_idx=entry["policy_call_idx"]))
                for modality in modalities:
                    writer.write(dict(**common, modality=modality, status=entry["status"], results=None))
                continue
            row = selected[(entry["episode"], entry["policy_call_idx"])]
            writer.set_context(row, stage="source_cache_authentication" if preparation else "exact_source_replay")
            sidecar, sidecar_digest = load_sidecar(row, args.metrics, manifest)
            probe_identity = None
            if preparation is not None:
                cache, cache_record = cache_helper.load_prepared_context(args.prepared, preparation, entry, manifest, row, sidecar)
                device = next(pipe["runner"].parameters()).device
                source = {key: value.to(device) for key, value in cache.items()}
                try:
                    ctx, probe_identity = make_fp32_probe_context(source, pipe["runner"])
                except (NonFiniteAttributionError, FloatingPointError) as error:
                    raw_failure = getattr(error, "raw_tensors", {})
                    artifact = writer.write_tensor_auxiliary("probe-failure-" + entry["context_id"] + ".pt", raw_failure) if raw_failure else None
                    for modality in modalities:
                        writer.write(dict(**common, modality=modality, status="numerical_failure",
                            failure_kind="probe_self_reference", source_replay_verified=True,
                            reason=str(error), diagnostics=getattr(error,"diagnostics",{}),
                            failure_artifact=artifact, results=None))
                    del source, cache, sidecar
                    continue
                probe_identity["prepared_cache"] = cache_record["cache_file"]
                del source, cache
            else:
                ctx = replay_context(args, row, sidecar, pipe, lang, verify_reference=True, strict=True)
            for modality in modalities:
                started = time.perf_counter()
                writer.set_context(row, stage="paired_ranking_and_response", source_attr_sha256=sidecar_digest)
                try:
                    actual, baseline, indices, axis, action_forward = modality_inputs(ctx, modality)
                    rankings = build_rankings(action_forward, ctx["ref_action"], actual, baseline, indices, axis,
                        protocol["numerics"][modality], context_id=entry["context_id"], modality=modality,
                        random_permutations=protocol["random_permutations"], random_seed=protocol["random_seed"])
                    results = evaluate_rankings(action_forward, ctx["ref_action"], actual, baseline, indices, axis,
                                                rankings, protocol["grid_percent"], protocol["denominator_min"])
                    writer.write(dict(**common, modality=modality, status="evaluated", source_replay_verified=True,
                        probe_identity=probe_identity, input_sha256=tensor_hash(actual), baseline_sha256=tensor_hash(baseline),
                        reference_sha256=tensor_hash(ctx["ref_action"]), noise_sha256=tensor_hash(ctx["initial_noise"]),
                        attribution_and_response_wall_seconds=time.perf_counter()-started, results=results))
                except (NonFiniteAttributionError, FloatingPointError) as error:
                    writer.write(dict(**common, modality=modality, status="numerical_failure",
                        failure_kind="modality_probe", source_replay_verified=True, probe_identity=probe_identity,
                        reason=str(error), diagnostics=getattr(error,"diagnostics",{}),
                        attribution_and_response_wall_seconds=time.perf_counter()-started, results=None))
            del ctx, sidecar
        if preparation is not None and cache_helper.runtime_settings() != applied:
            raise ValueError("Production runtime settings changed during execution")
        if any(file_hash(path) != expected for path, expected in
               ((args.protocol, configuration["protocol_sha256"]), (args.bank, configuration["bank_sha256"]),
                (args.e01_gate, configuration["e01_gate_sha256"]))):
            raise ValueError("Production protocol, bank or gate changed during execution")


def authenticate_action_results(results):
    """Recompute saved primitives from authenticated fp32 action coordinates.

    Cross-device reduction comparison permits 32 float32 eps relative error,
    a rounding check rather than a scientific numerical-accuracy threshold.
    """
    reference = torch.tensor(results["active_reference_action"], dtype=torch.float32)
    if reference.ndim != 3 or reference.shape[-1] != len(MANISKILL_INDICES) or not torch.isfinite(reference).all():
        raise ValueError("Invalid saved active reference")
    if tensor_hash(reference) != results["active_reference_sha256"]:
        raise ValueError("Saved active reference hash differs")
    if results["active_action_indices"] != list(MANISKILL_INDICES):
        raise ValueError("Saved active action coordinates differ")
    table = results["response_table"]
    for key, item in table.items():
        positions = item["baseline_positions"]
        if positions != sorted(set(positions)) or not set(positions) <= set(results["eligible_indices"]) or object_hash(positions) != key:
            raise ValueError("Saved intervention mask identity differs")
        if item["status"] == "numerical_failure":
            if any(item[response] is not None for response in ("Q", "L2", "RMS")):
                raise ValueError("Failed intervention contains a finite primitive")
            continue
        if item["status"] != "finite":
            raise ValueError("Unknown intervention outcome")
        action = torch.tensor(item["active_action"], dtype=torch.float32)
        if action.shape != reference.shape or not torch.isfinite(action).all() or tensor_hash(action) != item["active_action_sha256"]:
            raise ValueError("Saved active action shape, values or hash differs")
        recomputed = common_scores(action, reference, active_indices=list(range(reference.shape[-1])))
        for target, expected in recomputed.items():
            if not math.isclose(item[target], float(expected), rel_tol=32*torch.finfo(torch.float32).eps,
                                abs_tol=torch.finfo(torch.float32).tiny):
                raise ValueError("Saved primitive does not reproduce from actions: " + target)
    for ranking in results["rankings"].values():
        if ranking["status"] != "defined":
            continue
        exact = ranking.get("control_kind") == "exact_uniform_subset_expectation"
        for direction, curve in ranking["curves"].items():
            ids = curve["response_ids"] if exact else [[key] for key in curve["response_ids"]]
            for target in ("Q", "L2", "RMS"):
                groups = [[table[key][target] for key in group] for group in ids]
                values = [None if any(x is None for x in group) else math.fsum(group)/len(group) for group in groups]
                outcome = curve["responses"][target]
                if values != outcome["values"]:
                    raise ValueError("Saved curve differs from its primitive action responses")
                endpoints = [table[results["input_response_id"]][target],table[results["baseline_response_id"]][target]]
                if any(x is None for x in values+endpoints):
                    expected_raw = expected_normalized = None
                else:
                    expected_raw = float(np.trapezoid(values,results["realized_fractions"]))
                    expected_normalized = auc_normalized([100*x for x in results["realized_fractions"]],values,*endpoints,
                                                        denominator_min=results["denominator_min"].get(target,0))
                if outcome["raw_auc"] != expected_raw or outcome["normalized_auc"] != expected_normalized:
                    raise ValueError("Saved area differs from its exact response curve")


def load_completed_study(path):
    """Strict shared loader for global analyses of completed producer shards."""
    import json
    path = Path(path)
    completion = strict_json(path.with_name(path.name + ".completion.json"))
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = strict_json(manifest_path)
    if completion["status"] != "complete" or completion["output_sha256"] != file_hash(path):
        raise ValueError("Paired study export is incomplete or changed")
    if completion.get("manifest_sha256") != file_hash(manifest_path) or completion["evaluation_id"] != manifest["evaluation_id"]:
        raise ValueError("Completion does not bind the current paired manifest")
    configuration = manifest["configuration"]
    for name in ("protocol_sha256", "bank_sha256"):
        if completion.get(name) != configuration[name]:
            raise ValueError("Completion protocol or bank binding differs")
    for name, digest in completion.get("auxiliary_sha256", {}).items():
        if Path(name).name != name or file_hash(path.with_name(path.name + "." + name)) != digest:
            raise ValueError("Paired auxiliary hash differs")
    def unique(pairs):
        value = {}
        for key,item in pairs:
            if key in value:
                raise ValueError("Duplicate paired result key")
            value[key] = item
        return value
    rows = [json.loads(line,object_pairs_hook=unique) for line in path.read_text(encoding="utf-8").splitlines()]
    canonical_json(rows)
    if len(rows) != completion["expected_rows"] or completion["completed_rows"] != len(rows) or manifest["expected_rows"] != len(rows):
        raise ValueError("Paired result count differs from completion")
    expected = {(entry["episode_id"],entry["policy_call_idx"],modality):entry["context_id"]
                for entry in configuration["bank"]["contexts"] for modality in configuration["protocol"]["numerics"]}
    observed = {}
    for row in rows:
        key = (row["episode_id"],row["policy_call_idx"],row["modality"])
        if row["evaluation_id"] != manifest["evaluation_id"] or key in observed:
            raise ValueError("Repeated or foreign paired output decision")
        observed[key] = row["source_context_id"]
        if row["status"] == "evaluated":
            authenticate_action_results(row["results"])
        elif row["status"] not in {"numerical_failure","call_not_reached"}:
            raise ValueError("Unknown paired context outcome")
    if observed != expected:
        raise ValueError("Paired output does not match the full declared bank population")
    return rows,manifest,completion


def summarize_study(args):
    path = Path(args.input)
    rows, manifest, completion = load_completed_study(path)
    protocol = manifest["configuration"]["protocol"]
    analysis = protocol["analysis"]
    if (args.family_alpha,args.draws,args.seed) != (analysis["family_alpha"],analysis["draws"],analysis["seed"]):
        raise ValueError("Requested interval configuration differs from the locked analysis protocol")
    family = protocol["contrast_family"]
    if not isinstance(family, list) or not family:
        raise ValueError("Explicit contrast family required for summary")
    results = []
    for contrast in family:
        planned = [r for r in rows if r["modality"] == contrast["modality"]]
        if not planned:
            raise ValueError("A contrast has no planned modality population")
        result = paired_episode_summary(planned, **{k: contrast[k] for k in
            ("method", "control", "response", "direction", "metric")},
            alpha=args.family_alpha/len(family), draws=args.draws, seed=args.seed)
        result["modality"] = contrast["modality"]
        if protocol["stage"] == "variance_only_pilot":
            # This is variance-only reporting. Contrast hashes are reversible
            # and do not establish blinding of the analyst or raw outputs.
            result = {k: result[k] for k in ("planned_contexts", "valid_contexts", "planned_episodes",
                "analysis_episodes", "episode_variance", "episode_standard_error", "random_monte_carlo_standard_error",
                "outcome_counts", "conditioning")}
            result["contrast_id"] = object_hash(contrast)
        results.append(result)
    write_exclusive(args.out, dict(schema_version=1, stage=protocol["stage"],
        study_output_sha256=file_hash(path), study_manifest_sha256=file_hash(path.with_name(path.name+".manifest.json")),
        family_alpha=args.family_alpha, multiplicity_rule="Bonferroni across the declared contrast family",
        results=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bank = commands.add_parser("bank", help="Freeze recorded collector flags or an explicit prospective subset; CPU only")
    bank.add_argument("--metrics", type=Path, required=True)
    bank.add_argument("--selection", type=Path)
    bank.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="Execute the E01-gated paired protocol")
    add_replay_arguments(run)
    run.add_argument("--bank", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--e01-gate", type=Path, required=True)
    run.add_argument("--prepared", type=Path)
    run.add_argument("--prepared-sha256")
    summary = commands.add_parser("summarize", help="Episode-level paired estimates or variance-only pilot summaries")
    summary.add_argument("--input", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    summary.add_argument("--draws", type=int, required=True)
    summary.add_argument("--seed", type=int, required=True)
    summary.add_argument("--family-alpha", type=float, required=True)
    args = parser.parse_args()
    if args.command == "bank":
        write_exclusive(args.out, make_bank(args.metrics, strict_json(args.selection) if args.selection else None))
    elif args.command == "run":
        run_study(args)
    else:
        summarize_study(args)


if __name__ == "__main__":
    main()
