"""Exact median scale bounds for every executed call of a fixed pilot roster.

This module is independent of the frozen paired-inference implementation. Its
input is an upstream-authenticated endpoint bank, not a selected-call bank.
Hashes bind membership here; the exporter must authenticate actual terminals,
source rows, contexts and endpoint artifacts before supplying this structure.

For a fixed population median m, P(X <= m) >= 1/2 and P(X < m) <= 1/2.
Consequently binomial order-statistic tail bounds remain conservative with
atoms. Decisions below compare exact integer binomial sums to the supplied
alpha's exact floating-point ratio. No interpolated ranks or normal
approximations are used. Nonnegative support supplies a lower bound of zero
when the sample has no informative order bound; infinity is serialized as null.

Reference: https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/mediancl.htm
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from numbers import Real
from fractions import Fraction


MODALITIES = ("vision", "language", "state")
BANK_KIND = "complete_executed_call_baseline_endpoints"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Expected finite numeric " + name)
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError("Expected nonnegative finite " + name)
    return value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid integer " + name)
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing " + name)
    return value


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid SHA256 " + name)
    return value


def _lower_rank(n, alpha):
    """Largest k with P(Bin(n,.5) < k) <= alpha; k=0 uses support."""
    numerator, denominator = alpha.as_integer_ratio()
    total = 1 << n
    mass, cumulative, rank = 1, 0, 0
    accepted = 0
    for k in range(1, n + 1):
        cumulative += mass
        if cumulative * denominator > numerator * total:
            break
        rank, accepted = k, cumulative
        mass = mass * (n - k + 1) // k
    return rank, accepted / total


def exact_median_bounds(values, alpha_lower, alpha_upper):
    """Nonrandomized bounds for a fixed median of an iid nonnegative variable.

    The usual averaged middle-order sample median is only the point estimate.
    A zero lower bound is valid coverage information but cannot set a positive
    planning resolution. A null upper bound denotes positive infinity.
    """
    lower_alpha = _finite(alpha_lower, "lower-tail alpha", positive=True)
    upper_alpha = _finite(alpha_upper, "upper-tail alpha", positive=True)
    if lower_alpha >= 1 or upper_alpha >= 1 or lower_alpha + upper_alpha >= 1:
        raise ValueError("Tail alpha must be positive with sum below one")
    sample = sorted(_finite(value, "median observation") for value in values)
    if not sample:
        raise ValueError("Median bounds require every planned episode observation")
    n = len(sample)
    lower_rank, lower_error = _lower_rank(n, lower_alpha)
    reflected_rank, upper_error = _lower_rank(n, upper_alpha)
    upper_rank = n + 1 - reflected_rank
    point = sample[n // 2] if n % 2 else sample[n // 2 - 1] / 2 + sample[n // 2] / 2
    lower = sample[lower_rank - 1] if lower_rank else 0.
    upper = sample[upper_rank - 1] if upper_rank <= n else None
    return dict(value=point, lower=lower, upper=upper, n=n,
        lower_order_rank=lower_rank, upper_order_rank=upper_rank,
        lower_source="sample_order_statistic" if lower_rank else "known_nonnegative_support",
        upper_unbounded=upper is None,
        alpha_lower=lower_alpha, alpha_upper=upper_alpha,
        lower_noncoverage_bound=lower_error, upper_noncoverage_bound=upper_error,
        simultaneous_coverage_lower=1 - lower_error - upper_error,
        status="positive_exact_scale" if lower > 0 else "unresolved_nonpositive_scale_lower_bound",
        ties="conservative_binomial_tail_bounds_for_any_fixed_population_median",
        assumptions="independent_identically_distributed_finite_episode_variables; fixed_population_median")


def _plan(design, stage, look_weights, alpha):
    if design.get("stage") != "variance_only_pilot" or design.get("modalities") != list(MODALITIES):
        raise ValueError("Scale planning requires the declared three-modality pilot")
    strata = design["strata"]
    ids = [_identifier(s["id"], "stratum ID") for s in strata]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Missing or repeated strata")
    all_seeds = []
    for spec in strata:
        seeds = spec["reset_seeds"]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("A unique complete reset roster is required")
        all_seeds.extend(_integer(seed, "reset seed", 0) for seed in seeds)
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Reset seeds must be disjoint across strata")
    stages = design["pilot_stages"]
    if not stages or len(stages) != len(look_weights):
        raise ValueError("Every declared stage needs an explicit look weight")
    weights = [_finite(w, "look weight", positive=True) for w in look_weights]
    if sum(map(Fraction.from_float, weights)) > 1:
        raise ValueError("Look weights exceed their probability budget")
    total_alpha = _finite(alpha, "total simultaneous alpha", positive=True)
    if total_alpha >= 1:
        raise ValueError("Total alpha must be below one")
    previous = dict.fromkeys(ids, 0)
    stage_ids = []
    for item in stages:
        stage_ids.append(_identifier(item["id"], "stage ID"))
        counts = item["episodes_per_stratum"]
        if set(counts) != set(ids):
            raise ValueError("Stage must enumerate every stratum")
        for spec in strata:
            sid = spec["id"]
            count = _integer(counts[sid], "stage episode count")
            if not previous[sid] < count <= len(spec["reset_seeds"]):
                raise ValueError("Stages must be increasing prefixes of the reset roster")
            previous[sid] = count
    if len(set(stage_ids)) != len(stage_ids) or stage not in stage_ids:
        raise ValueError("Unknown or repeated pilot stage")
    groups = design["groups"]
    if not groups or len({g["id"] for g in groups}) != len(groups):
        raise ValueError("Missing or repeated fixed-weight groups")
    covered = set()
    for group in groups:
        _identifier(group["id"], "group ID")
        group_weights = group["weights"]
        if not group_weights or not set(group_weights) <= set(ids):
            raise ValueError("Group references unknown strata")
        values = [_finite(v, "fixed task weight", positive=True) for v in group_weights.values()]
        if not math.isclose(math.fsum(values), 1., rel_tol=0, abs_tol=1e-12):
            raise ValueError("Fixed task weights must sum to one")
        covered.update(group_weights)
    if covered != set(ids):
        raise ValueError("Every planned stratum must have a fixed group weight")
    index = stage_ids.index(stage)
    exact_alpha = Fraction.from_float(total_alpha) * Fraction.from_float(weights[index]) / (2 * len(ids) * len(MODALITIES))
    local_alpha = float(exact_alpha)
    if Fraction.from_float(local_alpha) > exact_alpha:
        local_alpha = math.nextafter(local_alpha, 0.)
    if local_alpha == 0:
        raise ValueError("Probability spending underflow cannot define a tail bound")
    return strata, stages[index], index, weights, total_alpha, local_alpha


def simultaneous_scale_report(design, endpoint_bank, stage, look_weights, alpha):
    """All-call episode means, stratum medians, and fixed-weight group bounds.

    endpoint_bank.strata[sid].episodes includes exactly the stage's reset-seed
    prefix. Every episode declares episode_id, reset_seed, terminal_record_sha256,
    terminal_policy_calls, executed_call_indices and calls. Calls declare
    policy_call_idx, context_id, source_row_sha256 and all three modalities.
    Each modality declares status, baseline_rms, endpoint_identity_sha256; an
    explicit failed status requires reason and baseline_rms=null. Missing rows
    are identity errors. Explicit endpoint failures produce unresolved scales.

    Schema 1 retains its positive-call requirement. Schema 2 additionally
    accepts an authenticated attempted episode with zero executed calls only
    when collection_status="failed", collection_failure_reason is nonempty,
    and both call lists are empty. Its undefined episode mean remains in the
    roster and makes every modality unresolved; it is never replaced by zero.
    """
    strata, stage_spec, index, weights, total_alpha, local_alpha = _plan(design, stage, look_weights, alpha)
    bank_version = endpoint_bank.get("schema_version")
    if type(bank_version) is not int or bank_version not in (1, 2) or endpoint_bank.get("kind") != BANK_KIND:
        raise ValueError("Unknown all-executed-call endpoint bank")
    if set(endpoint_bank["strata"]) != {s["id"] for s in strata}:
        raise ValueError("Endpoint bank must retain every planned stratum")
    seen_episodes, seen_contexts, seen_sources = set(), set(), set()
    output, counts, collection_counts = {}, Counter(), Counter()
    for spec in strata:
        sid = spec["id"]
        expected_seeds = spec["reset_seeds"][:stage_spec["episodes_per_stratum"][sid]]
        episodes = endpoint_bank["strata"][sid]["episodes"]
        by_seed = {}
        for episode in episodes:
            seed = _integer(episode["reset_seed"], "episode reset seed", 0)
            if seed in by_seed:
                raise ValueError("Repeated planned episode reset seed")
            by_seed[seed] = episode
        if set(by_seed) != set(expected_seeds):
            raise ValueError("Endpoint episodes differ from the exact planned stage prefix")
        membership, means, invalid = [], {m:[] for m in MODALITIES}, {m:[] for m in MODALITIES}
        for seed in expected_seeds:
            episode = by_seed[seed]
            eid = _identifier(episode["episode_id"], "episode ID")
            if eid in seen_episodes:
                raise ValueError("Repeated episode identity")
            seen_episodes.add(eid)
            terminal = _digest(episode["terminal_record_sha256"], "terminal record")
            n_calls = _integer(episode["terminal_policy_calls"], "terminal policy-call count", 0 if bank_version == 2 else 1)
            collection_failure = None
            if n_calls == 0:
                if episode.get("collection_status") != "failed":
                    raise ValueError("Zero-call episodes require explicit authenticated collection failure")
                collection_failure = _identifier(episode.get("collection_failure_reason"), "collection failure reason")
                if episode["executed_call_indices"] != [] or episode["calls"] != []:
                    raise ValueError("Zero-call collection failure must retain empty call lists")
                collection_counts["zero_call_failure"] += 1
            expected_calls = list(range(n_calls))
            declared = episode["executed_call_indices"]
            if (declared != expected_calls or any(type(i) is not int for i in declared)):
                raise ValueError("Executed call list must equal the authenticated terminal count")
            call_map = {}
            for call in episode["calls"]:
                number = _integer(call["policy_call_idx"], "policy call", 0)
                if number in call_map:
                    raise ValueError("Repeated executed call")
                call_map[number] = call
            if set(call_map) != set(expected_calls):
                raise ValueError("Missing or extra executed-call endpoint row")
            values, problems, contexts = {m:[] for m in MODALITIES}, {m:[] for m in MODALITIES}, []
            if collection_failure is not None:
                for modality in MODALITIES:
                    problems[modality].append(dict(call=None, context_id=None, status="collection_failure",
                        reason=collection_failure, terminal_record_sha256=terminal))
            for number in expected_calls:
                call = call_map[number]
                context = _identifier(call["context_id"], "context ID")
                source = _digest(call["source_row_sha256"], "source row")
                if context in seen_contexts or source in seen_sources:
                    raise ValueError("Repeated context or source-row identity")
                seen_contexts.add(context); seen_sources.add(source)
                if set(call["modalities"]) != set(MODALITIES):
                    raise ValueError("Each executed call must retain all three modality outcomes")
                descriptors = {}
                for modality in MODALITIES:
                    endpoint = call["modalities"][modality]
                    identity = _digest(endpoint["endpoint_identity_sha256"], "endpoint identity")
                    status = _identifier(endpoint["status"], "endpoint status")
                    counts[modality + ":" + status] += 1
                    descriptors[modality] = dict(status=status, endpoint_identity_sha256=identity)
                    if status == "finite":
                        values[modality].append(_finite(endpoint["baseline_rms"], "baseline RMS"))
                    else:
                        reason = _identifier(endpoint["reason"], "explicit endpoint failure reason")
                        if endpoint["baseline_rms"] is not None:
                            raise ValueError("A failed endpoint cannot fabricate a finite RMS")
                        problems[modality].append(dict(call=number, context_id=context, status=status, reason=reason))
                        descriptors[modality]["reason"] = reason
                contexts.append(dict(policy_call_idx=number, context_id=context, source_row_sha256=source,
                    within_episode_weight=1/n_calls, modalities=descriptors))
            episode_means = {}
            for modality in MODALITIES:
                if problems[modality]:
                    value = None
                    invalid[modality].append(dict(episode_id=eid, reset_seed=seed, failures=problems[modality]))
                else:
                    value = math.fsum(v/n_calls for v in values[modality])
                means[modality].append(value)
                episode_means[modality] = value
            member = dict(episode_id=eid, reset_seed=seed, terminal_record_sha256=terminal,
                terminal_policy_calls=n_calls, planned_episode_weight=1/len(expected_seeds),
                episode_mean_baseline_rms=episode_means, contexts=contexts)
            if collection_failure is not None:
                member.update(collection_status="failed", collection_failure_reason=collection_failure)
            membership.append(member)
        bounds = {}
        for modality in MODALITIES:
            if invalid[modality]:
                collection_failed = any(f["status"] == "collection_failure" for item in invalid[modality] for f in item["failures"])
                bounds[modality] = dict(value=None, lower=0., upper=None, n=len(expected_seeds),
                    upper_unbounded=True, status="unresolved_collection_failure" if collection_failed else "unresolved_endpoint_failure", invalid_episodes=invalid[modality],
                    confidence_bound_applicable=False)
            else:
                bounds[modality] = dict(exact_median_bounds(means[modality], local_alpha, local_alpha),
                    invalid_episodes=[], confidence_bound_applicable=True)
        output[sid] = dict(planned_episodes=len(expected_seeds), planned_calls=sum(e["terminal_policy_calls"] for e in membership),
            membership=membership, membership_sha256=_hash(membership), modalities=bounds)
    scales = {}
    for group in design["groups"]:
        for modality in MODALITIES:
            entries = [(w, output[sid]["modalities"][modality]) for sid,w in group["weights"].items()]
            failed = any(not item["confidence_bound_applicable"] for _,item in entries)
            point = None if failed else math.fsum(w*item["value"] for w,item in entries)
            lower = math.fsum(w*item["lower"] for w,item in entries)
            upper = None if any(item["upper"] is None for _,item in entries) else math.fsum(w*item["upper"] for w,item in entries)
            collection_failed = any(item["status"] == "unresolved_collection_failure" for _,item in entries)
            status = ("unresolved_collection_failure" if collection_failed else "unresolved_endpoint_failure") if failed else "positive_exact_scale" if lower > 0 else "unresolved_nonpositive_scale_lower_bound"
            scales[group["id"] + ":" + modality] = dict(value=point, lower=lower, upper=upper,
                halfwidth=.05*point if point is not None else None,
                substantial_difference=.10*point if point is not None else None,
                conservative_planning_halfwidth=.05*lower if not failed and lower > 0 else None,
                status=status, upper_unbounded=upper is None, confidence_bound_applicable=not failed,
                fixed_task_weights=group["weights"],
                definition="fixed_weighted_stratum_medians_of_episode_mean_all_executed_call_baseline_RMS",
                interpretation="benchmark_resolution_not_physical_utility_or_task_success_threshold")
    report = dict(kind="simultaneous_all_call_median_scale_report", schema_version=1,
        stage=stage, look_index=index, design_sha256=_hash(design), endpoint_bank_sha256=_hash(endpoint_bank),
        strata=output, scales=scales, endpoint_status_counts=dict(counts),
        endpoint_bank_schema_version=bank_version, collection_status_counts=dict(collection_counts),
        spending=dict(total_alpha=total_alpha, look_weights=weights, selected_look_weight=weights[index],
            alpha_per_stratum_modality_tail=local_alpha, tails=2, strata=len(strata), modalities=len(MODALITIES),
            total_allocated_alpha=total_alpha*math.fsum(weights), allocation="equal_stratum_modality_tail_then_prespecified_look_weight",
            simultaneous_over="all declared looks, strata and modalities; fixed nonnegative weighted groups inherit bounds"),
        assumptions=["Independent identically distributed well-defined finite episode-level variables within each stratum.",
            "Every executed call from every planned episode is retained; no endpoint or episode deletion.",
            "Stage prefixes and probability spending fixed before observing endpoint values; no cross-stratum independence needed for union bound.",
            "Upstream authenticates terminal counts, source/context identities and actual endpoint artifacts; this module verifies their supplied membership contract."],
        limits=["Median bounds do not bound the mean, variance, rare tails, attribution error, or task utility.",
            "They do not validate adaptive J/M/R recruitment or a full pilot stopping rule.",
            "Explicit endpoint failures leave the affected scale unresolved; zero lower bounds never receive an epsilon.",
            "Schema 2 zero-call collection failures retain undefined episode means and block all affected modality scales; schema 1 still requires positive calls."])
    _hash(report)  # Require a strictly finite JSON artifact; unbounded endpoints are null.
    return report
