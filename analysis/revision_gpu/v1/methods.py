"""Methods study (Must Fix 5): several attribution methods under Q and N with controls.

Every quantity is recomputed from the authenticated scoring records. The plan
is docs/nhsjs-revision/protocols/r1/analysis-plan-r1.json, recorded before
any production result existed, with amendments 1 and 2 beside it.
"""
from __future__ import annotations

from collections import defaultdict
import itertools
import math
from pathlib import Path

import numpy as np

from experiment_io import file_hash, strict_json
import revision_numerics as rn

RESPONSES = ("Q", "L2")
DIRECTIONS = ("deletion", "insertion")
DESCRIPTIVE_ALPHA = 0.05
EXTREME = -1.0


def load_bank(run_root, protocol_file, bank_file, bank_id, phases=("bf16", "fp32", "scoring"), only=None):
    """Every phase record of one bank, each authenticated by its completion hashes.

    With only (a set of context keys such as e0030-c017), other contexts are listed without phases.
    """
    protocol = strict_json(protocol_file)
    cfg = protocol["banks"][bank_id]
    if file_hash(bank_file) != cfg["bank_sha256"]:
        raise ValueError("Bank file differs from the protocol")
    bank = strict_json(bank_file)
    binding = dict(protocol_sha256=file_hash(protocol_file), bank_sha256=cfg["bank_sha256"])
    fp32_members = rn.resolve_members(cfg["fp32_contexts"], bank)
    contexts = []
    for entry in bank["contexts"]:
        item = dict(entry=entry)
        if entry["status"] == "available" and (only is None or rn.context_key(entry) in only):
            key = rn.context_key(entry)
            for phase in phases:
                if phase == "fp32" and entry["context_id"] not in fp32_members:
                    continue
                directory, _ = rn.load_phase(run_root, bank_id, key, phase, dict(binding, context_id=entry["context_id"]))
                item[phase] = strict_json(directory / "record.json")
                item[phase + "_dir"] = Path(directory)
        contexts.append(item)
    return protocol, cfg, bank, contexts


def round_rule_area(result, values, response):
    """Normalized area under the round count rule, the plan's sensitivity analysis, from stored values.

    The stored curve follows max1_round. The two rules differ only where round() gives zero groups
    and max1_round one; the zero-group response is the curve's first value, so the round-rule
    curve repeats it there and keeps every other point.
    """
    from faithfulness import auc_normalized
    if result["count_rule"] != "max1_round" or any(v is None for v in values):
        return None
    n = len(result["eligible_indices"])
    counts = rn.count_schedule(n, result["nominal_grid_percent"], "round")
    curve = []
    for count, stored, value in zip(counts, result["selected_counts"], values):
        if count != stored and count != 0:
            raise ValueError("Round and max1_round counts differ beyond the zero-group case")
        curve.append(value if count == stored else values[0])
    table = result["response_table"]
    endpoints = (table[result["input_response_id"]][response], table[result["baseline_response_id"]][response])
    if any(x is None for x in endpoints):
        return None
    return auc_normalized([100 * count / n for count in counts], curve, *endpoints,
                          denominator_min=result["denominator_min"].get(response, 0))


def value_of(source, name, direction, response, metric):
    """One ranking's area. source is a modality result; its rankings dict serves for the stored metrics."""
    rankings = source["rankings"] if "response_table" in source else source
    item = rankings.get(name)
    if item is None or item["status"] != "defined":
        return None
    outcome = item["curves"][direction]["responses"][response]
    if metric == "normalized_auc_round":
        return round_rule_area(source, outcome["values"], response)
    return outcome.get(metric)


def reason_of(result, name, direction, response):
    item = result["rankings"].get(name)
    if item is None:
        return f"{name} absent"
    if item["status"] != "defined":
        return f"{name} {item.get('reason', item['status'])}"
    return f"{name} {response} {item['curves'][direction]['responses'][response].get('status', 'undefined')}"


def exact_order_values(result, direction, response, metric):
    """Area of every ordering of a small population, from the stored subset responses.

    The exact control evaluates every subset at every count, so each order's
    nested prefixes are all in the response table.
    """
    from faithfulness import auc_normalized
    from scipy.integrate import trapezoid
    indices = result["eligible_indices"]
    if len(indices) > 8:
        raise ValueError("Exact order enumeration is limited to eight groups")
    table = result["response_table"]
    lookup = {frozenset(item["baseline_positions"]): item[response] for item in table.values()}
    counts = result["selected_counts"]
    if metric == "normalized_auc_round":
        counts = rn.count_schedule(len(indices), result["nominal_grid_percent"], "round")
    fractions = [count / len(indices) for count in counts]
    axis = fractions if metric != "nominal_normalized_auc" else [k / 100 for k in result["nominal_grid_percent"]]
    endpoints = (table[result["input_response_id"]][response], table[result["baseline_response_id"]][response])
    if any(value is None for value in endpoints):
        return None
    cutoff = result["denominator_min"].get(response, 0)
    full = frozenset(indices)
    values = []
    for order in itertools.permutations(indices):
        curve = []
        for count in counts:
            prefix = frozenset(order[:count])
            curve.append(lookup[prefix if direction == "deletion" else full - prefix])
        if any(value is None for value in curve):
            return None
        if metric == "raw_auc":
            values.append(float(trapezoid(curve, fractions)))
        else:
            values.append(auc_normalized([100 * x for x in axis], curve, *endpoints, denominator_min=cutoff))
    return values


def random_reference(result, direction, response, metric):
    """(mean, sd, kind) of the random-order control for one modality result."""
    rankings = result["rankings"]
    if "random_exact" in rankings:
        mean = value_of(result, "random_exact", direction, response, metric)
        values = exact_order_values(result, direction, response, metric)
        if mean is None or values is None or any(v is None for v in values):
            return None, None, "exact_undefined"
        return mean, float(np.std(values)), "exact_all_orders"
    names = sorted(name for name in rankings if name.startswith("random_"))
    values = [value_of(result, name, direction, response, metric) for name in names]
    if not values or any(v is None for v in values):
        return None, None, "random_undefined"
    return math.fsum(values) / len(values), float(np.std(values, ddof=1)), f"seeded_{len(values)}"


def advantage(value, reference, direction):
    """Positive favors the ranking: lower deletion or higher insertion area than random."""
    return reference - value if direction == "deletion" else value - reference


def context_table(contexts, modality, names, *, metric="normalized_auc"):
    """One row per planned call, ranking, response and direction with area, advantage, z and the reason for a gap.

    A planned call that is not available (for example a Month 8 call the episode never reached)
    gets rows for every ranking with no values, so the complete-case rule sees it.
    """
    rows = []
    for item in contexts:
        entry = item["entry"]
        base = dict(episode_id=entry["episode_id"], episode=entry["episode"], policy_call_idx=entry["policy_call_idx"],
                    context_id=entry["context_id"], modality=modality, status=entry["status"])
        result = item["scoring"]["modalities"][modality] if entry["status"] == "available" else None
        for response in RESPONSES:
            for direction in DIRECTIONS:
                mean, sd, kind = random_reference(result, direction, response, metric) if result else (None, None, None)
                for name in names:
                    value = value_of(result, name, direction, response, metric) if result else None
                    adv = None if value is None or mean is None else advantage(value, mean, direction)
                    reason = None
                    if result is None:
                        reason = entry["status"]
                    elif value is None:
                        reason = reason_of(result, name, direction, response)
                    elif mean is None:
                        reason = f"random control {kind}"
                    rows.append(dict(base, ranking=name, response=response, direction=direction, value=value,
                                     random_mean=mean, random_sd=sd, random_kind=kind, advantage=adv,
                                     z=None if adv is None or not sd else adv / sd, reason=reason))
    return rows


def complete_case(rows, names, direction, planned):
    """Episodes whose every planned call has a defined advantage for every name under both responses.

    Returns the kept episode ids in planned order and, for each excluded episode, the first reason.
    """
    reasons = {}
    seen = set()
    for row in rows:
        if row["direction"] != direction or row["ranking"] not in names:
            continue
        seen.add(row["episode_id"])
        if row["advantage"] is None and row["episode_id"] not in reasons:
            reasons[row["episode_id"]] = f"call {row['policy_call_idx']}: {row['reason']}"
    for episode in planned:
        if episode not in seen:
            reasons.setdefault(episode, "no planned call")
    return [episode for episode in planned if episode not in reasons], reasons


def episode_series(rows, field, episodes):
    """Equal-call mean of a field within each listed episode, in the listed order; None if any value is missing."""
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row[field])
    series = {}
    for episode in episodes:
        values = by_episode.get(episode, [])
        if values and all(v is not None for v in values):
            series[episode] = math.fsum(values) / len(values)
    return series


def available_series(rows, field):
    """Equal-episode means over all available defined calls (the 'all available calls' population)."""
    by_episode = defaultdict(list)
    for row in rows:
        if row[field] is not None:
            by_episode[row["episode_id"]].append(row[field])
    return {episode: math.fsum(values) / len(values) for episode, values in by_episode.items()}


def bootstrap_indices(n, draws, seed):
    """The resampling stream of paired_comparison.paired_episode_summary, as index blocks."""
    rng = np.random.Generator(np.random.PCG64(seed))
    for begin in range(0, draws, 1000):
        yield rng.integers(0, n, size=(min(begin + 1000, draws) - begin, n))


def resample(matrix, draws, seed):
    """Bootstrap means of every column over one shared episode resampling (rows are episodes)."""
    return np.concatenate([matrix[index].mean(axis=1) for index in bootstrap_indices(len(matrix), draws, seed)])


def percentile(samples, alpha):
    return np.quantile(samples, [alpha / 2, 1 - alpha / 2]).tolist()


def interval_record(values, samples, alpha, *, sensitivity=False):
    """A percentile interval with its specification; with sensitivity, BCa and t beside it (amendment 1)."""
    record = dict(interval_method="percentile", alpha=alpha, episodes=len(values), interval=percentile(samples, alpha))
    if sensitivity and len(values) > 1:
        from analysis.revision_gpu.v1.coverage import bca_interval, t_interval
        record["sensitivity"] = dict(bca=dict(interval_method="bca", alpha=alpha, episodes=len(values),
                                              interval=[float(x) for x in bca_interval(np.asarray(values), samples, alpha)]),
                                     t=dict(interval_method="t", alpha=alpha, episodes=len(values),
                                            interval=[float(x) for x in t_interval(np.asarray(values), alpha)]))
    return record


def extreme_episodes(rows, field, episodes):
    """Amendment 1: episodes with any call of the compared quantity below -1."""
    kept = set(episodes)
    return len({row["episode_id"] for row in rows if row["episode_id"] in kept and row[field] is not None
                and row[field] < EXTREME})


def joint_summary(series, episodes, *, draws, seed, alpha=DESCRIPTIVE_ALPHA, unique_alpha=None):
    """Means, intervals and best-method shares over one shared resampling of the listed episodes.

    Exact ties split a draw's probability_best credit; an exact tie in means leaves the preferred
    method undetermined (best is None and tie lists the leaders).
    """
    names = list(series)
    episodes = [episode for episode in episodes if all(episode in series[name] for name in names)]
    n = len(episodes)
    if n < 2 or len(names) < 2:
        return dict(episodes=n, names=names, status="fewer_than_two_episodes" if n < 2 else "one_series"), None
    matrix = np.array([[series[name][episode] for name in names] for episode in episodes], dtype=float)
    means = matrix.mean(axis=0)
    samples = resample(matrix, draws, seed)
    is_max = samples == samples.max(axis=1, keepdims=True)
    credit = (is_max / is_max.sum(axis=1, keepdims=True)).mean(axis=0)
    order = sorted(range(len(names)), key=lambda i: -means[i])
    leaders = [names[i] for i in order if means[i] == means[order[0]]]
    first, second = order[0], order[1]
    out = dict(episodes=n, episode_ids=episodes, names=names,
               mean={name: float(means[i]) for i, name in enumerate(names)},
               interval={name: interval_record(matrix[:, i], samples[:, i], alpha) for i, name in enumerate(names)},
               probability_best={name: float(credit[i]) for i, name in enumerate(names)},
               best=leaders[0] if len(leaders) == 1 else None, tie=leaders if len(leaders) > 1 else None,
               second=names[second])
    gap_values, gap_samples = matrix[:, first] - matrix[:, second], samples[:, first] - samples[:, second]
    out["best_minus_second"] = dict(mean=float(gap_values.mean()),
                                    descriptive=interval_record(gap_values, gap_samples, alpha))
    if unique_alpha is not None:
        adjusted = interval_record(gap_values, gap_samples, unique_alpha, sensitivity=True)
        out["best_minus_second"]["adjusted"] = adjusted
        out["unique"] = None if out["best"] is None else bool(adjusted["interval"][0] > 0)
    return out, dict(matrix=matrix, samples=samples, names=names, episodes=episodes)


def contrast_rows(contexts, contrast, metric):
    """Per-call value of one primary contrast (positive favors the method) with the reason for a gap."""
    rows = []
    for item in contexts:
        entry = item["entry"]
        base = dict(episode_id=entry["episode_id"], policy_call_idx=entry["policy_call_idx"])
        if entry["status"] != "available":
            rows.append(dict(base, value=None, reason=entry["status"]))
            continue
        result = item["scoring"]["modalities"]["vision"]
        direction = contrast["direction"]
        if contrast["contrast"] == "matched_calibrated":
            terms = [(contrast["method"], contrast["method_response"], 1), (contrast["control"], contrast["control_response"], -1)]
        elif contrast["contrast"] == "method_minus_method":
            terms = [(contrast["method"], contrast["response"], 1), (contrast["control"], contrast["response"], -1)]
        else:
            terms = [(contrast["method"], contrast["response"], 1)]
        total, reason = 0.0, None
        for name, response, sign in terms:
            mean, _, kind = random_reference(result, direction, response, metric)
            raw = value_of(result, name, direction, response, metric)
            if raw is None or mean is None:
                reason = reason_of(result, name, direction, response) if raw is None else f"random control {kind}"
                break
            total += sign * advantage(raw, mean, direction)
        rows.append(dict(base, value=None if reason else total, reason=reason))
    return rows


def contrast_population(contexts, contrast, metric):
    """Complete case under both responses for the rankings of one contrast, its random control included."""
    names = [contrast["method"]] + ([contrast["control"]] if contrast["contrast"] != "method_minus_random" else [])
    rows = context_table(contexts, "vision", names, metric=metric)
    planned = list(dict.fromkeys(item["entry"]["episode_id"] for item in contexts))
    return complete_case(rows, names, contrast["direction"], planned)


def primary_contrasts(contexts, plan, *, metric="normalized_auc", alpha=None):
    """The seven pre-registered vision-deletion contrasts.

    alpha None gives the registered Bonferroni level (family_alpha over the family size, primary bank);
    replication banks pass the descriptive 0.05.
    """
    family = plan["methods_study"]["primary_family"]
    boot = plan["methods_study"]["bootstrap"]
    alpha = plan["methods_study"]["family_alpha"] / len(family) if alpha is None else alpha
    planned = list(dict.fromkeys(item["entry"]["episode_id"] for item in contexts))
    out = []
    for contrast in family:
        kept, excluded = contrast_population(contexts, contrast, metric)
        rows = contrast_rows(contexts, contrast, metric)
        series = episode_series(rows, "value", kept)
        episodes = [e for e in kept if e in series]
        values = np.array([series[e] for e in episodes], dtype=float)
        record = dict(contrast, metric=metric, estimate=float(values.mean()) if len(values) else None,
                      episodes=len(values), planned_episodes=len(planned),
                      excluded_episodes={e: excluded[e] for e in planned if e in excluded},
                      extreme_episodes=extreme_episodes(rows, "value", episodes), alpha=alpha)
        if len(values) > 1:
            samples = resample(values[:, None], boot["draws"], boot["seed"])[:, 0]
            record.update(interval_record(values, samples, alpha, sensitivity=True))
        out.append(record)
    return out


def decisions(contexts, modality, methods, fixed, plan, *, metric="normalized_auc"):
    """Preferred method per response and direction, and the four ways calibration could change it."""
    boot = plan["methods_study"]["bootstrap"]
    unique_alpha = plan["methods_study"]["family_alpha"] / 4
    names = list(methods) + list(fixed)
    rows = context_table(contexts, modality, names, metric=metric)
    planned = list(dict.fromkeys(item["entry"]["episode_id"] for item in contexts))
    result = {}
    for direction in DIRECTIONS:
        kept, excluded = complete_case(rows, names, direction, planned)
        for response in RESPONSES:
            cell = [r for r in rows if r["response"] == response and r["direction"] == direction]
            by_name = defaultdict(list)
            for row in cell:
                by_name[row["ranking"]].append(row)
            series = {name: episode_series(by_name[name], "advantage", kept) for name in names}
            z_series = {name: episode_series(by_name[name], "z", kept) for name in methods}
            summary, draws = joint_summary({name: series[name] for name in methods}, kept, draws=boot["draws"],
                                           seed=boot["seed"], unique_alpha=unique_alpha)
            z_summary, _ = joint_summary(z_series, kept, draws=boot["draws"], seed=boot["seed"])
            everything, all_draws = joint_summary(series, kept, draws=boot["draws"], seed=boot["seed"])
            extremes = {name: extreme_episodes(by_name[name], "advantage", summary.get("episode_ids", [])) for name in methods}
            if summary.get("best") is not None:
                best, second = summary["best"], summary["second"]
                gap_rows = [dict(episode_id=a["episode_id"], gap=None if a["advantage"] is None or b["advantage"] is None
                                 else a["advantage"] - b["advantage"]) for a, b in zip(by_name[best], by_name[second])]
                summary["best_minus_second"]["extreme_episodes"] = extreme_episodes(gap_rows, "gap", summary["episode_ids"])
            beats = None
            if fixed and summary.get("best") is not None and all_draws is not None:
                columns = {name: i for i, name in enumerate(all_draws["names"])}
                best = summary["best"]
                means = everything["mean"]
                best_fixed = max(fixed, key=lambda name: means[name])
                gap_values = all_draws["matrix"][:, columns[best]] - all_draws["matrix"][:, columns[best_fixed]]
                gap_samples = all_draws["samples"][:, columns[best]] - all_draws["samples"][:, columns[best_fixed]]
                beats = dict(best_fixed=best_fixed, gaps={name: means[best] - means[name] for name in fixed},
                             all_fixed_below=all(means[best] > means[name] for name in fixed),
                             best_minus_best_fixed=dict(mean=float(gap_values.mean()),
                                                        descriptive=interval_record(gap_values, gap_samples, DESCRIPTIVE_ALPHA),
                                                        adjusted=interval_record(gap_values, gap_samples, unique_alpha)))
            bars = {}
            if direction == "deletion":
                for bar in (0.40, 0.5):
                    shares, passes = {}, {}
                    for name in methods:
                        values = [r["value"] for r in by_name[name] if r["episode_id"] in set(kept)]
                        shares[name] = None if not values else sum(v <= bar for v in values) / len(values)
                        mean = episode_series(by_name[name], "value", kept)
                        passes[name] = None if not mean else math.fsum(mean.values()) / len(mean) <= bar
                    bars[str(bar)] = dict(call_share_at_or_below=shares, episode_mean_passes=passes)
            all_available = {name: available_series(by_name[name], "advantage") for name in methods}
            available_means = {name: math.fsum(v.values()) / len(v) for name, v in all_available.items() if v}
            top = max(available_means.values()) if available_means else None
            leaders = [name for name, v in available_means.items() if v == top]
            result[f"{response}/{direction}"] = dict(
                advantage=summary, z_scaled=z_summary, with_fixed_orders=everything, beats_fixed_orders=beats,
                bars=bars, extreme_episodes=extremes,
                all_available=dict(mean=available_means, episodes={name: len(v) for name, v in all_available.items()},
                                   best=leaders[0] if len(leaders) == 1 else None))
        result[f"population/{direction}"] = dict(planned_episodes=len(planned), complete_case_episodes=len(kept),
                                                 excluded_episodes={e: excluded[e] for e in planned if e in excluded})
    return dict(modality=modality, metric=metric, cells=result, changes=calibration_changes(result, methods), rows=rows)


def _differs(a, b):
    """True or False when both preferred methods are determined, None otherwise."""
    return None if a is None or b is None else a != b


def calibration_changes(result, methods):
    """The four registered routes by which calibration could change the preferred method."""
    changes = {}
    for direction in DIRECTIONS:
        q, n = result[f"Q/{direction}"], result[f"L2/{direction}"]
        changes[direction] = dict(
            across_responses=_differs(q["advantage"].get("best"), n["advantage"].get("best")),
            under_z_scaling={r: _differs(result[f"{r}/{direction}"]["advantage"].get("best"),
                                         result[f"{r}/{direction}"]["z_scaled"].get("best")) for r in RESPONSES},
            through_population={r: _differs(result[f"{r}/{direction}"]["advantage"].get("best"),
                                            result[f"{r}/{direction}"]["all_available"]["best"]) for r in RESPONSES})
        if direction == "deletion":
            routes = {}
            for bar in ("0.4", "0.5"):
                route = {}
                for r in RESPONSES:
                    shares = {k: v for k, v in result[f"{r}/deletion"]["bars"][bar]["call_share_at_or_below"].items()
                              if v is not None}
                    preferred = result[f"{r}/deletion"]["advantage"].get("best")
                    top = max(shares.values()) if shares else None
                    leaders = sorted(name for name, v in shares.items() if v == top)
                    route[r] = dict(highest_share=leaders, preferred=preferred,
                                    differs=None if preferred is None or not leaders else preferred not in leaders)
                q_pass, n_pass = (result[f"{r}/deletion"]["bars"][bar]["episode_mean_passes"] for r in RESPONSES)
                route["pass_flips_between_responses"] = sorted(name for name in methods if q_pass.get(name) is not None
                                                               and n_pass.get(name) is not None and q_pass[name] != n_pass[name])
                routes[bar] = route
            changes[direction]["fixed_bars"] = routes
    return changes


def response_shift_rows(contexts, modality, names, *, metric="normalized_auc"):
    """Delta = AUC_Q - AUC_N of the same ranking, plus overshoot flags, per context and direction."""
    rows = []
    for item in contexts:
        if "scoring" not in item:
            continue
        entry = item["entry"]
        result = item["scoring"]["modalities"][modality]
        table = result["response_table"]
        f_baseline = table[result["baseline_response_id"]]["Q"]
        for name in names:
            ranking = result["rankings"].get(name)
            if ranking is None or ranking["status"] != "defined":
                continue
            for direction in DIRECTIONS:
                responses = ranking["curves"][direction]["responses"]
                q, n = responses["Q"].get(metric), responses["L2"].get(metric)
                values = responses["Q"]["values"]
                # Overshoot: an intervention farther from the reference than the all-baseline input (r > B).
                overshoot = None if f_baseline is None else any(v is not None and v < f_baseline for v in values[1:-1])
                rows.append(dict(episode_id=entry["episode_id"], episode=entry["episode"],
                                 policy_call_idx=entry["policy_call_idx"], modality=modality, ranking=name,
                                 direction=direction, auc_q=q, auc_n=n,
                                 delta=None if q is None or n is None else q - n, overshoot=overshoot,
                                 baseline_residual_q=f_baseline))
    return rows


def rms_units_rows(contexts, modality, names):
    """Amendment 2: RMS reported in units only, as each ranking's raw RMS area (action units) per call."""
    rows = []
    for item in contexts:
        if "scoring" not in item:
            continue
        entry = item["entry"]
        result = item["scoring"]["modalities"][modality]
        for name in names:
            for direction in DIRECTIONS:
                rows.append(dict(episode_id=entry["episode_id"], policy_call_idx=entry["policy_call_idx"], modality=modality,
                                 ranking=name, direction=direction,
                                 raw_rms_auc=value_of(result, name, direction, "RMS", "raw_auc")))
    return rows
