"""Numerics comparison (Must Fix 6): integration rule, budget and precision on preserved contexts.

Reads the authenticated bf16, fp32 and scoring records of each context and
returns flat tables. Map comparisons against the fp32 references are recomputed
here from the stored maps, one labelled family per reference: A2 T512 on every
fp32 context and A4 where it converged (amendment 2); fp32 rules the worker did
not store come from the worker's own comparisons. Self-comparisons are left out.
Ranking agreement is also measured on the orders the methods study scored.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import torch

import ig_quadrature as q
from revision_numerics import TARGETS, count_schedule

RULE = re.compile(r"^(T|E|M|S|RT|RM)(\d+)$")
QUALIFICATION_TOLERANCE = 1e-3


def rule_family(name):
    match = RULE.match(name)
    return (match.group(1), int(match.group(2))) if match else (name, None)


def residual_rows(bank_id, item):
    """Completeness residuals for every arm, rule and budget of one context.

    A0 sums its map in the input dtype, as the historical .sum().item() did, so its residual
    mixes the quadrature error with the bf16 reduction; the *_fp32_sum columns repeat it with
    the same map summed in fp32, which isolates the reduction.
    """
    entry = item["entry"]
    base = dict(bank=bank_id, context=entry["context_id"], episode=entry["episode"], call=entry["policy_call_idx"])
    rows = []
    for modality, record in item["bf16"]["modalities"].items():
        for target, cell in record["targets"].items():
            for name, hist in cell["historical"].items():
                totals = hist["nonfinite_totals"]
                gap = hist["expected_gap"]
                fp32_residual = None if gap is None else abs(hist["ig_sum_fp32"] - gap)
                rows.append(dict(base, modality=modality, target=target, arm="A0", rule="E", budget=hist["m"],
                                 expected_gap=gap, ig_sum=hist["ig_sum_input_dtype"],
                                 absolute_residual=hist["absolute_residual"], relative_residual=hist["relative_residual"],
                                 ig_sum_fp32=hist["ig_sum_fp32"], absolute_residual_fp32_sum=fp32_residual,
                                 relative_residual_fp32_sum=None if gap in (None, 0) else fp32_residual / abs(gap),
                                 nonfinite=totals["nan"] + totals["posinf"] + totals["neginf"],
                                 failed_forwards=totals["failed_forwards"]))
            nested = cell["nested"]
            for name, summary in nested.get("residuals", {}).items():
                family, budget = rule_family(name)
                rows.append(dict(base, modality=modality, target=target, arm="A1", rule=family, budget=budget,
                                 expected_gap=nested["expected_gap"], ig_sum=summary["ig_sum"],
                                 absolute_residual=summary["absolute_residual"],
                                 relative_residual=summary["relative_residual"], nonfinite=None, failed_forwards=None))
    if "fp32" in item and item["fp32"].get("status") == "evaluated":
        for modality, record in item["fp32"]["modalities"].items():
            for target, cell in record["targets"].items():
                nested = cell["nested"]
                for name, summary in nested.get("residuals", {}).items():
                    family, budget = rule_family(name)
                    rows.append(dict(base, modality=modality, target=target, arm="A2", rule=family, budget=budget,
                                     expected_gap=nested["expected_gap"], ig_sum=summary["ig_sum"],
                                     absolute_residual=summary["absolute_residual"],
                                     relative_residual=summary["relative_residual"], nonfinite=None, failed_forwards=None))
                for name, gl in cell.get("gauss_legendre", {}).items():
                    summary = gl.get("summary")
                    if summary:
                        rows.append(dict(base, modality=modality, target=target, arm="A3", rule="GL", budget=gl["n"],
                                         expected_gap=gl["expected_gap"], ig_sum=summary["ig_sum"],
                                         absolute_residual=summary["absolute_residual"],
                                         relative_residual=summary["relative_residual"], nonfinite=None, failed_forwards=None))
                adaptive = cell.get("adaptive")
                if adaptive and adaptive.get("summary"):
                    summary = adaptive["summary"]
                    rows.append(dict(base, modality=modality, target=target, arm="A4", rule="GK21",
                                     budget=adaptive["evaluations"], expected_gap=adaptive["expected_gap"],
                                     ig_sum=summary["ig_sum"], absolute_residual=summary["absolute_residual"],
                                     relative_residual=summary["relative_residual"], nonfinite=None, failed_forwards=None,
                                     converged=adaptive["converged"], error_l1=adaptive["error_l1"]))
    return rows


def comparison_rows(bank_id, item):
    """In-driver coordinate and ranking comparisons, flattened."""
    entry = item["entry"]
    base = dict(bank=bank_id, context=entry["context_id"], episode=entry["episode"], call=entry["policy_call_idx"])
    rows = []
    for modality, record in item["bf16"]["modalities"].items():
        for target, cell in record["targets"].items():
            for name, metrics in cell["nested"].get("versus_comparator", {}).items():
                rows.append(_flatten(dict(base, modality=modality, target=target, family="bf16_vs_A1_comparator",
                                          candidate=name), metrics))
    return rows


def converged_a4(cell):
    adaptive = cell.get("adaptive")
    return bool(adaptive and adaptive.get("converged") and adaptive.get("defined", True))


def qualification_rows(bank_id, item):
    """Plan reference rule: A4 where it converged, qualified when A4, GL128 and A2 T512 agree within
    relative L1 1e-3 pairwise; the largest pairwise disagreement is the resolution floor.

    A4's convergence test bounds the L1 error of the path-gradient integral (rtol on its L1 norm,
    a4_error_space), not of the attribution map, which weights each coordinate by its displacement.
    Convergence alone therefore does not bound the map error; the pairwise map-level agreement is
    the check that does. qualified is True only when A4 converged and the floor is within tolerance;
    qualification_reason says which condition failed.
    """
    if "fp32" not in item or item["fp32"].get("status") != "evaluated":
        return []
    entry = item["entry"]
    rows = []
    for modality, record in item["fp32"]["modalities"].items():
        for target, cell in record["targets"].items():
            pairs = cell.get("qualification")
            if not pairs:
                continue
            values = {pair: metrics.get("relative_l1") for pair, metrics in pairs.items()}
            defined = [v for v in values.values() if v is not None]
            floor = max(defined) if len(defined) == len(values) else None
            adaptive = cell["adaptive"]
            converged = converged_a4(cell)
            last = (adaptive.get("log") or [{}])[-1]
            integral_l1 = last.get("integral_l1")
            if floor is None:
                reason = "pair_undefined"
            elif not converged:
                reason = "a4_not_converged"
            elif floor > QUALIFICATION_TOLERANCE:
                reason = "floor_above_tolerance"
            else:
                reason = "qualified"
            rows.append(dict(bank=bank_id, context=entry["context_id"], episode=entry["episode"], call=entry["policy_call_idx"],
                             modality=modality, target=target, a4_converged=converged,
                             a4_evaluations=adaptive.get("evaluations"), a4_error_l1=adaptive.get("error_l1"),
                             a4_error_space="path_gradient_integral_l1",
                             a4_relative_error_estimate=(adaptive.get("error_l1") / integral_l1
                                                         if integral_l1 and adaptive.get("error_l1") is not None else None),
                             **{f"relative_l1[{pair}]": v for pair, v in sorted(values.items())},
                             resolution_floor=floor, qualified=None if floor is None else reason == "qualified",
                             qualification_reason=reason))
    return rows


def _flatten(row, metrics):
    row = dict(row)
    for key in ("relative_l1", "l1", "max_abs", "signed_sum_difference", "group_spearman", "top1_equal",
                "top5pct_overlap", "top5pct_size"):
        row[key] = metrics.get(key)
    jaccard = metrics.get("deleted_set_jaccard", [])
    row["min_deleted_set_jaccard"] = min((j["jaccard"] for j in jaccard), default=None)
    row["deleted_set_jaccard"] = json.dumps([[j["count"], j["jaccard"]] for j in jaccard])
    return row


def reference_rows(bank_id, item, protocol):
    """Every bf16 and fp32 map against each fp32 reference.

    Families fp32_vs_A2T512 and bf16_vs_A2T512 (the protocol's comparator rule, T512 in production)
    cover every evaluated fp32 context; fp32_vs_A4 and bf16_vs_A4 cover the contexts where A4
    converged. A reference is never compared with itself. Stored maps are recomputed here
    (source "recomputed"). fp32 rules whose maps the worker did not store come from the worker's
    own comparisons against the reference it used, versus_reference (source "worker"); for
    stored maps that reference's worker value is kept beside the recomputed one, worker_relative_l1.
    """
    if "fp32" not in item or item["fp32"].get("status") != "evaluated":
        return []
    entry = item["entry"]
    base = dict(bank=bank_id, context=entry["context_id"], episode=entry["episode"], call=entry["policy_call_idx"])
    maps = dict(bf16=torch.load(item["bf16_dir"] / "maps.pt", weights_only=True, map_location="cpu"),
                fp32=torch.load(item["fp32_dir"] / "maps.pt", weights_only=True, map_location="cpu"))
    grid = protocol["evaluation"]["grid_percent"]
    rows = []
    for modality, record in item["fp32"]["modalities"].items():
        axis, count = record["axis"], record["eligible_count"]
        eligible = q.Eligible(list(range(count)), axis)
        counts = count_schedule(count, grid, protocol["evaluation"]["count_rule"])
        for target in TARGETS:
            prefix = f"{modality}/{target}/"
            cell = record["targets"][target]
            worker_reference = cell.get("reference")
            worker = cell.get("versus_reference", {})
            references = ["A2" + protocol["numerics"]["comparator_rule"]] + (["A4"] if converged_a4(cell) else [])
            for reference_name in references:
                reference = maps["fp32"].get(prefix + reference_name)
                if reference is None:
                    continue
                from_worker = worker if reference_name == worker_reference else {}
                recomputed = set()
                for phase, stored in maps.items():
                    for key, value in stored.items():
                        if not key.startswith(prefix) or key.endswith("/group_scores") or "/A1PG" in key:
                            continue
                        candidate = key[len(prefix):]
                        if phase == "fp32" and candidate == reference_name:
                            continue
                        metrics = q.compare_maps(reference, value, eligible, counts=counts)
                        if phase == "fp32":
                            recomputed.add(candidate)
                        rows.append(_flatten(dict(base, modality=modality, target=target,
                                                  family=f"{phase}_vs_{reference_name}", reference=reference_name,
                                                  candidate=candidate, source="recomputed",
                                                  worker_relative_l1=(from_worker[candidate]["relative_l1"]
                                                                      if phase == "fp32" and candidate in from_worker
                                                                      else None)), metrics))
                for candidate, metrics in sorted(from_worker.items()):
                    if candidate in recomputed or candidate == reference_name:
                        continue
                    rows.append(_flatten(dict(base, modality=modality, target=target,
                                              family=f"fp32_vs_{reference_name}", reference=reference_name,
                                              candidate=candidate, source="worker", worker_relative_l1=None), metrics))
    return rows


def _order_spearman(reference, candidate):
    """Spearman correlation of two total orders of the same items, positions as ranks."""
    n = len(reference)
    if n < 2 or sorted(reference) != sorted(candidate):
        return None
    position = {value: i for i, value in enumerate(candidate)}
    squared = sum((i - position[value]) ** 2 for i, value in enumerate(reference))
    return 1 - 6 * squared / (n * (n * n - 1))


def ranking_rows(bank_id, item, comparator="T512"):
    """Ranking agreement on the orders the methods study scored, amendment-2 pairs.

    compare_maps ranks float64 group sums of the maps, while the scored rankings rank the group
    scores the worker stored (bf16 maps reduce in bf16, so ties appear) with the repository tie
    rule. These rows measure the scored rankings themselves, from the scoring record: Spearman of
    the order positions, Spearman of the scored group scores (average ranks), top-1, top-5% overlap
    and the deleted-set Jaccard at each realized count.
    """
    entry = item["entry"]
    base = dict(bank=bank_id, context=entry["context_id"], episode=entry["episode"], call=entry["policy_call_idx"])
    rows = []
    for modality, result in item["scoring"]["modalities"].items():
        rankings = result["rankings"]
        counts = list(dict.fromkeys(result["selected_counts"]))
        for target in TARGETS:
            for candidate, reference in automatic_pairs(item, modality, target, comparator):
                if candidate not in rankings or reference not in rankings:
                    continue
                cand, ref = rankings[candidate], rankings[reference]
                row = dict(base, modality=modality, target=target, candidate=candidate, reference=reference,
                           candidate_status=cand["status"], reference_status=ref["status"],
                           candidate_tied_positions=cand.get("tied_positions"),
                           reference_tied_positions=ref.get("tied_positions"))
                if cand["status"] != "defined" or ref["status"] != "defined":
                    rows.append(dict(row, order_spearman=None, scored_group_spearman=None, top1_equal=None,
                                     top5pct_overlap=None, top5pct_size=None, min_deleted_set_jaccard=None,
                                     deleted_set_jaccard=None))
                    continue
                a, b = ref["order"], cand["order"]
                n = len(a)
                top = max(1, math.ceil(0.05 * n))
                jaccard = [[c, len(set(a[:c]) & set(b[:c])) / len(set(a[:c]) | set(b[:c]))] for c in counts if c not in (0, n)]
                rows.append(dict(row, order_spearman=_order_spearman(a, b),
                                 scored_group_spearman=q.spearman(np.asarray(ref["scores"], dtype=float),
                                                                  np.asarray(cand["scores"], dtype=float)),
                                 top1_equal=a[0] == b[0], top5pct_overlap=len(set(a[:top]) & set(b[:top])) / top,
                                 top5pct_size=top, min_deleted_set_jaccard=min((j for _, j in jaccard), default=None),
                                 deleted_set_jaccard=json.dumps(jaccard)))
    return rows


def nonfinite_rows(bank_id, item):
    """Every node with a nonfinite score or gradient, or a failed forward, by arm, modality, target and sign.

    A forward that raised (the revised sampler refuses nonfinite values) leaves no score or
    gradient, so its sign columns are empty and sign_recorded is False; its stage, inner
    sampler step and nonfinite count are kept where the arm recorded them (A0 records the stage only).
    """
    entry = item["entry"]
    base = dict(bank=bank_id, context=entry["context_id"])
    rows = []
    for phase in ("bf16", "fp32"):
        if phase not in item or item[phase].get("status", "evaluated") != "evaluated":
            continue
        for modality, record in item[phase]["modalities"].items():
            for target, cell in record["targets"].items():
                passes = [("A1" if phase == "bf16" else "A2", cell["nested"]["node_records"])]
                if phase == "bf16":
                    passes += [(name, hist["node_records"]) for name, hist in cell["historical"].items()]
                else:
                    passes += [(f"A3{name}", gl["node_records"]) for name, gl in cell.get("gauss_legendre", {}).items()]
                    if cell.get("adaptive"):
                        passes.append(("A4", cell["adaptive"]["node_records"]))
                for arm, records in passes:
                    for record in records:
                        row = dict(base, phase=phase, modality=modality, target=target, arm=arm,
                                   node=record.get("node"), alpha=record.get("alpha"), status=record.get("status"),
                                   stage=record.get("stage"))
                        if record.get("status") == "forward_nonfinite":
                            rows.append(dict(row, forward_failure=True, sign_recorded=False,
                                             inner_step=record.get("inner_step"),
                                             forward_nonfinite_count=record.get("forward_nonfinite_count"),
                                             **{key: None for key in SIGN_FIELDS}))
                            continue
                        counts = record.get("gradient_nonfinite_full") or {}
                        score = record.get("score_nonfinite") or {}
                        if not (sum(counts.get(k, 0) for k in ("nan", "posinf", "neginf"))
                                or sum(score.get(k, 0) for k in ("nan", "posinf", "neginf"))):
                            continue
                        rows.append(dict(row, forward_failure=False, sign_recorded=True, inner_step=None,
                                         forward_nonfinite_count=None,
                                         nan=counts.get("nan", 0), posinf=counts.get("posinf", 0),
                                         neginf=counts.get("neginf", 0), score_nan=score.get("nan", 0),
                                         score_posinf=score.get("posinf", 0), score_neginf=score.get("neginf", 0)))
    return rows


SIGN_FIELDS = ("nan", "posinf", "neginf", "score_nan", "score_posinf", "score_neginf")
NONFINITE_FIELDS = ["bank", "context", "phase", "modality", "target", "arm", "node", "alpha", "status",
                    "forward_failure", "sign_recorded", "stage", "inner_step", "forward_nonfinite_count",
                    *SIGN_FIELDS]


def nonfinite_summary(node_counts, nonfinite):
    """Nodes evaluated, nodes with a failed forward, and nodes with any nonfinite gradient or score,
    zeros included, per bank, phase and arm. Failed forwards carry no sign counts."""
    totals = {}
    for row in node_counts:
        key = (row["bank"], row["phase"], row["modality"], row["target"], row["arm"])
        totals.setdefault(key, dict(zip(("bank", "phase", "modality", "target", "arm"), key), contexts=0, nodes=0,
                                    nodes_forward_failure=0, nodes_nonfinite_gradient=0, nodes_nonfinite_score=0,
                                    **{field: 0 for field in SIGN_FIELDS}))
        totals[key]["contexts"] += 1
        totals[key]["nodes"] += row["nodes"]
    for row in nonfinite:
        entry = totals[(row["bank"], row["phase"], row["modality"], row["target"], row["arm"])]
        if row["forward_failure"]:
            entry["nodes_forward_failure"] += 1
            continue
        entry["nodes_nonfinite_gradient"] += int(bool(row["nan"] or row["posinf"] or row["neginf"]))
        entry["nodes_nonfinite_score"] += int(bool(row["score_nan"] or row["score_posinf"] or row["score_neginf"]))
        for key in SIGN_FIELDS:
            entry[key] += row[key]
    return [totals[key] for key in sorted(totals)]


def fp32_status_rows(bank_id, item):
    """Outcome of the fp32 phase for each fp32 context: evaluated, or the recorded probe failure."""
    if "fp32" not in item:
        return []
    entry = item["entry"]
    record = item["fp32"]
    return [dict(bank=bank_id, context=entry["context_id"], episode=entry["episode"], call=entry["policy_call_idx"],
                 status=record.get("status", "evaluated"), reason=record.get("reason"))]


def node_count_rows(bank_id, item):
    """Denominators for the nonfinite table: evaluated nodes per arm."""
    entry = item["entry"]
    rows = []
    for phase in ("bf16", "fp32"):
        if phase not in item or item[phase].get("status", "evaluated") != "evaluated":
            continue
        for modality, record in item[phase]["modalities"].items():
            for target, cell in record["targets"].items():
                arms = {"A1" if phase == "bf16" else "A2": len(cell["nested"]["node_records"])}
                if phase == "bf16":
                    arms.update({name: len(hist["node_records"]) for name, hist in cell["historical"].items()})
                else:
                    arms.update({f"A3{name}": len(gl["node_records"]) for name, gl in cell.get("gauss_legendre", {}).items()})
                    if cell.get("adaptive"):
                        arms["A4"] = len(cell["adaptive"]["node_records"])
                for arm, count in arms.items():
                    rows.append(dict(bank=bank_id, context=entry["context_id"], phase=phase, modality=modality,
                                     target=target, arm=arm, nodes=count))
    return rows


def automatic_pairs(item, modality, target, comparator="T512"):
    """Amendment 2 pairs: every ranked bf16 and fp32 arm against the bf16 comparator A1 T512 everywhere,
    against the fp32 A2 T512 ranking on fp32 contexts, and against the A4 ranking where A4 converged.
    Pairs whose rankings a context lacks are skipped by auc_change_rows."""
    rankings = item["scoring"]["modalities"][modality]["rankings"]
    candidates = sorted(name for name in rankings if name.startswith((f"IG_{target}_", f"fp32:IG_{target}_")))
    references = [f"IG_{target}_A1{comparator}"]
    if "fp32" in item and item["fp32"].get("status") == "evaluated":
        cell = item["fp32"]["modalities"][modality]["targets"][target]
        references += [f"fp32:IG_{target}_A2{comparator}"] + ([f"fp32:IG_{target}_A4"] if converged_a4(cell) else [])
    return [(candidate, reference) for reference in references for candidate in candidates if candidate != reference]


def auc_change_rows(bank_id, item, pairs=None, comparator="T512"):
    """Normalized-area change of each arm's ranking against a reference ranking, common bf16 scoring.

    With pairs None the amendment-2 families of automatic_pairs are used.
    """
    entry = item["entry"]
    base = dict(bank=bank_id, context=entry["context_id"], episode=entry["episode"], call=entry["policy_call_idx"])
    rows = []
    for modality, result in item["scoring"]["modalities"].items():
        rankings = result["rankings"]
        listed = pairs if pairs is not None else [pair for target in TARGETS
                                                  for pair in automatic_pairs(item, modality, target, comparator)]
        for candidate, reference in listed:
            if candidate not in rankings or reference not in rankings:
                continue
            for direction in ("deletion", "insertion"):
                for response in ("Q", "L2"):
                    values = []
                    for name in (candidate, reference):
                        item_ = rankings[name]
                        values.append(None if item_["status"] != "defined" else
                                      item_["curves"][direction]["responses"][response]["normalized_auc"])
                    change = None if None in values else values[0] - values[1]
                    rows.append(dict(base, modality=modality, candidate=candidate, reference=reference,
                                     direction=direction, response=response, candidate_auc=values[0],
                                     reference_auc=values[1], change=change))
    return rows
