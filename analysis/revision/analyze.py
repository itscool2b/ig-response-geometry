"""Generate a new, strictly reconciled retrospective analysis artifact.

Run from the repository root:
    python -m analysis.revision.analyze --output analysis/revision/results/2026-09-30

The checked-in manifest is immutable input to this command. It is never rebuilt
from arbitrary current files. Existing output directories are refused.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fnmatch
import gzip
import io
import json
import math
from pathlib import Path
import platform
import re
import sys

import numpy as np

from .core import (IntegrityError, bootstrap_summary, episode_key, json_bytes,
                   load_file, normalized_auc, population_id,
                   random_order_counterexample, reconcile, sha256, transform_score)

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("input_manifest.json")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_gzip(path, blob):
    # No timestamp or original filename: clean rebuilds are byte reproducible.
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write(blob)


def csv_bytes(rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def select(records, patterns, event=None, norm_filter="all"):
    if isinstance(patterns, str):
        patterns = [patterns]
    chosen = [r for r in records
              if any(fnmatch.fnmatchcase(Path(r.source).name, p) for p in patterns)
              and (event is None or r.row["event"] == event)]
    if norm_filter == "all":
        return chosen
    if any("ref_norm_maniskill" not in r.row for r in chosen):
        raise IntegrityError("Action norm filter requested on missing norms")
    if norm_filter == "norm_ge15":
        return [r for r in chosen if r.row["ref_norm_maniskill"] >= 15]
    if norm_filter == "norm_lt15":
        return [r for r in chosen if r.row["ref_norm_maniskill"] < 15]
    raise IntegrityError("Unknown population filter")


def target_name(source):
    if "_l2_" in source or source.endswith("_l2.jsonl"):
        return "L2"
    if "_maxdev_" in source or source.endswith("_maxdev.jsonl"):
        return "maxdev"
    return "Q"


def check_stored_arithmetic(records):
    counts = Counter()
    worst = defaultdict(float)
    undefined = []
    for record in records:
        row = record.row
        if row["event"] == "faithfulness":
            for modality in ("vision", "lang"):
                for direction in ("insertion", "deletion"):
                    field = f"{modality}_{direction}_auc"
                    actual = normalized_auc(row["k_grid_auc"], row[f"{modality}_{direction}_curve"],
                                            row[f"{modality}_f_input"], row[f"{modality}_f_baseline"])
                    if actual is None:
                        undefined.append(dict(record=record.id, field=field))
                        continue
                    error = abs(actual - row[field])
                    worst["auc_absolute_error"] = max(worst["auc_absolute_error"], error)
                    if error > 1e-9 * max(1, abs(actual)):
                        raise IntegrityError(f"Stored AUC mismatch: {record.id}:{field}")
                    counts["auc_checks"] += 1
                for k in (1, 5, 10):
                    i = row["k_grid_auc"].index(k)
                    actual = row[f"{modality}_deletion_curve"][i] - row[f"{modality}_f_input"]
                    if abs(actual - row[f"{modality}_dlogp_k{k}"]) > 1e-9 * max(1, abs(actual)):
                        raise IntegrityError(f"Stored deletion delta mismatch: {record.id}")
                    counts["deletion_delta_checks"] += 1
        if row["event"] == "step":
            for modality in ("vision", "lang", "state"):
                gap = row[f"{modality}_gap"]
                actual = abs(row[f"{modality}_ig_sum"] - gap) / (abs(gap) + 1e-12)
                error = abs(actual - row[f"{modality}_err"])
                worst["completeness_absolute_error"] = max(worst["completeness_absolute_error"], error)
                if error > max(1e-6, 1e-5 * abs(actual)):
                    raise IntegrityError(f"Stored completeness mismatch: {record.id}:{modality}")
                counts["completeness_checks"] += 1
    return dict(counts=counts, maximum_discrepancies=dict(worst), undefined_auc=undefined)


class Registry:
    def __init__(self, draws, seed):
        self.draws, self.seed = draws, seed
        self.results, self.populations, self.diagnostics = [], {}, []

    def add(self, result_id, records, field, units, *, values=None, statistic="median",
            population="all", view="retrospective_last", limitation="",
            conditional_shared_context=False, episode_equal=False):
        if not records:
            raise IntegrityError(f"Empty requested result population: {result_id}")
        if values is None:
            if any(field not in r.row for r in records):
                raise IntegrityError(f"Missing field in {result_id}")
            values = [r.row[field] for r in records]
        if len(values) != len(records):
            raise IntegrityError(f"Value/population mismatch in {result_id}")
        digest = population_id(records)
        self.populations.setdefault(digest, [r.id for r in sorted(records, key=lambda r: r.id)])
        groups = [episode_key(r, conditional_shared_context=conditional_shared_context) for r in records]
        if episode_equal:
            pooled = defaultdict(list)
            for key, value in zip(groups, values):
                pooled[key].append(value)
            groups = sorted(pooled)
            values = [float(np.median(pooled[key])) for key in groups]
        summary = bootstrap_summary(values, groups, statistic=statistic,
                                    draws=self.draws, seed=self.seed)
        result = dict(result_id=result_id, field=field, units=units,
                      population_filter=population, occurrence_view=view,
                      estimand="median_of_episode_medians" if episode_equal else
                      ("call_weighted_median" if statistic == "median" else "call_weighted_mean"),
                      point_population_sha256=digest, ci_population_sha256=digest,
                      n_source_records=len(records), **summary,
                      bootstrap_draws=self.draws, bootstrap_seed=self.seed,
                      cluster_key="task,model,episode (conditional shared-context sensitivity)"
                      if conditional_shared_context else "task,model,recorded_seed,episode",
                      checkpoint_identity="unknown", context_identity="unknown",
                      interpretation="retrospective_descriptive",
                      limitations=limitation or "Intervals condition on the released episode mixture and one checkpoint; historical numerical/provenance limitations remain.")
        self.results.append(result)
        return result


def add_saved_summaries(registry, records, view):
    # Every population is named explicitly. All-valid and historical norm-filter
    # summaries coexist; action norm is not called a denominator validity test.
    for task in ("PickCube", "StackCube", "PickSingleYCB", "PegInsertionSide"):
        for pop in ("all", "norm_ge15", "norm_lt15"):
            steps = select(records, [f"m7_metrics_{task}-v1_170m_seed{s}.jsonl" for s in (42, 142)], "step", pop)
            for modality in ("vision", "lang", "state"):
                for suffix in ("err", "gap"):
                    field = f"{modality}_{suffix}"
                    registry.add(f"verification/{task}/{pop}/{field}", steps, field,
                                 "relative_error" if suffix == "err" else "Q_score",
                                 values=[abs(r.row[field]) for r in steps], population=pop, view=view)
                registry.add(f"verification/{task}/{pop}/{modality}_pct_le3", steps,
                             f"{modality}_err", "percent", statistic="mean",
                             values=[100.0 * (r.row[f"{modality}_err"] <= 0.03) for r in steps],
                             population=pop, view=view)
            faith = select(records, [f"m7_metrics_faithfulness_{task}-v1_170m_seed{s}.jsonl" for s in (42, 142)], "faithfulness", pop)
            for modality in ("vision", "lang"):
                for direction in ("insertion", "deletion"):
                    field = f"{modality}_{direction}_auc"
                    registry.add(f"verification/{task}/{pop}/{field}", faith, field, "dimensionless", population=pop, view=view)
                registry.add(f"verification/{task}/{pop}/{modality}_dQ_k5", faith,
                             f"{modality}_dlogp_k5", "Q_score", population=pop, view=view)
                gaps = [abs(r.row[f"{modality}_f_input"] - r.row[f"{modality}_f_baseline"]) for r in faith]
                registry.diagnostics.append(dict(
                    family="score_denominators", task=task, population=pop, modality=modality, view=view,
                    rows=len(faith), minimum=min(gaps), q01=float(np.quantile(gaps, .01)),
                    median=float(np.median(gaps)), zero_count=sum(g == 0 for g in gaps),
                    below_historical_1e9=sum(g < 1e-9 for g in gaps),
                    insertion_negative=sum(r.row[f"{modality}_insertion_auc"] < 0 for r in faith),
                    deletion_negative=sum(r.row[f"{modality}_deletion_auc"] < 0 for r in faith),
                    insertion_min=min(r.row[f"{modality}_insertion_auc"] for r in faith),
                    deletion_min=min(r.row[f"{modality}_deletion_auc"] for r in faith)))
            registry.diagnostics.append(dict(
                family="verification_population", task=task, population=pop, view=view,
                step_rows=len(steps), faith_rows=len(faith),
                norm_median=float(np.median([r.row["ref_norm_maniskill"] for r in steps]))))
        for phase in ("C1", "C2"):
            rows = select(records, f"m7_metrics_sanity_{phase}_{task}-v1_170m_seed*.jsonl")
            for field in ("spearman_vision", "spearman_language", "spearman_state"):
                registry.add(f"verification/{task}/{phase}/{field}", rows, field, "Spearman_rho", view=view,
                             limitation="Input-shuffle sensitivity is not a guarantee of learned-model grounding; m16 versus original m64 is a confound.")
    oneb = select(records, "metrics_faithfulness_metrics_PickCube-v1_1b_seed*_*.jsonl", "faithfulness")
    for modality in ("vision", "lang"):
        for direction in ("insertion", "deletion"):
            field = f"{modality}_{direction}_auc"
            for episode_equal in (False, True):
                estimand = "episode_equal" if episode_equal else "call_weighted"
                registry.add(f"oneb/{estimand}/{field}", oneb, field, "dimensionless", view=view, episode_equal=episode_equal,
                             limitation="Mixed historical language baselines; unstratified empirical mixture across evaluation seeds, not training replication.")
    for kind in ("frozen", "cascade"):
        rows = select(records, f"metrics_sanity_C1_{kind}_*.jsonl")
        for field in ("spearman_vision", "spearman_language", "spearman_state"):
            for shared in (False, True):
                label = "conditional_shared_context" if shared else "recorded_episodes"
                registry.add(f"sanity/{kind}/{label}/{field}", rows, field, "Spearman_rho", view=view,
                             conditional_shared_context=shared,
                             limitation="Observation reuse is unresolved. Shared task-episode grouping is a conditional dependence sensitivity, not authenticated pairing.")
    for task in ("PickCube", "StackCube"):
        rows = select(records, f"metrics_baseline_sensitivity_metrics_{task}-v1_*.jsonl")
        for field in ("rho_black_gray", "rho_black_blur", "rho_gray_blur"):
            registry.add(f"baseline/{task}/{field}", rows, field, "Spearman_rho", view=view,
                         limitation="Only two episode clusters. Percentile bounds do not establish calibrated 95% coverage.")
    for task in ("PegInsertionSide", "PickSingleYCB"):
        for seed in (42, 142):
            rows = select(records, f"metrics_{task}-v1_170m_seed{seed}_m128.jsonl", "step", "norm_ge15")
            registry.add(f"m128/{task}/seed{seed}/vision_pct_le3", rows, "vision_err", "percent", view=view,
                         values=[100.0 * (r.row["vision_err"] <= 0.03) for r in rows], statistic="mean", population="norm_ge15",
                         limitation="Retrospective m128 population; earlier m64 observations are not authenticated matched contexts.")
    for target, suffix in (("Q_regeneration", ""), ("Q_later", "_logpi"), ("L2", "_l2"), ("maxdev", "_maxdev")):
        for pop in ("all", "norm_ge15"):
            rows = select(records, [f"metrics_PickCube-v1_170m_seed{s}{suffix}.jsonl" for s in (42, 142)], "step", pop)
            registry.add(f"targets/{target}/{pop}/vision_gap", rows, "vision_gap", "Q_score" if target.startswith("Q") else "action_coordinate" if target == "maxdev" else "action_L2", values=[abs(r.row["vision_gap"]) for r in rows], view=view, population=pop)
            registry.add(f"targets/{target}/{pop}/vision_pct_le3", rows, "vision_err", "percent", values=[100.0 * (r.row["vision_err"] <= 0.03) for r in rows], statistic="mean", view=view, population=pop)
    for target, suffix in (("Q_later", "logpi"), ("L2", "l2"), ("maxdev", "maxdev")):
        rows = select(records, f"metrics_faithfulness_metrics_PickCube-v1_170m_seed*_{suffix}_*.jsonl", "faithfulness")
        for field in ("vision_insertion_auc", "vision_deletion_auc", "vision_dlogp_k5"):
            units = "dimensionless" if field.endswith("auc") else {"Q_later": "Q_score", "L2": "action_L2", "maxdev": "action_coordinate"}[target]
            registry.add(f"targets/{target}/native/{field}", rows, field, units, view=view,
                         limitation="Target-dependent response units/geometry. Cross-target observation/runtime identity is unresolved; these are unpaired descriptive summaries.")


def add_transformed_curves(registry, records, view):
    for ranking, suffix in (("Q", "logpi"), ("L2", "l2")):
        rows = select(records, f"metrics_faithfulness_metrics_PickCube-v1_170m_seed*_{suffix}_*.jsonl", "faithfulness")
        for response in ("Q", "L2"):
            for modality in ("vision", "lang"):
                for direction in ("insertion", "deletion"):
                    selected, values, excluded, denominators = [], [], [], []
                    for r in rows:
                        row = r.row
                        convert = lambda v: transform_score(v, ranking, response)
                        baseline = convert(row[f"{modality}_f_baseline"])
                        input_value = convert(row[f"{modality}_f_input"])
                        denominator = input_value - baseline
                        value = normalized_auc(row["k_grid_auc"], [convert(v) for v in row[f"{modality}_{direction}_curve"]], input_value, baseline)
                        if value is None:
                            excluded.append(r.id)
                        else:
                            selected.append(r)
                            values.append(value)
                            denominators.append(abs(denominator))
                    result_id = f"rescore/ranking_{ranking}/response_{response}/{modality}_{direction}"
                    registry.add(result_id, selected, f"{modality}_{direction}_auc", "dimensionless",
                                 values=values, view=view,
                                 limitation="The recorded ranking and curve samples are held fixed. Response transformation changes AUC; it does not demonstrate a better ranking. Cross-arm context matching remains unknown.")
                    registry.diagnostics.append(dict(
                        family="rescore", result_id=result_id, view=view, undefined_records=excluded,
                        denominator_min=min(denominators), denominator_median=float(np.median(denominators)),
                        denominator_below_legacy_1e9=sum(d < 1e-9 for d in denominators),
                        auc_min=min(values), auc_max=max(values)))


def reconciliation_diagnostics(last):
    by_source = defaultdict(list)
    for r in last:
        by_source[r.source].append(r)
    joins = []
    for source, rows in sorted(by_source.items()):
        if rows[0].row["event"] != "faithfulness":
            continue
        name = Path(source).name
        if name.startswith(("m5_", "m7_")):
            step_name = name.replace("metrics_faithfulness_", "metrics_", 1)
        else:
            step_name = re.sub(r"_\d{8}_\d{6}\.jsonl$", ".jsonl", name.replace("metrics_faithfulness_", "", 1))
        step_rows = by_source.get("data/" + step_name, [])
        index = {(r.row["task"], r.row["model"], r.row["seed"], r.row["episode"], r.row["policy_call_idx"]): r for r in step_rows if r.row["event"] == "step"}
        missing, norm_differences, fingerprint = [], [], []
        for r in rows:
            key = tuple(r.row[k] for k in ("task", "model", "seed", "episode", "policy_call_idx"))
            step = index.get(key)
            if step is None:
                missing.append(r.id)
                continue
            if r.row["ref_norm_maniskill"] != step.row["ref_norm_maniskill"]:
                norm_differences.append(r.id)
            for modality in ("vision", "lang"):
                a, b = -step.row[f"{modality}_gap"], r.row[f"{modality}_f_baseline"]
                if abs(a - b) > 1e-6 * max(abs(a), abs(b), 1e-12):
                    fingerprint.append(dict(record=r.id, modality=modality, relative_difference=abs(a-b)/max(abs(a),abs(b),1e-12)))
        joins.append(dict(faithfulness=source, steps="data/" + step_name, n_rows=len(rows),
                          missing_keys=missing, norm_differences=norm_differences,
                          scalar_fingerprint_disagreements=fingerprint,
                          interpretation="Legacy-key consistency check only. Matching norms/gaps do not authenticate tensors or contexts."))
    cross_target = []
    for seed in (42, 142):
        def indexed(suffix):
            return {(r.row["episode"], r.row["policy_call_idx"]): r
                    for r in by_source[f"data/metrics_PickCube-v1_170m_seed{seed}{suffix}.jsonl"] if r.row["event"] == "step"}
        base = indexed("")
        for target, suffix in (("Q_later", "_logpi"), ("L2", "_l2"), ("maxdev", "_maxdev")):
            other = indexed(suffix)
            common = sorted(base.keys() & other.keys())
            diffs = [abs(base[k].row["ref_norm_maniskill"] - other[k].row["ref_norm_maniskill"]) for k in common]
            cross_target.append(dict(seed=seed, comparison=f"Q_regeneration_vs_{target}", nominal_key_overlap=len(common),
                                     exact_reference_norm_matches=sum(d == 0 for d in diffs),
                                     median_abs_difference=float(np.median(diffs)), maximum_abs_difference=max(diffs),
                                     norm_filter_membership_changes=sum((base[k].row["ref_norm_maniskill"] >= 15) != (other[k].row["ref_norm_maniskill"] >= 15) for k in common),
                                     context_identity="unknown", paired_effect_permitted=False))
    return dict(legacy_key_joins=joins, nominal_cross_target_diagnostics=cross_target)


def duplicate_population_sensitivity(raw, last, clean):
    report = []
    for label, patterns, event, population, field in [
        ("L2_completeness", [f"metrics_PickCube-v1_170m_seed{s}_l2.jsonl" for s in (42,142)], "step", "norm_ge15", "vision_err"),
        ("maxdev_completeness", [f"metrics_PickCube-v1_170m_seed{s}_maxdev.jsonl" for s in (42,142)], "step", "norm_ge15", "vision_err"),
        ("L2_faithfulness", "metrics_faithfulness_metrics_PickCube-v1_170m_seed*_l2_*.jsonl", "faithfulness", "all", "vision_deletion_auc"),
        ("maxdev_faithfulness", "metrics_faithfulness_metrics_PickCube-v1_170m_seed*_maxdev_*.jsonl", "faithfulness", "all", "vision_deletion_auc"),
        ("PegInsertionSide_m128_seed142", "metrics_PegInsertionSide-v1_170m_seed142_m128.jsonl", "step", "norm_ge15", "vision_err"),
    ]:
        for view, source_rows in (("raw_inclusive", raw), ("retrospective_last", last), ("conflict_excluded", clean)):
            selected = select(source_rows, patterns, event, population)
            report.append(dict(family=label, occurrence_view=view, population_filter=population,
                               n_rows=len(selected), population_sha256=population_id(selected),
                               field=field, median=float(np.median([r.row[field] for r in selected])),
                               pct_le3=float(np.mean([r.row[field] <= .03 for r in selected])*100) if field == "vision_err" else None,
                               interpretation="Historical replay sensitivity only. Raw-inclusive rows do not represent independent additional policy decisions; no CI is assigned to this view."))
    return report


def displacement_endpoints(records, draws, seed):
    results = []
    for task in ("PickCube", "StackCube"):
        for model in ("170m", "1b"):
            rows = select(records, [f"m5_metrics_displacement_{task}-v1_{model}.jsonl",
                                    f"metrics_displacement_metrics_{task}-v1_{model}_seed*_*.jsonl"], "displacement")
            # Use the historical paper's k=5 and both released evaluation seeds.
            # Source identity keeps distinct files from silently coalescing.
            index = {(r.source, r.row["seed"], r.row["episode"], r.row["policy_call_idx"], r.row["solver_steps"]): r for r in rows if r.row["modality"] == "vision"}
            keys2 = {k[:4] for k in index if k[4] == 2}
            keys20 = {k[:4] for k in index if k[4] == 20}
            if keys2 != keys20:
                raise IntegrityError("Incomplete nominal solver endpoint pairing")
            pairs = [(index[k + (2,)], index[k + (20,)]) for k in sorted(keys2)]
            group_map = defaultdict(list)
            for i, (a, _) in enumerate(pairs):
                group_map[episode_key(a)].append(i)
            groups = list(group_map.values())
            a_values = np.array([a.row["rel_active"][a.row["del_grid"].index(5)] for a, _ in pairs])
            b_values = np.array([b.row["rel_active"][b.row["del_grid"].index(5)] for _, b in pairs])
            if np.median(a_values) <= 0:
                raise IntegrityError("Nonpositive solver ratio denominator")
            ratio = float(np.median(b_values) / np.median(a_values))
            rng = np.random.default_rng(seed)
            bootstrap = []
            for _ in range(draws):
                selected = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                denominator = np.median(a_values[selected])
                if denominator <= 0:
                    raise IntegrityError("Nonpositive bootstrap solver ratio denominator")
                bootstrap.append(float(np.median(b_values[selected]) / denominator))
            lo, hi = np.percentile(bootstrap, [2.5, 97.5])
            members = [r for pair in pairs for r in pair]
            results.append(dict(task=task, model=model, modality="vision", deletion_percent=5,
                                contrast="relative_displacement_median_T20_over_T2", point_ratio=ratio,
                                ci_lo_ratio=float(lo), ci_hi_ratio=float(hi), point_change_percent=100*(ratio-1),
                                ci_lo_change_percent=100*(float(lo)-1), ci_hi_change_percent=100*(float(hi)-1),
                                n_nominal_pairs=len(pairs), n_episode_groups=len(groups),
                                point_population_sha256=population_id(members), ci_population_sha256=population_id(members),
                                reference_norm_changes=sum(a.row["ref_action_norm_active"] != b.row["ref_action_norm_active"] for a,b in pairs),
                                bootstrap_draws=draws, bootstrap_seed=seed,
                                interpretation="Conditional paired legacy-key sensitivity within one solver file. Context/noise identities are not authenticated; this does not identify denoiser causality.",
                                records=[r.id for r in members]))
    return results


def run(repo, output, draws, seed):
    if output.exists():
        raise IntegrityError(f"Output path already exists; use a new directory: {output}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_paths = {entry["path"] for entry in manifest["files"]}
    actual_paths = {p.relative_to(repo).as_posix() for p in (repo / "data").glob("*.jsonl")}
    if actual_paths != expected_paths:
        raise IntegrityError(f"Input file inventory changed: missing={sorted(expected_paths-actual_paths)}, additional={sorted(actual_paths-expected_paths)}")
    raw, ledger = [], []
    for entry in manifest["files"]:
        records, lines = load_file(repo / entry["path"], repo, entry)
        raw.extend(records)
        ledger.extend(lines)
    last, clean, duplicates = reconcile(raw)
    arithmetic = check_stored_arithmetic(raw)
    registry = Registry(draws, seed)
    add_saved_summaries(registry, last, "retrospective_last")
    add_transformed_curves(registry, last, "retrospective_last")
    # Only report additional sensitivity rows where their selected population
    # changes. This keeps all unchanged source families from being duplicated.
    sensitivity = Registry(draws, seed)
    add_saved_summaries(sensitivity, clean, "conflict_excluded")
    add_transformed_curves(sensitivity, clean, "conflict_excluded")
    reference = {r["result_id"]: r for r in registry.results}
    changed = [r for r in sensitivity.results if r["point_population_sha256"] != reference[r["result_id"]]["point_population_sha256"]]
    registry.results.extend(changed)
    changed_hashes = {r["point_population_sha256"] for r in changed}
    registry.populations.update({k:v for k,v in sensitivity.populations.items() if k in changed_hashes})
    duplicate_lookup = {d["key_sha256"]: d for d in duplicates}
    for line in ledger:
        duplicate = duplicate_lookup.get(line.get("key_sha256"))
        if duplicate:
            line["duplicate_classification"] = duplicate["classification"]
            line["last_occurrence_selected"] = int(duplicate["retrospective_last_selected"].startswith(f"{line['source']}:{line['line']}:"))
            line["conflict_excluded_selected"] = int(duplicate["conflict_excluded_selected"] is not None and line["last_occurrence_selected"])
        elif line["status"] == "valid":
            line["duplicate_classification"] = "unique"
            line["last_occurrence_selected"] = line["conflict_excluded_selected"] = 1
    reconciliation = reconciliation_diagnostics(last)
    endpoints = displacement_endpoints(last, draws, seed)
    counterexample = random_order_counterexample()
    summary = dict(
        status="retrospective_reanalysis_not_submission_readiness",
        inputs=len(manifest["files"]), physical_lines=len(ledger), valid_records=len(raw),
        quarantined_lines=sum(x["status"] == "quarantined" for x in ledger),
        duplicate_extra_occurrences=len(raw)-len(last), duplicate_groups=len(duplicates),
        duplicate_classes=dict(Counter(d["classification"] for d in duplicates)),
        conflicting_groups=sum(d["classification"] == "conflicting_payload" for d in duplicates),
        retrospective_last_records=len(last), conflict_excluded_records=len(clean),
        results=len(registry.results), changed_population_sensitivity_results=len(changed),
        strict_arithmetic=arithmetic,
        limitations=[
            "Raw file/line identities are authenticated to this preserved snapshot. Model checkpoint, raw observation, reference-action tensor and noise identities remain unknown.",
            "Last occurrence is a declared retrospective selection convention, not evidence that its conflicting attempt is scientifically correct. All conflicting groups and a conflict-excluded sensitivity are retained.",
            "Cross-target reference norms disagree; no paired ranking improvement claim is supported.",
            "One known corrupt line is excluded only by exact source and physical-line hashes; no other parse error is skipped.",
            "The norm>=15 population is a historical sensitivity, not a small-denominator validity criterion.",
            "No empirical RDT random-ranking control, model rerun, training replication, or missing historical artifact recovery occurs here.",
            "Episode percentile intervals are conditional descriptive estimates, and do not establish calibration with few clusters or unknown shared contexts."])
    output.mkdir(parents=True)
    write_gzip(output / "raw_line_ledger.csv.gz", csv_bytes(ledger))
    write_gzip(output / "population_membership.json.gz", json_bytes(registry.populations) + b"\n")
    write_json(output / "duplicates.json", duplicates)
    write_json(output / "reconciliation.json", reconciliation)
    write_json(output / "duplicate_population_sensitivity.json", duplicate_population_sensitivity(raw, last, clean))
    write_json(output / "diagnostics.json", registry.diagnostics)
    write_json(output / "solver_endpoint_sensitivity.json", endpoints)
    write_json(output / "random_order_counterexample.json", counterexample)
    write_json(output / "summary.json", summary)
    (output / "results.csv").write_bytes(csv_bytes(registry.results))
    script_hashes = {p.name: sha256(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    artifacts = {p.name: sha256(p.read_bytes()) for p in sorted(output.iterdir())}
    write_json(output / "provenance.json", dict(
        schema_version=1, analysis="saved-data-revision-2026-09-30",
        manifest_sha256=sha256(MANIFEST.read_bytes()), code_sha256=script_hashes,
        inputs=manifest["files"], artifacts_sha256=artifacts,
        command=f"python -m analysis.revision.analyze --output <fresh-directory> --draws {draws} --seed {seed}",
        python=platform.python_version(), numpy=np.__version__,
        checkout_sha256={e["path"]: sha256((repo/e["path"]).read_bytes()) for e in manifest["files"]},
        checkout_policy="Accept exact canonical Git blob or predeclared exact CRLF expansion only; ledger/record identities use canonical original bytes. No raw file is modified.",
        rng="numpy.random.default_rng PCG64", percentile_method="linear 2.5/97.5",
        bootstrap_design="Unstratified empirical episode mixture; whole episode count resampling. Each row binds point and CI to one population hash.",
        analysis_status="retrospective; no preregistration claim"))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.draws < 2:
        parser.error("draws must be at least 2")
    try:
        result = run(args.repo.resolve(), args.output.resolve(), args.draws, args.seed)
    except IntegrityError as exc:
        parser.exit(1, f"Integrity failure: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
