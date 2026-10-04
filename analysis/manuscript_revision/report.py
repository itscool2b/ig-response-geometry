"""Versioned, offline reporting layer. No model execution or evidence edits.

The original paired protocol named the mean primary. This retrospective report
leads with the paired median and positive-change share for interpretation of
heavy-tailed ratio data. It preserves the original means, intervals and sources.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

from analysis.revision.core import IntegrityError, load_file, reconcile
from analysis.revision.verify import verify as verify_canonical
from analysis.paired_rescoring.influence import verify as verify_influence

ROOT = Path(__file__).resolve().parents[2]
VERSION = "2026-10-03-v2"
CANONICAL = Path("analysis/revision/results/2026-09-30-v2")
PAIRED = Path("analysis/paired_rescoring/results/2026-10-01-v1")
INFLUENCE = Path("analysis/paired_rescoring/influence_results/2026-10-01-v1")
NUMERICAL = Path("analysis/numerical_case/2026-10-01-v1")
OUTPUT = Path("analysis/manuscript_revision/results") / VERSION
CASES = tuple(f"{rank}:{modality}:{direction}" for rank in ("Q", "L2")
              for modality in ("vision", "lang") for direction in ("insertion", "deletion"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def csv_text(rows):
    stream = io.StringIO(newline="")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()})
    return stream.getvalue()


def _same(actual, expected, label):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-10, rel_tol=1e-10):
        raise IntegrityError(f"Reporting cross-check failed: {label}: {actual} != {expected}")


def _quantile(values, probability):
    """Linear interpolation on the observed range, including singleton groups."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _unique(rows, key, label):
    indexed = {}
    for row in rows:
        identity = key(row)
        if identity in indexed:
            raise IntegrityError(f"Duplicate {label}: {identity}")
        indexed[identity] = row
    return indexed


def authenticate(root=ROOT):
    """Authenticate frozen outputs, producers, protocols and every raw input."""
    root = Path(root)
    canonical = verify_canonical(root / CANONICAL)
    influence = verify_influence(root / INFLUENCE, root=root)
    paths = set()
    provenance = read_json(root / CANONICAL / "provenance.json")
    for name, expected in provenance["code_sha256"].items():
        path = Path("analysis/revision") / name
        if sha(root / path) != expected:
            raise IntegrityError(f"Frozen canonical producer changed: {path}")
        paths.add(path)
    manifest_path = Path("analysis/revision/input_manifest.json")
    if sha(root / manifest_path) != provenance["manifest_sha256"]:
        raise IntegrityError("Historical input manifest changed")
    paths.add(manifest_path)
    for entry in read_json(root / manifest_path)["files"]:
        path = Path(entry["path"])
        if sha(root / path) not in {entry["sha256"], entry.get("git_crlf_checkout_sha256")}:
            raise IntegrityError(f"Historical raw input changed: {path}")
        paths.add(path)
    for folder in (CANONICAL, PAIRED, INFLUENCE):
        p = read_json(root / folder / "provenance.json")
        paths.add(folder / "provenance.json")
        for name, expected in p["artifacts_sha256"].items():
            path = folder / name
            if sha(root / path) != expected:
                raise IntegrityError(f"Frozen result changed: {path}")
            paths.add(path)
    for folder, protocol in ((Path("analysis/paired_rescoring"), "protocol.json"),
                             (Path("analysis/paired_rescoring"), "influence_protocol.json")):
        paths.add(folder / protocol)
    for path in (root / "analysis/paired_rescoring").glob("*.py"):
        paths.add(path.relative_to(root))
    for path in (root / "analysis/paired_rescoring/influence").glob("*.py"):
        paths.add(path.relative_to(root))
    manifest = read_json(root / NUMERICAL / "outputs/manifest.json")
    for suffix, expected in [("inputs/sealed_v6.json", manifest["input_sha256"]),
                             ("summarize.py", manifest["source_sha256"]),
                             *(("outputs/" + name, digest) for name, digest in manifest["outputs"].items())]:
        path = NUMERICAL / suffix
        if sha(root / path) != expected:
            raise IntegrityError(f"Frozen numerical evidence changed: {path}")
        paths.add(path)
    paths.add(NUMERICAL / "outputs/manifest.json")
    alias = read_json(root / "analysis/revision/examples/nested_grid_aliasing.json")
    for name, expected in alias["source_sha256"].items():
        if sha(root / name) != expected:
            raise IntegrityError(f"Aliasing source changed: {name}")
        paths.add(Path(name))
    paths.update(map(Path, ("analysis/revision/examples/nested_grid_aliasing.json",
                           "analysis/revision/response_geometry_theory.py")))
    return {"canonical": canonical, "influence": influence,
            "numerical_outputs": len(manifest["outputs"])}, {
                path.as_posix(): sha(root / path) for path in sorted(paths)}


def summarize_case(case, calls, summary, geometry, omission):
    """Cross-check every aggregate against individual calls, never truncate tails."""
    if not calls or any(call["case"] != case for call in calls):
        raise IntegrityError("Empty or cross-cohort reporting case")
    if len({call["record_id"] for call in calls}) != len(calls):
        raise IntegrityError("A reporting case repeats a physical record")
    defined = [call for call in calls if call["paired_status"] == "defined"]
    if not defined:
        raise IntegrityError("No defined paired effects in reporting case")
    values = [float(call["delta"]) for call in defined]
    if not all(math.isfinite(value) for value in values):
        raise IntegrityError("Nonfinite paired reporting value")
    episodes = defaultdict(list)
    for call in defined:
        episodes[call["episode_id"]].append(call)
    mean = statistics.mean(statistics.mean(float(c["delta"]) for c in group) for group in episodes.values())
    contributions = {name: statistics.mean(statistics.mean(float(c["contribution_" + name]) for c in group)
                     for group in episodes.values())
                     for name in ("in_range", "high_distance_overshoot", "low_distance_undershoot")}
    _same(mean, summary["equal_episode_mean_point"], case + ":mean")
    _same(statistics.median(values), summary["paired_call_median_point"], case + ":median")
    _same(sum(contributions.values()), mean, case + ":decomposition")
    for name, value in contributions.items():
        _same(value, summary["equal_episode_contribution_" + name], case + ":" + name)
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    zero = sum(value == 0 for value in values)
    overshoot = sum(int(call["n_high_distance_overshoot"]) > 0 for call in calls)
    undershoot = sum(int(call["n_low_distance_undershoot"]) > 0 for call in calls)
    for actual, expected, label in ((len(calls), summary["n_raw_calls"], "raw calls"),
                                   (len(defined), summary["n_defined_calls"], "defined calls"),
                                   (len(episodes), summary["n_defined_episodes"], "episodes"),
                                   (overshoot, geometry["calls_any_high_distance_overshoot"], "overshoots"),
                                   (negative, summary["n_negative_delta"], "negative calls"),
                                   (zero, summary["n_zero_delta"], "zero calls")):
        _same(actual, expected, case + ":" + label)
    if summary["point_membership_sha256"] != summary["ci_membership_sha256"]:
        raise IntegrityError("Paired point and interval populations differ")
    in_range = [call for call in defined if int(call["n_high_distance_overshoot"]) == 0
                and int(call["n_low_distance_undershoot"]) == 0]
    overshooting = [call for call in defined if int(call["n_high_distance_overshoot"]) > 0]
    in_range_values = [float(call["delta"]) for call in in_range]
    in_range_positive = sum(value > 0 for value in in_range_values)
    overshoot_positive = sum(float(call["delta"]) > 0 for call in overshooting)
    rank, modality, direction = case.split(":")
    return dict(case=case, ranking=rank, modality=modality, direction=direction,
        n_raw_calls=len(calls), n_defined_calls=len(defined), n_undefined_calls=len(calls)-len(defined),
        n_episodes=len(episodes), calls_per_episode_min=min(map(len, episodes.values())),
        calls_per_episode_max=max(map(len, episodes.values())),
        median=statistics.median(values), median_ci_low=float(summary["paired_call_median_ci_lo"]),
        median_ci_high=float(summary["paired_call_median_ci_hi"]), positive_calls=positive,
        positive_percent=100*positive/len(defined), negative_calls=negative, zero_calls=zero,
        overshoot_calls=overshoot, overshoot_percent=100*overshoot/len(calls), undershoot_calls=undershoot,
        fully_in_range_calls=len(in_range), fully_in_range_percent=100*len(in_range)/len(defined),
        fully_in_range_positive_calls=in_range_positive,
        fully_in_range_positive_percent=100*in_range_positive/len(in_range) if in_range else None,
        fully_in_range_delta_q25=_quantile(in_range_values, .25),
        fully_in_range_delta_q75=_quantile(in_range_values, .75),
        overshoot_defined_calls=len(overshooting), overshoot_positive_calls=overshoot_positive,
        overshoot_positive_percent=100*overshoot_positive/len(overshooting) if overshooting else None,
        geometry_share_denominator="defined calls; positive fractions condition on their named geometry subgroup",
        mean=mean, mean_ci_low=float(summary["equal_episode_mean_ci_lo"]),
        mean_ci_high=float(summary["equal_episode_mean_ci_hi"]),
        in_range_contribution=contributions["in_range"], overshoot_contribution=contributions["high_distance_overshoot"],
        undershoot_contribution=contributions["low_distance_undershoot"],
        minimum=min(values), maximum=max(values),
        omission_sign_preserved=int(omission["n_sign_preserved"]), omissions=int(omission["n_omissions"]),
        omission_mean_min=float(omission["omission_mean_min"]), omission_mean_max=float(omission["omission_mean_max"]),
        membership_sha256=summary["point_membership_sha256"],
        mean_weighting="equal recorded episode", median_weighting="equal defined call",
        units="dimensionless AUC_Q minus AUC_N (native norm label L2)", bootstrap_draws=int(summary["bootstrap_draws"]),
        bootstrap_seed=int(summary["bootstrap_seed"]), interval="descriptive whole-episode percentile")


def success_counts(root):
    manifest = read_json(root / "analysis/revision/input_manifest.json")
    expressions = {
        "Q-ranking regeneration": r"data/metrics_PickCube-v1_170m_seed(?:42|142)_logpi\.jsonl$",
        "L-ranking regeneration": r"data/metrics_PickCube-v1_170m_seed(?:42|142)_l2\.jsonl$",
        "Verification": r"data/m7_metrics_(?:PickCube|StackCube|PickSingleYCB|PegInsertionSide)-v1_170m_seed(?:42|142)\.jsonl$",
        "Historical 1B": r"data/metrics_PickCube-v1_1b_seed(?:42|142|242)\.jsonl$",
    }
    out = []
    for source, pattern in expressions.items():
        members = []
        for entry in manifest["files"]:
            if re.fullmatch(pattern, entry["path"]):
                records, _ = load_file(root / entry["path"], root, entry)
                selected, _, _ = reconcile(records)
                endings = [record for record in selected if record.row["event"] == "episode_end"]
                if not endings or any(type(record.row.get("success")) is not bool for record in endings):
                    raise IntegrityError("Missing or non-boolean task-success records")
                members.extend(endings)
                out.append(dict(source=source, file=entry["path"], episodes=len(endings),
                                successes=sum(record.row["success"] for record in endings),
                                record_ids=[record.id for record in endings]))
        if not members:
            raise IntegrityError("Missing source for task success: " + source)
    return out


def numerical_rows(projection):
    """Keep IG and mean-path-gradient checks distinct, with raw denominators."""
    rows = []
    for cell in projection["rows"]:
        if cell["status"] != "complete_diagnostics":
            continue
        observations = {obs["m"]: obs for obs in cell["observations"] if obs["repeat"] == 0}
        for check in cell["checks"]:
            if check["criterion"] != "coordinate_relative_l1":
                continue
            reference = observations[check["reference_m"]]
            rows.append(dict(stratum=cell["stratum"], episode=cell["episode"], call=cell["call"],
                context_id=cell["context_id"], modality=cell["modality"], target=cell["target"],
                ranking=check["ranking"], candidate_m=check["candidate_m"], reference_m=check["reference_m"],
                coordinate_relative_l1=check["value"], coordinate_absolute_l1=check["absolute_l1_difference"],
                reference_coordinate_l1=check.get("reference_l1_inferred_from_reported_ratio"),
                coordinate_threshold=check["threshold"], coordinate_status=check["status"],
                reference_completeness=reference["relative_residual"], reference_gap=reference["expected_gap"],
                reference_sum=reference["ig_sum"], reference_absolute_residual=reference["absolute_residual"],
                report_sha256=cell["report_sha256"]))
    return rows


def summarize_call_indices(case, calls):
    """Retrospective first/later split, retaining all calls and episode weights.

    Group means and medians weight defined calls equally. Contributions use the
    full case's equal-episode mean weights, so the two contributions sum to that
    mean even if episode lengths differ. Undefined calls remain in the counts.
    """
    if not calls or any(row["case"] != case for row in calls):
        raise IntegrityError("Empty or cross-cohort call-index summary")
    if len({row["record_id"] for row in calls}) != len(calls):
        raise IntegrityError("Duplicate call in call-index summary")
    indices = {}
    episodes = defaultdict(list)
    for row in calls:
        raw_index = row["policy_call_idx"]
        try:
            index = int(raw_index)
        except (ValueError, TypeError) as exc:
            raise IntegrityError("Invalid policy-call index") from exc
        if str(index) != str(raw_index) or index < 0:
            raise IntegrityError("Policy-call index must be a nonnegative integer")
        indices[row["record_id"]] = index
        if row["paired_status"] == "defined":
            values = (float(row["delta"]), float(row["contribution_high_distance_overshoot"]))
            if not all(math.isfinite(value) for value in values):
                raise IntegrityError("Nonfinite call-index value")
            if values[1] > 1e-10:
                raise IntegrityError("Overshoot contribution must be nonpositive")
            episodes[row["episode_id"]].append(row)
    weights = {row["record_id"]: 1 / (len(episodes) * len(group))
               for group in episodes.values() for row in group}
    total_overshoot = math.fsum(weights[row["record_id"]] * float(row["contribution_high_distance_overshoot"])
                                for group in episodes.values() for row in group)
    result = dict(case=case, n_raw_calls=len(calls), n_defined_calls=len(weights),
                  n_defined_episodes=len(episodes), contribution_weighting="equal defined episode",
                  conditional_summary_weighting="equal defined call")
    for name, predicate in (("first", lambda index: index == 0), ("later", lambda index: index > 0)):
        raw = [row for row in calls if predicate(indices[row["record_id"]])]
        defined = [row for row in raw if row["paired_status"] == "defined"]
        values = [float(row["delta"]) for row in defined]
        overshoot = math.fsum(weights[row["record_id"]] * float(row["contribution_high_distance_overshoot"])
                              for row in defined)
        result.update({name + "_" + key: value for key, value in dict(
            raw_calls=len(raw), defined_calls=len(defined), undefined_calls=len(raw)-len(defined),
            mean=statistics.mean(values) if values else None,
            median=statistics.median(values) if values else None,
            mean_contribution=math.fsum(weights[row["record_id"]] * float(row["delta"]) for row in defined)
                              if weights else None,
            overshoot_calls=sum(int(row["n_high_distance_overshoot"]) > 0 for row in raw),
            overshoot_percent=100 * sum(int(row["n_high_distance_overshoot"]) > 0 for row in raw) / len(raw)
                              if raw else None,
            overshoot_contribution=overshoot if weights else None,
            overshoot_contribution_fraction=overshoot / total_overshoot if total_overshoot < 0 else None,
        ).items()})
    result["mean"] = math.fsum(result[name + "_mean_contribution"] for name in ("first", "later")) if weights else None
    return result


def summarize_gaps(case, calls):
    """Expose endpoint-gap magnitudes without removing small-gap calls."""
    if not calls or any(row["case"] != case for row in calls):
        raise IntegrityError("Empty or cross-cohort gap summary")
    result = dict(case=case, n_calls=len(calls))
    for field in ("residual_baseline", "gap_q", "gap_l2"):
        values = [float(row[field]) for row in calls]
        if not all(math.isfinite(value) for value in values):
            raise IntegrityError("Nonfinite endpoint gap")
        if field == "residual_baseline" and min(values) < 0:
            raise IntegrityError("Negative squared residual")
        for name, value in (("minimum", min(values)), ("median", statistics.median(values)),
                            ("maximum", max(values)), ("zero_count", values.count(0.0)),
                            ("q05", _quantile(values, .05)), ("q95", _quantile(values, .95))):
            result[field + "_" + name] = value
    for threshold, label in ((.1, "tenth"), (.5, "half")):
        selected = [row for row in calls if float(row["residual_baseline"]) < threshold]
        result["baseline_below_" + label] = len(selected)
        result["first_baseline_below_" + label] = sum(int(row["policy_call_idx"]) == 0 for row in selected)
    return result


def schedule_bound(grid_percent):
    """Universal upper bound when normalized endpoints agree exactly.

    Positive epsilon generally makes the fixed-baseline bound stricter. This
    universal bound is not a claim that a particular policy can attain it.
    """
    grid = [float(value) / 100 for value in grid_percent]
    if (len(grid) < 2 or not all(math.isfinite(value) for value in grid)
            or grid[0] != 0 or grid[-1] != 1
            or any(right <= left for left, right in zip(grid, grid[1:]))):
        raise IntegrityError("Expected a strictly increasing 0-to-100 schedule")
    weights = [(grid[1] - grid[0]) / 2]
    weights.extend((grid[index+1] - grid[index-1]) / 2 for index in range(1, len(grid)-1))
    weights.append((grid[-1] - grid[-2]) / 2)
    endpoint_weight = weights[0] + weights[-1]
    return dict(grid_percent=[100 * value for value in grid], trapezoid_weights=weights,
                endpoint_weight=endpoint_weight, upper_bound=(1-endpoint_weight)/4,
                assumptions="exact self-reference, positive baseline residual, matching normalized endpoints")


def summarize_grid(case, calls):
    """Check saved-grid facts, without inferring unpreserved feature masks."""
    if not calls or any(row["case"] != case for row in calls):
        raise IntegrityError("Empty or cross-cohort grid summary")
    grid = json.loads(calls[0]["grid_percent"])
    bound = schedule_bound(grid)
    one = grid.index(1) if 1 in grid else None
    five = grid.index(5) if 5 in grid else None
    same_ordinates = 0
    eligible = 0
    maxima = []
    for row in calls:
        if json.loads(row["grid_percent"]) != grid:
            raise IntegrityError("Mixed schedules within a reporting case")
        residuals = json.loads(row["residual_points"])
        if len(residuals) != len(grid) or any(not math.isfinite(value) or value < 0 for value in residuals):
            raise IntegrityError("Invalid saved residual curve")
        if one is not None and five is not None:
            same_ordinates += residuals[one] == residuals[five]
        if row["paired_status"] != "defined":
            continue
        q, n = (json.loads(row[field]) for field in ("normalized_q", "normalized_l2"))
        if any(len(values) != len(grid) or not all(math.isfinite(value) for value in values) for values in (q, n)):
            raise IntegrityError("Invalid normalized saved curve")
        if float(row["residual_actual"]) == 0 and float(row["residual_baseline"]) > 0:
            if any(abs(q[index] - n[index]) > 1e-10 for index in (0, -1)):
                raise IntegrityError("Normalized endpoints disagree")
            if any(a-b > .25+1e-10 for a, b in zip(q, n)):
                raise IntegrityError("Saved ordinate exceeds universal response bound")
            delta = float(row["delta"])
            if not math.isfinite(delta) or delta > bound["upper_bound"] + 1e-10:
                raise IntegrityError("Saved AUC difference exceeds schedule bound")
            eligible += 1
            maxima.append(delta)
    return dict(case=case, n_calls=len(calls), **bound,
                bound_checked_calls=eligible,
                repeated_one_five_residuals=same_ordinates if one is not None and five is not None else None,
                maximum_delta=max(maxima) if maxima else None,
                maximum_percent_of_bound=100*max(maxima)/bound["upper_bound"]
                                        if maxima and bound["upper_bound"] > 0 else None,
                interpretation="Equal saved residuals do not authenticate masks; both responses use the same nominal schedule.")


def summarize_cohort_relationship(calls):
    """Compare recorded labels and scalars, without asserting context identity."""
    def key(row):
        try:
            episode = json.loads(row["episode_key"])
            raw_call = row["policy_call_idx"]
            call = int(raw_call)
        except (ValueError, TypeError, KeyError) as exc:
            raise IntegrityError("Invalid recorded cohort identity") from exc
        if (not isinstance(episode, list) or len(episode) != 5
                or not all(isinstance(value, str) and value for value in episode[:3])
                or any(type(value) is not int or value < 0 for value in episode[3:])
                or str(call) != str(raw_call) or call < 0):
            raise IntegrityError("Invalid recorded cohort identity")
        return (*episode[1:], call)

    cohorts = {}
    for ranking in ("Q", "L2"):
        case = ranking + ":vision:insertion"
        selected = calls.get(case, [])
        if not selected or any(row["case"] != case for row in selected):
            raise IntegrityError("Missing or cross-cohort vision-insertion records")
        _unique(selected, lambda row: row["record_id"], "physical cohort record")
        cohorts[ranking] = _unique(selected, key, "recorded cohort key")
    if set(cohorts["Q"]) != set(cohorts["L2"]):
        raise IntegrityError("Recorded cohort keys do not align completely")
    keys = sorted(cohorts["Q"])
    baseline = {}
    for ranking, selected in cohorts.items():
        baseline[ranking] = {identity: float(row["residual_baseline"]) for identity, row in selected.items()}
        if any(not math.isfinite(value) or value < 0 for value in baseline[ranking].values()):
            raise IntegrityError("Invalid baseline residual in cohort relationship")
        if any(row["paired_status"] == "defined" and not math.isfinite(float(row["delta"]))
               for row in selected.values()):
            raise IntegrityError("Nonfinite cohort effect")
    correlations = []
    for call in sorted({identity[-1] for identity in keys}):
        selected = [identity for identity in keys if identity[-1] == call]
        positive = [identity for identity in selected if all(baseline[rank][identity] > 0 for rank in cohorts)]
        x, y = ([math.log(baseline[rank][identity]) for identity in positive] for rank in ("Q", "L2"))
        coefficient = statistics.correlation(x, y) if len(x) > 1 and len(set(x)) > 1 and len(set(y)) > 1 else None
        correlations.append(dict(call=call, aligned_calls=len(selected), positive_baseline_pairs=len(positive),
                                 log_baseline_correlation=coefficient))
    threshold = -100.0
    tails, episode_sets = {}, {}
    for ranking, selected in cohorts.items():
        extreme = [identity for identity, row in selected.items()
                   if row["paired_status"] == "defined" and float(row["delta"]) < threshold]
        episode_sets[ranking] = {identity[:-1] for identity in extreme}
        tails[ranking] = dict(extreme_calls=len(extreme), extreme_episodes=len(episode_sets[ranking]),
            total_episodes=len({identity[:-1] for identity in selected}),
            episode_labels=[list(identity) for identity in sorted(episode_sets[ranking])],
            record_ids=[selected[identity]["record_id"] for identity in sorted(extreme)])
    return dict(case="vision:insertion", aligned_calls=len(keys),
        alignment_fields=["task", "model", "recorded_seed", "episode", "policy_call_idx"],
        unequal_baseline_calls=sum(baseline["Q"][identity] != baseline["L2"][identity] for identity in keys),
        call_index_correlations=correlations, extreme_threshold=threshold, extreme_operator="<",
        extreme_by_ranking=tails,
        shared_extreme_episodes=[list(identity) for identity in sorted(episode_sets["Q"] & episode_sets["L2"])],
        extreme_episode_sets_equal=episode_sets["Q"] == episode_sets["L2"],
        interpretation="Shared recorded labels and correlated residuals indicate dependence; they do not authenticate identical initial states, full contexts, noise or checkpoints, or justify ranking-quality comparisons.")


def summarize_language_schedule(case, calls):
    """Omit only the nominal 1% node; keep both responses on each same grid."""
    if ":lang:" not in case or not calls or any(row["case"] != case for row in calls):
        raise IntegrityError("Expected one language reporting case")
    _unique(calls, lambda row: row["record_id"], "language schedule record")
    grid = json.loads(calls[0]["grid_percent"])
    original = schedule_bound(grid)
    if 1 not in grid:
        raise IntegrityError("Language sensitivity requires the nominal 1% node")
    keep = [index for index, value in enumerate(grid) if value != 1]
    reduced = schedule_bound([grid[index] for index in keep])
    old_values, new_values, in_range_changes = [], [], []
    removed = grid.index(1)
    for row in calls:
        residuals = json.loads(row["residual_points"])
        if (len(residuals) != len(grid) or any(not math.isfinite(value) or value < 0 for value in residuals)):
            raise IntegrityError("Invalid language residual curve")
    equal_next = all(json.loads(row["residual_points"])[removed] ==
                     json.loads(row["residual_points"])[removed+1] for row in calls)
    # Each in-range response difference is in [0, 1/4], with zero endpoints.
    # If the omitted and following ordinates are equal, combine their weights.
    coefficients = [-weight for weight in original["trapezoid_weights"]]
    for index, weight in zip(keep, reduced["trapezoid_weights"]):
        coefficients[index] += weight
    if equal_next:
        coefficients[removed+1] += coefficients[removed]
        coefficients[removed] = 0
    interior = coefficients[1:-1]
    in_range_bound = .25 * max(sum(value for value in interior if value > 0),
                              -sum(value for value in interior if value < 0))
    for row in calls:
        if json.loads(row["grid_percent"]) != grid:
            raise IntegrityError("Mixed language schedules")
        if row["paired_status"] != "defined":
            continue
        q, n, residuals = (json.loads(row[field]) for field in ("normalized_q", "normalized_l2", "residual_points"))
        if any(len(values) != len(grid) or any(not math.isfinite(value) for value in values)
               for values in (q, n, residuals)):
            raise IntegrityError("Invalid language schedule values")
        difference = [a-b for a, b in zip(q, n)]
        old = math.fsum(weight * value for weight, value in zip(original["trapezoid_weights"], difference))
        new = math.fsum(weight * difference[index] for weight, index in zip(reduced["trapezoid_weights"], keep))
        _same(old, row["delta"], case + ":original language AUC")
        old_values.append(old)
        new_values.append(new)
        actual, baseline = float(row["residual_actual"]), float(row["residual_baseline"])
        if not all(math.isfinite(value) and value >= 0 for value in (actual, baseline)):
            raise IntegrityError("Invalid language endpoint residual")
        if actual == 0 and baseline > 0 and all(0 <= value <= baseline for value in residuals):
            if abs(difference[0]) > 1e-10 or abs(difference[-1]) > 1e-10:
                raise IntegrityError("Language normalized endpoints disagree")
            if any(value < -1e-10 or value > .25+1e-10 for value in difference):
                raise IntegrityError("Invalid in-range language response difference")
            if abs(new-old) > in_range_bound + 1e-10:
                raise IntegrityError("Language schedule difference exceeds in-range bound")
            in_range_changes.append(abs(new-old))
    old_median = statistics.median(old_values) if old_values else None
    new_median = statistics.median(new_values) if new_values else None
    return dict(case=case, n_raw_calls=len(calls), n_defined_calls=len(old_values),
        original_grid_percent=grid, reduced_grid_percent=reduced["grid_percent"], removed_node_percent=1,
        original_median=old_median, reduced_median=new_median,
        median_change=new_median-old_median if old_values else None,
        fully_in_range_calls=len(in_range_changes),
        maximum_in_range_absolute_change=max(in_range_changes) if in_range_changes else None,
        in_range_per_call_bound=in_range_bound,
        bound_uses_equal_removed_next_ordinates=equal_next,
        interpretation="Retrospective nominal-node sensitivity, using the same trapezoid weights for both responses; equal ordinates do not authenticate feature masks.")


def numerical_evaluation(projection):
    """Project saved evaluation checks without pooling rankings or responses."""
    rows = []
    complete = [cell for cell in projection["rows"] if cell["status"] == "complete_diagnostics"]
    if not complete:
        raise IntegrityError("No completed numerical diagnostics")
    cell_fields = ("stratum", "episode", "call", "context_id", "modality", "target")
    _unique(complete, lambda cell: tuple(cell[name] for name in cell_fields if name != "context_id"), "numerical cell")
    for cell in complete:
        observations = _unique([obs for obs in cell["observations"] if obs["repeat"] == 0],
                               lambda obs: obs["m"], "numerical first-repeat observation")
        budgets = sorted(observations)
        selected = [check for check in cell["checks"] if check["criterion"] in
                    {"group_spearman", "normalized_auc_difference", "rms_curve_difference"}]
        identities = ("criterion", "ranking", "candidate_m", "reference_m", "direction", "response")
        _unique(selected, lambda check: tuple(check.get(name) for name in identities), "numerical evaluation check")
        coordinate_pairs = {(check["ranking"], check["candidate_m"], check["reference_m"])
                            for check in cell["checks"] if check["criterion"] == "coordinate_relative_l1"}
        expected_pairs = {(ranking, candidate, reference) for ranking in ("IG", "path_gradient")
                          for candidate in budgets for reference in budgets if candidate < reference}
        if len(budgets) < 2 or coordinate_pairs != expected_pairs:
            raise IntegrityError("Incomplete numerical coordinate comparison coverage")
        expected = set()
        for ranking, candidate, reference in coordinate_pairs:
            expected.add(("group_spearman", ranking, candidate, reference, None, None))
            for direction in ("insertion", "deletion"):
                expected.add(("rms_curve_difference", ranking, candidate, reference, direction, "RMS"))
                for response in ("Q", "L2", "RMS"):
                    expected.add(("normalized_auc_difference", ranking, candidate, reference, direction, response))
        if not coordinate_pairs or {tuple(check.get(name) for name in identities) for check in selected} != expected:
            raise IntegrityError("Incomplete numerical evaluation check coverage")
        for check in selected:
            value = check["value"]
            if (check["status"] not in {"satisfies", "violates", "not_applicable"}
                    or (value is not None and not math.isfinite(float(value)))
                    or (value is None and check["status"] != "not_applicable")):
                raise IntegrityError("Invalid numerical evaluation value")
            row = {name: cell[name] for name in cell_fields}
            row.update({name: check.get(name) for name in (*identities, "value", "threshold", "operator", "status", "reason")})
            row["report_sha256"] = cell["report_sha256"]
            row["baseline_rms"] = check.get("baseline_rms")
            row["relative_to_baseline_rms"] = (value / check["baseline_rms"]
                if value is not None and check.get("baseline_rms", 0) > 0 else None)
            rows.append(row)
    return rows


def summarize_numerical_violations(projection, numerical_summary):
    complete = [cell for cell in projection["rows"] if cell["status"] == "complete_diagnostics"]
    statuses = Counter(check["status"] for cell in complete for check in cell["checks"])
    if dict(statuses) != projection["criterion_status_counts"]:
        raise IntegrityError("Numerical criterion counts differ from sealed summary")
    grouped = defaultdict(list)
    for cell in complete:
        grouped[(cell["modality"], cell["target"])].append(cell)
    summaries = []
    expected = _unique(numerical_summary["modality_target_extrema"],
                       lambda row: (row["modality"], row["target"]), "numerical modality/target summary")
    if set(grouped) != set(expected):
        raise IntegrityError("Numerical modality/target coverage differs")
    for (modality, target), cells in sorted(grouped.items()):
        violations = 0
        for cell in cells:
            budgets = sorted({obs["m"] for obs in cell["observations"]})
            if len(budgets) < 2:
                raise IntegrityError("Missing numerical comparison budgets")
            violations += any(check["status"] == "violates" and check.get("candidate_m") == budgets[-2]
                              and check.get("reference_m") == budgets[-1] for check in cell["checks"])
        _same(violations, expected[(modality, target)]["contexts_with_finest_pair_threshold_violation"],
              "finest-pair violating arms")
        _same(len(cells), expected[(modality, target)]["completed_contexts"], "completed numerical arms")
        summaries.append(dict(modality=modality, target=target, completed_arms=len(cells),
                              arms_with_finest_pair_violation=violations))
    return dict(full_ladder_status_counts=dict(statuses),
        full_ladder_violation_criteria=dict(sorted(Counter(check["criterion"] for cell in complete
            for check in cell["checks"] if check["status"] == "violates").items())),
        finest_pair_by_modality_target=summaries,
        interpretation="Dependent recorded criteria across both IG and path-gradient rankings, not independent samples or failure counts.")


def build_report(root=ROOT):
    root = Path(root)
    verified, sources = authenticate(root)
    summaries = _unique(read_csv(root / PAIRED / "summary.csv"), lambda row: row["case"], "paired summary case")
    geometries = _unique(read_csv(root / PAIRED / "geometry.csv"), lambda row: row["case"], "geometry case")
    omissions = _unique(read_csv(root / INFLUENCE / "case_summary.csv"), lambda row: row["case"], "omission case")
    calls = defaultdict(list)
    for row in read_csv(root / PAIRED / "calls.csv.gz"):
        calls[row["case"]].append(row)
    if any(set(group) != set(CASES) for group in (summaries, geometries, omissions, calls)):
        raise IntegrityError("All eight named cases must be present exactly once")
    paired = [summarize_case(case, calls[case], summaries[case], geometries[case], omissions[case]) for case in CASES]
    for rank in ("Q", "L2"):
        selected = [set(c["record_id"] for c in calls[case]) for case in CASES if case.startswith(rank + ":")]
        if any(members != selected[0] for members in selected[1:]):
            raise IntegrityError("Dependent views no longer share their declared cohort")
    if set(c["record_id"] for c in calls[CASES[0]]) & set(c["record_id"] for c in calls[CASES[4]]):
        raise IntegrityError("Ranking cohorts unexpectedly share physical records")
    projection = read_json(root / NUMERICAL / "inputs/sealed_v6.json")
    numerical = numerical_rows(projection)
    successes = success_counts(root)
    marginal = [row for row in read_csv(root / CANONICAL / "results.csv")
                if row["result_id"].startswith("rescore/") and row["occurrence_view"] == "retrospective_last"]
    tails = read_json(root / INFLUENCE / "tail_diagnostics.json")
    context_keys = {(r["stratum"], r["episode"], r["call"]) for r in numerical}
    call_indices = [summarize_call_indices(case, calls[case]) for case in CASES]
    gaps = [summarize_gaps(case, calls[case]) for case in CASES]
    grids = [summarize_grid(case, calls[case]) for case in CASES]
    for split, row in zip(call_indices, paired):
        _same(split["mean"], row["mean"], row["case"] + ":call-index contributions")
    report = dict(schema_version=3, reporting_version=VERSION,
        interpretation="retrospective descriptive presentation; no new model execution",
        estimator_history="The frozen 2026-10-01 paired protocol names the equal-episode mean primary. The 2026-10-02 reporting layer introduced the paired-call median and positive-change share as the lead summaries. The 2026-10-03 layers add retrospective call-index, gap, geometry, cohort-relationship, schedule-sensitivity and numerical-evaluation summaries; original means, intervals and populations remain unchanged.",
        optional_ranking_comparison="not included: locally saved maps cover only two selected YCB state contexts; the 12-context map and common-response curve set is unavailable",
        verification=verified, paired=paired, task_success=successes,
        call_index_summary=call_indices, baseline_gap_summary=gaps, grid_diagnostics=grids,
        cohort_relationship=summarize_cohort_relationship(calls),
        language_schedule_sensitivity=[summarize_language_schedule(case, calls[case]) for case in CASES if ":lang:" in case],
        numerical_evaluation_checks=numerical_evaluation(projection),
        marginal_rescoring=marginal, numerical=numerical,
        numerical_coverage=dict(planned=projection["planned_contexts"], complete=len(context_keys),
            episodes=len({key[:2] for key in context_keys}), status_counts=projection["context_status_counts"],
            criterion_status_counts=projection["criterion_status_counts"]),
        tail_examples={case: tails[case]["example"] for case in CASES},
        archive=read_json(root / CANONICAL / "summary.json"),
        random_reference=read_json(root / CANONICAL / "random_order_counterexample.json"),
        source_sha256=sources)
    # The portable producer owns the precise repeatability-count definition.
    numerical_summary = read_json(root / NUMERICAL / "outputs/summary.json")
    report["numerical_violation_summary"] = summarize_numerical_violations(projection, numerical_summary)
    report["numerical_original_summary"] = {key: numerical_summary[key] for key in
        ("completed_arms",) if key in numerical_summary}
    report["numerical_original_summary"].update({key: numerical_summary[key] for key in
        ("repeatability_equality_check_counts", "repeated_condition_and_coverage_check_counts",
         "repeat_design", "modality_target_extrema")})
    report["numerical_coverage"]["planned_one_b"] = sum(row["stratum"].endswith("-1b") for row in numerical_summary["roster"])
    report["numerical_coverage"]["later_completion_status"] = numerical_summary["final_stop_completion_state"]
    return report, calls


def write_report(report, root=ROOT):
    root = Path(root)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    files = {"report.json": json_text(report), "paired_summary.csv": csv_text(report["paired"]),
             "numerical_by_context.csv": csv_text(report["numerical"]),
             "task_success.csv": csv_text(report["task_success"]),
             "call_index_summary.csv": csv_text(report["call_index_summary"]),
             "baseline_gap_summary.csv": csv_text(report["baseline_gap_summary"]),
             "grid_diagnostics.csv": csv_text(report["grid_diagnostics"]),
             "language_schedule_sensitivity.csv": csv_text(report["language_schedule_sensitivity"]),
             "numerical_evaluation_checks.csv": csv_text(report["numerical_evaluation_checks"]),
             "numerical_violation_summary.csv": csv_text(report["numerical_violation_summary"]["finest_pair_by_modality_target"])}
    for name, text in files.items():
        (output / name).write_text(text, encoding="utf-8", newline="\n")
    sources = {path.relative_to(root).as_posix(): sha(path)
               for path in sorted((root / "analysis/manuscript_revision").glob("*.py"))}
    manifest = dict(schema_version=1, reporting_version=VERSION, source_sha256=sources,
                    input_sha256=report["source_sha256"],
                    outputs={name: sha(output / name) for name in files})
    (output / "manifest.json").write_text(json_text(manifest), encoding="utf-8", newline="\n")
    return output, manifest


def reproduce(root=ROOT):
    """Fresh local reproduction of every retained frozen scientific family."""
    root = Path(root)
    from analysis.revision.analyze import run as canonical_run
    from analysis.paired_rescoring.analyze import run as paired_run
    from analysis.paired_rescoring.influence import run as influence_run
    from analysis.revision.nested_grid_aliasing import run_example
    counts = {}
    with tempfile.TemporaryDirectory(prefix="paper-science-reproduction-") as directory:
        scratch = Path(directory)
        for label, source, producer in (("canonical", CANONICAL, lambda dest: canonical_run(root, dest, 10000, 0)),
                                        ("paired", PAIRED, lambda dest: paired_run(dest, root=root)),
                                        ("influence", INFLUENCE, lambda dest: influence_run(dest, root=root))):
            destination = scratch / label
            producer(destination)
            expected = read_json(root / source / "provenance.json")["artifacts_sha256"]
            for name, digest in expected.items():
                if sha(destination / name) != digest:
                    raise IntegrityError(f"Fresh {label} result does not reproduce: {name}")
            counts[label] = len(expected)
        destination = scratch / "numerical"
        subprocess.run([sys.executable, str(root / NUMERICAL / "summarize.py"), "--output", str(destination)],
                       cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        expected = read_json(root / NUMERICAL / "outputs/manifest.json")["outputs"]
        for name, digest in expected.items():
            if sha(destination / name) != digest:
                raise IntegrityError("Fresh numerical result does not reproduce: " + name)
        counts["numerical"] = len(expected)
        if run_example() != read_json(root / "analysis/revision/examples/nested_grid_aliasing.json"):
            raise IntegrityError("Fresh aliasing toy does not reproduce")
        counts["aliasing"] = 1
    _, sources = authenticate(root)
    receipt = dict(status="fresh local CPU reproduction passed", families=counts,
                   total_scientific_artifacts=sum(counts.values()), input_sha256=sources)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    (output / "reproduction.json").write_text(json_text(receipt), encoding="utf-8", newline="\n")
    return receipt
