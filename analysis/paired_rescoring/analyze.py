"""CPU-only paired response analysis; never executes a model or changes inputs."""
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

import numpy as np

from analysis.revision.core import (
    IntegrityError, bootstrap_summary, json_bytes, load_file, normalized_auc,
    population_id, reconcile, sha256, transform_score,
)
from analysis.revision.analyze import check_stored_arithmetic
from analysis.revision.verify import verify as verify_canonical

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CANONICAL_PROVENANCE_SHA256 = "606cfd5307afeff873789ef48db969d914a42348552b447a6daf298182ff0a8b"
CATEGORIES = ("in_range", "high_distance_overshoot", "low_distance_undershoot")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_gzip(path, blob):
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            stream.write(blob)


def csv_bytes(rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json_bytes(v).decode() if isinstance(v, (list, dict)) else v
                         for k, v in row.items()})
    return stream.getvalue().encode("utf-8")


def episode_key(record):
    return (record.source,) + tuple(record.row[k] for k in ("task", "model", "seed", "episode"))


def classify(value, actual, baseline, tolerance=0.0):
    if not all(math.isfinite(x) for x in (value, actual, baseline, tolerance)) or tolerance < 0:
        raise IntegrityError("Invalid geometry value/tolerance")
    lo, hi = sorted((actual, baseline))
    if value > hi + tolerance:
        return "high_distance_overshoot"
    if value < lo - tolerance:
        return "low_distance_undershoot"
    return "in_range"


def paired_curve(grid, native_curve, native_actual, native_baseline, ranking,
                 *, dimension=512, epsilon=1e-12):
    """Pair readouts on one recorded curve, with explicit endpoint geometry.

    The generalized ordinate identity uses stored endpoints, including reversed
    gaps. Geometry is an analysis of sampled ordinates, not intermediate masks.
    The native L2 values are retained exactly; the canonical transform permits
    only its predeclared tiny negative squared-distance conversion roundoff.
    """
    # Validates the grid even when the endpoint gap is undefined.
    normalized_auc(grid, native_curve, native_actual, native_baseline)
    q = [transform_score(x, ranking, "Q", dimension=dimension, epsilon=epsilon)
         for x in [native_actual, native_baseline, *native_curve]]
    l2 = [transform_score(x, ranking, "L2", dimension=dimension, epsilon=epsilon)
          for x in [native_actual, native_baseline, *native_curve]]
    residuals = [-2 * dimension * x for x in q]
    actual, baseline, *points = residuals
    labels = [classify(x, actual, baseline) for x in points]
    tolerance = 1e-6 * max(1.0, abs(actual), abs(baseline))
    sensitive_labels = [classify(x, actual, baseline, tolerance) for x in points]
    q_auc = normalized_auc(grid, q[2:], q[0], q[1])
    l_auc = normalized_auc(grid, l2[2:], l2[0], l2[1])
    orientation = "baseline_farther" if baseline > actual else (
        "baseline_nearer" if baseline < actual else "equal")
    segment_counts = Counter()
    for a, b in zip(labels, labels[1:]):
        inside = int(a == "in_range") + int(b == "in_range")
        segment_counts[{2: "both_inside", 1: "one_inside_one_outside", 0: "both_outside"}[inside]] += 1
    result = dict(
        paired_status="defined" if q_auc is not None and l_auc is not None else "undefined_zero_gap",
        auc_q=q_auc, auc_l2=l_auc,
        undefined_responses=[name for name, value in (("Q", q_auc), ("L2", l_auc)) if value is None],
        delta=None, gap_q=q[0] - q[1], gap_l2=l2[0] - l2[1],
        residual_actual=actual, residual_baseline=baseline,
        orientation=orientation, grid_percent=list(grid), residual_points=points,
        native_actual=native_actual, native_baseline=native_baseline,
        native_curve=list(native_curve), normalized_q=None, normalized_l2=None,
        node_labels=labels, tolerance_node_labels=sensitive_labels,
        geometry_tolerance=tolerance, n_grid_nodes=len(points),
        n_label_changes=sum(a != b for a, b in zip(labels, sensitive_labels)),
        n_native_l2_domain_roundoff=sum(x * x - epsilon < 0 for x in
                                      [native_actual, native_baseline, *native_curve]) if ranking == "L2" else 0,
        segment_endpoint_counts=dict(segment_counts),
        decomposition_residual=None, generalized_identity_max_error=None,
        **{f"n_{c}": labels.count(c) for c in CATEGORIES},
        **{f"n_interior_{c}": labels[1:-1].count(c) for c in CATEGORIES},
        **{f"tolerance_n_{c}": sensitive_labels.count(c) for c in CATEGORIES},
        **{f"contribution_{c}": None for c in CATEGORIES},
    )
    if result["paired_status"] != "defined":
        return result
    nq = [(x - q[1]) / result["gap_q"] for x in q[2:]]
    nl = [(x - l2[1]) / result["gap_l2"] for x in l2[2:]]
    differences = [x - y for x, y in zip(nq, nl)]
    weights = [0.0] * len(grid)
    for i, (a, b) in enumerate(zip(grid, grid[1:])):
        weights[i] += (b - a) / 200
        weights[i + 1] += (b - a) / 200
    contributions = {c: math.fsum(w * d for w, d, label in zip(weights, differences, labels)
                                  if label == c) for c in CATEGORIES}
    delta = q_auc - l_auc
    decomposition_residual = delta - math.fsum(contributions.values())
    if not all(math.isfinite(x) for x in [*nq, *nl, *differences, *contributions.values(), delta, decomposition_residual]):
        raise IntegrityError("Nonfinite derived paired geometry")
    if abs(decomposition_residual) > 1e-10 * max(1.0, abs(q_auc), abs(l_auc)):
        raise IntegrityError("Paired trapezoid decomposition failed")
    a, b = math.sqrt(actual + epsilon), math.sqrt(baseline + epsilon)
    # Residual equality can arise from canonical tiny-negative-domain clamping.
    identity_error = None
    if actual != baseline:
        theory = [(b - math.sqrt(x + epsilon)) * (math.sqrt(x + epsilon) - a) /
                  (baseline - actual) for x in points]
        identity_error = max(abs(x - y) for x, y in zip(differences, theory))
    result.update(delta=delta, normalized_q=nq, normalized_l2=nl,
                  decomposition_residual=decomposition_residual,
                  generalized_identity_max_error=identity_error,
                  **{f"contribution_{c}": contributions[c] for c in CATEGORIES})
    return result


def summarize_calls(calls, *, draws=10000, seed=0):
    """Keep raw/defined populations distinct and bind both estimates to callers."""
    if not calls or len({r["record_id"] for r in calls}) != len(calls):
        raise IntegrityError("Empty or repeated call population")
    raw_groups = {r["episode_id"] for r in calls}
    kept = [r for r in calls if r["paired_status"] == "defined"]
    if any(r["delta"] is None or not math.isfinite(r["delta"]) for r in kept):
        raise IntegrityError("Nonfinite defined paired effect")
    by_episode = defaultdict(list)
    for row in kept:
        by_episode[row["episode_id"]].append(row)
    episodes = []
    for group, rows in sorted(by_episode.items()):
        episodes.append(dict(episode_id=group, n_defined_calls=len(rows),
                             n_raw_calls=sum(r["episode_id"] == group for r in calls),
                             mean_delta=float(np.mean([r["delta"] for r in rows])),
                             record_ids=sorted(r["record_id"] for r in rows),
                             **{f"mean_contribution_{c}": float(np.mean([r[f"contribution_{c}"] for r in rows]))
                                for c in CATEGORIES}))
    members = dict(raw=sorted(r["record_id"] for r in calls),
                   defined=sorted(r["record_id"] for r in kept),
                   undefined=[dict(record_id=r["record_id"], reason=r["paired_status"])
                              for r in calls if r["paired_status"] != "defined"],
                   episode_groups={r["episode_id"]: r["record_ids"] for r in episodes})
    digest = sha256(json_bytes(members))
    summary = dict(n_raw_calls=len(calls), n_defined_calls=len(kept),
                   n_undefined_calls=len(calls) - len(kept), n_raw_episodes=len(raw_groups),
                   n_defined_episodes=len(episodes), n_fully_undefined_episodes=len(raw_groups) - len(episodes),
                   point_membership_sha256=digest, ci_membership_sha256=digest,
                   bootstrap_draws=draws, bootstrap_seed=seed)
    empty = dict(point=None, ci_lo=None, ci_hi=None, interval_status="no_defined_episodes")
    mean = bootstrap_summary([r["mean_delta"] for r in episodes],
                             [r["episode_id"] for r in episodes], statistic="mean", draws=draws, seed=seed) if kept else empty
    median = bootstrap_summary([r["delta"] for r in kept], [r["episode_id"] for r in kept],
                               statistic="median", draws=draws, seed=seed) if kept else empty
    for label, values in (("equal_episode_mean", mean), ("paired_call_median", median)):
        for field in ("point", "ci_lo", "ci_hi", "interval_status"):
            summary[f"{label}_{field}"] = values[field]
    values = [r["delta"] for r in kept]
    summary.update(min_delta=min(values) if kept else None, max_delta=max(values) if kept else None,
                   n_negative_delta=sum(x < 0 for x in values), n_zero_delta=sum(x == 0 for x in values),
                   **{f"equal_episode_contribution_{c}": float(np.mean([r[f"mean_contribution_{c}"] for r in episodes]))
                      if kept else None for c in CATEGORIES})
    return summary, episodes, members


def geometry_summary(calls):
    n = len(calls)
    nodes = sum(r["n_grid_nodes"] for r in calls)
    interior = sum(max(0, r["n_grid_nodes"] - 2) for r in calls)
    result = dict(n_calls=n, n_grid_nodes=nodes, n_interior_nodes=interior,
                  n_calls_any_outside=sum(r["n_in_range"] != r["n_grid_nodes"] for r in calls),
                  n_calls_nonzero_actual_residual=sum(r["residual_actual"] != 0 for r in calls),
                  max_actual_residual=max(r["residual_actual"] for r in calls),
                  min_baseline_residual=min(r["residual_baseline"] for r in calls),
                  min_absolute_gap_q=min(abs(r["gap_q"]) for r in calls),
                  min_absolute_gap_l2=min(abs(r["gap_l2"]) for r in calls),
                  n_native_l2_domain_roundoff=sum(r["n_native_l2_domain_roundoff"] for r in calls),
                  n_tolerance_label_changes=sum(r["n_label_changes"] for r in calls),
                  **{f"n_{c}": sum(r["orientation"] == c for r in calls)
                     for c in ("baseline_farther", "baseline_nearer", "equal")})
    for category in CATEGORIES:
        count = sum(r[f"n_{category}"] for r in calls)
        interior_count = sum(r[f"n_interior_{category}"] for r in calls)
        result.update({f"nodes_{category}": count, f"node_fraction_{category}": count / nodes,
                       f"interior_nodes_{category}": interior_count,
                       f"interior_fraction_{category}": interior_count / interior if interior else None,
                       f"calls_any_{category}": sum(r[f"n_{category}"] > 0 for r in calls),
                       f"tolerance_nodes_{category}": sum(r[f"tolerance_n_{category}"] for r in calls)})
    for label in ("both_inside", "one_inside_one_outside", "both_outside"):
        result[f"segments_{label}"] = sum(r["segment_endpoint_counts"].get(label, 0) for r in calls)
    return result


def authenticate(root, protocol):
    canonical = root / protocol["canonical_results"]
    provenance_bytes = (canonical / "provenance.json").read_bytes()
    if sha256(provenance_bytes) != CANONICAL_PROVENANCE_SHA256:
        raise IntegrityError("Canonical provenance differs from the reviewed v2 anchor")
    provenance = json.loads(provenance_bytes)
    manifest_path = root / protocol["input_manifest"]
    if sha256(manifest_path.read_bytes()) != provenance["manifest_sha256"]:
        raise IntegrityError("Input manifest mismatch")
    for name, expected in provenance["code_sha256"].items():
        if sha256((root / "analysis/revision" / name).read_bytes()) != expected:
            raise IntegrityError(f"Canonical code mismatch: {name}")
    verification = verify_canonical(canonical)
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["files"] != provenance["inputs"]:
        raise IntegrityError("Canonical input/provenance mismatch")
    raw, ledgers, inputs = [], [], []
    for item in manifest["files"]:
        if any(fnmatch.fnmatchcase(Path(item["path"]).name, pattern)
               for pattern in protocol["cohorts"].values()):
            rows, ledger = load_file(root / item["path"], root, item)
            raw.extend(r for r in rows if r.row["event"] == protocol["event"])
            ledgers.extend(ledger)
            inputs.append(item)
    if not raw:
        raise IntegrityError("No declared source records")
    canonical_ledger = list(csv.DictReader(io.StringIO(gzip.decompress(
        (canonical / "raw_line_ledger.csv.gz").read_bytes()).decode())))
    known = {f"{r['source']}:{r['line']}:{r['line_sha256']}" for r in canonical_ledger if r["status"] == "valid"}
    if not {r.id for r in raw} <= known:
        raise IntegrityError("Saved curves are absent from the canonical physical-line ledger")
    last, no_conflicts, duplicates = reconcile(raw)
    arithmetic = check_stored_arithmetic(raw)
    return last, no_conflicts, dict(canonical_verification=verification, inputs=inputs,
                                    n_raw_occurrences=len(raw), n_selected_occurrences=len(last),
                                    duplicates=duplicates, native_arithmetic=arithmetic,
                                    unknown_identities=manifest["unknown_identities"])


def make_table(summaries):
    lines = [r"\begin{tabular}{lllrr}", r"\toprule",
             r"Ranking & Modality & Curve & Mean paired $\Delta$ & 95\% interval \\", r"\midrule"]
    for row in summaries:
        point, lo, hi = (row[f"equal_episode_mean_{x}"] for x in ("point", "ci_lo", "ci_hi"))
        p = "undefined" if point is None else f"{point:.4f}"
        ci = "unavailable" if lo is None else f"[{lo:.4f}, {hi:.4f}]"
        lines.append(f"{row['ranking']} & {row['modality']} & {row['direction']} & {p} & {ci} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  "% Delta = AUC_Q - AUC_stabilized_L2; equal-episode mean of within-call differences.",
                  "% Conditional descriptive 95% episode bootstrap intervals, B=10000, seed=0; unadjusted."])
    for row in summaries:
        lines.append(f"% {row['case']}: {row['n_defined_calls']}/{row['n_raw_calls']} calls; "
                     f"{row['n_defined_episodes']}/{row['n_raw_episodes']} recorded episodes.")
    return "\n".join(lines) + "\n"


def narrative(summaries, geometries, audit):
    lines = ["# Paired response rescoring: 2026-10-01-v1", "",
             "This is a retrospective response-only analysis of authenticated saved curves. "
             "The ranking, saved intervention ordinates and endpoints are fixed within each call. "
             "The signed difference is normalized AUC_Q minus normalized AUC_stabilized_L2. "
             "Q-ranking and L2-ranking cohorts are never paired with each other.", "",
             "The primary statistic gives equal weight to each recorded episode after averaging "
             "its paired call differences. This new estimand is distinct from the canonical marginal "
             "call medians; neither weighting is universally required by episode resampling. The median "
             "of paired call differences is retained as a prespecified descriptive sensitivity. "
             "Both intervals resample whole recorded episodes, B=10000, PCG64 seed 0, percentile 95%. "
             "All 16 intervals across eight cases are unadjusted, conditional empirical descriptions, "
             "without calibrated coverage or a global significance claim.", "",
             "| Ranking | Modality | Curve | Calls / episodes | Mean paired difference [95% interval] | Paired-call median [95% interval] |",
             "|---|---|---|---|---|---|"]
    for row in summaries:
        estimates = []
        for label in ("equal_episode_mean", "paired_call_median"):
            values = [row[f"{label}_{x}"] for x in ("point", "ci_lo", "ci_hi")]
            estimates.append("undefined/unavailable" if any(x is None for x in values)
                             else f"{values[0]:.6f} [{values[1]:.6f}, {values[2]:.6f}]")
        lines.append(f"| {row['ranking']} | {row['modality']} | {row['direction']} | "
                     f"{row['n_defined_calls']} / {row['n_defined_episodes']} | {estimates[0]} | {estimates[1]} |")
    lines.extend(["", "## Saved-point geometry", "",
                  "Classification compares squared discrepancy with both stored endpoint discrepancies. "
                  "In range means inside that closed interval; high-distance overshoot is above both "
                  "endpoints; low-distance undershoot is below both. If the baseline is nearer than the "
                  "actual endpoint, the usual in-range sign reverses. We never assume exact self-reference. "
                  "Reported node percentages use all saved grid ordinates, including endpoints, with "
                  "interior-only counts also retained in geometry.csv. Calls are counted as having an "
                  "overshoot if at least one recorded ordinate is above both endpoints. These are pooled "
                  "descriptive counts, not independent observations or additional inference units.", "",
                  "| Ranking | Modality | Curve | In-range nodes | High overshoot nodes | Low undershoot nodes | Calls with high overshoot |", "|---|---|---|---|---|---|---|"])
    for row in geometries:
        lines.append(f"| {row['ranking']} | {row['modality']} | {row['direction']} | "
                     f"{row['node_fraction_in_range']:.2%} | {row['node_fraction_high_distance_overshoot']:.2%} | "
                     f"{row['node_fraction_low_distance_undershoot']:.2%} | "
                     f"{row['calls_any_high_distance_overshoot']}/{row['n_calls']} |")
    lines.extend(["", "The exact trapezoid difference is decomposed by saved-node category in summary.csv "
                  "and calls.csv.gz. Segment labels refer only to their two sampled endpoints. No "
                  "continuous crossing or unseen intervention is inferred. The prespecified 1e-6 "
                  "endpoint-scale tolerance changes labels only; it cannot change an AUC, endpoint "
                  "gap, defined population or inference result.", "", "## Accounting and limits", "",
                  f"The input audit retains {audit['n_raw_occurrences']} physical faithfulness occurrences "
                  f"and selects {audit['n_selected_occurrences']} by the canonical last-occurrence rule. "
                  "The conflict-excluded population comparison is recorded for each ranking arm. "
                  "Every input source and physical line is hash-bound to the canonical v2 manifest and "
                  "ledger, and every native stored AUC is checked before transformation.", "",
                  "Undefined zero-gap calls remain in raw accounting and are excluded only from the "
                  "explicit paired-defined estimand. Negative gaps and negative effects are retained. "
                  "There is no action-norm filter. Native L2 conversion roundoff, actual endpoint "
                  "residuals, orientation and finite extremes remain visible in the outputs. Source-file "
                  "namespaces are checked against legacy episode group counts to detect unintended "
                  "regrouping.", "",
                  "The historical grid is nominal percentage, not authenticated realized mask fraction. "
                  "Checkpoint, observation, noise and reference action tensors are unavailable for "
                  "authentication. This analysis changes the response readout only, and supplies no "
                  "ranking efficacy, task success, causal effect or empirical random-control claim. "
                  "The saved-grid statistic also differs from a continuous path integral.", "",
                  "## Reproduction", "", "From the repository root, with NumPy installed:", "", "```text",
                  "python -m analysis.paired_rescoring.analyze --output <new-empty-directory>", "```", "",
                  "The command refuses an existing directory. Compare the generated scientific files "
                  "against artifacts_sha256 in provenance.json. The environment fields and exact new "
                  "source/protocol hashes document execution; they are not silently normalized. "
                  "Canonical revision code and outputs remain untouched.", ""])
    return "\n".join(lines)


def run(output, *, root=ROOT):
    output, root = Path(output), Path(root)
    if output.exists():
        raise IntegrityError("Refusing an existing output directory")
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_bytes())
    records, no_conflicts, audit = authenticate(root, protocol)
    summaries, geometries, episodes, all_calls, memberships = [], [], [], [], {}
    audit["cohorts"] = {}
    with (root / protocol["canonical_results"] / "results.csv").open(encoding="utf-8", newline="") as stream:
        canonical_results = {r["result_id"]: r for r in csv.DictReader(stream)
                             if r["occurrence_view"] == "retrospective_last"}
    audit["canonical_marginal_reproduction"] = []
    for ranking, pattern in protocol["cohorts"].items():
        cohort = sorted((r for r in records if fnmatch.fnmatchcase(Path(r.source).name, pattern)), key=lambda r: r.id)
        sensitivity = sorted((r for r in no_conflicts if fnmatch.fnmatchcase(Path(r.source).name, pattern)), key=lambda r: r.id)
        if not cohort:
            raise IntegrityError(f"Empty declared cohort {ranking}")
        group_count = len({episode_key(r) for r in cohort})
        legacy_count = len({episode_key(r)[1:] for r in cohort})
        audit["cohorts"][ranking] = dict(n_calls=len(cohort), n_episodes=group_count,
                                         n_legacy_episode_keys=legacy_count,
                                         source_namespace_changes_group_count=group_count != legacy_count,
                                         selected_population_sha256=population_id(cohort),
                                         conflict_excluded_population_sha256=population_id(sensitivity),
                                         conflict_excluded_membership_identical=[r.id for r in cohort] == [r.id for r in sensitivity])
        audit["cohorts"][ranking]["calls_per_episode_histogram"] = dict(sorted(Counter(
            Counter(episode_key(r) for r in cohort).values()).items()))
        # Changing the known canonical cohorts requires a new protocol version.
        if len(cohort) != 750 or group_count != 30 or legacy_count != 30:
            raise IntegrityError("Canonical paired cohort/group count differs from the declared 750/30 roster")
        for modality in protocol["modalities"]:
            for direction in protocol["directions"]:
                case = f"{ranking}:{modality}:{direction}"
                identity = dict(case=case, ranking=ranking, modality=modality, direction=direction)
                calls = []
                for record in cohort:
                    row = record.row
                    calls.append(dict(**identity, record_id=record.id, source=record.source,
                                      source_sha256=record.source_sha256, line=record.line,
                                      line_sha256=record.line_sha256,
                                      episode_id=sha256(json_bytes(episode_key(record))),
                                      episode_key=list(episode_key(record)), policy_call_idx=row["policy_call_idx"],
                                      **paired_curve(row["k_grid_auc"], row[f"{modality}_{direction}_curve"],
                                                     row[f"{modality}_f_input"], row[f"{modality}_f_baseline"], ranking,
                                                     dimension=protocol["dimension"], epsilon=protocol["stabilizer_epsilon"])))
                for response, field in (("Q", "auc_q"), ("L2", "auc_l2")):
                    key = f"rescore/ranking_{ranking}/response_{response}/{modality}_{direction}"
                    prior = canonical_results[key]
                    retained = [(r, c) for r, c in zip(cohort, calls) if c[field] is not None]
                    if population_id([r for r, c in retained]) != prior["point_population_sha256"]:
                        raise IntegrityError(f"Canonical transformed population mismatch: {key}")
                    value = float(np.median([c[field] for r, c in retained]))
                    error = abs(value - float(prior["point"]))
                    if error > 1e-14 * max(1, abs(value)):
                        raise IntegrityError(f"Canonical transformed marginal mismatch: {key}")
                    audit["canonical_marginal_reproduction"].append(dict(result_id=key, median=value,
                                                                         absolute_error=error,
                                                                         population_sha256=prior["point_population_sha256"]))
                summary, episode_rows, membership = summarize_calls(calls, **{
                    k: protocol["bootstrap"][k] for k in ("draws", "seed")})
                summaries.append(dict(**identity, **summary))
                geometries.append(dict(**identity, **geometry_summary(calls)))
                episodes.extend(dict(**identity, **r) for r in episode_rows)
                memberships[case] = membership
                all_calls.extend(calls)
    audit["max_generalized_identity_error"] = max((r["generalized_identity_max_error"] or 0 for r in all_calls), default=0)
    audit["max_decomposition_residual"] = max((abs(r["decomposition_residual"] or 0) for r in all_calls), default=0)
    audit["n_undefined_paired_case_calls"] = sum(r["paired_status"] != "defined" for r in all_calls)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("summary.csv", summaries), ("episodes.csv", episodes), ("geometry.csv", geometries)):
        (output / name).write_bytes(csv_bytes(rows))
    write_gzip(output / "calls.csv.gz", csv_bytes(all_calls))
    write_gzip(output / "membership.json.gz", json_bytes(memberships) + b"\n")
    write_json(output / "audit.json", audit)
    (output / "methods_results.md").write_text(narrative(summaries, geometries, audit), encoding="utf-8", newline="\n")
    (output / "paired_rescoring.tex").write_text(make_table(summaries), encoding="utf-8", newline="\n")
    artifacts = {p.name: sha256(p.read_bytes()) for p in sorted(output.iterdir())}
    canonical = root / protocol["canonical_results"]
    provenance = dict(schema_version=1, analysis_version=protocol["analysis_version"],
                      canonical_provenance_sha256=CANONICAL_PROVENANCE_SHA256,
                      protocol_sha256=sha256(protocol_path.read_bytes()),
                      source_sha256={p.name: sha256(p.read_bytes()) for p in sorted(HERE.glob("*.py"))},
                      canonical_code_sha256=json.loads((canonical / "provenance.json").read_bytes())["code_sha256"],
                      artifacts_sha256=artifacts,
                      environment=dict(python=platform.python_version(), numpy=np.__version__),
                      command="python -m analysis.paired_rescoring.analyze --output <new-empty-directory>",
                      units="dimensionless saved-grid normalized response AUC difference",
                      interpretation="retrospective_descriptive_no_efficacy_claim")
    write_json(output / "provenance.json", provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
