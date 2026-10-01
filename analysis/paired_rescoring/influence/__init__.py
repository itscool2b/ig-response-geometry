"""Exploratory recorded-episode influence; the frozen paired analysis is unchanged."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fnmatch
import gzip
import io
import json
import math
from pathlib import Path
import platform
from statistics import median

from analysis.paired_rescoring import analyze as paired
from analysis.revision.core import IntegrityError, json_bytes, sha256

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = Path("analysis/paired_rescoring/influence_protocol.json")
OUTPUT_NAMES = ("protocol.json", "case_summary.csv", "episode_omissions.csv",
                "tail_diagnostics.json", "influence_table.tex", "methods_results.md")


def mean(values):
    return math.fsum(values) / len(values)


def sign(value):
    return (value > 0) - (value < 0)


def membership_hash(ids):
    if len(ids) != len(set(ids)):
        raise IntegrityError("Repeated physical record in an influence population")
    return sha256(json_bytes(sorted(ids)))


def summarize(calls):
    """Drop each entire episode once, including for unequal episode lengths."""
    if not calls:
        raise IntegrityError("Empty influence population")
    membership_hash([r["record_id"] for r in calls])
    groups = defaultdict(list)
    for row in calls:
        if row.get("paired_status") != "defined" or not math.isfinite(row["delta"]):
            raise IntegrityError("Influence requires the explicitly paired-defined population")
        groups[row["episode_id"]].append(row)
    if len(groups) < 2:
        raise IntegrityError("Episode omission requires at least two recorded episodes")
    episode_means = {key: mean([r["delta"] for r in rows]) for key, rows in sorted(groups.items())}
    full_mean = mean(list(episode_means.values()))
    negative_mass = math.fsum(-v for v in episode_means.values() if v < 0)
    worst_episode = min(episode_means, key=lambda key: (episode_means[key], key))
    omissions = []
    for key, rows in sorted(groups.items()):
        retained = [r for r in calls if r["episode_id"] != key]
        remaining_mean = mean([v for group, v in episode_means.items() if group != key])
        omissions.append(dict(
            omitted_episode_id=key, omitted_record_ids=sorted(r["record_id"] for r in rows),
            omitted_population_sha256=membership_hash([r["record_id"] for r in rows]),
            retained_population_sha256=membership_hash([r["record_id"] for r in retained]),
            n_omitted_calls=len(rows), n_retained_calls=len(retained),
            n_retained_episodes=len(groups) - 1, omitted_episode_mean=episode_means[key],
            remaining_equal_episode_mean=remaining_mean,
            full_mean_sign=sign(full_mean), remaining_mean_sign=sign(remaining_mean),
            sign_preserved=sign(full_mean) == sign(remaining_mean)))
    summary = dict(
        n_defined_calls=len(calls), n_recorded_episodes=len(groups),
        population_sha256=membership_hash([r["record_id"] for r in calls]),
        equal_episode_mean=full_mean,
        omission_mean_min=min(r["remaining_equal_episode_mean"] for r in omissions),
        omission_mean_max=max(r["remaining_equal_episode_mean"] for r in omissions),
        n_omissions=len(omissions), n_sign_preserved=sum(r["sign_preserved"] for r in omissions),
        n_negative_episode_means=sum(v < 0 for v in episode_means.values()),
        negative_episode_mean_mass=negative_mass,
        largest_negative_episode_id=worst_episode if negative_mass else None,
        largest_negative_episode_mean=episode_means[worst_episode] if negative_mass else None,
        largest_negative_episode_mass_share=-episode_means[worst_episode] / negative_mass if negative_mass else None)
    return summary, omissions


def tail_diagnostic(calls, top_counts, historical_cutoff):
    """Describe the minimum paired difference in every case without exclusion."""
    worst = min(calls, key=lambda r: (r["delta"], r["record_id"]))
    negatives = sorted((r for r in calls if r["delta"] < 0), key=lambda r: (r["delta"], r["record_id"]))
    mass = math.fsum(-r["delta"] for r in negatives)
    maximum = max(worst["residual_points"])
    baseline = worst["residual_baseline"]
    return dict(
        selection="minimum paired call difference, physical-record-ID tie break, within this case only",
        n_negative_calls=len(negatives), negative_call_mass=mass,
        top_negative_call_mass=[dict(
            requested_count=k, included_count=min(k, len(negatives)),
            record_ids=[r["record_id"] for r in negatives[:k]],
            magnitude=math.fsum(-r["delta"] for r in negatives[:k]),
            share=math.fsum(-r["delta"] for r in negatives[:k]) / mass if mass else None)
            for k in top_counts],
        example={key: worst[key] for key in (
            "record_id", "source", "source_sha256", "line", "line_sha256", "episode_id", "episode_key",
            "policy_call_idx", "delta", "auc_q", "auc_l2", "gap_q", "gap_l2", "native_actual",
            "native_baseline", "native_curve", "grid_percent", "residual_actual", "residual_baseline",
            "residual_points", "orientation")},
        maximum_saved_squared_residual=maximum,
        maximum_to_baseline_squared_residual_ratio=maximum / baseline if baseline > 0 else None,
        baseline_weak_empirical_percentile=100 * sum(r["residual_baseline"] <= baseline for r in calls) / len(calls),
        maximum_residual_weak_empirical_percentile=100 * sum(max(r["residual_points"]) <= maximum for r in calls) / len(calls),
        absolute_q_gap_exceeds_historical_cutoff=abs(worst["gap_q"]) > historical_cutoff,
        historical_gap_cutoff_for_description_only=historical_cutoff)


def authenticate(root, protocol):
    """Bind saved paired calls to frozen outputs, physical bytes and grouping."""
    location = root / protocol["paired_results"]
    provenance_bytes = (location / "provenance.json").read_bytes()
    if sha256(provenance_bytes) != protocol["paired_provenance_sha256"]:
        raise IntegrityError("Frozen paired provenance anchor mismatch")
    provenance = json.loads(provenance_bytes)
    for name, digest in provenance["artifacts_sha256"].items():
        if sha256((location / name).read_bytes()) != digest:
            raise IntegrityError(f"Frozen paired artifact mismatch: {name}")
    paired_protocol_bytes = (root / protocol["paired_protocol"]).read_bytes()
    if sha256(paired_protocol_bytes) != provenance["protocol_sha256"]:
        raise IntegrityError("Frozen paired protocol mismatch")
    for name, digest in provenance["source_sha256"].items():
        if sha256((root / "analysis/paired_rescoring" / name).read_bytes()) != digest:
            raise IntegrityError(f"Frozen paired source mismatch: {name}")
    paired_protocol = json.loads(paired_protocol_bytes)
    if paired_protocol["input_manifest"] != protocol["input_manifest"]:
        raise IntegrityError("Input manifest identity differs from paired protocol")
    selected, clean, raw_audit = paired.authenticate(root, paired_protocol)
    by_id = {r.id: r for r in selected}
    clean_ids = {r.id for r in clean}
    memberships = json.loads(gzip.decompress((location / "membership.json.gz").read_bytes()))
    summaries = {r["case"]: r for r in csv.DictReader((location / "summary.csv").open(encoding="utf-8", newline=""))}
    calls = list(csv.DictReader(io.StringIO(gzip.decompress((location / "calls.csv.gz").read_bytes()).decode())))
    floats = ("delta", "auc_q", "auc_l2", "gap_q", "gap_l2", "native_actual", "native_baseline",
              "residual_actual", "residual_baseline")
    lists = ("episode_key", "residual_points", "native_curve", "grid_percent")
    cases = defaultdict(list)
    for row in calls:
        for key in floats:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise IntegrityError(f"Nonfinite saved paired value: {key}")
        for key in lists:
            row[key] = json.loads(row[key])
        for key in ("line", "policy_call_idx"):
            row[key] = int(row[key])
        record = by_id.get(row["record_id"])
        if record is None or row["record_id"] not in clean_ids:
            raise IntegrityError("Saved call is absent from authenticated selected/conflict-excluded population")
        validate_identity(row, record)
        original = record.row
        if not fnmatch.fnmatchcase(Path(record.source).name, paired_protocol["cohorts"][row["ranking"]]):
            raise IntegrityError("Saved call ranking/source mismatch")
        expected_case = f"{row['ranking']}:{row['modality']}:{row['direction']}"
        if row["case"] != expected_case:
            raise IntegrityError("Saved call case mismatch")
        computed = paired.paired_curve(original["k_grid_auc"], original[f"{row['modality']}_{row['direction']}_curve"],
                                       original[f"{row['modality']}_f_input"], original[f"{row['modality']}_f_baseline"],
                                       row["ranking"], dimension=paired_protocol["dimension"],
                                       epsilon=paired_protocol["stabilizer_epsilon"])
        for key in (*floats, "residual_points", "native_curve", "grid_percent", "orientation", "paired_status"):
            if row[key] != computed[key]:
                raise IntegrityError(f"Saved call derived value mismatch: {key}")
        cases[row["case"]].append(row)
    expected_cases = [f"{r}:{m}:{d}" for r in protocol["ranking_order"]
                      for m in protocol["modality_order"] for d in protocol["direction_order"]]
    if set(cases) != set(expected_cases) or set(summaries) != set(expected_cases):
        raise IntegrityError("Influence must retain exactly all eight paired cases")
    for case in expected_cases:
        rows = cases[case]
        ids = sorted(r["record_id"] for r in rows)
        members = memberships[case]
        if ids != members["raw"] or ids != members["defined"] or members["undefined"]:
            raise IntegrityError("Frozen paired defined/raw membership mismatch")
        groups = defaultdict(list)
        for row in rows:
            groups[row["episode_id"]].append(row["record_id"])
        if {k: sorted(v) for k, v in groups.items()} != members["episode_groups"]:
            raise IntegrityError("Frozen episode membership mismatch")
        roster = protocol["roster_per_case"]
        if len(rows) != roster["defined_calls"] or len(groups) != roster["recorded_episodes"] or \
                {len(v) for v in groups.values()} != {roster["calls_per_episode"]}:
            raise IntegrityError("Frozen influence roster mismatch")
        result, _ = summarize(rows)
        if not math.isclose(result["equal_episode_mean"], float(summaries[case]["equal_episode_mean_point"]), rel_tol=1e-13, abs_tol=1e-13):
            raise IntegrityError("Frozen full mean changed")
        if median(r["delta"] for r in rows) != float(summaries[case]["paired_call_median_point"]):
            raise IntegrityError("Frozen paired-call median changed")
    audit = dict(paired_provenance_sha256=sha256(provenance_bytes),
                 paired_artifacts_sha256=provenance["artifacts_sha256"],
                 paired_source_sha256=provenance["source_sha256"],
                 canonical_provenance_sha256=provenance["canonical_provenance_sha256"],
                 input_manifest_sha256=sha256((root / protocol["input_manifest"]).read_bytes()),
                 raw_inputs=raw_audit["inputs"], n_raw_occurrences=raw_audit["n_raw_occurrences"],
                 n_selected_occurrences=raw_audit["n_selected_occurrences"],
                 canonical_verification=raw_audit["canonical_verification"])
    return {case: cases[case] for case in expected_cases}, audit


def validate_identity(row, record):
    expected_episode_key = list(paired.episode_key(record))
    expected = dict(record_id=record.id, source=record.source, source_sha256=record.source_sha256,
                    line=record.line, line_sha256=record.line_sha256,
                    episode_key=expected_episode_key, episode_id=sha256(json_bytes(expected_episode_key)),
                    policy_call_idx=record.row["policy_call_idx"])
    if any(row.get(key) != value for key, value in expected.items()):
        raise IntegrityError("Saved physical/episode identity mismatch")


def table(rows):
    lines = [r"\begin{tabular}{lllrrr}", r"\toprule",
             r"Ranking & Modality & Curve & Full mean & Omission-mean range & Same sign \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['ranking']} & {row['modality']} & {row['direction']} & "
                     f"{row['equal_episode_mean']:.4f} & [{row['omission_mean_min']:.4f}, {row['omission_mean_max']:.4f}] & "
                     f"{row['n_sign_preserved']}/{row['n_omissions']} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  "% Exploratory whole-recorded-episode omissions; ranges are not confidence intervals.",
                  "% Full untrimmed estimands and frozen paired results remain unchanged."])
    return "\n".join(lines) + "\n"


def methods():
    return """# Exploratory episode influence, 2026-10-01-v1

This supplement follows an earlier exploratory audit; the saved protocol is not an outcome-blind preregistration. It preserves all eight paired cases and all original untrimmed means, medians and intervals. Each omission removes one entire recorded episode and then gives every remaining episode equal weight. All 240 omissions retain physical-record membership and hashes. Omission ranges are deterministic influence diagnostics, not confidence intervals or calibration of general-population coverage.

`case_summary.csv` reports the full mean, all-omission extrema, exact sign agreement, and concentration of negative episode means. The negative episode-mean denominator is the sum of magnitudes of negative within-episode means, not the signed net mean or raw call total. `episode_omissions.csv` identifies every omitted episode and its records and hashes the retained population. `tail_diagnostics.json` describes the minimum paired call difference in every case, with ties broken by physical record ID, and top negative-call concentration using the sum of magnitudes of negative call differences. A zero negative-mass denominator yields null, not zero. Weak empirical percentiles are 100 times the fraction of within-case values less than or equal to the example.

The finite squared residuals and normalization gaps retain the historical action-coordinate convention and imply no physical importance threshold. The historical 1e-9 gap cutoff is only a descriptive comparison; no new eligibility rule is applied. All examples remain within their own ranking cohort. Matching nominal labels cannot authenticate a cross-arm context, and recorded episodes do not establish independent randomized or training replications. Saved nodes do not identify unobserved continuous curves or realized masks. These diagnostics establish no ranking efficacy or task-success result.

From the repository root, using the same NumPy dependency as the paired analysis:

```text
python -m analysis.paired_rescoring.influence --output <new-empty-directory>
python -m analysis.paired_rescoring.influence --verify <result-directory>
python -m pytest tests/test_paired_influence.py -q
```

The producer authenticates the frozen paired and canonical source/output records and raw inputs before using saved calls, refuses an existing destination, and records input, source, protocol and output hashes. Reproduction requires no model, simulator, GPU or private file. The new package is nested so the frozen paired producer's top-level source-file enumeration is unchanged.
"""


def run(output, *, root=ROOT):
    output, root = Path(output), Path(root)
    if output.exists():
        raise IntegrityError("Refusing an existing influence output directory")
    protocol_bytes = (root / PROTOCOL_PATH).read_bytes()
    protocol = json.loads(protocol_bytes)
    cases, audit = authenticate(root, protocol)
    summaries, omissions, tails = [], [], {}
    for case, calls in cases.items():
        ranking, modality, direction = case.split(":")
        identity = dict(case=case, ranking=ranking, modality=modality, direction=direction)
        summary, case_omissions = summarize(calls)
        summaries.append(dict(**identity, **summary))
        by_episode = {r["episode_id"]: r for r in calls}
        for row in case_omissions:
            original = by_episode[row["omitted_episode_id"]]
            omissions.append(dict(**identity, omitted_episode_key=original["episode_key"], **row))
        tails[case] = tail_diagnostic(calls, protocol["negative_mass"]["top_call_counts"],
                                     protocol["historical_gap_cutoff_for_description_only"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(protocol_bytes)
    (output / "case_summary.csv").write_bytes(paired.csv_bytes(summaries))
    (output / "episode_omissions.csv").write_bytes(paired.csv_bytes(omissions))
    paired.write_json(output / "tail_diagnostics.json", tails)
    (output / "influence_table.tex").write_text(table(summaries), encoding="utf-8", newline="\n")
    (output / "methods_results.md").write_text(methods(), encoding="utf-8", newline="\n")
    source = Path(__file__).parent
    provenance = dict(schema_version=1, analysis_version=protocol["analysis_version"],
                      interpretation="exploratory_deterministic_influence_not_new_primary_inference",
                      protocol_sha256=sha256(protocol_bytes),
                      source_sha256={p.relative_to(root).as_posix(): sha256(p.read_bytes()) for p in sorted(source.glob("*.py"))},
                      inputs=audit, n_cases=len(summaries), n_episode_omissions=len(omissions),
                      artifacts_sha256={name: sha256((output / name).read_bytes()) for name in OUTPUT_NAMES},
                      environment=dict(python=platform.python_version()),
                      command="python -m analysis.paired_rescoring.influence --output <new-empty-directory>")
    paired.write_json(output / "provenance.json", provenance)
    return provenance


def verify(output, *, root=ROOT):
    output, root = Path(output), Path(root)
    provenance = json.loads((output / "provenance.json").read_bytes())
    if set(provenance["artifacts_sha256"]) != set(OUTPUT_NAMES):
        raise IntegrityError("Influence artifact roster mismatch")
    for name, digest in provenance["artifacts_sha256"].items():
        if sha256((output / name).read_bytes()) != digest:
            raise IntegrityError(f"Influence artifact mismatch: {name}")
    protocol_bytes = (root / PROTOCOL_PATH).read_bytes()
    if sha256(protocol_bytes) != provenance["protocol_sha256"] or (output / "protocol.json").read_bytes() != protocol_bytes:
        raise IntegrityError("Influence protocol mismatch")
    for name, digest in provenance["source_sha256"].items():
        if sha256((root / name).read_bytes()) != digest:
            raise IntegrityError(f"Influence source mismatch: {name}")
    _, audit = authenticate(root, json.loads(protocol_bytes))
    if audit != provenance["inputs"]:
        raise IntegrityError("Influence input provenance mismatch")
    return dict(status="verified", artifacts=len(OUTPUT_NAMES), n_cases=provenance["n_cases"],
                n_episode_omissions=provenance["n_episode_omissions"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--verify", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output) if args.output else verify(args.verify), indent=2))
