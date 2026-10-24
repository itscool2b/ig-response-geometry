"""Completeness residuals of the saved rankings and the endpoint example (CPU only).

(a) For each faithfulness cohort, its own step file is selected by the
provenance fingerprint among every declared candidate; residuals
|ig_sum - gap| / |gap| are summarized on the step rows of the rescored calls,
with a bf16 storage-rounding analysis of the stored sums.
(b) The reviewer's one-dimensional endpoint example (endpoint_example.py).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from analysis.completeness_audit import endpoint_example
from analysis.paired_rescoring import curve_stats as cs
from analysis.paired_rescoring import oneb_success as oneb
from analysis.revision.core import IntegrityError, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results" / "2026-10-23-v1"
OUTPUTS = ("protocol.json", "step_selection.json", "residuals.csv", "bf16.csv", "calls.csv.gz",
           "endpoint_example.csv", "endpoint_identity.csv")
PLATFORM_OUTPUTS = ("endpoint_numpy_leggauss.json",)
SOURCES = ("analysis/completeness_audit/__init__.py", "analysis/completeness_audit/__main__.py",
           "analysis/completeness_audit/endpoint_example.py",
           "analysis/paired_rescoring/curve_stats/__init__.py", "analysis/paired_rescoring/analyze.py",
           "analysis/revision/core.py", "analysis/revision/analyze.py", "analysis/revision/verify.py")
MODALITIES = ("vision", "lang", "state")
BF16_RELATIVE_BOUND = 2.0 ** -8


# --- bf16 arithmetic on Python floats ---------------------------------------

def bf16_round(x):
    """Round a float to the nearest bfloat16 value (8 significant bits, ties to even)."""
    if x == 0 or not math.isfinite(x):
        if not math.isfinite(x):
            raise IntegrityError("Nonfinite bf16 input")
        return x
    mantissa, exponent = math.frexp(x)
    if not -125 <= exponent <= 128:
        raise IntegrityError("Value outside the normal bf16 range")
    return math.ldexp(round(math.ldexp(mantissa, 8)), exponent - 8)


def bf16_half_spacings(v):
    """Half-distances from a bf16 value to its neighbors (toward zero, away from zero)."""
    if bf16_round(v) != v or v == 0:
        raise IntegrityError("Expected a nonzero bf16 value")
    mantissa, exponent = math.frexp(abs(v))
    k = int(math.ldexp(mantissa, 8))
    spacing = math.ldexp(1.0, exponent - 8)
    below = spacing / 2 if k == 128 else spacing
    return below / 2, spacing / 2


def rounding_interval(v):
    toward, away = bf16_half_spacings(v)
    return (v - toward, v + away) if v > 0 else (v - away, v + toward)


def fp32_exact(x):
    import numpy as np
    return float(np.float32(x)) == x


# --- inputs and step selection ----------------------------------------------

def load_protocol(root=ROOT):
    blob = (Path(root) / "analysis/completeness_audit/protocol.json").read_bytes()
    return blob, json.loads(blob)


def declared_files(protocol):
    names = {}
    for cohort in protocol["cohorts"]:
        for item in cohort["faithfulness"] + cohort["candidate_steps"]:
            names[item["path"]] = item["sha256"]
    for population in protocol["comparison_populations"]:
        for item in population["steps"] + population["fingerprint_against"]:
            names[item["path"]] = item["sha256"]
    return names


def select_steps(cohort, loaded):
    """A unique full general-form match selects; ambiguity needs a declared primary."""
    declared = cohort.get("declared_primary_when_ambiguous", {})
    selections = []
    for item in cohort["faithfulness"]:
        faith = cs.faithfulness_rows(loaded[item["path"]]["last"])
        matrix = []
        for candidate in cohort["candidate_steps"]:
            mark = cs.fingerprint(faith, loaded[candidate["path"]]["last"])
            matrix.append(dict(steps=candidate["path"], **mark))
        qualified = [m["steps"] for m in matrix if m["all_agree_general"]]
        declaration = declared.get(item["path"])
        if len(qualified) == 1 and declaration is None:
            status, selected = "fingerprint_unique", qualified[0]
        elif len(qualified) > 1 and declaration is not None and declaration["steps"] in qualified:
            status, selected = "fingerprint_ambiguous_declared_primary", declaration["steps"]
        else:
            raise IntegrityError(f"Own step file is not determined for {item['path']}: {qualified}")
        selections.append(dict(faithfulness=item["path"], n_faithfulness=len(faith), qualified=qualified,
                               selected=selected, selection_status=status,
                               alternatives=[q for q in qualified if q != selected],
                               declared_basis=declaration["basis"] if declaration else None, candidates=matrix))
    return selections


def joined_rows(faith_records, step_records):
    index = cs.step_index(step_records)
    out = []
    for record in faith_records:
        step = index.get(tuple(record.row[k] for k in cs.STEP_KEY))
        if step is None:
            raise IntegrityError("Selected step file lacks a rescored call")
        out.append(step)
    return out


def call_rows(cohort_id, step_file, steps):
    rows = []
    for record in steps:
        for modality in MODALITIES:
            gap, total = record.row[f"{modality}_gap"], record.row[f"{modality}_ig_sum"]
            defined = gap != 0
            row = dict(cohort=cohort_id, steps=step_file, record_id=record.id,
                       key=[record.row[k] for k in cs.STEP_KEY], modality=modality, gap=gap, ig_sum=total,
                       stored_err=record.row[f"{modality}_err"],
                       residual=abs(total - gap) / abs(gap) if defined else None,
                       relative_signed_error=(total - gap) / gap if defined else None,
                       ig_sum_bf16_exact=bf16_round(total) == total, gap_bf16_exact=bf16_round(gap) == gap,
                       ig_sum_fp32_exact=fp32_exact(total), gap_fp32_exact=fp32_exact(gap))
            if defined and total != 0 and row["ig_sum_bf16_exact"]:
                low, high = rounding_interval(total)
                toward, away = bf16_half_spacings(total)
                distance = 0.0 if low <= gap <= high else min(abs(gap - low), abs(gap - high))
                row.update(bf16_of_gap=bf16_round(gap), rounding_explains=bf16_round(gap) == total,
                           storage_bound=max(toward, away) / abs(gap),
                           adjusted_residual=distance / abs(gap),
                           definite_shortfall=(high < gap) if gap > 0 else (low > gap),
                           definite_excess=(low > gap) if gap > 0 else (high < gap),
                           rounding_only_shortfall=(bf16_round(gap) - gap) / gap < 0)
            rows.append(row)
    return rows


# --- summaries --------------------------------------------------------------

def residual_summary(rows):
    defined = [r for r in rows if r["residual"] is not None]
    values = sorted(r["residual"] for r in defined)
    signed = [r["relative_signed_error"] for r in defined]
    n = len(defined)
    out = dict(n_rows=len(rows), n_undefined_zero_gap=len(rows) - n)
    if not n:
        return out
    out.update(median=cs.percentile(values, 50), p05=cs.percentile(values, 5), p95=cs.percentile(values, 95),
               maximum=values[-1], share_over_1pct=sum(v > 0.01 for v in values) / n,
               share_over_3pct=sum(v > 0.03 for v in values) / n,
               shortfall_share=sum(v < 0 for v in signed) / n, excess_share=sum(v > 0 for v in signed) / n,
               exact_share=sum(v == 0 for v in signed) / n,
               ig_sum_below_gap_share=sum(r["ig_sum"] < r["gap"] for r in defined) / n,
               n_negative_gap=sum(r["gap"] < 0 for r in defined),
               population_sha256=sha256(json.dumps(sorted(r["record_id"] for r in rows)).encode()))
    return out


def bf16_summary(rows):
    defined = [r for r in rows if r["residual"] is not None]
    analysed = [r for r in defined if "storage_bound" in r]
    out = dict(n_rows=len(defined), n_ig_sum_bf16_exact=sum(r["ig_sum_bf16_exact"] for r in defined),
               n_gap_bf16_exact=sum(r["gap_bf16_exact"] for r in defined),
               n_ig_sum_fp32_exact=sum(r["ig_sum_fp32_exact"] for r in defined),
               n_gap_fp32_exact=sum(r["gap_fp32_exact"] for r in defined),
               bf16_relative_rounding_bound=BF16_RELATIVE_BOUND, n_analysed=len(analysed))
    if not analysed:
        return out
    bounds = sorted(r["storage_bound"] for r in analysed)
    adjusted = sorted(r["adjusted_residual"] for r in analysed)
    n = len(analysed)
    beyond = [r for r in analysed if not r["rounding_explains"]]
    out.update(storage_bound_median=cs.percentile(bounds, 50), storage_bound_maximum=bounds[-1],
               n_rounding_explains=n - len(beyond), share_rounding_explains=(n - len(beyond)) / n,
               adjusted_residual_median=cs.percentile(adjusted, 50), adjusted_residual_p95=cs.percentile(adjusted, 95),
               adjusted_share_over_1pct=sum(v > 0.01 for v in adjusted) / n,
               observed_shortfall_share=sum(r["relative_signed_error"] < 0 for r in analysed) / n,
               rounding_only_shortfall_share=sum(r["rounding_only_shortfall"] for r in analysed) / n,
               definite_shortfall_share=sum(r["definite_shortfall"] for r in analysed) / n,
               definite_excess_share=sum(r["definite_excess"] for r in analysed) / n,
               shortfall_share_beyond_rounding=sum(r["relative_signed_error"] < 0 for r in beyond) / len(beyond)
               if beyond else None)
    return out


def groupings(cohort, selections):
    groups = [("pooled", [s["selected"] for s in selections])]
    if len(selections) > 1:
        groups += [(Path(s["selected"]).name, [s["selected"]]) for s in selections]
    groups += [(f"alternative:{Path(a).name}", [a]) for s in selections for a in s["alternatives"]]
    return groups


def compute(root, protocol):
    root = Path(root)
    loaded, files, anchor = cs.authenticate_named(root, list(declared_files(protocol)), expected=declared_files(protocol))
    if anchor["input_manifest_sha256"] != protocol["anchors"]["input_manifest_sha256"]:
        raise IntegrityError("Input manifest differs from the protocol anchor")
    selection, calls, residuals, bf16 = {}, [], [], []
    for cohort in protocol["cohorts"]:
        chosen = select_steps(cohort, loaded)
        selection[cohort["id"]] = chosen
        per_file = {}
        for item, pick in zip(cohort["faithfulness"], chosen):
            faith = cs.faithfulness_rows(loaded[item["path"]]["last"])
            for role, name in [("selected", pick["selected"])] + [("alternative_qualified", a) for a in pick["alternatives"]]:
                steps = joined_rows(faith, loaded[name]["last"])
                clean_ids = {r.id for r in loaded[name]["clean"]}
                rows = call_rows(cohort["id"], name, steps)
                for row in rows:
                    row["selection_role"] = role
                    row["conflict_excluded_member"] = row["record_id"] in clean_ids
                per_file[name] = rows
                calls.extend(rows)
        for label, members in groupings(cohort, chosen):
            group_rows = [r for name in members for r in per_file[name]]
            for modality in MODALITIES:
                rows = [r for r in group_rows if r["modality"] == modality]
                identity = dict(cohort=cohort["id"], grouping=label, modality=modality)
                residuals.append(dict(identity, occurrence_view="last", **residual_summary(rows)))
                clean = [r for r in rows if r["conflict_excluded_member"]]
                if len(clean) != len(rows):
                    residuals.append(dict(identity, occurrence_view="conflict_excluded", **residual_summary(clean)))
                bf16.append(dict(identity, **bf16_summary(rows)))
    comparison = []
    for population in protocol["comparison_populations"]:
        steps = [r for item in population["steps"] for r in loaded[item["path"]]["last"] if r.row["event"] == "step"]
        marks = []
        for item in population["fingerprint_against"]:
            faith = cs.faithfulness_rows(loaded[item["path"]]["last"])
            for step_item in population["steps"]:
                marks.append(dict(faithfulness=item["path"], steps=step_item["path"],
                                  **cs.fingerprint(faith, loaded[step_item["path"]]["last"])))
        rows = [r for item in population["steps"] for r in call_rows(
            population["id"], item["path"], [s for s in loaded[item["path"]]["last"] if s.row["event"] == "step"])]
        for row in rows:
            row["selection_role"] = "comparison"
            row["conflict_excluded_member"] = None
        calls.extend(rows)
        for modality in MODALITIES:
            members = [r for r in rows if r["modality"] == modality]
            identity = dict(cohort=population["id"], grouping="pooled", modality=modality)
            residuals.append(dict(identity, occurrence_view="last", **residual_summary(members)))
            bf16.append(dict(identity, **bf16_summary(members)))
        comparison.append(dict(id=population["id"], steps=[i["path"] for i in population["steps"]],
                               n_step_rows=len(steps), fingerprints=marks))
    audit = dict(anchor=anchor, files=files, selection=selection, comparison_populations=comparison)
    rows, identity, numpy_part = endpoint_example.compute(protocol)
    return dict(audit=audit, calls=calls, residuals=residuals, bf16=bf16,
                endpoint_rows=rows, endpoint_identity=identity, endpoint_numpy=numpy_part)


def source_hashes(root):
    return {name: sha256((Path(root) / name).read_bytes()) for name in SOURCES}


def run(output, *, root=ROOT):
    output, root = Path(output), Path(root)
    if output.exists():
        raise IntegrityError("Refusing an existing output directory")
    protocol_bytes, protocol = load_protocol(root)
    result = compute(root, protocol)
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    cs.write_json(output / "step_selection.json", result["audit"])
    (output / "residuals.csv").write_bytes(cs.csv_bytes(result["residuals"]))
    (output / "bf16.csv").write_bytes(cs.csv_bytes(result["bf16"]))
    (output / "calls.csv.gz").write_bytes(cs.gzip_bytes(cs.csv_bytes(result["calls"])))
    (output / "endpoint_example.csv").write_bytes(cs.csv_bytes(result["endpoint_rows"]))
    (output / "endpoint_identity.csv").write_bytes(cs.csv_bytes(result["endpoint_identity"]))
    cs.write_json(output / "endpoint_numpy_leggauss.json", result["endpoint_numpy"])
    provenance = dict(
        schema_version=1, analysis=protocol["analysis"], analysis_version=protocol["analysis_version"],
        protocol_sha256=sha256(protocol_bytes), source_sha256=source_hashes(root),
        inputs=[dict(path=f["path"], sha256=f["sha256"], checkout_sha256=f["checkout_sha256"])
                for f in result["audit"]["files"]],
        artifacts={name: cs.artifact_record(output / name) for name in OUTPUTS},
        platform_dependent_artifacts={name: cs.artifact_record(output / name) for name in PLATFORM_OUTPUTS},
        environment=dict(oneb.environment(), mpmath=endpoint_example.mp.__version__),
        command="python -m analysis.completeness_audit --output <new-empty-directory>",
        verify_command="python -m analysis.completeness_audit --verify <result-directory>",
        interpretation="retrospective_descriptive_numerical_audit_no_ranking_accuracy_certificate")
    cs.write_json(output / "provenance.json", provenance)
    return provenance


def verify(output=RESULTS, *, root=ROOT):
    output, root = Path(output), Path(root)
    provenance = json.loads((output / "provenance.json").read_bytes())
    if set(provenance["artifacts"]) != set(OUTPUTS):
        raise IntegrityError("Artifact roster mismatch")
    for name, record in {**provenance["artifacts"], **provenance["platform_dependent_artifacts"]}.items():
        if cs.artifact_record(output / name) != record:
            raise IntegrityError(f"Artifact mismatch: {name}")
    protocol_bytes, protocol = load_protocol(root)
    if sha256(protocol_bytes) != provenance["protocol_sha256"] or (output / "protocol.json").read_bytes() != protocol_bytes:
        raise IntegrityError("Protocol mismatch")
    if source_hashes(root) != provenance["source_sha256"]:
        raise IntegrityError("Source mismatch")
    loaded, files, anchor = cs.authenticate_named(root, list(declared_files(protocol)), expected=declared_files(protocol))
    recorded = json.loads((output / "step_selection.json").read_bytes())
    selection = {c["id"]: select_steps(c, loaded) for c in protocol["cohorts"]}
    if recorded["selection"] != json.loads(cs.json_text(selection)) or recorded["files"] != json.loads(cs.json_text(files)):
        raise IntegrityError("Step selection or input provenance mismatch")
    return dict(status="verified", artifacts=len(OUTPUTS) + len(PLATFORM_OUTPUTS),
                analysis_version=provenance["analysis_version"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--verify", type=Path)
    args = parser.parse_args()
    print(cs.json_text(run(args.output) if args.output else verify(args.verify)), end="")
