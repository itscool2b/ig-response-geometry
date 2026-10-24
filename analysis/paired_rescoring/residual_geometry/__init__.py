"""Baseline residual B, maximum interior residual and paired Delta per saved call.

Descriptive only. The eight headline cases are recomputed from the
authenticated inputs with the frozen ``paired_curve`` and checked for exact
equality against the frozen ``calls.csv.gz``; the RDT-1B cases are recomputed
the same way and checked against the committed oneb_success calls.
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import gzip
import io
import json
from pathlib import Path

from analysis.paired_rescoring import analyze as paired
from analysis.paired_rescoring import curve_stats as cs
from analysis.paired_rescoring import oneb_success as oneb
from analysis.revision.core import IntegrityError, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results" / "2026-10-23-v1"
DATA_OUTPUTS = ("protocol.json", "calls.csv.gz", "correlations.csv", "bins.csv", "checks.json")
FIGURES = ("headline.pdf", "headline.png", "oneb.pdf", "oneb.png")
OUTPUTS = DATA_OUTPUTS + FIGURES
SOURCES = ("analysis/paired_rescoring/residual_geometry/__init__.py",
           "analysis/paired_rescoring/residual_geometry/__main__.py",
           "analysis/paired_rescoring/residual_geometry/figures.py", *oneb.SOURCES)
CALL_FIELDS = ("case", "figure", "cohort", "record_id", "episode_key", "episode_id", "policy_call_idx",
               "first_call", "paired_status", "delta", "auc_q", "auc_l2", "residual_actual", "residual_baseline",
               "max_interior_residual", "ratio_max_interior_to_baseline", "overshoot", "fully_in_range",
               "n_high_distance_overshoot", "n_low_distance_undershoot")
SUBSETS = ("all", "fully_in_range", "overshoot")


def load_protocol(root=ROOT):
    blob = (Path(root) / "analysis/paired_rescoring/residual_geometry/protocol.json").read_bytes()
    return blob, json.loads(blob)


def geometry_row(case, figure, cohort, call):
    baseline = call["residual_baseline"]
    maximum = call["max_interior_residual"]
    return dict(case=case, figure=figure, cohort=cohort, record_id=call["record_id"],
                episode_key=call["episode_key"], episode_id=call["episode_id"],
                policy_call_idx=call["policy_call_idx"], first_call=call["policy_call_idx"] == 0,
                paired_status=call["paired_status"], delta=call["delta"], auc_q=call["auc_q"],
                auc_l2=call["auc_l2"], residual_actual=call["residual_actual"], residual_baseline=baseline,
                max_interior_residual=maximum,
                ratio_max_interior_to_baseline=maximum / baseline if baseline > 0 else None,
                overshoot=cs.any_overshoot(call), fully_in_range=cs.fully_in_range(call),
                n_high_distance_overshoot=call["n_high_distance_overshoot"],
                n_low_distance_undershoot=call["n_low_distance_undershoot"])


def headline(root, protocol):
    """Recompute the eight frozen cases and require exact equality with them."""
    spec = protocol["headline"]
    paired_protocol_bytes = (root / spec["paired_protocol"]["path"]).read_bytes()
    if sha256(paired_protocol_bytes) != spec["paired_protocol"]["sha256"]:
        raise IntegrityError("Frozen paired protocol mismatch")
    provenance_bytes = (root / spec["paired_provenance"]["path"]).read_bytes()
    if sha256(provenance_bytes) != spec["paired_provenance"]["sha256"]:
        raise IntegrityError("Frozen paired provenance mismatch")
    frozen_bytes = (root / spec["frozen_calls"]["path"]).read_bytes()
    if sha256(frozen_bytes) != spec["frozen_calls"]["sha256"] or \
            json.loads(provenance_bytes)["artifacts_sha256"]["calls.csv.gz"] != spec["frozen_calls"]["sha256"]:
        raise IntegrityError("Frozen paired calls mismatch")
    paired_protocol = json.loads(paired_protocol_bytes)
    records, _, audit = paired.authenticate(root, paired_protocol)
    frozen = {}
    for row in csv.DictReader(io.StringIO(gzip.decompress(frozen_bytes).decode())):
        frozen[(row["case"], row["record_id"])] = row
    rows, compared, equal = [], 0, dict(delta=0, residual_baseline=0, residual_points=0)
    for ranking, pattern in paired_protocol["cohorts"].items():
        cohort = sorted((r for r in records if fnmatch.fnmatchcase(Path(r.source).name, pattern)), key=lambda r: r.id)
        for modality in paired_protocol["modalities"]:
            for direction in paired_protocol["directions"]:
                case = f"{ranking}:{modality}:{direction}"
                for record in cohort:
                    call = cs.rescore(record, modality, direction, native=ranking,
                                      dimension=paired_protocol["dimension"],
                                      epsilon=paired_protocol["stabilizer_epsilon"])
                    old = frozen.get((case, record.id))
                    if old is None:
                        raise IntegrityError(f"Recomputed call absent from the frozen calls: {case}")
                    compared += 1
                    equal["delta"] += float(old["delta"]) == call["delta"]
                    equal["residual_baseline"] += float(old["residual_baseline"]) == call["residual_baseline"]
                    equal["residual_points"] += json.loads(old["residual_points"]) == call["residual_points"]
                    rows.append(geometry_row(case, "headline", ranking, call))
    if compared != len(frozen) or any(v != compared for v in equal.values()):
        raise IntegrityError(f"Recomputed headline calls differ from the frozen calls: {compared}, {equal}")
    check = dict(frozen_calls=spec["frozen_calls"]["path"], frozen_calls_sha256=spec["frozen_calls"]["sha256"],
                 n_frozen_rows=len(frozen), n_recomputed=compared,
                 **{f"n_exact_{k}": v for k, v in equal.items()},
                 n_raw_occurrences=audit["n_raw_occurrences"], n_selected_occurrences=audit["n_selected_occurrences"])
    raw = [dict(path=item["path"], sha256=item["sha256"], checkout_sha256=sha256((root / item["path"]).read_bytes()))
           for item in audit["inputs"]]
    return rows, check, raw


def oneb_cases(root, protocol):
    """Recompute the primary RDT-1B calls; require exact equality with analysis 1."""
    spec = protocol["oneb"]
    oneb_bytes, oneb_protocol = oneb.load_protocol(root)
    if sha256(oneb_bytes) != spec["protocol_sha256"]:
        raise IntegrityError("oneb_success protocol mismatch")
    committed_bytes = (root / spec["committed_calls"]["path"]).read_bytes()
    if sha256(committed_bytes) != spec["committed_calls"]["sha256"]:
        raise IntegrityError("Committed oneb_success calls mismatch")
    committed = {(r["record_id"], r["modality"], r["direction"]): r for r in
                 csv.DictReader(io.StringIO(gzip.decompress(committed_bytes).decode()))}
    selected, audit = oneb.authenticate(root, oneb_protocol)
    rows, compared, equal = [], 0, 0
    for case in spec["cases"]:
        for label in case["runs"]:
            for record in selected[label]["faith"]:
                call = cs.rescore(record, case["modality"], case["direction"], native=oneb_protocol["native_scale"],
                                  dimension=oneb_protocol["dimension"], epsilon=oneb_protocol["stabilizer_epsilon"])
                old = committed[(record.id, case["modality"], case["direction"])]
                compared += 1
                equal += float(old["delta"]) == call["delta"]
                rows.append(geometry_row(case["case"], "oneb", label, call))
    if equal != compared:
        raise IntegrityError("Recomputed RDT-1B calls differ from the committed oneb_success calls")
    raw = [dict(path=f["path"], sha256=f["sha256"], checkout_sha256=f["checkout_sha256"]) for f in audit["files"]]
    return rows, dict(committed_calls=spec["committed_calls"]["path"], n_recomputed=compared, n_exact_delta=equal), raw


def subset(rows, name):
    defined = [r for r in rows if r["paired_status"] == "defined"]
    if name == "all":
        return defined
    if name == "fully_in_range":
        return [r for r in defined if r["fully_in_range"]]
    return [r for r in defined if r["overshoot"]]


def correlations(rows, cases):
    out = []
    for case in cases:
        members = [r for r in rows if r["case"] == case]
        for name in SUBSETS:
            chosen = subset(members, name)
            ratio = [r for r in chosen if r["ratio_max_interior_to_baseline"] is not None]
            out.append(dict(case=case, subset=name, n_calls=len(chosen),
                            spearman_delta_vs_B=cs.spearman([r["delta"] for r in chosen],
                                                            [r["residual_baseline"] for r in chosen]),
                            n_calls_with_ratio=len(ratio),
                            spearman_delta_vs_ratio=cs.spearman([r["delta"] for r in ratio],
                                                                [r["ratio_max_interior_to_baseline"] for r in ratio])))
    return out


def bin_label(value, edges, closed):
    """Label of value among string edges; closed='left' gives [a,b), 'right' (a,b]."""
    numbers = [float(e) for e in edges]
    if closed == "left":
        if value < numbers[0]:
            return f"<{edges[0]}"
        for (a, x), (b, y) in zip(zip(edges, numbers), zip(edges[1:], numbers[1:])):
            if x <= value < y:
                return f"[{a},{b})"
        return f">={edges[-1]}"
    if value <= numbers[0]:
        return f"<={edges[0]}"
    for (a, x), (b, y) in zip(zip(edges, numbers), zip(edges[1:], numbers[1:])):
        if x < value <= y:
            return f"({a},{b}]"
    return f">{edges[-1]}"


def bin_order(edges, closed):
    if closed == "left":
        return [f"<{edges[0]}", *(f"[{a},{b})" for a, b in zip(edges, edges[1:])), f">={edges[-1]}"]
    return [f"<={edges[0]}", *(f"({a},{b}]" for a, b in zip(edges, edges[1:])), f">{edges[-1]}"]


def bin_stats(members):
    values = [r["delta"] for r in members]
    return dict(n_calls=len(members), n_first_calls=sum(r["first_call"] for r in members),
                n_overshoot_calls=sum(r["overshoot"] for r in members),
                n_fully_in_range_calls=sum(r["fully_in_range"] for r in members),
                n_negative_delta=sum(v < 0 for v in values), n_delta_below_minus_1=sum(v < -1 for v in values),
                n_delta_below_minus_100=sum(v < -100 for v in values),
                min_delta=min(values) if values else None, median_delta=cs.median(values) if values else None,
                max_delta=max(values) if values else None)


def bins(rows, cases, protocol):
    spec = protocol["bins"]
    b_edges, r_edges = spec["B_edges"], spec["ratio_edges"]
    out = []
    for case in cases:
        members = [r for r in rows if r["case"] == case and r["paired_status"] == "defined"]
        if any(r["ratio_max_interior_to_baseline"] is None for r in members):
            raise IntegrityError("A defined call has a zero baseline residual")
        b_of = {r["record_id"]: bin_label(r["residual_baseline"], b_edges, "left") for r in members}
        r_of = {r["record_id"]: bin_label(r["ratio_max_interior_to_baseline"], r_edges, "right") for r in members}
        for label in bin_order(b_edges, "left"):
            chosen = [r for r in members if b_of[r["record_id"]] == label]
            out.append(dict(case=case, binning="B", B_bin=label, ratio_bin="", **bin_stats(chosen)))
        for label in bin_order(r_edges, "right"):
            chosen = [r for r in members if r_of[r["record_id"]] == label]
            out.append(dict(case=case, binning="ratio", B_bin="", ratio_bin=label, **bin_stats(chosen)))
        for b_label in bin_order(b_edges, "left"):
            for r_label in bin_order(r_edges, "right"):
                chosen = [r for r in members if b_of[r["record_id"]] == b_label and r_of[r["record_id"]] == r_label]
                if chosen:
                    out.append(dict(case=case, binning="joint", B_bin=b_label, ratio_bin=r_label, **bin_stats(chosen)))
    return out


def compute(root, protocol):
    head, head_check, head_inputs = headline(root, protocol)
    one, one_check, one_inputs = oneb_cases(root, protocol)
    rows = head + one
    cases = [*protocol["headline"]["cases"], *(c["case"] for c in protocol["oneb"]["cases"])]
    if [c for c in dict.fromkeys(r["case"] for r in rows)] != cases:
        raise IntegrityError("Case roster differs from the protocol")
    checks = dict(headline=head_check, oneb=one_check,
                  n_calls_by_case={c: sum(r["case"] == c for r in rows) for c in cases},
                  n_zero_baseline_calls=sum(r["residual_baseline"] == 0 for r in rows),
                  n_undefined_calls=sum(r["paired_status"] != "defined" for r in rows))
    declared = [dict(path=item["path"], sha256=item["sha256"]) for item in (
        protocol["headline"]["paired_protocol"], protocol["headline"]["paired_provenance"],
        protocol["headline"]["frozen_calls"], protocol["oneb"]["committed_calls"])]
    declared.append(dict(path="analysis/paired_rescoring/oneb_success/protocol.json", sha256=protocol["oneb"]["protocol_sha256"]))
    inputs = dict(declared=declared, headline_raw=head_inputs, oneb_raw=one_inputs)
    return dict(rows=rows, cases=cases, checks=checks, inputs=inputs,
                correlations=correlations(rows, cases), bins=bins(rows, cases, protocol))


def source_hashes(root):
    return {name: sha256((Path(root) / name).read_bytes()) for name in SOURCES}


def run(output, *, root=ROOT):
    from analysis.paired_rescoring.residual_geometry import figures
    output, root = Path(output), Path(root)
    if output.exists():
        raise IntegrityError("Refusing an existing output directory")
    protocol_bytes, protocol = load_protocol(root)
    result = compute(root, protocol)
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    (output / "calls.csv.gz").write_bytes(cs.gzip_bytes(cs.csv_bytes(result["rows"], CALL_FIELDS)))
    (output / "correlations.csv").write_bytes(cs.csv_bytes(result["correlations"]))
    (output / "bins.csv").write_bytes(cs.csv_bytes(result["bins"]))
    cs.write_json(output / "checks.json", result["checks"])
    figure_info = figures.render(result["rows"], protocol, output)
    provenance = dict(
        schema_version=1, analysis=protocol["analysis"], analysis_version=protocol["analysis_version"],
        protocol_sha256=sha256(protocol_bytes), source_sha256=source_hashes(root), inputs=result["inputs"],
        artifacts={name: cs.artifact_record(output / name) for name in OUTPUTS},
        png_pixels_sha256={name: figures.png_pixels_sha256(output / name) for name in FIGURES if name.endswith(".png")},
        figure_environment=figure_info, environment=oneb.environment(),
        command="python -m analysis.paired_rescoring.residual_geometry --output <new-empty-directory>",
        verify_command="python -m analysis.paired_rescoring.residual_geometry --verify <result-directory>",
        interpretation="descriptive_geometry_no_causal_claim")
    cs.write_json(output / "provenance.json", provenance)
    return provenance


def verify(output=RESULTS, *, root=ROOT):
    output, root = Path(output), Path(root)
    provenance = json.loads((output / "provenance.json").read_bytes())
    if set(provenance["artifacts"]) != set(OUTPUTS):
        raise IntegrityError("Artifact roster mismatch")
    for name, record in provenance["artifacts"].items():
        if cs.artifact_record(output / name) != record:
            raise IntegrityError(f"Artifact mismatch: {name}")
    protocol_bytes, protocol = load_protocol(root)
    if sha256(protocol_bytes) != provenance["protocol_sha256"] or (output / "protocol.json").read_bytes() != protocol_bytes:
        raise IntegrityError("Protocol mismatch")
    if source_hashes(root) != provenance["source_sha256"]:
        raise IntegrityError("Source mismatch")
    result = compute(root, protocol)
    if json.loads(cs.json_text(result["inputs"])) != provenance["inputs"]:
        raise IntegrityError("Input provenance mismatch")
    for name, blob in (("calls.csv.gz", cs.csv_bytes(result["rows"], CALL_FIELDS)),
                       ("correlations.csv", cs.csv_bytes(result["correlations"])),
                       ("bins.csv", cs.csv_bytes(result["bins"])),
                       ("checks.json", cs.json_text(result["checks"]).encode())):
        stored = (output / name).read_bytes()
        if (gzip.decompress(stored) if name.endswith(".gz") else stored) != blob:
            raise IntegrityError(f"Recomputed artifact differs: {name}")
    return dict(status="verified", artifacts=len(OUTPUTS), analysis_version=provenance["analysis_version"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--verify", type=Path)
    args = parser.parse_args()
    print(cs.json_text(run(args.output) if args.output else verify(args.verify)), end="")
