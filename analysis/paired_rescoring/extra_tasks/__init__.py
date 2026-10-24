"""PegInsertionSide and PickSingleYCB (RDT-170M, m=128) paired rescoring.

The run machinery is the oneb_success package's, applied to this package's own
protocol with a single outcome group: every saved episode of these runs failed.
The two pre-fix seed-42 m=128 faithfulness files are excluded by name; their
fingerprints are computed only to document the exclusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.paired_rescoring import curve_stats as cs
from analysis.paired_rescoring import oneb_success as base
from analysis.revision.core import IntegrityError, sha256

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results" / "2026-10-23-v1"
OUTPUTS = base.OUTPUTS
SOURCES = ("analysis/paired_rescoring/extra_tasks/__init__.py",
           "analysis/paired_rescoring/extra_tasks/__main__.py", *base.SOURCES)


def load_protocol(root=ROOT):
    blob = (Path(root) / "analysis/paired_rescoring/extra_tasks/protocol.json").read_bytes()
    return blob, json.loads(blob)


def cross_run_fingerprint(faith, steps):
    """Fingerprint ignoring the seed label: (task, model, episode, call) only."""
    index = {}
    for record in steps:
        if record.row["event"] == "step":
            index[tuple(record.row[k] for k in ("task", "model", "episode", "policy_call_idx"))] = record
    joined = agree = 0
    for record in faith:
        step = index.get(tuple(record.row[k] for k in ("task", "model", "episode", "policy_call_idx")))
        if step is None:
            continue
        joined += 1
        agree += all(cs.relative_close(step.row[f"{m}_gap"], record.row[f"{m}_f_input"] - record.row[f"{m}_f_baseline"])
                     for m in ("vision", "lang"))
    return dict(n_faithfulness=len(faith), n_joined=joined, n_agree_vision_and_lang=agree)


def exclusions(root, protocol):
    """Document each excluded file: own-step and cross-run fingerprints."""
    names, expected = [], {}
    for item in protocol["excluded"]:
        for kind in ("faithfulness", "nominal_steps", "clean_steps_same_task"):
            names.append(item[kind]["path"])
            expected[item[kind]["path"]] = item[kind]["sha256"]
    loaded, files, _ = cs.authenticate_named(root, list(dict.fromkeys(names)), expected=expected)
    out = []
    for item in protocol["excluded"]:
        faith = cs.faithfulness_rows(loaded[item["faithfulness"]["path"]]["last"])
        own = cs.fingerprint(faith, loaded[item["nominal_steps"]["path"]]["last"])
        other = cross_run_fingerprint(faith, loaded[item["clean_steps_same_task"]["path"]]["last"])
        out.append(dict(faithfulness=item["faithfulness"]["path"], nominal_steps=item["nominal_steps"]["path"],
                        clean_steps_same_task=item["clean_steps_same_task"]["path"], reason=item["reason"],
                        own_step_fingerprint=own, fingerprint_against_clean_steps_same_task=other,
                        rescored=False))
        if own["vision_general_agree"] or own["lang_general_agree"]:
            raise IntegrityError(f"Excluded file unexpectedly matches its own steps: {item['faithfulness']['path']}")
    return out, files


def compute(root, protocol):
    result = base.compute(root, protocol)
    excluded, files = exclusions(root, protocol)
    result["audit"]["excluded"] = excluded
    result["audit"]["excluded_files"] = files
    return result


def run(output, *, root=ROOT):
    output, root = Path(output), Path(root)
    if output.exists():
        raise IntegrityError("Refusing an existing output directory")
    protocol_bytes, protocol = load_protocol(root)
    result = compute(root, protocol)
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(protocol_bytes)
    cs.write_json(output / "inputs.json", result["audit"])
    (output / "calls.csv.gz").write_bytes(cs.gzip_bytes(cs.csv_bytes(result["calls"], base.CALL_FIELDS)))
    (output / "episodes.csv").write_bytes(cs.csv_bytes(result["episodes"], base.EPISODE_FIELDS))
    (output / "summary.csv").write_bytes(cs.csv_bytes(result["summaries"], base.SUMMARY_FIELDS))
    (output / "call_index.csv").write_bytes(cs.csv_bytes(result["indices"], base.INDEX_FIELDS))
    provenance = dict(
        schema_version=1, analysis=protocol["analysis"], analysis_version=protocol["analysis_version"],
        protocol_sha256=sha256(protocol_bytes),
        source_sha256={name: sha256((root / name).read_bytes()) for name in SOURCES},
        inputs=[dict(path=f["path"], sha256=f["sha256"], checkout_sha256=f["checkout_sha256"])
                for f in result["audit"]["files"] + result["audit"]["excluded_files"]],
        artifacts={name: cs.artifact_record(output / name) for name in OUTPUTS},
        environment=base.environment(),
        command="python -m analysis.paired_rescoring.extra_tasks --output <new-empty-directory>",
        verify_command="python -m analysis.paired_rescoring.extra_tasks --verify <result-directory>",
        interpretation="retrospective_descriptive_response_only_rescoring_no_efficacy_claim")
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
    if {name: sha256((root / name).read_bytes()) for name in SOURCES} != provenance["source_sha256"]:
        raise IntegrityError("Source mismatch")
    _, audit = base.authenticate(root, protocol)
    excluded, files = exclusions(root, protocol)
    recorded = json.loads((output / "inputs.json").read_bytes())
    observed = json.loads(cs.json_text(dict(audit, excluded=excluded, excluded_files=files)))
    if {k: recorded[k] for k in observed} != observed:
        raise IntegrityError("Input provenance mismatch")
    return dict(status="verified", artifacts=len(OUTPUTS), analysis_version=provenance["analysis_version"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--output", type=Path)
    operation.add_argument("--verify", type=Path)
    args = parser.parse_args()
    print(cs.json_text(run(args.output) if args.output else verify(args.verify)), end="")
