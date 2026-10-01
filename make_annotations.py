"""Export identity-bearing numerical diagnostics from authenticated current runs.

These labels describe completeness residuals. They are not behavioral failure
categories, attention quality judgments, or human annotations. The exact old
selection algorithm and its existing CSV remain historical artifacts.
"""
import argparse
import csv
import io
import math
from pathlib import Path

from experiment_io import atomic_bytes, canonical_json, file_hash, object_hash
from faithfulness import authenticated_source, load_sidecar


def categorize_residuals(row):
    values = [row[key] for key in ("vision_err", "lang_err", "state_err")]
    if any(value is None for value in values):
        return "undefined_relative_residual"
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Residuals must be finite nonnegative numbers or explicitly undefined")
    largest = max(values)
    if largest == 0:
        return "zero_reported_residuals"
    if min(values) / largest > .5:
        return "comparable_residuals"
    return ("vision", "language", "state")[values.index(largest)] + "_residual_largest"


def export_annotations(metrics_paths, output):
    output = Path(output).resolve()
    provenance_path = output.with_name(output.name + ".provenance.json")
    if output.exists() or provenance_path.exists():
        raise FileExistsError("Use new annotation and provenance paths")
    rows, sources, seen = [], [], set()
    for metrics in metrics_paths:
        metrics = Path(metrics).resolve()
        steps, manifest = authenticated_source(metrics)
        config = manifest["configuration"]
        sources.append({"metrics": str(metrics), "sha256": file_hash(metrics),
                        "run_id": manifest["run_id"], "configuration_sha256": manifest["configuration_sha256"]})
        for step in steps:
            _, digest = load_sidecar(step, metrics, manifest)
            key = manifest["run_id"], step["context_id"]
            if key in seen:
                raise ValueError("Duplicate run/context in annotation inputs")
            seen.add(key)
            rows.append({"run_id": key[0], "context_id": key[1],
                "configuration_sha256": manifest["configuration_sha256"],
                "checkpoint_identity_sha256": object_hash(config["pipeline"]),
                "task": step["task"], "model": step["model"], "seed": step["seed"],
                "episode": step["episode"], "policy_call_idx": step["policy_call_idx"],
                "target": config["target"], "m": config["m"], "quadrature": config["quadrature"],
                "solver_steps": config["solver_steps"], "sidecar_sha256": digest,
                "numerical_diagnostic": categorize_residuals(step),
                "vision_relative_residual": step["vision_err"], "language_relative_residual": step["lang_err"],
                "state_relative_residual": step["state_err"],
                "qualification": "Completeness accounting only; no behavioral or attribution-quality label"})
    if not rows:
        raise ValueError("No authenticated contexts to annotate")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(buffer.getvalue().encode("utf-8"))
    atomic_bytes(provenance_path, canonical_json({"sources": sources, "rows": len(rows),
        "csv_sha256": file_hash(output), "generator_sha256": file_hash(__file__),
        "selection": "all calls in the explicitly listed authenticated runs",
        "units": "dimensionless relative residual; blank means undefined, not zero"}) + b"\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = export_annotations(args.metrics, args.out)
    print(f"Exported {len(rows)} numerical diagnostics with run/context identity.")


if __name__ == "__main__":
    main()
