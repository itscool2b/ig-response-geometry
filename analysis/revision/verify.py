"""Verify generated artifact hashes and point/interval population contracts."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path

from .core import IntegrityError, json_bytes, sha256


def verify(output):
    provenance = json.loads((output / "provenance.json").read_text())
    for filename, expected in provenance["artifacts_sha256"].items():
        if sha256((output / filename).read_bytes()) != expected:
            raise IntegrityError(f"Artifact hash mismatch: {filename}")
    populations = json.loads(gzip.decompress((output / "population_membership.json.gz").read_bytes()))
    for digest, members in populations.items():
        if len(members) != len(set(members)) or sha256(json_bytes(sorted(members))) != digest:
            raise IntegrityError(f"Population identity mismatch: {digest}")
    ledger = list(csv.DictReader(gzip.decompress((output / "raw_line_ledger.csv.gz").read_bytes()).decode().splitlines()))
    valid_ids = {f"{r['source']}:{r['line']}:{r['line_sha256']}" for r in ledger if r["status"] == "valid"}
    results = list(csv.DictReader((output / "results.csv").open(encoding="utf-8", newline="")))
    keys = set()
    for result in results:
        key = result["result_id"], result["occurrence_view"]
        if key in keys:
            raise IntegrityError(f"Repeated result identity: {key}")
        keys.add(key)
        digest = result["point_population_sha256"]
        if digest != result["ci_population_sha256"]:
            raise IntegrityError(f"Point/CI population mismatch: {key}")
        members = populations[digest]
        if len(members) != int(result["n_source_records"]) or not set(members) <= valid_ids:
            raise IntegrityError(f"Population lineage mismatch: {key}")
        for field in ("point", "ci_lo", "ci_hi"):
            if result[field] and not math.isfinite(float(result[field])):
                raise IntegrityError(f"Nonfinite result: {key}/{field}")
        if result["ci_lo"] and float(result["ci_lo"]) > float(result["ci_hi"]):
            raise IntegrityError(f"Reversed interval: {key}")
    return dict(artifact_hashes=len(provenance["artifacts_sha256"]), results=len(results),
                populations=len(populations), valid_physical_records=len(valid_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))
