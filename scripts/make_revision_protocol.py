"""Assemble a revision GPU protocol from a committed template and frozen banks.

The template fixes every numerical, ranking and evaluation setting. This adds
each bank's file hash and the hashes of the exact source tree that will run
it, so a worker refuses any other bank or code. Banks hold collection data
only, no attribution outcome, so binding them here keeps the protocol ahead
of every result.

Usage: python scripts/make_revision_protocol.py --template T.json --banks BANKS.json --out PROTOCOL.json
BANKS.json maps a bank id to its settings plus "code_tree", the directory the
bank's workers run from.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revision_numerics import PROTOCOL_SCHEMA, SOURCE_FILES  # noqa: E402  (imports no torch)

BANK_FIELDS = {"metrics", "bank", "task", "model", "lang_dir", "modalities", "fp32_contexts", "adaptive_contexts",
               "full_node_contexts", "full_node_modalities", "source_replay_settings", "code_tree"}
OPTIONAL_FIELDS = {"checkpoint_path", "verify_contexts", "note", "numerics_override"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(template, banks):
    protocol = dict(template)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Template has the wrong schema")
    protocol["banks"] = {}
    for bank_id, spec in banks.items():
        missing, extra = BANK_FIELDS - set(spec), set(spec) - BANK_FIELDS - OPTIONAL_FIELDS
        if missing or extra:
            raise ValueError(f"Bank {bank_id}: missing {sorted(missing)}, unknown {sorted(extra)}")
        entry = {key: value for key, value in spec.items() if key != "code_tree"}
        entry["bank_sha256"] = sha256(spec["bank"])
        entry["source_sha256"] = {name: sha256(Path(spec["code_tree"]) / name) for name in SOURCE_FILES}
        protocol["banks"][bank_id] = entry
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--banks", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    protocol = assemble(json.loads(Path(args.template).read_text()), json.loads(Path(args.banks).read_text()))
    out = Path(args.out)
    with out.open("x") as handle:
        handle.write(json.dumps(protocol, indent=1, sort_keys=True) + "\n")
    print(json.dumps({bank: entry["bank_sha256"] for bank, entry in protocol["banks"].items()}))


if __name__ == "__main__":
    main()
