"""Build the Month 8 replay tree: the frozen revision code plus the exact Month 8 forward bytes.

The Month 8 context bank recorded mixed-line-ending working copies of
pipeline.py and per_step_attribution.py that git cannot rebuild (note 9.4,
section 2). Strict replay compares those hashes, so this copies the frozen
revision tree and replaces the replay-checked forward files with the recovered
snapshot bytes. Every other file keeps its frozen bytes. The record lists each
replaced file with the lines that differ after newline normalization, so a
reader can see that the Month 8 forward code, not a new one, ran the replay.

Usage: python scripts/build_m8_overlay.py --frozen DIR --snapshot DIR --manifest RUN_MANIFEST --out DIR
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

# The files faithfulness.check_replay_identity compares with the collection.
REPLAY_CHECKED = ("pipeline.py", "per_step_attribution.py", "rdt_sampling.py", "checkpoint_contract.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(frozen, snapshot, manifest, out):
    frozen, snapshot, out = Path(frozen), Path(snapshot), Path(out)
    if out.exists():
        raise FileExistsError(f"Overlay exists: {out}")
    recorded = json.loads(Path(manifest).read_text())["configuration"]["source_sha256"]
    replaced = {}
    for name in REPLAY_CHECKED:
        if sha256(snapshot / name) != recorded[name]:
            raise ValueError(f"Snapshot {name} is not the recorded Month 8 byte sequence")
        if sha256(frozen / name) != recorded[name]:
            old = (snapshot / name).read_bytes().replace(b"\r\n", b"\n").decode().splitlines()
            new = (frozen / name).read_bytes().replace(b"\r\n", b"\n").decode().splitlines()
            replaced[name] = dict(frozen_sha256=sha256(frozen / name), snapshot_sha256=recorded[name],
                                  normalized_diff=list(difflib.unified_diff(old, new, "month8", "frozen", n=0,
                                                                           lineterm="")))
    shutil.copytree(frozen, out, symlinks=False)
    for name in replaced:
        shutil.copyfile(snapshot / name, out / name)
    for name in REPLAY_CHECKED:
        if sha256(out / name) != recorded[name]:
            raise ValueError(f"Overlay {name} differs from the recorded hash")
    files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    record = dict(kind="month8_replay_overlay", frozen=str(frozen), snapshot=str(snapshot),
                  replaced=replaced, recorded_forward_sha256={k: recorded[k] for k in REPLAY_CHECKED},
                  files_sha256=files)
    (out.parent / (out.name + ".overlay.json")).write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--manifest", required=True, help="a Month 8 bank run manifest.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    record = build(args.frozen, args.snapshot, args.manifest, args.out)
    print(json.dumps(dict(replaced=sorted(record["replaced"]), files=len(record["files_sha256"]))))


if __name__ == "__main__":
    main()
