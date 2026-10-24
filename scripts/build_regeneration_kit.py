"""Assemble the R1 regeneration kit for one P context (Must Fix 2).

The kit mirrors the campaign's absolute paths under `workspace/`, so the
committed protocol works unchanged once that tree is copied to `/workspace` on
a fresh pod. It holds the full P collection run (bank authentication checks
every episode), the bank, the protocol, the language inputs, the stored phase
outputs of the chosen context (its fp32 phase too when the context is in the
protocol's fp32 subset, since scoring needs it), the regeneration code archive
with its manifest, the pinned package list and the README. MANIFEST.sha256
covers every file.

Usage (on a pod with the campaign volume):
  python scripts/build_regeneration_kit.py --context e0030-c017 --code-archive TAR --code-manifest SHA256 --out DIR
"""
import argparse
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "workspace/nhsjs/banks/r1/P": "/workspace/nhsjs/banks/r1/P",
    "workspace/nhsjs/banks/r1/P-bank.json": "/workspace/nhsjs/banks/r1/P-bank.json",
    "workspace/nhsjs/protocols/r1-p.json": "/workspace/nhsjs/protocols/r1-p.json",
    "workspace/tmlr/lang-20260930": "/workspace/tmlr/lang-20260930",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(context, code_archive, code_manifest, requirements, readme, out, stored_root="/workspace/nhsjs/runs/r1"):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    for target, source in SOURCES.items():
        destination = out / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if Path(source).is_dir():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.lock"))
        else:
            shutil.copyfile(source, destination)
    for phase in ("bf16", "fp32", "scoring"):
        # fp32 exists only for contexts in the protocol's fp32 subset; scoring needs it there.
        source = Path(stored_root) / "P" / context / phase
        if phase == "fp32" and not source.exists():
            continue
        if not (source / "completion.json").is_file():
            raise FileNotFoundError(f"Stored phase {phase} of {context} is not complete")
        shutil.copytree(source, out / "stored" / "P" / context / phase)
    (out / "code").mkdir()
    shutil.copyfile(code_archive, out / "code" / Path(code_archive).name)
    shutil.copyfile(code_manifest, out / "code" / Path(code_manifest).name)
    shutil.copyfile(requirements, out / "requirements-r1-venv.txt")
    shutil.copyfile(readme, out / "README.md")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    lines = [f"{sha256(p)}  {p.relative_to(out)}\n" for p in files]
    (out / "MANIFEST.sha256").write_text("".join(lines))
    return len(lines), sum(p.stat().st_size for p in files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--code-archive", required=True)
    parser.add_argument("--code-manifest", required=True)
    parser.add_argument("--requirements", default=str(ROOT / "docs/nhsjs-revision/regeneration/requirements-r1-venv.txt"))
    parser.add_argument("--readme", default=str(ROOT / "docs/nhsjs-revision/regeneration/README.md"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    count, size = build(args.context, args.code_archive, args.code_manifest, args.requirements, args.readme, args.out)
    print(f"{count} files, {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
