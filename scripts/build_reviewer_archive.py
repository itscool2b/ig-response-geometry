"""Assemble the R1 reviewer archive for Must Fix 2: what a reviewer needs to check the R1 results.

Layout under OUT:
  protocols/      docs/nhsjs-revision/protocols/r1 (collection, selection, analysis plan and amendments,
                  production protocols, banks)
  results/        the R1 analysis directory (tables, figures, checks)
  runs/           record.json, rankings.json and completion.json of every completed phase; the tensors
                  stay out and runs/TENSORS.sha256 lists each one with the hash its completion record binds.
                  Also every worker event log (<bank>/workers/*.jsonl) and the failure.json of every failed
                  attempt; runs/ATTEMPTS.sha256 lists every other file left in an attempt directory, and
                  runs/ATTEMPT_DIRECTORIES.txt every attempt directory with its state (failed, or interrupted
                  with or without files), empty ones included
  collections/    the collection runs named by --collection (every executed call's inputs and records)
  kit/            the regeneration kit, when given
  regeneration/   the fresh-pod regeneration reports, logs and identity (docs/nhsjs-revision/regeneration/fresh-pod-*)
  month8/         docs/nhsjs-revision/month8 (the recovered Month 8 bank, code snapshots, language inputs)
  identity/       the pod identity captures
  inventories/    the Must Fix 2 identity inventory and reproducibility classification
  README.md       what each directory holds and what the archive alone can and cannot check
  MANIFEST.sha256 every file above

With --anonymous every file's path and bytes are scanned for the research-supplement packager's
identifiers and anonymous-candidate tokens, any --identifier token and personal machine paths
(/Users/<name>/, /home/<name>/, C:\\Users\\), through the packager's identity_views: raw and
NUL-stripped bytes, gzip and compressed zip members, zlib streams of binaries, and PDF drawn text.
/workspace and /root are the pod mounts every RunPod container has and are not flagged. Any hit
stops the build and removes the partial output.

Usage: python scripts/build_reviewer_archive.py --results DIR --runs DIR --identity DIR
         [--collection NAME=DIR ...] [--kit DIR] [--anonymous [--identifier TOKEN ...]] --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_research_supplement import ANONYMOUS_TOKENS, IDENTIFIERS, audit_tokens, identity_views  # noqa: E402

RECORD_FILES = ("record.json", "rankings.json", "completion.json")
ARCHIVE_IDENTIFIERS = (*IDENTIFIERS, *ANONYMOUS_TOKENS)
PERSONAL_PATH = re.compile(r"[A-Za-z]:[\\/]Users[\\/]|/Users/[^/\s\"']+/|/home/[^/\s\"']+/", re.I)
INVENTORY_FILES = ("identities.json", "identities.md", "reproducibility_classification.json",
                   "reproducibility_classification.md", "rec4_facts.csv")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.lock", "__pycache__", ".DS_Store"))


def copy_runs(runs, destination):
    """Phase records of every completed phase, the hash list of the tensors left out, the worker
    event logs, and the failure records of failed attempts with a hash list of what else they kept."""
    runs = Path(runs)
    destination.mkdir(parents=True, exist_ok=True)
    tensors, tensor_bytes, kinds = [], 0, {}
    for completion in sorted(runs.glob("*/e*-c*/*/completion.json")):
        phase_dir = completion.parent
        target = destination / phase_dir.relative_to(runs)
        target.mkdir(parents=True, exist_ok=True)
        for name in RECORD_FILES:
            if (phase_dir / name).is_file():
                shutil.copyfile(phase_dir / name, target / name)
        bound = json.loads(completion.read_text())["files_sha256"]
        tensors += [f"{digest}  {(phase_dir / name).relative_to(runs)}\n" for name, digest in sorted(bound.items())
                    if name not in RECORD_FILES]
        tensor_bytes += sum((phase_dir / name).stat().st_size for name in bound if name not in RECORD_FILES)
        for name in bound:
            if name not in RECORD_FILES:
                kind = ("map files" if name == "maps.pt" else "node-gradient files" if name.startswith("nodes-")
                        else "source caches" if name.startswith("source-cache") else "other files")
                kinds[kind] = kinds.get(kind, 0) + 1
    (destination / "TENSORS.sha256").write_text("".join(tensors))
    for log in sorted(runs.glob("*/workers/*.jsonl")):
        target = destination / log.relative_to(runs)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log, target)
    kept, failures, states = [], 0, []
    for directory in sorted(p for p in runs.glob("*/e*-c*/attempts/*") if p.is_dir()):
        files = [p for p in directory.rglob("*") if p.is_file()]
        state = ("failed (failure.json kept)" if (directory / "failure.json").is_file()
                 else f"interrupted, {len(files)} files listed in ATTEMPTS.sha256" if files else "interrupted, empty")
        states.append(f"{directory.relative_to(runs)}/\t{state}\n")
    (destination / "ATTEMPT_DIRECTORIES.txt").write_text("".join(states))
    for path in sorted(p for p in runs.glob("*/e*-c*/attempts/**/*") if p.is_file()):
        relative = path.relative_to(runs)
        if path.name == "failure.json":
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination / relative)
            failures += 1
        else:
            kept.append(f"{sha256(path)}  {relative}\n")
    (destination / "ATTEMPTS.sha256").write_text("".join(kept))
    return len(tensors), failures, tensor_bytes, kinds


def identifying_hits(root, extra=()):
    """Files whose path or any view of whose bytes holds an identifier or a personal machine path."""
    tokens = (*ARCHIVE_IDENTIFIERS, *audit_tokens(list(extra)))
    hits = []
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        relative = str(path.relative_to(root))
        texts = [relative] + [view.decode("latin-1") for view in identity_views(path.read_bytes())]
        found = [token for token in tokens if any(token in text.lower() for text in texts)]
        if any(PERSONAL_PATH.search(text) for text in texts):
            found.append("personal machine path")
        if found:
            hits.append((relative, found))
    return hits


README = """# R1 reviewer archive

Records of the NHSJS-R1 GPU campaign for Must Fix 2: the protocols, the inputs of every policy call,
what each computation phase recorded, the registered analysis and the regeneration check. Check the
files first, from this directory:

    sha256sum -c MANIFEST.sha256        (on macOS: shasum -a 256 -c MANIFEST.sha256)

| Path | What it holds |
|---|---|
| `protocols/` | The collection and selection protocols, the analysis template, the analysis plan and its amendments, the production protocols and the banks. |
| `collections/` | The collection runs: every executed policy call's inputs and records. |
| `runs/` | `record.json`, `rankings.json` and `completion.json` of every completed phase, every worker event log, the failure record of every failed attempt, and `ATTEMPT_DIRECTORIES.txt`, which lists every attempt directory with its state. |
| `results/` | The registered analysis: tables, figures, populations and the checks made before unblinding. |
| `kit/` | The regeneration kit. `kit/README.md` regenerates one P context end to end on a GPU and checks its stored scoring on CPU. |
| `regeneration/` | The reports, logs and machine identity of that kit's run on a fresh machine. |
| `month8/` | The recovered Month 8 context bank, its code snapshots and language inputs (`month8/README.md`). |
| `identity/` | The identity captures of the campaign machines, and `phase_hosts.csv`, the machine that computed each phase. |
| `inventories/` | The identity inventory and the reproducibility classification of every reported result. |

The {tensors} tensor files ({kinds}; {gigabytes:.1f} GB) are not included. `runs/TENSORS.sha256` lists
each with the hash its completion record binds. The analysis loader authenticates every file a completion
record binds, so re-running the registered analysis, or the CPU check of a whole bank, needs those
tensors. Without them, each phase's recorded values can be read and compared with the result tables.
"""


SOURCES = dict(protocols=ROOT / "docs/nhsjs-revision/protocols/r1", month8=ROOT / "docs/nhsjs-revision/month8",
               inventories=ROOT / "docs/nhsjs-revision/mf2",
               regeneration=ROOT / "docs/nhsjs-revision/regeneration/fresh-pod-20261024")


def build(out, *, results, runs, identity, collections=(), kit=None, anonymous=False, identifiers=(),
          sources=SOURCES):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    try:
        copy_tree(sources["protocols"], out / "protocols")
        copy_tree(results, out / "results")
        tensors, failures, tensor_bytes, kinds = copy_runs(Path(runs), out / "runs")
        for name, source in collections:
            copy_tree(source, out / "collections" / name)
        if kit:
            copy_tree(kit, out / "kit")
        if sources.get("regeneration") and Path(sources["regeneration"]).is_dir():
            copy_tree(sources["regeneration"], out / "regeneration")
        copy_tree(sources["month8"], out / "month8")
        copy_tree(identity, out / "identity")
        (out / "inventories").mkdir()
        for name in INVENTORY_FILES:
            shutil.copyfile(Path(sources["inventories"]) / name, out / "inventories" / name)
        listed = [f"{count} {kind}" for kind, count in sorted(kinds.items())]
        kinds_text = ", ".join(listed[:-1]) + (" and " if len(listed) > 1 else "") + listed[-1] if listed else "none"
        (out / "README.md").write_text(README.format(tensors=tensors, kinds=kinds_text, gigabytes=tensor_bytes / 1e9))
        if anonymous:
            hits = identifying_hits(out, identifiers)
            if hits:
                raise ValueError(f"Identifying text in the anonymous archive ({len(hits)} files): {hits[:10]}")
        files = sorted(p for p in out.rglob("*") if p.is_file())
        (out / "MANIFEST.sha256").write_text("".join(f"{sha256(p)}  {p.relative_to(out)}\n" for p in files))
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return dict(files=len(files), bytes=sum(p.stat().st_size for p in files), tensors_listed=tensors,
                attempt_failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--collection", action="append", default=[], help="NAME=DIR")
    parser.add_argument("--kit")
    parser.add_argument("--anonymous", action="store_true")
    parser.add_argument("--identifier", action="append", default=[],
                        help="extra token the anonymous scan must not find (case-insensitive)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    collections = [tuple(item.split("=", 1)) for item in args.collection]
    print(json.dumps(build(args.out, results=args.results, runs=args.runs, identity=args.identity,
                           collections=collections, kit=args.kit, anonymous=args.anonymous,
                           identifiers=args.identifier)))


if __name__ == "__main__":
    main()
