"""Assemble the R1 reviewer archive for Must Fix 2: what a reviewer needs to check the R1 results.

Layout under OUT:
  protocols/      docs/nhsjs-revision/protocols/r1 (collection, selection, analysis plan and amendments,
                  production protocols, banks)
  results/        the R1 analysis directory (tables, figures, checks)
  runs/           record.json, rankings.json and completion.json of every completed phase; the tensors
                  stay out and runs/TENSORS.sha256 lists each one with the hash its completion record binds.
                  With --include-raw-runs, every raw file is included and authenticated instead.
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
         [--collection NAME=DIR ...] [--kit DIR] [--include-raw-runs]
         [--anonymous [--identifier TOKEN ...]] --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_research_supplement import (  # noqa: E402
    ANONYMOUS_TOKENS, IDENTIFIERS, audit_tokens, identity_views, pdf_reader_available,
)
from scripts.reproduce_reviewer_archive import member, extract_code  # noqa: E402

RECORD_FILES = ("record.json", "rankings.json", "completion.json")
ARCHIVE_IDENTIFIERS = (*IDENTIFIERS, *ANONYMOUS_TOKENS)
PERSONAL_PATH = re.compile(r"[A-Za-z]:[\\/]Users[\\/]|/Users/[^/\s\"']+/|/home/[^/\s\"']+/", re.I)
# The publicly pinned Google T5 tokenizer contains its upstream build-system
# path /placer/prod/scratch/home/brain-ogm-data-writer/. It is not an author
# identifier. Permit only these exact raw and NUL-stripped public bytes; any
# alteration loses the exception. The source URL and original digest are in
# analysis/language_groups/protocol.json. No tokenizer bytes are rewritten.
PUBLIC_TOKENIZER_VIEWS = frozenset({
    "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
    "d0f0e66dea80bc9adecfa60b908474088abb833885521d4c27a940a10c956bc7",
})
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


def reject_aliases(path):
    """Reject an aliased selection before resolve() can hide its path components."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if component.is_symlink() or (hasattr(component, "is_junction") and component.is_junction()):
            raise ValueError(f"Archive path contains a symbolic link or junction: {component}")
    return path


def validate_source_tree(source):
    """Check each directory before descending, including Windows junctions."""
    source = reject_aliases(source)
    if not source.is_dir():
        raise NotADirectoryError(source)
    root = source.resolve(strict=True)
    pending = [source]
    while pending:
        for path in pending.pop().iterdir():
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError(f"Archive source contains a symbolic link or junction: {path}")
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError(f"Archive source path escapes its selected tree: {path}")
            if path.is_dir():
                pending.append(path)


def bound_files(completion):
    """Resolve portable completion members without permitting path escapes."""
    bound = json.loads(completion.read_text(encoding="utf-8"))["files_sha256"]
    if not isinstance(bound, dict):
        raise ValueError(f"Invalid completion file map: {completion}")
    for name, digest in sorted(bound.items()):
        if (not isinstance(name, str) or not name or "\\" in name or ":" in name
                or PurePosixPath(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/"))
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError(f"Invalid completion member: {name!r}")
        path = reject_aliases(completion.parent / name)
        if not path.resolve().is_relative_to(completion.parent.resolve()):
            raise ValueError(f"Invalid completion member outside its phase: {name!r}")
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"Completion hash mismatch or missing file: {path}")
    return bound


def copy_runs(runs, destination, include_raw=False):
    """Phase records of every completed phase, the hash list of the tensors left out, the worker
    event logs, and the failure records of failed attempts with a hash list of what else they kept."""
    runs = Path(runs)
    completions = [(path, bound_files(path)) for path in sorted(runs.glob("*/e*-c*/*/completion.json"))]
    if include_raw:
        # Preserve every attempt and auxiliary file, including empty directories.
        shutil.copytree(runs, destination)
    else:
        destination.mkdir(parents=True, exist_ok=True)
    tensors, tensor_bytes, kinds = [], 0, {}
    for completion, bound in completions:
        phase_dir = completion.parent
        target = destination / phase_dir.relative_to(runs)
        target.mkdir(parents=True, exist_ok=True)
        for name in RECORD_FILES:
            if (phase_dir / name).is_file():
                shutil.copyfile(phase_dir / name, target / name)
        if include_raw:
            for name, digest in bound.items():
                if sha256(target / name) != digest:
                    raise ValueError(f"Copied completion hash mismatch: {target / name}")
        tensors += [f"{digest}  {(phase_dir / name).relative_to(runs).as_posix()}\n" for name, digest in sorted(bound.items())
                    if name not in RECORD_FILES]
        tensor_bytes += sum((phase_dir / name).stat().st_size for name in bound if name not in RECORD_FILES)
        for name in bound:
            if name not in RECORD_FILES:
                kind = ("map files" if name == "maps.pt" else "node-gradient files" if name.startswith("nodes-")
                        else "source caches" if name.startswith("source-cache") else "other files")
                kinds[kind] = kinds.get(kind, 0) + 1
    (destination / "TENSORS.sha256").write_text("".join(tensors), encoding="utf-8", newline="\n")
    for log in sorted(runs.glob("*/workers/*.jsonl")):
        target = destination / log.relative_to(runs)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log, target)
    kept, failures, states = [], 0, []
    for directory in sorted(p for p in runs.glob("*/e*-c*/attempts/*") if p.is_dir()):
        files = [p for p in directory.rglob("*") if p.is_file()]
        state = ("failed (failure.json kept)" if (directory / "failure.json").is_file()
                 else f"interrupted, {len(files)} files listed in ATTEMPTS.sha256" if files else "interrupted, empty")
        states.append(f"{directory.relative_to(runs).as_posix()}/\t{state}\n")
    (destination / "ATTEMPT_DIRECTORIES.txt").write_text("".join(states), encoding="utf-8", newline="\n")
    for path in sorted(p for p in runs.glob("*/e*-c*/attempts/**/*") if p.is_file()):
        relative = path.relative_to(runs)
        if path.name == "failure.json":
            (destination / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination / relative)
            failures += 1
        else:
            kept.append(f"{sha256(path)}  {relative.as_posix()}\n")
    (destination / "ATTEMPTS.sha256").write_text("".join(kept), encoding="utf-8", newline="\n")
    return len(tensors), failures, tensor_bytes, kinds


def stored_zip_hits(path, tokens, chunk_size=1 << 23):
    """Scan large uncompressed tensor ZIPs with bounded memory and boundary overlap.

    For a ZIP with only stored members, identity_views exposes exactly its raw
    bytes, NUL-stripped bytes and member names. Streaming those same views avoids
    several gigabytes of copies for a single saved gradient tensor. Compressed
    ZIPs keep the full recursive scanner instead.
    """
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if any(member.compress_type != zipfile.ZIP_STORED for member in members):
            return None
        names = "\n".join(member.filename for member in members).lower()
    found = {token for token in tokens if token in names}
    if PERSONAL_PATH.search(names):
        found.add("personal machine path")
    # identity_views decodes raw ZIP bytes as latin-1. A token outside that
    # alphabet cannot match that view, but still participates in filename scans.
    encoded = [(token, token.encode("latin-1")) for token in tokens if all(ord(c) < 256 for c in token)]
    overlap = max([256, *(len(value) for _, value in encoded)])
    tails = [b"", b""]
    # Prefix detection is conservative even when a very long username spans chunks.
    personal = re.compile(rb"[a-z]:[\\/]users[\\/]|/users/|/home/")
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            for index, view in enumerate((chunk, chunk.replace(b"\x00", b""))):
                view = tails[index] + view.lower()
                found.update(token for token, value in encoded if value in view)
                # Every regex alternative contains one of these lowercase
                # literals. Fast substring search avoids applying a regex to
                # gigabytes of numeric tensor bytes with no path prefix.
                if (b"users" in view or b"/home/" in view) and personal.search(view):
                    found.add("personal machine path")
                tails[index] = view[-overlap:]
    return sorted(found)


def identifying_hits(root, extra=()):
    """Files whose path or any view of whose bytes holds an identifier or a personal machine path."""
    tokens = (*ARCHIVE_IDENTIFIERS, *audit_tokens(list(extra)))
    hits = []
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        found = {token for token in tokens if token in relative.lower()}
        if PERSONAL_PATH.search(relative):
            found.add("personal machine path")
        with path.open("rb") as handle:
            header = handle.read(4)
        if header == b"%PDF" and not pdf_reader_available():
            raise ValueError(f"PDF identity scan requires PyMuPDF: {relative}")
        streamed = (stored_zip_hits(path, tokens) if path.stat().st_size > 1 << 26
                    and header == b"PK\x03\x04" else None)
        if streamed is not None:
            found.update(streamed)
        else:
            for view in identity_views(path.read_bytes()):
                text = view.decode("latin-1").lower()
                found.update(token for token in tokens if token in text)
                # This view is lowercase latin-1, so the prefilter is equivalent
                # to the regex's literal Users/home components. Filenames keep
                # their separate unrestricted Unicode regex check above.
                if (("users" in text or "/home/" in text) and PERSONAL_PATH.search(text)
                        and hashlib.sha256(view).hexdigest() not in PUBLIC_TOKENIZER_VIEWS):
                    found.add("personal machine path")
        if found:
            hits.append((relative, sorted(found)))
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


def add_reproduction(out, code, coverage):
    """Bind a portable replay to the exact frozen plan, coverage and raw inputs."""
    original = json.loads((out / "results/analysis/manifest.json").read_text(encoding="utf-8"))
    if original["kind"] != "nhsjs_r1_gpu_analysis":
        raise ValueError("Portable replay requires the complete registered analysis")
    for name, expected in original["outputs"].items():
        if sha256(out / "results/analysis" / member(name)) != expected:
            raise ValueError("Frozen analysis output mismatch: " + name)
    plans = [p for p in (out / "protocols").glob("analysis-plan*.json")
             if sha256(p) == original["plan_sha256"]]
    if len(plans) != 1:
        raise ValueError("Expected exactly one archive-contained registered plan")
    supplied = {Path(path).name: Path(path) for path in coverage}
    if len(supplied) != len(coverage):
        raise ValueError("Repeated coverage input name")
    coverage_paths = []
    for name, expected in original["coverage_sha256"].items():
        source = reject_aliases(supplied.pop(Path(member(name)).name, ROOT / name))
        if source.is_symlink() or not source.is_file() or sha256(source) != expected:
            raise ValueError("Coverage input missing or differs from registered analysis: " + name)
        target = out / "coverage" / Path(name).name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        coverage_paths.append(target.relative_to(out).as_posix())
    if supplied:
        raise ValueError("Unexpected coverage inputs: " + ", ".join(supplied))
    input_paths = {}
    prefix = "docs/nhsjs-revision/protocols/r1/"
    for spec in original["inputs"]["banks"]:
        for key in ("bank", "protocol"):
            name = member(spec[key])
            if not name.startswith(prefix):
                raise ValueError("Unreviewed registered input path: " + name)
            target = "protocols/" + name[len(prefix):]
            if not (out / target).is_file():
                raise ValueError("Missing archive-contained registered input: " + target)
            input_paths[name] = target
    metrics = {}
    for spec in original["inputs"]["banks"]:
        bank = member(spec["id"])
        if bank in ("P", "B1", "S"):
            target = "collections/" + bank + "/metrics.jsonl"
        elif bank.startswith("M8-") and "/" not in bank:
            target = "month8/context-bank-v1/" + bank[3:] + "/metrics.jsonl"
        else:
            raise ValueError("Unreviewed outcome bank: " + bank)
        record = json.loads((out / input_paths[spec["bank"]]).read_text(encoding="utf-8"))
        if (not (out / target).is_file()
                or sha256(out / target) != record.get("source_metrics_sha256")):
            raise ValueError("Missing collection outcomes: " + target)
        metrics[bank] = target
    # Authenticate the nested code inventory now, without executing it. The
    # launcher repeats the check before execution in a fresh extraction.
    temporary = out / ".validated-code"
    extract_code(code, temporary)
    shutil.rmtree(temporary)
    (out / "code").mkdir()
    shutil.copyfile(code, out / "code/research-supplement.zip")
    shutil.copyfile(ROOT / "scripts/reproduce_reviewer_archive.py", out / "reproduce.py")
    shutil.copyfile(ROOT / "requirements-replay-linux.txt", out / "requirements-replay-linux.txt")
    config = dict(schema_version=1, code="code/research-supplement.zip",
                  expected_manifest="results/analysis/manifest.json",
                  plan=plans[0].relative_to(out).as_posix(), coverage=coverage_paths,
                  input_paths=input_paths, metrics=metrics, runtime_lock="requirements-replay-linux.txt")
    (out / "reproduction.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")


def build(out, *, results, runs, identity, collections=(), kit=None, anonymous=False, identifiers=(),
          sources=SOURCES, include_raw_runs=False, reproduction_code=None, coverage=(), mathematical_supplement=None):
    out = reject_aliases(out)
    if out.exists():
        raise FileExistsError(out)
    if reproduction_code is not None and not include_raw_runs:
        raise ValueError("Portable registered-analysis replay requires complete raw runs")
    collections = list(collections)
    coverage = list(coverage)
    names = [name for name, _ in collections]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) for name in names):
        raise ValueError("Collection names must be unique single portable path components")
    inputs = [results, runs, identity, *sources.values(), *(source for _, source in collections)]
    if kit:
        inputs.append(kit)
    for source in (Path(source) for source in inputs if source is not None):
        validate_source_tree(source)
        if out.resolve().is_relative_to(source.resolve()):
            raise ValueError("Archive output must be outside every source tree")
    for source in (reproduction_code, mathematical_supplement, *coverage):
        if source is not None:
            path = reject_aliases(source)
            if not path.is_file():
                raise ValueError(f"Archive input must be a regular file: {path}")
    out.mkdir(parents=True)
    try:
        copy_tree(sources["protocols"], out / "protocols")
        copy_tree(results, out / "results")
        tensors, failures, tensor_bytes, kinds = copy_runs(Path(runs), out / "runs", include_raw_runs)
        for name, source in collections:
            copy_tree(source, out / "collections" / name)
        if kit:
            # A kit's own manifest can bind auxiliary files too. Preserve every
            # byte rather than applying the generic source-tree cache filter.
            shutil.copytree(kit, out / "kit")
        if sources.get("regeneration") and Path(sources["regeneration"]).is_dir():
            copy_tree(sources["regeneration"], out / "regeneration")
        copy_tree(sources["month8"], out / "month8")
        copy_tree(identity, out / "identity")
        (out / "inventories").mkdir()
        for name in INVENTORY_FILES:
            shutil.copyfile(Path(sources["inventories"]) / name, out / "inventories" / name)
        if reproduction_code is not None:
            add_reproduction(out, reproduction_code, coverage)
        if mathematical_supplement is not None:
            path = Path(mathematical_supplement)
            if path.is_symlink() or not path.is_file() or path.suffix not in (".tex", ".pdf"):
                raise ValueError("Expected a regular mathematical supplement PDF or TeX source")
            (out / "mathematics").mkdir()
            shutil.copyfile(path, out / "mathematics" / ("supplementary_derivations" + path.suffix))
        listed = [f"{count} {kind}" for kind, count in sorted(kinds.items())]
        kinds_text = ", ".join(listed[:-1]) + (" and " if len(listed) > 1 else "") + listed[-1] if listed else "none"
        readme = README.format(tensors=tensors, kinds=kinds_text, gigabytes=tensor_bytes / 1e9)
        if include_raw_runs:
            readme = readme[:readme.index("The " + str(tensors) + " tensor files")]
            readme += (f"All {tensors} completion-bound raw files ({kinds_text}; {tensor_bytes / 1e9:.1f} GB) are included. "
                       "`runs/TENSORS.sha256` authenticates them against their completion records. "
                       "The complete runs tree also preserves worker logs, failed and interrupted attempts, "
                       "and empty attempt directories. No raw file or hash-bound record has been redacted.\n\n"
                       "The archive includes the raw inputs required to rerun the registered analysis. "
                       "Use the accompanying research supplement's analysis code and the dedicated replay environment; "
                       "point its input configuration at this archive's runs, protocols and banks. "
                       "GPU regeneration still requires the pinned models and runtime documented in kit/README.md.\n")
        if reproduction_code is not None:
            readme += ("\n## Portable registered-analysis replay\n\n"
                       "The complete research code is in `code/research-supplement.zip`. Use Linux x86-64 "
                       "Python 3.12 and at least 24 GB RAM (32 GB recommended). In a fresh virtual environment, "
                       "install this archive's dedicated runtime and run:\n\n"
                       "    python -m pip install -r requirements-replay-linux.txt\n"
                       "    python reproduce.py --verify-only\n"
                       "    python reproduce.py --out ../registered-analysis-reproduction\n\n"
                       "The launcher verifies every archive file, authenticates and extracts the code into the fresh "
                       "output directory, creates runtime paths wholly inside this archive, and runs the full "
                       "registered CPU analysis. Both frozen coverage simulations and all nine banks' collection outcomes "
                       "are included. It compares every generated scientific output hash and records all differences "
                       "in `reproduction-report.json`; original records and their hashes remain unchanged. "
                       "A successful replay exits zero. A GPU and access to the original volume are unnecessary.\n")
        (out / "README.md").write_text(readme, encoding="utf-8", newline="\n")
        if anonymous:
            hits = identifying_hits(out, identifiers)
            if hits:
                raise ValueError(f"Identifying text in the anonymous archive ({len(hits)} files): {hits[:10]}")
        files = sorted(p for p in out.rglob("*") if p.is_file())
        (out / "MANIFEST.sha256").write_text("".join(f"{sha256(p)}  {p.relative_to(out).as_posix()}\n" for p in files),
                                            encoding="utf-8", newline="\n")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return dict(files=len(files), bytes=sum(p.stat().st_size for p in files), tensors_listed=tensors,
                attempt_failures=failures, raw_runs_included=include_raw_runs,
                tensors_included=tensors if include_raw_runs else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--collection", action="append", default=[], help="NAME=DIR")
    parser.add_argument("--kit")
    parser.add_argument("--anonymous", action="store_true")
    parser.add_argument("--include-raw-runs", action="store_true",
                        help="Include and authenticate the complete runs tree, including tensors and all attempts")
    parser.add_argument("--identifier", action="append", default=[],
                        help="extra token the anonymous scan must not find (case-insensitive)")
    parser.add_argument("--reproduction-code", help="Verified research supplement ZIP for a self-contained registered-analysis replay")
    parser.add_argument("--coverage", action="append", default=[], help="Frozen coverage JSON override, authenticated by the result manifest")
    parser.add_argument("--mathematical-supplement", help="Standalone mathematical supplement PDF or TeX source")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    collections = [tuple(item.split("=", 1)) for item in args.collection]
    print(json.dumps(build(args.out, results=args.results, runs=args.runs, identity=args.identity,
                           collections=collections, kit=args.kit, anonymous=args.anonymous,
                           identifiers=args.identifier, include_raw_runs=args.include_raw_runs,
                           reproduction_code=args.reproduction_code, coverage=args.coverage,
                           mathematical_supplement=args.mathematical_supplement)))


if __name__ == "__main__":
    main()
