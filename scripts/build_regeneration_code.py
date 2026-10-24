"""Build the code archive of the R1 regeneration kit from a frozen tag: only what the regeneration runs.

The archive holds the import closure of the worker, the regeneration script,
the pinned-model download and the collector (found by reading imports, as the
research-supplement packager does), the RDT-170M config, the pod script, the license and a
third-party notice. Bytes are git blobs of the tag, so the worker's pinned
source hashes still match. With --anonymous the license's copyright line names
an anonymous author; nothing else changes, and no member may carry an
identifying string.

Usage: python scripts/build_regeneration_code.py TAG OUTDIR [--anonymous]
Writes OUTDIR/<TAG>-regeneration[-anonymous].tar and the matching .sha256 (sha256sum -c format).
"""
import argparse
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_research_supplement import IDENTIFIERS, import_dependencies  # noqa: E402

# The worker, the regeneration and model-download scripts, and the collector: the source replay
# checks the collector files the collection configuration hashes (collect_contexts.py, per_step_ig.py).
ENTRYPOINTS = ("revision_numerics.py", "scripts/regenerate_result.py", "scripts/prepare_pinned_models.py",
               "collect_contexts.py", "per_step_ig.py", "scripts/launch_strict.py")
EXTRAS = ("configs/base_170m.yaml", "scripts/revision_pod.sh", "LICENSE")
OWN_NOTICE = b"Copyright (c) 2026 Arjun Bajpai"
ANONYMOUS_NOTICE = b"Copyright (c) 2026 Anonymous author"


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def members(tag, anonymous=False):
    available = set(git("ls-tree", "-r", "--name-only", tag).decode().splitlines())
    blob = lambda name: git("show", f"{tag}:{name}")  # noqa: E731
    selected, queue = set(), list(ENTRYPOINTS)
    while queue:
        name = queue.pop()
        if name in selected:
            continue
        selected.add(name)
        local, _ = import_dependencies(name, blob(name), available)
        queue.extend(sorted(local - selected))
    contents = {name: blob(name) for name in sorted(selected | set(EXTRAS))}
    sampler = contents["rdt_sampling.py"].decode("utf-8")
    notice = sampler.split("Original sampler license:\n", 1)[1].split('"""', 1)[0].strip()
    contents["THIRD_PARTY_NOTICES.txt"] = (
        "rdt_sampling.py and configs/base_170m.yaml derive from the official RDT project.\n"
        "Source: https://github.com/thu-ml/RoboticsDiffusionTransformer/tree/cd79363a1387e8f81c7724d070ef7e45fd23150f\n\n"
        + notice + "\n").encode()
    if anonymous:
        if contents["LICENSE"].count(OWN_NOTICE) != 1:
            raise ValueError("License notice differs from the reviewed anonymous replacement")
        contents["LICENSE"] = contents["LICENSE"].replace(OWN_NOTICE, ANONYMOUS_NOTICE)
    for name, data in contents.items():
        hits = [token for token in IDENTIFIERS if token in data.decode("utf-8", "replace").lower()]
        if hits and (anonymous or name != "LICENSE"):
            raise ValueError(f"Identifying text in {name}: {hits}")
    return contents


def build(tag, outdir, anonymous=False):
    contents = members(tag, anonymous)
    stem = f"{tag}-regeneration" + ("-anonymous" if anonymous else "")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in sorted(contents.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uid, info.gid = len(data), 1767225600, 0, 0
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tar_path = outdir / f"{stem}.tar"
    tar_path.write_bytes(stream.getvalue())
    manifest = outdir / f"{stem}.sha256"
    manifest.write_text("".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(contents.items())))
    return tar_path, manifest, len(contents)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag")
    parser.add_argument("outdir")
    parser.add_argument("--anonymous", action="store_true")
    args = parser.parse_args()
    tar_path, manifest, count = build(args.tag, args.outdir, args.anonymous)
    print(f"{tar_path} {hashlib.sha256(tar_path.read_bytes()).hexdigest()} files={count}")


if __name__ == "__main__":
    main()
