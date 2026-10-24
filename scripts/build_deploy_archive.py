"""Build the pod deployment archive for a frozen tag, with a per-file SHA-256 manifest.

Only git-archive bytes of the tag are deployed (working copies are how the
Month 8 line-ending mismatch happened). The archive holds the code, configs,
scripts, tests and the data the tests read, plus the archived historical kernel
and the revision protocols; the rest of docs/ stays out to keep it small.

Usage: python scripts/build_deploy_archive.py TAG OUTDIR
Writes OUTDIR/TAG.tar and OUTDIR/TAG.sha256 (sha256sum -c format, paths relative to the root).
"""
import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("configs", "scripts", "tests", "data", "analysis", "legacy", "paper", "docs/manuscript_revision",
               "docs/tmlr-readiness/archives/active-originals/2026-10-01-stable-base", "docs/nhsjs-revision/protocols")


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout


def build(tag, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    top = [line for line in git("ls-tree", tag).decode().splitlines()]
    blobs = [line.split("\t", 1)[1] for line in top if line.split()[1] == "blob"]
    docs = [line.split("\t", 1)[1] for line in git("ls-tree", f"{tag}:docs").decode().splitlines()
            if line.split()[1] == "blob"]
    paths = blobs + [f"docs/{name}" for name in docs] + list(DIRECTORIES)
    archive = git("archive", "--format=tar", tag, "--", *paths)
    tar_path = outdir / f"{tag}.tar"
    tar_path.write_bytes(archive)
    lines = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        for member in sorted(handle.getmembers(), key=lambda m: m.name):
            if member.isfile():
                data = handle.extractfile(member).read()
                lines.append(f"{hashlib.sha256(data).hexdigest()}  {member.name}\n")
    manifest = outdir / f"{tag}.sha256"
    manifest.write_text("".join(lines))
    return tar_path, manifest, len(lines)


if __name__ == "__main__":
    tar_path, manifest, count = build(sys.argv[1], sys.argv[2])
    print(f"{tar_path} {hashlib.sha256(tar_path.read_bytes()).hexdigest()} files={count}")
