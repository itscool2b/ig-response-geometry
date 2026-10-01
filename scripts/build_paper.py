#!/usr/bin/env python3
"""Build the current research paper and protect the archived historical paper."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def validate_revision_assets(root: Path) -> None:
    """Reject stale manuscript/display inputs before producing a candidate."""
    # Support both module imports and direct `python scripts/build_paper.py`.
    if __package__:
        from .validate_template_provenance import validate_template_provenance
    else:
        from validate_template_provenance import validate_template_provenance
    validate_template_provenance(root)
    registry_path = root / "paper/figures_revision/lineage.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    expected = dict(registry["generated_sha256"])
    expected.update(registry.get("manuscript_inputs_sha256", {}))
    expected.update(registry.get("numerical_case_inputs_sha256", {}))
    expected.update(registry.get("strengthening_inputs_sha256", {}))
    expected.update(registry.get("influence_inputs_sha256", {}))
    expected["paper/paper.tex"] = registry["manuscript_source_sha256"]
    mismatches = [name for name, digest in expected.items()
                  if not (root / name).is_file()
                  or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError("Stale or missing manuscript assets: " + ", ".join(mismatches)
                           + ". Run scripts/build_revision_assets.py before compiling.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anonymous", action="store_true", help="Build an anonymous audit copy and suppress identifying links/acknowledgments.")
    parser.add_argument("--output", type=Path, help="Destination PDF; either current identified filename also updates its compatibility alias.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paper = root / "paper"
    requested = args.output or paper / ("paper-anonymous-draft.pdf" if args.anonymous else "paper.pdf")
    output = requested.resolve()
    identified = [(paper / name).resolve() for name in ("paper.pdf", "paper-revision.pdf")]
    if args.anonymous and output in identified:
        parser.error("An anonymous build cannot overwrite either current identified PDF.")
    destinations = identified if not args.anonymous and output in identified else [output]
    if not args.anonymous and (paper / "paper-anonymous-draft.pdf").resolve() in destinations:
        parser.error("The reserved anonymous PDF requires --anonymous.")
    archive = root / "legacy/pre-response-geometry"
    if (requested.absolute().is_relative_to(archive.absolute())
            or any(path.is_relative_to(archive.resolve()) for path in destinations)):
        parser.error("The historical legacy/pre-response-geometry archive must not be overwritten.")
    validate_revision_assets(root)
    for executable in ("pdflatex", "bibtex"):
        if not shutil.which(executable):
            parser.error(f"{executable} is required (TeX Live with latex-extra, fonts-recommended, science and lmodern).")

    with tempfile.TemporaryDirectory(prefix="tmlr-paper-") as scratch:
        work = Path(scratch)
        for name in ("paper.tex", "appendix_aliasing.tex", "references.bib", "tmlr.sty", "tmlr.bst"):
            shutil.copyfile(paper / name, work / name)
        for directory in ("figures", "figures_revision", "tables_revision"):
            if (paper / directory).exists():
                shutil.copytree(paper / directory, work / directory)
        entry = (r"\def\TMLRAnonymous{1}" if args.anonymous else "") + r"\input{paper.tex}"
        latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-jobname=revision", entry]
        commands = (latex, ["bibtex", "revision"], latex, latex, latex)
        transcripts = []
        for command in commands:
            result = subprocess.run(command, cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            transcripts.append(result.stdout)
            if result.returncode:
                raise RuntimeError(result.stdout[-20000:])
        log = (work / "revision.log").read_text(errors="replace")
        if "There were undefined references" in log or ("Citation" in log and "undefined" in log) or "Label(s) may have changed" in log:
            raise RuntimeError("Undefined reference or citation in final LaTeX pass.\n" + log[-20000:])
        for destination in dict.fromkeys(destinations):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / "revision.pdf", destination)
            destination.with_suffix(".build.log").write_text("\n".join(transcripts), encoding="utf-8")
        warnings = [line for line in log.splitlines() if "Warning" in line or "Overfull" in line]
        print("Built " + ", ".join(str(path) for path in dict.fromkeys(destinations)))
        print("\n".join(warnings) if warnings else "No final-pass warnings or overfull boxes.")


if __name__ == "__main__":
    main()
