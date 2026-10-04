#!/usr/bin/env python3
"""Build the current research paper and protect the archived historical paper."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    expected.update(registry.get("reporting_inputs_sha256", {}))
    if "builder_sha256" in registry:
        expected["scripts/build_revision_assets.py"] = registry["builder_sha256"]
    expected["paper/paper.tex"] = registry["manuscript_source_sha256"]
    if any(not (root / name).resolve().is_relative_to(root.resolve()) for name in expected):
        raise RuntimeError("Manuscript lineage contains a path outside the repository")
    mismatches = [name for name, digest in expected.items()
                  if not (root / name).is_file()
                  or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    if mismatches:
        raise RuntimeError("Stale or missing manuscript assets: " + ", ".join(mismatches)
                           + ". Run scripts/build_revision_assets.py before compiling.")


def resolve_engine(requested: str, source: str) -> tuple[str, str]:
    """Choose an installed engine without installing or downloading anything."""
    if requested == "auto":
        if "\\usepackage{fontspec}" not in source and shutil.which("pdflatex"):
            requested = "pdflatex"
        else:
            cached = Path.home() / ".cache/tmlr-tectonic-0.17.0/tectonic.exe"
            requested = (shutil.which("tectonic") or os.environ.get("TECTONIC")
                         or (str(cached) if cached.is_file() else "tectonic"))
    kind = Path(requested).stem.lower()
    if kind not in {"pdflatex", "tectonic"}:
        raise ValueError("--engine must be auto, pdflatex, tectonic, or a path to one of those executables")
    executable = shutil.which(requested)
    if executable is None and Path(requested).is_file():
        executable = str(Path(requested).resolve())
    if not executable:
        raise ValueError(f"{requested} is unavailable; supply --engine with an installed Tectonic path")
    if kind == "pdflatex" and "\\usepackage{fontspec}" in source:
        raise ValueError("This manuscript uses fontspec; select Tectonic instead of pdfLaTeX")
    if kind == "pdflatex" and not shutil.which("bibtex"):
        raise ValueError("bibtex is required with the pdflatex engine")
    return kind, executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anonymous", action="store_true", help="Build an anonymous audit copy and suppress identifying links/acknowledgments.")
    parser.add_argument("--output", type=Path, help="Destination PDF; either current identified filename also updates its compatibility alias.")
    parser.add_argument("--engine", default="auto", help="auto, pdflatex, tectonic, or an executable path. Tectonic uses only its existing cache.")
    parser.add_argument("--allow-resource-downloads", action="store_true", help="Allow Tectonic to fetch missing public TeX resources into its local cache; manuscript files remain local")
    parser.add_argument("--keep-intermediates-dir", type=Path, help="Fresh local directory for the final AUX, BBL, logs and build identities, for document conversion and review")
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
    if args.keep_intermediates_dir:
        args.keep_intermediates_dir = args.keep_intermediates_dir.resolve()
        if args.keep_intermediates_dir.exists():
            parser.error("--keep-intermediates-dir must be a fresh directory")
        if args.keep_intermediates_dir.is_relative_to(archive.resolve()):
            parser.error("Build intermediates must not be written inside the historical archive")
    validate_revision_assets(root)
    registry_path = paper / "figures_revision/lineage.json"
    registry_before = registry_path.read_bytes() if registry_path.is_file() else None
    source = (paper / "paper.tex").read_text(encoding="utf-8")
    try:
        engine, executable = resolve_engine(args.engine, source)
    except ValueError as error:
        parser.error(str(error))
    if args.allow_resource_downloads and engine != "tectonic":
        parser.error("--allow-resource-downloads applies only to Tectonic")

    with tempfile.TemporaryDirectory(prefix="research-paper-") as scratch:
        work = Path(scratch)
        for name in ("paper.tex", "appendix_aliasing.tex", "references.bib", "tmlr.sty", "tmlr.bst"):
            shutil.copyfile(paper / name, work / name)
        for name in ("nhsjs.bst",):
            if (paper / name).is_file():
                shutil.copyfile(paper / name, work / name)
        for directory in ("figures", "figures_revision", "tables_revision"):
            if (paper / directory).exists():
                shutil.copytree(paper / directory, work / directory)
        entry = (r"\def\TMLRAnonymous{1}" if args.anonymous else "") + r"\input{paper.tex}"
        if engine == "tectonic":
            (work / "revision.tex").write_text(entry + "\n", encoding="utf-8")
            commands = ([executable, *([] if args.allow_resource_downloads else ["--only-cached"]), "--untrusted", "--keep-logs",
                         "--keep-intermediates", "--reruns", "3", "revision.tex"],)
        else:
            latex = [executable, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "-jobname=revision", entry]
            commands = (latex, ["bibtex", "revision"], latex, latex, latex)
        transcripts = []
        for command in commands:
            result = subprocess.run(command, cwd=work, text=True, encoding="utf-8", errors="replace",
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            transcripts.append(result.stdout)
            if result.returncode:
                raise RuntimeError(result.stdout[-20000:])
        log = (work / "revision.log").read_text(encoding="utf-8", errors="replace")
        if "There were undefined references" in log or ("Citation" in log and "undefined" in log) or "Label(s) may have changed" in log:
            raise RuntimeError("Undefined reference or citation in final LaTeX pass.\n" + log[-20000:])
        validate_revision_assets(root)
        if ((paper / "paper.tex").read_bytes() != (work / "paper.tex").read_bytes()
                or (registry_before is not None and registry_path.read_bytes() != registry_before)):
            raise RuntimeError("Manuscript or lineage changed during compilation; refresh assets and rebuild")
        if args.keep_intermediates_dir:
            retained = args.keep_intermediates_dir
            retained.mkdir(parents=True, exist_ok=False)
            for extension in ("aux", "bbl", "blg", "log", "out", "xdv"):
                intermediate = work / ("revision." + extension)
                if intermediate.is_file():
                    shutil.copyfile(intermediate, retained / intermediate.name)
            identities = {
                "engine": engine, "executable": executable, "anonymous": args.anonymous,
                "resource_downloads_allowed": args.allow_resource_downloads,
                "manuscript_sha256": hashlib.sha256((work / "paper.tex").read_bytes()).hexdigest(),
                "pdf_sha256": hashlib.sha256((work / "revision.pdf").read_bytes()).hexdigest(),
                "intermediates_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in sorted(retained.iterdir())},
            }
            (retained / "build.json").write_text(json.dumps(identities, indent=2) + "\n", encoding="utf-8")
        for destination in dict.fromkeys(destinations):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / "revision.pdf", destination)
            destination.with_suffix(".build.log").write_text("\n".join(transcripts), encoding="utf-8")
        warnings = [line for line in log.splitlines() if "Warning" in line or "Overfull" in line or "Underfull" in line]
        print("Built " + ", ".join(str(path) for path in dict.fromkeys(destinations)))
        print("\n".join(warnings) if warnings else "No final-pass LaTeX warnings or overfull/underfull boxes.")


if __name__ == "__main__":
    main()
