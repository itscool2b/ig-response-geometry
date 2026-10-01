"""Current paper filenames stay synchronized without requiring a TeX install."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import build_paper


@pytest.fixture
def builder(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    paper = root / "paper"
    paper.mkdir(parents=True)
    for name in ("paper.tex", "appendix_aliasing.tex", "references.bib", "tmlr.sty", "tmlr.bst"):
        (paper / name).write_text("fixture", encoding="utf-8")
    archive = root / "legacy/pre-response-geometry"
    archive.mkdir(parents=True)
    (archive / "paper.pdf").write_bytes(b"historical PDF")
    for name in ("paper.pdf", "paper-revision.pdf"):
        (paper / name).write_bytes(b"previous identified PDF")
    monkeypatch.setattr(build_paper, "__file__", str(root / "scripts/build_paper.py"))
    monkeypatch.setattr(build_paper, "validate_revision_assets", lambda supplied: None)
    monkeypatch.setattr(build_paper.shutil, "which", lambda name: "/mock/" + name)
    calls = []

    def compile_mock(command, *, cwd, **kwargs):
        calls.append(command)
        (cwd / "revision.pdf").write_bytes(b"new compiled PDF")
        (cwd / "revision.log").write_text("clean final log", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="mock compilation\n")

    monkeypatch.setattr(build_paper.subprocess, "run", compile_mock)

    def run(*args):
        monkeypatch.setattr(sys, "argv", ["build_paper.py", *map(str, args)])
        build_paper.main()

    return root, calls, run


@pytest.mark.parametrize("selected", [None, "paper.pdf", "paper-revision.pdf"])
def test_identified_default_and_either_explicit_name_sync_both(builder, selected):
    root, calls, run = builder
    run(*([] if selected is None else ["--output", root / "paper" / selected]))
    assert len(calls) == 5
    assert all("TMLRAnonymous" not in str(command) for command in calls)
    for name in ("paper", "paper-revision"):
        assert (root / "paper" / (name + ".pdf")).read_bytes() == b"new compiled PDF"
    assert (root / "paper/paper.build.log").read_bytes() == (root / "paper/paper-revision.build.log").read_bytes()
    assert (root / "legacy/pre-response-geometry/paper.pdf").read_bytes() == b"historical PDF"


def test_anonymous_default_leaves_identified_and_history_unchanged(builder):
    root, calls, run = builder
    run("--anonymous")
    assert "\\def\\TMLRAnonymous{1}" in calls[0][-1]
    assert (root / "paper/paper-anonymous-draft.pdf").read_bytes() == b"new compiled PDF"
    for name in ("paper.pdf", "paper-revision.pdf"):
        assert (root / "paper" / name).read_bytes() == b"previous identified PDF"
    assert (root / "legacy/pre-response-geometry/paper.pdf").read_bytes() == b"historical PDF"


@pytest.mark.parametrize("name", ["paper.pdf", "paper-revision.pdf"])
def test_anonymous_cannot_overwrite_either_identified_name(builder, name):
    root, calls, run = builder
    with pytest.raises(SystemExit) as error:
        run("--anonymous", "--output", root / "paper" / name)
    assert error.value.code == 2 and not calls
    assert (root / "paper" / name).read_bytes() == b"previous identified PDF"


def test_identified_cannot_overwrite_reserved_anonymous_name(builder):
    root, calls, run = builder
    anonymous = root / "paper/paper-anonymous-draft.pdf"
    anonymous.write_bytes(b"previous anonymous PDF")
    with pytest.raises(SystemExit) as error:
        run("--output", anonymous)
    assert error.value.code == 2 and not calls
    assert anonymous.read_bytes() == b"previous anonymous PDF"
    for name in ("paper.pdf", "paper-revision.pdf"):
        assert (root / "paper" / name).read_bytes() == b"previous identified PDF"


@pytest.mark.parametrize("anonymous", [False, True])
@pytest.mark.parametrize("name", ["paper.pdf", "nested/new.pdf"])
def test_archive_rejects_existing_and_new_nested_outputs(builder, anonymous, name):
    root, calls, run = builder
    destination = root / "legacy/pre-response-geometry" / name
    with pytest.raises(SystemExit) as error:
        run(*(["--anonymous"] if anonymous else []), "--output", destination)
    assert error.value.code == 2 and not calls
    assert (root / "legacy/pre-response-geometry/paper.pdf").read_bytes() == b"historical PDF"
    if name != "paper.pdf":
        assert not destination.exists()


@pytest.mark.parametrize("anonymous", [False, True])
def test_custom_output_does_not_change_identified_aliases(builder, anonymous):
    root, calls, run = builder
    destination = root / "custom/current.pdf"
    run(*(["--anonymous"] if anonymous else []), "--output", destination)
    assert destination.read_bytes() == b"new compiled PDF"
    for name in ("paper.pdf", "paper-revision.pdf"):
        assert (root / "paper" / name).read_bytes() == b"previous identified PDF"


def test_failed_compilation_preserves_both_identified_outputs(builder, monkeypatch):
    root, calls, run = builder
    monkeypatch.setattr(build_paper.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="TeX failed"))
    with pytest.raises(RuntimeError, match="TeX failed"):
        run()
    for name in ("paper.pdf", "paper-revision.pdf"):
        assert (root / "paper" / name).read_bytes() == b"previous identified PDF"
