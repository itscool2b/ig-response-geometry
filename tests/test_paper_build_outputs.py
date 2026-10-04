"""Current paper filenames stay synchronized without requiring a TeX install."""
from pathlib import Path
from types import SimpleNamespace
import json
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


def test_tectonic_build_uses_cache_only_and_retains_anonymous_wrapper(builder, monkeypatch):
    root, calls, run = builder
    original = build_paper.subprocess.run

    def inspect(command, *, cwd, **kwargs):
        assert (cwd / "revision.tex").read_text() == "\\def\\TMLRAnonymous{1}\\input{paper.tex}\n"
        return original(command, cwd=cwd, **kwargs)

    monkeypatch.setattr(build_paper.subprocess, "run", inspect)
    run("--anonymous", "--engine", "tectonic")
    assert len(calls) == 1
    assert {"--only-cached", "--untrusted", "--keep-logs", "--reruns"} <= set(calls[0])
    assert (root / "paper/paper-anonymous-draft.pdf").read_bytes() == b"new compiled PDF"


def test_fontspec_cannot_be_silently_compiled_with_pdflatex(builder):
    root, calls, run = builder
    (root / "paper/paper.tex").write_text(r"\usepackage{fontspec}")
    with pytest.raises(SystemExit):
        run("--engine", "pdflatex")
    assert not calls


def test_explicit_tectonic_path_is_accepted_without_path_install(tmp_path, monkeypatch):
    engine = tmp_path / "installed" / "tectonic.exe"
    engine.parent.mkdir()
    engine.write_bytes(b"fixture")
    monkeypatch.setattr(build_paper.shutil, "which", lambda value: None)
    kind, actual = build_paper.resolve_engine(str(engine), r"\usepackage{fontspec}")
    assert kind == "tectonic" and actual == str(engine.resolve())


def test_missing_engine_fails_before_compilation(builder, monkeypatch):
    root, calls, run = builder
    monkeypatch.setattr(build_paper.shutil, "which", lambda value: None)
    with pytest.raises(SystemExit):
        run("--engine", root / "missing/tectonic.exe")
    assert not calls


def test_explicit_resource_fetch_flag_and_intermediate_identities(builder):
    root, calls, run = builder
    retained = root / 'local-review/identified'
    run('--engine', 'tectonic', '--allow-resource-downloads', '--keep-intermediates-dir', retained)
    assert '--only-cached' not in calls[0]
    record = json.loads((retained / 'build.json').read_text())
    assert record['engine'] == 'tectonic' and record['resource_downloads_allowed']
    assert 'revision.log' in record['intermediates_sha256']
    assert record['pdf_sha256']
    before = len(calls)
    with pytest.raises(SystemExit):
        run('--keep-intermediates-dir', retained)
    assert len(calls) == before


def test_intermediates_cannot_be_written_under_protected_history(builder):
    root, calls, run = builder
    with pytest.raises(SystemExit):
        run('--keep-intermediates-dir', root / 'legacy/pre-response-geometry/new')
    assert not calls


def test_concurrent_source_change_does_not_publish_stale_pdf(builder, monkeypatch):
    root, _, run = builder
    compile_original = build_paper.subprocess.run

    def changing(command, *, cwd, **kwargs):
        result = compile_original(command, cwd=cwd, **kwargs)
        (root / 'paper/paper.tex').write_text('changed during compilation')
        return result

    monkeypatch.setattr(build_paper.subprocess, 'run', changing)
    with pytest.raises(RuntimeError, match='changed during compilation'):
        run('--engine', 'tectonic')
    assert (root / 'paper/paper.pdf').read_bytes() == b'previous identified PDF'
