"""Curated export integrity, without executing a model or publishing an artifact."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts import package_research_supplement as package
from scripts.validate_template_provenance import FILE_SHA256


@pytest.mark.parametrize("name", ["../LICENSE", "data/../LICENSE", "/LICENSE", "C:/LICENSE", "data\\x",
                                  "data//x", "./LICENSE", "data/./x", "data/x\n", ""])
def test_noncanonical_members_are_rejected(name):
    with pytest.raises(ValueError):
        package.member_name(name)


def test_local_and_relative_dependencies_are_found_without_importing_code():
    available={"analysis/revision/core.py", "pipeline.py", "scripts/validate_fp32_probe.py"}
    local,external=package.import_dependencies("analysis/revision/analyze.py",
        b"from .core import check\nimport pipeline\nfrom scripts import validate_fp32_probe\nimport numpy\nimport os\n",available)
    assert local==available
    assert external=={"numpy"}


def test_curated_python_closure_includes_runtime_hashed_implementation_sources():
    import weight_arrangement_control

    root = Path(__file__).resolve().parents[1]
    available = set(subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "*.py"], text=True).splitlines())
    selected = {path for path in package.ENTRYPOINTS if path.endswith(".py")}
    selected.update(path for path in available if path.startswith("tests/") and path not in package.OMITTED_TESTS)
    selected.update(path for path in package.PENDING_ADDITIONS if path.endswith(".py") and path in available)
    queue = list(selected)
    while queue:
        path = queue.pop()
        dependencies, _ = package.import_dependencies(path, (root / path).read_bytes(), available)
        queue.extend(dependencies - selected)
        selected.update(dependencies)
    # Exercise the actual runtime manifest rather than duplicating its source list.
    required = set(weight_arrangement_control.implementation_hashes())
    assert required <= selected, f"Runtime-hashed source omitted from supplement: {sorted(required - selected)}"


def test_privacy_scan_expands_gzip_and_distinguishes_examples():
    result=package.privacy_findings({"data.json.gz":gzip.compress(b'{"owner":"Arjun Bajpai"}'),
        "guide.md":b"Use /workspace/example for a new run", "figure.pdf":b"\xff"})
    assert {r["category"] for r in result}=={"identifying_text","machine_path_or_portable_example","binary_requires_format_review"}


def test_private_audit_tokens_are_supplied_externally_and_scan_compressed_content():
    content={"record.json.gz":gzip.compress(b'{"owner":"Example-Private-User"}')}
    assert package.privacy_findings(content)==[]
    findings=package.privacy_findings(content,["example-private-user"])
    assert findings[0]["identifiers"]==["example-private-user"]
    with pytest.raises(ValueError,match="nonempty strings"):
        package.audit_tokens([None])
    assert package.audit_tokens([" EXAMPLE-PRIVATE-USER ","example-private-user"])==("example-private-user",)


def test_snapshot_uses_committed_blobs_not_working_or_private_files(tmp_path):
    def git(*args):
        return subprocess.run(["git","-C",str(tmp_path),*args],check=True,stdout=subprocess.PIPE).stdout
    git("init","-q")
    (tmp_path/"file.py").write_text("value = 1\n")
    git("add","file.py")
    git("-c","user.name=Fixture","-c","user.email=fixture@example.invalid","commit","-qm","fixture")
    (tmp_path/"file.py").write_text("value = 2\n")
    (tmp_path/"private.txt").write_text("not tracked")
    commit,index=package.snapshot(tmp_path,"HEAD")
    assert set(index)=={"file.py"}
    assert b"value = 1" in package.git(tmp_path,"cat-file","blob",index["file.py"][1])


def test_build_preserves_notices_and_creates_deterministic_archive(tmp_path,monkeypatch):
    sampler=b'"""Original sampler license:\nMIT License\nCopyright (c) 2024 TSAIL group\nPermission notice preserved.\n"""\n'
    contents={"LICENSE":b"MIT License\nCopyright (c) 2026 Arjun Bajpai\n",
              "rdt_sampling.py":sampler,"data/fixture.jsonl":b'{"value":1}\n'}
    root = Path(__file__).resolve().parents[1]
    for name in (*FILE_SHA256, "tmlr-source.json"):
        contents["paper/"+name] = (root/"paper"/name).read_bytes()
    def collect(*args):
        return dict(contents),dict(commit="a"*40,raw_inputs=1,pending_uncommitted_additions=[])
    monkeypatch.setattr(package,"collect",collect)
    first=package.build(tmp_path,"HEAD",tmp_path/"a")
    second=package.build(tmp_path,"HEAD",tmp_path/"b")
    assert first["archive"]["sha256"]==second["archive"]["sha256"]
    with zipfile.ZipFile(tmp_path/"a"/first["archive"]["file"]) as archive:
        assert archive.read("LICENSE")==contents["LICENSE"]
        assert archive.read("rdt_sampling.py")==sampler
        assert archive.read("paper/tmlr.bst")==contents["paper/tmlr.bst"]
        assert b"LPPL-1.0-or-later" in archive.read("SUPPLEMENT_README.md")
        assert b"Copyright (c) 2024 TSAIL group" in archive.read("THIRD_PARTY_NOTICES.txt")
        manifest=json.loads(archive.read("SUPPLEMENT_MANIFEST.json"))
        assert manifest["status"]=="working_draft_not_submission"
        for path,record in manifest["members"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest()==record["sha256"]
    with pytest.raises(FileExistsError):
        package.build(tmp_path,"HEAD",tmp_path/"a")


def test_no_anonymous_license_replacement_or_rights_image_in_declared_scope():
    assert not (set(package.ENTRYPOINTS)|set(package.DOCUMENTS)) & package.RIGHTS_EXCLUDED
    assert "LICENSE" in package.DOCUMENTS


def test_private_candidate_changes_only_reviewed_members_and_preserves_terms():
    paper=b"\\author{\\name Arjun Bajpai\n\\addr Independent Researcher}\nScientific body.\n"
    license=b"MIT License\nCopyright (c) 2026 Arjun Bajpai\nPermission and disclaimer stay intact.\n"
    original={"LICENSE":license,"README.md":b"The sole current author is Arjun Bajpai.\n`CITATION.cff` identifies the deposit.\nPinned runtime recipe survives.\n",
        "paper/paper.tex":paper,"paper/figures_revision/lineage.json":json.dumps({"manuscript_source_sha256":package.digest(paper),"entries":[1]}).encode(),
        "rdt_sampling.py":b"Copyright (c) 2024 TSAIL group\nMIT permission.","data/fixture.jsonl":b'{"value":1}\n'}
    result,audit=package.anonymous_candidate(original,[])
    assert original["LICENSE"]==license
    assert result["LICENSE"]==license.replace(b"Arjun Bajpai",b"Anonymous author")
    assert result["rdt_sampling.py"]==original["rdt_sampling.py"]
    assert result["data/fixture.jsonl"]==original["data/fixture.jsonl"]
    assert b"Pinned runtime recipe survives." in result["README.md"]
    assert json.loads(result["paper/figures_revision/lineage.json"])["manuscript_source_sha256"]==package.digest(result["paper/paper.tex"])
    assert audit["licensing_status"]=="proposed_review_copy_attribution_not_approved_for_external_distribution"
    contaminated=dict(original,unexpected=b"Example-private-user")
    with pytest.raises(ValueError,match="identifying text"):
        package.anonymous_candidate(contaminated,["example-private-user"])


def test_private_candidate_cannot_be_written_inside_public_repository(tmp_path):
    with pytest.raises(ValueError,match="outside the canonical public"):
        package.build(tmp_path,"HEAD",tmp_path/"candidate",private_anonymous_candidate=True)


def test_research_packager_rejects_actual_committed_member_bytes_before_output(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    contents = {'paper/'+name: (root/'paper'/name).read_bytes()
                for name in (*FILE_SHA256, 'tmlr-source.json')}
    contents['paper/tmlr.sty'] += b'Changed committed blob'
    monkeypatch.setattr(package, 'collect', lambda *args: (contents, {}))
    output = tmp_path/'research-package'
    with pytest.raises(ValueError, match='template byte mismatch'):
        package.build(tmp_path, 'HEAD', output)
    assert not output.exists()


def test_working_snapshot_only_adds_reviewed_new_paths_and_reads_current_bytes(tmp_path):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True, stdout=subprocess.PIPE).stdout
    git('init', '-q')
    (tmp_path / 'tracked.py').write_text('value = 1\n')
    git('add', 'tracked.py')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
    (tmp_path / 'tracked.py').write_bytes(b'value = 2\n')
    report = tmp_path / 'analysis/manuscript_revision/report.py'
    report.parent.mkdir(parents=True)
    report.write_text('value = 3\n')
    exporter = tmp_path / 'scripts/export_online_docx.py'
    exporter.parent.mkdir(parents=True)
    exporter.write_text('def export(): pass\n')
    private_review = tmp_path / 'docs/manuscript_revision/review_build.md'
    private_review.parent.mkdir(parents=True)
    private_review.write_text('private review narrative\n')
    reviewed_documents = (
        'docs/manuscript_revision/final_local_verification.md',
        'docs/manuscript_revision/final_semantic_audit.md',
        'docs/manuscript_revision/claim_map.md',
        'docs/manuscript_revision/factual_corrections.md',
        'docs/manuscript_revision/revision_checklist.md',
    )
    for name in reviewed_documents:
        (tmp_path / name).write_text('Reviewed document\n')
    (tmp_path / 'private.txt').write_text('private material')
    _, index = package.working_snapshot(tmp_path)
    assert 'analysis/manuscript_revision/report.py' in index
    assert 'scripts/export_online_docx.py' in index
    assert set(reviewed_documents) <= set(index)
    assert 'docs/manuscript_revision/review_build.md' not in index
    assert 'private.txt' not in index
    assert package.working_file(tmp_path, 'tracked.py') == b'value = 2\n'
    assert index['analysis/manuscript_revision/report.py'][1] is None


def working_collection_fixture(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(['git', '-C', str(tmp_path), *args], check=True, stdout=subprocess.PIPE).stdout
    files = {
        'module.py': b'value = 1\n', 'paper/paper.tex': b'legacy manuscript\n',
        'data/fixture.jsonl': b'{"value":1}\n',
        'analysis/revision/results/2026-09-30-v2/provenance.json': b'{"artifacts_sha256":{}}\n',
    }
    files['analysis/revision/input_manifest.json'] = json.dumps({'files': [
        {'path': 'data/fixture.jsonl', 'sha256': hashlib.sha256(files['data/fixture.jsonl']).hexdigest(),
         'git_crlf_checkout_sha256': hashlib.sha256(files['data/fixture.jsonl'].replace(b'\n', b'\r\n')).hexdigest()}]}).encode()
    git('init', '-q')
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    git('add', '.')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
    monkeypatch.setattr(package, 'ENTRYPOINTS', ('module.py',))
    monkeypatch.setattr(package, 'DOCUMENTS', ('paper/paper.tex',))
    monkeypatch.setattr(package, 'PAPER_ASSETS', ())
    monkeypatch.setattr(package, 'RESULT_FILES', ('provenance.json',))
    monkeypatch.setattr(package, 'PENDING_ADDITIONS', ())
    return files


def test_working_collection_freezes_current_bytes_and_records_base_separately(tmp_path, monkeypatch):
    working_collection_fixture(tmp_path, monkeypatch)
    (tmp_path / 'module.py').write_bytes(b'value = 2\n')
    contents, audit = package.collect(tmp_path, 'HEAD', working_copy=True)
    assert contents['module.py'] == b'value = 2\n'
    assert audit['source_kind'] == 'working_copy_snapshot'
    assert audit['source_members']['module.py']['sha256'] == hashlib.sha256(b'value = 2\n').hexdigest()
    assert 'base_git_blob' in audit['source_members']['module.py']
    assert 'git_blob' not in audit['source_members']['module.py']
    assert len(audit['snapshot_sha256']) == 64
    committed, _ = package.collect(tmp_path, 'HEAD')
    assert committed['module.py'] == b'value = 1\n'


def test_canonical_package_omits_local_historical_replay_dependency(tmp_path, monkeypatch):
    working_collection_fixture(tmp_path, monkeypatch)
    local_files = {
        'scripts/assemble_approved_manuscript.py': b'BASELINE = "local_revision/private-baseline"\n',
        'tests/test_approved_assembly.py': b'from scripts import assemble_approved_manuscript\n',
    }
    for name, data in local_files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    subprocess.run(['git', '-C', str(tmp_path), 'add', *local_files], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Fixture',
                    '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'local review files'], check=True)
    contents, _ = package.collect(tmp_path, 'HEAD', working_copy=True)
    assert contents['paper/paper.tex'] == b'legacy manuscript\n'
    assert not set(local_files) & set(contents)


def test_package_guide_compiles_already_assembled_source(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    contents = {'paper/' + name: (root / 'paper' / name).read_bytes()
                for name in (*FILE_SHA256, 'tmlr-source.json')}
    contents['rdt_sampling.py'] = b'"""Original sampler license:\nMIT License\nPermission notice preserved.\n"""\n'
    contents['analysis/manuscript_revision/report.py'] = b'# reporting entry point\n'
    monkeypatch.setattr(package, 'collect', lambda *args: (dict(contents), {}))
    audit = package.build(tmp_path, 'HEAD', tmp_path / 'package')
    with zipfile.ZipFile(tmp_path / 'package' / audit['archive']['file']) as archive:
        guide = archive.read('SUPPLEMENT_README.md').decode()
    assert 'build entry point is the already assembled paper/paper.tex' in guide
    assert 'Compile the current source directly.' in guide
    assert 'preparation records are not required and are not included' in guide
    assert 'approval ledger' not in guide and 'final coverage' not in guide
    assert 'prior verification history' not in guide
    assert 'python scripts/build_paper.py --engine tectonic' in guide
    assert 'python scripts/assemble_approved_manuscript.py' not in guide


def test_working_collection_preserves_only_declared_exact_checkout_bytes(tmp_path, monkeypatch):
    files = working_collection_fixture(tmp_path, monkeypatch)
    path = tmp_path / 'data/fixture.jsonl'
    checkout = files['data/fixture.jsonl'].replace(b'\n', b'\r\n')
    path.write_bytes(checkout)
    contents, audit = package.collect(tmp_path, 'HEAD', working_copy=True)
    assert contents['data/fixture.jsonl'] == checkout
    assert audit['raw_input_representations']['data/fixture.jsonl'] == 'declared_git_crlf_checkout'
    path.write_bytes(checkout + b'\n')
    with pytest.raises(ValueError, match='Raw input differs'):
        package.collect(tmp_path, 'HEAD', working_copy=True)


@pytest.mark.parametrize('kind', ['raw', 'manifest', 'frozen_result'])
def test_working_collection_rejects_rewritten_historical_evidence(tmp_path, monkeypatch, kind):
    working_collection_fixture(tmp_path, monkeypatch)
    if kind == 'raw':
        (tmp_path / 'data/fixture.jsonl').write_bytes(b'{"value":2}\n')
    elif kind == 'manifest':
        (tmp_path / 'analysis/revision/input_manifest.json').write_text('{"files":[]}')
    else:
        (tmp_path / 'analysis/revision/results/2026-09-30-v2/provenance.json').write_text('{"artifacts_sha256":{},"changed":true}')
    with pytest.raises(ValueError, match='manifest|Frozen scientific artifact'):
        package.collect(tmp_path, 'HEAD', working_copy=True)


def test_working_collection_rejects_concurrent_changes(tmp_path, monkeypatch):
    working_collection_fixture(tmp_path, monkeypatch)
    original = package.working_file
    calls = 0

    def unstable(root, name):
        nonlocal calls
        data = original(root, name)
        if name == 'module.py':
            calls += 1
            if calls > 1:
                return b'value = 3\n'
        return data

    monkeypatch.setattr(package, 'working_file', unstable)
    with pytest.raises(ValueError, match='changed while its snapshot was captured'):
        package.collect(tmp_path, 'HEAD', working_copy=True)
