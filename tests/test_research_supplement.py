"""Curated export integrity, without executing a model or publishing an artifact."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from scripts import package_research_supplement as package


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
    def collect(*args):
        return dict(contents),dict(commit="a"*40,raw_inputs=1,pending_uncommitted_additions=[])
    monkeypatch.setattr(package,"collect",collect)
    first=package.build(tmp_path,"HEAD",tmp_path/"a")
    second=package.build(tmp_path,"HEAD",tmp_path/"b")
    assert first["archive"]["sha256"]==second["archive"]["sha256"]
    with zipfile.ZipFile(tmp_path/"a"/first["archive"]["file"]) as archive:
        assert archive.read("LICENSE")==contents["LICENSE"]
        assert archive.read("rdt_sampling.py")==sampler
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
