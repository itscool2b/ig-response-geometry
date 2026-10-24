"""The repository packaging tools: the regeneration-code builder and the reviewer archive.

Like tests/test_research_supplement.py, these test tools that build packages from the
repository, so the research supplement omits them.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from revision_numerics import SOURCE_FILES


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_tree(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


@pytest.mark.skipif(subprocess.run(["git", "rev-parse", "--verify", "-q", "nhsjs-r1-3^{commit}"],
                                   capture_output=True).returncode != 0, reason="frozen tag nhsjs-r1-3 not present")
def test_regeneration_code_holds_the_pinned_sources_and_is_deterministic(tmp_path):
    from scripts import build_regeneration_code
    protocol = json.loads(Path("docs/nhsjs-revision/protocols/r1/production/r1-p.json").read_text())
    outputs = {}
    for variant, anonymous in (("identified", False), ("anonymous", True)):
        paths = [build_regeneration_code.build("nhsjs-r1-3", tmp_path / f"{variant}-{run}", anonymous)[0] for run in (0, 1)]
        assert paths[0].read_bytes() == paths[1].read_bytes()
        with tarfile.open(paths[0]) as archive:
            outputs[variant] = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
    members = outputs["anonymous"]
    assert set(SOURCE_FILES) <= set(members)
    assert {name: sha(members[name]) for name in SOURCE_FILES} == protocol["banks"]["P"]["source_sha256"]
    assert {"collect_contexts.py", "per_step_ig.py", "configs/base_170m.yaml", "THIRD_PARTY_NOTICES.txt"} <= set(members)
    assert [n for n in members if outputs["identified"][n] != members[n]] == ["LICENSE"]
    assert b"Anonymous author" in members["LICENSE"]


def test_reviewer_archive_keeps_records_lists_tensors_and_screens_identifiers(tmp_path):
    from scripts import build_reviewer_archive as archive
    phase = tmp_path / "runs/P/e0000-c001/bf16"
    write_tree(phase, {"record.json": b"{}", "rankings.json": b"{}", "maps.pt": b"tensor bytes"})
    (phase / "completion.json").write_text(json.dumps({"files_sha256": {"record.json": sha(b"{}"),
                                                                        "rankings.json": sha(b"{}"),
                                                                        "maps.pt": sha(b"tensor bytes")}}))
    write_tree(tmp_path / "results", {"manifest.json": b"{}"})
    write_tree(tmp_path / "identity", {"pod1.json": b"{}"})
    write_tree(tmp_path / "protocols", {"plan.json": b"{}"})
    write_tree(tmp_path / "month8", {"README.md": b"bank"})
    write_tree(tmp_path / "mf2", {name: b"facts" for name in archive.INVENTORY_FILES})
    sources = dict(protocols=tmp_path / "protocols", month8=tmp_path / "month8", inventories=tmp_path / "mf2")
    report = archive.build(tmp_path / "out", results=tmp_path / "results", runs=tmp_path / "runs",
                           identity=tmp_path / "identity", anonymous=True, sources=sources)
    out = tmp_path / "out"
    assert (out / "runs/P/e0000-c001/bf16/record.json").is_file() and not (out / "runs/P/e0000-c001/bf16/maps.pt").exists()
    assert (out / "runs/TENSORS.sha256").read_text() == f"{sha(b'tensor bytes')}  P/e0000-c001/bf16/maps.pt\n"
    assert report["tensors_listed"] == 1 and "MANIFEST.sha256" not in (out / "MANIFEST.sha256").read_text()
    write_tree(tmp_path / "identity", {"named.json": ('{"author": "%s"}' % archive.ARCHIVE_IDENTIFIERS[0].title()).encode()})
    with pytest.raises(ValueError, match="Identifying text"):
        archive.build(tmp_path / "out2", results=tmp_path / "results", runs=tmp_path / "runs",
                      identity=tmp_path / "identity", anonymous=True, sources=sources)
    assert not (tmp_path / "out2").exists()


def test_reviewer_archive_keeps_worker_logs_and_failures_and_scans_every_encoding(tmp_path):
    from scripts import build_reviewer_archive as archive
    (tmp_path / "runs/P/e0000-c002/attempts/bf16-2").mkdir(parents=True)
    write_tree(tmp_path / "runs/P", {"workers/worker-00.jsonl": b'{"event": "worker_end"}\n',
                                     "e0000-c001/attempts/bf16-1/failure.json": b'{"error": "x"}',
                                     "e0000-c001/attempts/bf16-1/partial.pt": b"partial"})
    for name, files in (("results", {"manifest.json": b"{}"}), ("identity", {"pod1.json": b"{}"}),
                        ("protocols", {"plan.json": b"{}"}), ("month8", {"README.md": b"bank"}),
                        ("mf2", {n: b"facts" for n in archive.INVENTORY_FILES})):
        write_tree(tmp_path / name, files)
    sources = dict(protocols=tmp_path / "protocols", month8=tmp_path / "month8", inventories=tmp_path / "mf2")
    kwargs = dict(results=tmp_path / "results", runs=tmp_path / "runs", identity=tmp_path / "identity",
                  anonymous=True, sources=sources)
    report = archive.build(tmp_path / "out", **kwargs)
    out = tmp_path / "out/runs"
    assert (out / "P/workers/worker-00.jsonl").is_file() and (out / "P/e0000-c001/attempts/bf16-1/failure.json").is_file()
    assert (out / "ATTEMPTS.sha256").read_text() == f"{sha(b'partial')}  P/e0000-c001/attempts/bf16-1/partial.pt\n"
    assert (out / "ATTEMPT_DIRECTORIES.txt").read_text() == ("P/e0000-c001/attempts/bf16-1/\tfailed (failure.json kept)\n"
                                                             "P/e0000-c002/attempts/bf16-2/\tinterrupted, empty\n")
    assert report["attempt_failures"] == 1
    cases = {"latin1.bin": "caf\xe9 Bajpai".encode("latin-1"), "path.json": b'{"cwd": "/Users/someone/project"}',
             "token.txt": b"contact: secret-token-7"}
    for k, (name, data) in enumerate(cases.items()):
        write_tree(tmp_path / f"identity{k}", {name: data})
        with pytest.raises(ValueError, match="Identifying text"):
            archive.build(tmp_path / f"bad{k}", **dict(kwargs, identity=tmp_path / f"identity{k}"),
                          identifiers=["SECRET-token-7"])
        assert not (tmp_path / f"bad{k}").exists()
    write_tree(tmp_path / "identity-pod", {"pod.json": b'{"cwd": "/workspace/nhsjs", "home": "/root/.cache"}'})
    archive.build(tmp_path / "pod-ok", **dict(kwargs, identity=tmp_path / "identity-pod"))
