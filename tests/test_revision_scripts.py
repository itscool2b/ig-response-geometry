import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

import pytest

from scripts import build_m8_overlay, make_revision_protocol
from revision_numerics import PROTOCOL_SCHEMA, SOURCE_FILES


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_tree(root, files):
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def test_overlay_swaps_only_the_replay_checked_bytes(tmp_path):
    month8 = {"pipeline.py": b"a = 1\r\n# old comment\r\n", "per_step_attribution.py": b"b = 2\n",
              "rdt_sampling.py": b"c = 3\n", "checkpoint_contract.py": b"d = 4\n"}
    frozen = dict(month8, **{"pipeline.py": b"a = 1\n# new comment\n", "revision_numerics.py": b"x = 0\n"})
    write_tree(tmp_path / "frozen", frozen)
    write_tree(tmp_path / "snapshot", dict(month8, **{"other.py": b"never copied\n"}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"configuration": {"source_sha256": {k: sha(v) for k, v in month8.items()}}}))
    record = build_m8_overlay.build(tmp_path / "frozen", tmp_path / "snapshot", manifest, tmp_path / "overlay")
    assert sorted(record["replaced"]) == ["pipeline.py"]
    assert record["replaced"]["pipeline.py"]["normalized_diff"][-2:] == ["-# old comment", "+# new comment"]
    assert (tmp_path / "overlay/pipeline.py").read_bytes() == month8["pipeline.py"]
    assert (tmp_path / "overlay/revision_numerics.py").read_bytes() == b"x = 0\n"
    assert not (tmp_path / "overlay/other.py").exists()
    with pytest.raises(FileExistsError):
        build_m8_overlay.build(tmp_path / "frozen", tmp_path / "snapshot", manifest, tmp_path / "overlay")
    (tmp_path / "snapshot/rdt_sampling.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="not the recorded"):
        build_m8_overlay.build(tmp_path / "frozen", tmp_path / "snapshot", manifest, tmp_path / "overlay2")


def test_protocol_assembly_binds_bank_and_source_hashes(tmp_path):
    tree = tmp_path / "code"
    write_tree(tree, {name: name.encode() for name in SOURCE_FILES})
    (tmp_path / "bank.json").write_text("{}")
    spec = dict(metrics="m", bank=str(tmp_path / "bank.json"), task="PickCube-v1", model="170m", lang_dir="l",
                modalities=["vision"], fp32_contexts="none", adaptive_contexts="none",
                full_node_contexts="none", full_node_modalities=[], source_replay_settings={},
                code_tree=str(tree))
    protocol = make_revision_protocol.assemble(dict(schema=PROTOCOL_SCHEMA, numerics={}), {"P": spec})
    bank = protocol["banks"]["P"]
    assert bank["bank_sha256"] == sha(b"{}") and "code_tree" not in bank
    assert bank["source_sha256"] == {name: sha(name.encode()) for name in SOURCE_FILES}
    with pytest.raises(ValueError, match="unknown"):
        make_revision_protocol.assemble(dict(schema=PROTOCOL_SCHEMA), {"P": dict(spec, surprise=1)})
    with pytest.raises(ValueError, match="schema"):
        make_revision_protocol.assemble(dict(schema="other"), {"P": spec})


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
    write_tree(tmp_path / "identity", {"named.json": b'{"author": "Arjun Bajpai"}'})
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


NUMBERS_TOOL = Path(__file__).resolve().parents[1] / "docs/nhsjs-revision/tools/numbers.py"


@pytest.mark.skipif(not NUMBERS_TOOL.is_file(), reason="the numbers sheet tool is private-only")
def test_numbers_sheet_verification_goes_stale_when_a_source_changes():
    import importlib.util
    path = NUMBERS_TOOL
    spec = importlib.util.spec_from_file_location("numbers_sheet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = dict(id_prefix="audit/", status="re-derived", by="verifier", date="2026-10-23",
                  sources_sha256={"a.csv": "1" * 64})
    row = dict(id="audit/x/value", source="a.csv", source_sha256="1" * 64)
    assert module.verification_for(row, [record]) == "re-derived (verifier, 2026-10-23)"
    assert module.verification_for(dict(row, source_sha256="2" * 64), [record]).startswith("stale: a.csv changed")
    assert module.verification_for(dict(row, id="other/x"), [record]) == "not yet independently verified"
    assert module.verification_for(row, [dict(record, sources_sha256={})]).startswith("stale:")
