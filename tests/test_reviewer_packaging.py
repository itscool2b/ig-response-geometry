"""The repository packaging tools: the regeneration-code builder and the reviewer archive.

Like tests/test_research_supplement.py, these test tools that build packages from the
repository, so the research supplement omits them.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

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
    assert "The 1 tensor files (1 map files; 0.0 GB)" in (out / "README.md").read_text()
    assert "  README.md\n" in (out / "MANIFEST.sha256").read_text()
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
             "token.txt": b"contact: secret-token-7", "utf16.txt": "Windermere".encode("utf-16-le"),
             "bajpai-notes.json": b"{}"}
    for k, (name, data) in enumerate(cases.items()):
        write_tree(tmp_path / f"identity{k}", {name: data})
        with pytest.raises(ValueError, match="Identifying text"):
            archive.build(tmp_path / f"bad{k}", **dict(kwargs, identity=tmp_path / f"identity{k}"),
                          identifiers=["SECRET-token-7"])
        assert not (tmp_path / f"bad{k}").exists()
    write_tree(tmp_path / "identity-pod", {"pod.json": b'{"cwd": "/workspace/nhsjs", "home": "/root/.cache"}'})
    archive.build(tmp_path / "pod-ok", **dict(kwargs, identity=tmp_path / "identity-pod"))


def reviewer_inputs(tmp_path):
    from scripts import build_reviewer_archive as archive
    for name, files in (("results", {"manifest.json": b"{}"}), ("identity", {"pod.json": b"{}"}),
                        ("protocols", {"plan.json": b"{}"}), ("month8", {"README.md": b"bank"}),
                        ("mf2", {n: b"facts" for n in archive.INVENTORY_FILES})):
        write_tree(tmp_path / name, files)
    phase = tmp_path / "runs/P/e0000-c001/bf16"
    write_tree(phase, {"record.json": b"{}", "maps.pt": b"raw-map", "source-cache.pt": b"raw-cache"})
    (phase / "completion.json").write_text(json.dumps({"files_sha256": {
        "record.json": sha(b"{}"), "maps.pt": sha(b"raw-map"), "source-cache.pt": sha(b"raw-cache")}}))
    return dict(results=tmp_path / "results", runs=tmp_path / "runs", identity=tmp_path / "identity",
                sources=dict(protocols=tmp_path / "protocols", month8=tmp_path / "month8",
                             inventories=tmp_path / "mf2"))


def test_complete_reviewer_archive_keeps_and_authenticates_every_raw_run_file(tmp_path):
    from scripts import build_reviewer_archive as archive
    kwargs = reviewer_inputs(tmp_path)
    write_tree(tmp_path / "runs/P/e0000-c001/attempts/failed", {
        "failure.json": b'{"error":"retained"}', "partial.pt": b"partial bytes"})
    (tmp_path / "runs/P/e0000-c002/attempts/empty").mkdir(parents=True)
    write_tree(tmp_path / "runs/P/workers", {"worker-00.jsonl": b'{"event":"end"}\n'})
    report = archive.build(tmp_path / "full", **kwargs, include_raw_runs=True, anonymous=True)
    assert report["raw_runs_included"] is True and report["tensors_included"] == 2
    assert report["attempt_failures"] == 1
    for source in (tmp_path / "runs").rglob("*"):
        destination = tmp_path / "full/runs" / source.relative_to(tmp_path / "runs")
        assert destination.is_dir() if source.is_dir() else destination.read_bytes() == source.read_bytes()
    out = tmp_path / "full"
    manifest = (out / "MANIFEST.sha256").read_bytes()
    assert b"\r" not in manifest and b"\\" not in manifest
    for line in manifest.decode().splitlines():
        digest, name = line.split("  ", 1)
        assert sha((out / name).read_bytes()) == digest
    assert "All 2 completion-bound raw files" in (out / "README.md").read_text()
    assert "are not included" not in (out / "README.md").read_text()


@pytest.mark.parametrize("include_raw", [False, True])
@pytest.mark.parametrize("fault", ["missing", "corrupted", "escape", "absolute", "backslash", "bad-hash"])
def test_reviewer_archive_rejects_invalid_completion_bindings(tmp_path, include_raw, fault):
    from scripts import build_reviewer_archive as archive
    kwargs = reviewer_inputs(tmp_path)
    phase = tmp_path / "runs/P/e0000-c001/bf16"
    completion = phase / "completion.json"
    value = json.loads(completion.read_text())
    if fault == "missing":
        (phase / "maps.pt").unlink()
    elif fault == "corrupted":
        (phase / "maps.pt").write_bytes(b"changed")
    else:
        name = {"escape": "../outside.pt", "absolute": "/outside.pt",
                "backslash": "nested\\outside.pt", "bad-hash": "maps.pt"}[fault]
        value["files_sha256"][name] = "invalid" if fault == "bad-hash" else sha(b"outside")
        completion.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Completion hash|Invalid completion"):
        archive.build(tmp_path / "bad", **kwargs, include_raw_runs=include_raw)
    assert not (tmp_path / "bad").exists()


def test_complete_anonymous_archive_checks_attempt_data_without_rewriting_it(tmp_path):
    from scripts import build_reviewer_archive as archive
    kwargs = reviewer_inputs(tmp_path)
    private = b"Windermere in an interrupted file"
    write_tree(tmp_path / "runs/P/e0000-c001/attempts/interrupted", {"partial.pt": private})
    with pytest.raises(ValueError, match="Identifying text"):
        archive.build(tmp_path / "bad", **kwargs, include_raw_runs=True, anonymous=True)
    assert not (tmp_path / "bad").exists()
    assert (tmp_path / "runs/P/e0000-c001/attempts/interrupted/partial.pt").read_bytes() == private


@pytest.mark.parametrize("name", ["../outside", "nested/path", "C:drive", "", ".."])
def test_reviewer_archive_rejects_escaping_collection_names(tmp_path, name):
    from scripts import build_reviewer_archive as archive
    kwargs = reviewer_inputs(tmp_path)
    with pytest.raises(ValueError, match="Collection names"):
        archive.build(tmp_path / "bad", **kwargs, collections=[(name, tmp_path / "runs")])
    assert not (tmp_path / "bad").exists()


def test_reviewer_archive_rejects_output_inside_source_without_modifying_it(tmp_path):
    from scripts import build_reviewer_archive as archive
    kwargs = reviewer_inputs(tmp_path)
    with pytest.raises(ValueError, match="outside every source"):
        archive.build(tmp_path / "runs/output", **kwargs)
    assert not (tmp_path / "runs/output").exists()


def test_regeneration_kit_members_are_not_silently_filtered(tmp_path):
    from scripts import build_reviewer_archive as archive
    kwargs = reviewer_inputs(tmp_path)
    kit = tmp_path / 'kit'
    files = {'record.json': b'{}', 'source.lock': b'bound auxiliary', '__pycache__/saved.bin': b'bound cache'}
    write_tree(kit, files)
    archive.build(tmp_path / 'out', **kwargs, kit=kit)
    assert {p.relative_to(tmp_path / 'out/kit').as_posix(): p.read_bytes()
            for p in (tmp_path / 'out/kit').rglob('*') if p.is_file()} == files


def test_stored_tensor_privacy_scan_keeps_raw_and_utf16_matches_across_chunks(tmp_path):
    from scripts import build_reviewer_archive as archive
    path = tmp_path / "tensor.pt"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as output:
        output.writestr("tensor/data.pkl", b"metadata")
        output.writestr("tensor/data/0", b"x" * 63 + b"Bajpai" + b"x" * 63
                        + "Windermere".encode("utf-16-le") + b"x" * 63 + b"/Users/someone/project")
    assert archive.stored_zip_hits(path, ["bajpai", "windermere"], chunk_size=4) == [
        "bajpai", "personal machine path", "windermere"]
    compressed = tmp_path / "compressed.zip"
    with zipfile.ZipFile(compressed, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr("secret.txt", b"Bajpai")
    assert archive.stored_zip_hits(compressed, ["bajpai"], chunk_size=4) is None
    assert archive.stored_zip_hits(path, ['非公開'], chunk_size=4) == ['personal machine path']


def test_public_tokenizer_build_path_exception_requires_exact_upstream_bytes(tmp_path):
    from scripts import build_reviewer_archive as archive
    source = Path(__file__).resolve().parents[1] / 'analysis/language_groups/vendor/spiece.model'
    data = source.read_bytes()
    target = tmp_path / 'tokenizer.model'
    target.write_bytes(data)
    assert not archive.identifying_hits(tmp_path)
    target.write_bytes(data + b'changed')
    assert archive.identifying_hits(tmp_path) == [('tokenizer.model', ['personal machine path'])]


def test_binary_path_prefilter_preserves_mixed_separators_and_unicode_filename_checks(tmp_path):
    from scripts import build_reviewer_archive as archive
    values = [b'C:/Users/person/file', b'c:\\Users\\person\\file', b'C:/uSeRs\\person/file',
              b'C:\\uSErs/person/file', b'/Users/person/file', b'/HoMe/person/file']
    for index, value in enumerate(values):
        (tmp_path / f'{index}.bin').write_bytes(value)
    unicode_path = tmp_path / 'a/Uſers/person/file.txt'
    unicode_path.parent.mkdir(parents=True)
    unicode_path.write_text('ordinary bytes')
    hits = archive.identifying_hits(tmp_path)
    assert len(hits) == len(values) + 1
    assert all(tokens == ['personal machine path'] for _, tokens in hits)
    # Every possible latin-1 replacement is still evaluated by the same regex
    # whenever its literal Users/home component can match.
    for byte in range(256):
        text = ('C:/' + chr(byte) + 'sers/person/file').lower()
        assert bool(archive.PERSONAL_PATH.search(text)) == bool(
            ('users' in text or '/home/' in text) and archive.PERSONAL_PATH.search(text))


def test_anonymous_reviewer_archive_requires_pdf_drawn_text_scan(tmp_path, monkeypatch):
    from scripts import build_reviewer_archive as archive
    kwargs = reviewer_inputs(tmp_path)
    write_tree(tmp_path / "results", {"figure.pdf": b"%PDF-1.4\n"})
    monkeypatch.setattr(archive, "pdf_reader_available", lambda: False)
    with pytest.raises(ValueError, match="requires PyMuPDF"):
        archive.build(tmp_path / "bad", **kwargs, anonymous=True)
    assert not (tmp_path / "bad").exists()


def portable_fixture(tmp_path):
    from scripts import build_reviewer_archive as archive
    root = tmp_path / 'archive'
    protocol = 'docs/nhsjs-revision/protocols/r1/production/test.json'
    bank = 'docs/nhsjs-revision/protocols/r1/banks/test.json'
    expected = dict(kind='nhsjs_r1_gpu_analysis', inputs={'run_root': '../unavailable-volume',
        'banks': [dict(id='P', group='P', bank=bank, protocol=protocol)]}, population={'P': {'available': 1}},
        outputs={'result.csv': sha(b'x\n1\n')}, plan_sha256=sha(b'{}'),
        coverage_sha256={'original/coverage.json': sha(b'{}')})
    write_tree(root, {'results/analysis/manifest.json': json.dumps(expected).encode(),
        'results/analysis/result.csv': b'x\n1\n', 'protocols/analysis-plan-r1.json': b'{}',
        'protocols/production/test.json': b'{}', 'protocols/banks/test.json': b'{}',
        **{'collections/' + name + '/metrics.jsonl': b'{}' for name in ('P', 'B1', 'S')}})
    (root / 'runs').mkdir()
    code = b'''import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('inputs','plan','out'):p.add_argument('--'+name,required=True)
p.add_argument('--coverage',action='append');p.add_argument('--metrics-copy',action='append');a=p.parse_args()
inputs=json.loads(Path(a.inputs).read_text());assert Path(inputs['run_root']).is_dir()
for row in inputs['banks']:
 for key in ('bank','protocol'):assert Path(row[key]).is_file() and Path(row[key]).is_absolute()
for value in a.metrics_copy:assert Path(value.split('=',1)[1]).is_file()
sha=lambda b:hashlib.sha256(b).hexdigest()
out=Path(a.out);out.mkdir();(out/'result.csv').write_bytes(b'x\\n1\\n')
data=dict(kind='nhsjs_r1_gpu_analysis',inputs=inputs,population={'P':{'available':1}},outputs={'result.csv':sha(b'x\\n1\\n')},plan_sha256=sha(Path(a.plan).read_bytes()),coverage_sha256={x:sha(Path(x).read_bytes()) for x in a.coverage})
(out/'manifest.json').write_text(json.dumps(data))
'''
    contents = {'analysis/revision_gpu/v1/analyze.py': code}
    package = tmp_path / 'code.zip'
    with zipfile.ZipFile(package, 'w') as stream:
        for name, data in contents.items():
            stream.writestr(name, data)
        stream.writestr('SUPPLEMENT_MANIFEST.json', json.dumps({'members': {
            name: {'sha256': sha(data), 'bytes': len(data)} for name, data in contents.items()}}))
    coverage = tmp_path / 'coverage.json'
    coverage.write_bytes(b'{}')
    archive.add_reproduction(root, package, [coverage])
    files = sorted(p for p in root.rglob('*') if p.is_file())
    (root / 'MANIFEST.sha256').write_text(''.join(archive.sha256(p) + '  ' + p.relative_to(root).as_posix() + '\n' for p in files))
    return root


def test_portable_archive_replay_executes_without_checkout_or_original_paths(tmp_path):
    from scripts import reproduce_reviewer_archive as replay
    root = portable_fixture(tmp_path)
    report = replay.replay(root, tmp_path / 'reproduction')
    assert report['verified'] and report['outputs_compared'] == 1
    text = (tmp_path / 'reproduction/inputs.json').read_text()
    assert 'unavailable-volume' not in text
    assert str(root).replace('\\', '\\\\') in text
    (root / 'collections/P/metrics.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Archive hash mismatch'):
        replay.replay(root, tmp_path / 'second')
    assert not (tmp_path / 'second').exists()


@pytest.mark.parametrize('bad_name', ['../escape.py', 'a/../escape.py', 'C:/escape.py', 'a\\escape.py'])
def test_nested_code_manifest_cannot_escape_extraction(tmp_path, bad_name):
    from scripts.reproduce_reviewer_archive import extract_code
    package = tmp_path / 'code.zip'
    with zipfile.ZipFile(package, 'w') as stream:
        info = zipfile.ZipInfo('placeholder')
        info.filename = bad_name  # Bypass Windows ZipInfo's separator normalization in this adversarial fixture.
        stream.writestr(info, b'code')
        stream.writestr('SUPPLEMENT_MANIFEST.json', '{}')
    with pytest.raises(ValueError, match='Unsafe archive member'):
        extract_code(package, tmp_path / 'code')
    assert not (tmp_path / 'code').exists()


def test_replay_rejects_unlisted_files_and_duplicate_manifest_records(tmp_path):
    from scripts.reproduce_reviewer_archive import verify_archive
    root = portable_fixture(tmp_path)
    manifest = root / 'MANIFEST.sha256'
    original = manifest.read_text()
    manifest.write_text(original + original.splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='Repeated'):
        verify_archive(root)
    manifest.write_text(original)
    (root / 'unlisted.py').write_text('pass')
    with pytest.raises(ValueError, match='inventory differs'):
        verify_archive(root)
