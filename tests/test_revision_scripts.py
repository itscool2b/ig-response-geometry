import hashlib
import json
from pathlib import Path

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


NUMBERS_TOOL = Path(__file__).resolve().parents[1] / "docs/nhsjs-revision/tools/numbers.py"


@pytest.mark.skipif(not NUMBERS_TOOL.is_file(), reason="the numbers sheet tool is private-only")
def test_numbers_sheet_verification_goes_stale_when_a_source_changes():
    import importlib.util
    path = NUMBERS_TOOL
    spec = importlib.util.spec_from_file_location("numbers_sheet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = dict(id_prefix="audit/", status="re-derived", by="verifier", date="2026-10-23",
                  sources_sha256={"a.csv": "1" * 64}, sheet_commit="c1")
    sheets = {"c1": {"audit/x/value": "5"}}
    row = dict(id="audit/x/value", source="a.csv", source_sha256="1" * 64, value="5")

    def check(item, records=(record,)):
        return module.verification_for(item, list(records), sheet_of=sheets.__getitem__)

    assert check(row) == "re-derived (verifier, 2026-10-23)"
    assert check(dict(row, source_sha256="2" * 64)).startswith("stale: a.csv changed")
    assert check(dict(row, value="6")).startswith("stale: the value changed")
    # A row added later under the checked prefix, a source the check never read, an unbound check:
    assert check(dict(row, id="audit/y/value")) == "not yet independently verified"
    assert check(dict(row, source="b.csv")) == "not yet independently verified"
    assert check(row, [{k: v for k, v in record.items() if k != "sheet_commit"}]) == "not yet independently verified"
    assert check(dict(row, id="other/x")) == "not yet independently verified"


@pytest.mark.skipif(not NUMBERS_TOOL.is_file(), reason="the numbers sheet tool is private-only")
def test_numbers_sheet_keeps_the_unprefixed_bf16_comparator_rules_but_not_the_comparator_itself():
    import importlib.util
    spec = importlib.util.spec_from_file_location("numbers_sheet", NUMBERS_TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keep = module.r1_comparison
    family = "bf16_vs_A1_comparator"
    assert all(keep(dict(family=family, candidate=name)) for name in ("T16", "E64", "M128", "E512", "A0m64"))
    assert not any(keep(dict(family=family, candidate=name)) for name in ("T512", "RT16", "E8", "S16"))
    assert keep(dict(family="fp32_vs_A2T512", candidate="A2T256")) and not keep(dict(family="fp32_vs_A2T512", candidate="T256"))
