import csv
import gzip
import json
import shutil
import zlib

import pytest

from analysis.paired_rescoring import extra_tasks as extra
from analysis.revision.core import IntegrityError, Record
from tests.provenance_assertions import assert_artifacts_authentic, assert_same_input_audit


def record(source, line, **row):
    base = dict(event="faithfulness", task="T", model="M", seed=42, episode=0, policy_call_idx=0)
    base.update(row)
    return Record(source, "0" * 64, line, f"{line:064d}", base)


def assert_reproduces(fresh, committed):
    """Every scientific artifact must match the committed bytes (gzip by payload,
    and by compressed bytes when this zlib build matches the recorded one)."""
    recorded = json.loads((committed / "provenance.json").read_bytes())
    produced = json.loads((fresh / "provenance.json").read_bytes())
    assert_artifacts_authentic(committed, recorded)
    assert_artifacts_authentic(fresh, produced)
    same_zlib = recorded["environment"]["zlib_runtime"] == zlib.ZLIB_RUNTIME_VERSION
    for name, entry in recorded["artifacts"].items():
        mine = produced["artifacts"][name]
        if name == "inputs.json":
            assert_same_input_audit(json.loads((fresh / name).read_bytes()),
                                    json.loads((committed / name).read_bytes()), extra.ROOT)
        elif "uncompressed_sha256" in entry:
            assert mine["uncompressed_sha256"] == entry["uncompressed_sha256"], name
            if same_zlib:
                assert mine["sha256"] == entry["sha256"], name
        else:
            assert (fresh / name).read_bytes() == (committed / name).read_bytes(), name
    assert_same_input_audit(produced["inputs"], recorded["inputs"], extra.ROOT)
    for key in ("protocol_sha256", "source_sha256", "analysis_version"):
        assert produced[key] == recorded[key], key


def test_cross_run_fingerprint_ignores_only_the_seed_label():
    faith = [record("f", 1, seed=42, vision_f_input=0.0, vision_f_baseline=-2.0, lang_f_input=0.0, lang_f_baseline=-1.0),
             record("f", 2, seed=43, episode=1, vision_f_input=0.0, vision_f_baseline=-3.0, lang_f_input=0.0, lang_f_baseline=-1.0)]
    steps = [record("s", 1, event="step", seed=142, vision_gap=2.0, lang_gap=1.0),
             record("s", 2, event="step", seed=143, episode=1, vision_gap=9.0, lang_gap=1.0)]
    result = extra.cross_run_fingerprint(faith, steps)
    assert result == dict(n_faithfulness=2, n_joined=2, n_agree_vision_and_lang=1)


def test_protocol_names_every_input_exactly_and_never_rescores_exclusions():
    _, protocol = extra.load_protocol()
    included = {run[kind]["path"] for run in protocol["runs"] for kind in ("faithfulness", "steps")}
    excluded = {item["faithfulness"]["path"] for item in protocol["excluded"]}
    assert len(included) == 8 and len(excluded) == 2 and not included & excluded
    assert all("*" not in name for name in included | excluded)
    assert all("_seed42_m128_2026" in name for name in excluded)
    assert protocol["outcome_groups"] == ["all"]
    assert all("lang" not in p["modalities"] or len(p["runs"]) == 1 for p in protocol["populations"])
    for task in ("PegInsertionSide", "PickSingleYCB"):
        pooled = next(p for p in protocol["populations"] if p["id"] == f"{task}_pooled")
        assert sum(r["expected_roster"]["n_calls"] for r in protocol["runs"] if r["run"] in pooled["runs"]) == 400


def test_real_data_reproduction_duplicates_and_exclusions(tmp_path):
    fresh = tmp_path / "fresh"
    extra.run(fresh)
    assert_reproduces(fresh, extra.RESULTS)
    assert extra.verify(fresh)["status"] == "verified"
    audit = json.loads((extra.RESULTS / "inputs.json").read_bytes())
    files = {f["path"]: f for f in audit["files"]}
    peg_faith = files["data/metrics_faithfulness_metrics_PegInsertionSide-v1_170m_seed142_m128_20260529_160739.jsonl"]
    peg_steps = files["data/metrics_PegInsertionSide-v1_170m_seed142_m128.jsonl"]
    assert (peg_faith["valid_records"], peg_faith["selected_records"], peg_faith["duplicate_groups"]) == (219, 200, 19)
    assert peg_faith["duplicate_episodes"] == [1] and peg_steps["duplicate_episodes"] == [1]
    assert peg_steps["duplicate_classes"] == {"conflicting_payload": 19}
    for item in audit["excluded"]:
        own = item["own_step_fingerprint"]
        assert own["n_joined"] == 200 and own["vision_general_agree"] == own["lang_general_agree"] == 0
        assert item["rescored"] is False
    with (extra.RESULTS / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = {(r["population"], r["case"]): r for r in csv.DictReader(stream)}
    for task in ("PegInsertionSide", "PickSingleYCB"):
        for case in ("vision:insertion", "vision:deletion"):
            assert (rows[(f"{task}_pooled", case)]["n_defined_calls"], rows[(f"{task}_pooled", case)]["n_defined_episodes"]) == ("400", "16")
        assert (f"{task}_pooled", "lang:insertion") not in rows
    calls = gzip.decompress((extra.RESULTS / "calls.csv.gz").read_bytes()).decode()
    assert "seed42_m128_2026" not in calls
    copy = tmp_path / "copy"
    shutil.copytree(extra.RESULTS, copy)
    (copy / "episodes.csv").write_bytes(b"tampered\n")
    with pytest.raises(IntegrityError, match="Artifact mismatch"):
        extra.verify(copy)
