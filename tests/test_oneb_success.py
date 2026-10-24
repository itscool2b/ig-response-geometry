import csv
import gzip
import io
import json
import zlib

import pytest

from analysis.paired_rescoring import oneb_success as oneb
from analysis.revision.core import IntegrityError, Record, sha256


def q(r):
    return -r / 1024


def faith(episode, call, residuals, baseline=4.0, seed=42, source="f"):
    row = dict(event="faithfulness", task="PickCube-v1", model="1b", seed=seed, episode=episode,
               policy_call_idx=call, k_grid_auc=[0, 50, 100],
               vision_insertion_curve=[q(baseline), q(residuals), 0.0],
               vision_deletion_curve=[0.0, q(residuals), q(baseline)],
               vision_f_input=0.0, vision_f_baseline=q(baseline))
    line = episode * 10 + call + 1
    return Record(source, "0" * 64, line, sha256(str(line).encode()), row)


def toy_protocol():
    return dict(runs=[dict(run="r1", role="primary"), dict(run="r2", role="sensitivity")],
                modalities=["vision"], directions=["insertion", "deletion"], native_scale="Q",
                dimension=512, stabilizer_epsilon=1e-12, bootstrap=dict(draws=64, seed=0),
                outcome_groups=["all", "success", "failure"],
                populations=[dict(id="p1", runs=["r1"], modalities=["vision"]),
                             dict(id="p2", runs=["r2"], modalities=["vision"])])


def test_success_split_and_short_success_episodes(monkeypatch):
    # r1: episode 0 succeeds after one in-range call; episode 1 fails with an overshoot.
    r1 = [faith(0, 0, 1.0), faith(1, 0, 9.0), faith(1, 1, 2.0), faith(1, 2, 3.0)]
    r2 = [faith(0, 0, 2.0, source="g"), faith(1, 0, 1.0, source="g")]
    selected = dict(r1=dict(faith=r1, labels={("PickCube-v1", "1b", 42, 0): True, ("PickCube-v1", "1b", 42, 1): False}),
                    r2=dict(faith=r2, labels={("PickCube-v1", "1b", 42, 0): False, ("PickCube-v1", "1b", 42, 1): False}))
    monkeypatch.setattr(oneb, "authenticate", lambda root, protocol: (selected, dict(files=[], runs=[])))
    result = oneb.compute(None, toy_protocol())
    rows = {(r["population"], r["case"], r["outcome_group"]): r for r in result["summaries"]}
    success = rows[("p1", "vision:insertion", "success")]
    failure = rows[("p1", "vision:insertion", "failure")]
    everything = rows[("p1", "vision:insertion", "all")]
    assert (success["n_raw_calls"], success["n_raw_episodes"]) == (1, 1)
    assert (failure["n_raw_calls"], failure["n_raw_episodes"]) == (3, 1)
    assert everything["n_raw_calls"] == 4 and everything["n_raw_episodes"] == 2
    assert success["overshoot_calls"] == 0 and failure["overshoot_calls"] == 1
    assert success["interval_status"] == "insufficient_groups"
    # Equal-episode weighting gives the single short success half the weight.
    assert everything["equal_episode_mean"] == pytest.approx(
        (success["equal_episode_mean"] + failure["equal_episode_mean"]) / 2)
    assert rows[("p2", "vision:deletion", "success")]["n_raw_calls"] == 0
    assert len(result["calls"]) == 12 and {c["run"] for c in result["calls"]} == {"r1", "r2"}
    index = {(r["population"], r["case"], r["outcome_group"]): r for r in result["indices"]}
    assert index[("p1", "vision:insertion", "failure")]["first_raw_calls"] == 1
    assert len(result["episodes"]) == 8


def test_output_writers_refuse_existing_directory(tmp_path):
    with pytest.raises(IntegrityError, match="existing"):
        oneb.run(tmp_path)


def test_protocol_declares_exact_inputs_and_rosters():
    _, protocol = oneb.load_protocol()
    names = [run[kind]["path"] for run in protocol["runs"] for kind in ("faithfulness", "steps")]
    assert len(names) == len(set(names)) == 8 and all("*" not in name for name in names)
    primary = [run for run in protocol["runs"] if run["role"] == "primary"]
    assert sum(run["expected_roster"]["successes"] for run in primary) == 15
    assert sum(run["expected_roster"]["n_episodes"] for run in primary) == 60
    assert sum(run["expected_roster"]["success_calls"] for run in primary) == 74
    assert sum(run["expected_roster"]["failure_calls"] for run in primary) == 1125
    assert protocol["bootstrap"]["draws"] == 10000 and protocol["bootstrap"]["seed"] == 0
    assert all("lang" not in p["modalities"] or len(p["runs"]) == 1 for p in protocol["populations"])


def assert_reproduces(fresh, committed):
    """Every scientific artifact must match the committed bytes.

    A gzip artifact is compared by its decompressed payload, and also by its
    compressed bytes when this zlib build matches the recorded one.
    """
    recorded = json.loads((committed / "provenance.json").read_bytes())
    produced = json.loads((fresh / "provenance.json").read_bytes())
    same_zlib = recorded["environment"]["zlib_runtime"] == zlib.ZLIB_RUNTIME_VERSION
    for name, record in recorded["artifacts"].items():
        mine = produced["artifacts"][name]
        if "uncompressed_sha256" in record:
            assert mine["uncompressed_sha256"] == record["uncompressed_sha256"], name
            assert gzip.decompress((fresh / name).read_bytes()) == gzip.decompress((committed / name).read_bytes())
            if same_zlib:
                assert mine["sha256"] == record["sha256"], name
        else:
            assert (fresh / name).read_bytes() == (committed / name).read_bytes(), name
    for key in ("protocol_sha256", "source_sha256", "inputs", "analysis_version"):
        assert produced[key] == recorded[key], key


def test_real_data_reproduction_and_success_roster(tmp_path):
    fresh = tmp_path / "fresh"
    oneb.run(fresh)
    assert_reproduces(fresh, oneb.RESULTS)
    assert oneb.verify(oneb.RESULTS) == dict(status="verified", artifacts=6, analysis_version="2026-10-23-v1")
    with (oneb.RESULTS / "summary.csv").open(encoding="utf-8", newline="") as stream:
        rows = {(r["population"], r["case"], r["outcome_group"]): r for r in csv.DictReader(stream)}
    assert len(rows) == 2 * 3 + 3 * 4 * 3 + 4 * 3
    for case in ("vision:insertion", "vision:deletion"):
        success = rows[("pooled_primary", case, "success")]
        failure = rows[("pooled_primary", case, "failure")]
        assert (success["n_defined_calls"], success["n_defined_episodes"]) == ("74", "15")
        assert (failure["n_defined_calls"], failure["n_defined_episodes"]) == ("1125", "45")
        assert rows[("pooled_primary", case, "all")]["n_undefined_calls"] == "0"
    assert ("pooled_primary", "lang:insertion", "all") not in rows
    assert rows[("seed142", "vision:insertion", "success")]["interval_status"] == "insufficient_groups"
    audit = json.loads((oneb.RESULTS / "inputs.json").read_bytes())
    assert all(r["fingerprint"]["all_agree_original"] and r["fingerprint"]["all_agree_general"] for r in audit["runs"])
    calls = list(csv.DictReader(io.StringIO(gzip.decompress((oneb.RESULTS / "calls.csv.gz").read_bytes()).decode())))
    assert len(calls) == (1199 + 334) * 4
    with pytest.raises(IntegrityError, match="existing"):
        oneb.run(fresh)


def test_verify_rejects_tampered_results(tmp_path):
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(oneb.RESULTS, copy)
    (copy / "summary.csv").write_bytes((copy / "summary.csv").read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="Artifact mismatch"):
        oneb.verify(copy)
