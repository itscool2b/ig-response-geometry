import csv
import json

import pytest

from analysis.paired_rescoring import influence as a
from analysis.revision.core import IntegrityError, Record, json_bytes, sha256


def call(record, episode, delta):
    return dict(record_id=record, episode_id=episode, delta=delta, paired_status="defined")


def test_omission_removes_whole_episode_and_keeps_equal_episode_weights():
    calls = [call("a1", "A", -10), call("a2", "A", -10), call("a3", "A", 20),
             call("b", "B", 4), call("c", "C", -8)]
    summary, omissions = a.summarize(calls)
    results = {r["omitted_episode_id"]: r for r in omissions}
    assert summary["equal_episode_mean"] == pytest.approx(-4 / 3)
    assert results["A"]["n_omitted_calls"] == 3
    assert results["A"]["omitted_record_ids"] == ["a1", "a2", "a3"]
    assert results["A"]["remaining_equal_episode_mean"] == -2
    assert results["B"]["remaining_equal_episode_mean"] == -4
    # The retained call-weighted mean after omitting B would instead be -2.
    assert results["B"]["n_retained_calls"] == 4
    assert results["C"]["remaining_equal_episode_mean"] == 2
    assert summary["n_sign_preserved"] == 2
    assert results["A"]["retained_population_sha256"] == sha256(json_bytes(["b", "c"]))


def test_episode_concentration_uses_negative_mean_mass_not_net_or_call_total():
    calls = [call("a1", "A", -2), call("a2", "A", -2), call("b", "B", -6), call("c", "C", 20)]
    summary, _ = a.summarize(calls)
    assert summary["negative_episode_mean_mass"] == 8
    assert summary["largest_negative_episode_mass_share"] == .75
    assert summary["largest_negative_episode_id"] == "B"
    assert summary["equal_episode_mean"] == 4


def test_zero_is_a_separate_sign_and_missing_negative_mass_is_null():
    summary, omissions = a.summarize([call("a", "A", 0), call("b", "B", 2)])
    assert summary["n_sign_preserved"] == 1
    assert summary["largest_negative_episode_mass_share"] is None
    assert next(r for r in omissions if r["omitted_episode_id"] == "B")["remaining_mean_sign"] == 0


@pytest.mark.parametrize("calls", [[], [call("a", "A", 1)],
    [call("a", "A", 1), call("a", "B", 2)],
    [call("a", "A", float("nan")), call("b", "B", 2)],
    [dict(call("a", "A", 1), paired_status="undefined_zero_gap"), call("b", "B", 2)]])
def test_invalid_populations_fail_instead_of_silently_redefining_estimand(calls):
    with pytest.raises(IntegrityError):
        a.summarize(calls)


def test_physical_identity_includes_source_and_episode_not_just_legacy_labels():
    payload = dict(task="T", model="M", seed=42, episode=0, policy_call_idx=1)
    record = Record("source-a", "a" * 64, 2, "b" * 64, payload)
    key = list(a.paired.episode_key(record))
    row = dict(record_id=record.id, source=record.source, source_sha256=record.source_sha256,
               line=record.line, line_sha256=record.line_sha256, policy_call_idx=1,
               episode_key=key, episode_id=sha256(json_bytes(key)))
    a.validate_identity(row, record)
    for field, changed in (("source", "source-b"), ("line", 3), ("policy_call_idx", 2),
                           ("episode_id", "c" * 64), ("episode_key", ["source-b", *key[1:]])):
        with pytest.raises(IntegrityError, match="identity"):
            a.validate_identity(dict(row, **{field: changed}), record)


def test_tail_concentration_and_example_stay_within_case():
    base = dict(source="s", source_sha256="x", line=1, line_sha256="x", episode_key=["s"],
                policy_call_idx=0, auc_q=-5, auc_l2=-1, gap_q=.01, gap_l2=.1,
                native_actual=0, native_baseline=-.01, native_curve=[-.01, -1, 0],
                grid_percent=[0, 50, 100], residual_actual=0, residual_baseline=1,
                residual_points=[1, 100, 0], orientation="baseline_farther")
    calls = [dict(base, **call("b", "B", -2)), dict(base, **call("a", "A", -6)),
             dict(base, **call("c", "C", 20))]
    result = a.tail_diagnostic(calls, [1, 3], 1e-9)
    assert result["negative_call_mass"] == 8
    assert result["top_negative_call_mass"][0]["share"] == .75
    assert result["top_negative_call_mass"][1]["included_count"] == 2
    assert result["top_negative_call_mass"][1]["share"] == 1
    assert result["example"]["record_id"] == "a"
    assert result["maximum_to_baseline_squared_residual_ratio"] == 100
    assert result["baseline_weak_empirical_percentile"] == 100


def test_changed_paired_anchor_is_rejected_before_reading_calls(tmp_path):
    protocol = json.loads((a.ROOT / a.PROTOCOL_PATH).read_bytes())
    target = tmp_path / protocol["paired_results"]
    target.mkdir(parents=True)
    (target / "provenance.json").write_text("{}")
    with pytest.raises(IntegrityError, match="anchor"):
        a.authenticate(tmp_path, protocol)


def test_real_saved_data_reproduction_and_immutable_inputs(tmp_path):
    original = a.ROOT / "analysis/paired_rescoring/results/2026-10-01-v1/provenance.json"
    original_bytes = original.read_bytes()
    output = tmp_path / "fresh"
    provenance = a.run(output)
    assert a.verify(output) == dict(status="verified", artifacts=6, n_cases=8, n_episode_omissions=240)
    assert original.read_bytes() == original_bytes
    assert sorted(p.name for p in a.paired.HERE.glob("*.py")) == ["__init__.py", "analyze.py"]
    with (output / "case_summary.csv").open(encoding="utf-8", newline="") as stream:
        cases = {r["case"]: r for r in csv.DictReader(stream)}
    with (output / "episode_omissions.csv").open(encoding="utf-8", newline="") as stream:
        omissions = list(csv.DictReader(stream))
    assert len(omissions) == 240
    assert {int(r["n_omitted_calls"]) for r in omissions} == {25}
    assert {int(r["n_retained_calls"]) for r in omissions} == {725}
    assert cases["Q:lang:deletion"]["n_sign_preserved"] == "27"
    assert cases["L2:lang:deletion"]["n_sign_preserved"] == "29"
    for key, row in cases.items():
        if key.endswith("insertion"):
            assert float(row["omission_mean_max"]) < 0
    released = a.ROOT / "analysis/paired_rescoring/influence_results/2026-10-01-v1/provenance.json"
    if released.exists():
        assert provenance["artifacts_sha256"] == json.loads(released.read_bytes())["artifacts_sha256"]
    with pytest.raises(IntegrityError, match="existing"):
        a.run(output)
    (output / "episode_omissions.csv").write_bytes(b"tampered\n")
    with pytest.raises(IntegrityError, match="artifact mismatch"):
        a.verify(output)
