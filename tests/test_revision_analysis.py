"""Integrity and independent numerical checks for saved-data reanalysis."""
import json

import numpy as np
import pytest

from analysis.revision.analyze import Registry, select
from analysis.revision.core import (IntegrityError, Record, bootstrap_summary,
                                    json_bytes, load_file, normalized_auc,
                                    population_id, random_order_counterexample,
                                    reconcile, sha256, strict_json, transform_score)


def row(**overrides):
    result = dict(event="step", task="fixture", model="fixture", seed=42,
                  episode=0, policy_call_idx=0, value=1.0, wall_seconds=2.0)
    result.update(overrides)
    return result


def record(line=1, source="data/fixture.jsonl", **overrides):
    payload = row(**overrides)
    return Record(source, "source-hash", line, sha256(json_bytes(payload)), payload)


@pytest.mark.parametrize("raw", [b'{"event": "step", "event":"step"}',
                                  b"null", b"[]", b"{broken", b"", b"\0"])
def test_strict_json_rejects_malformed_or_ambiguous(raw):
    with pytest.raises(IntegrityError):
        strict_json(raw)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None])
def test_nonfinite_and_null_rejected_recursively(bad):
    payload = row(nested={"array": [1, bad]})
    with pytest.raises(IntegrityError):
        strict_json(json.dumps(payload).encode())


@pytest.mark.parametrize("overrides", [{"seed": True}, {"episode": -1},
                                       {"policy_call_idx": 1.2}, {"task": ""}])
def test_identity_types_cannot_silently_coalesce(overrides):
    with pytest.raises(IntegrityError):
        strict_json(json_bytes(row(**overrides)))


def test_exact_quarantine_and_checkout_variant(tmp_path):
    path = tmp_path / "fixture.jsonl"
    line = json_bytes(row()) + b"\n"
    bad = b"{\x00}\n"
    original = line + bad
    checkout = original.replace(b"\n", b"\r\n")
    path.write_bytes(checkout)
    expected = dict(sha256=sha256(original), physical_lines=2, valid_records=1,
                    git_crlf_checkout_sha256=sha256(checkout),
                    quarantine=[dict(line=2, line_sha256=sha256(bad), reason="fixture")])
    records, ledger = load_file(path, tmp_path, expected)
    assert records[0].line_sha256 == sha256(line)
    assert ledger[1]["status"] == "quarantined"
    assert path.read_bytes() == checkout  # The physical source was not rewritten.
    path.write_bytes(checkout.replace(b"fixture", b"changed"))
    with pytest.raises(IntegrityError, match="hash mismatch"):
        load_file(path, tmp_path, expected)


def test_unexpected_malformed_line_cannot_be_silently_skipped(tmp_path):
    path = tmp_path / "fixture.jsonl"
    raw = json_bytes(row()) + b"\n{bad}\n"
    path.write_bytes(raw)
    with pytest.raises(IntegrityError, match="Malformed JSON"):
        load_file(path, tmp_path, dict(sha256=sha256(raw), physical_lines=2, valid_records=1))


def test_duplicate_conflicts_preserve_all_occurrences_and_source_namespaces():
    records = [record(1), record(2, wall_seconds=9.0),
               record(3, policy_call_idx=1, value=4.0),
               record(4, policy_call_idx=1, value=7.0),
               record(1, source="data/different-run.jsonl")]
    last, clean, duplicates = reconcile(records)
    assert [r.line for r in last] == [2, 4, 1]
    assert [r.line for r in clean] == [2, 1]
    assert duplicates[0]["classification"] == "metadata_only"
    assert duplicates[1]["classification"] == "conflicting_payload"
    assert duplicates[1]["differing_fields"] == ["value"]
    assert duplicates[1]["conflict_excluded_selected"] is None
    assert len(duplicates[1]["occurrences"]) == 2
    assert population_id(last) != population_id(clean)
    with pytest.raises(IntegrityError, match="repeats"):
        population_id([last[0], last[0]])


def test_bootstrap_matches_independent_literal_cluster_resampling():
    values = np.array([1.0, 2.0, 90.0, 9.0, 10.0, 11.0])
    keys = [(0,), (0,), (1,), (2,), (2,), (2,)]
    groups = [values[[0, 1]], values[[2]], values[[3, 4, 5]]]
    for statistic in ("median", "mean"):
        rng = np.random.default_rng(81)
        statistic_fn = np.median if statistic == "median" else np.mean
        literal = [statistic_fn(np.concatenate([groups[i] for i in rng.integers(0, 3, 3)]))
                   for _ in range(1001)]
        observed = bootstrap_summary(values, keys, statistic=statistic, draws=1001, seed=81)
        assert observed["point"] == statistic_fn(values)
        assert [observed["ci_lo"], observed["ci_hi"]] == pytest.approx(np.percentile(literal, [2.5, 97.5]))
        assert observed["n_groups"] == 3


def test_single_and_two_cluster_precision_is_not_certified():
    one = bootstrap_summary([1, 7], [(0,), (0,)])
    assert one["ci_lo"] is None and one["ci_hi"] is None
    assert one["interval_status"] == "insufficient_groups"
    two = bootstrap_summary([1, 7], [(0,), (1,)], draws=20)
    assert two["interval_status"] == "two_group_descriptive_only"


def test_registry_population_is_shared_by_point_and_interval():
    records = [record(1, value=2), record(2, episode=1, value=9),
               record(3, episode=1, value=30)]
    registry = Registry(draws=100, seed=0)
    result = registry.add("fixture", records, "value", "fixture_units")
    assert result["point"] == 9
    assert result["point_population_sha256"] == result["ci_population_sha256"]
    assert result["n_rows"] == result["n_source_records"] == 3
    assert len(registry.populations[result["point_population_sha256"]]) == 3
    with pytest.raises(IntegrityError, match="Value/population mismatch"):
        registry.add("bad", records, "value", "units", values=[2])
    with pytest.raises(IntegrityError, match="Missing field"):
        registry.add("missing", records, "missing", "units")


def test_episode_equal_estimand_does_not_relabel_call_median():
    records = [record(1, episode=0, value=0), record(2, episode=0, value=0),
               record(3, episode=0, value=0), record(4, episode=1, value=10)]
    registry = Registry(draws=100, seed=0)
    call = registry.add("call", records, "value", "units")
    equal = registry.add("equal", records, "value", "units", episode_equal=True)
    assert call["point"] == 0
    assert equal["point"] == 5
    assert equal["n_rows"] == 2 and equal["n_source_records"] == 4
    assert call["point_population_sha256"] == equal["point_population_sha256"]


def test_auc_denominators_grid_and_unbounded_range():
    assert normalized_auc([0, 100], [0, 0], 0, 0) is None
    assert normalized_auc([0, 100], [0, 1e-12], 1e-12, 0) == pytest.approx(.5)
    assert normalized_auc([0, 100], [0, 1e-12], 1e-12, 0, denominator_min=1e-9) is None
    assert normalized_auc([0, 50, 100], [0, -100, 1], 1, 0) == -49.75
    for grid, curve in [([0, 10, 10, 100], [0, 0, 0, 1]), ([0, 50], [0, 1]), ([0, 100], [0])]:
        with pytest.raises(IntegrityError):
            normalized_auc(grid, curve, 1, 0)


def test_q_l2_transform_has_correct_distance_units_and_domain():
    q = -9 / 1024
    distance = transform_score(q, "Q", "L2")
    assert distance == pytest.approx(-np.sqrt(9 + 1e-12))
    assert transform_score(distance, "L2", "Q") == pytest.approx(q)
    with pytest.raises(IntegrityError):
        transform_score(0.2, "Q", "L2")
    with pytest.raises(IntegrityError):
        transform_score(0, "L2", "Q")


def test_equal_feature_random_order_counterexample():
    result = random_order_counterexample([0, 25, 50, 75, 100])
    # Explicitly evaluate every permutation of a four-feature linear policy.
    from itertools import permutations
    insertion, deletion = [], []
    for order in permutations(range(4)):
        ins, delete = [], []
        for k in range(5):
            kept = np.zeros(4)
            kept[list(order[:k])] = 1
            ins.append(-float((kept.sum() - 4) ** 2))
            delete.append(-float(kept.sum() ** 2))
        insertion.append(normalized_auc([0, 25, 50, 75, 100], ins, 0, -16))
        deletion.append(normalized_auc([0, 25, 50, 75, 100], delete, 0, -16))
    assert result["quadratic_grid_insertion"] == pytest.approx(np.mean(insertion))
    assert result["quadratic_grid_deletion"] == pytest.approx(np.mean(deletion))
    assert result["quadratic_continuous_integral"] == 2 / 3
    assert np.mean(insertion) != .5


def test_norm_filter_fails_if_requested_quantity_is_absent():
    with pytest.raises(IntegrityError, match="missing norms"):
        select([record()], "fixture.jsonl", norm_filter="norm_ge15")
