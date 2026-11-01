"""Scientific invariants of the explicitly retrospective reporting layer."""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.manuscript_revision.nhsjs_audit_2026_11_01 import report as r


def test_winner_outcomes_preserve_ties_and_pair_responses():
    q = np.array([[2, 1], [2, 1], [2, 2], [2, 1]], dtype=float)
    n = np.array([[3, 1], [1, 3], [3, 1], [2, 2]], dtype=float)
    assert r.paired_winner_counts(q, n) == {
        "same_unique_winner": 1, "different_unique_winners": 1, "at_least_one_tie": 2}
    matrix = np.array([[[1, 0], [0, 1], [2, 0]], [[1, 0], [0, 1], [2, 0]]], dtype=float)
    result = r.stability(matrix, matrix, ["a", "b"], draws=100)
    for item in result.values():
        assert item["bootstrap_counts"]["different_unique_winners"] == 0
        assert sum(item["bootstrap_counts"].values()) == 100


def test_clustered_median_retains_unequal_call_counts():
    groups = {"one": [0], "two": [10, 10, 10]}
    result = r.clustered_median(groups, draws=10000)
    assert result["median"] == 10
    assert result["ci_lo"] == 0 and result["ci_hi"] == 10
    assert result["episodes"] == 2 and result["calls"] == 4


def test_missing_or_duplicated_response_call_cannot_change_population_silently():
    rows = [{"direction": "deletion", "ranking": "a", "episode_id": "e", "context_id": "c",
             "response": response, "advantage": "1"} for response in ("Q", "L2")]
    np.testing.assert_array_equal(r.paired_matrix(rows, ["e"], ["a"]), np.ones((2, 1, 1)))
    with pytest.raises(ValueError, match="Unmatched"):
        r.paired_matrix(rows[:1], ["e"], ["a"])
    with pytest.raises(ValueError, match="Duplicate"):
        r.paired_matrix(rows + rows[:1], ["e"], ["a"])


def test_committed_reporting_reproduces_and_preserves_registered_decisions(tmp_path):
    expected = r.PACKAGE / "results/2026-11-01-v1/report.json"
    fresh = tmp_path / "fresh"
    result = r.build(fresh)
    assert (fresh / "report.json").read_bytes() == expected.read_bytes()
    assert {bank: len(item["episode_ids"]) for bank, item in result["methods"].items()} == {"P": 60, "B1": 40, "S": 30}
    for bank, record in result["methods"].items():
        for criterion in ("raw_mean", "scaled_mean", "episode_median"):
            item = record[criterion]
            assert sum(item["bootstrap_counts"].values()) == 10000
            q, n = (item["point"][response]["leaders"] for response in ("Q", "N"))
            assert (q != n) == (criterion == "raw_mean" or bank == "S")
    split = {(x["direction"], x["success"]): x for x in result["success_intervals"]}
    assert split["deletion", False]["episodes"] == 9
    assert split["deletion", False]["calls"] == 18
    assert split["deletion", False]["overshoot"] == 13
    assert split["deletion", True]["overshoot"] == 41
    assert split["deletion", False]["ci_lo"] <= 0 <= split["deletion", False]["ci_hi"]
    with pytest.raises(FileExistsError):
        r.build(fresh)
