import csv
import gzip
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from analysis.paired_rescoring import analyze as a
from analysis.revision.core import IntegrityError, Record, json_bytes, sha256


def qscore(r):
    return -r / 1024


def curve(residuals, actual=0, baseline=4, ranking="Q"):
    encode = qscore if ranking == "Q" else lambda x: -np.sqrt(x + 1e-12)
    return a.paired_curve([0, 50, 100], [encode(x) for x in residuals],
                          encode(actual), encode(baseline), ranking)


def call(identity, episode, delta, status="defined"):
    return dict(record_id=identity, episode_id=episode, delta=delta, paired_status=status,
                contribution_in_range=delta, contribution_high_distance_overshoot=0,
                contribution_low_distance_undershoot=0)


@pytest.mark.parametrize("ranking", ["Q", "L2"])
def test_nonzero_actual_generalized_identity_and_signed_decomposition(ranking):
    result = curve([4, 2.25, 1], actual=1, baseline=4, ranking=ranking)
    assert result["orientation"] == "baseline_farther"
    assert result["residual_actual"] == pytest.approx(1)
    assert result["delta"] == pytest.approx(1 / 24)
    assert result["generalized_identity_max_error"] < 1e-12
    assert sum(result[f"contribution_{c}"] for c in a.CATEGORIES) == pytest.approx(result["delta"])


def test_baseline_nearer_reverses_in_range_sign_without_exclusion():
    result = curve([1, 2.25, 4], actual=4, baseline=1)
    assert result["node_labels"] == ["in_range"] * 3
    assert result["orientation"] == "baseline_nearer"
    assert result["delta"] == pytest.approx(-1 / 24)


@pytest.mark.parametrize("middle, category", [(9, "high_distance_overshoot"), (0, "low_distance_undershoot")])
def test_outside_sign_categories_and_endpoint_only_segments(middle, category):
    result = curve([4, middle, 1], actual=1, baseline=4)
    assert result["node_labels"][1] == category
    assert result["delta"] < 0
    assert result[f"contribution_{category}"] == pytest.approx(result["delta"])
    assert result["segment_endpoint_counts"] == {"one_inside_one_outside": 2}


def test_tolerance_is_label_only():
    result = curve([4, 4 + 1e-8, 1], actual=1, baseline=4)
    assert result["node_labels"][1] == "high_distance_overshoot"
    assert result["tolerance_node_labels"][1] == "in_range"
    assert result["delta"] < 0
    assert result["paired_status"] == "defined"


def test_exact_zero_gap_and_domain_roundoff_remain_explicit():
    result = curve([1, 2, 1], actual=1, baseline=1)
    assert result["paired_status"] == "undefined_zero_gap"
    assert result["delta"] is None
    almost = float(np.float32(-1e-6))
    result = a.paired_curve([0, 50, 100], [-2, -1, almost], almost, -2, "L2")
    assert result["n_native_l2_domain_roundoff"] == 2
    assert result["residual_actual"] == 0
    assert result["native_actual"] == almost


def test_transform_can_make_only_one_endpoint_gap_undefined():
    result = curve([1e-40, 0.5e-40, 0], actual=0, baseline=1e-40)
    assert result["gap_q"] != 0
    assert result["gap_l2"] == 0
    assert result["undefined_responses"] == ["L2"]
    assert result["paired_status"] == "undefined_zero_gap"


@pytest.mark.parametrize("grid, values, actual, baseline", [
    ([0, 50, 100], [0, float("nan"), -1], 0, -1),
    ([0, 100, 50], [0, -0.5, -1], 0, -1),
    ([0, 50, 100], [0, -0.5, -1], float("inf"), -1),
    ([0, 50, 100], [0, 0.5, -1], 0, -1),
])
def test_malformed_curve_or_score_rejected(grid, values, actual, baseline):
    with pytest.raises(IntegrityError):
        a.paired_curve(grid, values, actual, baseline, "Q")


def test_equal_episode_mean_does_not_weight_longer_episode_more():
    rows = [call("a", "long", 0), call("b", "long", 0), call("c", "short", 3)]
    result, episodes, membership = a.summarize_calls(rows, draws=10000)
    assert result["equal_episode_mean_point"] == 1.5
    assert result["paired_call_median_point"] == 0
    assert result["point_membership_sha256"] == result["ci_membership_sha256"]
    assert result["point_membership_sha256"] == sha256(json_bytes(membership))
    assert {e["n_defined_calls"] for e in episodes} == {1, 2}


def test_paired_median_is_not_difference_of_marginal_medians():
    q, l2 = [0, 2, 3], [0, 100, 1]
    rows = [call(str(i), str(i), x - y) for i, (x, y) in enumerate(zip(q, l2))]
    result, _, _ = a.summarize_calls(rows, draws=100)
    assert result["paired_call_median_point"] == 0
    assert np.median(q) - np.median(l2) == 1


def test_undefined_calls_and_fully_undefined_episode_remain_in_membership():
    rows = [call("a", "one", 1), call("b", "one", None, "undefined_zero_gap"),
            call("c", "two", None, "undefined_zero_gap")]
    result, _, members = a.summarize_calls(rows, draws=100)
    assert result["n_raw_calls"] == 3
    assert result["n_defined_calls"] == 1
    assert result["n_raw_episodes"] == 2
    assert result["n_fully_undefined_episodes"] == 1
    assert result["equal_episode_mean_ci_lo"] is None
    assert len(members["undefined"]) == 2
    all_bad, _, _ = a.summarize_calls(rows[1:], draws=100)
    assert all_bad["equal_episode_mean_point"] is None


def test_repeated_call_or_nonfinite_defined_value_rejected():
    with pytest.raises(IntegrityError):
        a.summarize_calls([call("a", "one", 1), call("a", "two", 2)])
    with pytest.raises(IntegrityError):
        a.summarize_calls([call("a", "one", float("nan"))])


def test_source_namespace_prevents_guessed_episode_join():
    row = dict(task="x", model="y", seed=42, episode=0)
    left = Record("a", "", 1, "", row)
    right = Record("b", "", 1, "", row)
    assert a.episode_key(left) != a.episode_key(right)


def test_bootstrap_matches_explicit_whole_episode_sampling():
    rows = [call("a", "A", 0), call("b", "A", 2), call("c", "B", 4), call("d", "C", 9)]
    result, _, _ = a.summarize_calls(rows, draws=200, seed=0)
    rng = np.random.default_rng(0)
    means, medians = [], []
    groups = [[0, 2], [4], [9]]
    for selected in rng.integers(0, 3, (200, 3)):
        means.append(np.mean([np.mean(groups[i]) for i in selected]))
        medians.append(np.median([v for i in selected for v in groups[i]]))
    assert [result[f"equal_episode_mean_{x}"] for x in ("ci_lo", "ci_hi")] == pytest.approx(np.percentile(means, [2.5, 97.5]))
    assert [result[f"paired_call_median_{x}"] for x in ("ci_lo", "ci_hi")] == pytest.approx(np.percentile(medians, [2.5, 97.5]))


def test_canonical_provenance_tamper_is_fatal(tmp_path):
    location = tmp_path / "analysis/revision/results/2026-09-30-v2"
    location.mkdir(parents=True)
    (location / "provenance.json").write_text("{}")
    protocol = json.loads((a.HERE / "protocol.json").read_bytes())
    with pytest.raises(IntegrityError, match="Canonical provenance"):
        a.authenticate(tmp_path, protocol)


def test_changed_raw_curve_bytes_are_rejected_before_rescoring(tmp_path):
    protocol = json.loads((a.HERE / "protocol.json").read_bytes())
    canonical_relative = Path(protocol["canonical_results"])
    shutil.copytree(a.ROOT / canonical_relative, tmp_path / canonical_relative)
    manifest = a.ROOT / protocol["input_manifest"]
    shutil.copyfile(manifest, tmp_path / protocol["input_manifest"])
    for name in ("__init__.py", "core.py", "analyze.py", "verify.py"):
        shutil.copyfile(a.ROOT / "analysis/revision" / name, tmp_path / "analysis/revision" / name)
    item = next(r for r in json.loads(manifest.read_bytes())["files"] if any(
        a.fnmatch.fnmatchcase(Path(r["path"]).name, pattern) for pattern in protocol["cohorts"].values()))
    destination = tmp_path / item["path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes((a.ROOT / item["path"]).read_bytes() + b" ")
    with pytest.raises(IntegrityError, match="Source hash mismatch"):
        a.authenticate(tmp_path, protocol)


def test_real_data_reproduction_and_membership(tmp_path):
    output = tmp_path / "fresh"
    provenance = a.run(output)
    assert len(provenance["artifacts_sha256"]) == 8
    for name, digest in provenance["artifacts_sha256"].items():
        assert sha256((output / name).read_bytes()) == digest
    with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == 8
    assert {int(r["n_raw_calls"]) for r in summaries} == {750}
    assert {int(r["n_raw_episodes"]) for r in summaries} == {30}
    memberships = json.loads(gzip.decompress((output / "membership.json.gz").read_bytes()))
    for row in summaries:
        assert row["point_membership_sha256"] == sha256(json_bytes(memberships[row["case"]]))
        assert row["ci_membership_sha256"] == row["point_membership_sha256"]
    with pytest.raises(IntegrityError, match="existing"):
        a.run(output)
    # When a released result is present, this is also a byte-level regression.
    released = a.HERE / "results/2026-10-01-v1/provenance.json"
    if released.exists():
        expected = json.loads(released.read_bytes())["artifacts_sha256"]
        assert provenance["artifacts_sha256"] == expected
