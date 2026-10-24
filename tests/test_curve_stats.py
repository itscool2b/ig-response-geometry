import gzip
import math

import numpy as np
import pytest
from scipy import stats

from analysis.paired_rescoring import curve_stats as cs
from analysis.revision.core import IntegrityError, Record, bootstrap_summary


def call(record, episode, delta, index=0, high=0, low=0, in_range=None, status="defined"):
    return dict(record_id=record, episode_id=episode, episode_key=[episode], delta=delta,
                policy_call_idx=index, paired_status=status, n_high_distance_overshoot=high,
                n_low_distance_undershoot=low,
                contribution_in_range=delta if in_range is None else in_range,
                contribution_high_distance_overshoot=0 if in_range is None else delta - in_range,
                contribution_low_distance_undershoot=0)


@pytest.mark.parametrize("n", [1, 2, 3, 10, 101, 10000])
def test_percentile_reproduces_numpy_linear_exactly(n):
    rng = np.random.default_rng(n)
    values = sorted((rng.standard_normal(n) * 10.0 ** rng.integers(-8, 8, n)).tolist())
    for q in (0, 2.5, 25, 50, 97.5, 100):
        assert cs.percentile(values, q) == float(np.percentile(values, q))
    assert cs.median(values) == float(np.median(values))


def test_percentile_requires_sorted_nonempty_input():
    with pytest.raises(IntegrityError):
        cs.percentile([], 50)
    with pytest.raises(IntegrityError):
        cs.percentile([2.0, 1.0], 50)


def test_fsum_mean_is_exactly_rounded():
    values = [1e16, 1.0, -1e16, 1.0]
    assert cs.fmean(values) == 0.5
    naive = 0.0
    for value in values:
        naive += value
    assert naive / 4 != 0.5


def test_spearman_matches_scipy_with_ties_and_is_none_when_undefined():
    rng = np.random.default_rng(3)
    x = rng.integers(0, 5, 60).astype(float).tolist()
    y = (np.array(x) + rng.standard_normal(60)).tolist()
    assert cs.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-14)
    assert cs.spearman([1.0, 2.0], [2.0, 1.0]) is None
    assert cs.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert cs.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert cs.average_ranks([5.0, 1.0, 5.0, 2.0]) == [3.5, 1.0, 3.5, 2.0]


def test_bootstrap_uses_the_frozen_index_stream():
    rng = np.random.default_rng(11)
    calls, episodes = [], {}
    for e in range(7):
        for i in range(int(rng.integers(1, 6))):
            value = float(rng.standard_normal())
            calls.append(call(f"r{e}-{i}", f"E{e}", value, index=i))
            episodes.setdefault(f"E{e}", []).append(value)
    result = cs.episode_bootstrap(calls, draws=1000, seed=0)
    frozen = bootstrap_summary([c["delta"] for c in calls], [c["episode_id"] for c in calls],
                               statistic="median", draws=1000, seed=0)
    assert (result["median_ci_lo"], result["median_ci_hi"]) == (frozen["ci_lo"], frozen["ci_hi"])
    keys = sorted(episodes)
    frozen_mean = bootstrap_summary([float(np.mean(episodes[k])) for k in keys], keys,
                                    statistic="mean", draws=1000, seed=0)
    assert result["mean_ci_lo"] == pytest.approx(frozen_mean["ci_lo"], rel=1e-12)
    assert result["mean_ci_hi"] == pytest.approx(frozen_mean["ci_hi"], rel=1e-12)
    # Explicit resampling of whole episodes with the same blocked stream.
    explicit_means, explicit_medians = [], []
    for block in cs.bootstrap_blocks(len(keys), 1000, 0):
        for row in block.tolist():
            explicit_means.append(math.fsum(cs.fmean(episodes[keys[i]]) for i in row) / len(keys))
            explicit_medians.append(cs.median([v for i in row for v in episodes[keys[i]]]))
    assert result["mean_ci_lo"] == cs.percentile(sorted(explicit_means), 2.5)
    assert result["median_ci_hi"] == cs.percentile(sorted(explicit_medians), 97.5)
    assert result["interval_status"] == "conditional_empirical_episode_percentile"


def test_bootstrap_interval_status_for_one_and_two_episodes():
    one = cs.episode_bootstrap([call("a", "A", 1.0), call("b", "A", 3.0)], draws=50, seed=0)
    assert one["interval_status"] == "insufficient_groups" and one["median_ci_lo"] is None
    assert one["median_point"] == 2.0 and one["mean_point"] == 2.0
    two = cs.episode_bootstrap([call("a", "A", 1.0), call("b", "B", 3.0)], draws=50, seed=0)
    assert two["interval_status"] == "two_group_descriptive_only"
    with pytest.raises(IntegrityError):
        cs.episode_bootstrap([call("a", "A", float("nan"))], draws=50, seed=0)


def test_summary_keeps_episode_and_call_weights_distinct_and_contributions_exact():
    calls = [call("a1", "A", -10.0, 0, high=1, in_range=0.5), call("a2", "A", 2.0, 1),
             call("a3", "A", 2.0, 2), call("b1", "B", 4.0, 0), call("c1", "C", 1.0, 0, low=1)]
    s = cs.summarize(calls, draws=200, seed=0)
    assert s["equal_episode_mean"] == pytest.approx((-2.0 + 4.0 + 1.0) / 3)
    assert s["equal_call_mean"] == pytest.approx(-1 / 5)
    assert s["paired_median"] == 2.0
    assert s["positive_calls"] == 4 and s["negative_calls"] == 1
    assert s["overshoot_calls"] == 1 and s["overshoot_share"] == 0.2
    assert s["fully_in_range_calls"] == 3 and s["fully_in_range_share"] == 0.6
    assert s["undershoot_calls"] == 1 and s["overshoot_positive_share"] == 0.0
    total = sum(s[f"equal_episode_contribution_{c}"] for c in cs.CATEGORIES)
    assert total == pytest.approx(s["equal_episode_mean"], abs=1e-15)
    assert s["episodes_any_overshoot"] == 1 and s["episodes_fully_in_range"] == 1
    assert s["median_episode_mean"] == 1.0 and s["positive_episode_means"] == 2
    index = cs.call_index_summary(calls)
    assert index["first_raw_calls"] == 3 and index["later_raw_calls"] == 2
    assert index["first_median"] == 1.0 and index["later_median"] == 2.0
    assert index["first_plus_later_contribution"] == pytest.approx(s["equal_episode_mean"], abs=1e-15)
    assert index["first_overshoot_share"] == pytest.approx(1 / 3)


def test_undefined_calls_stay_in_raw_counts_only():
    calls = [call("a", "A", 1.0), call("b", "A", None, status="undefined_zero_gap"), call("c", "B", 3.0)]
    s = cs.summarize(calls, draws=50, seed=0)
    assert (s["n_raw_calls"], s["n_defined_calls"], s["n_undefined_calls"]) == (3, 2, 1)
    with pytest.raises(IntegrityError):
        cs.summarize([call("a", "A", 1.0), call("a", "B", 2.0)], draws=50, seed=0)
    empty = cs.summarize([], draws=50, seed=0)
    assert empty["n_raw_calls"] == 0 and "paired_median" not in empty


def record(source, line, **row):
    base = dict(event="faithfulness", task="T", model="M", seed=1, episode=0, policy_call_idx=0)
    base.update(row)
    return Record(source, "0" * 64, line, f"{line:064d}", base)


def test_fingerprint_original_and_general_forms():
    q = [record("f", 1, vision_f_input=0.0, vision_f_baseline=-0.5, lang_f_input=0.0, lang_f_baseline=-0.25)]
    n = [record("f", 1, vision_f_input=-1e-6, vision_f_baseline=-5.0, lang_f_input=-1e-6, lang_f_baseline=-2e-6)]
    steps = [record("s", 1, event="step", vision_gap=0.5, lang_gap=0.25)]
    mark = cs.fingerprint(q, steps)
    assert mark["all_agree_original"] and mark["all_agree_general"]
    l2_steps = [record("s", 1, event="step", vision_gap=5.0 - 1e-6, lang_gap=1e-6)]
    mark = cs.fingerprint(n, l2_steps)
    assert mark["all_agree_general"] and not mark["all_agree_original"]
    assert mark["lang_original_agree"] == 0 and mark["vision_original_agree"] == 1
    missing = cs.fingerprint(q, [record("s", 1, event="step", episode=5, vision_gap=0.5, lang_gap=0.25)])
    assert missing["n_missing"] == 1 and not missing["all_agree_general"]


def test_episode_structure_requires_contiguous_calls_and_one_end():
    faith = [record("f", i + 1, policy_call_idx=i) for i in range(3)]
    end = record("s", 9, event="episode_end", success=True, policy_calls=3)
    summary, labels = cs.episode_structure(faith, cs.episode_ends([end]))
    assert summary["successes"] == 1 and summary["success_calls"] == 3 and not summary["call_count_mismatches"]
    assert labels[("T", "M", 1, 0)] is True
    with pytest.raises(IntegrityError, match="Noncontiguous"):
        cs.episode_structure(faith[:1] + faith[2:], cs.episode_ends([end]))
    with pytest.raises(IntegrityError, match="lacks"):
        cs.episode_structure(faith, {})
    with pytest.raises(IntegrityError, match="non-boolean"):
        cs.episode_ends([record("s", 9, event="episode_end", success=1)])


def test_native_scale_check_rejects_nonzero_actual_scores():
    good = [record("f", 1, vision_f_input=-0.0, lang_f_input=0.0)]
    assert cs.native_scale_check(good, "Q")["n_rows"] == 1
    with pytest.raises(IntegrityError):
        cs.native_scale_check([record("f", 1, vision_f_input=-1e-6, lang_f_input=0.0)], "Q")


def test_rescore_reads_max_interior_residual_and_identity():
    row = dict(k_grid_auc=[0, 50, 100], vision_insertion_curve=[-4 / 1024, -9 / 1024, 0.0],
               vision_f_input=0.0, vision_f_baseline=-4 / 1024)
    result = cs.rescore(record("f", 1, **row), "vision", "insertion")
    assert result["residual_baseline"] == 4.0 and result["max_interior_residual"] == 9.0
    assert result["n_high_distance_overshoot"] == 1 and result["delta"] < 0
    assert result["episode_key"] == ["f", "T", "M", 1, 0]


def test_deterministic_writers():
    rows = [dict(a=0.1, b=None, c=[1, 2], d=True), dict(a=-0.0, b=1, c={"k": 1}, d=False)]
    blob = cs.csv_bytes(rows, ("a", "b", "c", "d"))
    assert blob == b'a,b,c,d\n0.1,,"[1,2]",True\n-0.0,1,"{""k"":1}",False\n'
    assert cs.gzip_bytes(blob) == cs.gzip_bytes(blob)
    assert gzip.decompress(cs.gzip_bytes(blob)) == blob
    with pytest.raises(IntegrityError):
        cs.csv_bytes([dict(a=float("inf"))], ("a",))
    assert cs.json_text({"b": 1, "a": [0.1]}) == '{\n  "a": [\n    0.1\n  ],\n  "b": 1\n}\n'


def test_named_authentication_rejects_unknown_or_rehashed_inputs():
    with pytest.raises(IntegrityError, match="absent"):
        cs.authenticate_named(cs.ROOT, ["data/not-a-file.jsonl"])
    name = "data/metrics_PickCube-v1_1b_seed42.jsonl"
    with pytest.raises(IntegrityError, match="Protocol hash"):
        cs.authenticate_named(cs.ROOT, [name], expected={name: "0" * 64})
    with pytest.raises(IntegrityError, match="repeated"):
        cs.authenticate_named(cs.ROOT, [name, name])
