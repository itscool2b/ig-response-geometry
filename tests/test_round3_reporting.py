"""Regression checks for the new descriptive summaries, using preserved data."""
from copy import deepcopy
import json
import math

import pytest

from analysis.manuscript_revision.report import (
    ROOT, NUMERICAL, build_report, read_json, schedule_bound,
    summarize_case, summarize_cohort_relationship, summarize_language_schedule,
    numerical_evaluation, summarize_numerical_violations,
)
from analysis.revision.core import IntegrityError
from analysis.revision.response_geometry_theory import normalized_responses


@pytest.fixture(scope="module")
def saved():
    return build_report()


@pytest.fixture(scope="module")
def projection():
    return read_json(ROOT / NUMERICAL / "inputs/sealed_v6.json")


def test_geometric_shares_separate_observed_strict_positivity_from_guaranteed_sign(saved):
    report, _ = saved
    assert [row["fully_in_range_calls"] for row in report["paired"]] == [438, 623, 195, 469, 458, 630, 259, 438]
    assert [row["overshoot_positive_calls"] for row in report["paired"]] == [223, 71, 372, 195, 217, 77, 372, 186]
    for row in report["paired"]:
        assert row["fully_in_range_positive_calls"] == row["fully_in_range_calls"]
        assert row["fully_in_range_calls"] + row["overshoot_defined_calls"] == row["n_defined_calls"]
    q = report["paired"][0]
    assert q["fully_in_range_delta_q25"] == pytest.approx(.09783559740650244)
    assert q["fully_in_range_delta_q75"] == pytest.approx(.13489796834069268)


def test_in_range_equality_case_is_not_reported_as_positive():
    case = "Q:vision:insertion"
    call = dict(case=case, record_id="one", paired_status="defined", episode_id="ep",
                delta="0", contribution_in_range="0", contribution_high_distance_overshoot="0",
                contribution_low_distance_undershoot="0", n_high_distance_overshoot="0", n_low_distance_undershoot="0")
    summary = dict(equal_episode_mean_point="0", paired_call_median_point="0", n_raw_calls="1",
                   n_defined_calls="1", n_defined_episodes="1", n_negative_delta="0", n_zero_delta="1",
                   equal_episode_contribution_in_range="0", equal_episode_contribution_high_distance_overshoot="0",
                   equal_episode_contribution_low_distance_undershoot="0", point_membership_sha256="one",
                   ci_membership_sha256="one", paired_call_median_ci_lo="0", paired_call_median_ci_hi="0",
                   equal_episode_mean_ci_lo="0", equal_episode_mean_ci_hi="0", bootstrap_draws="10000", bootstrap_seed="0")
    omission = dict(n_sign_preserved="0", n_omissions="0", omission_mean_min="0", omission_mean_max="0")
    row = summarize_case(case, [call], summary, {"calls_any_high_distance_overshoot": "0"}, omission)
    assert row["fully_in_range_calls"] == 1
    assert row["fully_in_range_positive_calls"] == 0
    assert row["fully_in_range_positive_percent"] == 0
    assert row["overshoot_positive_percent"] is None
    assert row["fully_in_range_delta_q25"] == row["fully_in_range_delta_q75"] == 0


def test_actual_cohort_relationship_uses_labels_not_physical_record_overlap(saved):
    report, _ = saved
    row = report["cohort_relationship"]
    assert row["aligned_calls"] == row["unequal_baseline_calls"] == 750
    assert row["extreme_operator"] == "<" and row["extreme_threshold"] == -100
    assert row["extreme_by_ranking"]["Q"]["extreme_calls"] == 7
    assert row["extreme_by_ranking"]["L2"]["extreme_calls"] == 8
    assert row["extreme_episode_sets_equal"]
    assert sorted(item[2] for item in row["shared_extreme_episodes"]) == [44, 46, 50, 54, 55, 147, 148]
    assert row["call_index_correlations"][0]["log_baseline_correlation"] == pytest.approx(.9915852518078411)
    assert row["call_index_correlations"][-1]["log_baseline_correlation"] == pytest.approx(.658049212916001)
    assert "do not authenticate" in row["interpretation"]


def _cohorts():
    result = {}
    for rank in ("Q", "L2"):
        case = rank + ":vision:insertion"
        result[case] = [dict(case=case, record_id=f"{rank}-{call}",
            episode_key=json.dumps([rank + ".jsonl", "task", "170m", 42, 0]),
            policy_call_idx=str(call), residual_baseline="0", delta=str(effect), paired_status="defined")
            for call, effect in enumerate([-100, -100.1])]
    return result


def test_cohort_tail_threshold_is_strict_and_missing_or_duplicate_keys_fail():
    calls = _cohorts()
    row = summarize_cohort_relationship(calls)
    assert row["extreme_by_ranking"]["Q"]["extreme_calls"] == 1
    assert all(item["log_baseline_correlation"] is None for item in row["call_index_correlations"])
    assert row["unequal_baseline_calls"] == 0
    for mutation, message in (("missing", "align"), ("duplicate", "Duplicate"), ("nonfinite", "baseline"), ("invalid", "identity")):
        changed = deepcopy(calls)
        selected = changed["L2:vision:insertion"]
        if mutation == "missing":
            selected.pop()
        elif mutation == "duplicate":
            selected.append(dict(selected[0], record_id="different-physical-occurrence"))
        elif mutation == "nonfinite":
            selected[0]["residual_baseline"] = "nan"
        else:
            selected[0]["policy_call_idx"] = "0.5"
        with pytest.raises(IntegrityError, match=message):
            summarize_cohort_relationship(changed)


def _language_call(residuals, grid=None):
    grid = [0, 1, 5, 100] if grid is None else grid
    responses = normalized_responses(residuals, 1)
    q, n = responses["quadratic"], responses["l2"]
    delta = sum(w * (a-b) for w, a, b in zip(schedule_bound(grid)["trapezoid_weights"], q, n))
    return dict(case="Q:lang:insertion", record_id="one", grid_percent=json.dumps(grid),
                residual_points=json.dumps(residuals), normalized_q=json.dumps(list(q)),
                normalized_l2=json.dumps(list(n)), delta=str(delta), paired_status="defined",
                residual_actual="0", residual_baseline="1")


def test_language_schedule_sensitivity_and_its_conditional_bound(saved):
    report, _ = saved
    rows = report["language_schedule_sensitivity"]
    assert [row["median_change"] for row in rows] == pytest.approx([
        .001128899265841088, -.0037135842937593005, .002083101815851529, -.003881999933132249])
    for row in rows:
        assert row["bound_uses_equal_removed_next_ordinates"]
        assert row["in_range_per_call_bound"] == pytest.approx(.005)
        assert row["maximum_in_range_absolute_change"] <= row["in_range_per_call_bound"]
    unequal = _language_call([1, .25, .5, 0])
    sensitivity = summarize_language_schedule(unequal["case"], [unequal])
    assert not sensitivity["bound_uses_equal_removed_next_ordinates"]
    assert sensitivity["in_range_per_call_bound"] == pytest.approx(.00625)
    # A fully in-range curve containing only endpoint residuals has zero change.
    equality = _language_call([1, 1, 1, 0])
    zero = summarize_language_schedule(equality["case"], [equality])
    assert zero["original_median"] == zero["reduced_median"] == zero["median_change"] == 0


def test_schedule_preserves_overshoots_and_undefined_counts_and_rejects_corruption():
    overshoot = _language_call([1, 100, 100, 0])
    undefined = dict(overshoot, record_id="undefined", paired_status="undefined_zero_gap", delta="")
    row = summarize_language_schedule(overshoot["case"], [overshoot, undefined])
    assert row["original_median"] < 0 and row["reduced_median"] < 0
    assert row["n_raw_calls"] == 2 and row["n_defined_calls"] == 1
    assert row["fully_in_range_calls"] == 0 and row["maximum_in_range_absolute_change"] is None
    no_defined = summarize_language_schedule(overshoot["case"], [undefined])
    assert no_defined["original_median"] is no_defined["median_change"] is None
    for bad in (dict(overshoot, delta="nan"), dict(overshoot, residual_points="[1]"),
                dict(overshoot, normalized_q="[0,NaN,0,1]")):
        with pytest.raises(IntegrityError):
            summarize_language_schedule(overshoot["case"], [bad])
    with pytest.raises(IntegrityError, match="Duplicate"):
        summarize_language_schedule(overshoot["case"], [overshoot, overshoot])


def test_numerical_evaluation_examples_remain_specific_to_map_response_and_budget(saved):
    report, _ = saved
    checks = report["numerical_evaluation_checks"]
    assert len(checks) == 3888
    def select(call, criterion, candidate, reference, **extra):
        matches = [row for row in checks if all(row[key] == value for key, value in dict(
            stratum="picksingleycb-170m", episode=1, call=call, modality="state", target="Q", ranking="IG",
            criterion=criterion, candidate_m=candidate, reference_m=reference, **extra).items())]
        assert len(matches) == 1
        assert matches[0]["report_sha256"]
        return matches[0]["value"]
    assert select(0, "group_spearman", 256, 512) == pytest.approx(5/6)
    assert select(0, "normalized_auc_difference", 256, 512, direction="insertion", response="Q") == pytest.approx(.19118083571074995)
    assert select(0, "normalized_auc_difference", 256, 512, direction="deletion", response="Q") == pytest.approx(.09993975055092673)
    assert select(12, "normalized_auc_difference", 512, 1024, direction="insertion", response="Q") == 0
    assert select(12, "rms_curve_difference", 512, 1024, direction="insertion", response="RMS") == 0
    assert report["numerical_violation_summary"]["full_ladder_status_counts"]["violates"] == 302
    assert [row["arms_with_finest_pair_violation"] for row in report["numerical_violation_summary"]["finest_pair_by_modality_target"]] == [0, 0, 2, 2, 4, 1]


def test_numerical_projection_rejects_missing_duplicate_or_nonfinite_checks(projection):
    cell = deepcopy(next(row for row in projection["rows"] if row["status"] == "complete_diagnostics"))
    check = next(row for row in cell["checks"] if row["criterion"] == "group_spearman")
    for mutation in ("duplicate", "missing", "nonfinite", "cell"):
        changed = deepcopy(cell)
        if mutation == "duplicate":
            changed["checks"].append(deepcopy(check))
        elif mutation == "missing":
            changed["checks"].remove(check)
        elif mutation == "nonfinite":
            next(row for row in changed["checks"] if row["criterion"] == "group_spearman")["value"] = math.nan
        rows = [changed, changed] if mutation == "cell" else [changed]
        with pytest.raises(IntegrityError):
            numerical_evaluation({"rows": rows})
    bad = deepcopy(projection)
    bad["criterion_status_counts"]["violates"] += 1
    with pytest.raises(IntegrityError, match="counts"):
        summarize_numerical_violations(bad, read_json(ROOT / NUMERICAL / "outputs/summary.json"))


def test_common_response_can_reverse_fixed_curve_auc_order_at_positive_epsilon():
    weights = [1/6, 1/3, 1/3, 1/6]
    scores = []
    for residuals in ([1, .25, .25, 0], [1, .64, 0, 0]):
        responses = normalized_responses(residuals, 1, epsilon=1e-12)
        q, n = responses["quadratic"], responses["l2"]
        scores.append(tuple(sum(w * ordinate for w, ordinate in zip(weights, response)) for response in (q, n)))
    assert scores[0][0] == pytest.approx(2/3)
    assert scores[1][0] == pytest.approx(.62)
    assert scores[0][1] == pytest.approx(.5, abs=1e-6)
    assert scores[1][1] == pytest.approx(17/30, abs=1e-6)
    assert scores[0][0] > scores[1][0] and scores[0][1] < scores[1][1]
