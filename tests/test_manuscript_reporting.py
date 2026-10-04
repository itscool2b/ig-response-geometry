"""Scientific and display invariants for the retrospective reporting layer."""
from copy import deepcopy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from analysis.manuscript_revision import figures
from analysis.manuscript_revision.report import (
    CASES, ROOT, build_report, read_csv, summarize_case, PAIRED, INFLUENCE,
    summarize_call_indices, summarize_gaps, summarize_grid, schedule_bound,
    write_report, OUTPUT, VERSION,
)
from analysis.revision.core import IntegrityError
from scripts.build_revision_assets import plain_abstract


@pytest.fixture(scope="module")
def saved():
    return build_report()


def test_all_eight_cases_preserve_populations_and_adverse_results(saved):
    report, calls = saved
    assert tuple(row["case"] for row in report["paired"]) == CASES
    assert len(calls) == 8
    assert sum(row["mean"] < 0 for row in report["paired"]) == 6
    assert all(row["median"] > 0 for row in report["paired"])
    assert all((row["n_raw_calls"], row["n_defined_calls"], row["n_episodes"]) == (750, 750, 30)
               for row in report["paired"])
    assert [row["overshoot_calls"] for row in report["paired"]] == [312, 127, 555, 281, 292, 120, 491, 312]
    assert [row["positive_calls"] for row in report["paired"]] == [661, 694, 567, 664, 675, 707, 631, 624]
    assert min(row["minimum"] for row in report["paired"]) < -5200
    assert max(row["maximum"] for row in report["paired"]) < .25
    assert "original" in report["estimator_history"]
    assert "not included" in report["optional_ranking_comparison"]


def test_decomposition_is_checked_before_rounding_and_sign_changes_are_kept(saved):
    report, _ = saved
    for row in report["paired"]:
        assert math.isclose(row["in_range_contribution"] + row["overshoot_contribution"] + row["undershoot_contribution"],
                            row["mean"], abs_tol=1e-10)
    assert [row["omission_sign_preserved"] for row in report["paired"]] == [30, 30, 30, 27, 30, 30, 30, 29]
    # The former four-decimal equation was not an exact equality after rounding.
    q = report["paired"][0]
    assert round(q["in_range_contribution"], 4) + round(q["overshoot_contribution"], 4) != round(q["mean"], 4)


def test_success_counts_use_episode_end_records_and_exclude_budget_extensions(saved):
    report, _ = saved
    actual = {}
    for row in report["task_success"]:
        count, wins = actual.get(row["source"], (0, 0))
        actual[row["source"]] = count + row["episodes"], wins + row["successes"]
        assert len(row["record_ids"]) == row["episodes"]
        assert "m128" not in row["file"]
    assert actual == {"Q-ranking regeneration": (30, 0), "L-ranking regeneration": (30, 0),
                      "Verification": (400, 2), "Historical 1B": (60, 15)}


def test_reporting_rejects_changed_call_value(saved):
    _, calls = saved
    case = CASES[0]
    selected = deepcopy(calls[case])
    selected[0]["delta"] = str(float(selected[0]["delta"]) + 1)
    summary = next(row for row in read_csv(ROOT / PAIRED / "summary.csv") if row["case"] == case)
    geometry = next(row for row in read_csv(ROOT / PAIRED / "geometry.csv") if row["case"] == case)
    omission = next(row for row in read_csv(ROOT / INFLUENCE / "case_summary.csv") if row["case"] == case)
    with pytest.raises(IntegrityError, match="cross-check failed"):
        summarize_case(case, selected, summary, geometry, omission)
    with pytest.raises(IntegrityError, match="repeats a physical record"):
        summarize_case(case, calls[case] + calls[case][:1], summary, geometry, omission)


def test_undefined_calls_do_not_enter_positive_fraction_but_remain_in_raw_accounting():
    case = CASES[0]
    common = dict(case=case, n_high_distance_overshoot="0", n_low_distance_undershoot="0")
    calls = [dict(common, record_id="a", paired_status="defined", episode_id="e1", delta=".2",
                  contribution_in_range=".2", contribution_high_distance_overshoot="0", contribution_low_distance_undershoot="0"),
             dict(common, record_id="b", paired_status="defined", episode_id="e2", delta="-.4",
                  contribution_in_range="0", contribution_high_distance_overshoot="-.4", contribution_low_distance_undershoot="0"),
             dict(common, record_id="c", paired_status="undefined_zero_gap", episode_id="e3", delta="")]
    calls[1]["n_high_distance_overshoot"] = "1"
    summary = dict(equal_episode_mean_point="-.1", paired_call_median_point="-.1", n_raw_calls="3",
                   n_defined_calls="2", n_defined_episodes="2", n_negative_delta="1", n_zero_delta="0",
                   equal_episode_contribution_in_range=".1", equal_episode_contribution_high_distance_overshoot="-.2",
                   equal_episode_contribution_low_distance_undershoot="0", point_membership_sha256="same",
                   ci_membership_sha256="same", paired_call_median_ci_lo="-.4", paired_call_median_ci_hi=".2",
                   equal_episode_mean_ci_lo="-.4", equal_episode_mean_ci_hi=".2", bootstrap_draws="10000", bootstrap_seed="0")
    omission = dict(n_sign_preserved="1", n_omissions="2", omission_mean_min="-.4", omission_mean_max=".2")
    result = summarize_case(case, calls, summary, {"calls_any_high_distance_overshoot": "1"}, omission)
    assert result["positive_percent"] == 50
    assert result["overshoot_percent"] == pytest.approx(100/3)
    assert result["n_undefined_calls"] == 1 and result["n_raw_calls"] == 3
    summary["ci_membership_sha256"] = "different"
    with pytest.raises(IntegrityError, match="populations differ"):
        summarize_case(case, calls, summary, {"calls_any_high_distance_overshoot": "1"}, omission)


def test_numerical_rankings_and_coverage_are_distinct(saved):
    report, _ = saved
    rows = report["numerical"]
    assert len(rows) == 432  # 72 cells, three budget pairs, two map types.
    assert {row["ranking"] for row in rows} == {"IG", "path_gradient"}
    assert len({row["context_id"] for row in rows}) == 12
    assert len({(row["stratum"], row["episode"]) for row in rows}) == 6
    assert all(row["report_sha256"] and row["reference_gap"] > 0 for row in rows)
    assert report["numerical_original_summary"]["repeated_condition_and_coverage_check_counts"] == {"satisfies": 2880}
    assert report["numerical_original_summary"]["repeatability_equality_check_counts"] == {"satisfies": 1800}
    assert report["numerical_coverage"]["planned_one_b"] == 8
    assert report["numerical_coverage"]["later_completion_status"] == "later_completion_unknown_and_scientifically_unassessed_here"


def test_first_call_breakdown_preserves_population_and_tail_denominators(saved):
    report, _ = saved
    for split, paired in zip(report["call_index_summary"], report["paired"]):
        assert split["first_raw_calls"] == split["first_defined_calls"] == 30
        assert split["later_raw_calls"] == split["later_defined_calls"] == 720
        assert split["n_defined_calls"] == split["n_raw_calls"] == 750
        assert split["first_mean_contribution"] + split["later_mean_contribution"] == pytest.approx(paired["mean"])
        assert split["later_mean_contribution"] == pytest.approx(split["later_mean"] * 720 / 750)
        assert split["first_overshoot_contribution"] + split["later_overshoot_contribution"] == pytest.approx(paired["overshoot_contribution"])
        assert split["first_overshoot_contribution_fraction"] + split["later_overshoot_contribution_fraction"] == pytest.approx(1)
    q, n = report["call_index_summary"][0], report["call_index_summary"][4]
    assert q["first_mean_contribution"] == pytest.approx(-16.91992930772643)
    assert n["first_mean_contribution"] == pytest.approx(-17.752352145011045)
    assert q["first_overshoot_contribution_fraction"] == pytest.approx(.9833394477396113)
    assert n["first_overshoot_contribution_fraction"] == pytest.approx(.9507792735298273)
    assert f"{n['first_median']:.2f}" == "-0.84"


def _indexed_call(name, episode, index, value, overshoot=0):
    return dict(case="Q:vision:insertion", record_id=name, episode_id=episode, policy_call_idx=str(index),
                paired_status="defined" if value is not None else "undefined_zero_gap",
                delta=str(value) if value is not None else "", contribution_high_distance_overshoot=str(overshoot),
                n_high_distance_overshoot=str(int(overshoot < 0)))


def test_call_index_contributions_keep_episode_weights_when_lengths_differ():
    calls = [_indexed_call("a", "e1", 0, 2), _indexed_call("b", "e2", 0, -4, -4),
             _indexed_call("c", "e2", 1, 12), _indexed_call("d", "e3", 0, None)]
    split = summarize_call_indices("Q:vision:insertion", calls)
    assert split["mean"] == pytest.approx(3)  # (2 + (-4 + 12)/2)/2, not 10/3.
    assert split["first_mean_contribution"] == pytest.approx(0)
    assert split["later_mean_contribution"] == pytest.approx(3)
    assert split["later_mean"] == 12
    assert split["first_undefined_calls"] == 1
    assert split["first_overshoot_contribution_fraction"] == 1


def test_empty_subgroups_undefined_gaps_and_zero_tails_remain_explicit():
    later_only = summarize_call_indices("Q:vision:insertion", [_indexed_call("a", "e1", 2, .2)])
    assert later_only["first_mean"] is None and later_only["first_median"] is None
    assert later_only["first_overshoot_percent"] is None
    assert later_only["first_mean_contribution"] == 0
    assert later_only["first_overshoot_contribution_fraction"] is None
    undefined = summarize_call_indices("Q:vision:insertion", [_indexed_call("a", "e1", 0, None)])
    assert undefined["n_raw_calls"] == 1 and undefined["n_defined_calls"] == 0
    assert undefined["mean"] is None and undefined["first_mean_contribution"] is None
    assert undefined["later_mean_contribution"] is None


def test_call_index_summary_rejects_invalid_indices_duplicates_and_nonfinite_values():
    call = _indexed_call("a", "e1", 0, .2)
    for index in ("-1", "0.5", "nan"):
        with pytest.raises(IntegrityError, match="index"):
            summarize_call_indices(call["case"], [dict(call, policy_call_idx=index)])
    with pytest.raises(IntegrityError, match="Duplicate"):
        summarize_call_indices(call["case"], [call, call])
    with pytest.raises(IntegrityError, match="Nonfinite"):
        summarize_call_indices(call["case"], [dict(call, delta="nan")])


def test_gap_and_schedule_diagnostics_are_saved_data_facts(saved):
    report, _ = saved
    gaps = report["baseline_gap_summary"][0]
    assert gaps["baseline_below_half"] == 17 and gaps["first_baseline_below_half"] == 14
    assert gaps["residual_baseline_minimum"] == pytest.approx(.017419494688510895)
    assert gaps["residual_baseline_median"] == pytest.approx(11.816057205200195)
    assert sum(row["repeated_one_five_residuals"] for row in report["grid_diagnostics"] if ":lang:" in row["case"]) == 3000
    assert sum(row["repeated_one_five_residuals"] for row in report["grid_diagnostics"] if ":vision:" in row["case"]) == 0
    for row in report["grid_diagnostics"]:
        assert row["bound_checked_calls"] == 750
        assert sum(row["trapezoid_weights"]) == pytest.approx(1)
        assert row["endpoint_weight"] == pytest.approx(.13)
        assert row["upper_bound"] == pytest.approx(.2175)
        assert row["maximum_delta"] <= row["upper_bound"]
    assert max(row["maximum_percent_of_bound"] for row in report["grid_diagnostics"]) == pytest.approx(98.26205002032998)


def test_schedule_validation_and_no_interior_node_bound():
    for grid in ([1, 100], [0, 100, 100], [0, 50, 20, 100], [0, math.nan, 100], [0]):
        with pytest.raises(IntegrityError, match="schedule"):
            schedule_bound(grid)
    assert schedule_bound([0, 100])["upper_bound"] == 0


def test_grid_summary_excludes_undefined_gaps_without_hiding_them():
    call = dict(case="Q:vision:insertion", grid_percent="[0,100]", residual_points="[0,0]",
                residual_actual="0", residual_baseline="0", paired_status="undefined_zero_gap")
    result = summarize_grid(call["case"], [call])
    assert result["n_calls"] == 1 and result["bound_checked_calls"] == 0
    assert result["maximum_delta"] is None and result["maximum_percent_of_bound"] is None
    call.update(gap_q="0", gap_l2="0", policy_call_idx="0")
    gap = summarize_gaps(call["case"], [call])
    assert gap["residual_baseline_zero_count"] == gap["gap_q_zero_count"] == gap["gap_l2_zero_count"] == 1


def test_schedule_bound_keeps_negative_overshoots_and_rejects_positive_violations():
    call = dict(case="Q:vision:insertion", grid_percent="[0,50,100]", residual_points="[1,10000,0]",
                residual_actual="0", residual_baseline="1", paired_status="defined",
                normalized_q="[0,-9999,1]", normalized_l2="[0,-99,1]", delta="-4950")
    result = summarize_grid(call["case"], [call])
    assert result["bound_checked_calls"] == 1 and result["maximum_delta"] == -4950
    assert result["upper_bound"] == .125
    with pytest.raises(IntegrityError, match="ordinate"):
        summarize_grid(call["case"], [dict(call, normalized_q="[0,1,1]", normalized_l2="[0,0,1]")])
    with pytest.raises(IntegrityError, match="AUC"):
        summarize_grid(call["case"], [dict(call, delta="0.2")])


def test_new_reporting_outputs_leave_previous_version_untouched(saved, tmp_path):
    report, _ = saved
    old = tmp_path / "analysis/manuscript_revision/results/2026-10-03-v1/report.json"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"preserved prior report")
    output, manifest = write_report(report, tmp_path)
    assert VERSION == "2026-10-03-v2" and output == tmp_path / OUTPUT
    assert old.read_bytes() == b"preserved prior report"
    assert {"call_index_summary.csv", "baseline_gap_summary.csv", "grid_diagnostics.csv"} <= set(manifest["outputs"])


def test_prose_macros_have_the_same_rounding_as_the_table(saved):
    report, _ = saved
    macros = figures.facts(report)
    assert macros["OvershootShareMin"]["display"] == "16.0"
    assert macros["OvershootShareMax"]["display"] == "74.0"
    assert macros["RescoreQDeletionQ"]["display"] == "0.448"
    assert macros["RescoreQDeletionL"]["display"] == "0.291"
    assert macros["VerificationSuccesses"]["value"] == 2
    assert macros["NumericalVisionQResidual"]["display"] == "0.124"
    assert macros["NumericalVisionNResidual"]["display"] == "0.045"
    assert macros["NumericalVisionQCoord"]["display"] == "2.05"
    assert macros["NumericalVisionNCoord"]["display"] == "1.23"
    assert macros["TailDiscrepancyRatio"]["display"] == "37{,}500"
    assert macros["NormVisionInsertionFirstMedian"]["display"] == "-0.845"
    table = figures.paired_table(report)
    assert "\\small" not in table.replace("\\smallskip", "")
    assert "\\begin{table}" not in table
    for row in report["paired"]:
        assert f"{row['median']:.3f}" in table
        assert f"{row['positive_percent']:.1f}\\%" in table


def test_full_tail_plot_keeps_every_point_and_does_not_join_cohorts(saved, monkeypatch, tmp_path):
    report, calls = saved
    captured = {}
    monkeypatch.setattr(figures, "save", lambda fig, directory, stem: captured.setdefault(stem, fig))
    figures.style()
    figures.paired_effects(tmp_path, report, calls)
    fig = captured["paired_effects"]
    ax = fig.axes[0]
    assert len(ax.collections) == 8
    assert sum(len(item.get_offsets()) for item in ax.collections) == 6000
    lower, upper = ax.get_xlim()
    assert all(lower < point[0] < upper for collection in ax.collections for point in collection.get_offsets())
    assert ax.get_xscale() == "symlog"
    assert all(len(line.get_xdata()) <= 2 for line in ax.lines)
    plt.close(fig)


def test_numerical_figure_never_clips_large_failures(saved, monkeypatch, tmp_path):
    report, _ = saved
    captured = {}
    monkeypatch.setattr(figures, "save", lambda fig, directory, stem: captured.setdefault(stem, fig))
    figures.numerical_diagnostics(tmp_path, report, ROOT)
    fig = captured["numerical_diagnostics"]
    assert len(fig.axes) == 6
    for ax in fig.axes:
        assert len(ax.lines) == 37  # 12 contexts per modality plus the threshold.
        top = ax.get_ylim()[1]
        assert all(0 <= value < top for line in ax.lines for value in line.get_ydata())
    plt.close(fig)


def test_pipeline_text_fits_within_the_saved_canvas(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(figures, "save", lambda fig, directory, stem: captured.setdefault(stem, fig))
    figures.style()
    figures.pipeline(tmp_path)
    fig = captured["pipeline"]
    fig.canvas.draw()
    bounds = fig.bbox
    renderer = fig.canvas.get_renderer()
    for text in fig.axes[0].texts:
        box = text.get_window_extent(renderer)
        assert bounds.x0 <= box.x0 <= box.x1 <= bounds.x1
        assert bounds.y0 <= box.y0 <= box.y1 <= bounds.y1
    plt.close(fig)


def test_plain_abstract_expands_whole_macro_names_only():
    source = r"\begin{abstract}Across \RescoreCalls{} calls, Q and $L$ differed by \PairedMedianMin{} to \PairedMedianMax{}.\end{abstract}"
    macros = {"RescoreCalls": {"display": "750"}, "PairedMedianMin": {"display": "0.104"}, "PairedMedianMax": {"display": "0.159"}}
    assert plain_abstract(source, macros) == "Across 750 calls, Q and L differed by 0.104 to 0.159.\n"
    with pytest.raises(ValueError, match="Unexpanded"):
        plain_abstract(source.replace("\\RescoreCalls", "\\RescoreCallsUnknown"), macros)


def test_plain_abstract_omits_review_comments_and_preserves_percentages():
    source = "\\begin{abstract}\n% BEGIN approved-00-00-0\nChanges occurred in 16.0\\% of calls. % internal note\n% END approved-00-00-0\n\\end{abstract}"
    assert plain_abstract(source, {}) == "Changes occurred in 16.0% of calls.\n"
