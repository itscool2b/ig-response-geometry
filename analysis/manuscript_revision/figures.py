"""Publication displays generated only from authenticated saved evidence."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter
import numpy as np

from analysis.revision.response_geometry_theory import normalized_responses, polynomial_rank_reversal
from .report import CASES, NUMERICAL, read_json

BLUE = "#235b78"
RUST = "#b04b2b"
GREEN = "#447b63"
INK = "#202c33"
PALE = "#eaf0f3"
STEMS = ("pipeline", "response_mechanism", "paired_effects", "analytic_toys", "numerical_diagnostics")


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": True, "pdf.fonttype": 42,
        "savefig.facecolor": "white", "text.color": INK, "axes.labelcolor": INK})


def save(fig, directory, stem):
    fig.savefig(directory / (stem + ".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(directory / (stem + ".png"), dpi=220)
    plt.close(fig)


def pipeline(directory):
    fig, ax = plt.subplots(figsize=(6.5, 2.7))
    fig.subplots_adjust(left=.015, right=.985, top=.97, bottom=.04)
    ax.set(xlim=(0, 10), ylim=(.6, 4))
    ax.axis("off")
    def box(x, y, w, h, text, color=PALE, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.07,rounding_size=.07",
                                   facecolor=color, edgecolor="#647985", linewidth=.8))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=10,
                weight="bold" if bold else "normal", linespacing=1.25)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color="#647985", lw=1.1))
    box(.15, 2.5, 2.75, 1.0, "One policy call:\nimage, instruction,\nstate")
    box(3.6, 2.5, 2.85, 1.0, "Choose target Q or N\nIntegrate gradients", "#e6f0f5", True)
    box(7.15, 2.5, 2.65, 1.0, "Rank input\nfeature groups")
    box(7.15, .9, 2.65, 1.0, "Insert or delete\nranked groups")
    box(3.6, .9, 2.85, 1.0, "Choose response\nQ or N", "#f6e9e2", True)
    box(.15, .9, 2.75, 1.0, "Normalize the curve\nCalculate area (AUC)")
    arrow((2.98, 3), (3.51, 3)); arrow((6.53, 3), (7.06, 3))
    arrow((8.47, 2.42), (8.47, 1.98))
    arrow((7.06, 1.4), (6.53, 1.4)); arrow((3.51, 1.4), (2.98, 1.4))
    ax.text(5.025, 3.75, "1. RANKING TARGET", color=INK, fontsize=9, weight="bold", ha="center")
    ax.text(5.025, 2.12, "2. SCORING RESPONSE", color=INK, fontsize=9, weight="bold", ha="center")
    save(fig, directory, "pipeline")


def representative_call(calls):
    """Select a median-nearest call deterministically, without selecting on range."""
    selected = [row for row in calls["Q:vision:insertion"] if row["paired_status"] == "defined"]
    median = float(np.median([float(row["delta"]) for row in selected]))
    return min(selected, key=lambda row: (abs(float(row["delta"])-median), row["record_id"]))


def response_mechanism(directory, report, calls):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), layout="constrained")
    u = np.linspace(0, 4, 501)
    response = normalized_responses(u, 1, epsilon=1e-12)
    ax = axes[0]
    ax.axvspan(0, 1, facecolor="#edf4ef", zorder=0)
    ax.plot(u, response["quadratic"], color=BLUE, label="Q response")
    ax.plot(u, response["l2"], color=RUST, ls="--", label="N response")
    ax.axvline(1, color="#75848a", lw=.8, ls=":")
    ax.axhline(0, color="#75848a", lw=.6)
    ax.set(xlabel=r"Squared residual ratio, $r/B$", ylabel="Normalized response",
           xlim=(0, 4), ylim=(-3.15, 1.25), title=r"A) $C_Q-C_N$ changes sign")
    typical = representative_call(calls)
    ratios = np.asarray(json.loads(typical["residual_points"])) / float(typical["residual_baseline"])
    ax.plot(ratios, np.full(len(ratios), 1.12), "|", color=INK, ms=7, markeredgewidth=1)
    ax.text(.06, -.65, "In range", fontsize=9, color=GREEN)
    ax.text(1.35, .35, "Overshoot", fontsize=9, color=RUST)
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    example = report["tail_examples"]["Q:vision:insertion"]
    call = next(row for row in calls["Q:vision:insertion"] if row["record_id"] == example["record_id"])
    x = np.array(json.loads(call["grid_percent"])) / 100
    for field, color, label, ls in (("normalized_q", BLUE, "Q response", "-"),
                                     ("normalized_l2", RUST, "N response", "--")):
        ax.plot(x, json.loads(call[field]), marker="o", ms=3.5, color=color, ls=ls, label=label)
    ax.axhline(0, color="#75848a", lw=.6)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(min(json.loads(call["normalized_q"]))*1.5, 1.4)
    ax.axhspan(ax.get_ylim()[0], 0, color="#f6e9e2", alpha=.45, zorder=0)
    ax.text(.55, .52, "Overshoot", transform=ax.transAxes, color=RUST, fontsize=9)
    ax.set(xlabel="Fraction of groups inserted", ylabel="Normalized response (symlog)",
           title="B) One saved insertion curve", xlim=(-.01, 1.01))
    ax.set_yticks([-10000, -100, -1, 0, 1], ["−10,000", "−100", "−1", "0", "1"])
    ax.grid(axis="y", color="#dddddd", lw=.5)
    ax.legend(frameon=False, loc="lower right")
    save(fig, directory, "response_mechanism")


def paired_effects(directory, report, calls):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.65), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1]}, layout="constrained")
    for index, case in enumerate(CASES):
        values = np.array([float(row["delta"]) for row in calls[case] if row["paired_status"] == "defined"])
        # Fixed arithmetic jitter makes display generation deterministic; it has
        # no statistical interpretation and never changes an x coordinate.
        jitter = ((np.arange(len(values)) * 137 % 751) / 750 - .5) * .48
        color = BLUE if case.startswith("Q:") else RUST
        for ax in axes:
            ax.scatter(values, index+jitter, s=3.5, color=color, alpha=.23, linewidths=0)
            ax.plot(np.median(values), index, "D", color="black", ms=4)
        hidden = int(np.sum((values < -.05) | (values > .25)))
        axes[1].text(-.043, index-.22, f"{hidden} left of zoom", ha="left", fontsize=8.5, color="#505c64",
                     bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.6))
    labels = [f"{'Q' if case.startswith('Q:') else 'N'}-ranked: {'vision' if ':vision:' in case else 'language'}\n"
              f"{case.split(':')[2]}" for case in CASES]
    axes[0].set_yticks(range(8), labels)
    axes[0].invert_yaxis()
    axes[0].set_xscale("symlog", linthresh=.25)
    axes[0].set_xlim(-6500, .26)
    axes[0].set_xticks([-1000, -10, 0], ["−1,000", "−10", "0"])
    axes[0].set_title("A) Every saved call")
    axes[1].set_title("B) Central range")
    axes[1].set_xlim(-.05, .25)
    axes[1].set_xticks([0, .1, .2])
    for ax in axes:
        ax.axhline(3.5, color="#768790", lw=.8)
        ax.axvline(0, color="#768790", ls=":", lw=.8)
        ax.grid(axis="x", color="#e5e5e5", lw=.5)
        ax.set_xlabel(r"Within-call $\Delta$ AUC (Q − N)")
    save(fig, directory, "paired_effects")


def analytic_toys(directory):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.1), layout="constrained")
    ax = axes[0]
    t = np.linspace(0, 1, 801)
    residual = (1-t)*(1+t+.8)
    q_weight = residual / 1.8
    l_weight = residual / np.sqrt(residual**2 + 1e-12)
    l_weight /= l_weight[0]
    ax.plot(t, q_weight, color=BLUE, label="Q weight / maximum")
    ax.plot(t, l_weight, color=RUST, ls="--", label="N weight / maximum")
    ax.set(title="A) Target changes path weight", xlabel="Path position", ylabel="Weight / own maximum",
           xlim=(0, 1), ylim=(-.42, 1.15))
    exact = polynomial_rank_reversal()
    ax.text(.02, -.18, "Q: (0.767, 0.853), feature 2 first", fontsize=9, color=BLUE)
    ax.text(.02, -.34, "N: approx. (1.000, 0.800), feature 1 first", fontsize=9, color=RUST)
    ax.legend(loc="lower left", bbox_to_anchor=(0, .28), frameon=False, fontsize=9)
    ax.set_yticks([0, .5, 1])
    assert exact["quadratic_ig_exact"] == ["23/30", "64/75"]
    ax = axes[1]
    # Show the first two periods of the actual K=128 construction. The full
    # function has identical zeros at every m=32,64,128 quadrature node.
    frequency = 128
    t = np.linspace(0, 2/frequency, 401)
    h = .4*np.sin(np.pi*frequency*t)**2
    ax.plot(t, h, color=GREEN, lw=1.6, label="Unseen oscillation")
    nodes = np.arange(3)/frequency
    ax.scatter(nodes, np.zeros(3), color=INK, s=25, zorder=3, label="m = 128 intervals")
    ax.set(title="B) Coarse nodes miss variation", xlabel="Path position (first two periods)",
           ylabel=r"Oscillatory term $h(\alpha)$", ylim=(-.16, .67), xlim=(-.06/frequency, 2.06/frequency))
    ax.set_xticks(nodes, ["0", "1/128", "2/128"])
    ax.set_yticks([0, .2, .4])
    ax.text(0, -.083, "K = 128; coarse (0.2, 0.3); exact (0.3, 0.2)", fontsize=8.4)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(.5, 1.01))
    save(fig, directory, "analytic_toys")


def numerical_diagnostics(directory, report, root):
    projection = read_json(root / NUMERICAL / "inputs/sealed_v6.json")
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.6), layout="constrained")
    colors = {"vision": "#75508a", "language": GREEN, "state": "#59636d"}
    markers = {"vision": "o", "language": "s", "state": "^"}
    for i, target in enumerate(("Q", "L2")):
        for j, metric in enumerate(("IG", "path_gradient", "completeness")):
            ax = axes[i, j]
            displayed_values = []
            for modality, color in colors.items():
                if metric == "completeness":
                    groups = []
                    for cell in projection["rows"]:
                        if cell["status"] == "complete_diagnostics" and cell["target"] == target and cell["modality"] == modality:
                            observations = [row for row in cell["observations"] if row["repeat"] == 0]
                            groups.append([(row["m"], 100*row["relative_residual"]) for row in observations])
                else:
                    by_context = defaultdict(list)
                    for row in report["numerical"]:
                        if row["target"] == target and row["modality"] == modality and row["ranking"] == metric and row["candidate_m"]*2 == row["reference_m"]:
                            by_context[row["context_id"]].append((row["reference_m"], 100*row["coordinate_relative_l1"]))
                    groups = list(by_context.values())
                for number, values in enumerate(groups):
                    values = sorted(values)
                    x, y = zip(*values)
                    displayed_values.extend(y)
                    # Exact zeros remain zero on symlog, never replaced by a
                    # plotting floor or omitted from the display.
                    ax.plot(x, y, color=color, marker=markers[modality], ms=2.5, lw=.9, alpha=.58,
                            label=modality.capitalize() if number == 0 else None)
            ax.axhline(1, color="#333333", ls=":", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("symlog", linthresh=.01)
            budgets = [128, 256, 512, 1024] if metric == "completeness" else [256, 512, 1024]
            ax.set_xticks(budgets, [str(value) for value in budgets])
            ax.set_xlim(118 if metric == "completeness" else 235, 1150)
            top = max(12, max(displayed_values)*1.6)
            ax.set_ylim(0, top)
            ticks = [value for value in [0, .01, .1, 1, 10, 100, 1000] if value <= top]
            ax.set_yticks(ticks, [f"{value:g}" for value in ticks])
            ax.grid(axis="y", color="#e5e5e5", lw=.5)
            ax.set_title(f"{'ABCDEF'[i*3+j]}) {'Q' if target == 'Q' else 'N'}: " +
                         {"IG": "IG coordinates", "path_gradient": "path gradients", "completeness": "completeness"}[metric], fontsize=9.4)
            if i == 1:
                ax.set_xlabel("Intervals m" if metric == "completeness" else "Finer budget 2m", fontsize=9)
            if j == 0:
                ax.set_ylabel(r"Coordinate change $D_m$ (%)", fontsize=9)
            elif j == 2:
                ax.set_ylabel(r"Completeness $E_m$ (%)", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3, loc="outside upper center")
    save(fig, directory, "numerical_diagnostics")


def _number(value, digits=3):
    # Math mode supplies a true minus sign rather than a text hyphen.
    return f"${value:.{digits}f}$"


def paired_table(report):
    out = [r"% Generated descriptive reporting; the original mean-primary protocol is preserved.",
           r"\begingroup\centering\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}",
           r"\textbf{A) Typical change and prevalence}\par\smallskip",
           r"\begin{tabular}{@{}llrrrr@{}}\toprule",
           r"Cohort & Input / curve & Median & 95\% interval & \shortstack{Positive\\share} & \shortstack{Overshoot\\share} \\", r"\midrule"]
    for index, row in enumerate(report["paired"]):
        if index == 4:
            out.append(r"\midrule")
        label = ("Q" if row["ranking"] == "Q" else "N") + "-ranked"
        direction = ("Vision" if row["modality"] == "vision" else "Language") + (" ins." if row["direction"] == "insertion" else " del.")
        out.append(f"{label} & {direction} & {_number(row['median'])} & $[{row['median_ci_low']:.3f}, {row['median_ci_high']:.3f}]$ & "
                   f"{row['positive_percent']:.1f}\\% & {row['overshoot_percent']:.1f}\\% " + r"\\")
    out.extend([r"\bottomrule\end{tabular}", r"\par\medskip\textbf{B) Means and tail contribution}\par\smallskip",
                r"\begin{tabular}{@{}llrrrr@{}}\toprule",
                r"Cohort & Input / curve & Mean & \shortstack{In-range\\contribution} & \shortstack{Overshoot\\contribution} & \shortstack{Same\\sign} \\", r"\midrule"])
    for index, row in enumerate(report["paired"]):
        if index == 4:
            out.append(r"\midrule")
        label = ("Q" if row["ranking"] == "Q" else "N") + "-ranked"
        direction = ("Vision" if row["modality"] == "vision" else "Language") + (" ins." if row["direction"] == "insertion" else " del.")
        fmt = lambda value: _number(value, 3)
        out.append(f"{label} & {direction} & {fmt(row['mean'])} & {_number(row['in_range_contribution'])} & {fmt(row['overshoot_contribution'])} & "
                   f"{row['omission_sign_preserved']}/{row['omissions']} " + r"\\")
    out.extend([r"\bottomrule\end{tabular}", r"\endgroup", ""])
    return "\n".join(out)


def facts(report):
    values = {}
    def add(name, value, digits=0):
        text = f"{value:,.{digits}f}".replace(",", "{,}")
        values[name] = dict(value=value, display=text, digits=digits)
    index = {row["case"]: row for row in report["paired"]}
    rows = report["paired"]
    marginal = {row["result_id"]: row for row in report["marginal_rescoring"]}
    for name, rank, response in (("RescoreQDeletionQ", "Q", "Q"), ("RescoreQDeletionL", "Q", "L2"),
                                 ("RescoreLDeletionL", "L2", "L2"), ("RescoreLDeletionQ", "L2", "Q")):
        add(name, float(marginal[f"rescore/ranking_{rank}/response_{response}/vision_deletion"]["point"]), 3)
    for prefix, field, digits in (("PairedMedian", "median", 3), ("PositiveShare", "positive_percent", 1),
                                  ("OvershootShare", "overshoot_percent", 1)):
        add(prefix + "Min", min(row[field] for row in rows), digits)
        add(prefix + "Max", max(row[field] for row in rows), digits)
    add("MaxPositiveChange", max(row["maximum"] for row in rows), 3)
    add("RescoreCalls", rows[0]["n_raw_calls"])
    add("RescoreEpisodes", rows[0]["n_episodes"])
    add("CallsPerEpisode", rows[0]["calls_per_episode_min"])
    add("BootstrapDraws", rows[0]["bootstrap_draws"])
    add("NegativeMeanCases", sum(row["mean"] < 0 for row in rows))
    add("NegativeMeanIntervalsBelowZero", sum(row["mean"] < 0 and row["mean_ci_high"] < 0 for row in rows))
    add("NegativeMeanIntervalsSpanZero", sum(row["mean"] < 0 and row["mean_ci_low"] <= 0 <= row["mean_ci_high"] for row in rows))
    add("PairedCases", len(rows))
    splits = {row["case"]: row for row in report["call_index_summary"]}
    for rank, prefix in (("Q", "Q"), ("L2", "Norm")):
        for modality, middle in (("vision", "Vision"), ("lang", "Language")):
            for direction, suffix in (("insertion", "Insertion"), ("deletion", "Deletion")):
                row = splits[f"{rank}:{modality}:{direction}"]
                stem = prefix + middle + suffix
                add(stem + "FirstContribution", row["first_mean_contribution"], 2)
                add(stem + "LaterContribution", row["later_mean_contribution"], 2)
                add(stem + "FirstMedian", row["first_median"], 3)
                add(stem + "LaterMedian", row["later_median"], 3)
                add(stem + "LaterMean", row["later_mean"], 3)
                add(stem + "FirstTailShare", 100*row["first_overshoot_contribution_fraction"])
                add(stem + "FirstOvershootShare", row["first_overshoot_percent"], 1)
                add(stem + "LaterOvershootShare", row["later_overshoot_percent"], 1)
    add("FirstCalls", report["call_index_summary"][0]["first_raw_calls"])
    add("LaterCalls", report["call_index_summary"][0]["later_raw_calls"])
    grids = report["grid_diagnostics"]
    if len({row["upper_bound"] for row in grids}) != 1:
        raise ValueError("A single schedule-bound macro requires a common bound")
    add("ScheduleDifferenceBound", grids[0]["upper_bound"], 4)
    add("MaxPositiveBoundPercent", max(row["maximum_percent_of_bound"] for row in grids))
    add("LanguageRepeatedPoints", sum(row["repeated_one_five_residuals"] for row in grids if ":lang:" in row["case"]))
    for rank, prefix in (("Q", "Q"), ("L2", "Norm")):
        gaps = next(row for row in report["baseline_gap_summary"] if row["case"] == rank + ":vision:insertion")
        add(prefix + "VisionBaselineMedian", gaps["residual_baseline_median"], 1)
        add(prefix + "VisionBaselineBelowHalf", gaps["baseline_below_half"])
        add(prefix + "VisionFirstBaselineBelowHalf", gaps["first_baseline_below_half"])
    for rank, prefix in (("Q", "QVisionInsertion"), ("L2", "NormVisionInsertion")):
        row = index[rank + ":vision:insertion"]
        for suffix, field, digits in (("Mean", "mean", 2), ("Median", "median", 3), ("Min", "minimum", 1),
                                       ("InRange", "in_range_contribution", 3), ("Overshoot", "overshoot_contribution", 2),
                                       ("MeanLow", "mean_ci_low", 2), ("MeanHigh", "mean_ci_high", 2),
                                       ("OvershootCalls", "overshoot_calls", 0), ("OvershootShare", "overshoot_percent", 1)):
            add(prefix + suffix, row[field], digits)
    for rank, prefix in (("Q", "Q"), ("L2", "Norm")):
        for modality, middle in (("vision", "Vision"), ("lang", "Language")):
            for direction, suffix in (("insertion", "Insertion"), ("deletion", "Deletion")):
                row = index[f"{rank}:{modality}:{direction}"]
                for metric, field, digits in (("Median", "median", 3), ("PositiveShare", "positive_percent", 1),
                                              ("OvershootShare", "overshoot_percent", 1), ("SignPreserved", "omission_sign_preserved", 0)):
                    add(prefix+middle+suffix+metric, row[field], digits)
    numerical = report["numerical"]
    for target, prefix in (("Q", "Q"), ("L2", "N")):
        state = [row for row in numerical if row["target"] == target and row["modality"] == "state" and row["ranking"] == "IG"
                 and row["reference_m"] == 1024 and row["candidate_m"] == 512]
        add("NumericalState"+prefix+"Coord", 100*max(row["coordinate_relative_l1"] for row in state), 1)
        add("NumericalState"+prefix+"Residual", 100*max(row["reference_completeness"] for row in state), 1)
        vision = next(row for row in numerical if row["target"] == target and row["modality"] == "vision"
                      and row["ranking"] == "IG" and row["stratum"] == "picksingleycb-170m"
                      and row["episode"] == 1 and row["call"] == 0
                      and row["candidate_m"] == 256 and row["reference_m"] == 512)
        add("NumericalVision"+prefix+"Coord", 100*vision["coordinate_relative_l1"], 2)
        add("NumericalVision"+prefix+"Residual", 100*vision["reference_completeness"], 3)
    coverage = report["numerical_coverage"]
    add("NumericalPlanned", coverage["planned"]); add("NumericalComplete", coverage["complete"])
    add("NumericalEpisodes", coverage["episodes"])
    add("NumericalPlannedOneB", coverage["planned_one_b"])
    add("NumericalUnreached", coverage["status_counts"]["call_not_reached"])
    add("NumericalUnfinished", coverage["status_counts"]["incomplete"])
    add("NumericalDependentViolations", coverage["criterion_status_counts"]["violates"])
    add("NumericalRepeatChecks", report["numerical_original_summary"]["repeated_condition_and_coverage_check_counts"]["satisfies"])
    add("NumericalEqualityChecks", report["numerical_original_summary"]["repeatability_equality_check_counts"]["satisfies"])
    for source, prefix in (("Q-ranking regeneration", "HeadlineQ"), ("L-ranking regeneration", "HeadlineL"),
                           ("Verification", "Verification"), ("Historical 1B", "OneB")):
        selected = [row for row in report["task_success"] if row["source"] == source]
        add(prefix+"Episodes", sum(row["episodes"] for row in selected))
        add(prefix+"Successes", sum(row["successes"] for row in selected))
    for name, key in (("ArchiveFiles", "inputs"), ("ArchiveValidRecords", "valid_records"),
                      ("ArchiveDuplicateGroups", "duplicate_groups"), ("ArchiveConflictingGroups", "conflicting_groups"),
                      ("ArchiveQuarantined", "quarantined_lines")):
        add(name, report["archive"][key])
    add("RandomGridAUC", report["random_reference"]["quadratic_grid_insertion"], 5)
    example = report["tail_examples"]["Q:vision:insertion"]
    for name, value, digits in (("TailBaselineSquared", example["residual_baseline"], 4),
                                ("TailMaximumSquared", max(example["residual_points"]), 0),
                                ("TailDiscrepancyRatio", max(example["residual_points"])/example["residual_baseline"], 0),
                                ("TailQAUC", example["auc_q"], 1), ("TailNormAUC", example["auc_l2"], 2)):
        add(name, value, digits)
    for prefix, field in (("FullyInRangeShare", "fully_in_range_percent"),
                          ("OvershootPositiveShare", "overshoot_positive_percent")):
        add(prefix + "Min", min(row[field] for row in rows), 1)
        add(prefix + "Max", max(row[field] for row in rows), 1)
    qvision = index["Q:vision:insertion"]
    add("QVisionInsertionInRangeQuartileLow", qvision["fully_in_range_delta_q25"], 3)
    add("QVisionInsertionInRangeQuartileHigh", qvision["fully_in_range_delta_q75"], 3)
    relationship = report["cohort_relationship"]
    add("CohortAlignedCalls", relationship["aligned_calls"])
    add("CohortUnequalBaselines", relationship["unequal_baseline_calls"])
    add("SharedExtremeEpisodes", len(relationship["shared_extreme_episodes"]))
    add("CohortInitialCorrelation", relationship["call_index_correlations"][0]["log_baseline_correlation"], 3)
    add("CohortFinalCorrelation", relationship["call_index_correlations"][-1]["log_baseline_correlation"], 2)
    sensitivity = report["language_schedule_sensitivity"]
    add("LanguageMedianSensitivity", max(abs(row["median_change"]) for row in sensitivity), 3)
    add("LanguageInRangeBound", max(row["in_range_per_call_bound"] for row in sensitivity), 3)
    for modality, stem in (("vision", "Vision"), ("state", "State"), ("language", "Language")):
        for target, label in (("Q", "Q"), ("L2", "N")):
            row = next(row for row in report["numerical_violation_summary"]["finest_pair_by_modality_target"]
                       if row["modality"] == modality and row["target"] == target)
            add("Finest" + stem + label + "Violations", row["arms_with_finest_pair_violation"])
    for name, criterion, direction, response in (
        ("YCBStateCoarseRankAgreement", "group_spearman", None, None),
        ("YCBStateCoarseInsertionChange", "normalized_auc_difference", "insertion", "Q"),
        ("YCBStateCoarseDeletionChange", "normalized_auc_difference", "deletion", "Q")):
        row = next(row for row in report["numerical_evaluation_checks"]
                   if row["stratum"] == "picksingleycb-170m" and row["episode"] == 1 and row["call"] == 0
                   and row["modality"] == "state" and row["target"] == "Q" and row["ranking"] == "IG"
                   and row["candidate_m"] == 256 and row["reference_m"] == 512
                   and row["criterion"] == criterion and row["direction"] == direction and row["response"] == response)
        add(name, row["value"], 2)
        values[name]["report_sha256"] = row["report_sha256"]
    later = next(row for row in numerical if row["stratum"] == "picksingleycb-170m" and row["episode"] == 1
                 and row["call"] == 12 and row["target"] == "Q" and row["modality"] == "state"
                 and row["ranking"] == "IG" and row["candidate_m"] == 512 and row["reference_m"] == 1024)
    add("YCBStateLaterCoordinateChange", 100*later["coordinate_relative_l1"], 1)
    pickcube = next(row for row in numerical if row["stratum"] == "pickcube-170m" and row["episode"] == 1
                    and row["call"] == 0 and row["target"] == "L2" and row["modality"] == "vision"
                    and row["ranking"] == "IG" and row["candidate_m"] == 256 and row["reference_m"] == 512)
    add("PickCubeVisionNCoordinateChange", 100*pickcube["coordinate_relative_l1"], 2)
    pickcube_auc = next(row for row in report["numerical_evaluation_checks"] if row["stratum"] == "pickcube-170m"
                        and row["episode"] == 1 and row["call"] == 0 and row["target"] == "L2"
                        and row["modality"] == "vision" and row["ranking"] == "IG"
                        and row["candidate_m"] == 256 and row["reference_m"] == 512
                        and row["criterion"] == "normalized_auc_difference"
                        and row["direction"] == "insertion" and row["response"] == "Q")
    add("PickCubeVisionNInsertionChange", pickcube_auc["value"], 3)
    values["PickCubeVisionNInsertionChange"]["report_sha256"] = pickcube_auc["report_sha256"]
    # Prose should not imply precision unsupported by its rounded input values.
    ratio = values["TailDiscrepancyRatio"]["value"]
    values["TailDiscrepancyRatio"]["display"] = f"{round(ratio, -2):,.0f}".replace(",", "{,}")
    values["TailDiscrepancyRatio"]["rounding"] = "nearest hundred"
    return values


def build_figures(root, report, calls):
    style()
    directory = root / "paper/figures_revision"
    directory.mkdir(parents=True, exist_ok=True)
    pipeline(directory)
    response_mechanism(directory, report, calls)
    paired_effects(directory, report, calls)
    analytic_toys(directory)
    numerical_diagnostics(directory, report, root)
    tables = root / "paper/tables_revision"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "paired_results.tex").write_text(paired_table(report), encoding="utf-8", newline="\n")
    macros = facts(report)
    (tables / "revision_facts.tex").write_text("% Generated from authenticated evidence by analysis.manuscript_revision.\n" +
        "\n".join("\\newcommand{\\" + name + "}{" + value["display"] + "}" for name, value in macros.items()) + "\n",
        encoding="utf-8", newline="\n")
    return macros
