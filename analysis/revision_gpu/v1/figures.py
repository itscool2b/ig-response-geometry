"""Figure files for the R1 results (PDF and 300 dpi PNG); captions are the author's.

Every figure is drawn from a CSV or JSON file written by analyze.py, which
serves as its table view. Colour roles follow the paper (blue for Q, orange for
N) using the validated reference palette: slots blue #2a78d6, orange #eb6834,
aqua #1baf7a (adjacent pairs pass the CVD and normal-vision checks). Every
figure with more than one series carries a legend; the aqua slot has low
contrast on white, so its identity never rests on colour alone. Axis and legend
strings are plain labels for the author to replace.

Usage: python -m analysis.revision_gpu.v1.figures --analysis DIR --out DIR
"""
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dddddd"
RESPONSE_COLOR = {"Q": BLUE, "L2": ORANGE, "N": ORANGE}
RULE_COLOR = {"T": BLUE, "E": ORANGE, "M": AQUA}


def style():
    matplotlib.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": MUTED,
                         "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42,
                         "svg.hashsalt": "nhsjs-r1", "axes.spines.top": False, "axes.spines.right": False})


def read_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if value in ("", "None"):
                row[key] = None
                continue
            if value in ("True", "False"):
                row[key] = value == "True"
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for extension, kwargs in (("pdf", dict(metadata={"CreationDate": None, "ModDate": None})), ("png", dict(dpi=300))):
        path = out / f"{name}.{extension}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plt.close(fig)
    return paths


def panel_population(rows, modality, target, field, min_budget):
    """The plotted points of one budget-figure panel and the contexts every point is drawn over.

    points maps (arm, rule, budget) to {context: value}, with ("A0", None, 64) for the historical star;
    shared is the sorted contexts that have a finite value at every one of those points.
    """
    points = defaultdict(dict)
    for r in rows:
        if not (r["modality"] == modality and r["target"] == target and isinstance(r.get(field), float)
                and math.isfinite(r[field])):
            continue
        if r["arm"] in ("A1", "A2") and r["rule"] in RULE_COLOR and r["budget"] and r["budget"] >= min_budget:
            points[(r["arm"], r["rule"], r["budget"])][r["context"]] = r[field]
        elif r["arm"] == "A0" and r["budget"] == 64:
            points[("A0", None, 64)][r["context"]] = r[field]
    shared = sorted(set.intersection(*(set(v) for v in points.values()))) if points else []
    return points, shared


def budget_figure(rows, bank, out, name, field, ylabel, min_budget=16):
    """Median of a per-context quantity against budget, per modality (rows) and target (columns).

    Colour identifies the rule (trapezoid, endpoint-inclusive average, midpoint); line style the
    forward precision (solid fp32, dashed bf16); the star is the historical bf16 kernel at m = 64.
    When the bank has fp32 (A2) rows, only those contexts are drawn. Within a panel every point is a
    median over the same contexts: those with a value at every plotted point of the panel (a rule
    compared on the pod only against its own reference can lack some). Panel titles give the count.
    """
    rows = [r for r in rows if r["bank"] == bank]
    fp32_contexts = {r["context"] for r in rows if r["arm"] == "A2"}
    if fp32_contexts:
        rows = [r for r in rows if r["context"] in fp32_contexts]
    modalities = [m for m in ("vision", "language", "state") if any(r["modality"] == m for r in rows)]
    drawn_rules, drawn_arms, drawn_historical, drawn_sparse = set(), set(), False, False
    fig, axes = plt.subplots(len(modalities), 2, figsize=(7.0, 2.0 * len(modalities) + 0.6), squeeze=False, sharex=True)
    for i, modality in enumerate(modalities):
        for j, target in enumerate(("Q", "N")):
            ax = axes[i][j]
            points, shared = panel_population(rows, modality, target, field, min_budget)
            ax.set_title(f"{modality}, target {target} ({len(shared)} contexts)", loc="left")
            if shared:
                for arm, linestyle in (("A2", "-"), ("A1", "--")):
                    for rule in ("T", "E", "M"):
                        xs = sorted(b for a, ru, b in points if a == arm and ru == rule)
                        if not xs:
                            continue
                        ys = [float(np.median([max(points[(arm, rule, x)][c], 1e-12) for c in shared])) for x in xs]
                        # A series kept at fewer than three budgets is drawn as points: a line would imply values between them.
                        sparse = len(xs) < 3
                        ax.plot(xs, ys, linestyle="none" if sparse else linestyle, color=RULE_COLOR[rule], lw=1.6,
                                marker="o", ms=4 if sparse else 2.5,
                                mfc="white" if sparse and arm == "A1" else RULE_COLOR[rule])
                        drawn_sparse = drawn_sparse or sparse
                        drawn_rules.add(rule)
                        drawn_arms.add(arm)
                if ("A0", None, 64) in points:
                    star = float(np.median([points[("A0", None, 64)][c] for c in shared]))
                    ax.plot([64], [max(star, 1e-12)], marker="*", ms=9, color=INK, ls="none")
                    drawn_historical = True
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="major", color=GRID, lw=0.5)
            if i == len(modalities) - 1:
                ax.set_xlabel("integration budget m (intervals)")
    rule_names = {"T": "trapezoid T_m", "E": "endpoint-inclusive E_m", "M": "midpoint M_n"}
    handles = [Line2D([], [], color=RULE_COLOR[r], lw=1.6, label=rule_names[r]) for r in ("T", "E", "M") if r in drawn_rules]
    if len(drawn_arms) > 1:
        handles += [Line2D([], [], color=MUTED, lw=1.6, ls="-", label="fp32 forward (A2)"),
                    Line2D([], [], color=MUTED, lw=1.6, ls="--", label="bf16 forward (A1)")]
    if drawn_historical:
        handles.append(Line2D([], [], color=INK, marker="*", ms=8, ls="none", label="historical bf16 kernel, m=64 (A0)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.supylabel(ylabel, fontsize=9)
    note = "each panel: the contexts with every plotted point" + (", all with an fp32 phase" if fp32_contexts else "")
    if drawn_sparse:
        note += "; points without a line: maps kept at those budgets only (open points: bf16)"
    fig.text(0.99, 0.005, note, ha="right", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    return save(fig, out, name)


def ranking_color(name):
    if name.startswith(("IG_", "PG_")):
        return RESPONSE_COLOR["N" if "_N_" in name else "Q"]
    return MUTED


def advantage_figure(decisions, out, name, modality="vision"):
    """Episode-mean calibrated advantage with its interval, per ranking, for the four response-direction cells.

    Each panel names its complete-case episode count; a cell without an estimate says so. Returns None,
    and writes nothing, when no cell has an estimate.
    """
    cells = decisions[modality]["cells"]
    keys = ("Q/deletion", "Q/insertion", "L2/deletion", "L2/insertion")
    if not any("mean" in cells[key]["with_fixed_orders"] for key in keys):
        return None
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharey=True)
    for k, key in enumerate(keys):
        ax = axes[k // 2][k % 2]
        summary = cells[key]["with_fixed_orders"]
        response, direction = key.split("/")
        episodes = summary.get("episodes")
        ax.set_title(f"response {'N' if response == 'L2' else 'Q'}, {direction} ({episodes} episodes)", loc="left")
        if "mean" not in summary:
            ax.text(0.5, 0.5, f"{episodes} complete-case episode{'' if episodes == 1 else 's'}: no estimate",
                    ha="center", va="center", transform=ax.transAxes, color=MUTED, fontsize=8)
            ax.set_xticks([])
            continue
        names = summary["names"]
        ys = np.arange(len(names))
        for y, rank_name in zip(ys, names):
            low, high = summary["interval"][rank_name]["interval"]
            color = ranking_color(rank_name)
            ax.plot([low, high], [y, y], color=color, lw=1.6, solid_capstyle="round")
            ax.plot(summary["mean"][rank_name], y, "o", ms=4.5, color=color)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(ys, names)
        ax.grid(True, axis="x", color=GRID, lw=0.5)
    axes[0][0].invert_yaxis()
    handles = [Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=4.5, label="ranking from a Q-target map"),
               Line2D([], [], color=ORANGE, lw=1.6, marker="o", ms=4.5, label="ranking from an N-target map"),
               Line2D([], [], color=MUTED, lw=1.6, marker="o", ms=4.5, label="input difference or fixed order")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("calibrated advantage over random order (episode mean)\n"
                  "intervals: 95% percentile, episode bootstrap", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, out, name)


def symlog_ticks(low, high, linthresh):
    """0, the linear-range edges and 0.1, then every power of ten out to the data on each side."""
    ticks = {0.0}
    for sign, bound in ((1.0, high), (-1.0, low)):
        if sign * bound <= 0:
            continue
        ticks.update({sign * 0.1, sign * linthresh})
        power = 1.0
        while power <= abs(bound):
            ticks.add(sign * power)
            power *= 10.0
    return sorted(ticks)


def shift_figure(groups, out, name, ranking, direction, linthresh=0.25):
    """Per-call response-only shift Delta = AUC_Q - AUC_N for one ranking, by bank and outcome (symlog axis)."""
    fig, ax = plt.subplots(figsize=(7.0, 0.55 * len(groups) + 1.3))
    rng = np.random.default_rng(0)
    finite = []
    for y, (label, values) in enumerate(groups):
        values = np.array([v for v in values if v is not None and math.isfinite(v)], dtype=float)
        if not len(values):
            continue
        finite.extend(values.tolist())
        ax.scatter(values, y + rng.uniform(-0.2, 0.2, len(values)), s=6, color=MUTED, alpha=0.45, linewidths=0)
        ax.plot(np.median(values), y, "D", ms=6, color=INK)
    low, high = min(finite + [-linthresh]), max(finite + [linthresh])
    ax.set_xscale("symlog", linthresh=linthresh)
    ticks = symlog_ticks(low, high, linthresh)
    ax.set_xticks(ticks, [f"{t:g}".replace("-", "\u2212") for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(low * 1.3 if low < -linthresh else -linthresh * 1.3, high * 1.3 if high > linthresh else linthresh * 1.3)
    ax.set_ylim(len(groups) - 0.5, -0.5)
    ax.set_yticks(range(len(groups)), [label for label, _ in groups])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(True, axis="x", color=GRID, lw=0.5)
    ax.plot([], [], "D", ms=6, color=INK, label="median")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax.set_xlabel(f"Delta = AUC_Q - AUC_N, ranking {ranking}, {direction} (symlog, linear within ±{linthresh:g})")
    fig.tight_layout()
    return save(fig, out, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ranking", default="IG_Q_A1T512")
    parser.add_argument("--min-budget", type=int, default=16)
    args = parser.parse_args()
    style()
    analysis = Path(args.analysis)
    manifest = json.loads((analysis / "manifest.json").read_text())
    hashes, skipped = {}, {}
    residuals = read_csv(analysis / "numerics/residuals.csv")
    references = read_csv(analysis / "numerics/references.csv") if (analysis / "numerics/references.csv").exists() else []
    against_t512 = []
    for row in references:
        candidate = row.get("candidate")
        if re.fullmatch(r"(fp32|bf16)_vs_A2T\d+", str(row["family"])) and isinstance(candidate, str) and candidate[:2] in ("A0", "A1", "A2"):
            rule = candidate[2:]
            family = "".join(c for c in rule if c.isalpha())
            digits = "".join(c for c in rule if c.isdigit())
            if digits and family in ("T", "E", "M", "m"):
                against_t512.append(dict(row, arm=candidate[:2], rule=family, budget=float(digits)))
    for bank in sorted({r["bank"] for r in residuals}):
        hashes.update(budget_figure(residuals, bank, args.out, f"residual_by_budget_{bank}", "relative_residual",
                                    "median relative completeness residual", args.min_budget))
        rows = [r for r in against_t512 if r["bank"] == bank]
        if rows:
            hashes.update(budget_figure(rows, bank, args.out, f"coordinate_error_by_budget_{bank}", "relative_l1",
                                        "median relative L1 to the fp32 trapezoid map at m = 512", args.min_budget))
    for bank in sorted(p.name for p in (analysis / "methods").iterdir()) if (analysis / "methods").exists() else []:
        decisions = json.loads((analysis / "methods" / bank / "decisions.json").read_text())
        if "vision" in decisions:
            drawn = advantage_figure(decisions, args.out, f"calibrated_advantage_{bank}_vision")
            if drawn is None:
                skipped[f"calibrated_advantage_{bank}_vision"] = "no complete-case episode with an estimate in any cell"
            else:
                hashes.update(drawn)
    groups = []
    for bank in sorted(p.name for p in (analysis / "shifts").iterdir()) if (analysis / "shifts").exists() else []:
        path = analysis / "shifts" / bank / "rows_vision.csv"
        if not path.exists():
            continue
        rows = [r for r in read_csv(path) if r["ranking"] == args.ranking and r["direction"] == "insertion"]
        for outcome, label in ((True, "success"), (False, "failure"), (None, "outcome not given")):
            values = [r["delta"] for r in rows if r.get("success") == outcome]
            if values:
                groups.append((f"{bank}, {label} (n={len(values)})", values))
    if groups:
        hashes.update(shift_figure(groups, args.out, "response_shift_vision_insertion", args.ranking, "insertion"))
    (Path(args.out) / "figures.json").write_text(json.dumps(dict(analysis_manifest=manifest.get("outputs", {}),
                                                                files=hashes, skipped=skipped),
                                                           indent=1, sort_keys=True) + "\n")
    print(json.dumps(sorted(hashes), indent=1))


if __name__ == "__main__":
    main()
