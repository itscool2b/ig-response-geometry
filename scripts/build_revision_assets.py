"""Generate manuscript tables/figures only from the verified v2 analysis.

The lineage registry binds every displayed estimate, interval and population
count to a named result/occurrence view and its physical-record membership.
Historical paper/figures and paper/paper.pdf are never written.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.revision.verify import verify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical_case_assets(tables):
    """Reproduce the frozen partial case before including its two displays."""
    directory = ROOT / "analysis/numerical_case/2026-10-01-v1"
    script = directory / "summarize.py"
    source = directory / "inputs/sealed_v6.json"
    manifest_path = directory / "outputs/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha(source) != manifest["input_sha256"] or sha(script) != manifest["source_sha256"]:
        raise ValueError("Numerical case input or reproducer differs from the recorded manifest")
    with tempfile.TemporaryDirectory(prefix="tmlr-numerical-display-") as scratch:
        output = Path(scratch)
        subprocess.run([sys.executable, str(script), "--input", str(source), "--output", str(output)], check=True)
        for name, expected in manifest["outputs"].items():
            if Path(name).name != name or sha(directory / "outputs" / name) != expected or sha(output / name) != expected:
                raise ValueError("Numerical case output does not reproduce: " + name)
        for name in ("numerical_roster.tex", "numerical_diagnostics.tex"):
            if name not in manifest["outputs"]:
                raise ValueError("Numerical case table lacks a manifest entry: " + name)
            (tables / name).write_bytes((output / name).read_bytes())
    inputs = [script, source, manifest_path, *(directory / "outputs" / name for name in manifest["outputs"])]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in inputs}


def strengthening_assets(tables):
    """Bind the additional CPU evidence without altering historical artifacts."""
    from analysis.revision.nested_grid_aliasing import run_example
    report = ROOT / "analysis/revision/examples/nested_grid_aliasing.json"
    expected = json.loads(report.read_text(encoding="utf-8"))
    if run_example() != expected:
        raise ValueError("Nested-grid example differs from the checked CPU report")
    paths = [report, ROOT / "analysis/revision/nested_grid_aliasing.py",
             ROOT / "integrated_gradients.py", ROOT / "paper/appendix_aliasing.tex"]
    from analysis.paired_rescoring.analyze import run as paired_run
    directory = ROOT / "analysis/paired_rescoring"
    results = directory / "results/2026-10-01-v1"
    provenance_path = results / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for name, expected_digest in provenance["source_sha256"].items():
        if Path(name).name != name or sha(directory / name) != expected_digest:
            raise ValueError("Paired rescoring source changed: " + name)
    if sha(directory / "protocol.json") != provenance["protocol_sha256"]:
        raise ValueError("Paired rescoring protocol changed")
    with tempfile.TemporaryDirectory(prefix="tmlr-paired-display-") as scratch:
        reproduced = Path(scratch) / "results"
        paired_run(reproduced, root=ROOT)
        for name, expected_digest in provenance["artifacts_sha256"].items():
            if (Path(name).name != name or sha(results / name) != expected_digest
                    or sha(reproduced / name) != expected_digest):
                raise ValueError("Paired rescoring output does not reproduce: " + name)
        with (reproduced / "summary.csv").open(encoding="utf-8", newline="") as stream:
            paired_rows = list(csv.DictReader(stream))
        # Show the prespecified median sensitivity beside the primary mean so
        # the tail-driven sign differences remain visible in the main table.
        table = [r"\begin{tabular}{lllrrr}", r"\toprule",
                 r"Ranking & Modality & Curve & Mean $\Delta$ & 95\% interval & Median $\Delta$ \\",
                 r"\midrule"]
        for row in paired_rows:
            mean = float(row["equal_episode_mean_point"])
            lo, hi = (float(row["equal_episode_mean_" + key]) for key in ("ci_lo", "ci_hi"))
            median = float(row["paired_call_median_point"])
            table.append(f"{row['ranking']} & {row['modality']} & {row['direction']} & "
                         f"{mean:.4f} & [{lo:.4f}, {hi:.4f}] & {median:.4f} " + r"\\")
        table.extend([r"\bottomrule", r"\end{tabular}"])
        (tables / "paired_rescoring.tex").write_text("\n".join(table) + "\n", encoding="utf-8", newline="\n")
    paths.extend([provenance_path, directory / "protocol.json",
                  *(directory / name for name in provenance["source_sha256"]),
                  *(results / name for name in provenance["artifacts_sha256"])])
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def influence_assets(tables):
    """Independently regenerate the explicitly exploratory appendix artifact."""
    from analysis.paired_rescoring.influence import run, verify as verify_influence
    directory = ROOT / "analysis/paired_rescoring/influence_results/2026-10-01-v1"
    verify_influence(directory, root=ROOT)
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="tmlr-influence-display-") as scratch:
        reproduced = Path(scratch) / "results"
        run(reproduced, root=ROOT)
        for name, expected in provenance["artifacts_sha256"].items():
            if Path(name).name != name or sha(reproduced / name) != expected:
                raise ValueError("Episode influence output does not reproduce: " + name)
        (tables / "episode_influence.tex").write_bytes((reproduced / "influence_table.tex").read_bytes())
    paths = [provenance_path, ROOT / "analysis/paired_rescoring/influence_protocol.json",
             *(ROOT / name for name in provenance["source_sha256"]),
             *(directory / name for name in provenance["artifacts_sha256"])]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    source = ROOT / "analysis/revision/results/2026-09-30-v2"
    verification = verify(source)
    rows = list(csv.DictReader((source / "results.csv").open(encoding="utf-8", newline="")))
    index = {(row["result_id"], row["occurrence_view"]): row for row in rows}
    figures = ROOT / "paper/figures_revision"
    tables = ROOT / "paper/tables_revision"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    numerical_inputs = numerical_case_assets(tables)
    strengthening_inputs = strengthening_assets(tables)
    influence_inputs = influence_assets(tables)
    lineage, macros, macro_displays = [], [], {}
    influence_results = ROOT / "analysis/paired_rescoring/influence_results/2026-10-01-v1"
    with (influence_results / "case_summary.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            lineage.append({"location": "table:episode-influence",
                            "source": (influence_results / "case_summary.csv").relative_to(ROOT).as_posix(),
                            "result": row, "interpretation": "exploratory_omission_range_not_confidence_interval",
                            "membership_source": "analysis/paired_rescoring/results/2026-10-01-v1/membership.json.gz",
                            "protocol_source": "analysis/paired_rescoring/influence_protocol.json"})
    tails = json.loads((influence_results / "tail_diagnostics.json").read_text(encoding="utf-8"))
    lineage.append({"location": "text:episode-influence-normalization-example",
                    "source": (influence_results / "tail_diagnostics.json").relative_to(ROOT).as_posix(),
                    "case": "Q:vision:insertion", "result": tails["Q:vision:insertion"]["example"],
                    "interpretation": "within_cohort_finite_endpoint_normalization_not_physical_importance"})
    paired_results = ROOT / "analysis/paired_rescoring/results/2026-10-01-v1"
    for filename, location in (("summary.csv", "table:paired-rescoring"),
                               ("geometry.csv", "text:paired-rescoring-geometry")):
        with (paired_results / filename).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                lineage.append({"location": location,
                                "source": (paired_results / filename).relative_to(ROOT).as_posix(),
                                "result": row,
                                "membership_source": "analysis/paired_rescoring/results/2026-10-01-v1/membership.json.gz",
                                "protocol_source": "analysis/paired_rescoring/protocol.json"})

    def result(key, location, view="retrospective_last"):
        row = index[key, view]
        lineage.append({"location": location, "source": "results.csv", "result": row})
        return row

    def estimate(key, location, scale=1, digits=3, view="retrospective_last", interval=True):
        row = result(key, location, view)
        point = float(row["point"]) * scale
        output = f"{point:.{digits}f}"
        if interval:
            if not row["ci_lo"] or not row["ci_hi"]:
                raise ValueError("Requested interval is absent: " + key)
            lo, hi = float(row["ci_lo"]) * scale, float(row["ci_hi"]) * scale
            output += f" [{lo:.{digits}f}, {hi:.{digits}f}]"
        lineage[-1].update(display=output, scale=scale, digits=digits, interval_displayed=interval)
        return output

    def counts(key, location, view="retrospective_last"):
        row = result(key, location, view)
        output = f"{int(row['n_source_records']):,} & {int(row['n_groups']):,}"
        lineage[-1].update(display=output, fields=["n_source_records", "n_groups"])
        return output

    def macro(name, key, digits=4, field="point", scale=1):
        row = result(key, "macro:" + name)
        value = float(row[field]) * scale
        displayed = f"{value:.{digits}f}" if digits else f"{int(value):,}".replace(",", "{,}")
        macros.append("\\newcommand{\\" + name + "}{" + displayed + "}")
        macro_displays[name] = displayed
        lineage[-1].update(display=displayed, field=field, scale=scale)

    def metadata_macro(name, filename, pointer, value, digits=0):
        display = f"{value:.{digits}f}" if digits else f"{int(value):,}".replace(",", "{,}")
        macros.append("\\newcommand{\\" + name + "}{" + display + "}")
        macro_displays[name] = display
        lineage.append({"location": "macro:" + name, "source": filename, "pointer": pointer, "display": display, "value": value})

    def table(name, columns, headers, body, caption, label):
        text = "\n".join([r"% Generated by scripts/build_revision_assets.py; do not hand-edit.",
            r"\begin{table}[htbp]", r"\centering\small", r"\setlength{\tabcolsep}{3.5pt}",
            r"\caption{" + caption + "}", r"\label{" + label + "}",
            r"\begin{tabular}{" + columns + "}", r"\toprule", headers + r" \\", r"\midrule",
            *[row + r" \\" for row in body], r"\bottomrule", r"\end{tabular}",
            r"\end{table}", ""])
        (tables / (name + ".tex")).write_text(text, encoding="utf-8")

    tasks = [("PegInsertionSide", "PegInsertion"), ("PickCube", "PickCube"),
             ("PickSingleYCB", "PickYCB"), ("StackCube", "StackCube")]
    body = []
    for task, display in tasks:
        base = f"verification/{task}/all/"
        body.append(display + " & " + counts(base + "vision_err", "table:completeness:counts:" + task) + " & " + " & ".join(
            estimate(base + modality + "_err", f"table:completeness:{task}:{modality}", scale=100, digits=2)
            for modality in ("vision", "lang", "state")))
    table("completeness", "lrrlll", r"Task & Calls & Episodes & Vision (\%) & Language (\%) & State (\%)", body,
        r"Month 7 verification campaign, all valid calls, $m=64$. Cells are median relative completeness residuals in percent with conditional 95\% episode-bootstrap intervals. Calls and episodes are the identical point/interval populations. These are historical numerical diagnostics, not coordinate-accuracy certifications.", "tab:correctness")

    body = []
    for task, display in tasks:
        for modality, modlabel in (("vision", "Vision"), ("lang", "Language")):
            base = f"verification/{task}/all/{modality}"
            body.append(display + " & " + modlabel + " & " + counts(base + "_insertion_auc", f"table:faithfulness:{task}:{modality}:counts") + " & " +
                estimate(base + "_insertion_auc", f"table:faithfulness:{task}:{modality}:insertion") + " & " +
                estimate(base + "_deletion_auc", f"table:faithfulness:{task}:{modality}:deletion"))
    table("faithfulness", "llrrll", r"Task & Modality & Calls & Episodes & Insertion AUC & Deletion AUC", body,
        r"Month 7 verification campaign, all valid calls and the native quadratic response. Medians and conditional 95\% intervals share one population. No entry is labeled a success against an uncalibrated chance threshold. The recorded token-fraction grids, endpoint conventions and provenance limits remain part of these descriptive estimates.", "tab:faithfulness")

    body = []
    for rank in ("Q", "L2"):
        for response in ("Q", "L2"):
            for modality, label in (("vision", "Vision"), ("lang", "Language")):
                base = f"rescore/ranking_{rank}/response_{response}/{modality}_"
                rank_label = "$Q$" if rank == "Q" else "$L$"
                response_label = "$Q$" if response == "Q" else "$L$"
                body.append(rank_label + " & " + response_label + " & " + label + " & " +
                    estimate(base + "insertion", f"table:rescore:{rank}:{response}:{modality}:insertion") + " & " +
                    estimate(base + "deletion", f"table:rescore:{rank}:{response}:{modality}:deletion"))
    table("rescore", "lllll", "Ranking target & Response & Modality & Insertion AUC & Deletion AUC", body,
        r"Within-ranking response transformation, with median AUC and conditional 95\% episode-bootstrap intervals. Each ranking cohort contains \RescoreCalls{} calls from \RescoreEpisodes{} episodes. The same recorded interventions are reused within a ranking cohort. The two ranking cohorts are not authenticated paired contexts; comparisons between them do not identify ranking improvement.", "tab:targets")

    body = []
    for task, display in tasks:
        for check in ("C1", "C2"):
            base = f"verification/{task}/{check}/spearman_"
            body.append(display + " & " + check + " & " + counts(base + "vision", f"table:sanity:{task}:{check}:counts") + " & " + " & ".join(
                estimate(base + modality, f"table:sanity:{task}:{check}:{modality}") for modality in ("vision", "language")))
    table("sanity", "llrrll", r"Task & Check & Calls & Episodes & Vision $\rho$ & Language $\rho$", body,
        r"Month 7 model randomization (C1) and input shuffle (C2), all recorded calls. Conditional 95\% episode intervals accompany median Spearman correlation. These diagnostics test different dependencies: low C2 correlation does not establish learned-model dependence, and C1 changes integration budget as well as weights.", "tab:sanity")

    body = []
    for weighting, label in (("call_weighted", "Median over calls"), ("episode_equal", "Median of episode medians")):
        for modality, modlabel in (("vision", "Vision"), ("lang", "Language")):
            base = f"oneb/{weighting}/{modality}_"
            body.append(label + " & " + modlabel + " & " +
                estimate(base + "insertion_auc", f"table:oneb:{weighting}:{modality}:insertion") + " & " +
                estimate(base + "deletion_auc", f"table:oneb:{weighting}:{modality}:deletion"))
    table("oneb", "llll", "Estimand & Modality & Insertion AUC & Deletion AUC", body,
        r"Saved RDT-1B PickCube campaign, \OneBCalls{} source calls and \OneBEpisodes{} episodes under both estimands. The first weights each call equally; the second weights each episode's median equally. Conditional 95\% episode intervals describe the released mixture. Evaluation seeds share a checkpoint, and the pooled language population spans two baseline provenances.", "tab:oneb")

    body = []
    for variant in ("frozen", "cascade"):
        for grouping, label in (("recorded_episodes", "Recorded episodes"), ("conditional_shared_context", "Shared-context sensitivity")):
            base = f"sanity/{variant}/{grouping}/spearman_"
            body.append(variant.capitalize() + " & " + label + " & " + counts(base + "vision", f"table:variants:{variant}:{grouping}:counts") + " & " +
                estimate(base + "vision", f"table:variants:{variant}:{grouping}:vision") + " & " +
                estimate(base + "language", f"table:variants:{variant}:{grouping}:language"))
    table("variants", "llrrll", r"Variant & Grouping & Calls & Groups & Vision $\rho$ & Language $\rho$", body,
        r"Saved backbone-randomization variants. Shared-context grouping joins nominal task/episode keys across recorded seeds as a dependence sensitivity; it does not authenticate shared inputs. Intervals are conditional on the stated grouping. Learned encoders/adaptors remain, and sidecar reuse and language-baseline deviations remain unresolved.", "tab:variants")

    body = []
    for task in ("PickCube", "StackCube"):
        base = f"baseline/{task}/rho_"
        body.append(task + " & " + counts(base + "black_gray", "table:baseline:" + task + ":counts") + " & " + " & ".join(
            estimate(base + pair, "table:baseline:" + task + ":" + pair, interval=False)
            for pair in ("black_gray", "black_blur", "gray_blur")))
    table("baseline", "lrrlll", "Task & Calls & Groups & Black/gray & Black/blur & Gray/blur", body,
        r"Saved baseline-ranking correlations. Only two nominal episode groups are available per task. Interval output is retained in the analysis artifact as a descriptive resampling diagnostic, but is not presented here as calibrated uncertainty. Positive correlation does not establish baseline interchangeability.", "tab:baseline")

    body = []
    for task, display in (tasks[0], tasks[2]):
        for seed in (42, 142):
            key = f"m128/{task}/seed{seed}/vision_pct_le3"
            body.append(display + " & " + str(seed) + " & " + counts(key, f"table:m128:{task}:{seed}:counts") + " & " +
                        estimate(key, f"table:m128:{task}:{seed}:rate", digits=1))
    key = "m128/PegInsertionSide/seed142/vision_pct_le3"
    body.append(r"PegInsertion (conflicts excluded) & 142 & " + counts(key, "table:m128:conflict:counts", "conflict_excluded") + " & " +
                estimate(key, "table:m128:conflict:rate", digits=1, view="conflict_excluded"))
    table("budget", "lrrrl", r"Task & Seed & Calls & Groups & Vision residual $\leq3\%$ (\%)", body,
        r"Reconstructible $m=128$ cohorts only. These conditional intervals describe recorded cohorts, not convergence under paired contexts. The final row removes every conflicting repeated key. Missing original PickCube/StackCube subsets are not reconstructed or replaced with constants.", "tab:m128")

    for name, rank, response in (("RescoreQDeletionQ", "Q", "Q"), ("RescoreQDeletionL", "Q", "L2"),
                                 ("RescoreLDeletionL", "L2", "L2"), ("RescoreLDeletionQ", "L2", "Q")):
        macro(name, f"rescore/ranking_{rank}/response_{response}/vision_deletion")
    macro("RescoreCalls", "rescore/ranking_Q/response_Q/vision_deletion", digits=0, field="n_source_records")
    macro("RescoreEpisodes", "rescore/ranking_Q/response_Q/vision_deletion", digits=0, field="n_groups")
    macro("OneBCalls", "oneb/call_weighted/vision_deletion_auc", digits=0, field="n_source_records")
    macro("OneBEpisodes", "oneb/call_weighted/vision_deletion_auc", digits=0, field="n_groups")
    summary = json.loads((source / "summary.json").read_text())
    for name, key in (("ArchiveFiles", "inputs"), ("ArchiveValidRecords", "valid_records"),
                      ("ArchiveDuplicateGroups", "duplicate_groups"), ("ArchiveConflictingGroups", "conflicting_groups"),
                      ("ArchiveQuarantined", "quarantined_lines")):
        metadata_macro(name, "summary.json", key, summary[key])
    null = json.loads((source / "random_order_counterexample.json").read_text())
    metadata_macro("RandomGridAUC", "random_order_counterexample.json", "quadratic_grid_insertion", null["quadratic_grid_insertion"], digits=5)
    # Denominator diagnostics are linked to the corresponding result population.
    diagnostics = json.loads((source / "diagnostics.json").read_text())
    for i, record in enumerate(diagnostics):
        if record.get("family") == "score_denominators" and record.get("task") == "PickCube" and record.get("population") == "all" and record.get("modality") == "vision" and record.get("view") == "retrospective_last":
            metadata_macro("PickCubeInsertionMinimum", "diagnostics.json", f"/{i}/insertion_min", record["insertion_min"], digits=2)
            lineage[-1]["population_result_id"] = "verification/PickCube/all/vision_insertion_auc"
            lineage[-1]["point_population_sha256"] = index["verification/PickCube/all/vision_insertion_auc", "retrospective_last"]["point_population_sha256"]
    (tables / "macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    manuscript = ROOT / "paper/paper.tex"
    abstract = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", manuscript.read_text(encoding="utf-8"), re.S).group(1)
    for name, display in macro_displays.items():
        abstract = abstract.replace("\\" + name + "{}", display)
    abstract = abstract.replace(r"$\ell_2$", "L2")
    if "\\" in abstract:
        raise ValueError("Unexpanded TeX command in plain-text abstract")
    abstract_path = ROOT / "paper/arxiv_abstract.txt"
    abstract_path.write_text(" ".join(abstract.split()) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True, layout="constrained")
    for i, (modality, modlabel) in enumerate((("vision", "Vision"), ("lang", "Language"))):
        for j, direction in enumerate(("insertion", "deletion")):
            ax = axes[i, j]
            for rank, color, marker in (("Q", "#245782", "o"), ("L2", "#bf5b35", "s")):
                selected = [result(f"rescore/ranking_{rank}/response_{response}/{modality}_{direction}", f"figure:response-geometry:{rank}:{response}:{modality}:{direction}") for response in ("Q", "L2")]
                points = np.array([float(row["point"]) for row in selected])
                errors = np.array([[float(row["point"]) - float(row["ci_lo"]) for row in selected], [float(row["ci_hi"]) - float(row["point"]) for row in selected]])
                offset = -.035 if rank == "Q" else .035
                ax.errorbar(np.array([0, 1]) + offset, points, yerr=errors, color=color, marker=marker,
                            capsize=3, lw=1.4, label=("Q ranking" if rank == "Q" else "L2 ranking"))
            ax.set_title(f"{modlabel}: {direction}", loc="left", fontweight="bold")
            ax.set_xticks([0, 1], ["Quadratic response", "L2 response"])
            ax.set_xlim(-.25, 1.25)
            ax.set_ylabel("Median normalized AUC")
            ax.grid(axis="y", color="#dddddd", lw=.6)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(figures / "response_geometry.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(figures / "response_geometry.png", dpi=220)
    plt.close(fig)

    solver = json.loads((source / "solver_endpoint_sensitivity.json").read_text())
    fig, ax = plt.subplots(figsize=(6.6, 2.7), layout="constrained")
    for i, record in enumerate(solver):
        point, lo, hi = [record[key] for key in ("point_change_percent", "ci_lo_change_percent", "ci_hi_change_percent")]
        ax.errorbar(point, i, xerr=[[point-lo], [hi-point]], fmt="o", color="#245782", capsize=4)
        lineage.append({"location": "figure:solver:" + record["task"] + ":" + record["model"],
                        "source": "solver_endpoint_sensitivity.json", "pointer": f"/{i}",
                        "result": {key: value for key, value in record.items() if key != "records"},
                        "record_ids": record["records"]})
    ax.set_yticks(range(len(solver)), [f"{r['task']} / {r['model'].upper()} ({r['n_episode_groups']} groups)" for r in solver])
    ax.invert_yaxis()
    ax.axvline(0, color="#777777", ls=":", lw=1)
    ax.set_xlabel("Change in median relative displacement, T=20 versus T=2 (%)")
    ax.grid(axis="x", color="#dddddd", lw=.6)
    fig.savefig(figures / "solver_endpoint.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(figures / "solver_endpoint.png", dpi=220)
    plt.close(fig)
    registry = {"status": "retrospective_descriptive_not_submission_readiness",
                "analysis_directory": source.relative_to(ROOT).as_posix(), "analysis_verification": verification,
                "builder_sha256": sha(Path(__file__)),
                "manuscript_source_sha256": sha(manuscript),
                "manuscript_inputs_sha256": {"paper/references.bib": sha(ROOT / "paper/references.bib")},
                "numerical_case_inputs_sha256": numerical_inputs,
                "strengthening_inputs_sha256": strengthening_inputs,
                "influence_inputs_sha256": influence_inputs,
                "input_sha256": {name: sha(source / name) for name in ("results.csv", "summary.json", "diagnostics.json", "random_order_counterexample.json", "solver_endpoint_sensitivity.json", "population_membership.json.gz", "raw_line_ledger.csv.gz", "provenance.json")},
                "entries": lineage,
                "generated_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in [abstract_path, *tables.glob("*.tex"), *figures.glob("*.png"), *figures.glob("*.pdf")]}}
    (figures / "lineage.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verification": verification, "lineage_entries": len(lineage), "generated_artifacts": len(registry["generated_sha256"])}))


if __name__ == "__main__":
    main()
