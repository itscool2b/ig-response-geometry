"""Generate current manuscript displays from immutable local evidence.

Use --reproduce to repeat retained scientific analyses in fresh temporary
directories. Historical producers, protocols, raw data and outputs are not edited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.manuscript_revision.report import (
    OUTPUT, VERSION, build_report, json_text, read_json, reproduce, sha, write_report,
)
from analysis.manuscript_revision.figures import STEMS, build_figures, representative_call


def plain_abstract(text, macros):
    match = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", text, re.S)
    if not match:
        raise ValueError("Manuscript abstract environment is missing")
    # Review trace comments belong to the source, not the exported prose.
    # Preserve escaped percent signs used in reported percentages.
    abstract = re.sub(r"(?<!\\)%[^\n]*", "", match.group(1))
    for name, record in sorted(macros.items(), key=lambda item: -len(item[0])):
        abstract = re.sub(r"\\" + name + r"(?![A-Za-z])(?:\{\})?", lambda _: record["display"], abstract)
    abstract = re.sub(r"\\(?:textbf|textit|emph|mathrm|operatorname)\{([^{}]*)\}", r"\1", abstract)
    for command, value in ((r"\ell_2", "L2"), (r"\ell", "L"), (r"\Delta", "Delta"),
                           (r"\%", "%"), (r"\(", ""), (r"\)", ""), (r"\,", " "),
                           (r"\Q", "Q"), (r"\Lnorm", "L")):
        abstract = abstract.replace(command, value)
    abstract = abstract.replace("$", "").replace("{,}", ",").replace("~", " ")
    if "\\" in abstract:
        raise ValueError("Unexpanded TeX command in plain-text abstract: " + abstract)
    return " ".join(abstract.replace("{", "").replace("}", "").split()) + "\n"


def build(root=ROOT, *, fresh_reproduction=False):
    root = Path(root)
    if fresh_reproduction:
        reproduce(root)
    report, calls = build_report(root)
    output, manifest = write_report(report, root)
    macros = build_figures(root, report, calls)
    manuscript = root / "paper/paper.tex"
    abstract_path = root / "paper/arxiv_abstract.txt"
    abstract_path.write_text(plain_abstract(manuscript.read_text(encoding="utf-8"), macros), encoding="utf-8", newline="\n")
    generated = [abstract_path, root / "paper/tables_revision/paired_results.tex", root / "paper/tables_revision/revision_facts.tex"]
    generated += [root / f"paper/figures_revision/{stem}.{extension}" for stem in STEMS for extension in ("pdf", "png")]
    generated += [output / name for name in manifest["outputs"]] + [output / "manifest.json"]
    inputs = dict(report["source_sha256"])
    inputs.update(manifest["source_sha256"])
    inputs["scripts/build_revision_assets.py"] = sha(root / "scripts/build_revision_assets.py")
    proof_inputs = {path.relative_to(root).as_posix(): sha(path)
                    for suffix in ("*.tex", "*.bib", "*.sty", "*.bst") for path in sorted((root / "paper").glob(suffix))}
    entries = [dict(location="table:paired-results", source=(OUTPUT / "paired_summary.csv").as_posix(),
                    result=row, population_membership=row["membership_sha256"],
                    dependency="same physical saved call under both responses") for row in report["paired"]]
    entries += [dict(location="macro:"+name, source=(OUTPUT / "report.json").as_posix(), **record)
                for name, record in macros.items()]
    entries += [dict(location="table:sources:task-success", source=(OUTPUT / "task_success.csv").as_posix(), result=row)
                for row in report["task_success"]]
    entries += [dict(location="figure:numerical-diagnostics", source=(OUTPUT / "numerical_by_context.csv").as_posix(), result=row)
                for row in report["numerical"]]
    for key, name in (("call_index_summary", "call-index"), ("baseline_gap_summary", "baseline-gaps"),
                       ("grid_diagnostics", "saved-grid"),
                       ("language_schedule_sensitivity", "language-sensitivity"),
                       ("numerical_evaluation_checks", "numerical-evaluation")):
        entries += [dict(location="report:" + name, source=(OUTPUT / (key + ".csv")).as_posix(), result=row)
                    for row in report[key]]
    entries += [dict(location="report:cohort-relationship", source=(OUTPUT / "report.json").as_posix(),
                     result=report["cohort_relationship"]),
                dict(location="report:numerical-violations", source=(OUTPUT / "report.json").as_posix(),
                     result=report["numerical_violation_summary"])]
    typical = representative_call(calls)
    receipt = output / "reproduction.json"
    reproduction_status = "not rerun by this refresh; frozen evidence hash-verified"
    if receipt.exists():
        saved = read_json(receipt)
        if saved["input_sha256"] != report["source_sha256"]:
            raise ValueError("Fresh reproduction receipt is stale")
        generated.append(receipt)
        reproduction_status = saved["status"]
    registry = dict(schema_version=2, reporting_version=VERSION,
        status="local descriptive revision; submission decisions remain separate",
        analysis_directory="analysis/revision/results/2026-09-30-v2",
        analysis_verification=report["verification"]["canonical"],
        builder_sha256=sha(root / "scripts/build_revision_assets.py"),
        manuscript_source_sha256=sha(manuscript), manuscript_inputs_sha256=proof_inputs,
        reporting_inputs_sha256=inputs, entries=entries, reproduction_status=reproduction_status,
        current_display_files=[f"paper/figures_revision/{stem}.pdf" for stem in STEMS] + ["paper/tables_revision/paired_results.tex"],
        display_count=7, display_count_note="Five figures, the paired-results table, and the source/claim table in the manuscript.",
        figure_data=dict(pipeline="Conceptual separation of target and response; no image inference or simulation",
            response_mechanism=dict(theory="normalized_responses with R=1 and epsilon=1e-12",
                                   example_record_id=report["tail_examples"]["Q:vision:insertion"]["record_id"],
                                   typical_record_id=typical["record_id"],
                                   typical_selection="Nearest to the Q-ranked vision-insertion median delta; record ID breaks ties",
                                   typical_residual_ratios=[value/float(typical["residual_baseline"])
                                       for value in json.loads(typical["residual_points"])],
                                   typical_delta=float(typical["delta"])),
            paired_effects=dict(cases=list(calls), all_defined_calls=sum(row["n_defined_calls"] for row in report["paired"]),
                                all_calls_visible_in_panel_A=True, panel_B_limits=[-.05, .25],
                                no_cross_cohort_connections=True),
            analytic_toys="Verified polynomial reversal and K=128 oscillation; no real-model ranking claim",
            numerical_diagnostics="Every completed context; separate IG and path-gradient coordinates; all completeness budgets"),
        generated_sha256={path.relative_to(root).as_posix(): sha(path) for path in generated})
    path = root / "paper/figures_revision/lineage.json"
    path.write_text(json_text(registry), encoding="utf-8", newline="\n")
    return dict(reporting_version=VERSION, cases=len(report["paired"]),
                numerical_comparisons=len(report["numerical"]), macros=len(macros),
                generated_artifacts=len(generated), lineage_entries=len(entries), reproduction=reproduction_status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", action="store_true", help="Fresh CPU reproduction before display generation")
    args = parser.parse_args()
    print(json.dumps(build(fresh_reproduction=args.reproduce), indent=2))


if __name__ == "__main__":
    main()
