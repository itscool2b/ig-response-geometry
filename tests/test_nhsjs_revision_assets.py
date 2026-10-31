"""Portability and data fidelity checks for the NHSJS revision displays."""
import csv
import io
from pathlib import Path
import runpy
import json
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_figure3_source_paths_are_portable():
    if not (ROOT / "docs/nhsjs-revision/tools/figure3_facts.py").is_file():
        pytest.skip("Private author-support tool is excluded from the public repository")
    module = runpy.run_path(str(ROOT / "docs/nhsjs-revision/tools/figure3_facts.py"))
    generated = module["table_text"](ROOT)
    rows = list(csv.DictReader(io.StringIO(generated)))
    assert len(rows) == 8
    assert {row["source"] for row in rows} == {
        "analysis/paired_rescoring/results/2026-10-01-v1/calls.csv.gz"
    }
    assert generated == (ROOT / "docs/nhsjs-revision/facts/figure3_scale.csv").read_text()


def test_primary_display_has_seven_contrasts_not_rounded_duplicates(tmp_path):
    module = runpy.run_path(str(ROOT / "scripts/build_nhsjs_revision_assets.py"))
    module["tables"](tmp_path)
    content = (tmp_path / "r1_methods_summary.tex").read_text()
    assert content.count(" minus ") == 7
    assert "[-11.4, -0.0115]" in content
    assert "Above fixed means" in content
    for path in tmp_path.glob("*.tex"):
        assert b"\r" not in path.read_bytes(), "Generated hash-bound files must use portable LF bytes"


def test_matched_numerical_display_records_actual_context_populations():
    registry = json.loads((ROOT / "paper/figures_revision/r1_lineage.json").read_text())
    assert {'analysis/revision/response_geometry_theory.py', 'analysis/revision_gpu/v1/figures.py'} <= set(registry['inputs_sha256'])
    populations = registry["populations"]["r1_numerics"]
    assert {target: populations[f"relative_residual/{target}"]["n"] for target in ("Q", "N")} == {"Q": 12, "N": 12}
    assert {target: populations[f"relative_l1/{target}"]["n"] for target in ("Q", "N")} == {"Q": 8, "N": 8}
    for panel in populations.values():
        assert len(panel["contexts"]) == len(set(panel["contexts"])) == panel["n"]


def test_r1_asset_validation_rejects_changed_or_escaped_sources(tmp_path, monkeypatch):
    from scripts.build_paper import validate_revision_assets
    from scripts import validate_template_provenance
    import hashlib
    monkeypatch.setattr(validate_template_provenance, "validate_template_provenance", lambda root: None)
    paper = tmp_path / "paper"
    (paper / "figures_revision").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (paper / "paper.tex").write_text(r"\includegraphics{figures_revision/r1_design.pdf}")
    asset = paper / "data.csv"
    asset.write_text("original")
    builder = tmp_path / "scripts/build_nhsjs_revision_assets.py"
    builder.write_text("builder")
    record = {"inputs_sha256": {"paper/data.csv": hashlib.sha256(asset.read_bytes()).hexdigest()},
              "outputs_sha256": {}, "manuscript_inputs_sha256": {},
              "builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest()}
    path = paper / "figures_revision/r1_lineage.json"
    path.write_text(json.dumps(record))
    validate_revision_assets(tmp_path)
    asset.write_text("modified")
    with pytest.raises(RuntimeError, match="Stale or missing R1 inputs"):
        validate_revision_assets(tmp_path)
    record["inputs_sha256"] = {"../outside.csv": "0" * 64}
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="outside the repository"):
        validate_revision_assets(tmp_path)
