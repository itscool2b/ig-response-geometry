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
    content = (tmp_path / "r1_methods_summary.tex").read_text(encoding="utf-8")
    assert len(module['POPULATIONS']['r1_methods_summary']['primary_contrasts']) == 7
    assert content.count(" minus ") == 4
    assert "[−11.4, −0.0115]" in content
    assert "Scaled mean" in content and "Episode median" in content
    assert 'Equal-weight IG-Q' in content
    assert 'A. Primary contrasts' in content and 'B. Point-estimate choices' in content
    assert (tmp_path/'r1_all_methods.tex').read_text(encoding="utf-8").count(' \\\\') >= 20
    assert 'Mean advantage' in (tmp_path/'r1_all_methods.tex').read_text(encoding='utf-8')
    identity=(tmp_path/'r1_identities.tex').read_text(encoding='utf-8')
    assert 'd10de3d61747' in identity and '384 × 384' in identity
    assert 'Runtime' in identity
    success=(tmp_path/'r1_success_summary.tex').read_text(encoding='utf-8')
    assert '142, encoded' in success and success.count('Not estimable')==2
    fixed=module['POPULATIONS']['r1_fixed_order_tails']
    raster=next(r for r in fixed if r['bank']=='P' and r['response']=='Q' and r['ranking']=='fixed_raster')
    assert raster['episodes']==60 and raster['positive_share']==.1
    assert raster['raw_mean']>7 and raster['episode_median']<0 and raster['scaled_mean']<0
    original=module['POPULATIONS']['r1_original_offline_summary']
    assert original['contexts']==12 and original['episodes']==6 and original['completed_arms']==72
    assert len(original['modality_target_rows'])==6
    assert 'not the mixed-task qualification sample' in (tmp_path/'r1_diagnostics.tex').read_text(encoding='utf-8')
    for path in tmp_path.glob("*.tex"):
        assert b"\r" not in path.read_bytes(), "Generated hash-bound files must use portable LF bytes"
        table_source=path.read_text(encoding='utf-8')
        assert table_source.count(r'\caption*')==table_source.count(r'\noalign{\addtocounter{table}{-1}}'), 'Unnumbered longtable continuations must adjust the counter outside alignment cells'


def test_matched_numerical_display_records_actual_context_populations():
    registry = json.loads((ROOT / "paper/figures_revision/r1_lineage.json").read_text())
    assert {'analysis/revision/response_geometry_theory.py', 'analysis/revision_gpu/v1/figures.py'} <= set(registry['inputs_sha256'])
    populations = registry["populations"]["r1_numerics"]
    assert {target: populations[f"relative_residual/{target}"]["n"] for target in ("Q", "N")} == {"Q": 12, "N": 12}
    assert {target: populations[f"relative_l1/{target}"]["n"] for target in ("Q", "N")} == {"Q": 8, "N": 8}
    for key, panel in populations.items():
        if key == 'display_notes':
            continue
        assert len(panel["contexts"]) == len(set(panel["contexts"])) == panel["n"]


def test_full_saved_zoom_counts_and_new_success_intervals_are_retained():
    registry=json.loads((ROOT/'paper/figures_revision/r1_lineage.json').read_text())
    cases=registry['populations']['r1_saved_geometry']['saved_cases']
    assert len(cases)==8 and all(r['calls']==750 for r in cases.values())
    assert all(r['outside_zoom_above']==0 for r in cases.values())
    assert cases['Q:vision:insertion']['outside_zoom_below']==74
    successes=registry['populations']['r1_saved_geometry']['success_task_rows']
    failed=next(r for r in successes if r['group']=='New 1B failure' and r['direction']=='deletion')
    assert failed['calls']==18 and failed['episodes']==9
    assert failed['ci_lo']<0<failed['ci_hi']
    assert failed['overshoot']==13 and failed['below_minus_1']==0


def test_figures_are_lossless_at_exact_placed_width():
    from PIL import Image
    for name in ('r1_design','r1_saved_geometry','r1_numerics'):
        with Image.open(ROOT/'paper/figures_revision'/f'{name}.png') as image:
            assert image.format=='PNG'
            assert image.width==1755  # 5.85 inches at 300 dpi, no post-layout shrink.


def test_original_offline_figure_preserves_every_completed_context():
    registry=json.loads((ROOT/'paper/figures_revision/r1_lineage.json').read_text())
    original=registry['populations']['r1_offline_numerics']
    assert original['completed_arms']==72 and original['contexts']==12
    assert len(original['panels'])==18
    assert all(len(ids)==12 and len(set(ids))==12 for ids in original['panels'].values())


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
