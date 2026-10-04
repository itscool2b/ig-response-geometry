"""A manuscript build must fail when its saved numerical evidence changes."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_paper import validate_revision_assets


@pytest.mark.parametrize("group,changed", [
    ("numerical_case_inputs_sha256", "analysis/numerical_case/summarize.py"),
    ("numerical_case_inputs_sha256", "analysis/numerical_case/case.json"),
    ("reporting_inputs_sha256", "analysis/manuscript_revision/report.py"),
    ("reporting_inputs_sha256", "analysis/manuscript_revision/figures.py"),
    ("reporting_inputs_sha256", "data/raw-evidence.jsonl"),
    ("builder_sha256", "scripts/build_revision_assets.py"),
])
def test_numerical_input_changes_invalidate_an_unchanged_display(tmp_path, group, changed):
    files = {
        "paper/paper.tex": b"Manuscript",
        "paper/tables_revision/numerical_roster.tex": b"Frozen display",
        "analysis/numerical_case/summarize.py": b"Reproducer",
        "analysis/numerical_case/case.json": b'{"completed":12}',
        "analysis/manuscript_revision/report.py": b"Reporting source",
        "analysis/manuscript_revision/figures.py": b"Display source",
        "data/raw-evidence.jsonl": b"Saved evidence",
        "scripts/build_revision_assets.py": b"Asset builder",
    }
    root = Path(__file__).resolve().parents[1]
    for name in ("tmlr.sty", "tmlr.bst", "tmlr-LICENSE", "tmlr-source.json"):
        files["paper/" + name] = (root / "paper" / name).read_bytes()
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    registry = {
        "manuscript_source_sha256": digest(files["paper/paper.tex"]),
        "generated_sha256": {name: digest(data) for name, data in files.items() if "tables_revision" in name},
        "numerical_case_inputs_sha256": {name: digest(data) for name, data in files.items() if name.startswith("analysis/")},
    }
    if group == "builder_sha256":
        registry[group] = digest(files[changed])
    else:
        registry[group] = {changed: digest(files[changed])}
    path = tmp_path / "paper/figures_revision/lineage.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry), encoding="utf-8")
    validate_revision_assets(tmp_path)
    (tmp_path / changed).write_bytes(files[changed] + b"changed")
    with pytest.raises(RuntimeError, match="Stale or missing manuscript assets") as caught:
        validate_revision_assets(tmp_path)
    assert changed in str(caught.value)
