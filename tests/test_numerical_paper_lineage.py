"""A manuscript build must fail when its saved numerical evidence changes."""
import hashlib
import json

import pytest

from scripts.build_paper import validate_revision_assets


@pytest.mark.parametrize("changed", ["analysis/numerical_case/summarize.py", "analysis/numerical_case/case.json"])
def test_numerical_input_changes_invalidate_an_unchanged_display(tmp_path, changed):
    files = {
        "paper/paper.tex": b"Manuscript",
        "paper/tables_revision/numerical_roster.tex": b"Frozen display",
        "analysis/numerical_case/summarize.py": b"Reproducer",
        "analysis/numerical_case/case.json": b'{"completed":12}',
    }
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
    path = tmp_path / "paper/figures_revision/lineage.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(registry), encoding="utf-8")
    validate_revision_assets(tmp_path)
    (tmp_path / changed).write_bytes(files[changed] + b"changed")
    with pytest.raises(RuntimeError, match="Stale or missing manuscript assets"):
        validate_revision_assets(tmp_path)
