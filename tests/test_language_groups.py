import csv
import json
import shutil
import subprocess

import pytest

from analysis import language_groups as lg
from analysis.revision.core import IntegrityError, Record


def test_historical_and_current_rules_with_half_even_rounding():
    assert [lg.historical_k(20, p) for p in (0, 1, 5, 10, 20, 30, 50, 75, 100)] == [0, 1, 1, 2, 4, 6, 10, 15, 20]
    assert [lg.current_k(20, p) for p in (0, 1, 5, 10, 20, 30, 50, 75, 100)] == [0, 0, 1, 2, 4, 6, 10, 15, 20]
    assert lg.historical_k(30, 5) == 2 and lg.historical_k(30, 75) == 22
    assert lg.historical_k(25, 10) == 2 and lg.historical_k(25, 30) == 8
    assert lg.historical_k(729, 50) == 364 and lg.historical_k(729, 1) == 7
    assert lg.historical_k(1, 1) == 1 and lg.current_k(1, 1) == 0


def test_count_rows_flags_floor_and_ties():
    rows = {r["nominal_percent"]: r for r in lg.count_rows("toy", 30, [0, 1, 5, 100])}
    assert rows[1]["historical_floor_active"] and not rows[5]["historical_floor_active"]
    assert rows[5]["half_way_tie"] and rows[5]["historical_k"] == 2
    assert rows[0]["node"] == "endpoint_not_masked_by_rule" and rows[100]["historical_fraction"] == 1.0
    assert rows[1]["historical_fraction_exact"] == "1/30" and rows[1]["current_fraction_exact"] == "0/30"


def test_encoder_facts_reads_the_literal_without_importing():
    source = 'import torch\ntask2lang = {"A-v1": "Do a thing."}\ndef f(sp, x):\n    token_ids = sp.Encode(x)\n    token_ids.append(1)\n'
    facts = lg.encoder_facts(source)
    assert facts == dict(instructions={"A-v1": "Do a thing."}, appends_eos=True, uses_sp_encode=True)
    assert not lg.encoder_facts(source.replace("append(1)", "append(2)"))["appends_eos"]
    with pytest.raises(IntegrityError):
        lg.encoder_facts("x = 1\n")


def test_equal_count_compares_saved_one_and_five_percent_nodes():
    grid = [0, 1, 5, 10, 20, 30, 50, 75, 100]
    rows = [Record("f", "0" * 64, i, f"{i:064d}", dict(k_grid_auc=grid, lang_insertion_curve=[0, a, b, 0, 0, 0, 0, 0, 0]))
            for i, (a, b) in enumerate([(1.0, 1.0), (1.0, 2.0)])]
    assert lg.equal_count(rows, "lang", "insertion") == (1, 2)


def test_vendored_tokenizer_is_the_pinned_file():
    _, protocol = lg.load_protocol()
    blob = (lg.ROOT / protocol["tokenizer"]["vendored_path"]).read_bytes()
    assert len(blob) == protocol["tokenizer"]["bytes"]
    assert lg.sha256(blob) == protocol["tokenizer"]["sha256"]


def history_available():
    _, protocol = lg.load_protocol()
    commits = protocol["git_history"]["commits"] + protocol["encoder"]["history_commits"]
    try:
        for commit in commits:
            subprocess.run(["git", "-C", str(lg.ROOT), "cat-file", "-e", f"{commit}^{{commit}}"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def authors_cached():
    try:
        lg.authors_embeddings(lg.load_protocol()[1], allow_download=False)
    except Exception:
        return False
    return True


def test_real_data_reproduction_counts_and_history(tmp_path, monkeypatch):
    pytest.importorskip("sentencepiece")
    if not history_available():
        pytest.skip("full git history (afd1249 to 934023d) is required")
    committed_authors = json.loads((lg.RESULTS / "authors_embeddings.json").read_bytes())
    cached = authors_cached()
    if not cached:
        monkeypatch.setattr(lg, "authors_embeddings", lambda protocol, allow_download=False: committed_authors)
    fresh = tmp_path / "fresh"
    lg.run(fresh)
    for name in lg.CORE_OUTPUTS + (lg.AUTHORS_OUTPUTS if cached else ()):
        assert (fresh / name).read_bytes() == (lg.RESULTS / name).read_bytes(), name
    produced = json.loads((fresh / "provenance.json").read_bytes())
    recorded = json.loads((lg.RESULTS / "provenance.json").read_bytes())
    for key in ("protocol_sha256", "source_sha256", "inputs", "vendored_sha256"):
        assert produced[key] == recorded[key], key
    assert lg.verify(lg.RESULTS)["status"] == "verified"
    tokens = json.loads((lg.RESULTS / "tokenization.json").read_bytes())
    assert {t: v["n_attended_positions"] for t, v in tokens["tasks"].items()} == {
        "PickCube-v1": 20, "StackCube-v1": 30, "PegInsertionSide-v1": 25, "PickSingleYCB-v1": 16}
    assert all(v["recorded_token_ids_equal"] for v in tokens["month8_cross_check"].values())
    assert all(all(h["instructions_equal_to_current"].values()) and h["appends_eos"] for h in tokens["encoder_history"])
    with (lg.RESULTS / "saved_curve_check.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    headline = [r for r in rows if r["scope"] == "headline_cohort"]
    assert sum(int(r["n_curves"]) for r in headline) == sum(int(r["n_equal_1_and_5"]) for r in headline) == 3000
    for row in rows:
        if row["scope"] == "all_faithfulness_files" and row["modality"] == "lang":
            expected = int(row["n_curves"]) if row["historical_rule_implies_equal"] == "True" else 0
            assert int(row["n_equal_1_and_5"]) == expected, row["population"]
    history = json.loads((lg.RESULTS / "git_history.json").read_bytes())
    assert [c["floor_present"] for c in history["commits"]] == [True, True, True, False, False]
    assert [c["no_floor_present"] for c in history["commits"]] == [False, False, False, True, True]
    assert history["archived_copy"]["equal_to_reference_after_crlf_to_lf"]
    copy = tmp_path / "copy"
    shutil.copytree(lg.RESULTS, copy)
    (copy / "group_counts.csv").write_bytes(b"x\n")
    with pytest.raises(IntegrityError, match="Artifact mismatch"):
        lg.verify(copy)
