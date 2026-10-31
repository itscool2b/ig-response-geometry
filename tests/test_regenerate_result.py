import json
import sys

import pytest
import torch

import revision_numerics as rn
from scripts import regenerate_result as rr
from tests.revision_toy import STRICT, stub_replay_layer, worker_args


def test_kit_manifest_detects_changed_and_unlisted_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/b.bin").write_bytes(b"b")
    (tmp_path / "MANIFEST.sha256").write_text(f"{rr.sha256(tmp_path / 'a.txt')}  a.txt\n{rr.sha256(tmp_path / 'sub/b.bin')}  sub/b.bin\n")
    assert rr.verify_kit(tmp_path)["status"] == "verified"
    (tmp_path / "sub/b.bin").write_bytes(b"c")
    (tmp_path / "extra").write_text("x")
    result = rr.verify_kit(tmp_path)
    assert result["mismatched"] == ["sub/b.bin"] and result["unlisted"] == ["extra"] and result["status"] == "failed"


def test_timing_fields_are_ignored_but_values_are_not():
    a = {"wall_seconds": 1.0, "x": [{"elapsed_seconds": 2, "y": 3}], "started_unix": 5}
    b = {"wall_seconds": 9.0, "x": [{"elapsed_seconds": 7, "y": 3}], "started_unix": 6}
    assert rr.strip_timing(a) == rr.strip_timing(b)
    b["x"][0]["y"] = 4
    assert rr.strip_timing(a) != rr.strip_timing(b)


@pytest.mark.parametrize("name", ["../outside", "/outside", "C:/outside", "sub\\file", "./file", "a//b", "MANIFEST.sha256"])
def test_kit_rejects_unsafe_or_noncanonical_manifest_paths(tmp_path, name):
    (tmp_path / "MANIFEST.sha256").write_text("0" * 64 + "  " + name + "\n")
    result = rr.verify_kit(tmp_path)
    assert result["status"] == "failed" and result["invalid"]


@pytest.mark.parametrize("contents", ["", "malformed\n", "0" * 63 + "  file\n"])
def test_kit_rejects_empty_or_malformed_manifest(tmp_path, contents):
    (tmp_path / "MANIFEST.sha256").write_text(contents)
    assert rr.verify_kit(tmp_path)["status"] == "failed"


def test_kit_detects_duplicate_entries_and_nested_manifest(tmp_path):
    (tmp_path / "file").write_bytes(b"a")
    line = f"{rr.sha256(tmp_path / 'file')}  file\n"
    (tmp_path / "MANIFEST.sha256").write_text(line * 2)
    assert rr.verify_kit(tmp_path)["invalid"]
    (tmp_path / "MANIFEST.sha256").write_text(line)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/MANIFEST.sha256").write_text("unlisted")
    assert rr.verify_kit(tmp_path)["unlisted"] == ["sub/MANIFEST.sha256"]


def test_map_comparison_reports_identity_and_differences():
    stored = {"vision/Q/A1T8": torch.ones(1, 729, 4), "state/Q/A1T8": torch.ones(1, 1, 8), "vision/Q/A1T8/group_scores": torch.ones(729)}
    fresh = {key: value.clone() for key, value in stored.items()}
    fresh["vision/Q/A1T8"][0, 0, 0] = 1.5
    report = rr.compare_maps(stored, fresh)
    assert sorted(report["identical"]) == ["state/Q/A1T8", "vision/Q/A1T8/group_scores"]
    assert report["differing"]["vision/Q/A1T8"]["relative_l1"] == pytest.approx(0.5 / (729 * 4))
    # Group scores are compared too, and missing or extra tensors are reported, not skipped.
    fresh["vision/Q/A1T8/group_scores"] = torch.arange(729.0)
    del fresh["state/Q/A1T8"]
    fresh["vision/Q/A9"] = torch.ones(1)
    report = rr.compare_maps(stored, fresh)
    assert "vision/Q/A1T8/group_scores" in report["differing"]
    assert report["missing"] == ["state/Q/A1T8"] and report["extra"] == ["vision/Q/A9"]


def test_area_comparison_counts_one_sided_areas_as_mismatches():
    def record(value):
        responses = {r: dict(normalized_auc=value, nominal_normalized_auc=value, raw_auc=value) for r in ("Q", "L2")}
        return dict(modalities=dict(vision=dict(rankings=dict(a=dict(status="defined", curves=dict(
            deletion=dict(responses=responses)))))))
    same = rr.area_differences(record(0.3), record(0.3))
    assert same["compared"] == 6 and same["max_abs_normalized"] == 0 and not same["definedness_mismatch"]
    one_sided = rr.area_differences(record(0.3), record(None))
    assert len(one_sided["definedness_mismatch"]) == 6 and one_sided["compared"] == 0


def test_cpu_regeneration_authenticates_every_scored_context(monkeypatch, tmp_path, capsys):
    protocol_path, protocol = stub_replay_layer(monkeypatch, tmp_path, contexts=2)
    rn.run_worker(worker_args(protocol_path, tmp_path / "out", "bf16,fp32,scoring"), protocol, dict(settings=STRICT))
    capsys.readouterr()
    args = type("Args", (), dict(root=tmp_path / "out", protocol=protocol_path, bank_file=tmp_path / "bank.json",
                                 bank_id="P", plan=None, contexts=None))()
    summary = rr.cpu_regeneration(args)
    assert summary["contexts"] == 2 and summary["modality_results_authenticated"] == 6
    assert summary["phases_authenticated"] == 2 + 1 + 2  # bf16 and scoring for both, fp32 for the first
    again = rr.cpu_regeneration(args)
    assert again["rows_fingerprint_sha256"] == summary["rows_fingerprint_sha256"]
    record = tmp_path / "out/P/e0001-c003/scoring/record.json"
    data = json.loads(record.read_text())
    key = next(iter(data["modalities"]["vision"]["response_table"]))
    data["modalities"]["vision"]["response_table"][key]["Q"] += 1.0
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        rr.cpu_regeneration(args)


def test_cpu_regeneration_can_check_one_stored_context(monkeypatch, tmp_path, capsys):
    import shutil
    protocol_path, protocol = stub_replay_layer(monkeypatch, tmp_path, contexts=2)
    rn.run_worker(worker_args(protocol_path, tmp_path / "out", "bf16,fp32,scoring"), protocol, dict(settings=STRICT))
    capsys.readouterr()
    shutil.rmtree(tmp_path / "out/P/e0000-c003")
    args = type("Args", (), dict(root=tmp_path / "out", protocol=protocol_path, bank_file=tmp_path / "bank.json",
                                 bank_id="P", plan=None, contexts="e0001-c003"))()
    assert rr.cpu_regeneration(args)["contexts"] == 1


@pytest.mark.parametrize("tamper", ["nominal", "order", "counts", "swap"])
def test_schedule_and_mask_check_catches_what_action_authentication_misses(monkeypatch, tmp_path, capsys, tamper):
    protocol_path, protocol = stub_replay_layer(monkeypatch, tmp_path, contexts=1)
    rn.run_worker(worker_args(protocol_path, tmp_path / "out", "bf16,fp32,scoring"), protocol, dict(settings=STRICT))
    capsys.readouterr()
    result = json.loads((tmp_path / "out/P/e0000-c003/scoring/record.json").read_text())["modalities"]["vision"]
    rr.verify_schedule_and_masks(result)
    ranking = result["rankings"]["IG_Q_A1T8"]
    if tamper == "nominal":
        ranking["curves"]["deletion"]["responses"]["Q"]["nominal_normalized_auc"] = 123.0
    elif tamper == "order":
        ranking["order"] = list(reversed(ranking["order"]))
    elif tamper == "counts":
        result["selected_counts"] = [0] * len(result["selected_counts"])
    else:
        ranking["curves"]["deletion"], ranking["curves"]["insertion"] = ranking["curves"]["insertion"], ranking["curves"]["deletion"]
    with pytest.raises(ValueError):
        rr.verify_schedule_and_masks(result)
