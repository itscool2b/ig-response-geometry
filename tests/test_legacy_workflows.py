"""Release-boundary and coordinate/identity tests without model downloads."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from experiment_io import RunStore, object_hash, tensor_hash
from generate_overlays import render_run
from legacy_status import DISPOSITIONS, status
from make_annotations import categorize_residuals, export_annotations
from overlays import (extract_language_per_token, extract_vision_heatmap,
                      model_input_rgb, observation_array, render_vision_panel,
                      render_tokens_only_figure, save_new_figure, selected_token_labels)
from scripts.run_workflow import plan

ROOT = Path(__file__).resolve().parents[1]


def fixture_run(tmp_path, *, seed=42, model="170m"):
    path = tmp_path / f"run-{model}-{seed}" / "metrics.jsonl"
    config = {"task": "PickCube-v1", "model": model, "target": "logpi", "m": 8,
        "quadrature": "trapezoid", "solver_steps": 5, "pipeline": {"checkpoint": "fixture"},
        "language": {}, "observation_pipeline": "one_current_external_camera_five_background_slots"}
    with RunStore(path, config).writer() as store:
        vision = torch.zeros(1, 4374, 2)
        vision[0, 3 * 729 + 13, :] = torch.tensor([1., .5])
        image = torch.zeros(384, 384, 3, dtype=torch.uint8)
        image[:192, :192, 0] = 255
        payload = dict(obs_image=image, proprio=torch.zeros(8), initial_noise=torch.ones(1, 2, 128),
            ref_action=torch.zeros(1, 2, 128), vision_attr=vision,
            lang_attr=torch.tensor([[[1., 2.], [10., 20.], [-2., -3.]]]),
            state_attr=torch.zeros(1, 1, 128), lang_attn_mask=torch.tensor([[True, False, True]]))
        noise = tensor_hash(payload["initial_noise"])
        context = object_hash(dict(episode=0, policy_call=0, configuration=store.identity,
            observation=tensor_hash(payload["obs_image"]), proprio=tensor_hash(payload["proprio"]),
            noise=noise, reference=tensor_hash(payload["ref_action"])))
        payload.update(initial_noise_sha256=noise, context_id=context)
        row = dict(event="step", task="PickCube-v1", model=model, seed=seed, episode=0, policy_call_idx=0,
            context_id=context, initial_noise_sha256=noise, vision_err=.01, lang_err=.02, state_err=.08)
        transaction = store.begin_episode(0)
        transaction.save_step(row, payload)
        transaction.commit(dict(event="episode_end", episode=0, policy_calls=1))
    return path


def test_historical_archives_are_byte_verified_and_not_silent_success():
    for name in DISPOSITIONS:
        assert status(name)["status"] == "historical_execution_retired"
    result = subprocess.run([sys.executable, str(ROOT / "finetune_rdt.py")], capture_output=True, text=True)
    assert result.returncode == 2 and "retired" in result.stderr
    result = subprocess.run([sys.executable, str(ROOT / "finetune_rdt.py"), "--status"], capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout)["entrypoint"] == "finetune_rdt.py"


def test_retired_rdt_demo_needs_no_model_packages_and_writes_nothing(tmp_path):
    # -S removes site packages: status and retirement must work without torch,
    # simulator, checkpoint downloads, or any model initialization.
    command = [sys.executable, "-S", str(ROOT / "ig_rdt.py")]
    result = subprocess.run(command + ["--status"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["entrypoint"] == "ig_rdt.py"
    assert report["archive"]["archive"] == "legacy/2026-10-01/ig_rdt.py.txt"
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and "retired" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_new_archive_tampering_is_rejected(tmp_path, monkeypatch):
    import legacy_status
    original = status("ig_rdt.py")["archive"]
    archive = tmp_path / "legacy/2026-10-01"
    archive.mkdir(parents=True)
    (archive / "manifest.json").write_text(json.dumps({"files": [original]}))
    (tmp_path / original["archive"]).write_text("modified historical source")
    monkeypatch.setattr(legacy_status, "ROOT", tmp_path)
    monkeypatch.setattr(legacy_status, "ENTRY_ARCHIVES", {"ig_rdt.py": archive})
    with pytest.raises(ValueError, match="Historical archive hash mismatch"):
        status("ig_rdt.py")


def test_exact_model_coordinates_invert_normalization_without_uncropping():
    rgb = torch.zeros(3, 8, 5)
    rgb[0, 1, 4] = 1
    mean, std = [.4, .5, .6], [.2, .3, .4]
    normalized = (rgb - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    display = model_input_rgb(normalized.unsqueeze(0), mean, std)
    assert display.shape == (8, 5, 3)
    np.testing.assert_allclose(display, rgb.permute(1, 2, 0), atol=1e-7)


def test_patch_and_image_extents_align_and_mask_holes_are_respected():
    fig, ax = plt.subplots()
    render_vision_panel(ax, torch.zeros(384,384,3,dtype=torch.uint8), np.zeros((27,27)))
    assert ax.images[0].get_extent() == ax.images[1].get_extent()
    assert ax.images[1].get_interpolation() == "nearest"
    plt.close(fig)
    values = torch.tensor([[[1.,2.],[30.,40.],[5.,6.]]])
    mask = torch.tensor([[True,False,True]])
    assert extract_language_per_token(values, mask).tolist() == [3.,11.]
    assert selected_token_labels(["a","b","c"], mask) == ["a","c"]
    assert selected_token_labels(None, mask) == ["position_0","position_2"]
    with pytest.raises(ValueError):
        observation_array(torch.zeros(512,512,3,dtype=torch.uint8))


def test_zero_map_and_signed_cancellation_are_explicit_display_conventions():
    values = torch.zeros(1,4374,2)
    assert np.isfinite(extract_vision_heatmap(values)).all()
    values[0,3*729,:] = torch.tensor([2.,-2.])
    assert extract_vision_heatmap(values).max() == 0
    values[0,3*729,0] = float("nan")
    with pytest.raises(ValueError):
        extract_vision_heatmap(values)


def test_language_figure_height_uses_attended_tokens_not_padded_capacity(tmp_path, monkeypatch):
    import overlays
    captured = {}
    def capture(figure, output, **kwargs):
        captured["height"] = figure.get_figheight()
        captured["labels"] = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
    monkeypatch.setattr(overlays, "save_new_figure", capture)
    mask = torch.zeros(1, 1024, dtype=torch.bool)
    mask[0, [0, 700, 1023]] = True
    render_tokens_only_figure({"lang_attr": torch.ones(1, 1024, 2)}, None,
                              tmp_path / "tokens.png", lang_attn_mask=mask)
    assert captured["height"] == 4
    assert captured["labels"] == ["position_0", "position_700", "position_1023"]


def test_rendered_run_preserves_context_identity_and_refuses_overwrite(tmp_path):
    metrics = fixture_run(tmp_path)
    output = tmp_path / "figures"
    record = render_run(metrics, "PickCube-v1", output, three_panel=True)
    assert record["status"] == "complete" and len(record["contexts"]) == 1
    assert len(record["artifacts"]) == 4
    for artifact in record["artifacts"]:
        assert (output / artifact["file"]).exists()
    with pytest.raises(FileExistsError):
        render_run(metrics, "PickCube-v1", output)
    metrics.write_bytes(metrics.read_bytes() + b"{bad}\n")
    with pytest.raises(ValueError):
        render_run(metrics, "PickCube-v1", tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_annotations_keep_seed_model_identity_and_numerical_scope(tmp_path):
    paths = [fixture_run(tmp_path, seed=42), fixture_run(tmp_path, seed=142, model="1b")]
    output = tmp_path / "annotations.csv"
    rows = export_annotations(paths, output)
    assert {row["seed"] for row in rows} == {42,142}
    assert {row["model"] for row in rows} == {"170m","1b"}
    assert len({row["run_id"] for row in rows}) == 2
    assert all(row["numerical_diagnostic"] == "state_residual_largest" for row in rows)
    assert len(list(csv.DictReader(output.open()))) == 2
    with pytest.raises(FileExistsError):
        export_annotations(paths, output)
    with pytest.raises(ValueError, match="Duplicate"):
        export_annotations([paths[0], paths[0]], tmp_path / "duplicates.csv")
    assert not (tmp_path / "duplicates.csv").exists()
    assert categorize_residuals(dict(vision_err=None,lang_err=0,state_err=0)) == "undefined_relative_residual"


def collection_args(output):
    return ["--task","PickCube-v1","--model","170m","--episodes","1","--max-policy-calls","1",
        "--m","8","--seed-base","42","--target","logpi","--quadrature","trapezoid",
        "--lang-dir","a path with spaces","--out",str(output)]


def test_wrapper_requires_explicit_scope_protocol_and_fresh_destination(tmp_path):
    decision = tmp_path / "protocol.json"
    decision.write_text(json.dumps({"decision_id":"CPU-fixture"}))
    out = tmp_path / "new metrics.jsonl"
    arguments = collection_args(out)
    result = plan("collect", arguments, decision)
    assert result["command"][-3].endswith("a path with spaces")
    assert result["command"][-2:] == ["--out",str(out)]
    with pytest.raises(ValueError, match="episodes"):
        plan("collect", [a for a in arguments if a != "--episodes"], decision)
    with pytest.raises(ValueError, match="preserved"):
        plan("collect", collection_args(ROOT / "data/new.jsonl"), decision)
    out.write_text("historical")
    with pytest.raises(FileExistsError):
        plan("collect", arguments, decision)
    assert out.read_text() == "historical"


def test_wrapper_cli_preserves_arguments_with_spaces(tmp_path):
    decision = tmp_path / "protocol file.json"
    decision.write_text(json.dumps({"decision_id":"CPU-fixture"}))
    command = [sys.executable,str(ROOT / "scripts/run_workflow.py"),"collect","--decision-file",str(decision),
               "--dry-run","--",*collection_args(tmp_path / "new metrics.jsonl")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert any(value.endswith("a path with spaces") for value in json.loads(result.stdout)["command"])
    assert not (tmp_path / "new metrics.jsonl").exists()


def test_audit_cli_exits_nonzero_on_missing_integrity_artifacts(tmp_path):
    result = subprocess.run([sys.executable,str(ROOT / "audit.py"),str(tmp_path)], capture_output=True)
    assert result.returncode != 0


def test_existing_plot_cannot_be_overwritten(tmp_path):
    path = tmp_path / "historical.png"
    path.write_bytes(b"original")
    fig, _ = plt.subplots()
    try:
        with pytest.raises(FileExistsError):
            save_new_figure(fig, path)
    finally:
        plt.close(fig)
    assert path.read_bytes() == b"original"
