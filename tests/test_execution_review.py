"""Independent fault regressions from the execution-integrity review."""

import json

import pytest
import torch

from checkpoint_contract import validate_and_load
from experiment_io import RunStore, object_hash


def test_resume_rebuilds_export_when_every_episode_already_committed(tmp_path):
    output = tmp_path / "metrics.jsonl"
    with RunStore(output, {"m": 64}).writer() as store:
        transaction = store.begin_episode(0)
        transaction.save_step(
            {"event": "step", "episode": 0, "policy_call_idx": 0},
            {"attribution": torch.ones(1)},
        )
        transaction.commit({"event": "episode_end", "episode": 0, "policy_calls": 1})
    expected = output.read_bytes()
    output.write_bytes(b'{"interrupted_export":')
    # Mirrors the collection CLI when the full requested episode set is done.
    # It must repair the derived export without collecting a new episode.
    with RunStore(output, {"m": 64}, resume=True).writer() as resumed:
        assert resumed.completed_episodes() == {0}
    assert output.read_bytes() == expected


@pytest.mark.parametrize("target_dtype", [torch.float32, torch.bfloat16])
def test_finite_checkpoint_cannot_overflow_destination_parameters(target_dtype):
    model = torch.nn.Linear(2, 1, dtype=target_dtype)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint = {
        "weight": torch.full((1, 2), 1e100, dtype=torch.float64),
        "bias": torch.zeros(1, dtype=torch.float64),
    }
    with pytest.raises(ValueError):
        validate_and_load(model, checkpoint)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_complex_checkpoint_cannot_silently_discard_imaginary_parameters():
    model = torch.nn.Linear(2, 1)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint = {
        "weight": torch.full((1, 2), 1 + 2j),
        "bias": torch.zeros(1, dtype=torch.complex64),
    }
    with pytest.raises(ValueError):
        validate_and_load(model, checkpoint)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_run_identity_snapshots_configuration_before_caller_mutation(tmp_path):
    config = {"m": 64, "model": {"checkpoint": "original"}}
    store = RunStore(tmp_path / "metrics.jsonl", config)
    config["model"]["checkpoint"] = "changed"
    with store.writer():
        assert store.configuration["model"]["checkpoint"] == "original"
        manifest = json.loads((store.root / "manifest.json").read_text())
        assert object_hash(manifest["configuration"]) == manifest["configuration_sha256"]
