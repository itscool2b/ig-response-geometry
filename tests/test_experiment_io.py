import json
from pathlib import Path

import pytest
import torch
from filelock import Timeout

from checkpoint_contract import normalized_state_dict, validate_and_load
from experiment_io import RunStore, tensor_hash


def commit_episode(store, episode=0):
    transaction = store.begin_episode(episode)
    row = transaction.save_step({'event': 'step', 'episode': episode, 'policy_call_idx': 0},
                                {'attribution': torch.ones(3)})
    transaction.commit({'event': 'episode_end', 'episode': episode, 'policy_calls': 1})
    return row


def test_partial_attempt_is_preserved_and_never_duplicates_completed_records(tmp_path):
    out = tmp_path / 'metrics.jsonl'
    store = RunStore(out, {'m': 64})
    with store.writer():
        partial = store.begin_episode(0)
        partial.save_step({'event': 'step', 'episode': 0, 'policy_call_idx': 0}, {'x': torch.ones(1)})
        partial.fail('injected interruption')
    with RunStore(out, {'m': 64}, resume=True).writer() as resumed:
        assert resumed.completed_episodes() == set()
        commit_episode(resumed)
        assert resumed.completed_episodes() == {0}
    assert partial.directory.joinpath('failure.json').exists()
    assert len(out.read_text().splitlines()) == 2
    with RunStore(out, {'m': 64}, resume=True).writer() as resumed:
        with pytest.raises(FileExistsError):
            resumed.begin_episode(0)


def test_truncated_export_recovers_from_verified_transactions(tmp_path):
    out = tmp_path / 'metrics.jsonl'
    with RunStore(out, {'m': 64}).writer() as store:
        commit_episode(store)
    correct = out.read_bytes()
    out.write_bytes(b'{"truncated":')
    with RunStore(out, {'m': 64}, resume=True).writer() as store:
        store.export()
    assert out.read_bytes() == correct
    backups = list(store.root.joinpath('attempts').glob('export-*.jsonl'))
    assert any(path.read_bytes() == b'{"truncated":' for path in backups)


def test_resume_rejects_budget_or_checkpoint_change(tmp_path):
    out = tmp_path / 'metrics.jsonl'
    with RunStore(out, {'m': 64, 'checkpoint': 'a'}).writer():
        pass
    for config in [{'m': 128, 'checkpoint': 'a'}, {'m': 64, 'checkpoint': 'b'}]:
        with pytest.raises(ValueError, match='configuration mismatch'):
            with RunStore(out, config, resume=True).writer():
                pass


def test_historical_and_existing_runs_are_not_overwritten(tmp_path):
    out = tmp_path / 'metrics.jsonl'
    out.write_bytes(b'old raw bytes\n')
    with pytest.raises(FileExistsError):
        with RunStore(out, {}, resume=True).writer():
            pass
    assert out.read_bytes() == b'old raw bytes\n'


@pytest.mark.parametrize('damage', ['delete', 'modify'])
def test_completed_sidecar_must_exist_and_match_hash(tmp_path, damage):
    out = tmp_path / 'metrics.jsonl'
    with RunStore(out, {}).writer() as store:
        row = commit_episode(store)
    sidecar = tmp_path / row['attr_file']
    if damage == 'delete':
        sidecar.unlink()
    else:
        sidecar.write_bytes(b'changed')
    with pytest.raises(ValueError, match='sidecar|Sidecar'):
        with RunStore(out, {}, resume=True).writer():
            pass


def test_nonfinite_metadata_rejected_before_creation(tmp_path):
    with pytest.raises(ValueError):
        RunStore(tmp_path / 'metrics.jsonl', {'x': float('nan')})
    assert not list(tmp_path.iterdir())


def test_conflicting_step_and_cross_run_sidecar_rejected(tmp_path):
    out = tmp_path / 'metrics.jsonl'
    with RunStore(out, {}).writer() as store:
        transaction = store.begin_episode(0)
        transaction.save_step({'event': 'step', 'episode': 0, 'policy_call_idx': 0}, {'x': torch.ones(1)})
        with pytest.raises(ValueError):
            transaction.save_step({'event': 'step', 'episode': 0, 'policy_call_idx': 0}, {})
        transaction.records[0]['attr_file'] = '../foreign.pt'
        with pytest.raises(ValueError):
            transaction.commit({'event': 'episode_end', 'episode': 0, 'policy_calls': 1})


def test_tensor_identity_includes_dtype_shape_and_bytes():
    a = torch.ones(2, dtype=torch.bfloat16)
    assert tensor_hash(a) == tensor_hash(a.clone())
    assert len({tensor_hash(a), tensor_hash(a.float()), tensor_hash(a.reshape(1, 2)), tensor_hash(a + 1)}) == 4


def test_mixed_prefix_is_normalized_per_key():
    model = torch.nn.Linear(2, 1)
    state = {'module.weight': torch.ones(1, 2), 'bias': torch.zeros(1)}
    validate_and_load(model, state)
    torch.testing.assert_close(model(torch.ones(2)), torch.tensor([2.0]))


def test_prefix_collision_is_rejected():
    with pytest.raises(ValueError, match='collision'):
        normalized_state_dict({'module.weight': torch.ones(1), 'weight': torch.ones(1)})


@pytest.mark.parametrize('damage', ['missing', 'unexpected', 'shape', 'nonfinite'])
def test_invalid_checkpoint_cannot_partially_mutate_model(damage):
    model = torch.nn.Linear(2, 1)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    state = {key: torch.zeros_like(value) for key, value in original.items()}
    if damage == 'missing':
        del state['bias']
    elif damage == 'unexpected':
        state['extra'] = torch.ones(1)
    elif damage == 'shape':
        state['weight'] = torch.ones(1, 3)
    else:
        state['weight'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        validate_and_load(model, state)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])
