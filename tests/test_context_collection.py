import json

import pytest
import torch
from PIL import Image

from collect_contexts import collect_episode, selected_stratum
from experiment_io import RunStore
from faithfulness import authenticated_source, load_sidecar


class Environment:
    def __init__(self, stop_after=None):
        self.stop_after = stop_after
        self.steps = 0

    def reset(self, seed):
        self.seed, self.steps = seed, 0
        return {'agent': {'qpos': torch.zeros(1, 9)}}, {'success': False}

    def step(self, action):
        assert action.shape == (1, 8)
        self.steps += 1
        truncated = self.stop_after is not None and self.steps >= self.stop_after
        return {'agent': {'qpos': torch.ones(1, 9) * self.steps}}, 0, False, truncated, {'success': False}


def prepare(image, proprio, seed):
    return dict(initial_noise=torch.zeros(1, 64, 128), ref_action=torch.zeros(1, 64, 128, dtype=torch.bfloat16),
                lang_attn_mask=torch.ones(1, 2, dtype=torch.bool), action_mask=torch.ones(1, 1, 128),
                ctrl_freqs=torch.tensor([25.]), sampler_metadata={'seed': seed})


def stratum():
    return dict(id='example', task='PickCube-v1', model='170m', checkpoint_mode='pretrained',
                episodes=1, seed_base=900, max_policy_calls=3, max_episode_steps=400,
                evaluate_calls=[0, 2])


def test_all_calls_preserved_but_only_prespecified_indices_selected(tmp_path):
    output = tmp_path / 'metrics.jsonl'
    env = Environment()
    with RunStore(output, {'solver_steps': 5}).writer() as store:
        end = collect_episode(env, store, 0, stratum(), prepare,
                              render=lambda env: Image.new('RGB', (2, 2), 'gray'))
    rows, manifest = authenticated_source(output)
    assert len(rows) == 3 and [r['selected_for_analysis'] for r in rows] == [True, False, True]
    assert end['selected_calls_present'] == [0, 2]
    assert end['selected_calls_unavailable'] == []
    assert end['env_steps'] == 48 and end['stop_reason'] == 'protocol_call_limit'
    assert env.seed == 900
    for row in rows:
        payload, _ = load_sidecar(row, output, manifest)
        assert payload['collector_type'] == 'context_only'
        assert 'vision_attr' not in payload


def test_early_termination_keeps_missing_selected_calls_in_denominator(tmp_path):
    env = Environment(stop_after=7)
    with RunStore(tmp_path / 'metrics.jsonl', {'solver_steps': 5}).writer() as store:
        end = collect_episode(env, store, 0, stratum(), prepare,
                              render=lambda env: Image.new('RGB', (2, 2)))
    assert end['selected_calls_present'] == [0]
    assert end['selected_calls_unavailable'] == [2]
    assert end['policy_calls'] == 1 and end['env_steps'] == 7
    assert end['stop_reason'] == 'truncated'


def test_failure_retains_partial_attempt_without_committing_episode(tmp_path):
    invocations = []

    def fail_second(*args):
        invocations.append(1)
        if len(invocations) == 2:
            raise ValueError('reproducible context failure')
        return prepare(*args)

    with RunStore(tmp_path / 'metrics.jsonl', {'solver_steps': 5}).writer() as store:
        with pytest.raises(ValueError, match='context failure'):
            collect_episode(Environment(), store, 0, stratum(), fail_second,
                            render=lambda env: Image.new('RGB', (2, 2)))
        assert store.completed_episodes() == set()
        failure = json.loads(next((store.root / 'attempts').glob('ep*/failure.json')).read_text())
        assert failure['completed_calls'] == 1
        assert len(list((store.root / 'attempts').glob('ep*/call*.pt'))) == 1


@pytest.mark.parametrize('changes', [dict(episodes=True), dict(seed_base=-1), dict(evaluate_calls=[2, 0]),
                                    dict(evaluate_calls=[0, 0]), dict(evaluate_calls=[3]), dict(checkpoint_mode=None)])
def test_invalid_or_ambiguous_sampling_protocol_rejected(changes):
    row = {**stratum(), **changes}
    with pytest.raises(ValueError):
        selected_stratum({'decision_id': 'E01', 'strata': [row]}, 'example')


def test_missing_or_duplicate_stratum_rejected():
    row = stratum()
    with pytest.raises(ValueError):
        selected_stratum({'decision_id': 'E01', 'strata': [row, row]}, 'example')


def test_separate_policy_stream_is_saved_and_step_cap_is_enforced(tmp_path):
    spec = {**stratum(), 'policy_seed_base': 1800, 'max_episode_steps': 19}
    assert selected_stratum({'decision_id': 'E02', 'strata': [spec]}, 'example') == spec
    seen = []
    def capture(image, proprio, seed):
        seen.append(seed)
        return prepare(image, proprio, seed)
    output = tmp_path / 'metrics.jsonl'
    env = Environment()  # This fixture does not implement its own step cap.
    with RunStore(output, {'solver_steps': 5}).writer() as store:
        end = collect_episode(env, store, 0, spec, capture,
                              render=lambda env: Image.new('RGB', (2, 2)))
    rows, manifest = authenticated_source(output)
    assert env.seed == 900 and seen == [1800, 1800]
    assert end['env_steps'] == 19 and end['policy_calls'] == 2
    assert end['stop_reason'] == 'collector_environment_step_cap'
    assert end['collector_step_cap'] and not end['truncated']
    assert end['policy_seed'] == 1800
    for row in rows:
        payload, _ = load_sidecar(row, output, manifest)
        assert row['seed'] == 900 and row['policy_seed'] == payload['policy_seed'] == 1800


@pytest.mark.parametrize('base', [900, True, -1, 2**32])
def test_overlapping_or_invalid_policy_stream_rejected(base):
    with pytest.raises(ValueError):
        selected_stratum({'decision_id': 'E02', 'strata': [{**stratum(), 'policy_seed_base': base}]}, 'example')


def test_seed_stream_overlap_between_strata_rejected():
    first = {**stratum(), 'episodes': 3, 'policy_seed_base': 1800}
    second = {**stratum(), 'id': 'other', 'seed_base': 1801, 'policy_seed_base': 2800}
    with pytest.raises(ValueError, match='overlap'):
        selected_stratum({'decision_id': 'E02', 'strata': [first, second]}, 'example')
