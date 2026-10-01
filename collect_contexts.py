"""Collect a protocol-selected context bank without computing IG maps.

Every executed policy call has a small immutable sidecar and record. Analysis
selection is declared before collection and never changes the policy rollout.
Early termination, failed attempts and full episode denominators are preserved.
"""
import argparse
import importlib.metadata
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image

from experiment_io import RunStore, file_hash, object_hash, strict_json, tensor_hash
from per_step_attribution import MANISKILL_INDICES, prepare_ig_context
from per_step_ig import ACTION_MAX, ACTION_MIN
from pipeline import load_lang, load_pipeline


def selected_stratum(protocol, stratum_id):
    if not isinstance(protocol.get('decision_id'), str) or not protocol['decision_id']:
        raise ValueError('A recorded experiment decision is required')
    matches = [row for row in protocol.get('strata', []) if row.get('id') == stratum_id]
    if len(matches) != 1:
        raise ValueError('Stratum must identify exactly one recorded configuration')
    row = matches[0]
    for name in ('episodes', 'max_policy_calls', 'max_episode_steps'):
        if type(row.get(name)) is not int or row[name] < 1:
            raise ValueError(f'Invalid positive protocol field: {name}')
    if type(row.get('seed_base')) is not int or row['seed_base'] < 0:
        raise ValueError('Invalid episode seed base')
    calls = row.get('evaluate_calls')
    if not isinstance(calls, list) or not calls or any(type(c) is not int or c < 0 or c >= row['max_policy_calls'] for c in calls):
        raise ValueError('Selected calls must be valid nonnegative policy-call indices')
    if calls != sorted(set(calls)):
        raise ValueError('Selected calls must be distinct and increasing')
    if row.get('model') not in {'170m', '1b'} or not isinstance(row.get('task'), str):
        raise ValueError('Invalid model/task protocol')
    if row.get('checkpoint_mode') not in {'pretrained', 'authors', 'lora'}:
        raise ValueError('An explicit checkpoint mode is required')
    return row


def render_observation(env):
    frame = env.render().squeeze(0).detach().cpu().numpy()
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError('Expected one uint8 RGB simulator frame')
    return Image.fromarray(frame).resize((384, 384))


def collect_episode(env, store, episode, stratum, prepare, render=render_observation):
    transaction = store.begin_episode(episode)
    seed = stratum['seed_base'] + episode
    selected = set(stratum['evaluate_calls'])
    try:
        observation, info = env.reset(seed=seed)
        terminated = truncated = False
        env_steps = 0
        calls = 0
        while not (terminated or truncated) and calls < stratum['max_policy_calls']:
            started = time.perf_counter()
            image = render(env)
            proprio = observation['agent']['qpos'][0, :8].detach().cpu()
            with torch.no_grad():
                context = prepare(image, proprio, seed)
            payload = {key: context[key].detach().cpu() for key in
                       ('initial_noise', 'ref_action', 'lang_attn_mask', 'action_mask', 'ctrl_freqs')}
            payload.update(obs_image=torch.from_numpy(np.asarray(image).copy()), proprio=proprio,
                           sampler_metadata=context['sampler_metadata'], collector_type='context_only')
            payload['initial_noise_sha256'] = tensor_hash(payload['initial_noise'])
            context_id = object_hash({'episode': episode, 'policy_call': calls,
                'configuration': store.identity, 'observation': tensor_hash(payload['obs_image']),
                'proprio': tensor_hash(proprio), 'noise': payload['initial_noise_sha256'],
                'reference': tensor_hash(payload['ref_action'])})
            payload['context_id'] = context_id
            transaction.save_step(dict(event='step', task=stratum['task'], model=stratum['model'],
                episode=episode, seed=seed, policy_call_idx=calls, env_step_at_call=env_steps,
                solver_steps=store.configuration['solver_steps'], target='logpi',
                collector_type='context_only', selected_for_analysis=calls in selected,
                ref_norm_maniskill=context['ref_action'][..., MANISKILL_INDICES].float().norm().item(),
                context_id=context_id, initial_noise_sha256=payload['initial_noise_sha256'],
                wall_seconds=time.perf_counter()-started), payload)
            reference = payload['ref_action'][0].float()[:, MANISKILL_INDICES]
            actions = ((reference + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN)[::4].numpy()
            calls += 1
            for action in actions:
                observation, _, terminated, truncated, info = env.step(action.reshape(1, 8))
                env_steps += 1
                if terminated or truncated:
                    break
            del context, payload
        terminal = dict(event='episode_end', task=stratum['task'], model=stratum['model'],
            episode=episode, seed=seed, env_steps=env_steps, policy_calls=calls,
            success=bool(info.get('success', False)), terminated=bool(terminated), truncated=bool(truncated),
            stop_reason='terminated' if terminated else 'truncated' if truncated else 'protocol_call_limit',
            selected_calls_present=sorted(c for c in selected if c < calls),
            selected_calls_unavailable=sorted(c for c in selected if c >= calls))
        transaction.commit(terminal)
        return terminal
    except Exception as error:
        transaction.fail(error, getattr(error, 'diagnostics', {}))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--stratum', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lang-dir', required=True)
    parser.add_argument('--checkpoint-path')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    protocol = strict_json(args.protocol)
    stratum = selected_stratum(protocol, args.stratum)
    pipe = load_pipeline(stratum['model'], enable_checkpoint=False,
        solver_steps=protocol['solver_steps'], checkpoint_mode=stratum['checkpoint_mode'],
        checkpoint_path=args.checkpoint_path, model_revision=stratum.get('model_revision'),
        vision_revision=protocol['vision_revision'])
    language = load_lang(stratum['task'], args.lang_dir)
    if stratum.get('checkpoint_sha256') and pipe['identity']['checkpoint']['sha256'] != stratum['checkpoint_sha256']:
        raise ValueError('Loaded checkpoint differs from the recorded stratum')
    source_files = ['collect_contexts.py', 'per_step_ig.py', 'per_step_attribution.py',
                    'pipeline.py', 'checkpoint_contract.py', 'experiment_io.py', 'rdt_sampling.py',
                    'integrated_gradients.py']
    config = dict(schema_version=1, collector_type='context_only', task=stratum['task'], model=stratum['model'],
        target='logpi', m=None, quadrature=None, pipeline=pipe['identity'], language=language['identity'],
        solver_steps=protocol['solver_steps'], episodes=stratum['episodes'], seed_base=stratum['seed_base'],
        max_policy_calls=stratum['max_policy_calls'], max_episode_steps=stratum['max_episode_steps'],
        protocol_sha256=file_hash(args.protocol), protocol=protocol, stratum_id=args.stratum,
        call_selection={'rule': 'prespecified_indices_with_all_executed_calls_preserved',
                        'evaluate_calls': stratum['evaluate_calls'], 'missing_rule': 'record_unavailable_no_replacement'},
        control_mode='pd_joint_pos', action_subsampling=4,
        observation_pipeline='one_current_external_camera_five_background_slots',
        noise_policy='fixed_episode_seed_stored_per_context', forward_dtype='torch.bfloat16',
        source_sha256={name: file_hash(Path(__file__).parent / name) for name in source_files},
        environment={name: importlib.metadata.version(name) for name in
                     ['torch', 'numpy', 'diffusers', 'transformers', 'mani_skill', 'sapien']})
    store = RunStore(args.out, config, resume=args.resume)

    def prepare(image, proprio, seed):
        return prepare_ig_context(pipe['runner'], pipe['vision_model'], image, proprio,
            language['lang_tokens'], language['lang_attn_mask'], language['lang_tokens_baseline'],
            pipe['bg_image_encoded'], pipe['img_tokens_baseline'], pipe['action_mask'], pipe['ctrl_freqs'],
            seed=seed, target='logpi')

    import gymnasium as gym
    import mani_skill.envs
    env = gym.make(stratum['task'], obs_mode='state_dict', num_envs=1, control_mode='pd_joint_pos',
                   render_mode='rgb_array', max_episode_steps=stratum['max_episode_steps'])
    try:
        with store.writer():
            completed = store.completed_episodes()
            for episode in range(stratum['episodes']):
                if episode in completed:
                    continue
                terminal = collect_episode(env, store, episode, stratum, prepare)
                print(f"{args.stratum}: episode {episode}, calls={terminal['policy_calls']}, selected={terminal['selected_calls_present']}, stop={terminal['stop_reason']}", flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
