"""Bounded CUDA, graphics, simulator and durable-output engineering probe."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import gymnasium as gym
    import mani_skill.envs
    start = time.perf_counter()
    env = gym.make('PickCube-v1', obs_mode='state_dict', num_envs=1,
                   control_mode='pd_joint_pos', render_mode='rgb_array', max_episode_steps=400)
    try:
        obs, info = env.reset(seed=910001)
        frame = env.render().squeeze(0).detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(frame).save(args.output_dir / 'initial-observation.png')
        # New initial state selected before attribution; not historical recovery.
        torch.save({'task': 'PickCube-v1', 'simulator_seed': 910001,
                    'proprio': obs['agent']['qpos'][0, :8].cpu(),
                    'image': torch.from_numpy(frame)}, args.output_dir / 'initial-context.pt')
        x = torch.ones(32, device='cuda', requires_grad=True)
        x.square().sum().backward()
        torch.cuda.synchronize()
        assert torch.equal(x.grad, torch.full_like(x, 2))
        result = {'purpose': 'engineering probe, not scientific confirmation',
                  'python': platform.python_version(), 'torch': torch.__version__,
                  'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
                  'device_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                  'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
                  'versions': {name: importlib.metadata.version(name) for name in
                               ['mani_skill', 'sapien', 'diffusers', 'transformers', 'timm', 'numpy']},
                  'frame_shape': list(frame.shape), 'finite_proprio': bool(torch.isfinite(obs['agent']['qpos']).all()),
                  'elapsed_seconds': time.perf_counter() - start,
                  'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output_dir.iterdir()}}
        (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
