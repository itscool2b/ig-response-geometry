"""Explicit-noise differentiable RDT sampling with a fresh scheduler per call.

Adapted from models/rdt_runner.py at thu-ml/RoboticsDiffusionTransformer
commit cd79363a1387e8f81c7724d070ef7e45fd23150f. The state/adaptor/model/solver
ordering and per-step dtype cast are preserved. Changes: noise is supplied,
the scheduler is local, inputs and finite arithmetic are checked, and no
process-global random state is reseeded. This is a versioned project adapter,
not a patch to an unknown installed upstream checkout.

Original sampler license:
MIT License
Copyright (c) 2024 TSAIL group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from copy import deepcopy
import math
import numbers

import torch

from experiment_io import tensor_hash
from integrated_gradients import _require_finite


UPSTREAM_COMMIT = "cd79363a1387e8f81c7724d070ef7e45fd23150f"
SAMPLER_ADAPTER_VERSION = "explicit_noise_local_scheduler_v1"


def _config_record(value):
    """Encode legal infinite scheduler settings explicitly in strict JSON.

    Diffusers uses lambda_min_clipped=-inf to mean no clipping. A tagged value
    preserves that setting without emitting invalid JSON Infinity constants.
    This encoding is for metadata only, not passed to the scheduler constructor.
    """
    if isinstance(value, dict):
        return {
            str(key): _config_record(sorted(item) if key == "_use_default_values" else item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_config_record(item) for item in value]
    if isinstance(value, numbers.Real) and not isinstance(value, (bool, numbers.Integral)):
        if not math.isfinite(value):
            return {"__float__": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
        return float(value)
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if hasattr(value, "tolist"):
        return _config_record(value.tolist())
    raise TypeError(f"Unsupported scheduler configuration value: {type(value).__name__}")


def make_initial_noise(runner, state_traj, seed):
    """Draw once from a private generator, saving low precision values in fp32.

    The generator uses the model device and dtype, matching the historical draw
    convention for that runtime. The saved tensor, not the seed alone, defines
    noise identity across runtimes. It is not recovery of historical noise.
    """
    generator = torch.Generator(device=state_traj.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        (state_traj.shape[0], runner.pred_horizon, runner.action_dim),
        device=state_traj.device, dtype=state_traj.dtype, generator=generator,
    )
    return noise.float() if noise.dtype in (torch.bfloat16, torch.float16) else noise


def sampler_metadata(runner, initial_noise):
    """Information required to identify this adapter and its solver settings."""
    scheduler = runner.noise_scheduler_sample
    return {
        "adapter_version": SAMPLER_ADAPTER_VERSION,
        "upstream_reference_commit": UPSTREAM_COMMIT,
        "scheduler_class": f"{type(scheduler).__module__}.{type(scheduler).__name__}",
        "scheduler_config": _config_record(dict(scheduler.config)),
        "solver_steps": runner.num_inference_timesteps,
        "initial_noise_sha256": tensor_hash(initial_noise),
        "initial_noise_shape": list(initial_noise.shape),
        "initial_noise_storage_dtype": str(initial_noise.dtype),
    }


def conditional_sample_with_noise(
    runner, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask,
    ctrl_freqs, initial_noise,
):
    """Run the pinned RDT sampling algorithm without consuming a random stream.

    Only deterministic DPM-Solver variants are supported. A stochastic solver
    would require additional explicitly saved per-step draws. Supplied noise
    must be exactly representable in the model dtype, so silently rounding a
    saved fp32 draw cannot masquerade as the same-noise precision comparison.
    The caller's noise and runner scheduler are never mutated.
    """
    if getattr(runner, "training", False):
        raise ValueError("Explicit-noise attribution requires a runner in evaluation mode")
    count = runner.num_inference_timesteps
    if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 1:
        raise ValueError("num_inference_timesteps must be a positive integer")
    config = deepcopy(dict(runner.noise_scheduler_sample.config))
    if config.get("algorithm_type") not in {"dpmsolver", "dpmsolver++"}:
        raise ValueError("Explicit-noise sampling requires a deterministic DPM-Solver algorithm")

    batch = state_traj.shape[0]
    expected_shape = (batch, runner.pred_horizon, runner.action_dim)
    if tuple(initial_noise.shape) != expected_shape or not initial_noise.is_floating_point():
        raise ValueError(f"initial_noise must be a floating point tensor of shape {expected_shape}")
    if tuple(action_mask.shape) != (batch, 1, runner.action_dim):
        raise ValueError("action_mask must have shape (batch, 1, action_dim)")
    if tuple(ctrl_freqs.shape) != (batch,):
        raise ValueError("ctrl_freqs must have shape (batch,)")
    if lang_attn_mask.dtype != torch.bool or tuple(lang_attn_mask.shape) != tuple(lang_cond.shape[:2]):
        raise ValueError("lang_attn_mask must be boolean with shape (batch, language_tokens)")
    if state_traj.ndim != 3 or state_traj.shape[1] != 1:
        raise ValueError("state_traj must have shape (batch, 1, hidden_size)")
    if lang_cond.ndim != 3 or img_cond.ndim != 3 or lang_cond.shape[0] != batch or img_cond.shape[0] != batch:
        raise ValueError("language and image conditions must be rank-three tensors with the same batch")
    for name, tensor in (
        ("language", lang_cond), ("image", img_cond), ("state", state_traj),
        ("action_mask", action_mask), ("ctrl_freqs", ctrl_freqs),
    ):
        if tensor.device != state_traj.device or tensor.dtype != state_traj.dtype:
            raise ValueError(f"{name} must have the model state device and dtype")
        _require_finite(tensor, f"sampler_{name}")
    if lang_attn_mask.device != state_traj.device:
        raise ValueError("lang_attn_mask must be on the model state device")
    if not bool(((action_mask == 0) | (action_mask == 1)).all()):
        raise ValueError("action_mask must contain only zeros and ones")
    _require_finite(initial_noise, "initial_noise")
    noisy_action = initial_noise.detach().to(device=state_traj.device, dtype=state_traj.dtype).clone()
    if not torch.equal(noisy_action.to(device=initial_noise.device, dtype=initial_noise.dtype), initial_noise):
        raise ValueError("initial_noise is not exactly representable in the model dtype")
    mask = action_mask.expand(-1, runner.pred_horizon, -1)
    # from_config creates clean solver history even if the runner's scheduler
    # has previously sampled, or if an earlier attribution raised an error.
    scheduler = type(runner.noise_scheduler_sample).from_config(config)
    scheduler.set_timesteps(count)
    for step_index, timestep in enumerate(scheduler.timesteps):
        action_tokens = runner.state_adaptor(torch.cat([noisy_action, mask], dim=2))
        state_action = torch.cat([state_traj, action_tokens], dim=1)
        model_output = runner.model(
            state_action, ctrl_freqs, timestep.unsqueeze(-1).to(state_traj.device),
            lang_cond, img_cond, lang_mask=lang_attn_mask,
        )
        _require_finite(model_output, "sampler_model_output", step=step_index)
        noisy_action = scheduler.step(model_output, timestep, noisy_action).prev_sample
        noisy_action = noisy_action.to(state_traj.dtype)
        _require_finite(noisy_action, "sampler_step", step=step_index)
    result = noisy_action * mask
    _require_finite(result, "sampler_result")
    return result
