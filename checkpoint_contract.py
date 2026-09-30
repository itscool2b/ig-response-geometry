"""Validate checkpoint keys and shapes before changing any model parameter."""
from collections.abc import Mapping
from pathlib import Path
import torch
from experiment_io import file_hash


def normalized_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError('Checkpoint must be a state dictionary or module wrapper')
    raw = checkpoint.get('module', checkpoint)
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError('Empty or invalid state dictionary')
    result = {}
    for original, value in raw.items():
        if not isinstance(original, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f'Invalid state entry: {original!r}')
        key = original.removeprefix('module.')
        if key in result:
            raise ValueError(f'Prefix normalization collision: {key}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite checkpoint tensor: {key}')
        result[key] = value
    return result


def validate_and_load(model, checkpoint, *, adapter_only=False):
    state = normalized_state_dict(checkpoint)
    expected = model.state_dict()
    selected = {key: value for key, value in expected.items()
                if not adapter_only or '.lora_' in key}
    if not selected:
        raise ValueError('No expected parameters for the requested checkpoint mode')
    missing = sorted(set(selected) - set(state))
    unexpected = sorted(set(state) - set(selected))
    shapes = [key for key in set(selected) & set(state) if state[key].shape != selected[key].shape]
    if missing or unexpected or shapes:
        raise ValueError(f'Checkpoint contract failed: missing={missing[:8]}, unexpected={unexpected[:8]}, shapes={shapes[:8]}')
    for key, value in state.items():
        destination = selected[key]
        if value.is_complex() and not destination.is_complex():
            raise ValueError(f'Complex-to-real checkpoint conversion is forbidden: {key}')
        if not torch.isfinite(value.to(dtype=destination.dtype)).all():
            raise ValueError(f'Checkpoint overflows destination dtype: {key}')
    model.load_state_dict(state, strict=not adapter_only)
    return {'loaded_keys': len(state), 'missing_keys': [], 'unexpected_keys': [],
            'adapter_only': adapter_only}


def load_checkpoint(model, path, *, adapter_only=False):
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if adapter_only:
        if not isinstance(checkpoint, Mapping) or 'lora_state_dict' not in checkpoint:
            raise ValueError('Explicit LoRA mode requires lora_state_dict')
        checkpoint = checkpoint['lora_state_dict']
    result = validate_and_load(model, checkpoint, adapter_only=adapter_only)
    return {**result, 'sha256': file_hash(path), 'filename': path.name}
