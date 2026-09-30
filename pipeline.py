"""
Shared RDT+SigLIP pipeline loader for faithfulness.py and sanity.py.

The model-loading block in per_step_ig.py is forked out here so downstream
post-processing scripts can reuse the same setup without re-reading that code.

What this loads (and what the caller gets back via `load_pipeline`):

- SigLIP vision tower (bf16 on GPU, stays loaded)
- `bg_image_encoded`    (1, 729, 1152) SigLIP tokens for one background slot
- `img_tokens_baseline` (1, 4374, 1152) full 6-slot baseline [bg,bg,bg,gray,bg,bg]
- RDTRunner backbone with pretrained weights loaded, frozen, optionally LoRA
  (for `1b`) and optionally gradient-checkpointed on every DiT block
- `action_mask`         (1, 1, 128) bf16, 1 at MANISKILL_INDICES else 0
- `ctrl_freqs`          (1,) bf16, the RDT control-frequency tensor

`load_lang` loads per-task T5-XXL language embeddings from disk and the [PAD, EOS]
two-token baseline (the baseline_bos_eos.pt file name is legacy; T5 has no BOS token);
kept separate because faithfulness and sanity iterate over tasks.
"""

import os
import sys
import subprocess
from pathlib import Path

import torch
import yaml
from PIL import Image

from checkpoint_contract import load_checkpoint
from experiment_io import file_hash, object_hash

from per_step_attribution import MANISKILL_INDICES

CONTROL_FREQ = 25


def load_pipeline(model="170m", enable_checkpoint=True, solver_steps=None, *,
                  checkpoint_mode=None, checkpoint_path=None, model_revision=None,
                  vision_revision=None, device="cuda", dtype=torch.bfloat16):
    """
    Load the static (task-independent) pieces of the per-step pipeline.

    Args:
        model: "170m" or "1b". The 1B path requires an explicit pretrained,
            authors or lora checkpoint mode. File existence never chooses it.
        enable_checkpoint: wrap every RDT DiT block forward in
            torch.utils.checkpoint.checkpoint(use_reentrant=False). Required
            on 12 GB cards; 24 GB can run with it off for ~1.4x speedup.
        solver_steps: Month 5 override of the DPM-Solver++ step count
            (num_inference_timesteps). None keeps the config value (5). Used by
            displacement.py to measure the action displacement as a function of
            T (that measurement describes solver-resolution sensitivity; see the
            displacement.py docstring), and by faithfulness re-runs that must
            match a non-default T.

    Returns dict with keys: runner, vision_model, bg_image_encoded,
    img_tokens_baseline, action_mask, ctrl_freqs, config.
    """
    source = Path(os.environ.get("RDT_SOURCE", os.path.expanduser("~/rdt-repo"))).resolve()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != "cd79363a1387e8f81c7724d070ef7e45fd23150f":
        raise ValueError("Unvalidated upstream revision; use the pinned RDT_SOURCE")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if dirty.strip():
        raise ValueError("Upstream source has unrecorded modifications")
    sys.path.insert(0, str(source))
    from models.rdt_runner import RDTRunner
    from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
    from configs.state_vec import STATE_VEC_IDX_MAPPING
    for name in ('models.rdt_runner', 'models.multimodal_encoder.siglip_encoder', 'configs.state_vec'):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(source):
            raise ValueError(f'Preloaded upstream module is outside RDT_SOURCE: {name}')
    expected_indices = [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
    if expected_indices != MANISKILL_INDICES:
        raise ValueError("Upstream action mapping differs from the validated contract")
    if model not in {"170m", "1b"}:
        raise ValueError("Unknown model size")
    if checkpoint_mode is None:
        if model == "1b":
            raise ValueError("1B requires explicit checkpoint_mode: pretrained, authors, or lora")
        checkpoint_mode = "pretrained"
    if checkpoint_mode not in {"pretrained", "authors", "lora"}:
        raise ValueError("Unknown checkpoint mode")
    if checkpoint_mode != "pretrained" and (model != "1b" or not checkpoint_path):
        raise ValueError("Authors/LoRA mode requires 1B and an explicit checkpoint path")
    if solver_steps is not None and (type(solver_steps) is not int or solver_steps < 1):
        raise ValueError("solver_steps must be a positive integer")
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    hub = HfApi()
    vision_id = "google/siglip-so400m-patch14-384"
    vision_revision = hub.model_info(vision_id, revision=vision_revision).sha
    vision_path = snapshot_download(vision_id, revision=vision_revision,
                                   allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model"])
    print("loading SigLIP...")
    vision_model = SiglipVisionTower(
        vision_tower=vision_path, args=None)
    vision_model = vision_model.to(device, dtype=dtype)
    image_processor = vision_model.image_processor
    NUM_PATCHES = vision_model.num_patches  # 729

    background_color = tuple(int(x * 255) for x in image_processor.image_mean)
    bg_image = Image.new("RGB", (384, 384), background_color)
    gray_image = Image.new("RGB", (384, 384), (128, 128, 128))

    bg_processed = image_processor.preprocess(bg_image, return_tensors="pt")["pixel_values"][0]
    bg_batched = bg_processed.unsqueeze(0).to(device, dtype=dtype)
    with torch.no_grad():
        bg_image_encoded = vision_model(bg_batched).detach()  # (1, 729, 1152)

    gray_processed = image_processor.preprocess(gray_image, return_tensors="pt")["pixel_values"][0]
    baseline_stack = torch.stack([bg_processed, bg_processed, bg_processed,
                                  gray_processed,
                                  bg_processed, bg_processed]).to(device, dtype=dtype)
    with torch.no_grad():
        baseline_tokens = vision_model(baseline_stack).detach()
        img_tokens_baseline = baseline_tokens.reshape(1, -1, vision_model.hidden_size)

    if model == "170m":
        config_path = "configs/base_170m.yaml"
    else:
        config_path = str(source / "configs/base.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    #Month 5: override the DPM-Solver++ step count in our config dict only, before
    #RDTRunner reads config["model"]["noise_scheduler"]. The ~/rdt-repo library is
    #never edited. Mirrors the override in per_step_ig.py.
    if solver_steps is not None:
        _old_T = config["model"]["noise_scheduler"]["num_inference_timesteps"]
        config["model"]["noise_scheduler"]["num_inference_timesteps"] = solver_steps
        print(f"OVERRIDE num_inference_timesteps {_old_T} -> {solver_steps}")

    print(f"loading RDT-{model} backbone from {config_path}...")
    runner = RDTRunner(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        lang_token_dim=config["model"]["lang_token_dim"],
        img_token_dim=config["model"]["img_token_dim"],
        state_token_dim=config["model"]["state_token_dim"],
        max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
        img_cond_len=config["common"]["img_history_size"] * config["common"]["num_cameras"] * NUM_PATCHES,
        img_pos_embed_config=[("image", (config["common"]["img_history_size"],
                                         config["common"]["num_cameras"], -NUM_PATCHES))],
        lang_pos_embed_config=[("lang", -config["dataset"]["tokenizer_max_length"])],
        dtype=dtype,
    )

    identity = {"upstream_commit": revision, "vision_model": vision_id,
                "vision_revision": vision_revision, "model_size": model,
                "checkpoint_mode": checkpoint_mode, "config_sha256": object_hash(config),
                "model_dtype": str(dtype), "gradient_checkpointing": enable_checkpoint}
    if checkpoint_mode == "authors":
        identity["checkpoint"] = load_checkpoint(runner, checkpoint_path)
    else:
        model_id = f"robotics-diffusion-transformer/rdt-{model}"
        model_revision = hub.model_info(model_id, revision=model_revision).sha
        weight_path = hf_hub_download(model_id, "pytorch_model.bin", revision=model_revision)
        identity.update(model_repository=model_id, model_revision=model_revision)
        identity["checkpoint"] = load_checkpoint(runner, weight_path)
        if checkpoint_mode == "lora":
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                target_modules=["attn.qkv", "attn.proj", "cross_attn.q", "cross_attn.kv", "cross_attn.proj"])
            runner.model = get_peft_model(runner.model, lora_cfg)
            identity["adapter"] = load_checkpoint(runner.model, checkpoint_path, adapter_only=True)
    runner = runner.to(device, dtype=dtype)
    runner.train(False)
    for p in runner.parameters():
        p.requires_grad_(False)

    if enable_checkpoint:
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        def _make_wrapped(orig_fn):
            def wrapped(x, c, mask=None):
                return ckpt_fn(orig_fn, x, c, mask, use_reentrant=False)
            return wrapped

        for block in runner.model.blocks:
            block.forward = _make_wrapped(block.forward)
        print(f"gradient checkpointing enabled on {len(runner.model.blocks)} blocks")
    else:
        print(f"gradient checkpointing DISABLED ({len(runner.model.blocks)} blocks)")

    state_elem_mask = torch.zeros(1, 128, device=device, dtype=dtype)
    state_elem_mask[0, MANISKILL_INDICES] = 1.0
    action_mask = state_elem_mask.unsqueeze(1)
    ctrl_freqs = torch.tensor([CONTROL_FREQ], device=device, dtype=dtype)

    return {
        "runner": runner,
        "vision_model": vision_model,
        "bg_image_encoded": bg_image_encoded,
        "img_tokens_baseline": img_tokens_baseline,
        "action_mask": action_mask,
        "ctrl_freqs": ctrl_freqs,
        "config": config,
        "identity": identity,
    }


def load_lang(task, lang_dir="data/lang_embeds", *, device="cuda", dtype=torch.bfloat16):
    """
    Load task embeddings and the shared PAD/EOS baseline (legacy filename).

    Returns dict with keys: lang_tokens (1, 1024, 4096) bf16 on GPU,
    lang_attn_mask (1, 1024) bool on GPU, lang_tokens_baseline (1, 1024, 4096)
    bf16 on GPU, tokens (list[str] or None).
    """
    task_path = os.path.join(lang_dir, f"{task}.pt")
    if not os.path.exists(task_path):
        raise FileNotFoundError(
            f"no language embeddings at {task_path}. "
            f"Run: .venv/bin/python encode_task_lang.py --task {task}")
    data = torch.load(task_path, weights_only=True)
    lang_tokens = data["embeds"].to(device, dtype=dtype)
    lang_attn_mask = data["attn_mask"].to(device, dtype=torch.bool)
    tokens = data.get("tokens", None)

    baseline_path = os.path.join(lang_dir, "baseline_bos_eos.pt")
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"no language baseline at {baseline_path}. "
            f"Run: .venv/bin/python encode_task_lang.py (the no-argument "
            f"all-tasks run is the invocation that writes the baseline file)")
    baseline = torch.load(baseline_path, weights_only=True)
    lang_tokens_baseline = baseline["embeds"].to(device, dtype=dtype)

    for label, value in (("task", data), ("baseline", baseline)):
        embeds, mask = value["embeds"], value["attn_mask"]
        if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[-1] != 4096 or mask.shape != embeds.shape[:2]:
            raise ValueError(f"Invalid {label} embedding shape/mask")
        if not torch.isfinite(embeds).all() or not torch.isin(mask, torch.tensor([0, 1], device=mask.device)).all():
            raise ValueError(f"Invalid {label} embedding values/mask")
        if not mask.bool().any():
            raise ValueError(f"Empty {label} attention mask")
    if lang_tokens.shape != lang_tokens_baseline.shape:
        raise ValueError("Task and baseline embedding shapes differ")
    return {
        "lang_tokens": lang_tokens,
        "lang_attn_mask": lang_attn_mask,
        "lang_tokens_baseline": lang_tokens_baseline,
        "tokens": tokens,
        "identity": {"task": task, "task_embedding_sha256": file_hash(task_path),
                     "baseline_embedding_sha256": file_hash(baseline_path),
                     "baseline_attention_policy": "task_mask_held_fixed"},
    }
