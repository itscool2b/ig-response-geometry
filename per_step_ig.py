"""Collect revised episode IG with strict identities and immutable episode commits.

New runs default to runs/<unique-id>/metrics.jsonl. Sidecars and unfinished
attempts stay inside its .run directory. --resume accepts only an identical
manifest and skips verified committed episodes. Historical JSONL cannot resume.
"""

import sys
import os
import gc
import json
import time
import argparse
import importlib.metadata
from pathlib import Path
from experiment_io import RunStore, file_hash, object_hash, tensor_hash
from pipeline import load_pipeline, load_lang
from collections import deque

import numpy as np
import torch
import yaml
from PIL import Image

from per_step_attribution import compute_ig_for_step, MANISKILL_INDICES

CONTROL_FREQ = 25

#Panda action bounds for denormalizing the 8-dim joint slice out of RDT's 128-dim
#unified action output. Taken verbatim from ~/rdt-repo/scripts/maniskill_model.py
#line 28 (DATA_STAT['action_min'/'action_max']) — RDT's _unformat_action_to_joint
#applies this same denormalization before passing actions to env.step.
ACTION_MIN = torch.tensor([-0.7472005486488342, -0.08631071448326111, -0.4995281398296356,
                           -2.658363103866577, -0.5751323103904724, 1.8290787935256958,
                           -2.245187997817993, -1.0])
ACTION_MAX = torch.tensor([0.7654682397842407, 1.4984270334243774, 0.46786263585090637,
                           -0.38181185722351074, 0.5517147779464722, 3.291581630706787,
                           2.575840711593628, 1.0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="PickCube-v1",
                   help="ManiSkill task id. Must have data/lang_embeds/{task}.pt.")
    p.add_argument("--episodes", type=int, default=10,
                   help="Prespecified number of episodes; this is not a power recommendation.")
    p.add_argument("--m", type=int, default=64,
                   help="Quadrature intervals per modality; validate for the selected workload.")
    p.add_argument("--model", choices=["170m", "1b"], default="170m",
                   help="RDT backbone size. Default 170m (dev); 1b for scale cross-check.")
    p.add_argument("--max-policy-calls", type=int, default=30,
                   help="Safety bound on policy calls per episode (prevents runaway episodes).")
    p.add_argument("--out", default=None,
                   help="Output JSONL path. Defaults to a unique directory under runs/.")
    p.add_argument("--video-dir", default=None,
                   help="If set, wrap env with RecordEpisode and save an mp4 per episode to this dir.")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable gradient checkpointing. Faster backward; needs more VRAM.")
    p.add_argument("--seed-base", type=int, default=42,
                   help="Base seed. Episode e uses seed_base+e; select before collection.")
    p.add_argument("--resume", action="store_true",
                   help="Resume only an identical manifest and verified committed episodes; "
                        "requires explicit --out. Historical JSONL cannot be resumed.")
    p.add_argument("--target", default="logpi",
                   choices=["logpi", "l2", "l2sq", "maxdev", "cosine"],
                   help="IG target function. logpi (default, auxiliary quadratic score) "
                        "or an alternative-target ablation (Month 4): l2, l2sq, maxdev, cosine.")
    p.add_argument("--solver-steps", type=int, default=None,
                   help="Override the DPM-Solver++ denoising step count "
                        "(num_inference_timesteps, default 5 in the config). Changing this "
                        "measures solver-resolution sensitivity, not denoiser contraction.")
    p.add_argument("--quadrature", choices=["trapezoid", "legacy_endpoint_average"], default="trapezoid")
    p.add_argument("--checkpoint-mode", choices=["pretrained", "authors", "lora"], default=None)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--model-revision", default=None)
    p.add_argument("--vision-revision", default=None)
    p.add_argument("--lang-dir", default="data/lang_embeds")
    return p.parse_args()


def main():
    args = parse_args()

    if args.episodes < 1 or args.m < 1 or args.max_policy_calls < 1:
        raise ValueError("Episodes, quadrature intervals and call limit must be positive")
    if args.resume and args.out is None:
        raise ValueError("--resume requires an explicit --out")
    if args.out is None:
        from uuid import uuid4
        args.out = f"runs/{args.task}_{args.model}_{args.target}_{uuid4().hex}/metrics.jsonl"
    if Path(args.out).exists() and not args.resume:
        raise FileExistsError("Output exists; use a new path or exact-compatible --resume")
    pipe = load_pipeline(args.model, enable_checkpoint=not args.no_checkpoint,
                         solver_steps=args.solver_steps, checkpoint_mode=args.checkpoint_mode,
                         checkpoint_path=args.checkpoint_path, model_revision=args.model_revision,
                         vision_revision=args.vision_revision)
    language = load_lang(args.task, args.lang_dir)
    runner, vision_model = pipe["runner"], pipe["vision_model"]
    bg_image_encoded, img_tokens_baseline = pipe["bg_image_encoded"], pipe["img_tokens_baseline"]
    action_mask, ctrl_freqs, config = pipe["action_mask"], pipe["ctrl_freqs"], pipe["config"]
    lang_tokens, lang_attn_mask = language["lang_tokens"], language["lang_attn_mask"]
    lang_tokens_baseline = language["lang_tokens_baseline"]
    source_files = ["per_step_ig.py", "per_step_attribution.py", "integrated_gradients.py",
                    "pipeline.py", "checkpoint_contract.py", "experiment_io.py", "rdt_sampling.py"]
    configuration = {
        "schema_version": 1, "task": args.task, "model": args.model,
        "episodes": args.episodes, "seed_base": args.seed_base,
        "max_policy_calls": args.max_policy_calls, "m": args.m,
        "quadrature": args.quadrature, "target": args.target,
        "pipeline": pipe["identity"], "language": language["identity"],
        "solver_steps": config["model"]["noise_scheduler"]["num_inference_timesteps"],
        "control_mode": "pd_joint_pos", "max_episode_steps": 400,
        "observation_pipeline": "one_current_external_camera_five_background_slots",
        "action_subsampling": 4, "noise_policy": "fixed_episode_seed_stored_per_context",
        "arithmetic_dtype": "torch.float32", "forward_dtype": "torch.bfloat16",
        "source_sha256": {name: file_hash(Path(__file__).parent / name) for name in source_files},
        "environment": {name: importlib.metadata.version(name) for name in
                        ["torch", "numpy", "diffusers", "transformers", "mani_skill", "sapien"]},
    }
    store = RunStore(args.out, configuration, resume=args.resume)

    #----- Env -----
    import gymnasium as gym
    import mani_skill.envs
    #control_mode must match what RDT's LoRA was fine-tuned on. eval_rdt_maniskill.py:58
    #in the RDT repo explicitly sets pd_joint_pos. ManiSkill's default for PickCube-v1 is
    #pd_joint_delta_pos with action space [-1,1]; ACTION_MIN/ACTION_MAX in this file are
    #for pd_joint_pos (absolute joint positions, ~[-2.9, 2.9]). Without this control_mode
    #override the env silently clips + reinterprets each RDT action as a tiny delta,
    #which drives 1B+LoRA from 76% success down to 0%.
    #
    #max_episode_steps must also be set: ManiSkill wraps PickCube-v1 in a TimeLimitWrapper
    #that truncates at 50 env steps by default (one chunk of policy actions barely fits).
    #RDT's official eval (eval_rdt_maniskill.py) uses MAX_EPISODE_STEPS=400. Without this,
    #the policy is truncated before it can grasp; this is the second of two bugs that
    #stopped 1B+LoRA from succeeding.
    env = gym.make(args.task, obs_mode="state_dict", num_envs=1,
                   control_mode="pd_joint_pos", render_mode="rgb_array",
                   max_episode_steps=400)

    if args.video_dir:
        from mani_skill.utils.wrappers import RecordEpisode
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=args.video_dir, save_trajectory=False,
                            save_video=True, video_fps=30, info_on_video=False,
                            trajectory_name=f"{args.task}_{args.model}_ep")
        print(f"=== recording videos to {args.video_dir} ===")

    def render_pil():
        arr = env.render().squeeze(0).detach().cpu().numpy().astype(np.uint8)
        return Image.fromarray(arr).resize((384, 384))

    #----- Episode loop -----
    print(f"\n=== running {args.episodes} episodes, m={args.m}, model={args.model} ===")
    print(f"=== output: {args.out} ===\n")
    transaction = None
    try:
        with store.writer():
            resume_episodes = store.completed_episodes()

            for ep in range(args.episodes):
                if ep in resume_episodes:
                    print(f"--- episode {ep}: SKIPPED (resume) ---")
                    continue
                print(f"\n--- episode {ep} (seed={args.seed_base + ep}) ---")
                transaction = store.begin_episode(ep)
                obs, info = env.reset(seed=args.seed_base + ep)
                frame_deque = deque(maxlen=2)
                frame_deque.append(None)               # no t-1 history at the very first call
                frame_deque.append(render_pil())

                call_idx = 0
                env_step_count = 0
                terminated = truncated = False

                while not (terminated or truncated) and call_idx < args.max_policy_calls:
                    obs_image = frame_deque[-1]
                    proprio = obs["agent"]["qpos"][0, :8].cpu()

                    t0 = time.time()
                    attr = compute_ig_for_step(
                        runner, vision_model, obs_image, proprio,
                        lang_tokens, lang_attn_mask, lang_tokens_baseline,
                        bg_image_encoded, img_tokens_baseline,
                        action_mask, ctrl_freqs,
                        seed=args.seed_base + ep, m=args.m, target=args.target,
                        quadrature=args.quadrature,
                        diagnostic_context={"run_id": store.manifest["run_id"], "episode": ep, "policy_call": call_idx},
                    )
                    wall = time.time() - t0

                    payload = {
                        "vision_attr": attr["vision"]["attribution"].cpu(),
                        "lang_attr": attr["language"]["attribution"].cpu(),
                        "state_attr": attr["state"]["attribution"].cpu(),
                        "ref_action": attr["ref_action"].cpu(),
                        "obs_image": torch.from_numpy(np.array(obs_image)),
                        "proprio": proprio.cpu(),
                        "initial_noise": attr["initial_noise"].cpu(),
                        "initial_noise_sha256": attr["initial_noise_sha256"],
                        "sampler_metadata": attr["sampler_metadata"],
                        "lang_attn_mask": lang_attn_mask.cpu(), "action_mask": action_mask.cpu(),
                        "ctrl_freqs": ctrl_freqs.cpu(),
                    }
                    context_id = object_hash({"episode": ep, "policy_call": call_idx,
                        "configuration": store.identity,
                        "observation": tensor_hash(payload["obs_image"]),
                        "proprio": tensor_hash(payload["proprio"]),
                        "noise": attr["initial_noise_sha256"],
                        "reference": tensor_hash(payload["ref_action"])})
                    payload["context_id"] = context_id

                    #JSONL summary row — summary stats only, safe for jq/grep.
                    #State per-joint (8 floats) stays inline; large attribution tensors
                    #are in the sidecar above.
                    row = {
                        "event": "step",
                        "task": args.task, "model": args.model,
                        "episode": ep, "seed": args.seed_base + ep,
                        "policy_call_idx": call_idx, "env_step_at_call": env_step_count,
                        "solver_steps": (args.solver_steps if args.solver_steps is not None
                                         else config["model"]["noise_scheduler"]["num_inference_timesteps"]),
                        "ref_norm_maniskill": attr["ref_norm_maniskill"],
                        "vision_err":   attr["vision"]["completeness_err"],
                        "vision_gap":   attr["vision"]["expected_gap"],
                        "vision_ig_sum": attr["vision"]["ig_sum"],
                        "lang_err":     attr["language"]["completeness_err"],
                        "lang_gap":     attr["language"]["expected_gap"],
                        "lang_ig_sum":  attr["language"]["ig_sum"],
                        "state_err":    attr["state"]["completeness_err"],
                        "state_gap":    attr["state"]["expected_gap"],
                        "state_ig_sum": attr["state"]["ig_sum"],
                        "state_per_joint": attr["state"]["per_joint"].tolist(),
                        "wall_seconds": wall,
                        "context_id": context_id, "target": args.target,
                        "m": args.m, "quadrature": args.quadrature,
                        "initial_noise_sha256": attr["initial_noise_sha256"],
                        "numerics": {key: attr[key]["numerics"] for key in ("vision", "language", "state")},
                    }
                    transaction.save_step(row, payload)
                    print(f"  call {call_idx}: errors=" +
                          str({key: attr[key]["completeness_err"] for key in ("vision", "language", "state")}) +
                          f" |ref|={attr['ref_norm_maniskill']:.2f} wall={wall:.1f}s")
                    call_idx += 1

                    #Apply the chunk — same 4x subsampling pattern as eval_maniskill.py:87.
                    #ref_action is the fixed-noise sample conditional_sample(real_inputs),
                    #which IS the action the policy would execute. No duplicate forward.
                    #
                    #RDT outputs a (1, 64, 128) unified action; ManiSkill's Panda controller
                    #wants a (1, 8) denormalized joint-space action per env.step. Replicate
                    #the RoboticDiffusionTransformerModel._unformat_action_to_joint path from
                    #~/rdt-repo/scripts/maniskill_model.py: slice MANISKILL_INDICES then
                    #denormalize from [-1, 1] to [ACTION_MIN, ACTION_MAX] per-dim.
                    action_unified = attr["ref_action"].squeeze(0).float().cpu()  # (64, 128)
                    action_joints = action_unified[:, MANISKILL_INDICES]           # (64, 8), still in [-1, 1]
                    action_denorm = (action_joints + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN
                    actions_np = action_denorm[::4, :].numpy()                     # (16, 8)
                    for action in actions_np:
                        obs, _r, terminated, truncated, info = env.step(action.reshape(1, 8))
                        env_step_count += 1
                        frame_deque.append(render_pil())
                        if terminated or truncated:
                            break

                #Episode-end row — each episode has one outcome record keyed on event
                success = bool(info.get("success", False))
                transaction.commit({
                    "task": args.task, "model": args.model, "episode": ep,
                    "seed": args.seed_base + ep, "event": "episode_end",
                    "env_steps": env_step_count, "policy_calls": call_idx,
                    "success": success,
                    "terminated": bool(terminated), "truncated": bool(truncated),
                })
                print(f"  episode {ep}: {call_idx} calls, {env_step_count} env steps, success={success}")

    except Exception as error:
        if transaction is not None:
            transaction.fail(error, getattr(error, "diagnostics", {}))
        raise
    finally:
        env.close()
    print(f"\n=== done. wrote {args.out} ===")


if __name__ == "__main__":
    main()
