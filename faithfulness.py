"""Replay authenticated attribution contexts and measure perturbation responses.

The legacy ``logpi`` target is an auxiliary quadratic action score, not a
normalized policy likelihood. Normalized AUC depends on the response geometry,
can lie outside [0,1], and has no universal random-ranking value of 0.5.
Revised evaluation integrates against realized eligible-position fractions.
Stable ties use increasing eligible index. Zero percent changes zero positions.

New input requires its run manifest, sidecar hashes, stored noise, and matching
model/language/runtime identities. Explicit --legacy-input records unverified
provenance. It never authenticates absent historical identities. Outputs are
new, exclusive files; failed evaluation is recorded and stops the job.
"""

import argparse
import json
import os
import time
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from PIL import Image

from per_step_attribution import (
    prepare_ig_context, build_forward_fns, MANISKILL_INDICES,
)
from pipeline import load_pipeline, load_lang
from experiment_io import (RunStore, atomic_bytes, canonical_json, file_hash,
                           object_hash, strict_json, tensor_hash)


#k-grid for B2 AUC (percent). 0 and 100 are the endpoints of the sweep.
AUC_K_GRID = [0, 1, 5, 10, 20, 30, 50, 75, 100]
#k values at which B1 Δlog p is reported.
B1_K = [1, 5, 10]
#slot 3 of the 6-slot image condition = external camera at time t; indices 2187..2915.
#Only this slot has a non-trivial IG path, the other 5 slots have input == baseline.
EXT_CAM_START = 3 * 729
EXT_CAM_END = 4 * 729


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True,
                   help="Path to per_step_ig.py JSONL output.")
    p.add_argument("--task", required=True, help="ManiSkill task id.")
    p.add_argument("--model", choices=["170m", "1b"], default=None)
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Disable gradient checkpointing on the pipeline load.")
    p.add_argument("--out", default=None,
                   help="Output JSONL. Auto-named from metrics if omitted.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N step rows. For smoke tests.")
    p.add_argument("--target", default=None,
                   choices=["logpi", "l2", "l2sq", "maxdev", "cosine"],
                   help="IG target the source JSONL used. Must match per_step_ig --target "
                        "so the perturbation Δ is measured on the same scalar.")
    p.add_argument("--legacy-input", action="store_true",
                   help="Allow a manifestless source only as explicitly unverified exploratory input; malformed or duplicate rows still fail.")
    p.add_argument("--auc-grid", choices=["realized", "legacy_nominal"], default="realized",
                   help="Primary integration axis. Both values are reported with explicit names.")
    p.add_argument("--denominator-min", type=float, default=0.0,
                   help="Declared absolute endpoint-gap threshold; default excludes only exact zero.")
    p.add_argument("--solver-steps", type=int, default=None)
    p.add_argument("--checkpoint-mode", choices=["pretrained", "authors", "lora"], default=None)
    p.add_argument("--checkpoint-path", default=None)
    p.add_argument("--model-revision", default=None)
    p.add_argument("--vision-revision", default=None)
    p.add_argument("--lang-dir", default="data/lang_embeds")
    return p.parse_args()


def load_step_rows(metrics_path):
    """Strict source parsing. Historical malformed lines/duplicates are errors."""
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value
    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    def finite_tree(value):
        if isinstance(value, dict):
            for item in value.values():
                finite_tree(item)
        elif isinstance(value, list):
            for item in value:
                finite_tree(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Nonfinite JSON number")
    rows, seen = [], set()
    with open(metrics_path, "rb") as handle:
        for number, line in enumerate(handle, 1):
            r = json.loads(line, object_pairs_hook=unique, parse_constant=nonfinite)
            finite_tree(r)
            if not isinstance(r, dict) or r.get("event") not in {"step", "episode_end"}:
                raise ValueError(f"Unknown source record at line {number}")
            if r["event"] == "episode_end":
                continue
            for field in ("episode", "seed", "policy_call_idx"):
                if type(r.get(field)) is not int or r[field] < 0:
                    raise ValueError(f"Invalid {field} at line {number}")
            if not isinstance(r.get("attr_file"), str) or not r["attr_file"]:
                raise ValueError(f"Missing attribution sidecar at line {number}")
            key = (r.get("run_id"), r.get("task"), r.get("model"), r["seed"], r["episode"], r["policy_call_idx"])
            if key in seen:
                raise ValueError(f"Duplicate source decision at line {number}")
            seen.add(key)
            rows.append(r)
    if not rows:
        raise ValueError("Source has no step records")
    return rows


def topk_mask(abs_scores, k_pct, n_total):
    """
    Boolean (n_total,) mask selecting the top k_pct% of positions by abs_scores.
    abs_scores is the ranked population (e.g. 729 ext-cam tokens, or the
    lang_attn_mask.sum() real tokens); n_total is its length.
    """
    if abs_scores.ndim != 1 or type(n_total) is not int or n_total < 1 or abs_scores.numel() != n_total:
        raise ValueError("Ranking population must be a nonempty one-dimensional score vector")
    if not math.isfinite(k_pct) or not 0 <= k_pct <= 100:
        raise ValueError("k_pct must be finite and between zero and 100")
    if not torch.isfinite(abs_scores).all() or (abs_scores < 0).any():
        raise ValueError("Ranking scores must be finite and nonnegative")
    k_count = int(round(n_total * k_pct / 100.0))
    idx = torch.argsort(abs_scores, descending=True, stable=True)[:k_count]
    mask = torch.zeros(n_total, dtype=torch.bool, device=abs_scores.device)
    mask[idx] = True
    return mask


def perturb_image(img_adapted, img_adapted_bl, ext_mask):
    """
    Return img_adapted with ext-cam token positions selected by `ext_mask`
    (shape (729,), bool) replaced by the corresponding baseline positions.
    """
    out = img_adapted.clone()
    start, end = EXT_CAM_START, EXT_CAM_END
    replace_idx = start + torch.nonzero(ext_mask, as_tuple=False).squeeze(-1)
    out[:, replace_idx, :] = img_adapted_bl[:, replace_idx, :]
    return out


def perturb_image_insertion(img_adapted, img_adapted_bl, ext_mask):
    """
    Symmetric to perturb_image: start from baseline, add input values at
    positions selected by `ext_mask`. Only the ext-cam slot differs; the other
    five slots already agree between input and baseline.
    """
    out = img_adapted_bl.clone()
    start, end = EXT_CAM_START, EXT_CAM_END
    replace_idx = start + torch.nonzero(ext_mask, as_tuple=False).squeeze(-1)
    out[:, replace_idx, :] = img_adapted[:, replace_idx, :]
    return out


def perturb_lang(lang_adapted, lang_adapted_bl, real_token_mask, full_mask):
    """
    Replace positions selected by `full_mask` (shape (1024,), bool) in
    `lang_adapted` with the corresponding baseline positions. `real_token_mask`
    is unused here but indicates which positions are real vs padding (for the
    caller to scope its top-k selection correctly).
    """
    out = lang_adapted.clone()
    replace_idx = torch.nonzero(full_mask, as_tuple=False).squeeze(-1)
    out[:, replace_idx, :] = lang_adapted_bl[:, replace_idx, :]
    return out


def perturb_lang_insertion(lang_adapted, lang_adapted_bl, full_mask):
    out = lang_adapted_bl.clone()
    replace_idx = torch.nonzero(full_mask, as_tuple=False).squeeze(-1)
    out[:, replace_idx, :] = lang_adapted[:, replace_idx, :]
    return out


def auc_normalized(k_grid, y_values, f_input, f_baseline, *, denominator_min=0.0):
    """
    Trapezoidal AUC of y_values over k_grid/100, with y linearly renormalized
    so f_input -> 1 and f_baseline -> 0. Values are unbounded outside [0,1]
    when intermediate responses exceed the endpoints, including valid responses.
    Returns None for a declared undefined gap. Nonfinite inputs are errors.
    """
    x = np.asarray(k_grid, dtype=np.float64) / 100.0
    values = np.asarray(y_values, dtype=np.float64)
    if x.ndim != 1 or values.ndim != 1 or len(x) != len(values) or len(x) < 2:
        raise ValueError("Invalid AUC grid/curve dimensions")
    if not np.isfinite(x).all() or not np.isfinite(values).all() or not all(map(math.isfinite, [f_input, f_baseline, denominator_min])):
        raise ValueError("Nonfinite AUC input")
    if x[0] != 0 or x[-1] != 1 or (np.diff(x) < 0).any() or denominator_min < 0:
        raise ValueError("AUC grid must be nondecreasing from 0 to 100; gap cutoff must be nonnegative")
    denom = f_input - f_baseline
    if not math.isfinite(denom):
        raise ValueError("Nonfinite endpoint-gap arithmetic")
    if abs(denom) <= denominator_min:
        return None
    y_norm = (values - f_baseline) / denom
    #numpy 2.0 renamed np.trapz to np.trapezoid; fall back to np.trapz on numpy 1.x.
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    value = float(trapezoid(y_norm, x))
    if not math.isfinite(value):
        raise ValueError("Nonfinite normalized AUC")
    return value


def compute_modality_metrics(
    fwd_fn, input_tensor, baseline_tensor, attribution,
    modality, real_mask=None, *, auc_grid="realized", denominator_min=0.0,
):
    """
    Run all B1 + B2 forward passes for one modality and one step.

    Args:
        fwd_fn: forward_fn_vision or forward_fn_language from build_forward_fns
        input_tensor: the modality's "real" input to fwd_fn (adapted embeddings)
        baseline_tensor: same-shape baseline to fwd_fn
        attribution: IG output, same shape as input_tensor
        modality: "vision" | "language"
        real_mask: (seq_len,) bool on GPU, for language; positions that are real
            tokens (rest are padding and must not be ranked by |IG|).
    Returns: dict with dlogp_k1/5/10, deletion_auc, insertion_auc,
             deletion_curve, insertion_curve
    """
    device = input_tensor.device
    if auc_grid not in {"realized", "legacy_nominal"}:
        raise ValueError("Unknown integration axis")
    if input_tensor.ndim != 3 or input_tensor.shape[0] != 1 or input_tensor.shape != baseline_tensor.shape or attribution.shape != input_tensor.shape:
        raise ValueError("Input, baseline, and attribution shapes must agree with batch size one")
    if any(not torch.isfinite(value).all() for value in (input_tensor, baseline_tensor, attribution)):
        raise ValueError("Nonfinite input, baseline, or attribution")

    #Reduce attribution to a per-position score by summing |IG| over hidden dim.
    abs_per_pos = attribution.float().abs().sum(dim=-1).squeeze(0)  # (seq_len,)

    if modality == "vision":
        #Rank only within the 729 ext-cam tokens; the other 5 slots are
        #attribution-zero by construction (input == baseline on them).
        ranked_scores = abs_per_pos[EXT_CAM_START:EXT_CAM_END]
        n_ranked = 729
        real_idx = torch.arange(EXT_CAM_START, EXT_CAM_END, device=device)
        if input_tensor.shape[1] < EXT_CAM_END:
            raise ValueError("Vision input is missing the external-camera token slot")
    elif modality == "language":
        if real_mask is None or real_mask.dtype != torch.bool or real_mask.shape != (input_tensor.shape[1],):
            raise ValueError("Language eligibility requires a boolean sequence mask")
        real_idx = torch.nonzero(real_mask, as_tuple=False).squeeze(-1)
        ranked_scores = abs_per_pos[real_idx]
        n_ranked = int(real_mask.sum().item())
    else:
        raise ValueError(f"unsupported modality: {modality}")
    if n_ranked == 0:
        raise ValueError("No eligible positions")
    eligible = torch.zeros(input_tensor.shape[1], dtype=torch.bool, device=device)
    eligible[real_idx] = True
    if not torch.equal(input_tensor[:, ~eligible], baseline_tensor[:, ~eligible]):
        raise ValueError("Input and attribution baseline differ outside the declared intervention population")

    def evaluate(value):
        with torch.no_grad():
            score = fwd_fn(value)
        if not isinstance(score, torch.Tensor) or score.numel() != 1 or not torch.isfinite(score).all():
            raise ValueError("Forward response must be one finite scalar")
        return float(score.item())

    #f_input and f_baseline, cached once.
    f_input = evaluate(input_tensor)
    f_baseline = evaluate(baseline_tensor)

    #B1: Δlog p at k in {1, 5, 10}.
    dlogp = {}
    for k in B1_K:
        ranked_mask = topk_mask(ranked_scores, k, n_ranked)
        if modality == "vision":
            perturbed = perturb_image(input_tensor, baseline_tensor, ranked_mask)
        else:
            full_mask = torch.zeros(input_tensor.shape[1], dtype=torch.bool, device=device)
            full_mask[real_idx] = ranked_mask
            perturbed = perturb_lang(input_tensor, baseline_tensor, real_mask, full_mask)
        f_del = evaluate(perturbed)
        dlogp[k] = f_del - f_input
        if not math.isfinite(dlogp[k]):
            raise ValueError("Nonfinite deletion-response arithmetic")
        del perturbed

    #B2: AUC curves. Share the ranked_mask computation with B1 where k overlaps.
    deletion_curve = []
    insertion_curve = []
    for k in AUC_K_GRID:
        if k == 0:
            deletion_curve.append(f_input)
            insertion_curve.append(f_baseline)
            continue
        if k == 100:
            deletion_curve.append(f_baseline)
            insertion_curve.append(f_input)
            continue
        ranked_mask = topk_mask(ranked_scores, k, n_ranked)
        if modality == "vision":
            p_del = perturb_image(input_tensor, baseline_tensor, ranked_mask)
            p_ins = perturb_image_insertion(input_tensor, baseline_tensor, ranked_mask)
        else:
            full_mask = torch.zeros(input_tensor.shape[1], dtype=torch.bool, device=device)
            full_mask[real_idx] = ranked_mask
            p_del = perturb_lang(input_tensor, baseline_tensor, real_mask, full_mask)
            p_ins = perturb_lang_insertion(input_tensor, baseline_tensor, full_mask)
        deletion_curve.append(evaluate(p_del))
        insertion_curve.append(evaluate(p_ins))
        del p_del, p_ins

    counts = [int(topk_mask(ranked_scores, k, n_ranked).sum().item()) for k in AUC_K_GRID]
    realized_grid = [100.0 * count / n_ranked for count in counts]
    primary_grid = realized_grid if auc_grid == "realized" else AUC_K_GRID
    deletion_auc = auc_normalized(primary_grid, deletion_curve, f_input, f_baseline, denominator_min=denominator_min)
    insertion_auc = auc_normalized(primary_grid, insertion_curve, f_input, f_baseline, denominator_min=denominator_min)
    order = real_idx[torch.argsort(ranked_scores, descending=True, stable=True)].cpu().tolist()

    return {
        "dlogp_k1": dlogp[1],
        "dlogp_k5": dlogp[5],
        "dlogp_k10": dlogp[10],
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "deletion_curve": deletion_curve,
        "insertion_curve": insertion_curve,
        "f_input": f_input,
        "f_baseline": f_baseline,
        "endpoint_gap": f_input - f_baseline,
        "denominator_min": denominator_min,
        "auc_status": "defined" if deletion_auc is not None else "undefined_endpoint_gap",
        "auc_axis": auc_grid,
        "nominal_grid_percent": list(AUC_K_GRID),
        "realized_grid_percent": realized_grid,
        "selected_counts": counts,
        "eligible_count": n_ranked,
        "eligible_indices": real_idx.cpu().tolist(),
        "ranking_indices": order,
        "ranking_sha256": object_hash(order),
        "mask_sha256": [object_hash(sorted(order[:count])) for count in counts],
        "tie_rule": "stable decreasing absolute attribution sum, increasing eligible index",
        "count_rule": "round_to_nearest_even(n_eligible*percent/100), including zero",
        "legacy_nominal_deletion_auc": auc_normalized(AUC_K_GRID, deletion_curve, f_input, f_baseline, denominator_min=denominator_min),
        "legacy_nominal_insertion_auc": auc_normalized(AUC_K_GRID, insertion_curve, f_input, f_baseline, denominator_min=denominator_min),
        "realized_deletion_auc": auc_normalized(realized_grid, deletion_curve, f_input, f_baseline, denominator_min=denominator_min),
        "realized_insertion_auc": auc_normalized(realized_grid, insertion_curve, f_input, f_baseline, denominator_min=denominator_min),
    }


def authenticated_source(metrics_path, *, legacy=False):
    """Authenticate a complete exported run before selecting an evaluation subset."""
    path = Path(metrics_path).resolve()
    rows = load_step_rows(path)
    manifest_path = path.with_name(path.name + ".run") / "manifest.json"
    if not manifest_path.exists():
        if legacy:
            return rows, None
        raise ValueError("Source run manifest is required; historical replay needs explicit --legacy-input and remains unverified")
    if legacy:
        raise ValueError("--legacy-input cannot bypass an existing revised run manifest")
    manifest = strict_json(manifest_path)
    configuration = manifest["configuration"]
    if object_hash(configuration) != manifest["configuration_sha256"]:
        raise ValueError("Source configuration hash mismatch")
    store = RunStore(path, configuration, resume=True)
    store.manifest = manifest
    store.completed_episodes()
    expected = b"".join(canonical_json(r) + b"\n"
                        for episode in sorted((store.root / "episodes").glob("ep*.json"))
                        for r in strict_json(episode)["records"])
    if expected != path.read_bytes():
        raise ValueError("Source export differs from committed episode transactions")
    return rows, manifest


def load_sidecar(row, metrics_path, manifest=None):
    source = Path(metrics_path).resolve()
    path = Path(row["attr_file"])
    if not path.is_absolute():
        path = source.parent / path
    path = path.resolve()
    if manifest is not None:
        source_run = source.with_name(source.name + ".run")
        if not path.is_relative_to(source_run):
            raise ValueError("Source sidecar is outside its run")
        if file_hash(path) != row["attr_sha256"]:
            raise ValueError("Sidecar byte hash mismatch")
    # Revised sidecars contain tensors and primitive metadata. No implicit
    # fallback to unrestricted pickle is permitted for historical numpy objects.
    payload = torch.load(path, weights_only=True, map_location="cpu")
    for name, value in payload.items():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite source sidecar tensor: {name}")
    if manifest is not None:
        for name in ("run_id", "configuration_sha256"):
            if row[name] != manifest[name] or payload[name] != manifest[name]:
                raise ValueError(f"Source sidecar {name} mismatch")
        noise_hash = tensor_hash(payload["initial_noise"])
        if noise_hash != payload["initial_noise_sha256"] or noise_hash != row["initial_noise_sha256"]:
            raise ValueError("Stored noise tensor hash mismatch")
        context_id = object_hash({"episode": row["episode"], "policy_call": row["policy_call_idx"],
                                  "configuration": manifest["configuration_sha256"],
                                  "observation": tensor_hash(payload["obs_image"]),
                                  "proprio": tensor_hash(payload["proprio"]), "noise": noise_hash,
                                  "reference": tensor_hash(payload["ref_action"])})
        if context_id != payload["context_id"] or context_id != row["context_id"]:
            raise ValueError("Source context tensor identity mismatch")
    return payload, file_hash(path)


def check_replay_identity(pipe, lang, source_manifest, *, allow_solver_change=False):
    configuration = source_manifest["configuration"]
    identity = dict(pipe["identity"])
    if allow_solver_change:
        from copy import deepcopy
        original_solver_config = deepcopy(pipe["config"])
        original_solver_config["model"]["noise_scheduler"]["num_inference_timesteps"] = configuration["solver_steps"]
        if object_hash(original_solver_config) != configuration["pipeline"]["config_sha256"]:
            raise ValueError("Solver comparison changed configuration beyond solver depth")
        identity["config_sha256"] = configuration["pipeline"]["config_sha256"]
    if identity != configuration["pipeline"]:
        raise ValueError("Evaluation pipeline identity differs from attribution source")
    if lang["identity"] != configuration["language"]:
        raise ValueError("Evaluation language/baseline identity differs from attribution source")
    for name in ("pipeline.py", "per_step_attribution.py", "rdt_sampling.py", "checkpoint_contract.py"):
        if file_hash(Path(__file__).parent / name) != configuration["source_sha256"][name]:
            raise ValueError(f"Evaluation forward source differs from attribution source: {name}")
    import importlib.metadata
    for name in ("torch", "numpy", "diffusers", "transformers"):
        if importlib.metadata.version(name) != configuration["environment"][name]:
            raise ValueError(f"Evaluation runtime differs from attribution source: {name}")


def add_replay_arguments(parser):
    """Common arguments for revised post-processing studies."""
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", choices=["170m", "1b"], default=None)
    parser.add_argument("--target", choices=["logpi", "l2", "l2sq", "maxdev", "cosine"], default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--legacy-input", action="store_true")
    parser.add_argument("--checkpoint-mode", choices=["pretrained", "authors", "lora"], default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--vision-revision", default=None)
    parser.add_argument("--lang-dir", default="data/lang_embeds")


def replay_inputs(args, *, apply_limit=True):
    digest = file_hash(args.metrics)
    rows, manifest = authenticated_source(args.metrics, legacy=args.legacy_input)
    if file_hash(args.metrics) != digest:
        raise ValueError("Source changed during authentication")
    configuration = manifest["configuration"] if manifest else {}
    args.model = args.model or configuration.get("model", "170m")
    args.target = args.target or configuration.get("target") or "logpi"
    identity_fields = ("task", "model") if configuration.get("collector_type") == "context_only" else ("task", "model", "target")
    if manifest and any(configuration[key] != getattr(args, key) for key in identity_fields):
        raise ValueError("Requested task/model/target differs from source attribution")
    if any(row["task"] != args.task or row["model"] != args.model for row in rows):
        raise ValueError("Source task/model differs from requested study")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Selection limit must be positive")
        if apply_limit:
            rows = rows[:args.limit]
    return rows, manifest, digest


def replay_pipeline(args, manifest, *, solver_steps=None, allow_solver_change=False):
    configuration = manifest["configuration"] if manifest else {}
    identity = configuration.get("pipeline", {})
    if solver_steps is None:
        solver_steps = configuration.get("solver_steps")
    checkpointing = identity.get("gradient_checkpointing", True) and not args.no_checkpoint
    pipe = load_pipeline(args.model, enable_checkpoint=checkpointing, solver_steps=solver_steps,
                         checkpoint_mode=args.checkpoint_mode or identity.get("checkpoint_mode"),
                         checkpoint_path=args.checkpoint_path,
                         model_revision=args.model_revision or identity.get("model_revision"),
                         vision_revision=args.vision_revision or identity.get("vision_revision"))
    lang = load_lang(args.task, args.lang_dir)
    if manifest:
        check_replay_identity(pipe, lang, manifest, allow_solver_change=allow_solver_change)
    return pipe, lang


def sidecar_image(sidecar):
    value = sidecar["obs_image"]
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    return Image.fromarray(value)


def replay_context(args, row, sidecar, pipe, lang, *, verify_reference=True,
                   obs_image=None, frozen_reference=None, strict=True):
    ctx = prepare_ig_context(
        pipe["runner"], pipe["vision_model"], sidecar_image(sidecar) if obs_image is None else obs_image,
        sidecar["proprio"], lang["lang_tokens"], lang["lang_attn_mask"], lang["lang_tokens_baseline"],
        pipe["bg_image_encoded"], pipe["img_tokens_baseline"], pipe["action_mask"], pipe["ctrl_freqs"],
        seed=row["seed"], target=args.target, initial_noise=sidecar.get("initial_noise"),
        frozen_ref_action=frozen_reference)
    if strict:
        fields = ["initial_noise", "lang_attn_mask", "action_mask", "ctrl_freqs"]
        if verify_reference:
            fields.append("ref_action")
        for name in fields:
            if tensor_hash(ctx[name]) != tensor_hash(sidecar[name]):
                raise ValueError(f"Replayed {name} differs from source")
    return ctx


def preflight_source_replays(args, rows, manifest, pipe, lang, *, writer=None):
    """Replay every original reference before any controlled model/input change."""
    if manifest is None:
        return dict(status="legacy_unverified", verified=False, contexts=[])
    records=[]
    for row in rows:
        if writer is not None:
            writer.set_context(row,stage="source_replay_preflight")
        sidecar,sidecar_hash=load_sidecar(row,args.metrics,manifest)
        ctx=replay_context(args,row,sidecar,pipe,lang,verify_reference=True,strict=True)
        records.append(dict(context_id=row["context_id"],sidecar_sha256=sidecar_hash,
                            reference_sha256=tensor_hash(ctx["ref_action"]),
                            noise_sha256=tensor_hash(ctx["initial_noise"]),status="exact"))
        del ctx,sidecar
    return dict(status="all_original_references_exact",verified=True,contexts=records)


class EvaluationWriter:
    """Exclusive derived-output writer with durable failure/completion records."""

    def __init__(self, args, manifest, source_digest, study, configuration, expected_rows):
        self.args, self.source_manifest, self.source_digest = args, manifest, source_digest
        self.path = Path(args.out or f"runs/{study}_{uuid4().hex}/metrics.jsonl")
        self.expected_rows, self.completed = expected_rows, 0
        self.context = {}
        self.identity_status = "authenticated_source_controlled_intervention" if manifest else "legacy_unverified"
        self.manifest = dict(schema_version=2, evaluation_id=uuid4().hex, study=study,
                             source_metrics_sha256=source_digest, source_run_id=manifest["run_id"] if manifest else None,
                             source_configuration_sha256=manifest["configuration_sha256"] if manifest else None,
                             identity_status=self.identity_status, expected_rows=expected_rows,
                             selected_source_limit=args.limit, configuration=configuration,
                             shared_evaluator_sha256=file_hash(__file__))

    def __enter__(self):
        if self.path.exists() or self.path.with_name(self.path.name + ".manifest.json").exists():
            raise FileExistsError("Evaluation output already exists")
        if file_hash(self.args.metrics) != self.source_digest:
            raise ValueError("Source changed before evaluation")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("xb")
        atomic_bytes(self.path.with_name(self.path.name + ".manifest.json"), canonical_json(self.manifest)+b"\n")
        return self

    def set_context(self, row, **extra):
        self.context = {key: row.get(key) for key in ("episode", "seed", "policy_call_idx")}
        self.context.update(source_context_id=row.get("context_id"), source_attr_sha256=row.get("attr_sha256"))
        self.context.update(extra)

    def write(self, record):
        value = {**record, **self.context, "evaluation_id":self.manifest["evaluation_id"],
                 "schema_version":2, "identity_status":self.identity_status}
        self.handle.write(canonical_json(value)+b"\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.completed += 1

    def write_auxiliary(self, name, value):
        if Path(name).name != name or not name.endswith(".json"):
            raise ValueError("Auxiliary name must be a local JSON filename")
        path=self.path.with_name(self.path.name+"."+name)
        with path.open("xb") as handle:
            handle.write(canonical_json(value)+b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return file_hash(path)

    def __exit__(self, kind, exception, traceback):
        failure = exception
        try:
            if failure is None and file_hash(self.args.metrics) != self.source_digest:
                failure = ValueError("Source changed during evaluation")
        except OSError as exc:
            failure = failure or exc
        if failure is None and self.completed != self.expected_rows:
            failure = ValueError("Evaluation ended with missing records")
        try:
            if failure is not None:
                self.handle.write(canonical_json(dict(event="evaluation_failure", **self.context,
                    evaluation_id=self.manifest["evaluation_id"], completed_rows=self.completed,
                    error_type=type(failure).__name__, reason=str(failure),
                    diagnostics=getattr(failure,"diagnostics",{})))+b"\n")
                self.handle.flush()
                os.fsync(self.handle.fileno())
        finally:
            self.handle.close()
        if failure is not None:
            if exception is None:
                raise failure
            return False
        atomic_bytes(self.path.with_name(self.path.name+".completion.json"), canonical_json(dict(
            evaluation_id=self.manifest["evaluation_id"], completed_rows=self.completed,
            expected_rows=self.expected_rows, status="complete", output_sha256=file_hash(self.path)))+b"\n")
        return False


def main():
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if not math.isfinite(args.denominator_min) or args.denominator_min < 0:
        raise ValueError("--denominator-min must be finite and nonnegative")
    source_digest = file_hash(args.metrics)
    rows, source_manifest = authenticated_source(args.metrics, legacy=args.legacy_input)
    if file_hash(args.metrics) != source_digest:
        raise ValueError("Source metrics changed during authentication")
    configuration = source_manifest["configuration"] if source_manifest else {}
    pipeline_identity = configuration.get("pipeline", {})
    args.model = args.model or configuration.get("model", "170m")
    args.target = args.target or configuration.get("target", "logpi")
    solver_steps = args.solver_steps if args.solver_steps is not None else configuration.get("solver_steps")
    if source_manifest and any((configuration["task"] != args.task, configuration["model"] != args.model,
                                configuration["target"] != args.target, configuration["solver_steps"] != solver_steps)):
        raise ValueError("Task/model/target/solver request differs from the attribution source")
    if any(row["task"] != args.task or row["model"] != args.model for row in rows):
        raise ValueError("Source rows do not match requested task/model")
    if args.limit is not None:
        rows = rows[:args.limit]
    if args.out is None:
        args.out = f"runs/faithfulness_{uuid4().hex}/metrics.jsonl"
    output = Path(args.out)
    if output.exists() or output.with_name(output.name + ".manifest.json").exists():
        raise FileExistsError("Evaluation output already exists; choose a new path")

    pipe = load_pipeline(args.model, enable_checkpoint=not args.no_checkpoint, solver_steps=solver_steps,
                         checkpoint_mode=args.checkpoint_mode or pipeline_identity.get("checkpoint_mode"),
                         checkpoint_path=args.checkpoint_path,
                         model_revision=args.model_revision or pipeline_identity.get("model_revision"),
                         vision_revision=args.vision_revision or pipeline_identity.get("vision_revision"))
    lang = load_lang(args.task, args.lang_dir)
    if source_manifest:
        check_replay_identity(pipe, lang, source_manifest)
    if file_hash(args.metrics) != source_digest:
        raise ValueError("Source metrics changed while loading the evaluation pipeline")
    evaluation_id = uuid4().hex
    evaluation_manifest = dict(schema_version=2, evaluation_id=evaluation_id,
                               source_metrics_sha256=source_digest,
                               source_run_id=source_manifest["run_id"] if source_manifest else None,
                               source_configuration_sha256=source_manifest["configuration_sha256"] if source_manifest else None,
                               pipeline=pipe["identity"], language=lang["identity"], target=args.target,
                               response_units={"logpi":"auxiliary_quadratic_action_score", "l2":"action_L2",
                                               "l2sq":"squared_action_L2", "maxdev":"action_coordinate",
                                               "cosine":"cosine_similarity"}[args.target],
                               auc_axis=args.auc_grid, denominator_min=args.denominator_min,
                               nominal_grid_percent=AUC_K_GRID, selected_rows=len(rows), selection_limit=args.limit,
                               identity_status="authenticated_replay" if source_manifest else "legacy_unverified",
                               evaluator_sha256=file_hash(__file__))
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with output.open("xb") as handle:
        atomic_bytes(output.with_name(output.name + ".manifest.json"), canonical_json(evaluation_manifest) + b"\n")
        for row in rows:
            started = time.time()
            try:
                sidecar, sidecar_hash = load_sidecar(row, args.metrics, source_manifest)
                image = sidecar["obs_image"]
                if isinstance(image, torch.Tensor):
                    image = image.numpy()
                ctx = prepare_ig_context(
                    pipe["runner"], pipe["vision_model"], Image.fromarray(image), sidecar["proprio"],
                    lang["lang_tokens"], lang["lang_attn_mask"], lang["lang_tokens_baseline"],
                    pipe["bg_image_encoded"], pipe["img_tokens_baseline"], pipe["action_mask"], pipe["ctrl_freqs"],
                    seed=row["seed"], sigma_sq=1.0, target=args.target,
                    initial_noise=sidecar.get("initial_noise"))
                if source_manifest:
                    for name in ("ref_action", "initial_noise", "lang_attn_mask", "action_mask", "ctrl_freqs"):
                        if tensor_hash(ctx[name]) != tensor_hash(sidecar[name]):
                            raise ValueError(f"Replayed {name} differs from source sidecar")
                fwd_v, fwd_l, _ = build_forward_fns(ctx)
                device = ctx["img_adapted"].device
                # Preserve the stored FP32 attribution. Do not quantize ranking
                # inputs back to bf16 after the numerical accumulator repair.
                vision_attr = sidecar["vision_attr"].to(device)
                lang_attr = sidecar["lang_attr"].to(device)
                v_metrics = compute_modality_metrics(fwd_v, ctx["img_adapted"], ctx["img_adapted_bl"], vision_attr,
                                                     "vision", auc_grid=args.auc_grid, denominator_min=args.denominator_min)
                l_metrics = compute_modality_metrics(fwd_l, ctx["lang_adapted"], ctx["lang_adapted_bl"], lang_attr,
                                                     "language", real_mask=ctx["lang_attn_mask"].squeeze(0),
                                                     auc_grid=args.auc_grid, denominator_min=args.denominator_min)
                out_row = dict(event="faithfulness", schema_version=2, evaluation_id=evaluation_id,
                               task=args.task, model=args.model, target=args.target, episode=row["episode"],
                               seed=row["seed"], policy_call_idx=row["policy_call_idx"],
                               source_run_id=row.get("run_id"), source_context_id=row.get("context_id"),
                               source_attr_sha256=sidecar_hash, identity_status=evaluation_manifest["identity_status"],
                               ref_norm_maniskill=row["ref_norm_maniskill"], wall_seconds=time.time()-started,
                               k_grid_auc_nominal=AUC_K_GRID, k_grid_b1=B1_K,
                               **{f"vision_{key}": value for key,value in v_metrics.items()},
                               **{f"lang_{key}": value for key,value in l_metrics.items()})
                handle.write(canonical_json(out_row) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed += 1
                print(f"completed {completed}/{len(rows)}: episode {row['episode']}, call {row['policy_call_idx']}")
                del ctx, sidecar, vision_attr, lang_attr
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                failure = dict(event="evaluation_failure", evaluation_id=evaluation_id,
                               episode=row["episode"], policy_call_idx=row["policy_call_idx"],
                               source_context_id=row.get("context_id"), completed_rows=completed,
                               error_type=type(exc).__name__, reason=str(exc),
                               diagnostics=getattr(exc,"diagnostics",{}))
                handle.write(canonical_json(failure) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                raise
        if file_hash(args.metrics) != source_digest:
            handle.write(canonical_json(dict(event="evaluation_failure", evaluation_id=evaluation_id,
                                              completed_rows=completed, reason="Source metrics changed during evaluation")) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
            raise ValueError("Source metrics changed during evaluation; no completion record is issued")
    atomic_bytes(output.with_name(output.name + ".completion.json"), canonical_json(dict(
        evaluation_id=evaluation_id, completed_rows=completed, expected_rows=len(rows),
        output_sha256=file_hash(output), status="complete")) + b"\n")
    print(f"wrote {completed} evaluated rows to {output}")


if __name__ == "__main__":
    main()
