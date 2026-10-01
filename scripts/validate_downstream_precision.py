"""Compare downstream RDT arithmetic on one authenticated, cached context.

The fp32 arm lifts the *bf16-rounded weight and representation values*. It is
not full-precision checkpoint recovery or end-to-end fp32 encoder evaluation.
Record a decision amendment before launching. This script never collects a
new episode and never modifies its source run or upstream source checkout.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import importlib.metadata
import itertools
import math
import os
from pathlib import Path
import platform
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_io import (RunStore, atomic_bytes, canonical_json, file_hash,
                           object_hash, strict_json, tensor_hash)
from integrated_gradients import _require_finite, integrated_gradients
from per_step_attribution import MANISKILL_INDICES
from rdt_sampling import conditional_sample_with_noise, sampler_metadata
from scripts.validate_rdt_numerics import comparison_metrics, group_values


CACHE_KEYS = (
    "lang_adapted", "lang_adapted_bl", "img_adapted", "img_adapted_bl",
    "state_input_actual", "state_input_baseline", "state_traj_actual",
    "action_mask", "ctrl_freqs", "lang_attn_mask", "initial_noise", "ref_action",
)
# These are explicit dtype attributes in the pinned upstream RDT source.
RDT_DTYPE_ATTRIBUTES = frozenset({
    "model.dtype", "model.t_embedder.dtype", "model.freq_embedder.dtype",
})


def cache_context(ctx):
    """Copy prepared inputs once. This function never calls an encoder/adaptor."""
    cache = {key: ctx[key].detach().cpu().clone() for key in CACHE_KEYS}
    for key, value in cache.items():
        _require_finite(value, "cache_" + key)
    return cache


def cache_identity(cache):
    return {key: {"sha256": tensor_hash(value), "dtype": str(value.dtype),
                  "shape": list(value.shape)} for key, value in cache.items()}


def _dtype_attributes(module):
    result = {}
    for module_name, child in module.named_modules():
        for name, value in vars(child).items():
            if isinstance(value, torch.dtype):
                result[f"{module_name}.{name}".lstrip(".")] = (child, name, value)
    return result


def _nested_plain_values(value, prefix, seen=None):
    """Inspect containers as well as direct attrs, without following modules."""
    seen = set() if seen is None else seen
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
        for key, child in value.items() if isinstance(value, dict) else enumerate(value):
            yield from _nested_plain_values(child, f"{prefix}[{key}]", seen)
    elif isinstance(value, (torch.Tensor, torch.dtype)):
        yield prefix, value


def storage_audit(module, expected_dtype):
    """Fail on missed floating parameters, buffers or plain tensor attributes."""
    counts = Counter()
    for kind, iterator in (("parameter", module.named_parameters()), ("buffer", module.named_buffers())):
        for name, value in iterator:
            if value.is_floating_point():
                if value.dtype != expected_dtype:
                    raise ValueError(f"Unexpected {kind} dtype: {name}={value.dtype}")
                _require_finite(value, "storage_" + name)
                counts[kind] += value.numel()
    for module_name, child in module.named_modules():
        for name, value in vars(child).items():
            # _parameters/_buffers were inspected above. An unregistered tensor
            # is not converted by Module.float(), and therefore fails closed.
            if name in {"_parameters", "_buffers", "_modules"}:
                continue
            for path, item in _nested_plain_values(value, f"{module_name}.{name}".lstrip(".")):
                if isinstance(item, torch.Tensor) and item.is_floating_point():
                    if item.dtype != expected_dtype:
                        raise ValueError(f"Unconverted plain tensor: {path}={item.dtype}")
                    _require_finite(item, "plain_tensor_" + path)
                    counts["plain_tensor"] += item.numel()
                elif isinstance(item, torch.dtype) and "[" in path:
                    raise ValueError(f"Uncontrolled nested dtype attribute: {path}={item}")
    return dict(counts)


def _value_hash(module):
    # Hash numeric values in fp32 so a dtype conversion can be distinguished
    # from changing weights. The hash binds tensor names and shapes as well.
    return object_hash({name: tensor_hash(value.detach().float())
                        for name, value in module.state_dict().items()})


def convert_downstream_to_fp32(runner, *, expected_attributes=RDT_DTYPE_ATTRIBUTES):
    """Convert this job's isolated runner, including audited plain dtype fields.

    No deep copy is used: checkpoint wrappers can capture original modules.
    The caller completes/saves the bf16 arm before this one-way conversion.
    Unexpected dtype attributes fail before mutation instead of guessing.
    """
    attributes = _dtype_attributes(runner)
    if set(attributes) != set(expected_attributes):
        raise ValueError(f"Dtype-attribute contract changed: found {sorted(attributes)}, expected {sorted(expected_attributes)}")
    if any(value != torch.bfloat16 for _, _, value in attributes.values()):
        raise ValueError("Expected the pinned bf16 dtype attributes before conversion")
    before_storage = storage_audit(runner, torch.bfloat16)
    before_values = _value_hash(runner)
    runner.float()
    changes = []
    for path, (child, name, value) in attributes.items():
        setattr(child, name, torch.float32)
        changes.append({"path": path, "before": str(value), "after": "torch.float32"})
    after_storage = storage_audit(runner, torch.float32)
    if any(value != torch.float32 for _, _, value in _dtype_attributes(runner).values()):
        raise ValueError("A floating dtype attribute survived conversion")
    after_values = _value_hash(runner)
    if before_values != after_values:
        raise ValueError("Converting to fp32 changed stored numerical weight/buffer values")
    return {"dtype_attributes": changes, "before_storage": before_storage,
            "after_storage": after_storage, "fp32_numeric_values_sha256": after_values,
            "value_preservation": "exact", "weight_scope": "bf16-rounded values lifted to fp32"}


class FP32OperationAudit(TorchDispatchMode):
    """Sentinel audit catches hidden low precision casts inside functional code.

    Integer/bool tensors are permitted for timesteps/masks. A successful audit
    is about observable tensor dtypes; it does not prove exact real arithmetic.
    """
    def __init__(self):
        super().__init__()
        self.operations = Counter()

    def _check(self, value, stage, operation):
        for item in tree_flatten(value)[0]:
            if isinstance(item, torch.Tensor) and item.is_floating_point() and item.dtype != torch.float32:
                raise ValueError(f"Non-fp32 {stage} in {operation}: {item.dtype}")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        operation = str(func)
        self._check((args, kwargs), "input", operation)
        result = func(*args, **kwargs)
        self._check(result, "output", operation)
        self.operations[operation] += 1
        return result


@contextmanager
def fp32_math_settings():
    """Disable autocast and reduced-precision float32 matmul for this arm only."""
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            yield {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                   "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                   "autocast": "disabled"}
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def response_scores(prediction, reference, *, sigma_sq=1.0):
    """Common fp32 readouts. Reference is frozen across every precision arm."""
    _require_finite(prediction, "response_prediction")
    _require_finite(reference, "response_reference")
    if not math.isfinite(sigma_sq) or sigma_sq <= 0:
        raise ValueError("sigma_sq must be finite and positive")
    active = prediction[..., MANISKILL_INDICES].float()
    difference = active - reference[..., MANISKILL_INDICES].float()
    squared = difference.square()
    return {"quadratic": -0.5 / sigma_sq * squared.mean(),
            "l2": -(squared.sum() + 1e-12).sqrt()}


def make_cached_functions(runner, cache, dtype, target, *, sigma_sq=1.0):
    """Build downstream functions without re-encoding or readapting conditions."""
    if target not in ("logpi", "l2"):
        raise ValueError("This precision protocol supports logpi or l2 attribution targets")
    device = next(runner.parameters()).device
    ctx = {}
    for key, value in cache.items():
        output_dtype = torch.bool if key == "lang_attn_mask" else dtype
        # The reference/noise stay in fp32 with exactly the recorded values.
        if key in ("initial_noise", "ref_action"):
            output_dtype = torch.float32
        ctx[key] = value.detach().to(device=device, dtype=output_dtype).clone()
        if not torch.equal(ctx[key].cpu().to(value.dtype), value):
            raise ValueError(f"Precision arm changes cached numeric values: {key}")

    def sample(language, image, state):
        return conditional_sample_with_noise(
            runner, language, ctx["lang_attn_mask"], image, state,
            ctx["action_mask"], ctx["ctrl_freqs"], ctx["initial_noise"],
        )

    def vision(value):
        return sample(ctx["lang_adapted"], value, ctx["state_traj_actual"])

    def language(value):
        return sample(value, ctx["img_adapted"], ctx["state_traj_actual"])

    def state(value):
        token = runner.state_adaptor(torch.cat([value, ctx["action_mask"]], dim=2))
        return sample(ctx["lang_adapted"], ctx["img_adapted"], token)

    actions = {"vision": vision, "language": language, "state": state}
    response_key = "quadratic" if target == "logpi" else "l2"
    scalars = {name: (lambda x, action=action: response_scores(action(x), ctx["ref_action"],
                          sigma_sq=sigma_sq)[response_key]) for name, action in actions.items()}
    pairs = {"vision": (ctx["img_adapted"], ctx["img_adapted_bl"]),
             "language": (ctx["lang_adapted"], ctx["lang_adapted_bl"]),
             "state": (ctx["state_input_actual"], ctx["state_input_baseline"])}
    return ctx, actions, scalars, pairs


def repeat_endpoints(action, actual, baseline, reference, repeats):
    if type(repeats) is not int or repeats < 2:
        raise ValueError("At least two endpoint repeats are required")
    result, samples = {}, {}
    with torch.no_grad():
        for name, value in (("actual", actual), ("baseline", baseline)):
            outputs = [action(value).detach() for _ in range(repeats)]
            rows = [{key: score.item() for key, score in response_scores(out, reference).items()}
                    for out in outputs]
            result[name] = {"actions_sha256": [tensor_hash(out) for out in outputs],
                            "scores": rows, "bitwise_repeatable": all(torch.equal(outputs[0], out) for out in outputs),
                            "max_action_repeat_difference": max((outputs[0].float() - out.float()).abs().max().item() for out in outputs),
                            "frozen_reference_max_difference": (outputs[0].float() - reference.float()).abs().max().item()}
            samples[name] = outputs[0].cpu()
    result["gap_repeat_range"] = {
        key: [min(row[key] for row in result["actual"]["scores"]) - max(row[key] for row in result["baseline"]["scores"]),
              max(row[key] for row in result["actual"]["scores"]) - min(row[key] for row in result["baseline"]["scores"])]
        for key in ("quadratic", "l2")}
    result["qualification"] = "Repeated exactness does not calibrate rounding error or establish a safe near-zero denominator."
    return result, samples


def eligible_groups(modality, actual, ctx):
    if modality == "state":
        return torch.tensor(MANISKILL_INDICES, device=actual.device, dtype=torch.long)
    if modality == "language":
        return ctx["lang_attn_mask"][0].nonzero().flatten()
    if actual.shape[1] % 6:
        raise ValueError("Expected six equal vision slots")
    count = actual.shape[1] // 6
    return torch.arange(3 * count, 4 * count, device=actual.device)


def ranked_group_indices(attributions, modality, ctx, actual):
    groups = group_values(attributions.to(actual.device), modality, ctx).abs()
    eligible = eligible_groups(modality, actual, ctx)
    if groups.numel() != eligible.numel() or not groups.numel():
        raise ValueError("Group scores and eligible coordinates disagree")
    return eligible[torch.argsort(groups, descending=True, stable=True)]


def replace_groups(start, donor, indices, modality):
    value = start.clone()
    if modality == "state":
        value[..., indices] = donor[..., indices]
    else:
        value[:, indices, :] = donor[:, indices, :]
    return value


def normalize_curve(scores, baseline_score, actual_score, realized_fractions):
    """Unclipped normalization; zero signed gaps have undefined AUC."""
    if len(scores) != len(realized_fractions) or len(scores) < 2:
        raise ValueError("A curve needs matching score/fraction arrays of length at least two")
    if not all(math.isfinite(value) for value in [baseline_score, actual_score, *scores, *realized_fractions]):
        raise ValueError("Curve values and endpoints must be finite")
    if sorted(set(realized_fractions)) != list(realized_fractions):
        raise ValueError("Realized fractions must increase strictly")
    gap = actual_score - baseline_score
    if gap == 0:
        return {"signed_gap": gap, "normalized": None, "auc": None, "status": "zero_gap"}
    values = [(score - baseline_score) / gap for score in scores]
    auc = sum((right - left) * (a + b) / 2 for left, right, a, b in
              zip(realized_fractions[:-1], realized_fractions[1:], values[:-1], values[1:]))
    if not all(math.isfinite(value) for value in [gap, *values, auc]):
        raise FloatingPointError("Nonfinite normalized curve")
    return {"signed_gap": gap, "normalized": values, "auc": auc,
            "status": "nonzero_gap_conditioning_not_certified"}


def common_response_curves(action, actual, baseline, reference, rankings, modality, fractions):
    """Score every ranking through the SAME fp32 model and frozen reference.

    Raw prediction tensors are returned for independent response reconstruction.
    No gradients from these perturbations are reused to construct rankings.
    """
    if not fractions or not all(math.isfinite(value) for value in fractions) or sorted(set(fractions)) != list(fractions) or fractions[0] != 0 or fractions[-1] != 1:
        raise ValueError("Fractions must be strictly increasing, from 0 through 1")
    if actual.dtype != torch.float32 or baseline.dtype != torch.float32:
        raise ValueError("Common response requires fp32 cached inputs")
    if not rankings:
        raise ValueError("At least one ranking is required")
    population = sorted(next(iter(rankings.values())).cpu().tolist())
    if any(sorted(ordering.cpu().tolist()) != population for ordering in rankings.values()):
        raise ValueError("Compared rankings must permute exactly the same eligible groups")
    report, predictions = {}, {}
    with torch.no_grad():
        endpoint_actions = {"actual": action(actual).detach(), "baseline": action(baseline).detach()}
        endpoint_scores = {name: {key: value.item() for key, value in response_scores(pred, reference).items()}
                           for name, pred in endpoint_actions.items()}
        predictions.update({"endpoint_" + key: value.cpu() for key, value in endpoint_actions.items()})
        for name, ordering in rankings.items():
            if ordering.numel() == 0 or ordering.unique().numel() != ordering.numel():
                raise ValueError("A ranking must be a nonempty permutation of eligible groups")
            n = ordering.numel()
            counts = sorted(set([int(round(fraction * n)) for fraction in fractions] + [max(1, int(round(.05 * n)))]))
            realized = [count / n for count in counts]
            curves = {}
            for direction, start, donor in (("deletion", actual, baseline), ("insertion", baseline, actual)):
                outputs = []
                for count in counts:
                    value = replace_groups(start, donor, ordering[:count], modality)
                    outputs.append(action(value).detach())
                matrix = torch.stack(outputs).cpu()
                predictions[name + "_" + direction] = matrix
                scores = {key: [response_scores(pred, reference)[key].item() for pred in outputs]
                          for key in ("quadratic", "l2")}
                curves[direction] = {"raw_scores": scores, "responses": {
                    key: normalize_curve(values, endpoint_scores["baseline"][key], endpoint_scores["actual"][key], realized)
                    for key, values in scores.items()}}
                five_index = counts.index(max(1, int(round(.05 * n))))
                curves[direction]["top_five_percent"] = {
                    "realized_fraction": realized[five_index],
                    "quadratic_change_from_actual": endpoint_scores["actual"]["quadratic"] - scores["quadratic"][five_index],
                    "l2_change_from_actual": endpoint_scores["actual"]["l2"] - scores["l2"][five_index],
                    "active_action_displacement_from_actual": (outputs[five_index][..., MANISKILL_INDICES] - endpoint_actions["actual"][..., MANISKILL_INDICES]).norm().item(),
                }
            report[name] = {"ordered_group_indices": ordering.cpu().tolist(), "eligible_groups": n,
                            "replacement_counts": counts, "realized_fractions": realized, "curves": curves}
    comparisons = []
    for left, right in itertools.combinations(report, 2):
        for direction in ("deletion", "insertion"):
            for response in ("quadratic", "l2"):
                a, b = report[left]["curves"][direction], report[right]["curves"][direction]
                auc_a, auc_b = a["responses"][response]["auc"], b["responses"][response]["auc"]
                comparisons.append({"left": left, "right": right, "direction": direction,
                    "response": response, "auc_right_minus_left": None if auc_a is None or auc_b is None else auc_b - auc_a,
                    "raw_score_right_minus_left": [y - x for x, y in zip(a["raw_scores"][response], b["raw_scores"][response])],
                    "top5_change_right_minus_left": b["top_five_percent"][response + "_change_from_actual"] - a["top_five_percent"][response + "_change_from_actual"]})
    return {"endpoints": endpoint_scores, "rankings": report, "paired_comparisons": comparisons,
            "normalization": "(score - baseline_score)/(actual_score - baseline_score), no clipping",
            "response_definition": {"quadratic": "-sum(active squared residual)/(2*D)",
                                    "l2": "-sqrt(sum(active squared residual)+1e-12)",
                                    "D": reference[..., MANISKILL_INDICES].numel()},
            "readout": "common fp32 downstream sampler; same fixed bf16 reference values",
            "interpretation": "Within-context numerical ranking stability, not attribution efficacy."}, predictions


def authenticate_context(run, sidecar):
    run, sidecar = Path(run).resolve(), Path(sidecar).resolve()
    manifest = strict_json(run.with_name(run.name + ".run") / "manifest.json")
    config = manifest["configuration"]
    if object_hash(config) != manifest["configuration_sha256"]:
        raise ValueError("Run configuration hash mismatch")
    store = RunStore(run, config, resume=True)
    store.manifest = manifest
    store.completed_episodes()
    rows = [row for path in sorted((store.root / "episodes").glob("ep*.json"))
            for row in strict_json(path)["records"] if row.get("event") == "step"
            and (run.parent / row["attr_file"]).resolve() == sidecar]
    if len(rows) != 1 or file_hash(sidecar) != rows[0]["attr_sha256"]:
        raise ValueError("Sidecar must match exactly one intact committed step")
    row = rows[0]
    payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    for name, expected in (("run_id", manifest["run_id"]), ("configuration_sha256", manifest["configuration_sha256"]),
                           ("context_id", row["context_id"])):
        if payload.get(name) != expected:
            raise ValueError(f"Sidecar {name} mismatch")
    if tensor_hash(payload["initial_noise"]) != row["initial_noise_sha256"]:
        raise ValueError("Stored noise hash mismatch")
    expected_context = object_hash({"episode": row["episode"], "policy_call": row["policy_call_idx"],
        "configuration": manifest["configuration_sha256"], "observation": tensor_hash(payload["obs_image"]),
        "proprio": tensor_hash(payload["proprio"]), "noise": row["initial_noise_sha256"],
        "reference": tensor_hash(payload["ref_action"])})
    if expected_context != row["context_id"]:
        raise ValueError("Context identity does not bind the stored inputs/reference")
    for name, expected in config["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Source differs from the recorded context: {name}")
    return manifest, row, payload


def _save_tensor_file(path, values):
    with path.open("xb") as stream:
        torch.save(values, stream)
    return {"file": path.name, "sha256": file_hash(path)}


def _run(args, output):
    from PIL import Image
    from pipeline import load_lang, load_pipeline
    from per_step_attribution import prepare_ig_context

    decision = strict_json(args.decision_file)
    if decision.get("decision_id") != args.decision_id:
        raise ValueError("Decision ID does not match the recorded protocol amendment")
    manifest, row, payload = authenticate_context(args.run, args.sidecar)
    config, identity = manifest["configuration"], manifest["configuration"]["pipeline"]
    target = args.target or config['target']
    pipe = load_pipeline(config["model"], enable_checkpoint=identity["gradient_checkpointing"],
        solver_steps=config["solver_steps"], checkpoint_mode=identity["checkpoint_mode"],
        checkpoint_path=args.checkpoint_path, model_revision=identity.get("model_revision"),
        vision_revision=identity["vision_revision"])
    language = load_lang(config["task"], args.lang_dir)
    if pipe["identity"] != identity or language["identity"] != config["language"]:
        raise ValueError("Model/language identity differs from the source run")
    for key, value in (("lang_attn_mask", language["lang_attn_mask"]), ("action_mask", pipe["action_mask"]),
                       ("ctrl_freqs", pipe["ctrl_freqs"])):
        if not torch.equal(payload[key], value.cpu()):
            raise ValueError("Source mask/frequency differs: " + key)
    prepared = prepare_ig_context(pipe["runner"], pipe["vision_model"],
        Image.fromarray(payload["obs_image"].numpy()), payload["proprio"],
        language["lang_tokens"], language["lang_attn_mask"], language["lang_tokens_baseline"],
        pipe["bg_image_encoded"], pipe["img_tokens_baseline"], pipe["action_mask"], pipe["ctrl_freqs"],
        seed=row["seed"], target=config["target"], initial_noise=payload["initial_noise"],
        frozen_ref_action=payload["ref_action"])
    cache = cache_context(prepared)
    cache_id = cache_identity(cache)
    cache_file = _save_tensor_file(output / "prepared-bf16-context.pt", cache)
    # The only encoder/preprocessing invocation was above, before either arm.
    runner = pipe["runner"]
    del prepared, language
    report = {"decision_id": args.decision_id, "decision_sha256": file_hash(args.decision_file),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "run_id": manifest["run_id"], "context_id": row["context_id"],
        "attribution_target": target, "source_target_convention": config['target'],
        "configuration_sha256": manifest["configuration_sha256"], "checkpoint_identity": identity,
        "source_sha256": config["source_sha256"], "script_sha256": file_hash(Path(__file__)),
        "comparison_helper_sha256": file_hash(ROOT / "scripts/validate_rdt_numerics.py"),
        "sidecar_sha256": row["attr_sha256"], "cached_tensors": cache_id, "cache_file": cache_file,
        "source_runtime": config["environment"],
        "current_runtime": {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "device_capability": list(torch.cuda.get_device_capability()),
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "versions": {name: importlib.metadata.version(name) for name in config["environment"]}},
        "path": {"master_dtype": "torch.float32", "quadrature": args.quadrature,
                 "endpoints": "exact cached values; no subtract/add endpoint reconstruction",
                 "bf16_path": "fp32 master point cast to bf16 before downstream forward",
                 "fp32_path": "identical fp32 master point without bf16 cast",
                 "state_mapping": "raw state -> precision-specific state adaptor on state path; vision/language hold the cached bf16 actual state token fixed"},
        "arms": {}, "comparisons": [], "common_response": {},
        "limits": ["One engineering context, no population conclusion.",
                   "fp32 arithmetic on bf16-rounded weights and adapted features, not full-precision checkpoint recovery.",
                   "Scalar completeness and budget agreement do not prove coordinate accuracy."]}

    def checkpoint():
        atomic_bytes(output / "report.json", canonical_json(report) + b"\n")

    checkpoint()
    all_maps = {}
    for arm, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        if arm == "fp32":
            report["conversion"] = convert_downstream_to_fp32(runner)
        ctx, actions, scalars, pairs = make_cached_functions(runner, cache, dtype, target)
        report["arms"][arm] = {"sampler": sampler_metadata(runner, ctx["initial_noise"]),
                               "endpoint_repeats": {}, "maps": {}, "operation_audits": {}}
        with fp32_math_settings() if arm == "fp32" else nullcontext() as settings:
            report["arms"][arm]["math_settings"] = settings or {
                "precision": "source-runtime settings held for exact bf16 replay",
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
            }
            with torch.no_grad():
                state_token = runner.state_adaptor(torch.cat([ctx["state_input_actual"], ctx["action_mask"]], dim=2))
            report["arms"][arm]["state_actual_mapping"] = {
                "fresh_state_token_sha256": tensor_hash(state_token),
                "cached_state_token_sha256": tensor_hash(ctx["state_traj_actual"]),
                "max_difference_from_cached_state_token": (state_token.float() - ctx["state_traj_actual"].float()).abs().max().item(),
                "cached_token_held_fixed_for": ["vision", "language"],
                "fresh_token_used_for": ["state"],
            }
            for modality in args.modalities:
                actual, baseline = pairs[modality]
                repeats, samples = repeat_endpoints(actions[modality], actual, baseline, ctx["ref_action"], args.repeats)
                report["arms"][arm]["endpoint_repeats"][modality] = repeats
                report["arms"][arm]["endpoint_repeats"][modality]["action_file"] = _save_tensor_file(output / f"{arm}-{modality}-endpoints.pt", samples)
                if arm == "bf16" and (not torch.equal(samples["actual"], payload["ref_action"]) or
                        not repeats["actual"]["bitwise_repeatable"]):
                    raise ValueError("Exact stored-reference bf16 replay failed for " + modality)
                if arm == "fp32":
                    audit = FP32OperationAudit()
                    gradient_norms = []
                    with audit, torch.enable_grad():
                        for point in (baseline, (baseline + actual) / 2, actual):
                            leaf = point.detach().requires_grad_(True)
                            score = scalars[modality](leaf)
                            gradient = torch.autograd.grad(score, leaf)[0]
                            _require_finite(score, "fp32_sentinel_score")
                            _require_finite(gradient, "fp32_sentinel_gradient")
                            gradient_norms.append(gradient.norm().item())
                    report["arms"][arm]["operation_audits"][modality] = {"points": [0, .5, 1],
                        "includes_backward": True, "gradient_norms": gradient_norms, "operations": dict(audit.operations)}
                checkpoint()
                for budget in args.budgets:
                    print(f"downstream precision: {arm}, {modality}, m={budget}", flush=True)
                    result = integrated_gradients(scalars[modality], actual, baseline, m=budget,
                        quadrature=args.quadrature, arithmetic_dtype=torch.float32, forward_dtype=dtype,
                        return_result=True, diagnostic_context={"context_id": row["context_id"], "modality": modality, "arm": arm})
                    key = f"{modality}_m{budget}"
                    all_maps[arm, key] = result.attributions.detach().cpu()
                    artifact = _save_tensor_file(output / f"{arm}-{key}.pt", {"attribution": all_maps[arm, key]})
                    report["arms"][arm]["maps"][key] = {**result.diagnostics(), **artifact}
                    checkpoint()
            if arm == "fp32":
                for modality in args.modalities:
                    actual, baseline = pairs[modality]
                    for budget in args.budgets:
                        key = f"{modality}_m{budget}"
                        low, high = all_maps["bf16", key], all_maps["fp32", key]
                        report["comparisons"].append({"modality": modality, "m": budget,
                            "reference": "fp32", "candidate": "bf16", **comparison_metrics(high, low,
                            reference_groups=group_values(high.to(actual.device), modality, ctx),
                            candidate_groups=group_values(low.to(actual.device), modality, ctx))})
                        rankings = {name: ranked_group_indices(all_maps[name, key], modality, ctx, actual) for name in ("bf16", "fp32")}
                        curves, predictions = common_response_curves(actions[modality], actual, baseline,
                            ctx["ref_action"], rankings, modality, args.fractions)
                        curves["prediction_file"] = _save_tensor_file(output / f"common-response-{key}.pt", predictions)
                        report["common_response"][key] = curves
                        checkpoint()
                for arm_name in ("bf16", "fp32"):
                    for modality in args.modalities:
                        for left, right in zip(args.budgets[:-1], args.budgets[1:]):
                            low, high = all_maps[arm_name, f"{modality}_m{left}"], all_maps[arm_name, f"{modality}_m{right}"]
                            report["comparisons"].append({"arm": arm_name, "modality": modality,
                                "reference_m": right, "candidate_m": left, **comparison_metrics(high, low,
                                reference_groups=group_values(high.to(actual.device), modality, ctx),
                                candidate_groups=group_values(low.to(actual.device), modality, ctx))})
        if cache_identity(cache) != cache_id:
            raise ValueError("A precision arm mutated the fixed cache")
        checkpoint()
    report["execution_status"] = "complete"
    checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--quadrature", choices=["trapezoid", "legacy_endpoint_average"], required=True)
    parser.add_argument("--fractions", type=float, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--modalities", nargs="+", choices=["vision", "language", "state"], default=["vision", "language", "state"])
    parser.add_argument("--lang-dir", default="data/lang_embeds")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--target", choices=["logpi", "l2"],
                        help="Explicit controlled score choice on the same authenticated context/reference")
    args = parser.parse_args()
    if sorted(set(args.budgets)) != args.budgets or not args.budgets or min(args.budgets) < 1:
        parser.error("Budgets must be distinct, increasing positive integers")
    if args.repeats < 2 or len(set(args.modalities)) != len(args.modalities):
        parser.error("Require at least two repeats and distinct modalities")
    if not all(math.isfinite(value) for value in args.fractions) or sorted(set(args.fractions)) != args.fractions or args.fractions[0] != 0 or args.fractions[-1] != 1:
        parser.error("Fractions must increase strictly from 0 through 1")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        _run(args, output)
    except Exception as error:
        atomic_bytes(output / "failure.json", canonical_json({"decision_id": args.decision_id,
            "reason": str(error), "exception_type": type(error).__name__,
            "script_sha256": file_hash(Path(__file__)),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "diagnostics": getattr(error, "diagnostics", {})}) + b"\n")
        raise


if __name__ == "__main__":
    main()
