"""Engineering numerical comparisons on one authenticated stored RDT context.

This does not collect a rollout or certify historical maps. Record the job's
scientific question, context selection, budgets, criteria and stopping rule
before launching it, and pass that experiment-decision ID. Outputs are fresh
derivatives; the recorded run and sidecars are read only.
"""

import argparse
import itertools
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_io import RunStore, atomic_bytes, canonical_json, file_hash, object_hash, strict_json, tensor_hash
from integrated_gradients import _require_finite, integrated_gradients


def same_gradient_stream(forward, actual, baseline, m, *, context=None):
    """Isolate accumulation using one historical low precision path/gradient stream.

    Both comparison maps use the same (rounded historical) displacement and
    fp32 division/product. Their only difference is accumulation dtype. The
    separately returned historical map also uses low precision finishing
    arithmetic, so it must not be labeled an accumulation-only comparison.
    """
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError("m must be a positive integer")
    if actual.dtype != baseline.dtype or actual.shape != baseline.shape:
        raise ValueError("Historical stream requires matching input/baseline dtype and shape")
    _require_finite(actual, "stream_input", context=context)
    _require_finite(baseline, "stream_baseline", context=context)
    started = time.perf_counter()
    displacement = actual.detach() - baseline.detach()
    _require_finite(displacement, "stream_displacement", context=context)
    low_sum = torch.zeros_like(actual)
    fp32_sum = torch.zeros_like(actual, dtype=torch.float32)
    endpoints = {}
    for k in range(m + 1):
        alpha = k / m
        check = {"step": k, "alpha": alpha, "context": context}
        # Deliberately reconstruct endpoints as the historical implementation
        # did. A difference from the exact input is recorded, not concealed.
        with torch.enable_grad():
            point = (baseline + alpha * displacement).detach().requires_grad_(True)
            _require_finite(point, "stream_path", **check)
            score = forward(point)
            if not isinstance(score, torch.Tensor) or score.numel() != 1:
                raise ValueError("forward must return a scalar Tensor")
            _require_finite(score, "stream_score", **check)
            gradient = torch.autograd.grad(score, point)[0]
        _require_finite(gradient, "stream_gradient", **check)
        low_sum.add_(gradient)
        fp32_sum.add_(gradient.float())
        _require_finite(low_sum, "stream_low_precision_sum", **check)
        _require_finite(fp32_sum, "stream_fp32_sum", **check)
        if k in (0, m):
            endpoints[k] = score.detach().float().item()
        del point, score, gradient
    historical_map = displacement * (low_sum / (m + 1))
    historical_sum = historical_map.sum()
    _require_finite(historical_sum, "historical_attribution_sum", context=context)
    maps = {
        "same_stream_low_sum_fp32_finish": displacement.float() * (low_sum.float() / (m + 1)),
        "same_stream_fp32_sum_fp32_finish": displacement.float() * (fp32_sum / (m + 1)),
        "historical_arithmetic_diagnostic": historical_map.float(),
    }
    for name, value in maps.items():
        _require_finite(value, name, context=context)
    reconstructed_input = baseline + displacement
    return {
        "maps": maps,
        "metadata": {
            "m": m, "quadrature": "legacy_endpoint_average",
            "path_dtype": str(actual.dtype), "low_accumulation_dtype": str(actual.dtype),
            "high_accumulation_dtype": "torch.float32", "forward_evaluations": m + 1,
            "baseline_score": endpoints[0], "reconstructed_input_score": endpoints[m],
            "path_endpoint_gap": endpoints[m] - endpoints[0],
            "historical_attribution_sum_low_precision": historical_sum.item(),
            "max_endpoint_reconstruction_error": (reconstructed_input.float() - actual.float()).abs().max().item(),
            "nonfinite_count": 0, "elapsed_seconds": time.perf_counter() - started,
            "validity": "diagnostic_only_not_a_production_attribution",
        },
    }


def comparison_metrics(reference, candidate, *, reference_groups=None, candidate_groups=None):
    """Coordinate differences plus optional group-rank/top-five-percent agreement."""
    from scipy.stats import spearmanr

    _require_finite(reference, "comparison_reference")
    _require_finite(candidate, "comparison_candidate")
    a, b = reference.detach().double().flatten(), candidate.detach().double().flatten()
    if a.shape != b.shape or not a.numel():
        raise ValueError("Compared maps must have the same nonempty shape")
    difference = (a - b).abs()
    l1 = a.abs().sum().item()
    result = {
        "absolute_l1_difference": difference.sum().item(),
        "relative_l1_difference": difference.sum().item() / l1 if l1 else None,
        "max_absolute_coordinate_difference": difference.max().item(),
        "reference_signed_sum": a.sum().item(), "candidate_signed_sum": b.sum().item(),
    }
    if reference_groups is not None:
        _require_finite(reference_groups, "comparison_reference_groups")
        _require_finite(candidate_groups, "comparison_candidate_groups")
        a = reference_groups.detach().double().flatten().abs()
        b = candidate_groups.detach().double().flatten().abs()
        if a.shape != b.shape or not a.numel():
            raise ValueError("Compared groups must have the same nonempty shape")
        constant = bool((a == a[0]).all() or (b == b[0]).all())
        result["absolute_group_spearman"] = None if constant else float(spearmanr(a.cpu().numpy(), b.cpu().numpy()).statistic)
        k = max(1, int(round(0.05 * a.numel())))
        top_a = set(torch.argsort(a, descending=True, stable=True)[:k].cpu().tolist())
        top_b = set(torch.argsort(b, descending=True, stable=True)[:k].cpu().tolist())
        result.update(
            top_five_percent_count=k, eligible_groups=a.numel(),
            top_five_percent_overlap=len(top_a & top_b) / k,
            rank_status="constant_group_scores" if constant else "defined",
            tie_break="stable_coordinate_order",
            group_reduction="sum_of_absolute_coordinate_attributions",
        )
    return result


def group_values(attribution, modality, ctx):
    from per_step_attribution import MANISKILL_INDICES
    if modality == "state":
        return attribution.reshape(-1)[MANISKILL_INDICES]
    # Match the perturbation evaluator's ranking, not the signed plotting sum.
    values = attribution.abs().sum(dim=-1).squeeze(0)
    if modality == "language":
        return values[ctx["lang_attn_mask"].squeeze(0)]
    if values.numel() % 6:
        raise ValueError("Vision validation expects the documented six equal image slots")
    count = values.numel() // 6
    return values[3 * count:4 * count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Recorded metrics.jsonl path")
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--out", required=True, help="New directory; must not already exist")
    parser.add_argument("--budgets", type=int, nargs="+", required=True)
    parser.add_argument("--modalities", nargs="+", choices=["vision", "language", "state"],
                        default=["vision", "language", "state"])
    parser.add_argument("--lang-dir", default="data/lang_embeds")
    parser.add_argument("--checkpoint-path", default=None)
    args = parser.parse_args()
    if any(m < 1 for m in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Budgets must be distinct positive interval counts")
    if len(set(args.modalities)) != len(args.modalities):
        raise ValueError("Modalities must be distinct")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        _run(args, output)
    except Exception as error:
        atomic_bytes(output / "failure.json", canonical_json({
            "decision_id": args.decision_id, "reason": str(error),
            "diagnostics": getattr(error, "diagnostics", {}),
        }) + b"\n")
        raise


def _run(args, output):
    from PIL import Image
    from pipeline import load_lang, load_pipeline
    from per_step_attribution import build_forward_fns, prepare_ig_context

    run = Path(args.run).resolve()
    manifest_path = run.with_name(run.name + ".run") / "manifest.json"
    manifest = strict_json(manifest_path)
    configuration = manifest["configuration"]
    if object_hash(configuration) != manifest["configuration_sha256"]:
        raise ValueError("Run manifest configuration hash mismatch")
    # Read-only verification of immutable committed episodes and sidecars.
    store = RunStore(run, configuration, resume=True)
    store.manifest = manifest
    store.completed_episodes()
    sidecar_path = Path(args.sidecar).resolve()
    matching = [row for path in sorted((store.root / "episodes").glob("ep*.json"))
                for row in strict_json(path)["records"] if row.get("event") == "step"
                and (run.parent / row["attr_file"]).resolve() == sidecar_path]
    if len(matching) != 1:
        raise ValueError("The selected sidecar must belong to exactly one committed step")
    row = matching[0]
    if file_hash(sidecar_path) != row["attr_sha256"]:
        raise ValueError("Selected sidecar hash mismatch")
    payload = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    for key, expected in (("run_id", manifest["run_id"]),
                          ("configuration_sha256", manifest["configuration_sha256"]),
                          ("context_id", row["context_id"])):
        if payload[key] != expected:
            raise ValueError(f"Sidecar {key} mismatch")
    if tensor_hash(payload["initial_noise"]) != row["initial_noise_sha256"]:
        raise ValueError("Stored noise identity mismatch")
    for name, expected in configuration["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Source differs from recorded context: {name}; collect a newly identified context")
    pipeline_id = configuration["pipeline"]
    pipe = load_pipeline(
        configuration["model"], enable_checkpoint=pipeline_id["gradient_checkpointing"],
        solver_steps=configuration["solver_steps"], checkpoint_mode=pipeline_id["checkpoint_mode"],
        checkpoint_path=args.checkpoint_path, model_revision=pipeline_id.get("model_revision"),
        vision_revision=pipeline_id["vision_revision"],
    )
    language = load_lang(configuration["task"], args.lang_dir)
    if pipe["identity"] != pipeline_id or language["identity"] != configuration["language"]:
        raise ValueError("Loaded model/language identity differs from the selected context")
    ctx = prepare_ig_context(
        pipe["runner"], pipe["vision_model"], Image.fromarray(payload["obs_image"].numpy()),
        payload["proprio"], language["lang_tokens"], language["lang_attn_mask"],
        language["lang_tokens_baseline"], pipe["bg_image_encoded"], pipe["img_tokens_baseline"],
        pipe["action_mask"], pipe["ctrl_freqs"], seed=row["seed"],
        target=configuration["target"], initial_noise=payload["initial_noise"],
        frozen_ref_action=payload["ref_action"],
    )
    with torch.no_grad():
        replay = ctx["seeded_conditional_sample"](ctx["lang_adapted"], ctx["img_adapted"], ctx["state_traj_actual"])
    if not torch.equal(replay.cpu(), payload["ref_action"]):
        raise ValueError("Exact stored-reference replay failed; investigate identity or runtime determinism")
    fns = dict(zip(["vision", "language", "state"], build_forward_fns(ctx)))
    pairs = {
        "vision": (ctx["img_adapted"], ctx["img_adapted_bl"]),
        "language": (ctx["lang_adapted"], ctx["lang_adapted_bl"]),
        "state": (ctx["state_input_actual"], ctx["state_input_baseline"]),
    }
    report = {
        "decision_id": args.decision_id, "run_id": manifest["run_id"], "context_id": row["context_id"],
        "sidecar_sha256": row["attr_sha256"], "sampler": ctx["sampler_metadata"],
        "budgets": args.budgets, "modalities": args.modalities, "reference_replay": "exact",
        "script_sha256": file_hash(Path(__file__)), "results": [],
        "limits": ["Single engineering context; no population accuracy claim.",
                   "Downstream fp32 model control is not executed by this script.",
                   "No ranking efficacy claim or downstream perturbation assay is performed.",
                   "Budget stability and completeness do not prove coordinate accuracy."],
    }
    for modality in args.modalities:
        actual, baseline = pairs[modality]
        previous = None
        for budget in args.budgets:
            print(f"numerics: {modality}, m={budget}", flush=True)
            diagnostic_context = {"context_id": row["context_id"], "modality": modality}
            stream = same_gradient_stream(fns[modality], actual, baseline, budget, context=diagnostic_context)
            maps, metadata = dict(stream["maps"]), {"historical_stream": stream["metadata"]}
            for rule in ("legacy_endpoint_average", "trapezoid"):
                result = integrated_gradients(
                    fns[modality], actual, baseline, m=budget, quadrature=rule,
                    return_result=True, diagnostic_context=diagnostic_context,
                )
                name = "master_" + rule
                maps[name], metadata[name] = result.attributions, result.diagnostics()
            comparisons = []
            for left, right in itertools.combinations(maps, 2):
                comparisons.append({"reference": left, "candidate": right, **comparison_metrics(
                    maps[left], maps[right], reference_groups=group_values(maps[left], modality, ctx),
                    candidate_groups=group_values(maps[right], modality, ctx),
                )})
            adjacent_budget_comparisons = []
            if previous is not None:
                previous_budget, previous_maps = previous
                for name, previous_map in previous_maps.items():
                    current = maps[name].detach().cpu()
                    # Use CPU copies for adjacent-budget comparisons so the
                    # previous budget cannot retain large GPU maps indefinitely.
                    old_groups = group_values(previous_map.to(actual.device), modality, ctx).cpu()
                    new_groups = group_values(maps[name], modality, ctx).cpu()
                    adjacent_budget_comparisons.append({
                        "method": name, "reference_m": previous_budget, "candidate_m": budget,
                        **comparison_metrics(previous_map, current,
                            reference_groups=old_groups, candidate_groups=new_groups),
                    })
            previous = (budget, {name: maps[name].detach().cpu() for name in
                                 ("master_legacy_endpoint_average", "master_trapezoid")})
            path = output / f"{modality}_m{budget}_maps.pt"
            with path.open("xb") as handle:
                torch.save({name: value.detach().cpu() for name, value in maps.items()}, handle)
            report["results"].append({
                "modality": modality, "m": budget, "metadata": metadata, "comparisons": comparisons,
                "adjacent_budget_comparisons": adjacent_budget_comparisons,
                "maps_file": path.name, "maps_sha256": file_hash(path),
            })
            atomic_bytes(output / "report.json", canonical_json(report) + b"\n")


if __name__ == "__main__":
    main()
