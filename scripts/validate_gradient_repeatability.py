"""Repeat path gradients and IG maps on a saved precision-validation cache.

Launch baseline and deterministic modes in separate fresh processes after a
recorded decision amendment. This module intentionally imports no torch at
module load: strict CUBLAS workspace configuration precedes torch startup.
The result measures observed repeat variation, not its cause or an accuracy
bound. No source context, upstream source or prior validator output is changed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configure_startup(mode):
    """Called before any torch import in the diagnostic process."""
    if mode not in ("baseline", "deterministic"):
        raise ValueError("Unknown execution mode")
    if "torch" in sys.modules:
        raise RuntimeError("Start a fresh process: torch was already imported")
    inherited = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if mode == "deterministic":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return {"mode": mode, "torch_not_preimported": True,
            "inherited_cublas_workspace_config": inherited,
            "effective_cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def rng_identity():
    import torch
    from experiment_io import tensor_hash
    values = {"cpu": tensor_hash(torch.get_rng_state())}
    if torch.cuda.is_initialized():
        values["cuda"] = [tensor_hash(value) for value in torch.cuda.get_rng_state_all()]
    return values


def path_gradient(forward, actual, baseline, alpha, forward_dtype):
    """One fresh graph using the same fp32 master path as the revised core."""
    import torch
    from integrated_gradients import _require_finite
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    if actual.shape != baseline.shape or actual.device != baseline.device:
        raise ValueError("Endpoints must have the same shape and device")
    actual, baseline = actual.detach().float(), baseline.detach().float()
    _require_finite(actual, "repeat_actual", alpha=alpha)
    _require_finite(baseline, "repeat_baseline", alpha=alpha)
    displacement = actual - baseline
    _require_finite(displacement, "repeat_displacement", alpha=alpha)
    point = baseline if alpha == 0 else actual if alpha == 1 else baseline + alpha * displacement
    _require_finite(point, "repeat_master_point", alpha=alpha)
    with torch.enable_grad():
        leaf = point.detach().requires_grad_(True)
        model_input = leaf.to(forward_dtype)
        _require_finite(model_input, "repeat_model_input", alpha=alpha)
        score = forward(model_input)
        if not isinstance(score, torch.Tensor) or score.numel() != 1 or not score.is_floating_point():
            raise ValueError("The forward must return a floating scalar tensor")
        _require_finite(score, "repeat_score", alpha=alpha)
        gradient = torch.autograd.grad(score, leaf)[0]
    _require_finite(gradient, "repeat_gradient", alpha=alpha)
    return score.detach().float().item(), gradient.detach(), point.detach()


def pairwise_variation(values, group_fn=None):
    """Compare every pair, avoiding a favorable single chosen repeat."""
    import torch
    from scripts.validate_rdt_numerics import comparison_metrics
    if len(values) < 2:
        raise ValueError("At least two repeats are required")
    pairs = []
    for left, right in itertools.combinations(range(len(values)), 2):
        a, b = values[left], values[right]
        options = {} if group_fn is None else {
            "reference_groups": group_fn(a), "candidate_groups": group_fn(b)}
        pairs.append({"reference_repeat": left, "candidate_repeat": right,
                      "bitwise_equal": torch.equal(a, b),
                      **comparison_metrics(a, b, **options)})
    defined_relative = [row["relative_l1_difference"] for row in pairs
                        if row["relative_l1_difference"] is not None]
    return {"pairs": pairs, "all_bitwise_equal": all(row["bitwise_equal"] for row in pairs),
            "max_absolute_coordinate_difference": max(row["max_absolute_coordinate_difference"] for row in pairs),
            "max_absolute_l1_difference": max(row["absolute_l1_difference"] for row in pairs),
            "max_relative_l1_difference": max(defined_relative) if defined_relative else None,
            "relative_denominator": "L1 norm of the lower-index repeat; undefined when zero"}


def _save(path, payload):
    import torch
    from experiment_io import file_hash
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return {"file": path.name, "sha256": file_hash(path)}


def load_authenticated_cache(source_report_path, manifest, row, payload, *, report_snapshot):
    """Bind a prepared cache to its prior validator and authenticated sidecar."""
    import torch
    from experiment_io import file_hash, strict_json, tensor_hash
    from scripts.validate_downstream_precision import cache_identity, CACHE_KEYS
    source_report_path = Path(source_report_path).resolve()
    # The caller saved one atomic read of a possibly still-running report.
    # Its hash must describe the exact parsed bytes, not a later progress file.
    source = strict_json(report_snapshot)
    expected = {"run_id": manifest["run_id"], "context_id": row["context_id"],
                "configuration_sha256": manifest["configuration_sha256"],
                "sidecar_sha256": row["attr_sha256"],
                "checkpoint_identity": manifest["configuration"]["pipeline"],
                "source_sha256": manifest["configuration"]["source_sha256"]}
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError("Precision cache report identity mismatch: " + key)
    relative = Path(source["cache_file"]["file"])
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("Cache file must be a sibling basename")
    cache_path = source_report_path.parent / relative
    if file_hash(cache_path) != source["cache_file"]["sha256"]:
        raise ValueError("Prepared cache file hash mismatch")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if set(cache) != set(CACHE_KEYS) or cache_identity(cache) != source["cached_tensors"]:
        raise ValueError("Prepared cache tensors differ from recorded identity")
    for key in ("initial_noise", "ref_action", "lang_attn_mask", "action_mask", "ctrl_freqs"):
        if tensor_hash(cache[key]) != tensor_hash(payload[key]):
            raise ValueError("Prepared cache differs from source sidecar: " + key)
    return cache, source, {"report_file": str(source_report_path),
        "report_snapshot": str(report_snapshot),
        "report_sha256": file_hash(report_snapshot), "cache_file": str(cache_path),
        "cache_sha256": file_hash(cache_path),
        "source_execution_status": source.get("execution_status", "incomplete_snapshot"),
        "qualification": "Cache identity is checked even if the originating precision run is unfinished."}


def _run(args, output, startup):
    import torch
    from experiment_io import atomic_bytes, canonical_json, file_hash, strict_json, tensor_hash
    from integrated_gradients import integrated_gradients
    from pipeline import load_pipeline
    from scripts.validate_downstream_precision import (
        authenticate_context, cache_identity, convert_downstream_to_fp32,
        fp32_math_settings, make_cached_functions, FP32OperationAudit,
    )
    from scripts.validate_rdt_numerics import group_values

    decision = strict_json(args.decision_file)
    if decision.get("decision_id") != args.decision_id:
        raise ValueError("Decision ID differs from recorded amendment")
    torch.use_deterministic_algorithms(args.mode == "deterministic", warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.mode == "deterministic"
    manifest, row, payload = authenticate_context(args.run, args.sidecar)
    snapshot = output / "source-precision-report.json"
    atomic_bytes(snapshot, args.cache_report.read_bytes())
    cache, source, cache_provenance = load_authenticated_cache(
        args.cache_report, manifest, row, payload, report_snapshot=snapshot)
    if args.mode == "baseline" and startup["effective_cublas_workspace_config"] != source["current_runtime"].get("cublas_workspace_config"):
        raise ValueError("Baseline process CUBLAS workspace differs from the source runtime")
    cache_hashes = cache_identity(cache)
    config, identity = manifest["configuration"], manifest["configuration"]["pipeline"]
    versions = {name: importlib.metadata.version(name) for name in config["environment"]}
    if versions != source["current_runtime"]["versions"]:
        raise ValueError("Package versions differ from the source precision run")
    pipe = load_pipeline(config["model"], enable_checkpoint=identity["gradient_checkpointing"],
        solver_steps=config["solver_steps"], checkpoint_mode=identity["checkpoint_mode"],
        checkpoint_path=args.checkpoint_path, model_revision=identity.get("model_revision"),
        vision_revision=identity["vision_revision"])
    if pipe["identity"] != identity:
        raise ValueError("Loaded pipeline identity differs from source")
    runner = pipe["runner"]
    conversion = convert_downstream_to_fp32(runner) if args.arm == "fp32" else None
    dtype = torch.float32 if args.arm == "fp32" else torch.bfloat16
    ctx, actions, scalars, pairs = make_cached_functions(runner, cache, dtype, config["target"])
    actual, baseline = pairs[args.modality]
    forward = scalars[args.modality]
    report = {"schema_version": 1, "execution_status": "running", "decision_id": args.decision_id,
        "decision_sha256": file_hash(args.decision_file), "startup": startup,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "run_id": manifest["run_id"], "context_id": row["context_id"],
        "configuration_sha256": manifest["configuration_sha256"],
        "sidecar_sha256": row["attr_sha256"], "source_sha256": config["source_sha256"],
        "cache_provenance": cache_provenance, "cached_tensors": cache_hashes,
        "script_sha256": file_hash(Path(__file__)),
        "helper_sha256": {name: file_hash(ROOT / name) for name in
            ("scripts/validate_downstream_precision.py", "scripts/validate_rdt_numerics.py")},
        "checkpoint_identity": identity, "conversion": conversion,
        "source_arm_math_settings": source.get("arms", {}).get(args.arm, {}).get("math_settings"),
        "setting_scope": "A new controlled protocol pins highest float32 matmul precision, disables matmul/cuDNN TF32 and autocast in BOTH arms/modes; source settings are preserved separately and are not assumed identical.",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
            "device_capability": list(torch.cuda.get_device_capability()),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "versions": versions},
        "gradient_probes": [], "map_repeats": [], "gradient_variation": {},
        "limits": ["Observed finite-repeat variation is not an error bound or a causal diagnosis.",
                   "Deterministic mode can reject unsupported operations; failure is diagnostic evidence.",
                   "Compare modes only on identical cache, weights, arm, target, path, budget and helper hashes.",
                   "One cached context does not establish population convergence."]}

    def checkpoint():
        atomic_bytes(output / "report.json", canonical_json(report) + b"\n")

    group_fn = lambda value: group_values(value.to(actual.device), args.modality, ctx)
    gradients = {alpha: [] for alpha in args.alphas}
    maps = []
    # This also controls fp32 operations inside the bf16 arm. It does not cast
    # its model to fp32: dtype still selects the cached-forward arm above.
    with fp32_math_settings() as settings:
        report["math_settings"] = settings
        with torch.no_grad():
            replay = actions[args.modality](actual).detach()
        report["actual_replay"] = {"action_sha256": tensor_hash(replay),
            "max_difference_from_stored_reference": (replay.float() - ctx["ref_action"]).abs().max().item(),
            "equal_to_stored_reference": torch.equal(replay.cpu(), payload["ref_action"])}
        if args.arm == "bf16" and not report["actual_replay"]["equal_to_stored_reference"]:
            raise ValueError("bf16 cached action does not exactly replay stored reference")
        if args.arm == "fp32":
            audit = FP32OperationAudit()
            with audit:
                path_gradient(forward, actual, baseline, .5, dtype)
            report["fp32_operation_audit"] = dict(audit.operations)
        checkpoint()
        # Repeat rounds preserve alpha order; every probe builds a fresh graph.
        for repeat in range(args.repeats):
            for alpha_index, alpha in enumerate(args.alphas):
                before, started = rng_identity(), time.perf_counter()
                score, gradient, point = path_gradient(forward, actual, baseline, alpha, dtype)
                after = rng_identity()
                gradient = gradient.cpu()
                artifact = _save(output / f"gradient-r{repeat}-a{alpha_index}.pt", {"gradient": gradient})
                gradients[alpha].append(gradient)
                report["gradient_probes"].append({"repeat": repeat, "alpha": alpha, "score": score,
                    "point_sha256": tensor_hash(point), "gradient_sha256": tensor_hash(gradient),
                    "gradient_l1": gradient.double().abs().sum().item(),
                    "rng_before": before, "rng_after": after, "rng_unchanged": before == after,
                    "elapsed_seconds": time.perf_counter() - started, **artifact})
                checkpoint()
        report["gradient_variation"] = {str(alpha): pairwise_variation(values, group_fn)
                                        for alpha, values in gradients.items()}
        checkpoint()
        for repeat in range(args.repeats):
            print(f"gradient repeatability: {args.mode}, {args.arm}, {args.modality}, m={args.budget}, repeat={repeat}", flush=True)
            before = rng_identity()
            result = integrated_gradients(forward, actual, baseline, m=args.budget,
                quadrature=args.quadrature, arithmetic_dtype=torch.float32, forward_dtype=dtype,
                return_result=True, diagnostic_context={"context_id": row["context_id"],
                    "modality": args.modality, "repeat": repeat, "mode": args.mode})
            after = rng_identity()
            value = result.attributions.detach().cpu()
            maps.append(value)
            artifact = _save(output / f"map-r{repeat}.pt", {"attribution": value})
            report["map_repeats"].append({"repeat": repeat, **result.diagnostics(),
                "rng_before": before, "rng_after": after, "rng_unchanged": before == after,
                "attribution_sha256": tensor_hash(value), **artifact})
            checkpoint()
        report["map_variation"] = pairwise_variation(maps, group_fn)
    if cache_identity(cache) != cache_hashes:
        raise ValueError("The fixed cache was mutated")
    report["execution_status"] = "complete"
    checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=["bf16", "fp32"], required=True)
    parser.add_argument("--mode", choices=["baseline", "deterministic"], required=True)
    parser.add_argument("--modality", choices=["vision", "language", "state"], required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--quadrature", choices=["trapezoid", "legacy_endpoint_average"], required=True)
    parser.add_argument("--checkpoint-path")
    args = parser.parse_args()
    if args.repeats < 2 or args.budget < 1:
        parser.error("Require at least two repeats and a positive interval count")
    if not all(math.isfinite(a) and 0 <= a <= 1 for a in args.alphas) or sorted(set(args.alphas)) != args.alphas:
        parser.error("Alphas must be distinct, increasing finite values in [0,1]")
    startup = configure_startup(args.mode)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        _run(args, output, startup)
    except Exception as error:
        from experiment_io import atomic_bytes, canonical_json, file_hash
        atomic_bytes(output / "failure.json", canonical_json({"decision_id": args.decision_id,
            "exception_type": type(error).__name__, "reason": str(error), "startup": startup,
            "script_sha256": file_hash(Path(__file__)),
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "diagnostics": getattr(error, "diagnostics", {})}) + b"\n")
        raise


if __name__ == "__main__":
    main()
