"""Shared immutable prepared-cache authentication for fp32 probe workflows.

Importing this module does not import torch; strict startup remains possible.
Call verify_preparation before load_prepared_context. Both return read-only
provenance and CPU tensors and never execute a model or modify a source run.
"""
from pathlib import Path
import sys


def process_startup(stage, source_settings=None):
    """Configure CUBLAS in a fresh process, before importing torch."""
    import os
    if "torch" in sys.modules:
        raise RuntimeError("A fresh process is required before torch import")
    inherited = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    value = ":4096:8" if stage == "run" else source_settings["cublas_workspace_config"]
    if value is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG",None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = value
    return dict(stage=stage,torch_not_preimported=True,inherited_cublas_workspace_config=inherited,
                effective_cublas_workspace_config=value)


def runtime_settings():
    import os
    import torch
    return dict(cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)


def apply_settings(settings):
    import torch
    import os
    if set(settings) != set(runtime_settings()):
        raise ValueError("Every runtime setting must be declared explicitly")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != settings["cublas_workspace_config"]:
        raise ValueError("CUBLAS settings differ from the fresh-process declaration")
    torch.use_deterministic_algorithms(settings["deterministic_algorithms"],warn_only=settings["deterministic_warn_only"])
    torch.backends.cudnn.benchmark=settings["cudnn_benchmark"]
    torch.backends.cudnn.deterministic=settings["cudnn_deterministic"]
    torch.set_float32_matmul_precision(settings["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32=settings["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32=settings["cudnn_allow_tf32"]
    if runtime_settings()!=settings:
        raise ValueError("Requested runtime settings were not applied exactly")
    return runtime_settings()


def strict_candidate_settings():
    return dict(cublas_workspace_config=":4096:8",deterministic_algorithms=True,deterministic_warn_only=False,
                cudnn_benchmark=False,cudnn_deterministic=True,float32_matmul_precision="highest",
                cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False)


def verify_preparation(directory,expected_completion_hash):
    from experiment_io import file_hash,strict_json
    directory=Path(directory).resolve()
    completion_path=directory/"completion.json"
    if file_hash(completion_path)!=expected_completion_hash:
        raise ValueError("Prepared completion hash mismatch")
    completion=strict_json(completion_path)
    if "report.json" not in completion.get("artifacts_sha256", {}):
        raise ValueError("Prepared completion does not bind its report")
    if completion.get("status")!="complete_source_preparation":
        raise ValueError("Source preparation is incomplete")
    for name,digest in completion["artifacts_sha256"].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory) or file_hash(path)!=digest:
            raise ValueError("Prepared artifact path or byte hash mismatch")
    report=strict_json(directory/"report.json")
    if report.get("kind") != "fp32_probe_source_preparation":
        raise ValueError("Unknown prepared-cache contract")
    if report.get("status")!="complete_source_preparation":
        raise ValueError("Prepared report is incomplete")
    return report


def load_prepared_context(directory,report,entry,manifest,source_row,payload):
    """Reusable read-only prepared-cache contract for the production launcher."""
    import torch
    from experiment_io import file_hash,tensor_hash,object_hash
    from scripts.validate_downstream_precision import CACHE_KEYS,cache_identity
    if report["source_run_id"]!=manifest["run_id"] or report["source_configuration_sha256"]!=manifest["configuration_sha256"]:
        raise ValueError("Prepared source run/configuration identity mismatch")
    if report["source_pipeline"]!=manifest["configuration"]["pipeline"] or report["source_language"]!=manifest["configuration"]["language"]:
        raise ValueError("Prepared checkpoint or language identity mismatch")
    if report["source_forward_sha256"] != manifest["configuration"]["source_sha256"]:
        raise ValueError("Prepared forward source differs")
    if payload["context_id"] != source_row["context_id"] or source_row["context_id"] != entry["context_id"]:
        raise ValueError("Prepared context source identity differs")
    if object_hash(source_row) != entry["row_sha256"]:
        raise ValueError("Prepared source row hash differs")
    matches=[item for item in report["contexts"] if item["context_id"]==entry["context_id"]]
    if len(matches)!=1:
        raise ValueError("Prepared context must match exactly one selected bank entry")
    item=matches[0]
    if not item.get("source_replay_verified") or item["sidecar_sha256"]!=source_row["attr_sha256"]:
        raise ValueError("Prepared source replay or sidecar binding is absent")
    for name in ("episode","policy_call_idx","episode_id","reset_seed"):
        if item[name]!=entry[name]:
            raise ValueError("Prepared episode/call population identity differs")
    path=Path(directory).resolve()/item["cache_file"]["file"]
    if Path(item["cache_file"]["file"]).name!=item["cache_file"]["file"] or file_hash(path)!=item["cache_file"]["sha256"]:
        raise ValueError("Prepared cache file/hash mismatch")
    cache=torch.load(path,weights_only=True,map_location="cpu")
    if any(not isinstance(value,torch.Tensor) or not torch.isfinite(value).all() for value in cache.values()):
        raise ValueError("Nonfinite or invalid prepared tensor")
    if set(cache)!=set(CACHE_KEYS) or cache_identity(cache)!=item["cached_tensors"]:
        raise ValueError("Prepared tensor identities differ")
    for name in ("initial_noise","ref_action","lang_attn_mask","action_mask","ctrl_freqs"):
        if tensor_hash(cache[name])!=tensor_hash(payload[name]):
            raise ValueError("Prepared cache differs from original sidecar: "+name)
    return cache,item
