"""Record the software, hardware, model and simulator identity of one pod.

Writes one JSON file per call (never overwritten). Sections that need a tool
the machine lacks are recorded as unavailable rather than skipped. Environment
variables whose names suggest credentials are reduced to "set"/"unset".

Usage:
  python scripts/capture_identity.py --out FILE [--hf-home DIR] [--rdt-source DIR]
         [--hash-models] [--load-model 170m|1b --checkpoint-path P] [--tasks PickCube-v1 ...]
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PINNED = {
    "google/siglip-so400m-patch14-384": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    "robotics-diffusion-transformer/rdt-170m": "8aa386cac3bbfd9540676c75b3d767cc7f88a10a",
    "robotics-diffusion-transformer/maniskill-model": "9622afab2b7ce2312a6cf1febc526928589b77eb",
    "google/t5-v1_1-xxl": "3db67ab1af984cf10548a73467f0e5bca2aaaeb2",
}
HASHED_REPOS = ("google/siglip-so400m-patch14-384", "robotics-diffusion-transformer/rdt-170m",
                "robotics-diffusion-transformer/maniskill-model")
PACKAGES = ("torch", "torchvision", "numpy", "scipy", "diffusers", "transformers", "huggingface-hub", "safetensors",
            "tokenizers", "sentencepiece", "timm", "peft", "mani_skill", "sapien", "gymnasium", "pillow", "filelock",
            "pyyaml", "accelerate", "einops")
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "CREDENTIAL", "AUTH")
ENV_PREFIXES = ("CUDA", "CUBLAS", "CUDNN", "NVIDIA", "TORCH", "PYTORCH", "HF_", "HUGGINGFACE", "RDT_", "OMP_",
                "MKL_", "RUNPOD_", "PYTHON", "LD_LIBRARY", "VK_", "DISPLAY")


def run(command, timeout=120):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return dict(returncode=done.returncode, stdout=done.stdout, stderr=done.stderr[-4000:])
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(status="unavailable", reason=repr(error))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    values = {}
    for name in sorted(os.environ):
        if not name.startswith(ENV_PREFIXES):
            continue
        secret = any(marker in name.upper() for marker in SECRET_MARKERS)
        values[name] = "set" if secret else os.environ[name]
    return values


def packages():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def torch_identity():
    import torch
    info = dict(version=torch.__version__, cuda=torch.version.cuda, git=getattr(torch.version, "git_version", None),
                cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                config=torch.__config__.show(), parallel=torch.__config__.parallel_info(),
                cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        info.update(device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                    device_count=torch.cuda.device_count(),
                    total_memory=torch.cuda.get_device_properties(0).total_memory,
                    multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count)
    info["flags"] = dict(
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cuda_matmul_allow_bf16_reduced=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        cuda_matmul_allow_fp16_reduced=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        flash_sdp=torch.backends.cuda.flash_sdp_enabled(), mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
        math_sdp=torch.backends.cuda.math_sdp_enabled(),
        cudnn_sdp=getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: None)())
    return info


def hf_snapshots(hf_home, hash_files):
    hub = Path(hf_home) / "hub"
    found = {}
    for repo, revision in PINNED.items():
        folder = hub / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
        if not folder.is_dir():
            found[repo] = dict(revision=revision, status="absent")
            continue
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        entry = dict(revision=revision, status="present",
                     files={str(p.relative_to(folder)): p.stat().st_size for p in files})
        if hash_files and repo in HASHED_REPOS:
            entry["sha256"] = {str(p.relative_to(folder)): sha256(p) for p in files}
        found[repo] = entry
    return found


def upstream(source):
    source = Path(source)
    git = ["git", "-c", f"safe.directory={source}", "-C", str(source)]
    return dict(path=str(source), head=run(git + ["rev-parse", "HEAD"]), status=run(git + ["status", "--porcelain"]),
                tree=run(git + ["rev-parse", "HEAD^{tree}"]), remote=run(git + ["remote", "-v"]))


def model_identity(model, checkpoint_path):
    import torch
    from experiment_io import object_hash, tensor_hash
    from pipeline import load_pipeline
    mode = "authors" if model == "1b" else "pretrained"
    pipe = load_pipeline(model, enable_checkpoint=False, checkpoint_mode=mode, checkpoint_path=checkpoint_path,
                         model_revision=None if model == "1b" else PINNED["robotics-diffusion-transformer/rdt-170m"],
                         vision_revision=PINNED["google/siglip-so400m-patch14-384"])
    runner = pipe["runner"]
    adaptors = {}
    for name in ("lang_adaptor", "img_adaptor", "state_adaptor"):
        module = getattr(runner, name)
        adaptors[name] = object_hash({key: tensor_hash(value) for key, value in module.state_dict().items()})
    processor = pipe["vision_model"].image_processor
    return dict(identity=pipe["identity"], adaptor_parameter_sha256=adaptors,
                vision_preprocessor=json.loads(processor.to_json_string()),
                baseline_image_tokens_sha256=tensor_hash(pipe["img_tokens_baseline"]),
                background_tokens_sha256=tensor_hash(pipe["bg_image_encoded"]),
                runner_parameter_count=sum(p.numel() for p in runner.parameters()),
                runner_dtypes=sorted({str(p.dtype) for p in runner.parameters()}),
                device=str(next(runner.parameters()).device), cuda_memory=torch.cuda.max_memory_allocated())


def simulator(tasks):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    found = {}
    for task in tasks:
        env = gym.make(task, obs_mode="state_dict", num_envs=1, control_mode="pd_joint_pos",
                       render_mode="rgb_array", max_episode_steps=400)
        try:
            base = env.unwrapped
            env.reset(seed=0)
            frame = env.render()
            cameras = {}
            for name, camera in getattr(base, "_human_render_camera_configs", {}).items():
                cameras[name] = {key: repr(getattr(camera, key, None)) for key in
                                 ("uid", "pose", "width", "height", "fov", "near", "far", "intrinsic", "shader_pack")}
            found[task] = dict(sim_freq=getattr(base, "sim_freq", None), control_freq=getattr(base, "control_freq", None),
                               sim_backend=repr(getattr(base, "backend", getattr(base, "sim_backend", None))),
                               sim_config=repr(getattr(base, "sim_config", None)),
                               control_mode=getattr(base, "control_mode", None),
                               robot_uids=repr(getattr(base, "robot_uids", None)),
                               render_frame=dict(shape=list(frame.shape), dtype=str(frame.dtype)),
                               render_cameras=cameras, max_episode_steps=env.spec.max_episode_steps)
        finally:
            env.close()
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    parser.add_argument("--rdt-source", default=os.environ.get("RDT_SOURCE"))
    parser.add_argument("--hash-models", action="store_true")
    parser.add_argument("--load-model", choices=["170m", "1b"])
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--code-root", default=str(ROOT), help="tree whose modules are imported and hashed")
    args = parser.parse_args()
    code_root = Path(args.code_root).resolve()
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    record = dict(kind="pod_identity_capture", python=dict(version=sys.version, executable=sys.executable,
                  prefix=sys.prefix, base_prefix=sys.base_prefix), platform=platform.platform(),
                  node=platform.node(), environment=environment(), packages=packages(),
                  pip_freeze=run([sys.executable, "-m", "pip", "freeze", "--all"]),
                  nvidia_smi_query=run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,pci.bus_id,"
                                        "vbios_version,compute_cap", "--format=csv"]),
                  nvidia_smi_full=run(["nvidia-smi", "-q"]), vulkan=run(["vulkaninfo", "--summary"]),
                  os_release=Path("/etc/os-release").read_text() if Path("/etc/os-release").exists() else None,
                  torch=torch_identity(), hf=hf_snapshots(args.hf_home, args.hash_models),
                  code_root=str(code_root), source_tree={name: sha256(code_root / name) for name in sorted(
                      str(p.relative_to(code_root)) for p in code_root.glob("*.py"))})
    if args.rdt_source:
        record["rdt_upstream"] = upstream(args.rdt_source)
    if args.load_model:
        record["model"] = model_identity(args.load_model, args.checkpoint_path)
    if args.tasks:
        record["simulator"] = simulator(args.tasks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        handle.write(json.dumps(record, indent=1, sort_keys=True, default=repr) + "\n")
    print(json.dumps(dict(out=str(out), sha256=sha256(out))))


if __name__ == "__main__":
    main()
