#!/usr/bin/env bash
# Every command the NHSJS revision ran on its RunPod pods, so each GPU step can be repeated exactly.
#
#   scripts/revision_pod.sh setup
#   scripts/revision_pod.sh deploy  ARCHIVE.tar DEST MANIFEST   extract a git archive, verify every file
#   scripts/revision_pod.sh collect CODE PROTOCOL STRATUM OUT [1b]
#   scripts/revision_pod.sh bank    CODE METRICS SELECTION OUT
#   scripts/revision_pod.sh work    CODE PROTOCOL BANK PHASES WORKER WORKERS OUT [online]
#   scripts/revision_pod.sh identity CODE OUT [extra capture_identity arguments]
#
# The volume layout is the Month 8 one: the Hugging Face cache at /workspace/huggingface,
# the venv and RDT upstream checkout under /workspace/tmlr. Work runs offline unless the
# last argument of "work" is "online" (the Month 8 forward code asks the Hub for revisions).
set -euo pipefail

export HF_HOME=/workspace/huggingface
export RDT_SOURCE=/workspace/tmlr/rdt-upstream
export PYTHONDONTWRITEBYTECODE=1
PY=/workspace/tmlr/venv/bin/python
LANG_DIR=/workspace/tmlr/lang-20260930
CKPT_1B=/workspace/huggingface/hub/models--robotics-diffusion-transformer--maniskill-model/snapshots/9622afab2b7ce2312a6cf1febc526928589b77eb/rdt/mp_rank_00_model_states.pt

command=${1:?command}
shift
case "$command" in
  setup)
    # The container disk is fresh on every pod: Vulkan loader and NVIDIA ICD files for SAPIEN
    # rendering (the system part of scripts/setup_runtime.sh; the venv lives on the volume).
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan1 vulkan-tools libglvnd-dev libegl1 libgl1 libglib2.0-0
    mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
    test -f /usr/share/vulkan/icd.d/nvidia_icd.json || \
      echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.155"}}' > /usr/share/vulkan/icd.d/nvidia_icd.json
    test -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json || \
      echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
    vulkaninfo --summary 2>/dev/null | grep -m1 deviceName
    git config --global --add safe.directory "$RDT_SOURCE"
    git -C "$RDT_SOURCE" rev-parse HEAD
    test -z "$(git -C "$RDT_SOURCE" status --porcelain)"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
    df -h /workspace
    "$PY" -c "import torch, numpy, diffusers, transformers; print(torch.__version__, numpy.__version__, diffusers.__version__, transformers.__version__)"
    ;;
  deploy)
    archive=${1:?archive}; dest=${2:?dest}; manifest=${3:?manifest}
    test ! -e "$dest"
    mkdir -p "$dest"
    tar -x -f "$archive" -C "$dest"
    (cd "$dest" && sha256sum -c --quiet "$manifest")
    test "$(cd "$dest" && find . -type f | wc -l)" -eq "$(wc -l < "$manifest")"
    echo "deployed $dest"
    ;;
  collect)
    code=${1:?code}; protocol=${2:?protocol}; stratum=${3:?stratum}; out=${4:?out}; model=${5:-170m}
    extra=()
    if [ "$model" = "1b" ]; then extra=(--checkpoint-path "$CKPT_1B"); fi
    if [ -d "$out.run" ]; then extra+=(--resume); fi
    cd "$code"
    HF_HUB_OFFLINE=1 "$PY" scripts/launch_strict.py --protocol "$protocol" --stratum "$stratum" \
      --out "$out" --lang-dir "$LANG_DIR" "${extra[@]}"
    ;;
  bank)
    code=${1:?code}; metrics=${2:?metrics}; selection=${3:?selection}; out=${4:?out}
    cd "$code"
    "$PY" paired_comparison.py bank --metrics "$metrics" --selection "$selection" --out "$out"
    sha256sum "$out"
    ;;
  work)
    code=${1:?code}; protocol=${2:?protocol}; bank=${3:?bank}; phases=${4:?phases}
    worker=${5:?worker}; workers=${6:?workers}; out=${7:?out}; mode=${8:-offline}
    cd "$code"
    if [ "$mode" = "online" ]; then unset HF_HUB_OFFLINE; else export HF_HUB_OFFLINE=1; fi
    "$PY" revision_numerics.py --protocol "$protocol" --bank "$bank" --phases "$phases" \
      --worker "$worker" --workers "$workers" --out "$out"
    ;;
  identity)
    code=${1:?code}; out=${2:?out}; shift 2
    cd "$code"
    "$PY" scripts/capture_identity.py --out "$out" --hf-home "$HF_HOME" --rdt-source "$RDT_SOURCE" "$@"
    ;;
  *)
    echo "unknown command $command" >&2
    exit 2
    ;;
esac
