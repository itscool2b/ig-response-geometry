#!/usr/bin/env bash
# Run inside a dedicated Linux GPU container, with /workspace durable storage.
# No credentials are embedded; model downloads use public upstream repositories.
set -euo pipefail
TASK_ROOT=${1:-/workspace/tmlr}
mkdir -p "$TASK_ROOT"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y libvulkan1 vulkan-tools libglvnd-dev libegl1 libgl1 libglib2.0-0
mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.2.155"}}
JSON
fi
if [ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'JSON'
{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}
JSON
fi
if [ ! -d "$TASK_ROOT/venv" ]; then
python -m venv --system-site-packages "$TASK_ROOT/venv"
fi
"$TASK_ROOT/venv/bin/python" -m pip install -r "$TASK_ROOT/requirements-gpu.txt"
if [ ! -d "$TASK_ROOT/rdt-upstream/.git" ]; then
git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git "$TASK_ROOT/rdt-upstream"
git -C "$TASK_ROOT/rdt-upstream" checkout --detach cd79363a1387e8f81c7724d070ef7e45fd23150f
fi
test "$(git -C "$TASK_ROOT/rdt-upstream" rev-parse HEAD)" = cd79363a1387e8f81c7724d070ef7e45fd23150f
"$TASK_ROOT/venv/bin/python" -m pip freeze > "$TASK_ROOT/environment-freeze.txt"
vulkaninfo --summary > "$TASK_ROOT/vulkan-summary.txt" 2>&1
