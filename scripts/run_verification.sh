#!/usr/bin/env bash
# Historical orchestration is preserved in legacy/2026-09-30.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$repo_root/legacy_status.py" "scripts/run_verification.sh" "$@"
