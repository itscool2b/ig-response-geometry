#!/usr/bin/env bash
# One explicit current job. See docs/legacy_workflows.md; archived campaign source is retained.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$repo_root/scripts/run_workflow.py" overlays "$@"
