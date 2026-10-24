"""Run the unchanged collect_contexts.py under the strict deterministic settings.

CUBLAS is configured before torch is imported and the strict bundle
(fp32_probe_cache.strict_candidate_settings) is applied; only then does the
collector run, in this process, so its run configuration records these
settings and every later replay can use the same bundle.

Usage: python scripts/launch_strict.py <collect_contexts.py arguments>
"""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fp32_probe_cache import process_startup  # noqa: E402  (imports no torch)


def main(argv):
    startup = process_startup("run")
    from fp32_probe_cache import apply_settings, runtime_settings, strict_candidate_settings
    applied = apply_settings(strict_candidate_settings())
    print(json.dumps(dict(event="strict_launch", startup=startup, settings=applied), sort_keys=True), flush=True)
    collector = ROOT / "collect_contexts.py"
    sys.argv = [str(collector), *argv]
    runpy.run_path(str(collector), run_name="__main__")
    if runtime_settings() != applied:
        raise RuntimeError("The collector changed the strict settings")


if __name__ == "__main__":
    main(sys.argv[1:])
