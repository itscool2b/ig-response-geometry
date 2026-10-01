"""Explicit execution boundaries for preserved historical workflows."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "legacy/2026-09-30"
DISPOSITIONS = {
    "eval_maniskill.py": ("The old evaluator constructed an unloaded backbone and did not apply PEFT adapters. Its success message was not a loading check.", "Use the identified pipeline and collector described in docs/per_step_ig.md; this is a new observation protocol."),
    "finetune_rdt.py": ("Training consumed replay data with unverified controller/state/success identity and saved incomplete adapter-only checkpoints. The retained study uses published checkpoints, not project retraining.", "See docs/legacy_workflows.md for loss/update accounting and requirements before any future training support."),
    "generate_demos.py": ("The historical replay did not bind controller metadata or compare replay success/states with the source trajectory. The original HDF5 identity is unavailable.", "Use the original source only as a historical record; authenticated replay needs a separately validated protocol."),
    "ig_llava.py": ("The quantized demo lacks a pinned, validated image-feature API and active evaluation-mode checkpointing contract.", "See docs/ig_llava.md. No current LLaVA runtime or memory guarantee is advertised."),
    "verify_models.py": ("Loading several demo models did not verify numerical correctness or the RDT research pipeline.", "Use scripts/runtime_probe.py and the explicit numerical validation commands in the current README."),
    "analyze_month4.py": ("The old analyzer silently skipped malformed rows and mixed historical selection/aggregation conventions.", "Use python -m analysis.revision.analyze with a fresh output directory."),
    "bootstrap_ci.py": ("The old aggregate workflow is preserved, but current results require the reconciled source-line populations and strict integrity checks.", "Use python -m analysis.revision.analyze and python -m analysis.revision.verify."),
    "make_month4_figs.py": ("The historical generator writes fixed figure paths and contains historical populations or constants.", "Use scripts/build_revision_assets.py for current revision assets."),
    "make_month5_figs.py": ("The historical generator writes fixed figure paths and does not authenticate missing original contexts.", "Use scripts/build_revision_assets.py for current revision assets."),
    "make_paper_figs.py": ("The historical generator can overwrite preserved figures and includes constants whose originals are unavailable.", "Use scripts/build_revision_assets.py for current revision assets."),
    "patch_nb_alttarget.py": ("This one-off notebook mutation is a historical edit, not a current reproducible analysis stage.", "Keep the historical notebook and use the current analysis package."),
    "scripts/run_all.sh": ("The old orchestration implicitly selected study sizes and appended to released data while mixing historical stages.", "Run one declared current workflow with an explicit protocol and fresh output."),
    "scripts/run_paper_experiments.sh": ("The old multi-campaign script relied on filename existence, obsolete populations and incompatible resume/sidecar identity.", "Use the current single-run wrappers and an explicit protocol; fresh runs do not recover old data."),
    "scripts/run_target_ablation.sh": ("The old target arms are not authenticated as paired and the script appends to historical paths.", "Use the identified paired-comparison workflow after recording its protocol."),
    "scripts/run_verification.sh": ("The old verification script reuses historical destinations and does not establish original context identity.", "Use current collection/evaluation commands with a new run and a recorded protocol."),
}


def status(entrypoint):
    reason, replacement = DISPOSITIONS[entrypoint]
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    item = next(row for row in manifest["files"] if row["source"] == entrypoint)
    archived = ROOT / item["archive"]
    if hashlib.sha256(archived.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("Historical archive hash mismatch: " + entrypoint)
    return {"entrypoint": entrypoint, "status": "historical_execution_retired",
            "reason": reason, "current_route": replacement, "archive": item,
            "qualification": "No old model, demonstration or scientific result is validated by this status check."}


def main(entrypoint=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if entrypoint is None:
        parser.add_argument("entrypoint", choices=sorted(DISPOSITIONS))
    parser.add_argument("--status", action="store_true", help="Read disposition and verify preserved source hash; execute no workflow")
    args = parser.parse_args()
    result = status(entrypoint or args.entrypoint)
    print(json.dumps(result, indent=2))
    if not args.status:
        parser.exit(2, "Historical execution is retired. See docs/legacy_workflows.md.\n")


if __name__ == "__main__":
    main()
