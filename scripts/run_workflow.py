"""Run one explicit current workflow under a recorded protocol.

This replaces historical filename globs and implicit experiment sizes. It
does not recover old runs or select a scientific protocol for the caller.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_io import canonical_json, file_hash, strict_json

ENTRYPOINTS = {"collect": "per_step_ig.py", "faithfulness": "faithfulness.py",
               "sanity": "sanity.py", "displacement": "displacement.py", "overlays": "generate_overlays.py"}


def option(arguments, name):
    matches = [i for i, value in enumerate(arguments) if value == name or value.startswith(name + "=")]
    if len(matches) != 1:
        raise ValueError("Specify exactly one " + name)
    index = matches[0]
    if "=" in arguments[index]:
        value = arguments[index].split("=", 1)[1]
    else:
        if index + 1 == len(arguments) or arguments[index + 1].startswith("--"):
            raise ValueError("Missing value for " + name)
        value = arguments[index + 1]
    if not value:
        raise ValueError("Empty value for " + name)
    return value


def plan(workflow, arguments, decision_file, python=sys.executable):
    decision_file = Path(decision_file).resolve()
    decision = strict_json(decision_file)
    if not isinstance(decision.get("decision_id"), str) or not decision["decision_id"].strip():
        raise ValueError("Protocol JSON must name its decision_id")
    required = ["--task", "--out"]
    if workflow == "collect":
        required += ["--model", "--episodes", "--max-policy-calls", "--m", "--seed-base", "--target", "--quadrature", "--lang-dir"]
        if option(arguments, "--model") == "1b":
            mode = option(arguments, "--checkpoint-mode")
            if mode in ("authors", "lora"):
                option(arguments, "--checkpoint-path")
    else:
        required += ["--metrics"]
    if workflow == "sanity":
        required += ["--phase", "--randomization-seed"]
        if option(arguments, "--phase") == "C2":
            required += ["--c2-mode", "--eos-policy", "--shuffle-seed"]
    if workflow == "displacement":
        required += ["--solver-steps", "--del-grid", "--modality", "--selection", "--signal-filter"]
    for name in required:
        option(arguments, name)
    destination = Path(option(arguments, "--out")).resolve()
    protected = [ROOT / name for name in ("data", "out", "output", "paper", "legacy", "analysis/revision/results")]
    if any(destination.is_relative_to(path.resolve()) for path in protected):
        raise ValueError("Use a fresh run destination outside preserved artifact directories")
    if destination.exists() and "--resume" not in arguments:
        raise FileExistsError("Destination exists; use a fresh output or explicit exact-compatible resume")
    if "--resume" in arguments and workflow != "collect":
        raise ValueError("Only collection has an authenticated resume contract")
    # The child runs from ROOT. Resolve caller-relative filesystem arguments
    # now so a different launching cwd cannot bypass destination checks.
    normalized = list(arguments)
    for name in ("--out", "--metrics", "--lang-dir", "--checkpoint-path", "--lang-embeds", "--shuffled-language"):
        indices = [i for i, value in enumerate(normalized) if value == name or value.startswith(name + "=")]
        if not indices:
            continue
        value = str(Path(option(normalized, name)).resolve())
        index = indices[0]
        if "=" in normalized[index]:
            normalized[index] = name + "=" + value
        else:
            normalized[index + 1] = value
    return {"workflow": workflow, "decision_id": decision["decision_id"],
            "decision_file": str(decision_file), "decision_sha256": file_hash(decision_file),
            "entrypoint_sha256": file_hash(ROOT / ENTRYPOINTS[workflow]),
            "command": [python, str(ROOT / ENTRYPOINTS[workflow]), *normalized],
            "cwd": str(ROOT), "destination": str(destination),
            "scope": "One caller-selected current job; historical files are not reproduction targets."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=sorted(ENTRYPOINTS))
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    raw = sys.argv[1:]
    if "--" not in raw:
        if "--help" in raw or "-h" in raw:
            parser.parse_args(raw)
        parser.error("Separate explicit workflow arguments with --")
    separator = raw.index("--")
    args = parser.parse_args(raw[:separator])
    arguments = raw[separator + 1:]
    record = plan(args.workflow, arguments, args.decision_file)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    destination = Path(record["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Each invocation is separately identified, including a requested resume.
    from uuid import uuid4
    invocation = destination.with_name(destination.name + ".invocation-" + uuid4().hex + ".json")
    with invocation.open("xb") as stream:
        stream.write(canonical_json(record) + b"\n")
    raise SystemExit(subprocess.run(record["command"], cwd=ROOT).returncode)


if __name__ == "__main__":
    main()
