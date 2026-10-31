"""Regenerate R1 results from preserved inputs and compare them with the stored outputs (Must Fix 2).

Three commands:

  verify-kit  DIR                     check every file of a regeneration kit against its MANIFEST.sha256
  cpu         --protocol P --bank-file B --bank-id ID --root R [--out F]
              authenticate every completed phase (bf16, fp32 where present, scoring) by its completion
              hashes; recompute every response, curve and area (realized and nominal axes) from the
              stored active actions; check the count schedule, the realized fractions and that every
              mask follows its ranking order (deletion) or complement (insertion); optionally recompute
              the primary contrasts
  gpu         --protocol P --bank-file B --bank-id ID --context KEY --stored-root R --fresh-root F
              rerun the bf16 and scoring phases of one context (and its fp32 phase when the protocol
              puts it in the fp32 subset, which scoring needs) with the frozen worker into a fresh
              root, then compare every stored map and group-score tensor (missing or extra ones
              fail), every record, ranking file and area: bit for bit (same GPU type and wheels),
              else within tolerances

The GPU command needs the protocol's paths to exist (the kit README lays them out). It never
writes into the stored root.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TIMING_KEYS = {"elapsed_seconds", "wall_seconds", "repeat_seconds", "seconds", "started_unix", "ended_unix", "host",
               "attribution_and_response_wall_seconds"}
TOLERANCE = dict(relative_l1=0.05, spearman=0.95, abs_auc=0.02)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_kit(directory):
    directory = Path(directory).resolve()
    manifest = (directory / "MANIFEST.sha256").read_text().splitlines()
    bad, invalid, listed = [], [], set()
    for lineno, line in enumerate(manifest, 1):
        parts = line.split("  ", 1)
        if len(parts) != 2:
            invalid.append(f"line {lineno}: expected SHA-256 and relative POSIX path")
            continue
        digest, name = parts
        relative = PurePosixPath(name)
        if (not re.fullmatch(r"[0-9a-f]{64}", digest) or not name
                or "\\" in name or relative.is_absolute() or PureWindowsPath(name).drive
                or any(part in ("", ".", "..") for part in name.split("/"))
                or name == "MANIFEST.sha256" or name in listed):
            invalid.append(f"line {lineno}: invalid or duplicate entry")
            continue
        listed.add(name)
        path = directory / name
        # A manifest must never authenticate a file outside its own kit, even
        # when a directory component is a symlink or Windows junction.
        if (not path.resolve().is_relative_to(directory) or path.is_symlink()
                or not path.is_file() or sha256(path) != digest):
            bad.append(name)
    unlisted = sorted(p.relative_to(directory).as_posix() for p in directory.rglob("*")
                      if p.is_file() and p != directory / "MANIFEST.sha256"
                      and p.relative_to(directory).as_posix() not in listed)
    if not manifest:
        invalid.append("empty manifest")
    return dict(files=len(manifest), mismatched=bad, unlisted=unlisted, invalid=invalid,
                status="verified" if not bad and not unlisted and not invalid else "failed")


def strip_timing(value):
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def verify_schedule_and_masks(result):
    """What authenticate_action_results leaves open: the count schedule, the realized fractions, the
    nominal-axis area, and that each mask follows the ranking order (deletion) or its complement (insertion)."""
    from itertools import combinations
    from experiment_io import object_hash
    from faithfulness import auc_normalized
    import revision_numerics as rn
    indices = result["eligible_indices"]
    n = len(indices)
    counts = rn.count_schedule(n, result["nominal_grid_percent"], result["count_rule"])
    if result["selected_counts"] != counts:
        raise ValueError("Stored counts do not follow the count rule")
    if result["realized_fractions"] != [count / n for count in counts]:
        raise ValueError("Stored realized fractions differ from counts over groups")
    table = result["response_table"]
    for name, ranking in result["rankings"].items():
        if ranking["status"] != "defined":
            continue
        exact = ranking.get("control_kind") == "exact_uniform_subset_expectation"
        for direction, curve in ranking["curves"].items():
            if exact:
                masks = [[list(group) for group in combinations(indices, count)] for count in counts]
                ids = curve["response_ids"]
            else:
                order = ranking["order"]
                if sorted(order) != indices:
                    raise ValueError(f"{name} does not order the eligible groups")
                masks = [[sorted(order[:count])] for count in counts]
                ids = [[key] for key in curve["response_ids"]]
                if curve["selected_positions"] != [group[0] for group in masks]:
                    raise ValueError(f"{name} masks do not follow its order")
            for groups, keys in zip(masks, ids):
                expected = [mask if direction == "deletion" else sorted(set(indices) - set(mask)) for mask in groups]
                if [table[key]["baseline_positions"] for key in keys] != expected or \
                        keys != [object_hash(positions) for positions in expected]:
                    raise ValueError(f"{name} {direction} responses are not those of its masks")
            endpoints = {response: (table[result["input_response_id"]][response], table[result["baseline_response_id"]][response])
                         for response in ("Q", "L2", "RMS")}
            for response, outcome in curve["responses"].items():
                values = outcome["values"]
                if any(x is None for x in values + list(endpoints[response])):
                    expected_nominal = None
                else:
                    expected_nominal = auc_normalized(list(result["nominal_grid_percent"]), values, *endpoints[response],
                                                      denominator_min=result["denominator_min"].get(response, 0))
                if outcome["nominal_normalized_auc"] != expected_nominal:
                    raise ValueError(f"{name} {direction} {response} nominal area differs")


def cpu_regeneration(args):
    from analysis.revision_gpu.v1 import methods
    from paired_comparison import authenticate_action_results
    only = set(args.contexts.split(",")) if args.contexts else None
    protocol, cfg, bank, contexts = methods.load_bank(args.root, args.protocol, args.bank_file, args.bank_id,
                                                      phases=("bf16", "fp32", "scoring"), only=only)
    checked, phases = 0, 0
    for item in contexts:
        phases += sum(phase in item for phase in ("bf16", "fp32", "scoring"))
        if "scoring" not in item:
            continue
        for modality, result in item["scoring"]["modalities"].items():
            # Each saved Q, L2 and RMS value, curve and area must follow from the stored active actions.
            authenticate_action_results(result)
            verify_schedule_and_masks(result)
            checked += 1
    scored = [c for c in contexts if "scoring" in c]
    names = [name for name in scored[0]["scoring"]["modalities"]["vision"]["rankings"]
             if not name.startswith("random_")]
    rows = methods.context_table(scored, "vision", names)
    summary = dict(kind="cpu_regeneration", bank=args.bank_id, contexts=len(scored), phases_authenticated=phases,
                   modality_results_authenticated=checked, vision_rows=len(rows),
                   rows_fingerprint_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
    if args.plan:
        plan = json.loads(Path(args.plan).read_text())
        summary["primary_contrasts"] = methods.primary_contrasts(scored, plan)
    return summary


def compare_maps(stored, fresh):
    """Every stored tensor against the fresh one: identical, differing (with metrics), missing or extra."""
    import torch
    from ig_quadrature import spearman
    report = dict(identical=[], differing={}, missing=sorted(set(stored) - set(fresh)), extra=sorted(set(fresh) - set(stored)))
    for key in sorted(set(stored) & set(fresh)):
        value, other = stored[key], fresh[key]
        if value.shape == other.shape and value.dtype == other.dtype and torch.equal(value, other):
            report["identical"].append(key)
            continue
        if value.shape != other.shape:
            report["differing"][key] = dict(relative_l1=None, spearman=None, reason="shape differs")
            continue
        a, b = value.double(), other.double()
        scale = float(a.abs().sum())
        if key.endswith("/group_scores"):
            ranks = spearman(a.flatten().numpy(), b.flatten().numpy())
        else:
            axis = 1 if value.ndim == 3 and value.shape[1] > 1 else value.ndim - 1
            ranks = spearman(a.abs().sum(dim=tuple(d for d in range(a.ndim) if d != axis)).numpy(),
                             b.abs().sum(dim=tuple(d for d in range(b.ndim) if d != axis)).numpy())
        report["differing"][key] = dict(relative_l1=float((a - b).abs().sum()) / scale if scale else None, spearman=ranks)
    return report


def area_differences(stored, fresh):
    """Every area of every ranking, both directions and responses: realized, nominal and raw."""
    report = dict(compared=0, both_undefined=0, definedness_mismatch=[], max_abs_normalized=0.0, max_abs_raw=0.0,
                  rankings_missing=[], rankings_extra=[])
    for modality, result in stored["modalities"].items():
        other = fresh["modalities"][modality]["rankings"]
        report["rankings_missing"] += [f"{modality}/{n}" for n in sorted(set(result["rankings"]) - set(other))]
        report["rankings_extra"] += [f"{modality}/{n}" for n in sorted(set(other) - set(result["rankings"]))]
        for name, ranking in result["rankings"].items():
            if name not in other:
                continue
            if (ranking.get("status") == "defined") != (other[name].get("status") == "defined"):
                report["definedness_mismatch"].append(f"{modality}/{name}")
                continue
            if ranking.get("status") != "defined":
                continue
            for direction, curve in ranking["curves"].items():
                for response, value in curve["responses"].items():
                    fresh_value = other[name]["curves"][direction]["responses"][response]
                    for metric in ("normalized_auc", "nominal_normalized_auc", "raw_auc"):
                        a, b = value.get(metric), fresh_value.get(metric)
                        if a is None and b is None:
                            report["both_undefined"] += 1
                        elif a is None or b is None:
                            report["definedness_mismatch"].append(f"{modality}/{name}/{direction}/{response}/{metric}")
                        else:
                            report["compared"] += 1
                            key = "max_abs_raw" if metric == "raw_auc" else "max_abs_normalized"
                            report[key] = max(report[key], abs(a - b))
    return report


def gpu_regeneration(args):
    import torch
    import revision_numerics as rn
    from experiment_io import strict_json
    bank = strict_json(args.bank_file)
    cfg = strict_json(args.protocol)["banks"][args.bank_id]
    available = [entry for entry in bank["contexts"] if entry["status"] == "available"]
    index = next(i for i, entry in enumerate(available) if rn.context_key(entry) == args.context)
    member = available[index]["context_id"] in rn.resolve_members(cfg["fp32_contexts"], bank)
    phases = ("bf16", "fp32", "scoring") if member else ("bf16", "scoring")
    fresh = Path(args.fresh_root)
    if fresh.exists() and any(fresh.iterdir()):
        raise FileExistsError("Fresh root must be empty")
    command = [sys.executable, str(ROOT / "revision_numerics.py"), "--protocol", args.protocol, "--bank", args.bank_id,
               "--phases", ",".join(phases), "--worker", str(index), "--workers", str(len(available)), "--out", str(fresh)]
    done = subprocess.run(command, cwd=ROOT)
    if done.returncode:
        raise RuntimeError(f"Worker failed with exit code {done.returncode}")
    report = dict(kind="gpu_regeneration", context=args.context, command=command, phases={})
    for phase in phases:
        stored_dir = Path(args.stored_root) / args.bank_id / args.context / phase
        fresh_dir = fresh / args.bank_id / args.context / phase
        entry = dict()
        stored_record = strip_timing(json.loads((stored_dir / "record.json").read_text()))
        fresh_record = strip_timing(json.loads((fresh_dir / "record.json").read_text()))
        entry["record_equal_without_timing"] = stored_record == fresh_record
        if phase in ("bf16", "fp32"):
            entry["rankings_bytes_equal"] = (stored_dir / "rankings.json").read_bytes() == (fresh_dir / "rankings.json").read_bytes()
            maps = compare_maps(torch.load(stored_dir / "maps.pt", weights_only=True),
                                torch.load(fresh_dir / "maps.pt", weights_only=True))
            entry["maps"] = dict(identical=len(maps["identical"]), differing=maps["differing"], missing=maps["missing"],
                                 extra=maps["extra"])
        else:
            entry["areas"] = area_differences(json.loads((stored_dir / "record.json").read_text()),
                                              json.loads((fresh_dir / "record.json").read_text()))
        report["phases"][phase] = entry
    mapped = [report["phases"][phase] for phase in phases if phase != "scoring"]
    areas = report["phases"]["scoring"]["areas"]
    complete = all(not p["maps"]["missing"] and not p["maps"]["extra"] for p in mapped) and \
        not areas["rankings_missing"] and not areas["rankings_extra"] and not areas["definedness_mismatch"]
    bit = complete and all(p["record_equal_without_timing"] for p in report["phases"].values()) and \
        all(p["rankings_bytes_equal"] and not p["maps"]["differing"] for p in mapped)
    within = complete and all(v["relative_l1"] is not None and v["relative_l1"] <= TOLERANCE["relative_l1"]
                              and (v["spearman"] or 0) >= TOLERANCE["spearman"]
                              for p in mapped for v in p["maps"]["differing"].values()) and \
        areas["max_abs_normalized"] <= TOLERANCE["abs_auc"]
    report["status"] = "bit_identical" if bit else "within_tolerance" if within else "outside_tolerance"
    report["tolerance"] = TOLERANCE
    report["device"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    kit = commands.add_parser("verify-kit")
    kit.add_argument("directory")
    cpu = commands.add_parser("cpu")
    gpu = commands.add_parser("gpu")
    for sub in (cpu, gpu):
        sub.add_argument("--protocol", required=True)
        sub.add_argument("--bank-file", required=True)
        sub.add_argument("--bank-id", required=True)
        sub.add_argument("--out")
    cpu.add_argument("--root", required=True)
    cpu.add_argument("--plan", help="analysis plan JSON; adds the primary contrasts")
    cpu.add_argument("--contexts", help="comma-separated context keys to check (default: every available context)")
    gpu.add_argument("--context", required=True)
    gpu.add_argument("--stored-root", required=True)
    gpu.add_argument("--fresh-root", required=True)
    args = parser.parse_args()
    if args.command == "verify-kit":
        result = verify_kit(args.directory)
    elif args.command == "cpu":
        result = cpu_regeneration(args)
    else:
        result = gpu_regeneration(args)
    text = json.dumps(result, indent=1, sort_keys=True, default=str)
    if getattr(args, "out", None):
        Path(args.out).write_text(text + "\n")
    print(text[:4000])
    if args.command == "cpu":
        return 0
    return 0 if result["status"] in ("verified", "bit_identical", "within_tolerance") else 1


if __name__ == "__main__":
    sys.exit(main())
