"""Compare the R1 fp32 arm with the sealed Month 8 v6 probe on the same contexts.

Month 8 computed fp32 trapezoid maps at several budgets on the fp32 probe of
each Month 8 context (numerics-probe-v6-numpyfix). R1 recomputes the same
function through a different driver (the nested grid of revision_numerics). If
the source tensors and the arithmetic agree, the A2 trapezoid maps must equal
the v6 maps bit for bit. This is a read-only check; run it with CUDA hidden.

The output lists every v6 context with its status (compared, missing, r1_fp32_missing),
every map compared, the v6 budgets R1 does not store (skipped_not_stored) and the
source-cache keys compared; the printed summary counts each, so a context that compared
nothing is visible rather than silently absent.

Usage: python -m analysis.revision_gpu.v1.v6_check --v6 DIR --r1 DIR --out FILE
  --v6  .../numerics-probe-v6-numpyfix   --r1  .../runs/r1
"""
import argparse
import glob
import json
from pathlib import Path

import torch

from per_step_attribution import MANISKILL_INDICES

STRATA = ("pickcube-170m", "stackcube-170m", "peginsertionside-170m", "picksingleycb-170m", "pickcube-1b", "stackcube-1b")
TARGET_NAMES = {"Q": "Q", "L2": "N"}


def eligible_slice(full, modality, report):
    if modality == "vision":
        return full[:, 3 * 729:4 * 729]
    if modality == "language":
        return full[:, report["eligible_indices"]]
    return full[:, :, MANISKILL_INDICES]


def compare_context(v6_dir, r1_dir):
    out = dict(cache={}, maps={}, skipped_not_stored=[])
    prepared = sorted(glob.glob(str(v6_dir.parents[2] / "prepared" / v6_dir.parents[0].name / "*.pt")))
    r1_cache_path = r1_dir / "bf16" / "source-cache.pt"
    if prepared and r1_cache_path.exists():
        v6_cache = torch.load(prepared[0], weights_only=True, map_location="cpu")
        r1_cache = torch.load(r1_cache_path, weights_only=True, map_location="cpu")
        for key in sorted(set(v6_cache) & set(r1_cache)):
            out["cache"][key] = bool(v6_cache[key].shape == r1_cache[key].shape and torch.equal(v6_cache[key], r1_cache[key]))
    r1_maps_path = r1_dir / "fp32" / "maps.pt"
    if not r1_maps_path.exists():
        out["status"] = "r1_fp32_missing"
        return out
    r1_maps = torch.load(r1_maps_path, weights_only=True, map_location="cpu")
    for cell in sorted(glob.glob(str(v6_dir / "*-*"))):
        cell = Path(cell)
        if not cell.is_dir():
            continue
        modality, v6_target = cell.name.split("-")
        target = TARGET_NAMES[v6_target]
        report = json.loads((cell / "report.json").read_text())
        for map_file in sorted(cell.glob("m*-r0.pt")):
            budget = int(map_file.name[1:].split("-")[0])
            key = f"{modality}/{target}/A2T{budget}"
            if key not in r1_maps:
                out["skipped_not_stored"].append(key)
                continue
            v6_full = torch.load(map_file, weights_only=True, map_location="cpu")["attribution"]
            v6_map = eligible_slice(v6_full, modality, report)
            ours = r1_maps[key]
            entry = dict(shape_equal=tuple(v6_map.shape) == tuple(ours.shape))
            if entry["shape_equal"]:
                entry["bit_identical"] = bool(torch.equal(v6_map, ours))
                if not entry["bit_identical"]:
                    difference = (v6_map.double() - ours.double()).abs()
                    entry["relative_l1"] = float(difference.sum() / v6_map.double().abs().sum())
                    entry["max_abs"] = float(difference.max())
            out["maps"][key] = entry
    out["status"] = "compared"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v6", required=True)
    parser.add_argument("--r1", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results = {}
    for stratum in STRATA:
        for v6_context in sorted(glob.glob(f"{args.v6}/results/{stratum}-e*-c*")):
            name = Path(v6_context).name
            episode, call = name.split("-")[-2:]
            r1_dir = Path(args.r1) / f"M8-{stratum}" / f"e{int(episode[1:]):04d}-c{int(call[1:]):03d}"
            inner = sorted(Path(v6_context).glob("ep*-call*"))
            if not inner or not r1_dir.exists():
                results[name] = dict(status="missing")
                continue
            results[name] = compare_context(inner[0], r1_dir)
    summary = summarize(results)
    Path(args.out).write_text(json.dumps(dict(contexts=results, summary=summary), indent=1, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


def summarize(results):
    maps = [(key, m) for r in results.values() for key, m in r.get("maps", {}).items()]
    statuses = {}
    for r in results.values():
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return dict(contexts=len(results), by_status=statuses,
                compared_contexts_with_no_map=sorted(name for name, r in results.items()
                                                     if r["status"] == "compared" and not r["maps"]),
                maps_compared=len(maps), bit_identical=sum(bool(m.get("bit_identical")) for _, m in maps),
                shape_mismatches=sum(not m["shape_equal"] for _, m in maps),
                budgets_compared=sorted({int(key.rsplit("A2T", 1)[1]) for key, _ in maps}),
                maps_skipped_not_stored=sum(len(r.get("skipped_not_stored", [])) for r in results.values()),
                budgets_skipped=sorted({int(key.rsplit("A2T", 1)[1]) for r in results.values()
                                        for key in r.get("skipped_not_stored", [])}),
                cache_keys_compared=sum(len(r.get("cache", {})) for r in results.values()),
                cache_keys_equal=sum(sum(r.get("cache", {}).values()) for r in results.values()),
                max_relative_l1=max((m["relative_l1"] for _, m in maps if "relative_l1" in m), default=None))


if __name__ == "__main__":
    main()
