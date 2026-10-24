"""Rebuild the stored quadrature maps of contexts that kept node gradients, on CPU.

A phase that kept node gradients (nodes-<modality>-<target>.pt: node_gradients,
nodes, displacement) lets this check recompute every stored rule map from them
and compare it with the stored map of the same name:

- trapezoid T_m, endpoint-inclusive E_m, midpoint M_n and the path gradient
  (the T_m average before the displacement): replayed in fp32 in node order
  with the worker's weights, expected bit-identical. CUDA divides a tensor by a
  host scalar as a multiply by the scalar's reciprocal while CPU divides
  exactly; the two can differ only where the divisor is not a power of two
  (E_m, Simpson, Romberg), so those are replayed both ways and the matching
  way is recorded;
- Simpson S_2n and the Romberg tables RT and RM: the worker's float64 formulas
  applied to the replayed T and M maps;
- every map also in float64 with exact sums, the size of the fp32 rounding.

It checks E_m = T_m + [d (f(0) + f(1))/2 - T_m]/(m + 1) and T_2n = (T_n + M_n)/2
on the stored maps themselves (in float64), and reports every phase it
inspects, including cells with missing nodes or no defined rule. Read-only.

Usage: python -m analysis.revision_gpu.v1.node_check --run-root DIR --out FILE
"""
import argparse
import json
from pathlib import Path

import torch

PHASE_ARMS = {"bf16": "A1", "fp32": "A2"}


def node_rules(gradients, displacement, dtype, N=512, reciprocal=False):
    """T_m, E_m, M_n and the T_m path-gradient averages from the N + 1 node gradients, in node order."""
    g = gradients.to(dtype)
    d = displacement.to(dtype)
    one = torch.tensor(1.0, dtype=dtype)
    rules, averages = {}, {}
    m = 1
    while m <= N:
        step = N // m
        trap = torch.zeros_like(d)
        end = torch.zeros_like(d)
        for k in range(m + 1):
            trap.add_(g[k * step], alpha=0.5 if k in (0, m) else 1.0)
            end.add_(g[k * step])
        averages[f"T{m}"] = trap / m
        rules[f"T{m}"] = d * averages[f"T{m}"]
        rules[f"E{m}"] = d * (end * (one / (m + 1)) if reciprocal else end / (m + 1))
        if m < N:
            mid = torch.zeros_like(d)
            for k in range(m):
                mid.add_(g[k * step + step // 2])
            rules[f"M{m}"] = d * (mid / m)
        m *= 2
    return rules, averages


def derived_rules(rules, levels, reciprocal=False):
    """ig_quadrature.simpson_and_romberg, with the scalar divisions done exactly or as reciprocal multiplies."""
    def divide(x, d):
        return x * (torch.tensor(1.0, dtype=torch.float64) / d) if reciprocal else x / d
    derived = {}
    for l in range(levels):
        n = 2 ** l
        derived[f"S{2 * n}"] = divide(rules[f"T{n}"].double() + 2 * rules[f"M{n}"].double(), 3).to(rules[f"T{n}"].dtype)
    for prefix, top in (("T", levels), ("M", levels - 1)):
        previous = None
        for k in range(top + 1):
            row = [rules[f"{prefix}{2 ** k}"].double()]
            for j in range(1, k + 1):
                row.append(row[j - 1] + divide(row[j - 1] - previous[j - 1], 4 ** j - 1))
            previous = row
            derived[f"R{prefix}{2 ** k}"] = row[-1].to(rules[f"{prefix}{2 ** k}"].dtype)
    return derived


def compare(stored, ours):
    stored64, ours64 = stored.double(), ours.double()
    scale = stored64.abs().sum()
    difference = (stored64 - ours64).abs()
    return dict(bit_identical=bool(torch.equal(stored, ours.to(stored.dtype))),
                relative_l1=float(difference.sum() / scale) if scale > 0 else float(difference.sum()),
                max_abs=float(difference.max()))


def replay(stored, candidates):
    """Bit identity against the exact-division replay, else the reciprocal one; records which matched."""
    exact = compare(stored, candidates["exact"])
    if exact["bit_identical"] or "reciprocal" not in candidates:
        return dict(exact, division="exact")
    reciprocal = compare(stored, candidates["reciprocal"])
    return dict(reciprocal, division="reciprocal") if reciprocal["bit_identical"] else dict(exact, division="neither")


def stored_identities(maps, prefix, arm, f0, f1, d, N):
    """Largest relative violation of the E_m and T_2n identities on the stored maps (float64)."""
    out = {}
    get = lambda name: maps.get(f"{prefix}{arm}{name}")  # noqa: E731
    m = 1
    while m <= N:
        t, e = get(f"T{m}"), get(f"E{m}")
        if t is not None and e is not None:
            rhs = t.double() + (d * (f0 + f1) / 2 - t.double()) / (m + 1)
            out[f"E{m}"] = float((rhs - e.double()).abs().sum() / e.double().abs().sum().clamp_min(1e-300))
        if m < N and t is not None and get(f"M{m}") is not None and get(f"T{2 * m}") is not None:
            t2 = get(f"T{2 * m}").double()
            out[f"T{2 * m}"] = float(((t.double() + get(f"M{m}").double()) / 2 - t2).abs().sum()
                                     / t2.abs().sum().clamp_min(1e-300))
        m *= 2
    return out


def check_phase(directory, arm):
    maps = torch.load(directory / "maps.pt", weights_only=True, map_location="cpu")
    out = {}
    for node_file in sorted(directory.glob("nodes-*-*.pt")):
        modality, target = node_file.stem.split("-")[1:]
        saved = torch.load(node_file, weights_only=True, map_location="cpu")
        nodes = saved["nodes"].tolist()
        if nodes != list(range(len(nodes))):
            out[f"{modality}/{target}"] = dict(status="incomplete_nodes", nodes=len(nodes))
            continue
        N = len(nodes) - 1
        levels = N.bit_length() - 1
        prefix = f"{modality}/{target}/"
        fp32 = {mode: node_rules(saved["node_gradients"], saved["displacement"], torch.float32, N, mode == "reciprocal")
                for mode in ("exact", "reciprocal")}
        fp64, _ = node_rules(saved["node_gradients"], saved["displacement"], torch.float64, N)
        candidates = {name: {mode: fp32[mode][0][name] for mode in fp32} for name in fp64}
        for mode in ("exact", "reciprocal"):
            for name, value in derived_rules(fp32["exact"][0], levels, reciprocal=mode == "reciprocal").items():
                candidates.setdefault(name, {})[mode] = value
        fp64.update(derived_rules(fp64, levels))
        entry = dict(status="checked", rules={})
        for name, options in candidates.items():
            key = f"{prefix}{arm}{name}"
            if key in maps:
                entry["rules"][name] = dict(fp32_replay=replay(maps[key], options), float64=compare(maps[key], fp64[name]))
        for name, average in fp32["exact"][1].items():
            key = f"{prefix}{arm}PG{name}"
            if key in maps:
                entry["rules"][f"PG{name}"] = dict(fp32_replay=replay(maps[key], {"exact": average}),
                                                    float64=compare(maps[key], average.double()))
        own = [k[len(prefix) + len(arm):] for k in maps
               if k.startswith(prefix + arm) and not k.endswith("/group_scores")]
        entry["unchecked_stored_maps"] = sorted(name for name in own if name not in entry["rules"])
        d = saved["displacement"].double()
        f0, f1 = saved["node_gradients"][0].double(), saved["node_gradients"][N].double()
        entry["stored_identity_relative"] = stored_identities(maps, prefix, arm, f0, f1, d, N)
        if not entry["rules"]:
            entry["status"] = "no_stored_rule"
        out[f"{modality}/{target}"] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    results, without_nodes = {}, 0
    for phase, arm in PHASE_ARMS.items():
        for directory in sorted(Path(args.run_root).glob(f"*/e*-c*/{phase}")):
            if not (directory / "completion.json").is_file():
                continue
            if not any(directory.glob("nodes-*-*.pt")):
                without_nodes += 1
                continue
            results[f"{directory.parts[-3]}/{directory.parts[-2]}/{phase}"] = check_phase(directory, arm)
    Path(args.out).write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    cells = [cell for phase in results.values() for cell in phase.values()]
    checked = [r for cell in cells for r in cell.get("rules", {}).values()]
    identities = [v for cell in cells for v in cell.get("stored_identity_relative", {}).values()]
    print(json.dumps(dict(phases_with_nodes=len(results), phases_without_nodes=without_nodes, cells=len(cells),
                          cells_incomplete_nodes=sum(c["status"] == "incomplete_nodes" for c in cells),
                          cells_without_stored_rule=sum(c["status"] == "no_stored_rule" for c in cells),
                          unchecked_stored_maps=sum(len(c.get("unchecked_stored_maps", [])) for c in cells),
                          rules_compared=len(checked),
                          fp32_bit_identical=sum(r["fp32_replay"]["bit_identical"] for r in checked),
                          max_float64_relative_l1=max((r["float64"]["relative_l1"] for r in checked), default=None),
                          max_stored_identity_relative=max(identities, default=None))))


if __name__ == "__main__":
    main()
