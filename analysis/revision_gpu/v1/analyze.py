"""Build the R1 result tables from authenticated GPU phase records.

Usage: python -m analysis.revision_gpu.v1.analyze --inputs INPUTS.json --plan PLAN.json --out DIR
                                                  [--coverage FILE ...] [--metrics-copy BANK=PATH ...] [--partial]

INPUTS.json: {"run_root": DIR, "banks": [{"id": "P", "protocol": FILE, "bank": FILE, "group": "P"}, ...]}
Every phase record is loaded through its completion hashes. The analysis follows
docs/nhsjs-revision/protocols/r1/analysis-plan-r1.json with amendments 1, 2 and 3.
Numerics summaries are kept per group (the Month 8 strata pool into one group, as
the plan's population) and per bank. Outputs are deterministic CSV and JSON files;
floats are written with repr precision.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import revision_numerics as rn
from analysis.revision_gpu.v1 import methods, numerics
from experiment_io import file_hash, strict_json

VISION_METHODS = ["IG_Q_A1T512", "IG_N_A1T512", "PG_Q_A1T512", "PG_N_A1T512", "input_difference", "IG_Q_A0m64", "IG_N_A0m64"]
FIXED = dict(vision=["fixed_raster", "fixed_reverse_raster", "fixed_centre_out"],
             language=["fixed_token_position", "fixed_reverse_token_position"],
             state=["fixed_joint_order", "fixed_reverse_joint_order"])
SENSITIVITY_METRICS = dict(round="normalized_auc_round", nominal="nominal_normalized_auc")


def csv_bytes(rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else
                               repr(value) if isinstance(value, float) else value) for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def write(out, name, data, fields=None):
    path = Path(out) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = csv_bytes(data, fields) if name.endswith(".csv") else \
        (json.dumps(data, indent=1, sort_keys=True, default=float) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def completed_contexts(run_root, spec, protocol_cfg, bank):
    """Context keys whose every required phase has a completion record (partial mode only)."""
    fp32 = rn.resolve_members(protocol_cfg["fp32_contexts"], bank)
    keys = set()
    for entry in bank["contexts"]:
        if entry["status"] != "available":
            continue
        key = rn.context_key(entry)
        phases = ["bf16", "scoring"] + (["fp32"] if entry["context_id"] in fp32 else [])
        if all((Path(run_root) / spec["id"] / key / phase / "completion.json").is_file() for phase in phases):
            keys.add(key)
    return keys


def load_all(inputs, partial=False):
    """Per bank: every planned context (unreached calls included, for the complete-case rule) and the available ones."""
    loaded = {}
    for spec in inputs["banks"]:
        only = None
        if partial:
            protocol = json.loads(Path(spec["protocol"]).read_text())
            bank_json = json.loads(Path(spec["bank"]).read_text())
            only = completed_contexts(inputs["run_root"], spec, protocol["banks"][spec["id"]], bank_json)
            if not only:
                continue
        protocol, cfg, bank, contexts = methods.load_bank(inputs["run_root"], spec["protocol"], spec["bank"], spec["id"],
                                                          only=only)
        if partial:
            contexts = [c for c in contexts if c["entry"]["status"] == "available" and rn.context_key(c["entry"]) in only]
        loaded[spec["id"]] = dict(spec=spec, group=spec.get("group", spec["id"]), protocol=rn.bank_protocol(protocol, spec["id"]),
                                  cfg=cfg, bank=bank, contexts=contexts,
                                  available=[c for c in contexts if c["entry"]["status"] == "available"],
                                  unavailable=[c["entry"] for c in contexts if c["entry"]["status"] != "available"])
    return loaded


def summarize(rows, keys, field, thresholds=()):
    groups = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is None or isinstance(value, bool) or (isinstance(value, float) and not math.isfinite(value)):
            continue
        groups[tuple(row.get(k) for k in keys)].append(value)
    out = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        values = np.array(values, dtype=float)
        row = dict(zip(keys, key))
        row.update(n=len(values), median=float(np.median(values)), mean=float(values.mean()), max=float(values.max()),
                   min=float(values.min()))
        for name, test in thresholds:
            row[name] = int(sum(1 for v in values if test(v)))
        out.append(row)
    return out


def numerics_tables(loaded, auc_pairs=None):
    """Numerics rows for every available context and their summaries per group and per bank."""
    tables = defaultdict(list)
    for bank_id, data in loaded.items():
        for item in data["available"]:
            produced = dict(residuals=numerics.residual_rows(bank_id, item),
                            comparisons=numerics.comparison_rows(bank_id, item),
                            references=numerics.reference_rows(bank_id, item, data["protocol"]),
                            qualification=numerics.qualification_rows(bank_id, item),
                            nonfinite=numerics.nonfinite_rows(bank_id, item),
                            node_counts=numerics.node_count_rows(bank_id, item),
                            fp32_status=numerics.fp32_status_rows(bank_id, item),
                            auc_changes=numerics.auc_change_rows(bank_id, item, auc_pairs,
                                                                 data["protocol"]["numerics"]["comparator_rule"]),
                            scored_rankings=numerics.ranking_rows(bank_id, item,
                                                                  data["protocol"]["numerics"]["comparator_rule"]))
            for name, rows in produced.items():
                for row in rows:
                    row["group"] = data["group"]
                tables[name] += rows
    for row in tables["auc_changes"]:
        row["abs_change"] = None if row["change"] is None else abs(row["change"])
    maps = tables["comparisons"] + tables["references"]
    for unit in ("group", "bank"):
        suffix = "" if unit == "group" else "_by_bank"
        tables[f"residual_summary{suffix}"] = summarize(
            tables["residuals"], [unit, "modality", "target", "arm", "rule", "budget"], "relative_residual",
            [("over_1pct", lambda v: v > 0.01), ("over_5pct", lambda v: v > 0.05)])
        keys = [unit, "modality", "target", "family", "reference", "candidate"]
        tables[f"comparison_summary{suffix}"] = summarize(maps, keys, "relative_l1", [("over_0_01", lambda v: v > 0.01)])
        tables[f"spearman_summary{suffix}"] = summarize(maps, keys, "group_spearman", [("below_0_99", lambda v: v < 0.99)])
        tables[f"top5_overlap_summary{suffix}"] = summarize(maps, keys, "top5pct_overlap", [("below_0_95", lambda v: v < 0.95)])
        scored = [unit, "modality", "target", "reference", "candidate"]
        tables[f"scored_order_spearman_summary{suffix}"] = summarize(
            tables["scored_rankings"], scored, "order_spearman", [("below_0_99", lambda v: v < 0.99)])
        tables[f"scored_top5_overlap_summary{suffix}"] = summarize(
            tables["scored_rankings"], scored, "top5pct_overlap", [("below_0_95", lambda v: v < 0.95)])
        tables[f"scored_min_jaccard_summary{suffix}"] = summarize(
            tables["scored_rankings"], scored, "min_deleted_set_jaccard", [("below_0_9", lambda v: v < 0.9)])
        tables[f"auc_change_summary{suffix}"] = summarize(
            tables["auc_changes"], [unit, "modality", "candidate", "reference", "direction", "response"], "abs_change",
            [("over_0_01", lambda v: v > 0.01)])
        tables[f"qualification_summary{suffix}"] = qualification_summary(tables["qualification"], unit)
    tables["nonfinite_summary"] = numerics.nonfinite_summary(tables["node_counts"], tables["nonfinite"])
    return dict(tables)


def verify_gate_rows(run_root, loaded):
    """The verify phase's smoke gates for every context a bank's protocol sends through it.

    Each gate is read from the authenticated phase record: the repeated A1 passes bit-identical,
    the pinned integrated_gradients trapezoid bit-identical with A1 T_m and zero outside the
    population, and the archived historical kernel bit-identical with the A0 replica. A context
    whose verify phase is missing or fails authentication gets a row saying so.
    """
    rows = []
    for bank_id, data in loaded.items():
        members = rn.resolve_members(data["cfg"].get("verify_contexts", "none"), data["bank"])
        binding = dict(protocol_sha256=file_hash(data["spec"]["protocol"]), bank_sha256=data["cfg"]["bank_sha256"])
        for entry in data["bank"]["contexts"]:
            if entry["context_id"] not in members:
                continue
            base = dict(bank=bank_id, context=entry["context_id"])
            try:
                directory, _ = rn.load_phase(run_root, bank_id, rn.context_key(entry), "verify",
                                             dict(binding, context_id=entry["context_id"]))
            except ValueError as error:
                rows.append(dict(base, status="not_loaded", reason=str(error)))
                continue
            checks = strict_json(directory / "record.json")["checks"]
            for target, item in sorted(checks["targets"].items()):
                pinned, archived = item.get("pinned_kernel", {}), item["archived_kernel"]
                replica = archived["replica_nonfinite"]
                row = dict(base, status="loaded", modality=checks["modality"], target=target,
                           repeats=len(item["repeats_defined"]), repeats_defined=all(item["repeats_defined"]),
                           repeats_bit_identical=item["repeats_bit_identical"],
                           pinned_status=pinned.get("status", "evaluated"),
                           pinned_trapezoid_bit_identical=pinned.get("trapezoid_bit_identical"),
                           pinned_outside_population_nonzero=pinned.get("outside_population_nonzero"),
                           archived_bit_identical=archived["bit_identical"],
                           replica_failed_forwards=replica["failed_forwards"],
                           replica_nonfinite=replica["nan"] + replica["posinf"] + replica["neginf"])
                row["all_gates_pass"] = bool(row["repeats_bit_identical"] and row["pinned_trapezoid_bit_identical"]
                                             and row["pinned_outside_population_nonzero"] == 0
                                             and row["archived_bit_identical"])
                rows.append(row)
    return rows


def qualification_summary(rows, unit):
    groups = defaultdict(list)
    for row in rows:
        groups[(row[unit], row["modality"], row["target"])].append(row)
    out = []
    for key, items in sorted(groups.items()):
        floors = [r["resolution_floor"] for r in items if r["resolution_floor"] is not None]
        out.append(dict(zip((unit, "modality", "target"), key), contexts=len(items),
                        a4_converged=sum(bool(r["a4_converged"]) for r in items),
                        qualified=sum(r["qualified"] is True for r in items),
                        floor_median=float(np.median(floors)) if floors else None,
                        floor_max=float(max(floors)) if floors else None))
    return out


def summarize_rms(rows, plan):
    """Raw RMS area in action units, episode-weighted over all available calls, with a 95% interval."""
    boot = plan["methods_study"]["bootstrap"]
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["modality"], row["ranking"], row["direction"])].append(row)
    out = []
    for key, items in sorted(by_key.items()):
        series = methods.available_series(items, "raw_rms_auc")
        values = np.array([series[e] for e in dict.fromkeys(r["episode_id"] for r in items) if e in series], dtype=float)
        record = dict(zip(("modality", "ranking", "direction"), key), episodes=len(values),
                      mean=float(values.mean()) if len(values) else None)
        if len(values) > 1:
            samples = methods.resample(values[:, None], boot["draws"], boot["seed"])[:, 0]
            record["interval"] = methods.percentile(samples, methods.DESCRIPTIVE_ALPHA)
        out.append(record)
    return out


def exclusion_rows(bank_id, decisions, contrasts):
    rows = []
    for modality, decision in decisions.items():
        for direction in methods.DIRECTIONS:
            for episode, reason in decision["cells"][f"population/{direction}"]["excluded_episodes"].items():
                rows.append(dict(bank=bank_id, scope=f"decisions/{modality}/{direction}", episode_id=episode, reason=reason))
    for contrast in contrasts or []:
        for episode, reason in contrast["excluded_episodes"].items():
            rows.append(dict(bank=bank_id, scope=f"contrast/{contrast['id']}", episode_id=episode, reason=reason))
    return rows


def methods_tables(loaded, plan, primary_bank, method_names=VISION_METHODS, fixed=FIXED):
    out = {}
    for bank_id, data in loaded.items():
        scored = [c for c in data["available"] if "scoring" in c]
        if not scored:
            continue
        contexts = data["contexts"]
        modalities = sorted(scored[0]["scoring"]["modalities"])
        bank_out = dict(decisions={}, sensitivity={name: {} for name in SENSITIVITY_METRICS}, context_rows={},
                        shifts={}, rms=[])
        for modality in modalities:
            names = [n for n in method_names if n in scored[0]["scoring"]["modalities"][modality]["rankings"]]
            if not names:
                continue
            decision = methods.decisions(contexts, modality, names, fixed[modality], plan)
            bank_out["context_rows"][modality] = decision.pop("rows")
            bank_out["decisions"][modality] = decision
            for label, metric in SENSITIVITY_METRICS.items():
                other = methods.decisions(contexts, modality, names, fixed[modality], plan, metric=metric)
                other.pop("rows")
                bank_out["sensitivity"][label][modality] = other
            bank_out["shifts"][modality] = methods.response_shift_rows(scored, modality, names + fixed[modality])
            bank_out["rms"] += methods.rms_units_rows(scored, modality, names + fixed[modality])
        if "vision" in modalities:
            primary = bank_id == primary_bank
            bank_out["contrasts"] = dict(
                role="primary family, Bonferroni over 7" if primary else "replication, descriptive 95%",
                normalized_auc=methods.primary_contrasts(contexts, plan, alpha=None if primary else methods.DESCRIPTIVE_ALPHA),
                normalized_auc_round=methods.primary_contrasts(contexts, plan, metric="normalized_auc_round",
                                                               alpha=None if primary else methods.DESCRIPTIVE_ALPHA))
        out[bank_id] = bank_out
    return out


def population_counts(loaded, plan, primary_bank, method_names=VISION_METHODS, fixed=FIXED):
    """Complete-case episode counts of every decision cell and contrast, from definedness alone.

    Used before unblinding to choose the episode counts of the coverage simulation (amendment 2);
    no area, advantage or interval is summarized or written.
    """
    counts = {}
    for bank_id, data in loaded.items():
        scored = [c for c in data["available"] if "scoring" in c]
        if not scored:
            continue
        contexts = data["contexts"]
        planned = list(dict.fromkeys(item["entry"]["episode_id"] for item in contexts))
        for modality in sorted(scored[0]["scoring"]["modalities"]):
            names = [n for n in method_names if n in scored[0]["scoring"]["modalities"][modality]["rankings"]]
            rows = methods.context_table(contexts, modality, names + fixed[modality])
            for direction in methods.DIRECTIONS:
                kept, _ = methods.complete_case(rows, names + fixed[modality], direction, planned)
                counts[f"{bank_id}/decisions/{modality}/{direction}"] = len(kept)
        if "vision" in scored[0]["scoring"]["modalities"]:
            for contrast in plan["methods_study"]["primary_family"]:
                kept, _ = methods.contrast_population(contexts, contrast, "normalized_auc")
                counts[f"{bank_id}/contrast/{contrast['id']}"] = len(kept)
    return counts


def coverage_lookup(documents):
    """{(n, method, alpha): {scenario: coverage}} from the coverage simulation files."""
    table = defaultdict(dict)
    for document in documents:
        for key, cells in document["results"].items():
            scenario, n = key.rsplit("/n", 1)
            for cell, value in cells.items():
                method, alpha = cell.split("/")
                table[(int(n), method, round(float(alpha), 6))][scenario] = value["coverage"]
    return table


def attach_coverage(node, table):
    """Amendment 1: every interval record carries its method's simulated coverage at the realized episode count."""
    if isinstance(node, dict):
        if {"interval_method", "alpha", "episodes", "interval"} <= set(node):
            found = table.get((node["episodes"], node["interval_method"], round(node["alpha"], 6)))
            node["simulated_coverage"] = found if found else f"not simulated at {node['episodes']} episodes"
        for value in node.values():
            attach_coverage(value, table)
    elif isinstance(node, list):
        for value in node:
            attach_coverage(value, table)


def shift_summary(rows):
    keys = ["modality", "ranking", "direction", "success"]
    summary = summarize(rows, keys, "delta", [("positive", lambda v: v > 0), ("below_minus_1", lambda v: v < -1)])
    overshoot = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        k = tuple(row.get(key) for key in keys)
        overshoot[k][0] += int(row["overshoot"] is True)
        overshoot[k][1] += int(row["overshoot"] is None)
        overshoot[k][2] += 1
    for row in summary:
        k = tuple(row.get(key) for key in keys)
        row["overshoot_calls"], row["overshoot_undefined"], row["calls"] = overshoot[k]
    return summary


def episode_outcomes(bank_metrics_path):
    outcomes = {}
    for line in Path(bank_metrics_path).read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "episode_end":
            outcomes[row["episode"]] = bool(row["success"])
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--primary-bank", default="P")
    parser.add_argument("--partial", action="store_true",
                        help="engineering check on completed contexts only; not the registered analysis")
    parser.add_argument("--metrics-copy", action="append", default=[],
                        help="BANK=PATH of a local copy of a bank's metrics.jsonl, for episode outcomes")
    parser.add_argument("--coverage", action="append", default=[],
                        help="coverage simulation JSON (amendment 1); every interval gets its simulated coverage")
    parser.add_argument("--population-only", action="store_true",
                        help="write only the complete-case episode counts (population.json), for the coverage run")
    args = parser.parse_args()
    inputs = json.loads(Path(args.inputs).read_text())
    plan = json.loads(Path(args.plan).read_text())
    loaded = load_all(inputs, partial=args.partial)
    if args.population_only:
        counts = population_counts(loaded, plan, args.primary_bank, inputs.get("methods", VISION_METHODS))
        write(args.out, "population.json", dict(complete_case_episodes=counts,
                                                distinct_counts=sorted(set(counts.values()))))
        print(json.dumps(sorted(set(counts.values()))))
        return
    hashes = {}
    method_names = inputs.get("methods", VISION_METHODS)
    auc_pairs = [tuple(pair) for pair in inputs["auc_pairs"]] if "auc_pairs" in inputs else None
    for name, rows in numerics_tables(loaded, auc_pairs).items():
        fields = numerics.NONFINITE_FIELDS + ["group"] if name == "nonfinite" else None
        hashes[f"numerics/{name}.csv"] = write(args.out, f"numerics/{name}.csv", rows, fields)
    hashes["numerics/verify_gates.csv"] = write(args.out, "numerics/verify_gates.csv",
                                                verify_gate_rows(inputs["run_root"], loaded))
    coverage = coverage_lookup([json.loads(Path(path).read_text()) for path in args.coverage])
    results = methods_tables(loaded, plan, args.primary_bank, method_names)
    copies = dict(item.split("=", 1) for item in args.metrics_copy)
    for bank_id, bank_out in results.items():
        attach_coverage(bank_out["decisions"], coverage)
        attach_coverage(bank_out["sensitivity"], coverage)
        attach_coverage(bank_out.get("contrasts"), coverage)
        files = {f"methods/{bank_id}/decisions.json": bank_out["decisions"],
                 f"methods/{bank_id}/decisions_round_rule.json": bank_out["sensitivity"]["round"],
                 f"methods/{bank_id}/decisions_nominal_axis.json": bank_out["sensitivity"]["nominal"],
                 f"methods/{bank_id}/rms_units.csv": summarize_rms(bank_out["rms"], plan),
                 f"methods/{bank_id}/exclusions.csv": exclusion_rows(
                     bank_id, bank_out["decisions"], (bank_out.get("contrasts") or {}).get("normalized_auc"))}
        if "contrasts" in bank_out:
            files[f"methods/{bank_id}/contrasts.json"] = bank_out["contrasts"]
        for modality, rows in bank_out["context_rows"].items():
            files[f"methods/{bank_id}/context_rows_{modality}.csv"] = rows
        for name, data in files.items():
            fields = ["bank", "scope", "episode_id", "reason"] if name.endswith("exclusions.csv") else None
            hashes[name] = write(args.out, name, data, fields)
        outcomes = episode_outcomes(copies[bank_id]) if bank_id in copies else {}
        for modality, rows in bank_out["shifts"].items():
            for row in rows:
                row["success"] = outcomes.get(row["episode"])
            hashes[f"shifts/{bank_id}/rows_{modality}.csv"] = write(args.out, f"shifts/{bank_id}/rows_{modality}.csv", rows)
            hashes[f"shifts/{bank_id}/summary_{modality}.csv"] = write(args.out, f"shifts/{bank_id}/summary_{modality}.csv",
                                                                       shift_summary(rows))
    population = {bank_id: dict(planned=len(data["contexts"]), available=len(data["available"]),
                                unavailable=len(data["unavailable"]),
                                with_scoring=sum(1 for c in data["available"] if "scoring" in c),
                                with_fp32_evaluated=sum(1 for c in data["available"]
                                                        if c.get("fp32", {}).get("status") == "evaluated"),
                                fp32_probe_failures=sum(1 for c in data["available"]
                                                        if "fp32" in c and c["fp32"].get("status") != "evaluated"))
                  for bank_id, data in loaded.items()}
    manifest = dict(kind="nhsjs_r1_gpu_analysis_partial_engineering_check" if args.partial else "nhsjs_r1_gpu_analysis",
                    inputs=inputs, plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),
                    coverage_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in args.coverage},
                    population=population, outputs=hashes)
    write(args.out, "manifest.json", manifest)
    print(json.dumps(population, indent=1))


if __name__ == "__main__":
    main()
