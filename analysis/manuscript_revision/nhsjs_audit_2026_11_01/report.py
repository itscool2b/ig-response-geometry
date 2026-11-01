"""Reproduce retrospective winner stability and clustered success intervals.

This reporting layer does not amend the registered experiment or its outcomes.
Only archived context rows are read. Resampling always retains whole episodes,
with the same episode indices for both responses and every candidate method.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
R1 = ROOT / "analysis/revision_gpu/v1/results/r1-2026-10-24/analysis"
METHODS = ["IG_Q_A1T512", "IG_N_A1T512", "PG_Q_A1T512", "PG_N_A1T512",
           "input_difference", "IG_Q_A0m64", "IG_N_A0m64"]
SEED = 930011
DRAWS = 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def bootstrap_indices(n, draws=DRAWS, seed=SEED):
    if n < 2:
        raise ValueError("At least two independent episodes are required")
    rng = np.random.Generator(np.random.PCG64(seed))
    for begin in range(0, draws, 500):
        yield rng.integers(0, n, (min(500, draws - begin), n))


def paired_matrix(rows, episodes, methods=METHODS, field="advantage"):
    """Two responses x episodes x methods; reject any changed membership.

    Within each episode the archived calls receive equal weight. The episode
    order and exact complete-case membership come from the registered decisions.
    Scaled scores use the archived per-context random-order standard deviation.
    """
    values = defaultdict(dict)
    for r in rows:
        if r["direction"] != "deletion" or r["ranking"] not in methods or r["episode_id"] not in episodes:
            continue
        if r["response"] not in ("Q", "L2"):
            raise ValueError("Unexpected scoring response")
        key = r["response"], r["episode_id"], r["ranking"]
        call = r["context_id"]
        if call in values[key]:
            raise ValueError("Duplicate context/method/response row")
        value = r[field]
        if value in (None, "", "null") or not math.isfinite(float(value)):
            raise ValueError("Undefined value in the registered complete-case population")
        values[key][call] = float(value)
    matrix = []
    for response in ("Q", "L2"):
        population = []
        for episode in episodes:
            expected = set(values[("Q", episode, methods[0])])
            if not expected:
                raise ValueError("Missing registered episode")
            row = []
            for name in methods:
                current = values[(response, episode, name)]
                if set(current) != expected:
                    raise ValueError("Unmatched calls across responses or methods")
                row.append(math.fsum(current[c] for c in sorted(current)) / len(current))
            population.append(row)
        matrix.append(population)
    return np.asarray(matrix, dtype=float)


def leaders(values, methods):
    return [name for name, value in zip(methods, values) if value == np.max(values)]


def paired_winner_counts(q, n):
    """Classify each bootstrap draw without breaking an exact tie arbitrarily."""
    qmax = q == q.max(axis=1, keepdims=True)
    nmax = n == n.max(axis=1, keepdims=True)
    tied = (qmax.sum(axis=1) != 1) | (nmax.sum(axis=1) != 1)
    same = (qmax & nmax).any(axis=1) & ~tied
    different = ~tied & ~same
    return {"same_unique_winner": int(same.sum()),
            "different_unique_winners": int(different.sum()),
            "at_least_one_tie": int(tied.sum())}


def stability(raw, scaled, methods=METHODS, draws=DRAWS, seed=SEED):
    if raw.shape != scaled.shape or raw.ndim != 3 or raw.shape[0] != 2:
        raise ValueError("Expected matched response/episode/method arrays")
    if raw.shape[2] != len(methods) or not np.isfinite(raw).all() or not np.isfinite(scaled).all():
        raise ValueError("Invalid candidate values")
    out = {}
    for name, matrix, statistic in (("raw_mean", raw, np.mean),
                                    ("scaled_mean", scaled, np.mean),
                                    ("episode_median", raw, np.median)):
        point = statistic(matrix, axis=1)
        counts = dict.fromkeys(("same_unique_winner", "different_unique_winners", "at_least_one_tie"), 0)
        for index in bootstrap_indices(raw.shape[1], draws, seed):
            sampled = statistic(matrix[:, index, :], axis=2)
            current = paired_winner_counts(sampled[0], sampled[1])
            for key, value in current.items():
                counts[key] += value
        out[name] = {"episodes": raw.shape[1], "draws": draws, "seed": seed,
                     "point": {response: {"values": dict(zip(methods, map(float, vals))),
                                          "leaders": leaders(vals, methods)}
                               for response, vals in zip(("Q", "N"), point)},
                     "bootstrap_counts": counts,
                     "bootstrap_shares": {key: value / draws for key, value in counts.items()}}
    return out


def clustered_median(values_by_episode, draws=DRAWS, seed=SEED):
    """Percentile interval for the pooled call median, resampling episodes."""
    episodes = list(values_by_episode)
    groups = [np.asarray(values_by_episode[e], dtype=float) for e in episodes]
    if any(not len(g) or not np.isfinite(g).all() for g in groups):
        raise ValueError("Empty or nonfinite episode")
    samples = []
    if len({len(g) for g in groups}) == 1:
        matrix = np.asarray(groups)
        for index in bootstrap_indices(len(groups), draws, seed):
            samples.extend(np.median(matrix[index].reshape(len(index), -1), axis=1).tolist())
    else:
        for index in bootstrap_indices(len(groups), draws, seed):
            samples.extend(float(np.median(np.concatenate([groups[i] for i in draw]))) for draw in index)
    return {"median": float(np.median(np.concatenate(groups))),
            "ci_lo": float(np.quantile(samples, .025)), "ci_hi": float(np.quantile(samples, .975)),
            "episodes": len(groups), "calls": sum(map(len, groups)), "draws": draws, "seed": seed}


def success_summary(rows, draws=DRAWS, seed=SEED):
    out = []
    for direction in ("insertion", "deletion"):
        for success in ("True", "False"):
            chosen = [r for r in rows if r["ranking"] == "IG_Q_A1T512" and r["direction"] == direction and r["success"] == success]
            by_episode = defaultdict(list)
            for row in chosen:
                by_episode[row["episode_id"]].append(float(row["delta"]))
            record = clustered_median(dict(sorted(by_episode.items())), draws, seed)
            record.update(direction=direction, success=success == "True", ranking="IG_Q_A1T512",
                          positive=sum(float(r["delta"]) > 0 for r in chosen),
                          overshoot=sum(r["overshoot"] == "True" for r in chosen),
                          below_minus_1=sum(float(r["delta"]) < -1 for r in chosen),
                          minimum=min(float(r["delta"]) for r in chosen),
                          episode_ids=sorted(by_episode))
            out.append(record)
    return out


def build(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    used = {Path(__file__).relative_to(ROOT).as_posix(): sha(__file__)}
    def read(path, kind="csv"):
        used[path.relative_to(ROOT).as_posix()] = sha(path)
        return read_csv(path) if kind == "csv" else json.loads(path.read_text(encoding="utf-8"))
    report = {"schema_version": 1, "status": "retrospective descriptive reporting",
              "bootstrap": {"unit": "whole episode", "draws": DRAWS, "seed": SEED, "generator": "PCG64",
                            "pairing": "same episode draw across both responses and all seven methods",
                            "ties": "exact ties remain an explicit undetermined category",
                            "interpretation": "resampling stability, not probability of a true population winner"},
              "methods": {}}
    for bank in ("P", "B1", "S"):
        decisions = read(R1 / "methods" / bank / "decisions.json", "json")["vision"]["cells"]
        episodes = decisions["Q/deletion"]["advantage"]["episode_ids"]
        if episodes != decisions["L2/deletion"]["advantage"]["episode_ids"]:
            raise ValueError("Registered response populations differ")
        rows = read(R1 / "methods" / bank / "context_rows_vision.csv")
        raw = paired_matrix(rows, episodes)
        scaled = paired_matrix(rows, episodes, field="z")
        result = stability(raw, scaled)
        result["episode_ids"] = episodes
        for criterion, archived in (("raw_mean", "advantage"), ("scaled_mean", "z_scaled")):
            for response, key in (("Q", "Q/deletion"), ("N", "L2/deletion")):
                expected = decisions[key][archived]
                current = result[criterion]["point"][response]
                if current["leaders"] != ([expected["best"]] if expected["best"] else expected["tie"]):
                    raise ValueError("Recomputed winner differs from registered output")
                if any(not math.isclose(current["values"][m], expected["mean"][m], rel_tol=1e-12, abs_tol=1e-12) for m in METHODS):
                    raise ValueError("Recomputed mean differs from registered output")
        report["methods"][bank] = result
    report["success_intervals"] = success_summary(read(R1 / "shifts/B1/rows_vision.csv"))
    report["inputs_sha256"] = dict(sorted(used.items()))
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {path}; {len(used)} authenticated inputs")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new directory; existing results are never overwritten")
    build(parser.parse_args().output)
