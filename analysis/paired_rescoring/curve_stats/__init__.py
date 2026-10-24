"""Shared helpers for the 2026-10-23 saved-curve analyses (CPU only).

Nothing here executes a model or rewrites an input. Every input is bound to
the canonical input manifest, physical-line ledger and provenance anchor that
the frozen paired analysis uses. Means use ``math.fsum`` and order statistics
use exact comparisons, so the written bytes do not depend on a platform's
summation order. The bootstrap index stream is the frozen paired analysis'
stream (``analysis.revision.core.bootstrap_summary``): numpy PCG64 seeded once,
drawing whole recorded episodes in blocks of at most 128 draws.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import io
import json
import math
from pathlib import Path

import numpy as np

from analysis.paired_rescoring import analyze as paired
from analysis.revision.analyze import check_stored_arithmetic
from analysis.revision.core import IntegrityError, json_bytes, load_file, reconcile, sha256
from analysis.revision.verify import verify as verify_canonical

ROOT = Path(__file__).resolve().parents[3]
CANONICAL = "analysis/revision/results/2026-09-30-v2"
MANIFEST = "analysis/revision/input_manifest.json"
CATEGORIES = paired.CATEGORIES
BLOCK = 128
STEP_KEY = ("task", "model", "seed", "episode", "policy_call_idx")
EPISODE_FIELDS = ("task", "model", "seed", "episode")


# --- deterministic output -------------------------------------------------

def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path, value):
    Path(path).write_text(json_text(value), encoding="utf-8", newline="\n")


def csv_bytes(rows, fields=None):
    """CSV with a fixed column order; floats use repr, lists/dicts JSON."""
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        clean = {}
        for key in fields:
            value = row.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                raise IntegrityError(f"Nonfinite output value: {key}")
            clean[key] = json_bytes(value).decode() if isinstance(value, (list, dict)) else value
        writer.writerow(clean)
    return stream.getvalue().encode("utf-8")


def gzip_bytes(blob):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as stream:
        stream.write(blob)
    return buffer.getvalue()


def artifact_record(path):
    """SHA-256 of the stored bytes and, for gzip, of the decompressed payload."""
    blob = Path(path).read_bytes()
    record = dict(sha256=sha256(blob), bytes=len(blob))
    if Path(path).suffix == ".gz":
        record["uncompressed_sha256"] = sha256(gzip.decompress(blob))
    return record


# --- exact summaries ------------------------------------------------------

def fmean(values):
    values = list(values)
    if not values:
        raise IntegrityError("Mean of an empty population")
    return math.fsum(values) / len(values)


def median(values):
    ordered = sorted(values)
    if not ordered:
        raise IntegrityError("Median of an empty population")
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2


def percentile(sorted_values, q_percent):
    """numpy.percentile(method='linear') reproduced in pure Python arithmetic."""
    n = len(sorted_values)
    if not n:
        raise IntegrityError("Percentile of an empty population")
    if any(b < a for a, b in zip(sorted_values, sorted_values[1:])):
        raise IntegrityError("Percentile input must be sorted")
    q = q_percent / 100
    virtual = (n - 1) * q
    if virtual >= n - 1:
        return sorted_values[-1]
    if virtual < 0:
        return sorted_values[0]
    previous = math.floor(virtual)
    gamma = virtual - previous
    a, b = sorted_values[previous], sorted_values[previous + 1]
    difference = b - a
    if gamma >= 0.5:
        return b - difference * (1 - gamma)
    return a + difference * gamma


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman rho with average ranks; None when undefined (n<3 or constant)."""
    if len(x) != len(y):
        raise IntegrityError("Spearman inputs differ in length")
    if any(not math.isfinite(v) for v in [*x, *y]):
        raise IntegrityError("Nonfinite Spearman input")
    n = len(x)
    if n < 3:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = math.fsum(rx) / n, math.fsum(ry) / n
    dx = [v - mx for v in rx]
    dy = [v - my for v in ry]
    sxx, syy = math.fsum(v * v for v in dx), math.fsum(v * v for v in dy)
    if sxx == 0 or syy == 0:
        return None
    return math.fsum(a * b for a, b in zip(dx, dy)) / math.sqrt(sxx * syy)


def bootstrap_blocks(n_groups, draws, seed):
    """Yield whole-episode index blocks exactly as core.bootstrap_summary draws them."""
    rng = np.random.default_rng(seed)
    for start in range(0, draws, BLOCK):
        yield rng.integers(0, n_groups, size=(min(BLOCK, draws - start), n_groups))


def episode_bootstrap(calls, *, draws, seed):
    """Paired call median and equal-episode mean with one shared resampling.

    Each draw selects as many whole recorded episodes as observed, with
    replacement; all of a selected episode's defined calls stay together.
    The median uses every resampled call with multiplicity; the mean averages
    the resampled within-episode means with math.fsum.
    """
    groups = sorted({row["episode_id"] for row in calls})
    lookup = {key: i for i, key in enumerate(groups)}
    by_group = defaultdict(list)
    for row in calls:
        by_group[row["episode_id"]].append(row["delta"])
    episode_means = [fmean(by_group[key]) for key in groups]
    values = [row["delta"] for row in calls]
    if not values or any(v is None or not math.isfinite(v) for v in values):
        raise IntegrityError("Bootstrap requires a nonempty finite defined population")
    result = dict(n_groups=len(groups), n_rows=len(values), draws=draws, seed=seed,
                  median_point=median(values), mean_point=fmean(episode_means))
    if len(groups) < 2:
        result.update(median_ci_lo=None, median_ci_hi=None, mean_ci_lo=None, mean_ci_hi=None,
                      interval_status="insufficient_groups")
        return result
    order = sorted(range(len(values)), key=lambda i: values[i])
    ordered_values = [values[i] for i in order]
    ordered_codes = np.array([lookup[calls[i]["episode_id"]] for i in order], dtype=np.int64)
    medians, means = [], []
    for sampled in bootstrap_blocks(len(groups), draws, seed):
        n = sampled.shape[0]
        counts = np.zeros((n, len(groups)), dtype=np.int64)
        np.add.at(counts, (np.arange(n)[:, None], sampled), 1)
        weights = counts[:, ordered_codes]
        cumulative = weights.cumsum(axis=1)
        totals = weights.sum(axis=1)
        low = (totals - 1) // 2
        high = totals // 2
        first = np.argmax(cumulative > low[:, None], axis=1).tolist()
        second = np.argmax(cumulative > high[:, None], axis=1).tolist()
        medians.extend((ordered_values[a] + ordered_values[b]) / 2 for a, b in zip(first, second))
        for row in sampled.tolist():
            means.append(math.fsum(episode_means[i] for i in row) / len(groups))
    medians.sort()
    means.sort()
    result.update(median_ci_lo=percentile(medians, 2.5), median_ci_hi=percentile(medians, 97.5),
                  mean_ci_lo=percentile(means, 2.5), mean_ci_hi=percentile(means, 97.5),
                  interval_status="two_group_descriptive_only" if len(groups) == 2
                  else "conditional_empirical_episode_percentile")
    return result


# --- input authentication -------------------------------------------------

def canonical_anchor(root):
    canonical = Path(root) / CANONICAL
    provenance_bytes = (canonical / "provenance.json").read_bytes()
    if sha256(provenance_bytes) != paired.CANONICAL_PROVENANCE_SHA256:
        raise IntegrityError("Canonical provenance differs from the reviewed v2 anchor")
    provenance = json.loads(provenance_bytes)
    manifest_bytes = (Path(root) / MANIFEST).read_bytes()
    if sha256(manifest_bytes) != provenance["manifest_sha256"]:
        raise IntegrityError("Input manifest mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest["files"] != provenance["inputs"]:
        raise IntegrityError("Canonical input/provenance mismatch")
    for name, expected in provenance["code_sha256"].items():
        if sha256((Path(root) / "analysis/revision" / name).read_bytes()) != expected:
            raise IntegrityError(f"Canonical code mismatch: {name}")
    return provenance, manifest


def authenticate_named(root, names, *, expected=None):
    """Load exactly the named manifest files; every record must be in the ledger.

    ``expected`` optionally maps a path to the canonical SHA-256 declared in a
    protocol; a mismatch with the manifest fails before any record is read.
    """
    root = Path(root)
    provenance, manifest = canonical_anchor(root)
    verification = verify_canonical(root / CANONICAL)
    entries = {entry["path"]: entry for entry in manifest["files"]}
    if len(set(names)) != len(names):
        raise IntegrityError("A declared input is repeated")
    ledger = csv.DictReader(io.StringIO(gzip.decompress(
        (root / CANONICAL / "raw_line_ledger.csv.gz").read_bytes()).decode()))
    known = {f"{r['source']}:{r['line']}:{r['line_sha256']}" for r in ledger if r["status"] == "valid"}
    loaded, audit = {}, []
    for name in names:
        entry = entries.get(name)
        if entry is None:
            raise IntegrityError(f"Declared input is absent from the manifest: {name}")
        if expected is not None and expected.get(name) != entry["sha256"]:
            raise IntegrityError(f"Protocol hash differs from the manifest: {name}")
        observed = sha256((root / name).read_bytes())
        records, _ = load_file(root / name, root, entry)
        if not {r.id for r in records} <= known:
            raise IntegrityError(f"Records absent from the canonical ledger: {name}")
        last, clean, duplicates = reconcile(records)
        arithmetic = check_stored_arithmetic(records)
        loaded[name] = dict(records=records, last=last, clean=clean, duplicates=duplicates)
        audit.append(dict(
            path=name, sha256=entry["sha256"], checkout_sha256=observed,
            checkout_form="canonical" if observed == entry["sha256"] else "declared_git_crlf",
            physical_lines=entry["physical_lines"], valid_records=entry["valid_records"],
            events=dict(sorted(Counter(r.row["event"] for r in records).items())),
            selected_records=len(last), conflict_excluded_records=len(clean),
            duplicate_groups=len(duplicates),
            duplicate_classes=dict(sorted(Counter(d["classification"] for d in duplicates).items())),
            duplicate_episodes=sorted({d["key"][5] for d in duplicates}),
            duplicate_differing_fields=sorted({f for d in duplicates for f in d["differing_fields"]}),
            stored_arithmetic_checks=dict(sorted(arithmetic["counts"].items())),
            stored_arithmetic_maximum_discrepancies=dict(sorted(arithmetic["maximum_discrepancies"].items())),
            stored_arithmetic_undefined_auc=len(arithmetic["undefined_auc"])))
    anchor = dict(canonical_provenance_sha256=paired.CANONICAL_PROVENANCE_SHA256,
                  input_manifest_sha256=provenance["manifest_sha256"],
                  canonical_verification=verification)
    return loaded, audit, anchor


# --- joins and fingerprints -----------------------------------------------

def relative_close(a, b, tolerance=1e-6):
    return abs(a - b) <= tolerance * max(abs(a), abs(b), 1e-12)


def step_index(records):
    index = {}
    for record in records:
        if record.row["event"] != "step":
            continue
        key = tuple(record.row[k] for k in STEP_KEY)
        if key in index:
            raise IntegrityError("Repeated step key after reconciliation")
        index[key] = record
    return index


def fingerprint(faith, steps, modalities=("vision", "lang")):
    """Month 7 own-sidecar fingerprint, plus the endpoint-general form.

    ``original``: -step gap equals the faithfulness baseline score.
    ``general``: step gap equals faithfulness f_input - f_baseline; it equals
    ``original`` when the stored actual score is exactly zero.
    """
    index = step_index(steps)
    out = dict(n_faithfulness=len(faith), n_joined=0, missing=[])
    for modality in modalities:
        out[f"{modality}_original_agree"] = 0
        out[f"{modality}_general_agree"] = 0
    for record in faith:
        step = index.get(tuple(record.row[k] for k in STEP_KEY))
        if step is None:
            out["missing"].append(record.id)
            continue
        out["n_joined"] += 1
        for modality in modalities:
            gap = step.row[f"{modality}_gap"]
            f_input, f_baseline = record.row[f"{modality}_f_input"], record.row[f"{modality}_f_baseline"]
            out[f"{modality}_original_agree"] += relative_close(-gap, f_baseline)
            out[f"{modality}_general_agree"] += relative_close(gap, f_input - f_baseline)
    out["n_missing"] = len(out.pop("missing"))
    for form in ("original", "general"):
        out[f"all_agree_{form}"] = out["n_missing"] == 0 and all(
            out[f"{m}_{form}_agree"] == len(faith) for m in modalities)
    return out


def episode_ends(records):
    ends = {}
    for record in records:
        if record.row["event"] != "episode_end":
            continue
        key = tuple(record.row[k] for k in EPISODE_FIELDS)
        if key in ends:
            raise IntegrityError("Repeated episode_end after reconciliation")
        if type(record.row.get("success")) is not bool:
            raise IntegrityError("Missing or non-boolean success label")
        ends[key] = record
    return ends


def faithfulness_rows(records):
    rows = [r for r in records if r.row["event"] == "faithfulness"]
    if len(rows) != len(records):
        raise IntegrityError("Faithfulness file carries another event type")
    return sorted(rows, key=lambda r: (tuple(r.row[k] for k in STEP_KEY), r.line))


def episode_structure(faith, ends):
    """Every episode needs contiguous calls 0..n-1 and one matching episode_end."""
    calls = defaultdict(list)
    for record in faith:
        calls[tuple(record.row[k] for k in EPISODE_FIELDS)].append(record.row["policy_call_idx"])
    summary = dict(n_episodes=len(calls), n_calls=len(faith), successes=0, failures=0,
                   success_calls=0, failure_calls=0, call_count_mismatches=0)
    labels = {}
    for key, indices in sorted(calls.items()):
        if sorted(indices) != list(range(len(indices))):
            raise IntegrityError(f"Noncontiguous policy calls in episode {key}")
        end = ends.get(key)
        if end is None:
            raise IntegrityError(f"Faithfulness episode lacks an episode_end: {key}")
        success = end.row["success"]
        labels[key] = success
        summary["successes" if success else "failures"] += 1
        summary["success_calls" if success else "failure_calls"] += len(indices)
        summary["call_count_mismatches"] += end.row.get("policy_calls") != len(indices)
    return summary, labels


# --- rescoring ------------------------------------------------------------

def native_scale_check(faith, native, modalities=("vision", "lang")):
    """A declared native Q curve stores exactly zero at the actual input."""
    if native != "Q":
        raise IntegrityError("These analyses declare Q-native saved curves only")
    bad = [r.id for r in faith for m in modalities if r.row[f"{m}_f_input"] != 0]
    if bad:
        raise IntegrityError(f"Declared Q-native curve has a nonzero actual score: {bad[0]}")
    return dict(native_scale=native, n_rows=len(faith), n_nonzero_actual_scores=0)


def rescore(record, modality, direction, *, native="Q", dimension=512, epsilon=1e-12):
    row = record.row
    result = paired.paired_curve(row["k_grid_auc"], row[f"{modality}_{direction}_curve"],
                                 row[f"{modality}_f_input"], row[f"{modality}_f_baseline"],
                                 native, dimension=dimension, epsilon=epsilon)
    interior = result["residual_points"][1:-1]
    key = list(paired.episode_key(record))
    return dict(record_id=record.id, source=record.source, line=record.line,
                episode_key=key, episode_id=sha256(json_bytes(key)),
                policy_call_idx=row["policy_call_idx"], modality=modality, direction=direction,
                paired_status=result["paired_status"], delta=result["delta"],
                auc_q=result["auc_q"], auc_l2=result["auc_l2"],
                gap_q=result["gap_q"], gap_l2=result["gap_l2"],
                residual_actual=result["residual_actual"], residual_baseline=result["residual_baseline"],
                max_interior_residual=max(interior) if interior else None,
                residual_points=result["residual_points"],
                n_grid_nodes=result["n_grid_nodes"],
                **{f"n_{c}": result[f"n_{c}"] for c in CATEGORIES},
                **{f"contribution_{c}": result[f"contribution_{c}"] for c in CATEGORIES},
                decomposition_residual=result["decomposition_residual"],
                generalized_identity_max_error=result["generalized_identity_max_error"])


def any_overshoot(row):
    return row["n_high_distance_overshoot"] > 0


def fully_in_range(row):
    return row["n_high_distance_overshoot"] == 0 and row["n_low_distance_undershoot"] == 0


def share(count, total):
    return count / total if total else None


# --- population summaries -------------------------------------------------

def summarize(calls, *, draws, seed):
    """Call-level and episode-level statistics for one population and case.

    An empty population (for example, the success group of a run without a
    success) keeps its row with zero counts and no statistics.
    """
    if len({r["record_id"] for r in calls}) != len(calls):
        raise IntegrityError("Repeated call in a population")
    defined = [r for r in calls if r["paired_status"] == "defined"]
    if any(r["delta"] is None or not math.isfinite(r["delta"]) for r in defined):
        raise IntegrityError("Nonfinite defined paired effect")
    raw_episodes = {r["episode_id"] for r in calls}
    by_episode = defaultdict(list)
    for row in defined:
        by_episode[row["episode_id"]].append(row)
    membership = sha256(json_bytes(sorted(r["record_id"] for r in calls)))
    out = dict(n_raw_calls=len(calls), n_defined_calls=len(defined),
               n_undefined_calls=len(calls) - len(defined), n_raw_episodes=len(raw_episodes),
               n_defined_episodes=len(by_episode), population_sha256=membership)
    if not defined:
        return out
    values = [r["delta"] for r in defined]
    lengths = [len(rows) for rows in by_episode.values()]
    overshoot = [r for r in calls if any_overshoot(r)]
    defined_overshoot = [r for r in defined if any_overshoot(r)]
    in_range = [r for r in defined if fully_in_range(r)]
    boot = episode_bootstrap(defined, draws=draws, seed=seed)
    episode_means = {k: fmean([r["delta"] for r in rows]) for k, rows in by_episode.items()}
    contribution = {c: fmean([fmean([r[f"contribution_{c}"] for r in rows]) for rows in by_episode.values()])
                    for c in CATEGORIES}
    episode_overshoot = sum(any(any_overshoot(r) for r in rows) for rows in by_episode.values())
    episode_in_range = sum(all(fully_in_range(r) for r in rows) for rows in by_episode.values())
    out.update(
        calls_per_episode_min=min(lengths), calls_per_episode_max=max(lengths),
        paired_median=boot["median_point"], paired_median_ci_lo=boot["median_ci_lo"],
        paired_median_ci_hi=boot["median_ci_hi"],
        positive_calls=sum(v > 0 for v in values), negative_calls=sum(v < 0 for v in values),
        zero_calls=sum(v == 0 for v in values),
        positive_share=share(sum(v > 0 for v in values), len(values)),
        overshoot_calls=len(overshoot), overshoot_share=share(len(overshoot), len(calls)),
        undershoot_calls=sum(r["n_low_distance_undershoot"] > 0 for r in calls),
        fully_in_range_calls=len(in_range), fully_in_range_share=share(len(in_range), len(defined)),
        fully_in_range_positive_calls=sum(r["delta"] > 0 for r in in_range),
        overshoot_positive_calls=sum(r["delta"] > 0 for r in defined_overshoot),
        overshoot_positive_share=share(sum(r["delta"] > 0 for r in defined_overshoot), len(defined_overshoot)),
        min_delta=min(values), max_delta=max(values), equal_call_mean=fmean(values),
        equal_episode_mean=boot["mean_point"], equal_episode_mean_ci_lo=boot["mean_ci_lo"],
        equal_episode_mean_ci_hi=boot["mean_ci_hi"],
        **{f"equal_episode_contribution_{c}": contribution[c] for c in CATEGORIES},
        decomposition_check=boot["mean_point"] - math.fsum(contribution.values()),
        median_episode_mean=median(episode_means.values()),
        positive_episode_means=sum(v > 0 for v in episode_means.values()),
        positive_episode_mean_share=share(sum(v > 0 for v in episode_means.values()), len(episode_means)),
        episodes_any_overshoot=episode_overshoot,
        episodes_any_overshoot_share=share(episode_overshoot, len(by_episode)),
        episodes_fully_in_range=episode_in_range,
        episodes_fully_in_range_share=share(episode_in_range, len(by_episode)),
        min_episode_mean=min(episode_means.values()), max_episode_mean=max(episode_means.values()),
        interval_status=boot["interval_status"], bootstrap_draws=draws, bootstrap_seed=seed)
    if abs(out["decomposition_check"]) > 1e-9 * max(1.0, abs(boot["mean_point"])):
        raise IntegrityError("Equal-episode contributions do not sum to the mean")
    return out


def call_index_summary(calls):
    """First (policy_call_idx 0) versus later calls on the case's own weights.

    Contributions use the equal-episode weights 1/(E*n_e) of the whole
    population, so first plus later contributions sum to the equal-episode
    mean. Medians, means and shares inside each subgroup weight calls equally.
    """
    defined = [r for r in calls if r["paired_status"] == "defined"]
    by_episode = defaultdict(list)
    for row in defined:
        by_episode[row["episode_id"]].append(row)
    weight = {r["record_id"]: 1 / (len(by_episode) * len(rows)) for rows in by_episode.values() for r in rows}
    out = dict(n_raw_calls=len(calls), n_defined_calls=len(defined), n_defined_episodes=len(by_episode))
    for name, predicate in (("first", lambda i: i == 0), ("later", lambda i: i > 0)):
        raw = [r for r in calls if predicate(r["policy_call_idx"])]
        kept = [r for r in raw if r["paired_status"] == "defined"]
        values = [r["delta"] for r in kept]
        part = dict(raw_calls=len(raw), defined_calls=len(kept),
                    median=median(values) if values else None,
                    equal_call_mean=fmean(values) if values else None,
                    min_delta=min(values) if values else None, max_delta=max(values) if values else None,
                    positive_share=share(sum(v > 0 for v in values), len(values)),
                    overshoot_calls=sum(any_overshoot(r) for r in raw),
                    overshoot_share=share(sum(any_overshoot(r) for r in raw), len(raw)),
                    fully_in_range_share=share(sum(fully_in_range(r) for r in kept), len(kept)),
                    mean_contribution=math.fsum(weight[r["record_id"]] * r["delta"] for r in kept) if weight else None,
                    **{f"{c}_contribution": math.fsum(weight[r["record_id"]] * r[f"contribution_{c}"] for r in kept)
                       if weight else None for c in ("in_range", "high_distance_overshoot")})
        out.update({f"{name}_{key}": value for key, value in part.items()})
    if weight:
        out["first_plus_later_contribution"] = out["first_mean_contribution"] + out["later_mean_contribution"]
    return out


def episode_rows(calls):
    by_episode = defaultdict(list)
    for row in calls:
        by_episode[row["episode_id"]].append(row)
    rows = []
    for key, members in sorted(by_episode.items(), key=lambda item: item[1][0]["episode_key"]):
        defined = [r for r in members if r["paired_status"] == "defined"]
        first = [r for r in members if r["policy_call_idx"] == 0]
        rows.append(dict(
            episode_id=key, episode_key=members[0]["episode_key"], n_calls=len(members),
            n_defined_calls=len(defined),
            mean_delta=fmean([r["delta"] for r in defined]) if defined else None,
            median_delta=median([r["delta"] for r in defined]) if defined else None,
            **{f"mean_contribution_{c}": fmean([r[f"contribution_{c}"] for r in defined]) if defined else None
               for c in CATEGORIES},
            overshoot_calls=sum(any_overshoot(r) for r in members),
            fully_in_range_calls=sum(fully_in_range(r) for r in defined),
            first_call_delta=first[0]["delta"] if first else None))
    return rows
