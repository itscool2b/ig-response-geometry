"""Record identity, reconciliation, curve arithmetic and episode resampling.

No repository legacy loader or bootstrap implementation is imported here.
Historical context identity is never inferred from filenames or scalar norms.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class IntegrityError(ValueError):
    """An input cannot be included without changing the declared contract."""


def sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def check_finite(value, location="record"):
    if isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            check_finite(item, f"{location}[{i}]")
    elif value is None:
        raise IntegrityError(f"Unexpected null: {location}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise IntegrityError(f"Nonfinite number: {location}")


def strict_json(raw: bytes) -> dict:
    try:
        row = json.loads(raw, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Malformed JSON: {exc}") from exc
    if not isinstance(row, dict):
        raise IntegrityError("Expected a JSON object")
    check_finite(row)
    for key in ("event", "task", "model"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise IntegrityError(f"Missing/invalid identity field {key}")
    for key in ("seed", "episode"):
        if type(row.get(key)) is not int or row[key] < 0:
            raise IntegrityError(f"Missing/invalid identity field {key}")
    if row["event"] != "episode_end":
        if type(row.get("policy_call_idx")) is not int or row["policy_call_idx"] < 0:
            raise IntegrityError("Missing/invalid policy_call_idx")
    return row


@dataclass(frozen=True)
class Record:
    source: str
    source_sha256: str
    line: int
    line_sha256: str
    row: dict

    @property
    def id(self):
        return f"{self.source}:{self.line}:{self.line_sha256}"

    @property
    def key(self):
        # Filename is a run namespace, not proof of context identity. Including
        # the source prevents accidental pooling of distinct historical runs.
        return (self.source,) + tuple(self.row.get(k) for k in (
            "event", "task", "model", "seed", "episode", "policy_call_idx",
            "solver_steps", "modality"))


def load_file(path: Path, root: Path, expected: dict) -> tuple[list[Record], list[dict]]:
    source = path.relative_to(root).as_posix()
    blob = path.read_bytes()
    digest = sha256(blob)
    if digest != expected["sha256"]:
        # Git may expand LF to CRLF on checkout. Accept only a separately
        # declared exact byte variant, then authenticate the original bytes.
        # Do not accept arbitrary whitespace, mixed endings, or JSON rewrites.
        if digest != expected.get("git_crlf_checkout_sha256"):
            raise IntegrityError(f"Source hash mismatch: {source}")
        blob = blob.replace(b"\r\n", b"\n")
        digest = sha256(blob)
        if digest != expected["sha256"]:
            raise IntegrityError(f"Checkout conversion mismatch: {source}")
    lines = blob.splitlines(keepends=True)
    if len(lines) != expected["physical_lines"]:
        raise IntegrityError(f"Physical line count mismatch: {source}")
    quarantines = {q["line"]: q for q in expected.get("quarantine", [])}
    records, ledger = [], []
    for number, raw in enumerate(lines, 1):
        line_hash = sha256(raw)
        entry = dict(source=source, source_sha256=digest, line=number,
                     line_sha256=line_hash, bytes=len(raw))
        if number in quarantines:
            quarantine = quarantines[number]
            if line_hash != quarantine["line_sha256"]:
                raise IntegrityError(f"Quarantine hash mismatch: {source}:{number}")
            # Only the exact predeclared malformed bytes may be excluded.
            try:
                strict_json(raw)
            except IntegrityError:
                pass
            else:
                raise IntegrityError("Quarantine unexpectedly contains a valid row")
            entry.update(status="quarantined", reason=quarantine["reason"])
        else:
            try:
                row = strict_json(raw)
            except IntegrityError as exc:
                raise IntegrityError(f"{source}:{number}: {exc}") from exc
            record = Record(source, digest, number, line_hash, row)
            records.append(record)
            entry.update(status="valid", event=row["event"],
                         key_sha256=sha256(json_bytes(record.key)),
                         payload_sha256=sha256(json_bytes(row)),
                         context_identity="unknown", checkpoint_identity="unknown")
        ledger.append(entry)
    if len(records) != expected["valid_records"]:
        raise IntegrityError(f"Valid record count mismatch: {source}")
    return records, ledger


def reconcile(records: list[Record]):
    groups = defaultdict(list)
    for record in records:
        groups[record.key].append(record)
    last, conflict_excluded, duplicates = [], [], []
    for key, group in groups.items():
        selected = group[-1]
        last.append(selected)
        # Wall time is execution metadata; every other field is preserved in
        # scientific payload comparisons, including source sidecar references.
        payloads = [json_bytes({k: v for k, v in r.row.items()
                                if k != "wall_seconds"}) for r in group]
        conflict = len(set(payloads)) > 1
        if not conflict:
            conflict_excluded.append(selected)
        if len(group) > 1:
            fields = sorted(set().union(*(r.row.keys() for r in group)))
            changed = [k for k in fields
                       if len({json_bytes(r.row.get(k)) for r in group}) > 1]
            duplicates.append(dict(
                source=selected.source, key=list(key),
                key_sha256=sha256(json_bytes(key)), occurrences=[r.id for r in group],
                extra_occurrences=len(group) - 1, differing_fields=changed,
                classification="conflicting_payload" if conflict else
                ("metadata_only" if changed else "identical_payload"),
                retrospective_last_selected=selected.id,
                conflict_excluded_selected=None if conflict else selected.id))
    return last, conflict_excluded, duplicates


def population_id(records: list[Record]) -> str:
    identities = [record.id for record in records]
    if len(identities) != len(set(identities)):
        raise IntegrityError("A result population repeats a physical record")
    return sha256(json_bytes(sorted(identities)))


def normalized_auc(grid, curve, f_input, f_baseline, *, denominator_min=0.0):
    """Return undefined explicitly for zero/small endpoint gaps; never clip AUC.

    The default excludes only exact degeneracy. A positive cutoff is a declared
    sensitivity, not an implicit action-norm proxy. Historical code used 1e-9.
    """
    check_finite([grid, curve, f_input, f_baseline, denominator_min])
    if denominator_min < 0 or len(grid) != len(curve) or len(grid) < 2:
        raise IntegrityError("Invalid curve lengths or denominator cutoff")
    if grid[0] != 0 or grid[-1] != 100:
        raise IntegrityError("Curve grid must span 0 to 100 percent")
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise IntegrityError("Curve grid must be strictly increasing")
    denominator = f_input - f_baseline
    if abs(denominator) <= denominator_min:
        return None
    y = [(value - f_baseline) / denominator for value in curve]
    result = sum((b - a) / 100 * (u + v) / 2
                 for a, b, u, v in zip(grid, grid[1:], y, y[1:]))
    if not math.isfinite(result):
        raise IntegrityError("Nonfinite normalized AUC")
    return result


def transform_score(value, original, target, *, dimension=512, epsilon=1e-12):
    """Convert Q=-d^2/(2D) and L2=-sqrt(d^2+eps) on one saved curve."""
    if not math.isfinite(value) or dimension <= 0 or epsilon < 0:
        raise IntegrityError("Invalid score transform input")
    if original not in ("Q", "L2") or target not in ("Q", "L2"):
        raise IntegrityError("Unknown response scale")
    if value > 0:
        raise IntegrityError("Q and negative L2 scores cannot be positive")
    if original == target:
        return value
    if original == "Q":
        return -math.sqrt(-2 * dimension * value + epsilon)
    squared = value * value - epsilon
    # Native FP32 endpoint -sqrt(eps) has tiny roundoff relative to eps.
    if squared < -1e-6 * max(epsilon, 1e-300):
        raise IntegrityError("L2 score is outside its negative-distance domain")
    return -max(squared, 0.0) / (2 * dimension)


def episode_key(record, *, conditional_shared_context=False):
    row = record.row
    if conditional_shared_context:
        return row["task"], row["model"], row["episode"]
    return row["task"], row["model"], row["seed"], row["episode"]


def bootstrap_summary(values, groups, *, statistic="median", draws=10000, seed=0):
    """Independent count-weighted episode bootstrap of the call estimand.

    Each draw selects the observed number of entire episode groups. Sorted
    order statistics implement call medians without flattening resampled rows.
    Point and interval always receive exactly the same values/groups.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise IntegrityError("Result values must be a nonempty finite vector")
    if len(values) != len(groups) or draws < 2:
        raise IntegrityError("Invalid bootstrap population/draw count")
    if statistic not in ("median", "mean"):
        raise IntegrityError("Unknown statistic")
    distinct = sorted(set(groups))
    lookup = {key: i for i, key in enumerate(distinct)}
    codes = np.array([lookup[key] for key in groups])
    point = float(np.median(values) if statistic == "median" else values.mean())
    n_groups = len(distinct)
    if n_groups < 2:
        return dict(point=point, ci_lo=None, ci_hi=None, n_groups=n_groups,
                    n_rows=len(values), interval_status="insufficient_groups")
    rng = np.random.default_rng(seed)
    boots = []
    order = np.argsort(values, kind="stable")
    ordered_values, ordered_codes = values[order], codes[order]
    for start in range(0, draws, 128):
        n = min(128, draws - start)
        sampled = rng.integers(0, n_groups, size=(n, n_groups))
        counts = np.zeros((n, n_groups), dtype=int)
        np.add.at(counts, (np.arange(n)[:, None], sampled), 1)
        weights = counts[:, ordered_codes]
        totals = weights.sum(axis=1)
        if statistic == "median":
            cumulative = weights.cumsum(axis=1)
            low = (totals - 1) // 2
            high = totals // 2
            a = np.argmax(cumulative > low[:, None], axis=1)
            b = np.argmax(cumulative > high[:, None], axis=1)
            boots.extend((ordered_values[a] + ordered_values[b]) / 2)
        else:
            boots.extend((weights @ ordered_values) / totals)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(point=point, ci_lo=float(lo), ci_hi=float(hi),
                n_groups=n_groups, n_rows=len(values),
                interval_status="two_group_descriptive_only" if n_groups == 2
                else "conditional_empirical_episode_percentile")


def random_order_counterexample(grid=(0, 1, 5, 10, 20, 30, 50, 75, 100)):
    """Equal additive features make every ordering random-equivalent.

    For n finite features, f(x)=sum(x_i), x_i=1, baseline=0 and
    self-reference f(x)=n, prefix interventions realize only q=k/n.
    Q normalization gives 1-(1-q)^2 on insertion and 1-q^2 on deletion.
    Their polynomial continuous-fraction extensions integrate to 2/3;
    the all-prefix trapezoid is instead 2/3 - 1/(6*n**2). With n=100,
    every fraction on the default integer-percent grid is realized exactly.
    The returned continuous-integral field describes the extension; the
    grid fields give finite-grid areas. Returned historical data is unchanged.
    """
    x = np.asarray(grid, dtype=float) / 100
    ins = (1 - (1 - x) ** 2).tolist()
    delete = (1 - x ** 2).tolist()
    return dict(
        construction="f(x)=sum_i x_i, x_i=1, baseline=0; every permutation identical",
        grid_percent=list(grid), normalized_insertion=ins, normalized_deletion=delete,
        quadratic_continuous_integral=2 / 3,
        quadratic_grid_insertion=normalized_auc(grid, ins, 1, 0),
        quadratic_grid_deletion=normalized_auc(grid, delete, 1, 0),
        unstabilized_l2_continuous_integral=0.5,
        interpretation="AUC 0.5 is not a universal random-order reference; this is an analytic control, not an empirical RDT null.")
