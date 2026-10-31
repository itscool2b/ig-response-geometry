"""Read-only, fail-closed comparison of a complete registered analysis replay.

SHA-256 identity and numerical equivalence are separate results. The only
permitted numerical differences are binary64 map L1/ratio/signed reductions,
summaries independently reconstructed from those rows, and explicitly certified
optional BCa/t sensitivity endpoints. All other floats,
counts, labels, missing values, method choices and intervals must be exact.
No frozen result, input, or generated result is changed by this verifier.

The caller authenticates the complete archive and executes its preserved code.
This verifier additionally authenticates the results, relevant source files,
and map bytes used to establish the operation-specific error bounds. It does
not prove that an arbitrary directory was actually produced by executing code.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import types
import zipfile


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\Z")
INTEGER = re.compile(r"[-+]?\d+\Z")
COUNT = re.compile(r"(^n$|^n_|^num_|^nodes|^count|^over_|_count$|_size$|^episodes$|^contexts$|^call$|^episode$|^seed$|^draws$)")
SOURCES = ("ig_quadrature.py", "analysis/revision_gpu/v1/analyze.py",
           "analysis/revision_gpu/v1/numerics.py", "analysis/revision_gpu/v1/methods.py",
           "analysis/revision_gpu/v1/coverage.py")
THRESHOLDS = (-1, 0.001, 0.01, 0.05, 0.4, 0.5, 0.9, 0.95, 0.99, 1)
UNIT_ROUNDOFF = 2.0 ** -53


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def _root(path):
    lexical = Path(path).absolute()
    if ".." in lexical.parts:
        raise ValueError("parent traversal in root")
    for entry in (lexical, *lexical.parents):
        if entry.is_symlink() or (hasattr(entry, "is_junction") and entry.is_junction()):
            raise ValueError("aliased root or ancestor")
    return lexical.resolve()


def _path(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", name):
        raise ValueError("unsafe member name")
    parts = PurePosixPath(name).parts
    if any(p in (".", "..") for p in parts):
        raise ValueError("unsafe member name")
    target = Path(root)
    for part in parts:
        target = target / part
        if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
            raise ValueError("aliased member")
    if not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("member escapes root")
    return target


def _read(path):
    before = path.stat()
    blob = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(blob) != after.st_size:
        raise ValueError("file changed while being read")
    return blob


def _json(blob):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("nonfinite JSON number")
    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("nonfinite or overflowed JSON number")
        return value
    return json.loads(blob, object_pairs_hook=unique, parse_constant=invalid, parse_float=finite)


def _gamma(n):
    if not 0 <= n * UNIT_ROUNDOFF < 0.01:
        raise ValueError("unsupported reduction size")
    return n * UNIT_ROUNDOFF / (1 - n * UNIT_ROUNDOFF)


def _ordered(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return (~bits & ((1 << 64) - 1)) if bits >> 63 else bits | (1 << 63)


def _archive_index(root):
    result = {}
    for line in _read(_path(root, "MANIFEST.sha256")).decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError("invalid archive manifest record")
        digest, name = match.groups()
        _path(root, name)
        if name in result or name == "MANIFEST.sha256":
            raise ValueError("duplicate or self-referential archive member")
        result[name] = digest
    return result


def _authenticate(root, index, path):
    name = path.relative_to(root).as_posix()
    if name not in index or _sha(_path(root, name)) != index[name]:
        raise ValueError("archive member hash mismatch: " + name)


def _metadata(archive, index, code):
    import torch

    configuration = archive / "reproduction.json"
    _authenticate(archive, index, configuration)
    code_zip = _path(archive, _json(_read(configuration))["code"])
    _authenticate(archive, index, code_zip)
    source_hashes = {}
    with zipfile.ZipFile(code_zip) as source:
        if len(source.namelist()) != len(set(source.namelist())):
            raise ValueError("duplicate source ZIP member")
        members = _json(source.read("SUPPLEMENT_MANIFEST.json"))["members"]
        for name in SOURCES:
            blob = source.read(name)
            digest = hashlib.sha256(blob).hexdigest()
            if digest != members[name]["sha256"] or len(blob) != members[name]["bytes"]:
                raise ValueError("source ZIP member hash mismatch: " + name)
            if _sha(_path(code, name)) != digest:
                raise ValueError("executed source hash mismatch: " + name)
            source_hashes[name] = digest
    metadata = dict(files=0, tensors=0, maximum_coordinates=0, dtypes={}, source_sha256=source_hashes,
                    files_sha256={}, shapes={})
    for path in sorted((archive / "runs").glob("*/*/*/maps.pt")):
        if path.parent.name not in ("bf16", "fp32"):
            continue
        name = path.relative_to(archive).as_posix()
        before = path.stat()
        _authenticate(archive, index, path)
        maps = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(maps, dict):
            raise ValueError("map payload is not a tensor dictionary")
        for tensor in maps.values():
            if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("map dtype is outside the exact binary64-cast bound")
            if not torch.isfinite(tensor).all().item():
                raise ValueError("nonfinite map tensor invalidates finite reduction bound")
            metadata["tensors"] += 1
            metadata["maximum_coordinates"] = max(metadata["maximum_coordinates"], tensor.numel())
            dtype, shape = str(tensor.dtype), str(list(tensor.shape))
            metadata["dtypes"][dtype] = metadata["dtypes"].get(dtype, 0) + 1
            metadata["shapes"][shape] = metadata["shapes"].get(shape, 0) + 1
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("map changed during metadata inspection")
        metadata["files"] += 1
        metadata["files_sha256"][name] = index[name]
    if not metadata["maximum_coordinates"]:
        raise ValueError("no authenticated maps establish a reduction bound")
    return metadata


def _csv(blob):
    rows = list(csv.reader(io.StringIO(blob.decode("utf-8"))))
    if not rows or len(rows[0]) != len(set(rows[0])):
        raise ValueError("missing or duplicate CSV headers")
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError("CSV row width mismatch")
    return rows[0], [dict(zip(rows[0], row)) for row in rows[1:]]


def _reference_bound(field, left, right, n):
    if left.get("source") != "recomputed" or right.get("source") != "recomputed":
        return None
    gamma = _gamma(n + 3)
    a, b = float(left[field]), float(right[field])
    if field == "l1":
        return 2 * gamma / (1 - gamma) * max(abs(a), abs(b))
    if field == "relative_l1":
        return 4 * gamma / (1 - gamma) ** 2 * max(abs(a), abs(b))
    if field == "signed_sum_difference":
        # ||candidate||_1 <= ||reference||_1 + ||candidate-reference||_1.
        # Inflate the norm recovered from the rounded L1/relative-L1 ratio.
        scales = []
        for row in (left, right):
            l1 = float(row["l1"])
            ratio = float(row["relative_l1"]) if row.get("relative_l1") else 0.
            if l1 < 0 or ratio < 0:
                return None
            if ratio:
                norm = l1 / ratio / (1 - gamma) ** 3
                scales.append(2 * norm + l1 / (1 - gamma))
            elif not row.get("relative_l1"):
                # Missing relative L1 means the reference norm is exactly zero.
                scales.append(l1 / (1 - gamma))
            elif l1 == 0 and a == b == 0:
                scales.append(0.)
            else:
                return None
        return 2 * gamma / (1 - gamma) * max(scales)
    return None


def _summary_values(blobs, unit):
    groups = defaultdict(list)
    keys = [unit, "modality", "target", "family", "reference", "candidate"]
    for name in ("numerics/comparisons.csv", "numerics/references.csv"):
        _, rows = _csv(blobs[name])
        for row in rows:
            text = row.get("relative_l1", "")
            if text and NUMBER.fullmatch(text) and math.isfinite(float(text)):
                value = float(text)
                if value < 0:
                    raise ValueError("relative L1 cannot be negative")
                # The producer's worker comparison rows omit this optional
                # key; csv.DictWriter represents its summary value as blank.
                groups[tuple(row.get(k, "") if k == "reference" else row[k] for k in keys)].append(value)
    result = {}
    for key, values in groups.items():
        ordered = sorted(values)
        n = len(values)
        middle = ordered[n // 2] if n % 2 else math.fsum(ordered[n // 2 - 1:n // 2 + 1]) / 2
        mean = math.fsum(values) / n
        result[key] = dict(n=n, over_0_01=sum(x > .01 for x in values), median=middle,
                           mean=mean, min=ordered[0], max=ordered[-1],
                           mean_bound=_gamma(n + 2) / (1 - _gamma(n + 2)) * mean,
                           median_bound=_gamma(3) / (1 - _gamma(3)) * middle)
    return keys, result


class _Compare:
    def __init__(self, report, name, maximum_coordinates, summaries, intervals=None):
        self.report, self.name = report, name
        self.maximum_coordinates, self.summaries = maximum_coordinates, summaries
        self.record = report["files"][name]
        self.intervals = intervals or {}

    def fail(self, location, reason, expected=None, actual=None):
        self.report["failures"].append(dict(file=self.name, location=location, reason=reason,
                                             expected=expected, actual=actual))

    def count(self, key):
        self.report["checks"][key] += 1

    def numeric(self, a, b, location, bound=None):
        self.count("numeric_values")
        if not math.isfinite(a) or not math.isfinite(b):
            self.fail(location, "nonfinite numeric value", str(a), str(b))
            return
        if a == b:
            if a == 0 and math.copysign(1, a) != math.copysign(1, b):
                self.fail(location, "signed zero changed", a, b)
            return
        error = abs(a - b)
        accepted = bound is not None and math.isfinite(bound) and error <= bound
        interval = self.intervals.get(location)
        if interval is not None:
            from decimal import Decimal
            lo, hi = map(Decimal, interval["enclosure"])
            accepted = lo <= Decimal(a) <= hi and lo <= Decimal(b) <= hi
            self.count("certified_optional_interval_endpoints")
        item = dict(location=location, expected=a, actual=b, absolute_error=error,
                    symmetric_relative_error=error / max(abs(a), abs(b)),
                    ulp_distance=abs(_ordered(a) - _ordered(b)), accepted=accepted,
                    allowed_absolute_error=bound)
        self.record["numeric_differences"].append(item)
        if interval is not None:
            item["independent_operation_enclosure"] = interval["enclosure"]
        if not accepted:
            self.fail(location, "difference exceeds derived bound" if bound is not None else
                      "no approved numerical equivalence rule", a, b)
        if (a > 0) - (a < 0) != (b > 0) - (b < 0):
            self.fail(location, "numeric sign changed", a, b)
        for threshold in THRESHOLDS:
            if (a <= threshold) != (b <= threshold):
                self.fail(location, "diagnostic threshold decision changed", a, b)
        for fmt in (".3g", ".4f", ".6g"):
            if format(a, fmt) != format(b, fmt):
                self.fail(location, "scientific display rounding changed: " + fmt, a, b)

    def value(self, a, b, location):
        if type(a) is not type(b):
            self.fail(location, "JSON type changed", type(a).__name__, type(b).__name__)
        elif isinstance(a, dict):
            self.count("object_key_sets")
            if a.keys() != b.keys():
                self.fail(location, "object keys changed", sorted(a), sorted(b))
            else:
                for key in a:
                    self.value(a[key], b[key], location + "/" + key)
        elif isinstance(a, list):
            self.count("array_lengths")
            if len(a) != len(b):
                self.fail(location, "array length changed", len(a), len(b))
            else:
                if location.endswith("/interval") and len(a) == 2:
                    self.count("interval_interpretations")
                    if all(type(v) in (int, float) and math.isfinite(v) for v in a + b):
                        state = lambda x: (x[0] > 0, x[1] < 0, x[0] <= 0 <= x[1], x[0] <= x[1])
                        if state(a) != state(b):
                            self.fail(location, "interval interpretation changed", a, b)
                for index, (x, y) in enumerate(zip(a, b)):
                    self.value(x, y, location + "/" + str(index))
        elif isinstance(a, float):
            self.numeric(a, b, location)
        else:
            self.count("boolean_values" if isinstance(a, bool) else "integer_values" if isinstance(a, int) else "categorical_values")
            if location.rsplit("/", 1)[-1] in ("best", "preferred", "second", "tie", "unique", "all_fixed_below", "across_responses", "differs"):
                self.count("method_preference_tie_calibration_fields")
            if a != b:
                self.fail(location, "categorical, integer, Boolean or missing value changed", a, b)

    def csv(self, left, right):
        lh, rows_a = _csv(left)
        rh, rows_b = _csv(right)
        if lh != rh or len(rows_a) != len(rows_b):
            self.fail("", "CSV header or row count changed", [lh, len(rows_a)], [rh, len(rows_b)])
            return
        for index, (a, b) in enumerate(zip(rows_a, rows_b), 2):
            for key in lh:
                x, y = a[key], b[key]
                location = f"row{index}/{key}"
                # Registered undefined quantities use empty fields or JSON
                # null. No textual nonfinite markers occur in this archive.
                markers = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
                if x.strip().lower() in markers or y.strip().lower() in markers:
                    self.fail(location, "unregistered nonfinite CSV marker", x, y)
                    continue
                if NUMBER.fullmatch(x) and NUMBER.fullmatch(y):
                    if COUNT.search(key) or INTEGER.fullmatch(x) and INTEGER.fullmatch(y):
                        self.count("integer_values")
                        if x != y:
                            self.fail(location, "integer/count changed", x, y)
                    else:
                        bound = None
                        if self.name == "numerics/references.csv" and key in ("l1", "relative_l1", "signed_sum_difference"):
                            bound = _reference_bound(key, a, b, self.maximum_coordinates)
                            self.count("map_reduction_bound_checks")
                        if self.name in self.summaries and key in ("mean", "median", "max", "min"):
                            bound = self.summaries[self.name].get((index, key))
                        self.numeric(float(x), float(y), location, bound)
                        if x != y and float(x) == float(y):
                            self.fail(location, "numeric representation changed without an arithmetic difference", x, y)
                elif x.startswith(("[", "{")) and y.startswith(("[", "{")):
                    self.value(_json(x), _json(y), location)
                else:
                    self.count("categorical_values")
                    if x != y:
                        self.fail(location, "categorical or missing value changed", x, y)


def _interval_helper(archive, index, supplied=None):
    """Compile authenticated immutable source bytes, never import from sys.path."""
    path = _path(archive, "reviewer_interval_certificate.py")
    blob = _read(path)
    if hashlib.sha256(blob).hexdigest() != index.get("reviewer_interval_certificate.py"):
        raise ValueError("interval helper source hash mismatch")
    if supplied is not None:
        supplied_path, supplied_bytes = supplied
        if _root(supplied_path) != path or supplied_bytes != blob:
            raise ValueError("supplied interval helper differs from authenticated archive")
    module = types.ModuleType("authenticated_reviewer_interval_certificate")
    module.__file__ = str(path)
    exec(compile(blob, str(path), "exec"), module.__dict__)
    if _read(path) != blob:
        raise ValueError("interval helper changed during import")
    return module, path, blob


def verify(expected_results, actual_results, archive_root, code_root, *, interval_source=None):
    """Return an anonymous JSON-compatible audit; failures never rewrite evidence."""
    supplied_roots = (expected_results, actual_results, archive_root, code_root)
    report = dict(schema_version=1, verified=False, byte_identity=False, complete=False,
                  policy="Exact except authenticated map reductions, reconstructed summaries, and independently enclosed optional BCa/t endpoints under explicit checked post hoc accuracy contracts:16u normal CDF,8u inverse CDFs/powers. Primary estimates, percentile intervals and decisions remain exact.",
                  comparator_sha256=_sha(Path(__file__)), files={}, failures=[], checks=Counter())
    try:
        expected, actual, archive, code = map(_root, supplied_roots)
        if expected == actual or not expected.is_relative_to(archive):
            raise ValueError("independent output and contained expected results are required")
        index = _archive_index(archive)
        _authenticate(archive, index, expected / "manifest.json")
        old, new = _json(_read(_path(expected, "manifest.json"))), _json(_read(_path(actual, "manifest.json")))
        if old["kind"] != new["kind"] or old["kind"] != "nhsjs_r1_gpu_analysis":
            raise ValueError("full registered analysis kind is required")
        if old["population"] != new["population"] or old["plan_sha256"] != new["plan_sha256"] or \
                sorted(old["coverage_sha256"].values()) != sorted(new["coverage_sha256"].values()):
            raise ValueError("population, plan or coverage binding changed")
        if not old["outputs"] or old["outputs"].keys() != new["outputs"].keys():
            raise ValueError("registered output inventory changed")
        actual_names = {p.relative_to(actual).as_posix() for p in actual.rglob("*") if p.is_file()}
        if actual_names != set(old["outputs"]) | {"manifest.json"}:
            raise ValueError("missing or unexpected generated file")
        report["manifest_sha256"] = dict(expected=_sha(expected / "manifest.json"), actual=_sha(actual / "manifest.json"))
        report["map_metadata"] = _metadata(archive, index, code)
        report["bounds"] = dict(unit_roundoff=UNIT_ROUNDOFF,
            maximum_coordinates=report["map_metadata"]["maximum_coordinates"],
            gamma="gamma_k=k*u/(1-k*u), k=max_coordinates+3",
            l1="2*gamma/(1-gamma)*max(abs(expected),abs(actual))",
            relative_l1="4*gamma/(1-gamma)^2*max(abs(expected),abs(actual))",
            signed_sum="2*gamma/(1-gamma)*(2*reference_L1+map_difference_L1), with recovered norm inflated for ratio rounding",
            summary="Each side must match independently reconstructed count, threshold, min/max, median and fsum mean within its own reduction bound.")
        blobs = [{}, {}]
        for name in sorted(old["outputs"]):
            a, b = _path(expected, name), _path(actual, name)
            _authenticate(archive, index, a)
            left, right = _read(a), _read(b)
            ha, hb = hashlib.sha256(left).hexdigest(), hashlib.sha256(right).hexdigest()
            if ha != old["outputs"][name] or hb != new["outputs"][name]:
                raise ValueError("generated or frozen output manifest hash mismatch: " + name)
            blobs[0][name], blobs[1][name] = left, right
            report["files"][name] = dict(expected_sha256=ha, actual_sha256=hb, byte_identity=left == right,
                                          numeric_differences=[])
        summaries = {}
        for name, unit in (("numerics/comparison_summary.csv", "group"), ("numerics/comparison_summary_by_bank.csv", "bank")):
            if name not in old["outputs"]:
                continue
            derived = [_summary_values(side, unit) for side in blobs]
            summary_rows = [_csv(side[name])[1] for side in blobs]
            summaries[name] = {}
            for side, (keys, groups) in enumerate(derived):
                rows = summary_rows[side]
                if len(rows) != len(groups) or {tuple(row[k] for k in keys) for row in rows} != set(groups):
                    raise ValueError("summary group inventory differs from underlying rows")
                for row in rows:
                    calculated = groups[tuple(row[k] for k in keys)]
                    for key in ("n", "over_0_01", "min", "max", "median", "mean"):
                        value = float(row[key])
                        bound = calculated.get(key + "_bound", 0.)
                        report["checks"]["summary_source_consistency_checks"] += 1
                        if abs(value - calculated[key]) > bound:
                            report["failures"].append(dict(file=name, location=key, reason="summary exceeds its own input reduction bound", side="expected" if side == 0 else "actual"))
            for row_index, row in enumerate(summary_rows[0], 2):
                group = tuple(row[k] for k in derived[0][0])
                a, b = derived[0][1][group], derived[1][1].get(group)
                if b is not None:
                    for key in ("min", "max", "median", "mean"):
                        summaries[name][row_index, key] = abs(a[key] - b[key]) + a.get(key + "_bound", 0.) + b.get(key + "_bound", 0.)
        certificates = None
        helper = None
        source_bindings = {}
        def read_bound(name):
            path = _path(archive, name)
            blob = _read(path)
            digest = hashlib.sha256(blob).hexdigest()
            if index.get(name) != digest:
                raise ValueError("certificate input source hash mismatch: " + name)
            source_bindings[name] = digest
            return blob
        for name, record in report["files"].items():
            a, b = blobs[0][name], blobs[1][name]
            intervals = {}
            if a != b and re.fullmatch(r"methods/(P|B1|S)/(decisions(?:_round_rule|_nominal_axis)?|contrasts)\.json", name):
                if certificates is None:
                    module, helper_path, helper_bytes = _interval_helper(archive, index, interval_source)
                    helper = helper_path, helper_bytes
                    certificates = module.Certificates(read_bound, _json, old)
                    report["interval_helper_sha256"] = hashlib.sha256(helper_bytes).hexdigest()
                intervals = certificates.for_file(name, _json(a), _json(b))
            compare = _Compare(report, name, report["map_metadata"]["maximum_coordinates"], summaries, intervals)
            if name.endswith(".csv"):
                compare.csv(a, b)
            elif name.endswith(".json"):
                compare.value(_json(a), _json(b), "")
            elif a != b:
                compare.fail("", "unknown changed output category")
            if a != b and not record["numeric_differences"] and not any(f.get("file") == name for f in report["failures"]):
                compare.fail("", "unexplained byte representation difference")
        if certificates is not None:
            report["optional_interval_certificates"] = certificates.records
            report["interval_source_bindings"] = source_bindings
            for name, digest in source_bindings.items():
                if hashlib.sha256(_read(_path(archive, name))).hexdigest() != digest:
                    raise ValueError("certificate input changed during verification")
            if _read(helper[0]) != helper[1]:
                raise ValueError("interval helper changed during verification")
        report["complete"] = True
        report["byte_identity"] = all(record["byte_identity"] for record in report["files"].values())
        report["verified"] = not report["failures"]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile) as error:
        # Never emit an absolute workspace path in anonymous review evidence.
        reason = str(error)
        for root in supplied_roots:
            reason = reason.replace(str(Path(root)), "<root>")
        report["failures"].append(dict(reason=reason))
    report["checks"] = dict(report["checks"])
    report["outputs_compared"] = len(report["files"])
    report["files_byte_identical"] = sum(item["byte_identity"] for item in report["files"].values())
    report["files_with_hash_differences"] = len(report["files"]) - report["files_byte_identical"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("expected-results", "actual-results", "archive-root", "code-root", "report"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.expected_results, args.actual_results, args.archive_root, args.code_root)
    with args.report.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: result[key] for key in ("verified", "byte_identity", "complete", "outputs_compared", "files_with_hash_differences")}))
    raise SystemExit(0 if result["verified"] else 1)
