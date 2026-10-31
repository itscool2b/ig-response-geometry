"""Compare frozen science without pretending checkout provenance is portable.

The archived verifiers bind the exact checkout used for each result. A new run
must verify against its own recorded bytes. Cross-checkout reproduction compares
canonical inputs, while authenticating both allowed historical representations
and every new checkout fingerprint before excluding those two metadata fields.
"""

import hashlib
import json
from pathlib import Path

from analysis.paired_rescoring import curve_stats as cs


def canonical_audit(value, root, entries, *, current):
    """Check, then remove only declared checkout digest/form fields recursively."""
    if isinstance(value, list):
        return [canonical_audit(item, root, entries, current=current) for item in value]
    if not isinstance(value, dict):
        return value
    checkout_fields = set()
    if "checkout_sha256" in value:
        name = value["path"]
        entry = entries[name]
        assert value["sha256"] == entry["sha256"], name
        observed = value["checkout_sha256"]
        allowed = {entry["sha256"]}
        if entry.get("git_crlf_checkout_sha256"):
            allowed.add(entry["git_crlf_checkout_sha256"])
        assert observed in allowed, name
        form = "canonical" if observed == entry["sha256"] else "declared_git_crlf"
        if "checkout_form" in value:
            assert value["checkout_form"] == form, name
        if current:
            blob = (Path(root) / name).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == observed, name
            canonical = blob if form == "canonical" else blob.replace(b"\r\n", b"\n")
            assert hashlib.sha256(canonical).hexdigest() == entry["sha256"], name
        checkout_fields = {"checkout_sha256", "checkout_form"}
    else:
        assert "checkout_form" not in value, "Checkout form lacks its byte digest"
    return {key: canonical_audit(item, root, entries, current=current)
            for key, item in value.items() if key not in checkout_fields}


def assert_same_input_audit(produced, recorded, root):
    manifest = json.loads((Path(root) / cs.MANIFEST).read_bytes())
    entries = {entry["path"]: entry for entry in manifest["files"]}
    assert canonical_audit(produced, root, entries, current=True) == canonical_audit(
        recorded, root, entries, current=False)


def assert_artifacts_authentic(directory, provenance):
    """Authenticate exact artifact bytes before comparing canonical audit values."""
    artifacts = dict(provenance["artifacts"])
    artifacts.update(provenance.get("platform_dependent_artifacts", {}))
    for name, record in artifacts.items():
        assert cs.artifact_record(Path(directory) / name) == record, name


def canonical_git_text_record(record, root, reference_bytes, *, current):
    """Authenticate a text-file fingerprint against a pinned Git blob.

    Some older inventories record an actual-byte digest in ``sha256`` without
    separate canonical and checkout fields. Permit exactly the pinned LF blob
    or its complete CRLF checkout, then retain every other metadata field.
    """
    assert b"\r" not in reference_bytes, "Pinned Git reference must be canonical LF"
    canonical_digest = hashlib.sha256(reference_bytes).hexdigest()
    allowed = {canonical_digest,
               hashlib.sha256(reference_bytes.replace(b"\n", b"\r\n")).hexdigest()}
    assert record["sha256"] in allowed, record["path"]
    if current:
        blob = (Path(root) / record["path"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == record["sha256"], record["path"]
        assert blob.replace(b"\r\n", b"\n") == reference_bytes, record["path"]
    return {**record, "sha256": canonical_digest}
