import copy
import hashlib

import pytest

from tests.provenance_assertions import canonical_audit


def fixture(tmp_path):
    canonical = b'{"value":1}\n{"value":2}\n'
    crlf = canonical.replace(b"\n", b"\r\n")
    name = "data.jsonl"
    entry = dict(path=name, sha256=hashlib.sha256(canonical).hexdigest(),
                 git_crlf_checkout_sha256=hashlib.sha256(crlf).hexdigest())
    audit = dict(files=[dict(path=name, sha256=entry["sha256"],
                            checkout_sha256=entry["sha256"], checkout_form="canonical",
                            valid_records=2)], median=1.5)
    return canonical, crlf, {name: entry}, audit


@pytest.mark.parametrize("form", ["canonical", "declared_git_crlf"])
def test_portable_comparison_authenticates_each_exact_checkout(tmp_path, form):
    canonical, crlf, entries, archived = fixture(tmp_path)
    blob = canonical if form == "canonical" else crlf
    (tmp_path / "data.jsonl").write_bytes(blob)
    fresh = copy.deepcopy(archived)
    fresh["files"][0].update(checkout_form=form, checkout_sha256=hashlib.sha256(blob).hexdigest())
    assert canonical_audit(fresh, tmp_path, entries, current=True) == canonical_audit(
        archived, tmp_path, entries, current=False)
    # Scientific values remain in the comparison.
    fresh["median"] = 7.0
    assert canonical_audit(fresh, tmp_path, entries, current=True) != canonical_audit(
        archived, tmp_path, entries, current=False)


@pytest.mark.parametrize("change", ["stale_digest", "wrong_form", "canonical_digest", "mixed_endings", "rewritten_json"])
def test_checkout_metadata_cannot_hide_changed_input(tmp_path, change):
    canonical, crlf, entries, audit = fixture(tmp_path)
    blob = canonical
    if change == "stale_digest":
        blob = crlf
    elif change == "wrong_form":
        audit["files"][0]["checkout_form"] = "declared_git_crlf"
    elif change == "canonical_digest":
        audit["files"][0]["sha256"] = "0" * 64
    elif change == "mixed_endings":
        blob = canonical.replace(b"\n", b"\r\n", 1)
        audit["files"][0]["checkout_sha256"] = hashlib.sha256(blob).hexdigest()
    else:
        blob = canonical.replace(b'"value":', b'"value": ')
        audit["files"][0]["checkout_sha256"] = hashlib.sha256(blob).hexdigest()
    (tmp_path / "data.jsonl").write_bytes(blob)
    with pytest.raises(AssertionError):
        canonical_audit(audit, tmp_path, entries, current=True)
