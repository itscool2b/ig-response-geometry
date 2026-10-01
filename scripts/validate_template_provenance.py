"""Validate the exact pinned TMLR bytes without network access or rewriting files.

The constants bind the separately checked upstream Git objects, not whichever
hashes happen to be listed in an editable manifest. Historical CRLF checkout
hashes are retained as provenance but never accepted as package identities.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path


SOURCE = "https://github.com/JmlrOrg/tmlr-style-file"
COMMIT = "7bf90efe3a0debbba703c05c43f3ff7e4d4a2992"
FILE_SHA256 = {
    "tmlr.sty": "816214ff5919aa457b6b443bee52b15d9561421417b7f8a50cc84651519f0002",
    "tmlr.bst": "306fd454cf40771bee01293eeb98d2c1cd5f4e11ed0cd7296b335f354fc45206",
    "tmlr-LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
}
HISTORICAL_CRLF_SHA256 = {
    "tmlr.sty": "a067e8286a3cf8e3ad292e3d8bc5feb4690e62bc100fa6d903a5e22f947ea6cb",
    "tmlr.bst": "ded3dbf610d9db328c4c1bc07fc24957c034b722de9a663c0350912f044676c8",
    "tmlr-LICENSE": "1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6",
}
MANIFEST_NAME = "tmlr-source.json"
EXPECTED_PROVENANCE = {
    "schema_version": 2,
    "source": SOURCE,
    "commit": COMMIT,
    "retrieved": "2026-09-30",
    "upstream_bytes_verified": "2026-10-01",
    "files": FILE_SHA256,
    "representation": "Exact pinned upstream Git-blob bytes with LF line endings.",
    "historical_checkout": {
        "representation": "CRLF checkout conversion of the same upstream bytes; these were the hashes recorded in the original manifest.",
        "files": HISTORICAL_CRLF_SHA256,
        "accepted_for_build_or_package": False,
    },
    "license_notices": {
        "upstream_repository": {"license": "Apache-2.0", "file": "tmlr-LICENSE", "upstream_path": "LICENSE"},
        "bibliography_header": {"license": "LPPL-1.0-or-later", "file": "tmlr.bst", "notice_lines": "15-18"},
        "interpretation": "The upstream repository license and the bibliography file's retained license notice are both preserved. No claim is made that one overrides the other.",
    },
    "modifications": "No changes to upstream style, bibliography style or license bytes. System fancyhdr is used.",
}


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate TMLR provenance key: " + name)
        result[name] = value
    return result


def validate_template_bytes(contents: Mapping[str, bytes]) -> dict:
    """Validate actual members, accepting either bare or paper-prefixed names.

    Other package members are ignored. All three fixed template files and the
    full reviewed provenance document are mandatory. No newline normalization
    or manifest repair is performed here.
    """
    names = {*FILE_SHA256, MANIFEST_NAME}
    bare = names <= contents.keys()
    prefixed = {"paper/" + name for name in names} <= contents.keys()
    if bare == prefixed:
        raise ValueError("TMLR provenance requires one complete, unambiguous template member set")
    prefix = "" if bare else "paper/"
    try:
        manifest = json.loads(contents[prefix + MANIFEST_NAME].decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid TMLR provenance document") from exc
    if manifest != EXPECTED_PROVENANCE:
        raise ValueError("TMLR provenance differs from the reviewed pinned source, exact file set, or license notices")
    actual = {name: hashlib.sha256(contents[prefix + name]).hexdigest() for name in FILE_SHA256}
    mismatches = [name for name in FILE_SHA256 if actual[name] != FILE_SHA256[name]]
    if mismatches:
        raise ValueError("TMLR template byte mismatch: " + ", ".join(mismatches))
    return {"source": SOURCE, "commit": COMMIT, "files_sha256": actual,
            "manifest_sha256": hashlib.sha256(contents[prefix + MANIFEST_NAME]).hexdigest()}


def validate_template_provenance(root: Path) -> dict:
    """Check a repository or extracted research-supplement tree before build."""
    paper = Path(root) / "paper"
    names = {*FILE_SHA256, MANIFEST_NAME}
    missing = sorted(name for name in names if not (paper / name).is_file())
    if missing:
        raise ValueError("Missing TMLR provenance files: " + ", ".join(missing))
    return validate_template_bytes({name: (paper / name).read_bytes() for name in names})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(validate_template_provenance(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
