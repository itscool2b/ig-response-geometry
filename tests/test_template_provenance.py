"""Exact upstream template identities and package-boundary regressions."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_template_provenance import (
    EXPECTED_PROVENANCE, FILE_SHA256, HISTORICAL_CRLF_SHA256,
    validate_template_bytes, validate_template_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


def template_members():
    return {name: (ROOT/'paper'/name).read_bytes() for name in (*FILE_SHA256, 'tmlr-source.json')}


def test_upstream_bytes_and_historical_crlf_hashes_are_distinct_and_reconstructible():
    members = template_members()
    result = validate_template_provenance(ROOT)
    assert result['files_sha256'] == FILE_SHA256
    assert set(json.loads(members['tmlr-source.json'])['files']) == set(FILE_SHA256)
    for name, expected in FILE_SHA256.items():
        data = members[name]
        assert b'\r' not in data
        assert hashlib.sha256(data).hexdigest() == expected
        assert hashlib.sha256(data.replace(b'\n', b'\r\n')).hexdigest() == HISTORICAL_CRLF_SHA256[name]
        assert expected != HISTORICAL_CRLF_SHA256[name]
    assert b'LaTeX Project Public License' in members['tmlr.bst']
    assert b'Apache License' in members['tmlr-LICENSE']


@pytest.mark.parametrize('name', FILE_SHA256)
@pytest.mark.parametrize('mutation', ['append', 'crlf'])
def test_changed_or_translated_template_bytes_fail(name, mutation):
    members = template_members()
    members[name] = members[name]+b' changed' if mutation == 'append' else members[name].replace(b'\n', b'\r\n')
    with pytest.raises(ValueError, match='template byte mismatch'):
        validate_template_bytes(members)


@pytest.mark.parametrize('change', ['missing_file', 'extra_file', 'changed_pin', 'changed_license', 'changed_history', 'forged_hash'])
def test_manifest_cannot_redefine_the_approved_template(change):
    members = template_members()
    manifest = copy.deepcopy(EXPECTED_PROVENANCE)
    if change == 'missing_file':
        del manifest['files']['tmlr.bst']
    elif change == 'extra_file':
        manifest['files']['unreviewed.sty'] = 'a'*64
    elif change == 'changed_pin':
        manifest['commit'] = 'a'*40
    elif change == 'changed_license':
        del manifest['license_notices']['bibliography_header']
    elif change == 'changed_history':
        manifest['historical_checkout']['accepted_for_build_or_package'] = True
    else:
        members['tmlr.bst'] += b'changed'
        manifest['files']['tmlr.bst'] = hashlib.sha256(members['tmlr.bst']).hexdigest()
    members['tmlr-source.json'] = json.dumps(manifest).encode()
    with pytest.raises(ValueError, match='provenance differs'):
        validate_template_bytes(members)


def test_missing_duplicate_and_ambiguous_provenance_is_rejected():
    members = template_members()
    with pytest.raises(ValueError, match='complete, unambiguous'):
        validate_template_bytes({k:v for k,v in members.items() if k != 'tmlr-source.json'})
    with pytest.raises(ValueError, match='complete, unambiguous'):
        validate_template_bytes({**members, **{'paper/'+k:v for k,v in members.items()}})
    members['tmlr-source.json'] = b'{"files":{},"files":{}}'
    with pytest.raises(ValueError, match='Duplicate TMLR provenance key'):
        validate_template_bytes(members)
