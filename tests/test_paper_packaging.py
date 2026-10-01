from pathlib import Path
import json
import zipfile

import pytest

from scripts.package_paper_sources import anonymous_source, build


def test_anonymous_derivative_removes_inactive_identity_and_acknowledgments():
    root = Path(__file__).resolve().parents[1]
    result = anonymous_source((root/'paper/paper.tex').read_text(encoding='utf-8'))
    for value in ('Arjun', 'arjunbajpai', 'itscool2b', 'Acknowledgments', r'\ifanonymous', r'\ifdefined'):
        assert value not in result
    assert r'\author{\name Anonymous authors}' in result
    assert r'\usepackage{tmlr}' in result and r'\usepackage[preprint]' not in result
    assert 'AI tools assisted' in result
    assert r'\bibliography{references}' in result


def test_source_package_is_curated_and_does_not_include_private_provenance(tmp_path):
    root = Path(__file__).resolve().parents[1]
    original = (root/'paper/paper.tex').read_bytes()
    first = build(root,tmp_path/'first')
    second = build(root,tmp_path/'second')
    assert first['package_sha256'] == second['package_sha256']
    with zipfile.ZipFile(tmp_path/'first/anonymous-manuscript-sources-draft.zip') as archive:
        names = archive.namelist()
        assert 'private-provenance.json' not in names
        assert 'tmlr-LICENSE' in names and 'tmlr-source.json' in names
        assert not any('legacy' in n or 'lineage' in n or 'paper.pdf' in n for n in names)
        manifest = json.loads(archive.read('manifest.json'))
        assert set(manifest['artifacts_sha256']) == set(names)-{'manifest.json'}
    assert (root/'paper/paper.tex').read_bytes() == original
    with pytest.raises(FileExistsError):
        build(root,tmp_path/'first')


def test_changed_author_or_malformed_conditionals_fail_closed():
    with pytest.raises(ValueError,match='conditional'):
        anonymous_source(r'\ifanonymous unclosed')
    with pytest.raises(ValueError,match='author declaration'):
        anonymous_source(r'\author{Different syntax}')
