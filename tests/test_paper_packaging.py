from pathlib import Path
import json
import zipfile

import pytest

from scripts.package_paper_sources import anonymous_source, assert_anonymous, build


def test_standalone_math_supplement_has_no_unreviewed_dependencies():
    from scripts.package_paper_sources import manuscript_dependencies
    root = Path(__file__).resolve().parents[1]
    path = root / 'paper/supplementary_derivations.tex'
    if not path.exists():
        pytest.skip('Current mathematical supplement absent from this historical snapshot')
    text = path.read_text(encoding='utf-8')
    assert not manuscript_dependencies(text, top_level=True, supplement=True)
    assert_anonymous(text, path.name)
    with pytest.raises(ValueError, match='Unreviewed manuscript input'):
        manuscript_dependencies(text + '\n\\input{private}', top_level=True, supplement=True)


def test_anonymous_derivative_removes_inactive_identity_and_acknowledgments():
    root = Path(__file__).resolve().parents[1]
    source = (root/'paper/paper.tex').read_text(encoding='utf-8')
    result = anonymous_source(source)
    for value in ('Arjun', 'arjunbajpai', 'itscool2b', 'Acknowledgments', r'\ifanonymous', r'\ifdefined'):
        assert value not in result
    if r'\bibliographystyle{nhsjs}' in source:
        assert any(author in result for author in (r'\author{Anonymous author}', r'\author{}'))
        assert r'\usepackage{fontspec}' in result and r'\usepackage[preprint]{tmlr}' not in result
    else:
        assert r'\author{\name Anonymous authors}' in result
        assert r'\usepackage[preprint]{tmlr}' in result and r'\usepackage{tmlr}' not in result
    assert r'\appendix' in result
    for value in ('working draft', 'remain pending', 'under review as submission', 'paper under double-blind review'):
        assert value not in result.lower()
    assert r'\bibliography{references}' in result


def test_source_package_is_curated_and_does_not_include_private_provenance(tmp_path):
    root = Path(__file__).resolve().parents[1]
    original = (root/'paper/paper.tex').read_bytes()
    first = build(root,tmp_path/'first')
    second = build(root,tmp_path/'second')
    assert first['package_sha256'] == second['package_sha256']
    with zipfile.ZipFile(tmp_path/'first/anonymous-manuscript-sources-draft.zip') as archive:
        names = archive.namelist()
        assert archive.read('paper.tex') == anonymous_source(original.decode('utf-8')).encode('utf-8')
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


def test_authorized_review_package_preserves_every_scientific_member(tmp_path):
    root = Path(__file__).resolve().parents[1]
    build(root, tmp_path/'draft')
    build(root, tmp_path/'review', authorized_review=True)
    with zipfile.ZipFile(tmp_path/'draft/anonymous-manuscript-sources-draft.zip') as before, \
            zipfile.ZipFile(tmp_path/'review/anonymous-manuscript-sources.zip') as after:
        assert before.namelist() == after.namelist()
        for name in before.namelist():
            if name not in {'README.txt', 'manifest.json'}:
                assert before.read(name) == after.read(name), name
        readme = after.read('README.txt').decode()
        assert 'working draft' not in readme and 'submission or receipt' in readme
        manifest = json.loads(after.read('manifest.json'))
        assert manifest['status'] == 'author_authorized_review_copy_not_journal_submission'
        assert 'author_authorized' not in json.loads(before.read('manifest.json'))['status']


def test_comment_only_review_markers_do_not_create_paragraphs_in_captions():
    source = "\\author{Anonymous author}\n\\caption{\n% BEGIN approved-caption\nA 16.0\\% overshoot share.\n% END approved-caption\n}\n"
    result = anonymous_source(source)
    assert result == "\\author{Anonymous author}\n\\caption{\nA 16.0\\% overshoot share.\n}\n"


@pytest.mark.parametrize('path', [r'C:\Users\example\paper.tex', '/home/example/paper.tex', '/workspace/project/paper.tex'])
def test_machine_paths_are_rejected_without_hardcoding_a_private_username(path):
    with pytest.raises(ValueError,match='Local machine path'):
        assert_anonymous(path, 'fixture.tex')


def test_article_anonymization_keeps_only_review_branch_and_no_author_details():
    source = r'''\newif\ifanonymous
\ifdefined\TMLRAnonymous\anonymoustrue\else\anonymousfalse\fi
\ifanonymous
\author{Anonymous author}
\hypersetup{pdfauthor={}}
\else
\author{Arjun Bajpai\\Independent Researcher\\\texttt{arjunbajpai2009@gmail.com}}
\hypersetup{pdfauthor={Arjun Bajpai}}
\fi
Scientific body.
\ifanonymous\else\section*{Acknowledgments} Private mentor.\fi
'''
    result = anonymous_source(source)
    assert r'\author{Anonymous author}' in result and result.count(r'\author{') == 1
    assert 'Scientific body.' in result
    for value in ('Arjun', 'gmail', 'Private mentor', 'Acknowledgments', 'anonymousfalse'):
        assert value not in result
