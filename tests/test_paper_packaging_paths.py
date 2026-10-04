"""Fail-closed dependency and archive-member regressions on toy manuscripts."""
from pathlib import Path
import zipfile

import pytest

import scripts.package_paper_sources as package


@pytest.mark.parametrize('name', ['../paper/references.bib', './tables/a.tex',
    'tables//a.tex', '/absolute/a.tex', 'C:/absolute/a.tex', r'C:\absolute\a.tex',
    r'tables\a.tex', 'a/../b.tex', ' a.tex', 'a.tex ', 'a\n.tex', ''])
def test_member_path_requires_one_relative_posix_spelling(name):
    with pytest.raises(ValueError, match='normalized relative POSIX'):
        package.member_name(name)


def manuscript(extra=''):
    return '\n'.join([r'\documentclass{article}', r'\usepackage{tmlr}',
        '\\author{\\name Example\n\\addr Example}',
        r'\bibliographystyle{tmlr}', r'\bibliography{references}', extra])


def toy_root(tmp_path, monkeypatch, extra=''):
    root=tmp_path/'repository';paper=root/'paper';paper.mkdir(parents=True)
    for name in package.BASE_FILES:
        if name.startswith('tmlr'):
            (paper/name).write_bytes((Path(__file__).resolve().parents[1]/'paper'/name).read_bytes())
        else:
            (paper/name).write_text('benign toy dependency\n')
    (paper/'paper.tex').write_text(manuscript(extra))
    monkeypatch.setattr(package,'validate_revision_assets',lambda root:None)
    return root


@pytest.mark.parametrize('absolute', [False, True])
def test_in_paper_dependency_alias_cannot_escape_output_tree(tmp_path,monkeypatch,absolute):
    root=toy_root(tmp_path,monkeypatch)
    # The temporary absolute path can contain the workstation username. The
    # path boundary must reject it independently of the identity blacklist.
    monkeypatch.setattr(package,'assert_anonymous',lambda *args: None)
    name=(root/'paper/references.bib').as_posix() if absolute else '../paper/references.bib'
    (root/'paper/paper.tex').write_text(manuscript(r'\input{'+name+'}'))
    before={p.name:p.read_bytes() for p in (root/'paper').iterdir()}
    output=tmp_path/'package'
    with pytest.raises(ValueError,match='normalized relative POSIX'):
        package.build(root,output)
    assert not output.exists()
    assert {p.name:p.read_bytes() for p in (root/'paper').iterdir()}==before


@pytest.mark.parametrize('extra', [r'\input secret.tex',r'\include{secret}',
    r'\includegraphics*{figures_revision/response_geometry.pdf}',
    r'\input{private-notes.tex}',r'\graphicspath{{../private/}}'])
def test_unsupported_or_uncurated_loading_fails_before_output(tmp_path,monkeypatch,extra):
    root=toy_root(tmp_path,monkeypatch,extra)
    (root/'paper/private-notes.tex').write_text('A name outside the identifier blacklist')
    with pytest.raises(ValueError):
        package.build(root,tmp_path/'package')
    assert not (tmp_path/'package').exists()


def test_reviewed_table_name_cannot_hide_nested_file_load(tmp_path,monkeypatch):
    root=toy_root(tmp_path,monkeypatch,r'\input{tables_revision/macros.tex}')
    folder=root/'paper/tables_revision';folder.mkdir()
    (folder/'macros.tex').write_text(r'\input{private-notes.tex}')
    with pytest.raises(ValueError,match='Nested manuscript data directives'):
        package.build(root,tmp_path/'package')
    assert not (tmp_path/'package').exists()


def test_reviewed_appendix_is_included_but_cannot_hide_nested_load(tmp_path, monkeypatch):
    root = toy_root(tmp_path, monkeypatch, r'\input{appendix_aliasing.tex}')
    appendix = root/'paper/appendix_aliasing.tex'
    appendix.write_text(r'\section{Finite sampling} A constructive proof.')
    package.build(root, tmp_path/'included')
    with zipfile.ZipFile(tmp_path/'included/anonymous-manuscript-sources-draft.zip') as archive:
        assert archive.read('appendix_aliasing.tex') == appendix.read_bytes()
    appendix.write_text(r'\input{private-notes.tex}')
    with pytest.raises(ValueError, match='Nested manuscript data directives'):
        package.build(root, tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


@pytest.mark.parametrize('family', ['strengthening_inputs_sha256', 'influence_inputs_sha256', 'manuscript_inputs_sha256', 'reporting_inputs_sha256'])
def test_changed_bound_evidence_blocks_pdf_build(tmp_path, family):
    import hashlib
    import json
    from scripts.build_paper import validate_revision_assets
    paper = tmp_path/'paper'
    (paper/'figures_revision').mkdir(parents=True)
    for name in ('tmlr.sty', 'tmlr.bst', 'tmlr-LICENSE', 'tmlr-source.json'):
        (paper/name).write_bytes((Path(__file__).resolve().parents[1]/'paper'/name).read_bytes())
    (paper/'paper.tex').write_bytes(b'manuscript')
    appendix = paper/'appendix_aliasing.tex'
    appendix.write_bytes(b'reviewed proof')
    registry = dict(generated_sha256={},
                    manuscript_source_sha256=hashlib.sha256(b'manuscript').hexdigest())
    registry[family] = {'paper/appendix_aliasing.tex': hashlib.sha256(b'reviewed proof').hexdigest()}
    (paper/'figures_revision/lineage.json').write_text(json.dumps(registry))
    validate_revision_assets(tmp_path)
    appendix.write_bytes(b'changed proof')
    with pytest.raises(RuntimeError, match='appendix_aliasing.tex'):
        validate_revision_assets(tmp_path)


@pytest.mark.parametrize(('original','replacement'), [
    (r'\bibliography{references}',r'\bibliography{../private/references}'),
    (r'\bibliographystyle{tmlr}',r'\bibliographystyle{other}'),
    (r'\documentclass{article}',r'\documentclass{private-template}'),
    (r'\usepackage{tmlr}',r'\usepackage{../private/style}')])
def test_template_and_bibliography_targets_are_curated(tmp_path,monkeypatch,original,replacement):
    root=toy_root(tmp_path,monkeypatch)
    (root/'paper/paper.tex').write_text(manuscript().replace(original,replacement))
    with pytest.raises(ValueError):
        package.build(root,tmp_path/'package')
    assert not (tmp_path/'package').exists()


def test_toy_package_all_members_are_normalized_and_generated_templates_present(tmp_path,monkeypatch):
    root=toy_root(tmp_path,monkeypatch)
    output=tmp_path/'package'
    package.build(root,output)
    with zipfile.ZipFile(output/'anonymous-manuscript-sources-draft.zip') as archive:
        assert all(package.member_name(name)==name for name in archive.namelist())
        assert {'paper.tex','README.txt','manifest.json'}<=set(archive.namelist())


def test_source_packager_rejects_untrusted_template_before_output(tmp_path, monkeypatch):
    root = toy_root(tmp_path, monkeypatch)
    (root/'paper/tmlr.bst').write_bytes(b'Unreviewed style')
    output = tmp_path/'source-package'
    with pytest.raises(ValueError, match='template byte mismatch'):
        package.build(root, output)
    assert not output.exists()


def test_new_article_dependencies_include_only_reviewed_revision_assets():
    source = '\n'.join([r'\documentclass[12pt,letterpaper]{article}',
        r'\usepackage{fontspec}', r'\setmainfont{Times New Roman}',
        r'\usepackage[margin=1in]{geometry}', r'\usepackage[super,sort&compress]{natbib}',
        r'\bibliographystyle{nhsjs}', r'\bibliography{references}',
        r'\input{tables_revision/revision_facts.tex}', r'\input{tables_revision/paired_results.tex}',
        r'\includegraphics[width=\linewidth]{figures_revision/paired_effects.pdf}'])
    names = package.manuscript_dependencies(source, top_level=True)
    assert {'nhsjs.bst', 'tables_revision/revision_facts.tex', 'tables_revision/paired_results.tex',
            'figures_revision/paired_effects.pdf'} <= names
    with pytest.raises(ValueError, match='Unreviewed manuscript main font'):
        package.manuscript_dependencies(source.replace('Times New Roman', '../private/font'), top_level=True)
    with pytest.raises(ValueError, match='Unreviewed manuscript main font'):
        package.manuscript_dependencies(source.replace(r'\setmainfont{', r'\setmainfont[Path=../private/]{'), top_level=True)


def test_changed_asset_builder_blocks_pdf_build(tmp_path):
    import hashlib
    import json
    from scripts.build_paper import validate_revision_assets
    paper = tmp_path / 'paper'
    (paper / 'figures_revision').mkdir(parents=True)
    for name in ('tmlr.sty', 'tmlr.bst', 'tmlr-LICENSE', 'tmlr-source.json'):
        (paper / name).write_bytes((Path(__file__).resolve().parents[1] / 'paper' / name).read_bytes())
    (paper / 'paper.tex').write_bytes(b'manuscript')
    (tmp_path / 'scripts').mkdir()
    builder = tmp_path / 'scripts/build_revision_assets.py'
    builder.write_bytes(b'reviewed builder')
    registry = dict(generated_sha256={}, manuscript_source_sha256=hashlib.sha256(b'manuscript').hexdigest(),
                    builder_sha256=hashlib.sha256(builder.read_bytes()).hexdigest())
    (paper / 'figures_revision/lineage.json').write_text(json.dumps(registry))
    validate_revision_assets(tmp_path)
    builder.write_bytes(b'changed builder')
    with pytest.raises(RuntimeError, match='build_revision_assets.py'):
        validate_revision_assets(tmp_path)
