"""Fail-closed dependency and archive-member regressions on toy manuscripts."""
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
