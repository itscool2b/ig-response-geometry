"""Private review capabilities must never enter default or identified outputs."""
import hashlib
import json

import pytest

from scripts.review_route import MARKER, load_review_url, render_review_route, review_binding
from scripts.package_paper_sources import anonymous_source, selected_source, build


def test_private_route_requires_explicit_anonymous_build(tmp_path):
    path = tmp_path / 'private.json'
    url = 'https://files.example.test/unlisted-review/index.html'
    path.write_text(json.dumps({'review_archive_url': url}))
    review = load_review_url(path)
    source = r'Available in the \href{' + MARKER + r'}{anonymous research archive}.'
    assert render_review_route(source, review, anonymous=True) == source.replace(MARKER, url)
    for setting in (None, review):
        public = render_review_route(source, setting, anonymous=False)
        assert public == 'Available in the anonymous research archive.'
        assert url not in public and MARKER not in public
    assert render_review_route(source, anonymous=True) == 'Available in the anonymous research archive.'
    assert review_binding(review) == {
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'url_sha256': hashlib.sha256(url.encode()).hexdigest()}
    assert url not in json.dumps(review_binding(review))


def test_anonymous_source_cannot_retain_identified_branch(tmp_path):
    path = tmp_path/'url.txt'
    path.write_text('https://files.example.test/private/index.html\n')
    source = (r'\author{Anonymous author}' '\n' r'\ifanonymous' '\n'
              r'\href{' + MARKER + r'}{research files}' '\n' r'\else' '\n'
              r'\href{https://github.com/itscool2b/identified}{public repository}' '\n' r'\fi')
    result = anonymous_source(source, review_route=load_review_url(path))
    assert 'private/index.html' in result and 'itscool2b' not in result and MARKER not in result
    assert 'https://' not in anonymous_source(source)


@pytest.mark.parametrize('url', ['http://example.test/a', 'https://name:pass@example.test/a',
    'https://example.test/a?secret=x', 'https://example.test/a#x',
    'https://example.test/a}\\input{bad}', MARKER])
def test_route_file_rejects_unreviewed_url_syntax(tmp_path, url):
    path=tmp_path/'url.txt';path.write_text(url)
    with pytest.raises(ValueError):load_review_url(path)


def test_markers_cannot_silently_duplicate_or_escape_href():
    source = r'\href{' + MARKER + r'}{files}'
    with pytest.raises(ValueError):render_review_route(source+source, anonymous=True)
    with pytest.raises(ValueError):render_review_route(MARKER, anonymous=True)
    assert render_review_route('historical source', anonymous=True) == 'historical source'


def test_identified_source_selects_only_public_branches():
    text = (r'\newif\ifanonymous' '\n' r'\ifanonymous' '\n'
            r'\author{Anonymous author}' '\n' r'\href{' + MARKER + r'}{files}' '\n'
            r'\else' '\n' r'\author{Named author}' '\n' 'stable public location' '\n' r'\fi')
    result = selected_source(text, anonymous=False)
    assert 'Named author' in result and 'stable public location' in result
    assert MARKER not in result and 'Anonymous author' not in result


def test_identified_package_refuses_private_route_before_any_io(tmp_path):
    with pytest.raises(ValueError, match='Identified packages'):
        build(tmp_path/'repo', tmp_path/'out', identified=True, review_url_file=tmp_path/'private.json')


def test_public_research_release_refuses_private_options_before_any_io(tmp_path):
    from scripts.package_research_supplement import build as research_build
    for options in ({'review_url_file':tmp_path/'private.json'}, {'authorized_anonymous_review':True}):
        with pytest.raises(ValueError, match='Public releases'):
            research_build(tmp_path/'repo','HEAD',tmp_path/'out',public_release=True,**options)
