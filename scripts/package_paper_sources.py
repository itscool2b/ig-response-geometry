"""Create a curated anonymous manuscript-source component, never a submission.

The identified canonical source remains unchanged. This component contains only
referenced manuscript assets and third-party style/license files. It does not
claim that the scientific, code-supplement or author-specific submission gates
have passed. Its identifying provenance ledger stays outside the review ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_paper import validate_revision_assets

IDENTIFIERS = ('arjun bajpai', 'arjunbajpai2009', 'itscool2b',
               'the-readout-not-the-denoiser-repo')

BASE_FILES = frozenset({'references.bib', 'tmlr.sty', 'tmlr.bst', 'tmlr-LICENSE', 'tmlr-source.json'})
TABLE_FILES = frozenset('tables_revision/' + name + '.tex' for name in
                        ('baseline', 'budget', 'completeness', 'faithfulness', 'macros',
                         'oneb', 'rescore', 'sanity', 'variants', 'numerical_roster', 'numerical_diagnostics'))
FIGURE_FILES = frozenset({'figures_revision/response_geometry.pdf',
                          'figures_revision/solver_endpoint.pdf'})
TEX_PACKAGES = frozenset({'tmlr', 'url', 'hyperref', 'inputenc', 'caption', 'graphicx',
                          'placeins', 'amsmath', 'amssymb', 'amsthm', 'booktabs',
                          'algorithm', 'algorithmic', 'microtype', 'xcolor'})
DATA_DIRECTIVES = ('input', 'includegraphics', 'bibliography', 'bibliographystyle',
                   'documentclass', 'usepackage')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def anonymous_source(text):
    # No verbatim/literal-percent environment occurs in this manuscript. Refuse
    # that extension until the transformation has a corresponding parser.
    if re.search(r'\\begin\{(?:verbatim\*?|lstlisting|minted)\}', text):
        raise ValueError('Literal environments require a reviewed anonymization parser')
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    text = re.sub(r'\\newif\\ifanonymous\s*', '', text)
    token = re.compile(r'\\ifdefined\\TMLRAnonymous\b|\\ifanonymous\b|\\else\b|\\fi\b')
    stack, active, pieces, start = [], True, [], 0
    for match in token.finditer(text):
        if active:
            pieces.append(text[start:match.start()])
        value = match.group()
        if value in (r'\ifdefined\TMLRAnonymous', r'\ifanonymous'):
            stack.append((active, False))
        elif value == r'\else':
            if not stack or stack[-1][1]:
                raise ValueError('Unmatched or repeated anonymous conditional else')
            parent, _ = stack[-1]
            stack[-1] = (parent, True)
            active = False
        else:
            if not stack:
                raise ValueError('Unmatched anonymous conditional end')
            active = stack.pop()[0]
        start = match.end()
        # TeX control words consume following whitespace. Leaving the command's
        # empty line behind can otherwise introduce a new paragraph boundary.
        following_line_end = re.match(r'[ \t]*\r?\n', text[start:])
        if following_line_end:
            start += following_line_end.end()
    if stack:
        raise ValueError('Unclosed anonymous conditional')
    if active:
        pieces.append(text[start:])
    text = ''.join(pieces)
    text = re.sub(r'\\anonymoustrue\s*', '', text)
    # This explicit source boundary fails closed if author syntax changes.
    text, count = re.subn(r'\\author\{\\name [^\n]*\n\\addr [^\n]*\}',
                         lambda _: r'\author{\name Anonymous authors}', text)
    if count != 1:
        raise ValueError('Expected exactly one reviewed author declaration')
    assert_anonymous(text, 'paper.tex')
    return text


def assert_anonymous(text, name):
    lowered = text.lower()
    hits = [value for value in IDENTIFIERS if value in lowered]
    if hits:
        raise ValueError('Identifying text in ' + name + ': ' + ', '.join(hits))
    if re.search(r'(?i)(?:[a-z]:[\\/](?:users|home)[\\/]|/(?:Users|home|workspace)/)', text):
        raise ValueError('Local machine path in ' + name)


def member_name(name):
    """Require the same safe spelling for source lookup, disk output and ZIP."""
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', name):
        raise ValueError('Invalid normalized relative POSIX member path: ' + repr(name))
    if any(part in {'.', '..'} for part in name.split('/')):
        raise ValueError('Invalid normalized relative POSIX member path: ' + repr(name))
    return name


def directives(text, command):
    # This is deliberately bounded to the current reviewed source syntax.
    # Bare filenames, macros, stars and nested option/argument braces require
    # a parser extension rather than being silently left outside the ZIP.
    pattern = re.compile(r'\\' + command + r'\b(?:\[[^\[\]{}]*\])?\{([^{}]+)\}')
    found = list(pattern.finditer(text))
    if len(found) != len(re.findall(r'\\' + command + r'\b', text)):
        raise ValueError('Unsupported manuscript data directive: ' + command)
    return [match.group(1) for match in found]


def manuscript_dependencies(text, *, top_level):
    # Third-party style files are separately preserved, reviewed dependencies.
    # This check applies to the manuscript and its table fragments, which must
    # not acquire another uncurated loading route inside an otherwise safe name.
    if re.search(r'\\(?:include|includeonly|graphicspath|DeclareGraphicsExtensions|'
                 r'InputIfFileExists|IfFileExists|openin|read|readline|catcode|csname|'
                 r'directlua|write18|RequirePackage|LoadClass)\b', text):
        raise ValueError('Unsupported manuscript data-loading directive')
    found = {command: directives(text, command) for command in DATA_DIRECTIVES}
    if not top_level:
        if any(found.values()):
            raise ValueError('Nested manuscript data directives require explicit review')
        return set()
    if found['bibliography'] != ['references'] or found['bibliographystyle'] != ['tmlr']:
        raise ValueError('Changed bibliography or style target requires explicit review')
    if found['documentclass'] != ['article']:
        raise ValueError('Changed document class requires explicit review')
    packages = [name for group in found['usepackage'] for name in group.split(',')]
    if not packages or 'tmlr' not in packages or any(name not in TEX_PACKAGES for name in packages):
        raise ValueError('Unreviewed template package or package path')
    names = set(BASE_FILES)
    for name in found['input']:
        name = member_name(name)
        name = name if Path(name).suffix else name + '.tex'
        if name not in TABLE_FILES:
            raise ValueError('Unreviewed manuscript input: ' + name)
        names.add(name)
    for name in found['includegraphics']:
        name = member_name(name)
        if name not in FIGURE_FILES:
            raise ValueError('Unreviewed manuscript figure: ' + name)
        names.add(name)
    return names


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    validate_revision_assets(root)
    if output.exists():
        raise FileExistsError('Choose a fresh package directory')
    source = (root / 'paper/paper.tex').read_text(encoding='utf-8')
    transformed = anonymous_source(source)
    files = {'paper.tex': transformed.encode('utf-8')}
    names = manuscript_dependencies(transformed, top_level=True)
    inputs = {'paper/paper.tex': digest((root / 'paper/paper.tex').read_bytes())}
    for name in sorted(names):
        member_name(name)
        path = (root / 'paper' / name).resolve()
        if not path.is_relative_to(root / 'paper') or not path.is_file():
            raise ValueError('Invalid manuscript dependency: ' + name)
        data = path.read_bytes()
        if path.suffix != '.pdf':
            assert_anonymous(data.decode('utf-8'), name)
        if name in TABLE_FILES:
            manuscript_dependencies(data.decode('utf-8'), top_level=False)
        files[name] = data
        inputs['paper/' + name] = digest(data)
    files['README.txt'] = (
        'Anonymous manuscript source component, working draft.\n'
        'Build: pdflatex paper.tex; bibtex paper; pdflatex paper.tex; pdflatex paper.tex.\n'
        'The official TMLR style and its Apache2 license are preserved unchanged.\n'
        'This contains manuscript sources, not the complete code/data supplement.\n'
        'Scientific validation and final submission review remain open.\n'
    ).encode()
    manifest = dict(kind='anonymous_manuscript_sources', status='working_draft_not_submission_ready',
                    artifacts_sha256={name:digest(data) for name,data in sorted(files.items())})
    files['manifest.json'] = (json.dumps(manifest,indent=2,sort_keys=True)+'\n').encode()
    # Validate every output member, including generated templates, before
    # creating any output. A source-path containment check alone is insufficient.
    for name in files:
        member_name(name)
    output.mkdir(parents=True, exist_ok=False)
    work = output / 'sources'
    work.mkdir()
    for name,data in files.items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    archive = output / 'anonymous-manuscript-sources-draft.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as package:
        for name,data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            package.writestr(info,data)
    provenance = dict(canonical_inputs_sha256=inputs, source_transform='anonymous_branch_only_comments_and_author_removed',
                      package_sha256=digest(archive.read_bytes()), files=len(files),
                      further_checks=['compile_exact_derivative','all_page_render_review','PDF_text_metadata_links_embedded_assets',
                                      'complete_anonymous_code_data_supplement','scientific_and_author_specific_submission_gates'])
    (output / 'private-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(archive=str(archive),sha256=provenance['package_sha256'],files=len(files))))
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.out)
