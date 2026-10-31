"""Authenticate and replay a complete reviewer archive, without a source checkout.

This file is copied to the archive root. Install the included research
supplement's pinned CPU requirements first, then run this file with --out DIR.
The original records and their path metadata are never rewritten. Only a new
runtime input configuration is generated, using paths inside this extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def member(name):
    if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', name)
            or '\\' in name or ':' in name
            or PurePosixPath(name).is_absolute()
            or any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError('Unsafe archive member: ' + repr(name))
    return name


def contained(root, name):
    path = root / member(name)
    current = root
    for part in PurePosixPath(name).parts:
        current = current / part
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError('Archive aliases are forbidden: ' + name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Archive path escapes extraction: ' + name)
    return path


def verify_archive(root):
    root = Path(root).resolve()
    expected = {}
    for line in (root / 'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        match = re.fullmatch(r'([a-f0-9]{64})  (.+)', line)
        if not match:
            raise ValueError('Invalid archive manifest record')
        digest, name = match.groups()
        member(name)
        if name in expected or name == 'MANIFEST.sha256':
            raise ValueError('Repeated or self-referential manifest member: ' + name)
        expected[name] = digest
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(expected) | {'MANIFEST.sha256'}:
        raise ValueError('Archive inventory differs from its manifest')
    for name, digest in expected.items():
        if sha256(contained(root, name)) != digest:
            raise ValueError('Archive hash mismatch: ' + name)
    return len(expected)


def extract_code(archive, destination):
    """Validate every ZIP name and streamed byte before executing any source."""
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate code archive members')
        for info in source.infolist():
            member(info.orig_filename)
            member(info.filename)
            if info.is_dir() or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Code archive must contain regular files only')
        manifest = json.loads(source.read('SUPPLEMENT_MANIFEST.json'))
        expected = manifest['members']
        if set(names) != set(expected) | {'SUPPLEMENT_MANIFEST.json'}:
            raise ValueError('Code supplement inventory differs from its manifest')
        destination.mkdir(parents=True, exist_ok=False)
        for name in names:
            target = contained(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            size, digest = 0, hashlib.sha256()
            with source.open(name) as stream, target.open('xb') as output:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    size += len(block)
                    digest.update(block)
                    output.write(block)
            if name in expected and (size != expected[name]['bytes'] or digest.hexdigest() != expected[name]['sha256']):
                raise ValueError('Code supplement hash mismatch: ' + name)


def validate_outcome_inputs(root, config, original):
    """Require a hash-authenticated outcome file for every registered bank."""
    banks = original['inputs']['banks']
    ids = [spec['id'] for spec in banks]
    if len(ids) != len(set(ids)) or set(config['metrics']) != set(ids):
        raise ValueError('Outcome inputs must cover exactly every registered bank')
    for spec in banks:
        bank = json.loads(contained(root, config['input_paths'][spec['bank']]).read_text(encoding='utf-8'))
        path = contained(root, config['metrics'][spec['id']])
        if not path.is_file() or sha256(path) != bank.get('source_metrics_sha256'):
            raise ValueError('Outcome metrics hash mismatch: ' + spec['id'])


def replay(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('Choose a fresh output directory outside the archive')
    count = verify_archive(root)
    config = json.loads((root / 'reproduction.json').read_text(encoding='utf-8'))
    original = json.loads(contained(root, config['expected_manifest']).read_text(encoding='utf-8'))
    validate_outcome_inputs(root, config, original)
    inputs = json.loads(json.dumps(original['inputs']))
    inputs['run_root'] = str(contained(root, 'runs'))
    inputs.pop('note', None)
    for spec in inputs['banks']:
        for key in ('bank', 'protocol'):
            spec[key] = str(contained(root, config['input_paths'][spec[key]]))
    out.mkdir(parents=True)
    code = out / 'code'
    extract_code(contained(root, config['code']), code)
    input_file = out / 'inputs.json'
    input_file.write_text(json.dumps(inputs, indent=2) + '\n', encoding='utf-8', newline='\n')
    command = [sys.executable, '-m', 'analysis.revision_gpu.v1.analyze', '--inputs', str(input_file),
               '--plan', str(contained(root, config['plan'])), '--out', str(out / 'analysis')]
    for path in config['coverage']:
        command += ['--coverage', str(contained(root, path))]
    for bank, path in config['metrics'].items():
        command += ['--metrics-copy', bank + '=' + str(contained(root, path))]
    subprocess.run(command, cwd=code, check=True)
    actual = json.loads((out / 'analysis/manifest.json').read_text(encoding='utf-8'))
    differences = {}
    for name in sorted(set(original['outputs']) | set(actual['outputs'])):
        expected = original['outputs'].get(name)
        path = contained(out / 'analysis', name)
        value = sha256(path) if path.is_file() else None
        if value != actual['outputs'].get(name):
            raise ValueError('Generated analysis manifest does not bind its output: ' + name)
        if value != expected:
            differences[name] = {'expected_sha256': expected, 'actual_sha256': value}
    report = {'archive_files_verified': count, 'analysis_kind': actual['kind'],
              'outputs_compared': len(original['outputs']), 'differences': differences,
              'population_matches': actual['population'] == original['population'],
              'plan_matches': actual['plan_sha256'] == original['plan_sha256'],
              'coverage_matches': sorted(actual['coverage_sha256'].values()) == sorted(original['coverage_sha256'].values()),
              'full_registered_analysis': actual['kind'] == 'nhsjs_r1_gpu_analysis'}
    report['verified'] = not differences and all(report[key] for key in
        ('population_matches', 'plan_matches', 'coverage_matches', 'full_registered_analysis'))
    (out / 'reproduction-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps({'archive_files_verified': verify_archive(args.archive)}))
    elif args.out is None:
        parser.error('--out is required for analysis replay')
    else:
        raise SystemExit(0 if replay(args.archive, args.out)['verified'] else 1)
