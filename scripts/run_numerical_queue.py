"""Freeze and partition the recorded E01 numerical-coverage jobs.

Workers use disjoint deterministic queue indices, exclusive claim/log files,
and unique outputs. Any failed job stops that worker for diagnosis. This does
not change the study population or promote an engineering result to a claim.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_io import RunStore, atomic_bytes, canonical_json, file_hash, object_hash, strict_json


def build(args):
    protocol = strict_json(args.protocol)
    output = args.output.resolve()
    jobs, missing = [], []
    for stratum in protocol['strata']:
        run = args.bank_root.resolve() / stratum['id'] / 'metrics.jsonl'
        manifest = strict_json(run.with_name(run.name + '.run') / 'manifest.json')
        config = manifest['configuration']
        if object_hash(config) != manifest['configuration_sha256'] or config['protocol_sha256'] != file_hash(args.protocol):
            raise ValueError('Bank protocol/configuration hash mismatch')
        if config['stratum_id'] != stratum['id']:
            raise ValueError('Bank stratum differs from protocol')
        store = RunStore(run, config, resume=True)
        store.manifest = manifest
        if store.completed_episodes() != set(range(stratum['episodes'])):
            raise ValueError('Every declared episode needs complete accounting before queue creation')
        documents = [strict_json(p) for p in sorted((store.root / 'episodes').glob('ep*.json'))]
        rows = [r for d in documents for r in d['records'] if r['event'] == 'step']
        lookup = {(r['episode'], r['policy_call_idx']): r for r in rows}
        for episode in range(stratum['episodes']):
            for call in stratum['evaluate_calls']:
                row = lookup.get((episode, call))
                if row is None:
                    terminal = documents[episode]['records'][-1]
                    if call < terminal['policy_calls']:
                        raise ValueError('An executed call is missing from its committed episode')
                    missing.append(dict(stratum=stratum['id'], episode=episode, call=call,
                                        reason='call_not_reached', terminal=terminal))
                    continue
                if row.get('selected_for_analysis') is not True:
                    raise ValueError('Source selection flag differs from protocol')
                for target in protocol['comparison']['targets']:
                    job_id = f"{stratum['id']}-e{episode:03d}-c{call:03d}-{target}"
                    comparison = protocol['comparison']
                    command = [sys.executable, str(ROOT / 'scripts/validate_downstream_precision.py'),
                        '--run', str(run), '--sidecar', str(run.parent / row['attr_file']),
                        '--decision-id', protocol['decision_id'], '--decision-file', str(args.protocol.resolve()),
                        '--out', str(output / 'results' / job_id), '--target', target,
                        '--budgets', *map(str, comparison['budgets']), '--quadrature', comparison['quadrature'],
                        '--modalities', *comparison['modalities'], '--repeats', str(comparison['endpoint_repeats']),
                        '--fractions', *map(str, comparison['fractions']), '--lang-dir', str(args.lang_dir.resolve())]
                    if stratum['checkpoint_mode'] != 'pretrained':
                        if args.checkpoint_path is None:
                            raise ValueError('Explicit authors checkpoint path is required')
                        # Preserve the named snapshot entry. Resolving an HF
                        # symlink changes the filename field in the checkpoint
                        # identity even though its authenticated bytes match.
                        command.extend(['--checkpoint-path', str(args.checkpoint_path.absolute())])
                    jobs.append(dict(job_id=job_id, context_id=row['context_id'],
                                     source_manifest_sha256=file_hash(store.root / 'manifest.json'),
                                     source_sidecar_sha256=row['attr_sha256'], command=command))
    queue = dict(decision_id=protocol['decision_id'], protocol_sha256=file_hash(args.protocol),
                 builder_sha256=file_hash(__file__), validator_sha256=file_hash(ROOT / 'scripts/validate_downstream_precision.py'),
                 planned_contexts=sum(s['episodes']*len(s['evaluate_calls']) for s in protocol['strata']),
                 unavailable=missing, jobs=jobs)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('claims', 'logs', 'completion', 'results'):
        (output / name).mkdir()
    atomic_bytes(output / 'queue.json', canonical_json(queue)+b'\n')
    print({'planned_contexts': queue['planned_contexts'], 'unavailable_contexts': len(missing),
           'jobs': len(jobs), 'queue_sha256': file_hash(output / 'queue.json')}, flush=True)


def worker(args):
    if args.workers < 1 or not 0 <= args.index < args.workers:
        raise ValueError('Invalid worker partition')
    queue_path = args.queue.resolve()
    queue = strict_json(queue_path)
    queue_hash = file_hash(queue_path)
    if file_hash(ROOT / 'scripts/validate_downstream_precision.py') != queue['validator_sha256']:
        raise ValueError('Validator changed after queue freeze')
    if file_hash(__file__) != queue['builder_sha256']:
        raise ValueError('Queue orchestrator changed after freeze')
    for index, job in enumerate(queue['jobs']):
        if index % args.workers != args.index:
            continue
        completion = queue_path.parent / 'completion' / (job['job_id']+'.json')
        if completion.exists():
            prior = strict_json(completion)
            if prior['queue_sha256'] != queue_hash or prior['returncode'] != 0:
                raise ValueError('Existing failed or inconsistent job needs explicit diagnosis')
            continue
        claim = dict(job_id=job['job_id'], queue_sha256=queue_hash, partition=args.index,
                     partitions=args.workers, started_unix=time.time(), hostname=os.uname().nodename)
        with (queue_path.parent / 'claims' / (job['job_id']+'.json')).open('xb') as handle:
            handle.write(canonical_json(claim)+b'\n')
        print(f"starting {index+1}/{len(queue['jobs'])}: {job['job_id']}", flush=True)
        started = time.perf_counter()
        with (queue_path.parent / 'logs' / (job['job_id']+'.log')).open('xb') as handle:
            result = subprocess.run(job['command'], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
        atomic_bytes(completion, canonical_json({**claim, 'returncode': result.returncode,
                     'elapsed_seconds': time.perf_counter()-started, 'ended_unix': time.time()})+b'\n')
        if result.returncode:
            raise RuntimeError(f"Job {job['job_id']} failed; worker stopped for diagnosis")
        if file_hash(queue_path) != queue_hash:
            raise ValueError('Queue changed during execution')
    print(f'worker {args.index}: assigned jobs complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('build')
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--bank-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lang-dir', type=Path, required=True)
    p.add_argument('--checkpoint-path', type=Path)
    p = sub.add_parser('worker')
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    (build if args.mode == 'build' else worker)(args)


if __name__ == '__main__':
    main()
