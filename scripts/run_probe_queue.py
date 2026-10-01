"""Freeze and execute independent exact-production numerical validation jobs.

Every job has a derived immutable decision before execution. Source replay and
candidate validation run in separate Python processes. Failed attempts and
claims are retained; an automatic retry cannot silently replace them.
"""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_io import atomic_bytes, canonical_json, file_hash, object_hash, strict_json

SOURCE_FILES = ('paired_comparison.py', 'fp32_probe_cache.py', 'faithfulness.py',
                'integrated_gradients.py', 'experiment_io.py', 'pipeline.py',
                'checkpoint_contract.py', 'per_step_attribution.py', 'rdt_sampling.py',
                'scripts/validate_fp32_probe.py',
                'scripts/validate_downstream_precision.py', 'scripts/validate_gradient_repeatability.py',
                'scripts/validate_rdt_numerics.py', 'scripts/run_probe_queue.py')


def exclusive_json(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical_json(value) + b'\n')


def verify_sources(queue):
    if {name: file_hash(ROOT / name) for name in SOURCE_FILES} != queue['source_sha256']:
        raise ValueError('Numerical implementation changed after queue freeze')


def verify_completion(path, queue_hash):
    prior = strict_json(path)
    if prior['queue_sha256'] != queue_hash or prior['returncode'] != 0:
        raise ValueError('Failed or inconsistent prior job requires an explicit recorded retry')
    for item in prior['artifacts']:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Completed numerical artifact changed')
    return prior


def collection_protocol_hash(metrics, bank):
    """Authenticate the collection protocol independently of call-selection mode."""
    manifest = strict_json(Path(str(metrics) + '.run') / 'manifest.json')
    config = manifest['configuration']
    if object_hash(config) != manifest['configuration_sha256']:
        raise ValueError('Collection configuration hash differs from its manifest')
    if manifest['configuration_sha256'] != bank['source_configuration_sha256']:
        raise ValueError('Bank and collection configuration differ')
    if file_hash(metrics) != bank['source_metrics_sha256']:
        raise ValueError('Collection changed after bank authentication')
    return config['protocol_sha256']


def build(args):
    import paired_comparison as paired
    protocol = strict_json(args.protocol)
    if protocol.get('recorded_before_execution') is not True:
        raise ValueError('Record the numerical protocol before queue construction')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('banks', 'decisions', 'claims', 'logs', 'completion', 'prepared', 'results'):
        (output / name).mkdir()
    jobs = []
    for stratum in protocol['strata']:
        metrics = args.bank_root.resolve() / stratum['id'] / 'metrics.jsonl'
        selection = stratum.get('uniform_call_selection', protocol.get('uniform_call_selection'))
        bank = paired.make_bank(metrics, selection=selection)
        if bank['task'] != stratum['task'] or bank['model'] != stratum['model']:
            raise ValueError('Source stratum differs from the recorded scope')
        if bank['pipeline']['checkpoint_mode'] != stratum['checkpoint_mode']:
            raise ValueError('Source checkpoint mode differs from the recorded stratum')
        if stratum.get('checkpoint_sha256') and bank['pipeline']['checkpoint']['sha256'] != stratum['checkpoint_sha256']:
            raise ValueError('Source checkpoint bytes differ from the recorded stratum')
        if collection_protocol_hash(metrics, bank) != protocol['source_collection_protocol_sha256']:
            raise ValueError('Context source collection protocol differs')
        bank_file = output / 'banks' / (stratum['id'] + '.json')
        exclusive_json(bank_file, bank)
        for entry in bank['contexts']:
            episode, call = entry['episode'], entry['policy_call_idx']
            job_id = f"{stratum['id']}-e{episode:03d}-c{call:03d}"
            decision = {**protocol['validation'], 'decision_id': protocol['decision_id'],
                        'protocol_version': protocol['protocol_version'],
                        'recorded_before_execution': True,
                        'parent_protocol_sha256': file_hash(args.protocol),
                        'bank_sha256': file_hash(bank_file),
                        'contexts': [{'episode': episode, 'policy_call_idx': call}]}
            decision_file = output / 'decisions' / (job_id + '.json')
            exclusive_json(decision_file, decision)
            from scripts.validate_fp32_probe import validate_decision
            validate_decision(decision, bank, file_hash(bank_file))
            common = ['--metrics', str(metrics), '--bank', str(bank_file),
                      '--decision-file', str(decision_file), '--lang-dir', str(args.lang_dir.resolve())]
            if stratum['checkpoint_mode'] != 'pretrained':
                if args.checkpoint_path is None:
                    raise ValueError('An explicit authors checkpoint path is required')
                # The source identity records the named snapshot filename;
                # following an HF symlink replaces it with the blob basename.
                common.extend(['--checkpoint-path', str(args.checkpoint_path.absolute())])
            jobs.append(dict(job_id=job_id, planned_status=entry['status'], context_id=entry['context_id'],
                             decision_sha256=file_hash(decision_file), bank_sha256=file_hash(bank_file),
                             decision_file=str(decision_file), bank_file=str(bank_file), common_arguments=common))
    queue = dict(decision_id=protocol['decision_id'], protocol_sha256=file_hash(args.protocol),
                 protocol_path=str(args.protocol.resolve()),
                 source_sha256={name: file_hash(ROOT / name) for name in SOURCE_FILES}, jobs=jobs,
                 scope='Exact production probe; diagnostics complete is not numerical approval')
    exclusive_json(output / 'queue.json', queue)
    print(dict(jobs=len(jobs), available=sum(j['planned_status'] == 'available' for j in jobs),
               queue_sha256=file_hash(output / 'queue.json')), flush=True)


def worker(args):
    if args.workers < 1 or not 0 <= args.index < args.workers:
        raise ValueError('Invalid queue partition')
    queue_path = args.queue.resolve()
    queue = strict_json(queue_path)
    queue_hash = file_hash(queue_path)
    verify_sources(queue)
    if file_hash(queue['protocol_path']) != queue['protocol_sha256']:
        raise ValueError('Parent numerical protocol changed')
    for index, job in enumerate(queue['jobs']):
        if index % args.workers != args.index:
            continue
        completion_path = queue_path.parent / 'completion' / (job['job_id'] + '.json')
        if completion_path.exists():
            verify_completion(completion_path, queue_hash)
            continue
        if file_hash(job['decision_file']) != job['decision_sha256'] or file_hash(job['bank_file']) != job['bank_sha256']:
            raise ValueError('Derived decision or frozen bank changed')
        verify_sources(queue)
        claim = dict(job_id=job['job_id'], queue_sha256=queue_hash, partition=args.index,
                     partitions=args.workers, hostname=platform.node(), started_unix=time.time())
        exclusive_json(queue_path.parent / 'claims' / (job['job_id'] + '.json'), claim)
        prepared = queue_path.parent / 'prepared' / job['job_id']
        results = queue_path.parent / 'results' / job['job_id']
        started = time.perf_counter()
        artifacts, stages = [], []
        for stage, destination in (('prepare', prepared), ('run', results)):
            verify_sources(queue)
            command = [sys.executable, str(ROOT / 'scripts/validate_fp32_probe.py'), stage,
                       *job['common_arguments'], '--out', str(destination)]
            if stage == 'run':
                command.extend(['--prepared', str(prepared), '--prepared-sha256', file_hash(prepared / 'completion.json')])
            print(f"worker {args.index}: {index+1}/{len(queue['jobs'])} {job['job_id']} {stage}", flush=True)
            log = queue_path.parent / 'logs' / (job['job_id'] + '-' + stage + '.log')
            with log.open('xb') as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            verify_sources(queue)
            stages.append(dict(stage=stage, returncode=result.returncode, log=str(log)))
            if result.returncode:
                break
            completion = strict_json(destination / 'completion.json')
            for name, digest in completion['artifacts_sha256'].items():
                path = (destination / name).resolve()
                if not path.is_relative_to(destination) or file_hash(path) != digest:
                    raise ValueError('Invalid completed-stage artifact')
                artifacts.append(dict(path=str(path), sha256=digest))
            artifacts.append(dict(path=str(destination / 'completion.json'), sha256=file_hash(destination / 'completion.json')))
        verify_sources(queue)
        exclusive_json(completion_path, {**claim, 'returncode': stages[-1]['returncode'], 'stages': stages,
                       'elapsed_seconds': time.perf_counter()-started, 'ended_unix': time.time(), 'artifacts': artifacts})
        if stages[-1]['returncode']:
            raise RuntimeError('Numerical job failed; worker stopped without erasing its attempt')
        if file_hash(queue_path) != queue_hash:
            raise ValueError('Numerical queue changed during execution')
    print(f'worker {args.index}: exact-probe queue partition complete', flush=True)


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
