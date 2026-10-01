"""Immutable run identities and episode transactions for new experiments.

Historical JSONL files are never resumed. A JSONL export is a derived view of
committed episodes; partial attempts and failures are retained separately.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import uuid

from filelock import FileLock


_WINDOWS = os.name == 'nt'
_REPLACE_RETRY_DELAYS = (0.01, 0.05, 0.20)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def object_hash(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(canonical_json({'shape': list(value.shape), 'dtype': str(value.dtype)}))
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def atomic_bytes(path, content):
    """Publish complete bytes atomically, retaining a failed replacement candidate.

    Windows access-denied errors can clear on a later attempt, but the error
    alone does not identify their cause. Retry only the observed WinError 5
    class, for at most 0.26 seconds of scheduled waiting. Other errors and
    persistent failures propagate. A retained ``.tmp`` is evidence, never a
    committed record or an automatically recoverable transaction.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    candidate_sha256 = hashlib.sha256(content).hexdigest()
    owned = False
    try:
        with temp.open('xb') as handle:
            owned = True
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException as error:
        # Do not delete a pre-existing name if exclusive creation failed, or
        # replace the primary write/fsync exception with a cleanup exception.
        if owned:
            try:
                temp.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error.add_note(f'Incomplete atomic-write candidate retained at {temp}; '
                               f'cleanup failed: {cleanup_error!r}')
        raise

    attempts = 0
    try:
        while True:
            attempts += 1
            try:
                os.replace(temp, path)
                return
            except PermissionError as error:
                if not (_WINDOWS and getattr(error, 'winerror', None) == 5
                        and attempts <= len(_REPLACE_RETRY_DELAYS)):
                    raise
                time.sleep(_REPLACE_RETRY_DELAYS[attempts - 1])
    except BaseException as error:
        # Keep the complete pending report even when retry waiting is
        # interrupted. Never delete the destination or use a copying fallback.
        error.add_note(f'Atomic replacement failed after {attempts} attempt(s); '
                       f'uncommitted candidate retained at {temp}; '
                       f'candidate_sha256={candidate_sha256}; destination={path}')
        raise


def strict_json(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'Duplicate JSON key {key!r} in {path}')
            obj[key] = value
        return obj
    def bad(value):
        raise ValueError(f'Nonfinite JSON value {value} in {path}')
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique, parse_constant=bad)


class RunStore:
    """Single-writer run store. Hold ``with store.writer():`` for every mutation."""

    def __init__(self, output, configuration, *, resume=False):
        self.output = Path(output).resolve()
        self.root = self.output.with_name(self.output.name + '.run')
        # Snapshot mutable caller input and normalize JSON types before hashing.
        self.configuration = json.loads(canonical_json(configuration))
        self.identity = object_hash(self.configuration)
        self.resume = resume
        self.lock = FileLock(str(self.root) + '.lock', timeout=0)
        self.manifest = None

    @contextmanager
    def writer(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            manifest_path = self.root / 'manifest.json'
            if manifest_path.exists():
                if not self.resume:
                    raise FileExistsError(f'Run exists: {self.root}; use an exact-compatible --resume')
                self.manifest = strict_json(manifest_path)
                if self.manifest.get('configuration') != self.configuration:
                    raise ValueError('Resume configuration mismatch; create a new output/run')
                if self.manifest.get('configuration_sha256') != self.identity:
                    raise ValueError('Run manifest identity mismatch')
            else:
                if self.output.exists() or self.root.exists():
                    raise FileExistsError('Existing output without a valid manifest; historical or incomplete data cannot be resumed')
                if self.resume:
                    raise FileNotFoundError('Cannot resume a nonexistent run')
                self.root.mkdir()
                self.manifest = {'schema_version': 1, 'run_id': uuid.uuid4().hex,
                                 'configuration': self.configuration,
                                 'configuration_sha256': self.identity,
                                 'host': platform.node()}
                atomic_bytes(manifest_path, canonical_json(self.manifest) + b'\n')
            (self.root / 'episodes').mkdir(exist_ok=True)
            (self.root / 'attempts').mkdir(exist_ok=True)
            self.completed_episodes()  # verify every committed artifact before reuse
            self.export()  # repair derived exports even when every episode is complete
            yield self

    def completed_episodes(self):
        completed = set()
        for path in sorted((self.root / 'episodes').glob('ep*.json')):
            doc = strict_json(path)
            episode = doc['episode']
            if episode in completed or path.name != f'ep{episode:06d}.json':
                raise ValueError(f'Duplicate/invalid episode identity: {path}')
            if doc['run_id'] != self.manifest['run_id'] or object_hash(doc['records']) != doc['records_sha256']:
                raise ValueError(f'Committed episode integrity failure: {path}')
            self._validate_records(episode, doc['records'])
            completed.add(episode)
        return completed

    def _validate_records(self, episode, records):
        if not records or records[-1].get('event') != 'episode_end':
            raise ValueError('Episode transaction lacks terminal record')
        seen = set()
        for index, row in enumerate(records):
            if row.get('episode') != episode or row.get('run_id') != self.manifest['run_id']:
                raise ValueError('Record identity differs from episode/run')
            if row.get('event') == 'step':
                key = row['policy_call_idx']
                if key in seen or key != len(seen):
                    raise ValueError('Duplicate or noncontiguous policy call')
                seen.add(key)
                sidecar = Path(row['attr_file'])
                if not sidecar.is_absolute():
                    sidecar = self.output.parent / sidecar
                sidecar = sidecar.resolve()
                if not sidecar.is_relative_to(self.root) or not sidecar.is_file():
                    raise ValueError('Missing sidecar or sidecar outside this run')
                if file_hash(sidecar) != row['attr_sha256']:
                    raise ValueError('Sidecar hash mismatch')
            elif row.get('event') != 'episode_end' or index != len(records) - 1:
                raise ValueError('Unknown event or misplaced terminal event')
        if records[-1].get('policy_calls') != len(seen):
            raise ValueError('Terminal policy-call count mismatch')

    def begin_episode(self, episode):
        if type(episode) is not int or episode < 0:
            raise ValueError('Episode must be a nonnegative integer')
        if episode in self.completed_episodes():
            raise FileExistsError('Episode already committed')
        attempt = self.root / 'attempts' / f'ep{episode:06d}-{uuid.uuid4().hex}'
        attempt.mkdir()
        return EpisodeTransaction(self, episode, attempt)

    def export(self):
        self.completed_episodes()
        data = b''.join(canonical_json(row) + b'\n'
                        for path in sorted((self.root / 'episodes').glob('ep*.json'))
                        for row in strict_json(path)['records'])
        if self.output.exists() and self.output.read_bytes() != data:
            # Exports are disposable, but retain interrupted/previous bytes too.
            backup = self.root / 'attempts' / ('export-' + uuid.uuid4().hex + '.jsonl')
            atomic_bytes(backup, self.output.read_bytes())
        atomic_bytes(self.output, data)


class EpisodeTransaction:
    def __init__(self, store, episode, directory):
        self.store, self.episode, self.directory = store, episode, directory
        self.records = []

    def save_step(self, row, payload):
        import torch
        call = row['policy_call_idx']
        if call != len(self.records) or row.get('episode') != self.episode:
            raise ValueError('Step is duplicate, out of order, or belongs to another episode')
        path = self.directory / f'call{call:06d}.pt'
        if path.exists():
            raise FileExistsError(path)
        payload = {**payload, 'run_id': self.store.manifest['run_id'],
                   'configuration_sha256': self.store.identity}
        temp = path.with_suffix('.tmp')
        with temp.open('xb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        row = {**row, 'run_id': self.store.manifest['run_id'],
               'configuration_sha256': self.store.identity,
               'attr_file': path.relative_to(self.store.output.parent).as_posix(),
               'attr_sha256': file_hash(path)}
        canonical_json(row)
        self.records.append(row)
        atomic_bytes(self.directory / 'progress.json', canonical_json(self.records) + b'\n')
        return row

    def fail(self, reason, diagnostics=None):
        atomic_bytes(self.directory / 'failure.json', canonical_json({
            'episode': self.episode, 'run_id': self.store.manifest['run_id'],
            'reason': str(reason), 'diagnostics': diagnostics or {},
            'completed_calls': len(self.records)}) + b'\n')

    def commit(self, terminal):
        terminal = {**terminal, 'run_id': self.store.manifest['run_id']}
        records = self.records + [terminal]
        self.store._validate_records(self.episode, records)
        target = self.store.root / 'episodes' / f'ep{self.episode:06d}.json'
        if target.exists():
            raise FileExistsError('Committed episode cannot be overwritten')
        doc = {'episode': self.episode, 'run_id': self.store.manifest['run_id'],
               'records': records, 'records_sha256': object_hash(records)}
        atomic_bytes(target, canonical_json(doc) + b'\n')
        self.store.export()
