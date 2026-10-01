"""Atomic publication failures must remain visible and preserve their evidence."""
import errno
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiment_io as io


def access_denied(winerror=5):
    error = PermissionError(errno.EACCES, 'injected access denied')
    if winerror is not None:
        error.winerror = winerror
    return error


def candidate_for(path):
    candidates = list(path.parent.glob(path.name + '.*.tmp'))
    assert len(candidates) == 1
    return candidates[0]


def test_transient_windows_failure_retries_same_complete_candidate(tmp_path, monkeypatch):
    target = tmp_path / 'report.json'
    target.write_bytes(b'old complete report')
    content = b'new complete report'
    real_replace = io.os.replace
    real_fsync = io.os.fsync
    attempts, waits, syncs = [], [], []

    def replace(source, destination):
        assert target.read_bytes() == b'old complete report'
        assert Path(source).read_bytes() == content
        attempts.append(Path(source))
        if len(attempts) < 3:
            raise access_denied()
        real_replace(source, destination)

    def fsync(fd):
        syncs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(io, '_WINDOWS', True)
    monkeypatch.setattr(io.os, 'replace', replace)
    monkeypatch.setattr(io.os, 'fsync', fsync)
    monkeypatch.setattr(io.time, 'sleep', waits.append)
    io.atomic_bytes(target, content)
    assert target.read_bytes() == content
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert waits == [0.01, 0.05]
    assert len(syncs) == 1
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('destination_exists', [False, True])
def test_persistent_windows_failure_keeps_candidate_and_raises_original(
        tmp_path, monkeypatch, destination_exists):
    target = tmp_path / 'report.json'
    if destination_exists:
        target.write_bytes(b'old complete report')
    error = access_denied()
    calls, waits = [], []

    def replace(source, destination):
        calls.append((source, destination))
        raise error

    monkeypatch.setattr(io, '_WINDOWS', True)
    monkeypatch.setattr(io.os, 'replace', replace)
    monkeypatch.setattr(io.time, 'sleep', waits.append)
    with pytest.raises(PermissionError) as caught:
        io.atomic_bytes(target, b'pending report')
    assert caught.value is error
    assert len(calls) == 4 and waits == [0.01, 0.05, 0.20]
    assert target.exists() == destination_exists
    if destination_exists:
        assert target.read_bytes() == b'old complete report'
    pending = candidate_for(target)
    assert pending.read_bytes() == b'pending report'
    note = error.__notes__[-1]
    assert str(pending) in note and '4 attempt(s)' in note
    assert hashlib.sha256(b'pending report').hexdigest() in note
    # A later fresh publication must not erase an earlier failed candidate.
    monkeypatch.undo()
    io.atomic_bytes(target, b'later complete report')
    assert target.read_bytes() == b'later complete report'
    assert pending.read_bytes() == b'pending report'


@pytest.mark.parametrize('windows,error', [
    (False, access_denied()),
    (True, access_denied(None)),
    (True, access_denied(32)),
    (True, access_denied(13)),
    (True, OSError(errno.ENOSPC, 'injected disk full')),
    (True, FileNotFoundError(errno.ENOENT, 'injected missing directory')),
])
def test_other_platforms_or_errors_are_not_retried(tmp_path, monkeypatch, windows, error):
    target = tmp_path / 'report.json'
    target.write_bytes(b'previous')
    attempts, waits = [], []

    def replace(source, destination):
        attempts.append(source)
        raise error

    monkeypatch.setattr(io, '_WINDOWS', windows)
    monkeypatch.setattr(io.os, 'replace', replace)
    monkeypatch.setattr(io.time, 'sleep', waits.append)
    with pytest.raises(type(error)) as caught:
        io.atomic_bytes(target, b'pending')
    assert caught.value is error
    assert len(attempts) == 1 and waits == []
    assert target.read_bytes() == b'previous'
    assert candidate_for(target).read_bytes() == b'pending'


def test_interrupted_retry_preserves_candidate_without_publishing(tmp_path, monkeypatch):
    target = tmp_path / 'report.json'
    interruption = KeyboardInterrupt()

    def replace(*args):
        raise access_denied()

    def interrupt(delay):
        raise interruption

    monkeypatch.setattr(io, '_WINDOWS', True)
    monkeypatch.setattr(io.os, 'replace', replace)
    monkeypatch.setattr(io.time, 'sleep', interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        io.atomic_bytes(target, b'pending')
    assert caught.value is interruption
    assert not target.exists()
    assert candidate_for(target).read_bytes() == b'pending'
    assert '1 attempt(s)' in interruption.__notes__[-1]


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_fsync_failure_does_not_publish_or_mask_original(tmp_path, monkeypatch, cleanup_fails):
    target = tmp_path / 'report.json'
    target.write_bytes(b'previous')
    error = OSError(errno.EIO, 'injected fsync failure')
    replacements = []

    def fsync(fd):
        raise error

    def failed_cleanup(*args, **kwargs):
        raise PermissionError('injected cleanup failure')

    monkeypatch.setattr(io.os, 'fsync', fsync)
    monkeypatch.setattr(io.os, 'replace', lambda *args: replacements.append(args))
    if cleanup_fails:
        monkeypatch.setattr(Path, 'unlink', failed_cleanup)
    with pytest.raises(OSError) as caught:
        io.atomic_bytes(target, b'pending')
    assert caught.value is error
    assert target.read_bytes() == b'previous' and replacements == []
    if cleanup_fails:
        assert candidate_for(target).read_bytes() == b'pending'
        assert 'Incomplete' in error.__notes__[-1] and 'cleanup failed' in error.__notes__[-1]
    else:
        assert not list(tmp_path.glob('*.tmp'))


def test_failed_exclusive_create_never_deletes_someone_elses_file(tmp_path, monkeypatch):
    target = tmp_path / 'report.json'
    collision = tmp_path / 'report.json.fixed.tmp'
    collision.write_bytes(b'foreign bytes')
    monkeypatch.setattr(io.uuid, 'uuid4', lambda: SimpleNamespace(hex='fixed'))
    with pytest.raises(FileExistsError):
        io.atomic_bytes(target, b'new bytes')
    assert collision.read_bytes() == b'foreign bytes'
    assert not target.exists()


def test_failed_episode_commit_retains_progress_and_is_not_accepted_on_resume(tmp_path, monkeypatch):
    store = io.RunStore(tmp_path / 'metrics.jsonl', {'test': 'atomic failure'})
    real_replace = io.os.replace
    waits = []

    def fail_commit(source, destination):
        if Path(destination).parent == store.root / 'episodes':
            raise access_denied()
        return real_replace(source, destination)

    with store.writer():
        transaction = store.begin_episode(0)
        transaction.save_step({'event': 'step', 'episode': 0, 'policy_call_idx': 0}, {})
        progress = (transaction.directory / 'progress.json').read_bytes()
        monkeypatch.setattr(io, '_WINDOWS', True)
        monkeypatch.setattr(io.os, 'replace', fail_commit)
        monkeypatch.setattr(io.time, 'sleep', waits.append)
        with pytest.raises(PermissionError):
            transaction.commit({'event': 'episode_end', 'episode': 0, 'policy_calls': 1})
        candidate = candidate_for(store.root / 'episodes' / 'ep000000.json')
        assert io.strict_json(candidate)['records'][-1]['event'] == 'episode_end'
        assert store.completed_episodes() == set()
        assert (transaction.directory / 'progress.json').read_bytes() == progress
    assert waits == [0.01, 0.05, 0.20]
    monkeypatch.undo()
    with io.RunStore(store.output, {'test': 'atomic failure'}, resume=True).writer() as resumed:
        assert resumed.completed_episodes() == set()
        assert resumed.output.read_bytes() == b''
        assert candidate.exists()
