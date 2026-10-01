"""Numerical queue integrity and failed-attempt boundaries without GPU work."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiment_io import file_hash, strict_json
from scripts import run_probe_queue as queue


def fixture_queue(tmp_path, monkeypatch, count=2):
    source = tmp_path / 'source'
    source.mkdir()
    for name in queue.SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('frozen numerical code\n')
    monkeypatch.setattr(queue, 'ROOT', source)
    root = tmp_path / 'queue'
    root.mkdir()
    for name in ('claims','logs','completion','prepared','results'):
        (root / name).mkdir()
    protocol = root / 'protocol.json'
    queue.exclusive_json(protocol, {'decision_id':'E01'})
    jobs = []
    for index in range(count):
        decision, bank = root / f'decision{index}.json', root / f'bank{index}.json'
        queue.exclusive_json(decision, {'index':index})
        queue.exclusive_json(bank, {'source':'authenticated fixture'})
        jobs.append(dict(job_id=f'context{index}', decision_file=str(decision), bank_file=str(bank),
                         decision_sha256=file_hash(decision), bank_sha256=file_hash(bank), common_arguments=[]))
    value = dict(source_sha256={name:file_hash(source/name) for name in queue.SOURCE_FILES},
                 protocol_path=str(protocol), protocol_sha256=file_hash(protocol), jobs=jobs)
    queue.exclusive_json(root/'queue.json',value)
    return root, SimpleNamespace(queue=root/'queue.json',index=0,workers=1)


def test_failed_prepare_keeps_claim_and_log_and_does_not_launch_later_context(tmp_path,monkeypatch):
    root,args=fixture_queue(tmp_path,monkeypatch)
    calls=[]
    def fail(command,**kwargs):
        calls.append(command)
        kwargs['stdout'].write(b'precise source replay failure\n')
        return SimpleNamespace(returncode=3)
    monkeypatch.setattr(queue.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='without erasing'):
        queue.worker(args)
    assert len(calls)==1 and calls[0][2]=='prepare'
    assert (root/'claims/context0.json').is_file()
    assert not (root/'claims/context1.json').exists()
    assert strict_json(root/'completion/context0.json')['returncode']==3
    assert b'source replay failure' in (root/'logs/context0-prepare.log').read_bytes()
    with pytest.raises(ValueError,match='recorded retry'):
        queue.worker(args)
    assert len(calls)==1


def test_completed_outputs_are_authenticated_before_skip(tmp_path,monkeypatch):
    root,args=fixture_queue(tmp_path,monkeypatch,count=1)
    calls=[]
    def succeed(command,**kwargs):
        calls.append(command)
        output=Path(command[command.index('--out')+1])
        output.mkdir()
        queue.exclusive_json(output/'report.json',{'status':'fixture_complete'})
        queue.exclusive_json(output/'completion.json',{'status':'fixture_complete',
                            'artifacts_sha256':{'report.json':file_hash(output/'report.json')}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue.subprocess,'run',succeed)
    queue.worker(args)
    assert [c[2] for c in calls]==['prepare','run']
    assert '--prepared-sha256' in calls[1]
    queue.worker(args)
    assert len(calls)==2
    (root/'results/context0/report.json').write_text('changed output')
    with pytest.raises(ValueError,match='artifact changed'):
        queue.worker(args)


def test_source_or_frozen_decision_change_prevents_new_job(tmp_path,monkeypatch):
    root,args=fixture_queue(tmp_path,monkeypatch,count=1)
    (root/'decision0.json').write_text('{}')
    with pytest.raises(ValueError,match='decision or frozen bank changed'):
        queue.worker(args)
    assert not list((root/'claims').iterdir())
    (queue.ROOT/queue.SOURCE_FILES[0]).write_text('changed implementation')
    with pytest.raises(ValueError,match='implementation changed'):
        queue.worker(args)
