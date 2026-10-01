"""Numerical queue integrity and failed-attempt boundaries without GPU work."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiment_io import file_hash, object_hash, strict_json
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


@pytest.mark.parametrize('selection', [None, {'rule':'uniform_executed_calls_v1','seed':930001}])
def test_queue_preserves_named_snapshot_checkpoint_identity(tmp_path, monkeypatch, selection):
    """HF symlink resolution must not replace a recorded checkpoint filename."""
    import paired_comparison as paired
    from scripts import validate_fp32_probe
    root, _ = fixture_queue(tmp_path, monkeypatch, count=1)
    snapshot = tmp_path / 'snapshots' / 'revision' / 'mp_rank_00_model_states.pt'
    blob = tmp_path / 'blobs' / ('a' * 64)
    original_resolve = Path.resolve
    def hf_resolve(path, *args, **kwargs):
        return blob if path == snapshot else original_resolve(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'resolve', hf_resolve)
    assert snapshot.resolve().name != snapshot.name
    protocol = root / 'selection.json'
    queue.exclusive_json(protocol, dict(recorded_before_execution=True, decision_id='E01',
        protocol_version=6, source_collection_protocol_sha256='collection', validation={}, uniform_call_selection=selection,
        strata=[dict(id='one', task='PickCube-v1', model='1b', checkpoint_mode='authors', checkpoint_sha256='bytes')]))
    bank = dict(task='PickCube-v1', model='1b', pipeline=dict(checkpoint_mode='authors', checkpoint=dict(sha256='bytes')),
        selection=dict(collector_protocol_sha256='collection'),
        contexts=[dict(episode=0, policy_call_idx=0, status='available', context_id='context')])
    metrics = tmp_path / 'bank/one/metrics.jsonl'
    metrics.parent.mkdir(parents=True)
    metrics.write_text('authenticated metrics fixture\n')
    manifest_dir = Path(str(metrics) + '.run')
    manifest_dir.mkdir()
    config = dict(protocol_sha256='collection')
    queue.exclusive_json(manifest_dir/'manifest.json',dict(configuration=config,configuration_sha256=object_hash(config)))
    bank.update(source_configuration_sha256=object_hash(config),source_metrics_sha256=file_hash(metrics))
    selections = []
    def authenticated_bank(metrics, selection=None):
        selections.append(selection)
        return bank
    monkeypatch.setattr(paired, 'make_bank', authenticated_bank)
    monkeypatch.setattr(validate_fp32_probe, 'validate_decision', lambda *args: None)
    output = tmp_path / 'built'
    queue.build(SimpleNamespace(protocol=protocol, bank_root=tmp_path / 'bank', output=output,
                                lang_dir=tmp_path / 'language', checkpoint_path=snapshot))
    arguments = strict_json(output / 'queue.json')['jobs'][0]['common_arguments']
    actual = arguments[arguments.index('--checkpoint-path') + 1]
    assert actual == str(snapshot.absolute())
    assert Path(actual).name == 'mp_rank_00_model_states.pt'
    assert selections == [selection]


def test_collection_identity_works_for_uniform_banks_and_rejects_stale_manifest(tmp_path):
    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_text('fixed complete episode export\n')
    directory = Path(str(metrics) + '.run')
    directory.mkdir()
    config = dict(protocol_sha256='heldout-collection', call_selection='all stored calls')
    manifest = dict(configuration=config,configuration_sha256=object_hash(config))
    queue.exclusive_json(directory/'manifest.json',manifest)
    bank = dict(selection=dict(rule='uniform_executed_calls_v1'),
                source_configuration_sha256=object_hash(config),source_metrics_sha256=file_hash(metrics))
    assert queue.collection_protocol_hash(metrics,bank) == 'heldout-collection'
    config['protocol_sha256'] = 'different-collection'
    (directory/'manifest.json').write_bytes(queue.canonical_json(manifest))
    with pytest.raises(ValueError,match='configuration hash'):
        queue.collection_protocol_hash(metrics,bank)
