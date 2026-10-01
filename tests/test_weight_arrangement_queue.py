"""Episode queue membership and immutable execution boundaries, without GPU work."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiment_io import canonical_json, file_hash, object_hash, strict_json
from scripts import run_weight_arrangement_queue as queue


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value)+b"\n")
    return path


def queue_fixture(tmp_path, monkeypatch, count=2):
    root=tmp_path/"queue";root.mkdir()
    for name in ("claims","logs","completion","prepared","results"):(root/name).mkdir()
    implementation={"fixture.py":"locked implementation"}
    monkeypatch.setattr(queue,"source_hashes",lambda:implementation)
    protocol=write(root/"protocol.json",{"kind":"prospective fixture"})
    native=write(root/"native.json",{"kind":"registry fixture"})
    jobs=[]
    for index in range(count):
        entries=[dict(episode=index,episode_id=f"episode-{index}",policy_call_idx=call,
                      context_id=f"context-{index}-{call}",status="available" if call==0 else "source_not_reached") for call in (0,12)]
        decision=write(root/f"decision{index}.json",dict(parameter_draws=2,parameter_master_seed=7,
                       parameter_seed_namespace="fixture",numerics={"locked":"criteria"},contexts=entries))
        bank=write(root/f"bank{index}.json",dict(contexts=entries))
        metrics=write(root/f"metrics{index}.jsonl",{"row":"authenticated source fixture"})
        manifest=write(root/f"manifest{index}.json",{"source":"fixture"})
        jobs.append(dict(job_id=f"fixture-e{index:03d}",episode_id=f"episode-{index}",contexts=entries,parameter_draws=2,
            decision_file=str(decision),decision_sha256=file_hash(decision),bank_file=str(bank),bank_sha256=file_hash(bank),
            metrics_file=str(metrics),metrics_sha256=file_hash(metrics),source_manifest_file=str(manifest),source_manifest_sha256=file_hash(manifest),
            structure_sha256="structure",checkpoint_path=None,checkpoint_sha256="checkpoint",common_arguments=[]))
    value=dict(kind="e05_episode_numerical_queue",schema_version=1,protocol_path=str(protocol),protocol_sha256=file_hash(protocol),
        native_registry_path=str(native),native_registry_sha256=file_hash(native),source_sha256=implementation,
        validation_implementation_sha256={"validator.py":"frozen validator"},jobs=jobs,retry_lineage=None)
    path=write(root/"queue.json",value)
    return root,SimpleNamespace(queue=path,queue_sha256=file_hash(path),index=0,workers=1),value


def stage_report(destination, stage, job, value, failure=False):
    destination.mkdir()
    status="complete_source_preparation" if stage=="prepare" else "complete_diagnostics_not_approval"
    report=dict(status=status,decision_sha256=job["decision_sha256"],bank_sha256=job["bank_sha256"],
        source_metrics_sha256=job["metrics_sha256"],native_registry_sha256=value["native_registry_sha256"],
        implementation_sha256=value["validation_implementation_sha256"])
    if stage=="prepare":
        report.update(kind="fp32_probe_source_preparation",contexts=deepcopy(job["contexts"]))
    else:
        rows=[]
        for draw in range(job["parameter_draws"]):
            for entry in job["contexts"]:
                status_arm="source_unavailable" if entry["status"]!="available" else "numerical_failure" if failure else "complete_diagnostics_not_approval"
                rows.append({**entry,"draw_index":draw,"numerics":[dict(modality=modality,target=target,status=status_arm)
                    for modality in ("vision","language","state") for target in ("Q","L2")]})
        report.update(kind="weight_arrangement_numerical_validation",planned_roster=job["contexts"],
            decision=strict_json(job["decision_file"]),structure_sha256=job["structure_sha256"],
            parameter_draws=job["parameter_draws"],planned_context_draw_target_modality_cells=len(job["contexts"])*job["parameter_draws"]*6,contexts=rows)
    write(destination/"report.json",report)
    write(destination/"raw.pt",{"raw":"stand-in saved tensor bytes"})
    reseal(destination)
    return report


def reseal(destination):
    report=strict_json(destination/"report.json")
    write(destination/"completion.json",dict(status=report["status"],artifacts_sha256={str(p.relative_to(destination)):file_hash(p)
        for p in destination.rglob("*") if p.is_file() and p.name!="completion.json"}))


def successful_child(value, calls, failure=False):
    def child(command,**kwargs):
        calls.append(command)
        destination=Path(command[command.index("--out")+1])
        job=next(job for job in value["jobs"] if job["job_id"]==destination.name)
        stage_report(destination,command[2],job,value,failure=failure)
        return SimpleNamespace(returncode=0)
    return child


@pytest.mark.parametrize("stage",["prepare","run"])
def test_failed_process_preserves_attempt_and_stops_later_episode(tmp_path,monkeypatch,stage):
    root,args,value=queue_fixture(tmp_path,monkeypatch)
    calls=[];succeed=successful_child(value,calls)
    def child(command,**kwargs):
        if command[2]!=stage:return succeed(command,**kwargs)
        calls.append(command);kwargs["stdout"].write(b"explicit retained failure\n")
        return SimpleNamespace(returncode=9)
    monkeypatch.setattr(queue.subprocess,"run",child)
    with pytest.raises(RuntimeError,match="without automatic retry"):queue.worker(args)
    record=strict_json(root/"completion/fixture-e000.json")
    assert record["returncode"]==9 and record["stages"][-1]["stage"]==stage
    assert (root/"claims/fixture-e000.json").exists() and not (root/"claims/fixture-e001.json").exists()
    assert "retained failure" in (root/f"logs/fixture-e000-{stage}.log").read_text()
    assert any(item["path"].endswith(f"-{stage}.log") for item in record["attempt_files"])
    before=len(calls)
    with pytest.raises(ValueError,match="explicit new retry queue"):queue.worker(args)
    assert len(calls)==before


def test_completed_failure_cells_are_diagnostics_and_artifacts_checked_before_skip(tmp_path,monkeypatch):
    root,args,value=queue_fixture(tmp_path,monkeypatch,count=1);calls=[]
    monkeypatch.setattr(queue.subprocess,"run",successful_child(value,calls,failure=True))
    queue.worker(args)
    assert [command[2] for command in calls]==["prepare","run"]
    assert calls[1][calls[1].index("--prepared-sha256")+1]==file_hash(root/"prepared/fixture-e000/completion.json")
    assert strict_json(root/"completion/fixture-e000.json")["returncode"]==0
    report=strict_json(root/"results/fixture-e000/report.json")
    assert len(report["contexts"])==4
    assert {arm["status"] for row in report["contexts"] for arm in row["numerics"]}=={"numerical_failure","source_unavailable"}
    queue.worker(args);assert len(calls)==2
    (root/"results/fixture-e000/raw.pt").write_bytes(b"changed")
    with pytest.raises(ValueError,match="attempt file changed"):queue.worker(args)


@pytest.mark.parametrize("mutation",["implementation","decision","queue"])
def test_midprocess_mutation_seals_failed_attempt_and_never_starts_candidate(tmp_path,monkeypatch,mutation):
    root,args,value=queue_fixture(tmp_path,monkeypatch);calls=[];succeed=successful_child(value,calls)
    def child(command,**kwargs):
        result=succeed(command,**kwargs)
        if mutation=="implementation":monkeypatch.setattr(queue,"source_hashes",lambda:{"changed":"code"})
        elif mutation=="decision":Path(value["jobs"][0]["decision_file"]).write_bytes(b"{}")
        else:args.queue.write_bytes(b"{}")
        return result
    monkeypatch.setattr(queue.subprocess,"run",child)
    with pytest.raises(RuntimeError):queue.worker(args)
    record=strict_json(root/"completion/fixture-e000.json")
    assert record["returncode"]==125 and record["failure"]["exception_type"]=="ValueError"
    assert len(calls)==1
    assert not (root/"claims/fixture-e001.json").exists()


def test_claim_collision_never_launches_or_overwrites_prior_claim(tmp_path,monkeypatch):
    root,args,value=queue_fixture(tmp_path,monkeypatch,count=1)
    claim=write(root/"claims/fixture-e000.json",{"unsealed":"prior worker"});digest=file_hash(claim)
    monkeypatch.setattr(queue.subprocess,"run",lambda *a,**k:pytest.fail("competing launch"))
    with pytest.raises(FileExistsError):queue.worker(args)
    assert file_hash(claim)==digest and not (root/"completion/fixture-e000.json").exists()


@pytest.mark.parametrize("mutation",["missing_cell","duplicate_cell","pending","wrong_context","wrong_draws","extra_file","path_escape"])
def test_completed_stage_rejects_roster_or_file_corruption(tmp_path,monkeypatch,mutation):
    root,args,value=queue_fixture(tmp_path,monkeypatch,count=1);job=value["jobs"][0]
    destination=root/"results"/job["job_id"]
    report=stage_report(destination,"run",job,value)
    if mutation=="missing_cell":report["contexts"][0]["numerics"].pop()
    elif mutation=="duplicate_cell":report["contexts"][0]["numerics"][0]=report["contexts"][0]["numerics"][1]
    elif mutation=="pending":report["contexts"][0]["numerics"][0]["status"]="own_maps_complete_pending_trained_responses"
    elif mutation=="wrong_context":report["contexts"][0]["context_id"]="different"
    elif mutation=="wrong_draws":report["parameter_draws"]=3
    elif mutation=="extra_file":write(destination/"undeclared.json",{})
    elif mutation=="path_escape":
        completion=strict_json(destination/"completion.json");completion["artifacts_sha256"]["../escape.json"]="anything"
        write(destination/"completion.json",completion)
    if mutation not in {"extra_file","path_escape"}:write(destination/"report.json",report);reseal(destination)
    with pytest.raises(ValueError):queue.stage_artifacts(destination,"run",job,value)


def build_fixture(tmp_path,monkeypatch,checkpoint_mode="pretrained"):
    import paired_comparison as paired
    import weight_arrangement_control as control
    from scripts import validate_weight_arrangement as validator
    from test_weight_arrangement_validation import decision
    validation,_=decision()
    native=write(tmp_path/"native.json",{"structural":"fixture"});validation["native_registry_sha256"]=file_hash(native)
    checkpoint=tmp_path/"snapshots/revision/mp_rank_00_model_states.pt"
    checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b"fixed synthetic weights")
    banks=tmp_path/"banks";sources=tmp_path/"sources"
    metrics=write(sources/"one/metrics.jsonl",{"synthetic":"source export"})
    configuration=dict(protocol_sha256="source-collection")
    write(Path(str(metrics)+".run")/"manifest.json",dict(configuration=configuration,configuration_sha256=object_hash(configuration)))
    entries=[dict(episode=ep,episode_id=f"ep-{ep}",policy_call_idx=call,context_id=f"ctx-{ep}-{call}",
                  status="source_not_reached" if (ep,call)==(1,12) else "available") for ep in (0,1) for call in (0,12)]
    bank=dict(task="PickCube-v1",model="1b",pipeline=dict(checkpoint_mode=checkpoint_mode,
        checkpoint=dict(sha256=file_hash(checkpoint),filename=checkpoint.name)),
        selection=dict(rule="exact_collector_prespecified_call_indices"),contexts=entries,
        source_configuration_sha256=object_hash(configuration),source_metrics_sha256=file_hash(metrics))
    bank_path=write(banks/"one.json",bank)
    protocol=dict(recorded_before_execution=True,decision_id="E05-fixture",protocol_version=1,validation=validation,
        source_collection_protocol_sha256="source-collection",strata=[dict(id="one",task=bank["task"],model=bank["model"],
            checkpoint_mode=checkpoint_mode,checkpoint_sha256=file_hash(checkpoint),checkpoint_identity=deepcopy(bank["pipeline"]["checkpoint"]),bank_sha256=file_hash(bank_path))],
        expected_population=dict(strata=1,episodes=2,planned_contexts=4,available_contexts=3,planned_cells=48))
    protocol_path=write(tmp_path/"protocol.json",protocol)
    args=SimpleNamespace(protocol=protocol_path,protocol_sha256=file_hash(protocol_path),native_registry=native,
        bank_directory=banks,source_root=sources,lang_dir=tmp_path/"language",output=tmp_path/"built",
        checkpoint_path=checkpoint if checkpoint_mode=="authors" else None,retry_queue=None,retry_amendment=None)
    monkeypatch.setattr(paired,"make_bank",lambda *a,**k:deepcopy(bank))
    monkeypatch.setattr(control,"structure_from_native_registry",lambda *a:{"parameter":"structure"})
    return args,protocol,bank


def test_builder_groups_all_calls_per_episode_and_retains_unavailable_member(tmp_path,monkeypatch):
    args,protocol,bank=build_fixture(tmp_path,monkeypatch)
    queue.build(args)
    result=strict_json(args.output/"queue.json")
    assert result["population"]==protocol["expected_population"]
    assert len(result["jobs"])==2
    assert all([e["policy_call_idx"] for e in job["contexts"]]==[0,12] for job in result["jobs"])
    assert result["jobs"][1]["contexts"][1]["status"]=="source_not_reached"
    assert file_hash(args.output/"banks/one.json")==file_hash(args.bank_directory/"one.json")
    decisions=[strict_json(job["decision_file"]) for job in result["jobs"]]
    assert all(d["parameter_seed_namespace"]==protocol["validation"]["parameter_seed_namespace"] for d in decisions)
    assert all(d["parameter_master_seed"]==protocol["validation"]["parameter_master_seed"] for d in decisions)
    assert all(d["parameter_draws"]==2 and len(d["contexts"])==2 for d in decisions)
    assert not list((args.output/"claims").iterdir())
    with pytest.raises(FileExistsError):queue.build(args)


def test_builder_preserves_snapshot_filename_and_verifies_checkpoint_bytes(tmp_path,monkeypatch):
    args,_,_=build_fixture(tmp_path,monkeypatch,"authors")
    snapshot=args.checkpoint_path;original=Path.resolve
    monkeypatch.setattr(Path,"resolve",lambda p,*a,**k:tmp_path/"blobs/opaque-hash" if p==snapshot else original(p,*a,**k))
    queue.build(args)
    value=strict_json(args.output/"queue.json");job=value["jobs"][0]
    assert job["checkpoint_path"]==str(snapshot.absolute())
    assert Path(job["checkpoint_path"]).name=="mp_rank_00_model_states.pt"
    snapshot.write_bytes(b"changed weights")
    with pytest.raises(ValueError,match="checkpoint bytes changed"):queue.verify_inputs(value,job)


@pytest.mark.parametrize("mutation",["bank","counts","seed","draws","checkpoint_mode","source_protocol"])
def test_builder_rejects_unbound_population_or_protocol_changes(tmp_path,monkeypatch,mutation):
    args,protocol,_=build_fixture(tmp_path,monkeypatch)
    if mutation=="bank":(args.bank_directory/"one.json").write_bytes(b"{}")
    elif mutation=="counts":protocol["expected_population"]["available_contexts"]=4
    elif mutation=="seed":del protocol["validation"]["parameter_master_seed"]
    elif mutation=="draws":protocol["validation"]["parameter_draws"]=1
    elif mutation=="checkpoint_mode":protocol["strata"][0]["checkpoint_mode"]="authors"
    elif mutation=="source_protocol":protocol["source_collection_protocol_sha256"]="other"
    write(args.protocol,protocol);args.protocol_sha256=file_hash(args.protocol)
    with pytest.raises((ValueError,KeyError)):queue.build(args)
    assert not (args.output/"queue.json").exists()


def retry_fixture(tmp_path,monkeypatch,status="failure"):
    root,args,value=queue_fixture(tmp_path,monkeypatch,count=2)
    job=value["jobs"][0]
    if status!="never_claimed":
        write(root/"claims"/(job["job_id"]+".json"),{"prior":"claim"})
    if status in {"failure","success","malformed"}:
        record=dict(job_id=job["job_id"],queue_sha256=file_hash(args.queue),returncode=3 if status=="failure" else 0 if status=="success" else None)
        write(root/"completion"/(job["job_id"]+".json"),record)
    amendment=write(tmp_path/"amendment.json",dict(recorded_before_retry=True,reason="Preserved setup error; repair execution only",
        parent_queue_sha256=file_hash(args.queue),protocol_sha256=value["protocol_sha256"],replacement_implementation_sha256=value["source_sha256"],replaces_jobs=[job["job_id"]]))
    retry_args=SimpleNamespace(retry_queue=args.queue,retry_amendment=amendment)
    return root,value,retry_args


@pytest.mark.parametrize("status",["failure","never_claimed"])
def test_explicit_retry_preserves_original_attempt_and_future_lineage(tmp_path,monkeypatch,status):
    root,value,args=retry_fixture(tmp_path,monkeypatch,status)
    jobs,lineage=queue.retry_membership(args,value["jobs"],value["protocol_sha256"],value["source_sha256"])
    assert len(jobs)==1 and lineage["unreplaced_parent_jobs"]==["fixture-e001"]
    assert lineage["prior_attempts"][0]["status"]==("sealed_failure" if status=="failure" else "never_claimed")
    retry={**value,"retry_lineage":lineage};queue.verify_retry_lineage(retry)
    if status=="failure":(root/"completion/fixture-e000.json").write_bytes(b"changed")
    else:write(root/"claims/fixture-e000.json",{"competing":"original worker"})
    with pytest.raises(ValueError):queue.verify_retry_lineage(retry)


@pytest.mark.parametrize("status",["success","unsealed","malformed"])
def test_retry_cannot_replace_success_unsealed_claim_or_unknown_outcome(tmp_path,monkeypatch,status):
    _,value,args=retry_fixture(tmp_path,monkeypatch,status)
    with pytest.raises(ValueError):queue.retry_membership(args,value["jobs"],value["protocol_sha256"],value["source_sha256"])


def test_retry_cannot_change_scientific_criteria_or_draw_seeds(tmp_path,monkeypatch):
    _,value,args=retry_fixture(tmp_path,monkeypatch)
    jobs=deepcopy(value["jobs"]);changed=strict_json(jobs[0]["decision_file"]);changed["parameter_master_seed"]+=1
    new=write(tmp_path/"new-decision.json",changed);jobs[0].update(decision_file=str(new),decision_sha256=file_hash(new))
    with pytest.raises(ValueError,match="criteria, seeds, draws or scope"):
        queue.retry_membership(args,jobs,value["protocol_sha256"],value["source_sha256"])


def test_retry_authenticates_retained_failed_logs_and_raw_files(tmp_path,monkeypatch):
    root,value,args=retry_fixture(tmp_path,monkeypatch)
    raw=root/"results/fixture-e000/partial-map.pt";raw.parent.mkdir(parents=True);raw.write_bytes(b"failed raw bytes")
    completion_path=root/"completion/fixture-e000.json";record=strict_json(completion_path)
    record["attempt_files"]=queue.attempt_files(root,"fixture-e000");write(completion_path,record)
    _,lineage=queue.retry_membership(args,value["jobs"],value["protocol_sha256"],value["source_sha256"])
    raw.write_bytes(b"changed failed bytes")
    with pytest.raises(ValueError,match="attempt file changed"):queue.verify_retry_lineage({**value,"retry_lineage":lineage})
