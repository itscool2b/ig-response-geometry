"""Freeze and execute episode-grouped E05 numerical jobs under a locked protocol.

Source preparation and candidate validation are separate processes. Every call
of an episode shares its parameter draws. Failed attempts are immutable and
require a new queue with an explicit retry amendment; this tool never launches
a job during queue construction or grants a numerical approval.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiment_io import canonical_json,file_hash,strict_json
from scripts.run_probe_queue import collection_protocol_hash,exclusive_json


def source_hashes():
    from scripts.validate_weight_arrangement import implementation_hashes
    extra=("scripts/run_probe_queue.py","scripts/run_weight_arrangement_queue.py")
    return {**implementation_hashes(),**{name:file_hash(ROOT/name) for name in extra}}


def verify_sources(queue):
    if source_hashes()!=queue["source_sha256"]:
        raise ValueError("E05 queue implementation changed after freeze")


def verify_inputs(queue,job):
    verify_retry_lineage(queue)
    fields=((queue,"protocol_path","protocol_sha256"),(queue,"native_registry_path","native_registry_sha256"),
        (job,"decision_file","decision_sha256"),(job,"bank_file","bank_sha256"),
        (job,"metrics_file","metrics_sha256"),(job,"source_manifest_file","source_manifest_sha256"))
    for value,path_key,hash_key in fields:
        if file_hash(value[path_key])!=value[hash_key]:
            raise ValueError("E05 frozen input changed: "+path_key)
    if job.get("checkpoint_path") and file_hash(job["checkpoint_path"])!=job["checkpoint_sha256"]:
        raise ValueError("E05 explicit checkpoint bytes changed")


def verify_retry_lineage(queue):
    """Keep the predecessor evidence fixed throughout a replacement attempt."""
    lineage=queue.get("retry_lineage")
    if not lineage:return
    for path_key,hash_key in (("parent_queue_path","parent_queue_sha256"),("amendment_path","amendment_sha256")):
        if file_hash(lineage[path_key])!=lineage[hash_key]:
            raise ValueError("E05 retry lineage changed: "+path_key)
    parent_root=Path(lineage["parent_queue_path"]).parent
    for attempt in lineage["prior_attempts"]:
        if attempt["status"]=="sealed_failure":
            if file_hash(attempt["completion_path"])!=attempt["completion_sha256"]:
                raise ValueError("Prior E05 failed completion changed")
            verify_attempt_files(strict_json(attempt["completion_path"]))
        elif attempt["status"]=="never_claimed":
            if any((parent_root/folder/(attempt["job_id"]+".json")).exists() for folder in ("claims","completion")):
                raise ValueError("Previously unclaimed E05 job now has a competing original attempt")
        else:raise ValueError("Unknown E05 retry attempt status")


def attempt_files(root,job_id):
    """Bind logs, claims and partial raw files even when a subprocess fails."""
    root=Path(root).resolve()
    paths=[root/"claims"/(job_id+".json")]
    paths.extend(root/"logs"/(job_id+"-"+stage+".log") for stage in ("prepare","run"))
    for folder in ("prepared","results"):
        paths.extend(path for path in (root/folder/job_id).rglob("*") if path.is_file())
    result=[]
    for path in sorted(paths):
        if not path.exists():continue
        if not path.resolve().is_relative_to(root):raise ValueError("E05 attempt artifact escapes its queue directory")
        result.append(dict(path=str(path),sha256=file_hash(path)))
    return result


def verify_attempt_files(completion):
    for item in completion.get("attempt_files",[]):
        if file_hash(item["path"])!=item["sha256"]:raise ValueError("E05 retained attempt file changed")


def stage_artifacts(destination,stage,job,queue):
    """Authenticate all stage files and the expected episode/draw population."""
    destination=Path(destination).resolve()
    seal=destination/"completion.json"
    completion=strict_json(seal)
    expected="complete_source_preparation" if stage=="prepare" else "complete_diagnostics_not_approval"
    if completion.get("status")!=expected or "report.json" not in completion.get("artifacts_sha256",{}):
        raise ValueError("E05 stage did not seal its expected report")
    declared=completion["artifacts_sha256"]
    observed={str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file() and path!=seal}
    if set(declared)!=observed:
        raise ValueError("E05 completed stage artifact membership differs")
    artifacts=[]
    for name,digest in declared.items():
        path=(destination/name).resolve()
        if not path.is_relative_to(destination) or file_hash(path)!=digest:
            raise ValueError("E05 completed stage artifact path or hash differs")
        artifacts.append(dict(path=str(path),sha256=digest))
    report=strict_json(destination/"report.json")
    for key,expected_value in (("decision_sha256",job["decision_sha256"]),("bank_sha256",job["bank_sha256"]),
        ("source_metrics_sha256",job["metrics_sha256"]),("native_registry_sha256",queue["native_registry_sha256"])):
        if report.get(key)!=expected_value:raise ValueError("E05 stage identity differs: "+key)
    if report.get("implementation_sha256")!=queue["validation_implementation_sha256"]:
        raise ValueError("E05 stage implementation identity differs")
    if report.get("status")!=expected:raise ValueError("E05 report status differs from its completion")
    if stage=="prepare":
        if report.get("kind")!="fp32_probe_source_preparation":raise ValueError("Wrong E05 preparation kind")
        actual=[{key:item[key] for key in entry} for entry,item in zip(job["contexts"],report.get("contexts",[]))]
        if len(actual)!=len(job["contexts"]) or len(report["contexts"])!=len(actual) or actual!=job["contexts"]:
            raise ValueError("E05 preparation episode/call population differs")
    else:
        if report.get("kind")!="weight_arrangement_numerical_validation" or report.get("planned_roster")!=job["contexts"]:
            raise ValueError("E05 candidate planned episode/call population differs")
        if report.get("decision")!=strict_json(job["decision_file"]) or report.get("structure_sha256")!=job["structure_sha256"]:
            raise ValueError("E05 candidate decision or parameter structure differs")
        if report.get("parameter_draws")!=job["parameter_draws"]:
            raise ValueError("E05 candidate draw count differs")
        expected_cells={(entry["episode_id"],entry["policy_call_idx"],draw,modality,target)
            for entry in job["contexts"] for draw in range(job["parameter_draws"])
            for modality in ("vision","language","state") for target in ("Q","L2")}
        keys=[];lookup={(e["episode_id"],e["policy_call_idx"]):e for e in job["contexts"]}
        for item in report.get("contexts",[]):
            entry=lookup.get((item["episode_id"],item["policy_call_idx"]))
            if entry is None or item.get("context_id")!=entry["context_id"]:
                raise ValueError("E05 candidate context identity differs")
            for arm in item.get("numerics",[]):
                keys.append((item["episode_id"],item["policy_call_idx"],item["draw_index"],arm["modality"],arm["target"]))
                allowed={"source_unavailable"} if entry["status"]!="available" else {"complete_diagnostics_not_approval","numerical_failure"}
                if arm["status"] not in allowed:raise ValueError("E05 candidate retained an unfinished or misclassified arm")
        if len(keys)!=len(expected_cells) or set(keys)!=expected_cells or report.get("planned_context_draw_target_modality_cells")!=len(expected_cells):
            raise ValueError("E05 candidate lost or duplicated a planned cell")
    artifacts.append(dict(path=str(seal),sha256=file_hash(seal)))
    return artifacts


def retry_membership(args,jobs,protocol_hash,implementation):
    """Only a named sealed failure or never-claimed job can be replaced."""
    if not args.retry_queue and not args.retry_amendment:return jobs,None
    if not args.retry_queue or not args.retry_amendment:raise ValueError("Retry queue and prospective amendment must be supplied together")
    parent_path=Path(args.retry_queue).absolute();parent=strict_json(parent_path)
    if parent.get("kind")!="e05_episode_numerical_queue":raise ValueError("E05 retry parent has the wrong queue kind")
    amendment=strict_json(args.retry_amendment)
    if (amendment.get("recorded_before_retry") is not True or not amendment.get("reason") or
        amendment.get("parent_queue_sha256")!=file_hash(parent_path) or amendment.get("protocol_sha256")!=protocol_hash or
        amendment.get("replacement_implementation_sha256")!=implementation):
        raise ValueError("E05 retry amendment does not bind the prior queue and new implementation/protocol")
    replacements=amendment.get("replaces_jobs",[])
    old={job["job_id"]:job for job in parent["jobs"]};new={job["job_id"]:job for job in jobs}
    if not replacements or len(set(replacements))!=len(replacements) or not set(replacements)<=old.keys() or not set(replacements)<=new.keys():
        raise ValueError("E05 retry job membership differs")
    attempts=[]
    for job_id in replacements:
        previous,replacement=old[job_id],new[job_id]
        if any(previous[key]!=replacement[key] for key in ("contexts","bank_sha256","metrics_sha256","source_manifest_sha256","parameter_draws","structure_sha256")):
            raise ValueError("E05 retry changes the planned scientific population or structure")
        old_decision,new_decision=strict_json(previous["decision_file"]),strict_json(replacement["decision_file"])
        if file_hash(previous["decision_file"])!=previous["decision_sha256"]:
            raise ValueError("Prior E05 decision changed")
        for value in (old_decision,new_decision):
            for field in ("parent_protocol_sha256","implementation_sha256"):value.pop(field,None)
        if old_decision!=new_decision:raise ValueError("E05 execution retry changes numerical criteria, seeds, draws or scope")
        completion=parent_path.parent/"completion"/(job_id+".json")
        claim=parent_path.parent/"claims"/(job_id+".json")
        if completion.exists():
            record=strict_json(completion)
            if record.get("job_id")!=job_id or record.get("queue_sha256")!=file_hash(parent_path):
                raise ValueError("Prior E05 attempt identity differs")
            if type(record.get("returncode")) is not int:raise ValueError("Prior E05 attempt lacks an explicit integer outcome")
            if record.get("returncode")==0:raise ValueError("An E05 retry cannot replace completed diagnostics")
            verify_attempt_files(record)
            attempts.append(dict(job_id=job_id,status="sealed_failure",completion_path=str(completion),completion_sha256=file_hash(completion)))
        elif claim.exists():
            raise ValueError("An unsealed E05 claim requires separate interrupted-attempt diagnosis")
        else:
            attempts.append(dict(job_id=job_id,status="never_claimed"))
    lineage=dict(parent_queue_path=str(parent_path),parent_queue_sha256=file_hash(parent_path),
        amendment_path=str(Path(args.retry_amendment).absolute()),amendment_sha256=file_hash(args.retry_amendment),prior_attempts=attempts,
        unreplaced_parent_jobs=sorted(set(old)-set(replacements)))
    return [job for job in jobs if job["job_id"] in replacements],lineage


def build(args):
    import paired_comparison as paired
    import weight_arrangement_control as control
    from scripts import validate_weight_arrangement as validator
    from experiment_io import object_hash
    protocol=strict_json(args.protocol)
    if file_hash(args.protocol)!=args.protocol_sha256 or protocol.get("recorded_before_execution") is not True:
        raise ValueError("A trusted prospective E05 protocol is required")
    native_hash=file_hash(args.native_registry);native=strict_json(args.native_registry)
    if protocol["validation"]["native_registry_sha256"]!=native_hash:
        raise ValueError("Native registry differs from the numerical protocol")
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    for name in ("banks","decisions","claims","logs","completion","prepared","results"):(output/name).mkdir()
    jobs=[];strata=set();context_count=available=0
    for stratum in protocol["strata"]:
        identifier=stratum["id"]
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*",identifier) or identifier in strata:raise ValueError("Invalid or repeated E05 stratum ID")
        strata.add(identifier)
        original_bank=args.bank_directory/(identifier+".json")
        if file_hash(original_bank)!=stratum["bank_sha256"]:raise ValueError("Existing E05 bank differs from its protocol digest")
        bank=strict_json(original_bank);metrics=args.source_root.absolute()/identifier/"metrics.jsonl"
        selection=bank["selection"]
        rebuilt=paired.make_bank(metrics,None if selection["rule"]=="exact_collector_prespecified_call_indices" else selection)
        if rebuilt!=bank:raise ValueError("Existing E05 bank differs from authenticated collection")
        if (bank["task"]!=stratum["task"] or bank["model"]!=stratum["model"] or
            bank["pipeline"]["checkpoint_mode"]!=stratum["checkpoint_mode"] or
            bank["pipeline"]["checkpoint"]["sha256"]!=stratum["checkpoint_sha256"] or
            bank["pipeline"]["checkpoint"]!=stratum["checkpoint_identity"]):
            raise ValueError("E05 task/model/checkpoint stratum identity differs")
        if stratum["checkpoint_mode"] not in {"pretrained","authors"}:
            raise ValueError("E05 episode queue supports the declared pretrained or authors checkpoints only")
        if collection_protocol_hash(metrics,bank)!=protocol["source_collection_protocol_sha256"]:
            raise ValueError("E05 source collection protocol differs")
        bank_file=output/"banks"/(identifier+".json")
        with bank_file.open("xb") as stream:stream.write(original_bank.read_bytes())
        if file_hash(bank_file)!=stratum["bank_sha256"]:raise ValueError("Existing E05 bank changed while copying")
        structure=control.structure_from_native_registry(native,bank["model"])
        episodes=defaultdict(list)
        for entry in bank["contexts"]:episodes[entry["episode_id"]].append(entry)
        for episode_id,entries in sorted(episodes.items()):
            entries=sorted(entries,key=lambda entry:entry["policy_call_idx"])
            episode=entries[0]["episode"]
            if any(entry["episode"]!=episode for entry in entries):raise ValueError("Episode identity spans different collector episodes")
            job_id=f"{identifier}-e{episode:03d}"
            decision={**protocol["validation"],"decision_id":protocol["decision_id"],"protocol_version":protocol["protocol_version"],
                "recorded_before_execution":True,"parent_protocol_sha256":args.protocol_sha256,"stratum_id":identifier,
                "bank_sha256":file_hash(bank_file),"checkpoint_mode":stratum["checkpoint_mode"],
                "checkpoint_identity":bank["pipeline"]["checkpoint"],
                "contexts":[dict(episode=entry["episode"],policy_call_idx=entry["policy_call_idx"]) for entry in entries]}
            validator.validate_decision(decision,bank,file_hash(bank_file),native_hash)
            decision_file=output/"decisions"/(job_id+".json");exclusive_json(decision_file,decision)
            common=["--metrics",str(metrics),"--bank",str(bank_file),"--decision-file",str(decision_file),
                "--decision-sha256",file_hash(decision_file),"--native-registry",str(args.native_registry.absolute()),
                "--lang-dir",str(args.lang_dir.absolute())]
            checkpoint_path=None
            if stratum["checkpoint_mode"]!="pretrained":
                if args.checkpoint_path is None:raise ValueError("An explicit named authors checkpoint is required")
                checkpoint_path=str(args.checkpoint_path.absolute())
                if file_hash(checkpoint_path)!=stratum["checkpoint_sha256"]:raise ValueError("Explicit authors checkpoint bytes differ")
                if Path(checkpoint_path).name!=stratum["checkpoint_identity"]["filename"]:
                    raise ValueError("Explicit authors checkpoint filename differs from its recorded identity")
                common.extend(["--checkpoint-path",checkpoint_path])
            source_manifest=Path(str(metrics)+".run")/"manifest.json"
            jobs.append(dict(job_id=job_id,stratum_id=identifier,episode_id=episode_id,episode=episode,contexts=entries,
                parameter_draws=decision["parameter_draws"],structure_sha256=object_hash(structure),
                metrics_file=str(metrics),metrics_sha256=file_hash(metrics),source_manifest_file=str(source_manifest),source_manifest_sha256=file_hash(source_manifest),
                bank_file=str(bank_file),bank_sha256=file_hash(bank_file),decision_file=str(decision_file),decision_sha256=file_hash(decision_file),
                checkpoint_path=checkpoint_path,checkpoint_sha256=stratum["checkpoint_sha256"],common_arguments=common))
            context_count+=len(entries);available+=sum(entry["status"]=="available" for entry in entries)
    population=dict(strata=len(strata),episodes=len(jobs),planned_contexts=context_count,available_contexts=available,
                    planned_cells=context_count*protocol["validation"]["parameter_draws"]*6)
    if population!=protocol["expected_population"]:raise ValueError("E05 bank population differs from the declared counts")
    implementation=source_hashes()
    jobs,lineage=retry_membership(args,jobs,args.protocol_sha256,implementation)
    queue=dict(kind="e05_episode_numerical_queue",schema_version=1,protocol_path=str(args.protocol.absolute()),protocol_sha256=args.protocol_sha256,
        native_registry_path=str(args.native_registry.absolute()),native_registry_sha256=native_hash,
        source_sha256=implementation,validation_implementation_sha256=validator.implementation_hashes(),
        population=population,jobs=jobs,retry_lineage=lineage,scope="Fixed ladders; every episode draw coherent across selected calls; diagnostic completion is not approval")
    for job in jobs:verify_inputs(queue,job)
    verify_sources(queue)
    exclusive_json(output/"queue.json",queue)
    print(dict(jobs=len(jobs),population=population,queue_sha256=file_hash(output/"queue.json")),flush=True)


def verify_previous(path,queue_hash,queue,job):
    completion=strict_json(path)
    if completion.get("queue_sha256")!=queue_hash or completion.get("job_id")!=job["job_id"] or type(completion.get("returncode")) is not int or completion["returncode"]!=0:
        raise ValueError("Failed or inconsistent E05 attempt requires an explicit new retry queue")
    if [stage["stage"] for stage in completion.get("stages",[])]!=["prepare","run"]:
        raise ValueError("E05 completed attempt lacks both process stages")
    if any(type(stage.get("returncode")) is not int or stage["returncode"]!=0 for stage in completion["stages"]):
        raise ValueError("E05 completed attempt contains an unsuccessful process stage")
    verify_attempt_files(completion)
    for item in completion["artifacts"]:
        if file_hash(item["path"])!=item["sha256"]:raise ValueError("Completed E05 artifact changed")
    actual=[]
    for stage,folder in (("prepare","prepared"),("run","results")):
        actual.extend(stage_artifacts(path.parent.parent/folder/job["job_id"],stage,job,queue))
    if sorted(actual,key=lambda item:item["path"])!=sorted(completion["artifacts"],key=lambda item:item["path"]):
        raise ValueError("E05 completed attempt artifact inventory differs from its stages")


def worker(args):
    if args.workers<1 or not 0<=args.index<args.workers:raise ValueError("Invalid E05 worker partition")
    queue_path=args.queue.resolve();queue=strict_json(queue_path);queue_hash=file_hash(queue_path)
    if queue_hash!=args.queue_sha256 or queue.get("kind")!="e05_episode_numerical_queue":raise ValueError("E05 queue differs from its trusted identity")
    verify_sources(queue)
    def verify_job(job):
        if file_hash(queue_path)!=queue_hash:raise ValueError("E05 queue changed during execution")
        verify_inputs(queue,job);verify_sources(queue)
    for index,job in enumerate(queue["jobs"]):
        if index%args.workers!=args.index:continue
        verify_job(job)
        completion_path=queue_path.parent/"completion"/(job["job_id"]+".json")
        if completion_path.exists():
            verify_previous(completion_path,queue_hash,queue,job)
            continue
        claim=dict(job_id=job["job_id"],queue_sha256=queue_hash,partition=args.index,partitions=args.workers,
                   hostname=platform.node(),started_unix=time.time())
        exclusive_json(queue_path.parent/"claims"/(job["job_id"]+".json"),claim)
        started=time.perf_counter();stages=[];artifacts=[];failure=None;returncode=125
        prepared=queue_path.parent/"prepared"/job["job_id"]
        try:
            for stage,destination in (("prepare",prepared),("run",queue_path.parent/"results"/job["job_id"])):
                verify_job(job)
                command=[sys.executable,str(ROOT/"scripts/validate_weight_arrangement.py"),stage,*job["common_arguments"],"--out",str(destination)]
                if stage=="run":command.extend(["--prepared",str(prepared),"--prepared-sha256",file_hash(prepared/"completion.json")])
                log=queue_path.parent/"logs"/(job["job_id"]+"-"+stage+".log")
                print(f"E05 worker {args.index}: {index+1}/{len(queue['jobs'])} {job['job_id']} {stage}",flush=True)
                stage_record=dict(stage=stage,log=str(log),returncode=None);stages.append(stage_record)
                with log.open("xb") as stream:
                    result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
                stage_record["returncode"]=result.returncode;returncode=result.returncode
                verify_job(job)
                if returncode:break
                artifacts.extend(stage_artifacts(destination,stage,job,queue))
            verify_job(job)
        except Exception as error:
            returncode=returncode or 125
            failure=dict(exception_type=type(error).__name__,reason=str(error))
        retained_files=attempt_files(queue_path.parent,job["job_id"])
        exclusive_json(completion_path,{**claim,"returncode":returncode,"stages":stages,"artifacts":artifacts,
            "attempt_files":retained_files,"failure":failure,"elapsed_seconds":time.perf_counter()-started,"ended_unix":time.time()})
        if returncode:raise RuntimeError("E05 numerical job failed; attempt retained and worker stopped without automatic retry")
    print(f"E05 worker {args.index}: partition diagnostics complete",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="mode",required=True)
    build_parser=sub.add_parser("build")
    for name in ("protocol","bank-directory","source-root","native-registry","lang-dir","output"):
        build_parser.add_argument("--"+name,type=Path,required=True)
    build_parser.add_argument("--protocol-sha256",required=True)
    for name in ("checkpoint-path","retry-queue","retry-amendment"):build_parser.add_argument("--"+name,type=Path)
    worker_parser=sub.add_parser("worker")
    worker_parser.add_argument("--queue",type=Path,required=True);worker_parser.add_argument("--queue-sha256",required=True)
    worker_parser.add_argument("--index",type=int,required=True);worker_parser.add_argument("--workers",type=int,required=True)
    args=parser.parse_args();(build if args.mode=="build" else worker)(args)


if __name__=="__main__":main()
