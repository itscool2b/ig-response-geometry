"""Bounded matched pilot-rule characterization. This command cannot approve a study.

The operational deadline terminates unfinished worker processes, records their
identities, and leaves the planned denominators unchanged. A later resume uses
the same seeds. Only complete matched pairs enter the authenticated ledger.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"

import argparse
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
from filelock import FileLock

import paired_study_analysis as study
import pilot_selection as selector
import pilot_selection_simulation as simulation
import pilot_rule_characterization as matched
from scripts.validate_pilot_selection import exact_bounds
from experiment_io import atomic_bytes,canonical_json,file_hash,object_hash,strict_json


SOURCES=("paired_study_analysis.py","pilot_selection.py","pilot_selection_simulation.py","pilot_scale_bounds.py",
    "pilot_rule_characterization.py","scripts/characterize_pilot_rules.py","scripts/validate_pilot_selection.py","experiment_io.py")
DIAGNOSTICS=(*matched.INTERNAL_DIAGNOSTICS,*matched.DIRECT_COMPONENTS,"future_interval_noncoverage",
    "legacy_internal_or_direct_failure")
CONTRACT="direct_outcome_characterization_with_indispensable_separate_exact_count_coverage_gate_v2"


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    return {name:file_hash(root/name) for name in SOURCES}


def validate_plan(plan,design,rules):
    if plan.get("schema_version")!=2 or plan.get("kind")!="matched_pilot_rule_characterization_plan":
        raise ValueError("Unknown matched characterization plan")
    if plan.get("acceptance")!=CONTRACT or plan.get("can_approve") is not False:
        raise ValueError("Characterization never approves a study")
    if plan.get("source_sha256")!=source_hashes():raise ValueError("Characterization source changed")
    if plan.get("design_object_sha256")!=object_hash(design) or plan.get("rules_object_sha256")!=object_hash(rules):
        raise ValueError("The frozen design or rule comparison changed")
    if len(rules)<2 or len({r["id"] for r in rules})!=len(rules):raise ValueError("At least two unique prespecified rules are required")
    for rule in rules:selector.validate_protocol(rule["protocol"],design)
    matched.union_candidates(rules)
    if plan.get("regimes")!=list(simulation.REGIMES) or plan.get("regime_definitions_sha256")!=object_hash(simulation.REGIMES):
        raise ValueError("All prespecified regimes must be retained")
    study.integer(plan["seed"],"matched validation seed",0)
    study.integer(plan["trials_per_regime"],"matched trials",1)
    study.integer(plan["workers"],"bounded workers",1)
    if not 0<study.finite(plan["wall_seconds"],"wall cap")<=86400:raise ValueError("An explicit bounded wall cap is required")
    if not 0<study.finite(plan["simulation_confidence_alpha"],"simulation confidence")<1:raise ValueError("Invalid confidence budget")
    if set(plan["risk_targets"])!={"unconditional_direct_unsafe","conditional_direct_unsafe_given_selected"}:
        raise ValueError("Both direct risks must be declared")
    if any(not 0<study.finite(v,"risk target")<1 for v in plan["risk_targets"].values()):raise ValueError("Invalid risk target")
    minimum=plan["minimum_resolution_probability"]
    if set(minimum)!=set(simulation.REGIMES) or any(not 0<=study.finite(v,"resolution")<1 for v in minimum.values()):
        raise ValueError("Every regime needs a resolution target")
    if not any(v>0 for v in minimum.values()):raise ValueError("Always inconclusive cannot be the whole reference design")
    if plan.get("exact_count_coverage_gate_required") is not True or plan.get("arbitrary_tail_guarantee") is not False or not plan.get("domain_statement"):
        raise ValueError("Coverage and synthetic-domain limitations must be explicit")
    settings=design["analysis"]
    for family in study.inferential_families(design):
        required=study.bootstrap_requirement(design["global_alpha"]*design["family_weights"][family],len(study.family_registry(design,family)),
            settings["min_tail_draws"],settings["tail_relative_mcse"])
        if settings["draws"]<required or settings["mc_repeats"]<2:raise ValueError("Use exact production interval precision")
    return plan


def summarize(plan,rules,rows):
    # Three directional claims for each rule/regime: risk upper, conditional
    # risk upper, resolution lower. Other sides and diagnostics are descriptive.
    local=plan["simulation_confidence_alpha"]/(len(rules)*len(plan["regimes"])*3)
    result={};complete=True
    for rule in rules:
        cells={}
        for regime in plan["regimes"]:
            pairs=[r for r in rows if r["regime"]==regime]
            values=[r["rules"][rule["id"]] for r in pairs];n=len(values)
            regime_complete=n==plan["trials_per_regime"]
            complete=complete and regime_complete
            selected=sum(v["selected"] for v in values);unsafe=sum(v["direct_unsafe_selection"] for v in values)
            _,upper=exact_bounds(unsafe,n,local);_,conditional_upper=exact_bounds(unsafe,selected,local)
            lower,_=exact_bounds(selected,n,local);target=plan["minimum_resolution_probability"][regime]
            cells[regime]=dict(trials=n,planned_trials=plan["trials_per_regime"],selected=selected,inconclusive=n-selected,
                fixed_regime_roster_complete=regime_complete,
                bound_interpretation=("fixed_planned_binomial_denominator" if regime_complete else
                    "descriptive_completed_pair_diagnostics_only; runtime_censoring_can_be_informative; no_valid_risk_or_resolution_inference"),
                direct_unsafe_count=unsafe,
                unconditional_direct_unsafe=dict(upper=upper,target=plan["risk_targets"]["unconditional_direct_unsafe"],
                    target_met=upper<=plan["risk_targets"]["unconditional_direct_unsafe"] if regime_complete else None),
                conditional_direct_unsafe_given_selected=dict(trials=selected,upper=conditional_upper,
                    target=plan["risk_targets"]["conditional_direct_unsafe_given_selected"],
                    status="unresolved_no_selected_trials" if selected==0 else "estimated",
                    target_met=(selected>0 and conditional_upper<=plan["risk_targets"]["conditional_direct_unsafe_given_selected"]) if regime_complete else None),
                resolution=dict(lower=lower,target=target,target_met=lower>=target if regime_complete else None),
                diagnostic_counts={key:sum(v[key] for v in values) for key in DIAGNOSTICS},
                interval_noncoverage_role="diagnostic; separate executed exact-count coverage calibration remains indispensable",
                selected_candidates={cid:sum(v["selected_candidate"]==cid for v in values) for cid in sorted({c["id"] for c in rule["protocol"]["candidates"]})})
        result[rule["id"]]=cells
    return dict(status="characterization_complete_not_approval" if complete else "characterization_incomplete_not_approval",
        can_approve=False,all_planned_pairs_complete=complete,rules=result,simulation_local_alpha=local,
        simultaneous_directional_claim_count=len(rules)*len(plan["regimes"])*3,
        exact_count_coverage_gate_required=True,
        conditional_risk_caveat="zero selected trials leave conditional risk unresolved; no epsilon or zero-risk substitution",
        limitation="fixed synthetic domain only; internal bound misses are diagnostic, not distribution-free guarantees",
        complete_matched_pairs=len(rows),completed_full_rule_trials=len(rows)*len(rules))


def _validate_row(row,identity,plan,rules):
    if row["identity_sha256"]!=identity or row["regime"] not in plan["regimes"] or not 0<=row["trial"]<plan["trials_per_regime"]:
        raise ValueError("Unexpected matched trial scope")
    if row["seed_sha256"]!=object_hash(dict(seed=plan["seed"],regime=row["regime"],trial=row["trial"])):
        raise ValueError("Matched trial stream changed")
    if set(row["rules"])!={r["id"] for r in rules}:raise ValueError("Incomplete matched rule result")
    for value in row["rules"].values():
        for field in (*DIAGNOSTICS,"selected","direct_unsafe_selection"):
            if type(value[field]) is not bool:raise ValueError("Invalid synthetic event")
        expected=value["selected"] and any(value[k] for k in matched.DIRECT_COMPONENTS)
        if value["direct_unsafe_selection"]!=expected:raise ValueError("Direct risk event changed")


def _load_rows(out,identity,plan,rules):
    import json
    path=out/"ledger.jsonl";rows=[];head="0"*64;seen=set()
    if not path.exists():return rows,head
    for line in path.read_text(encoding="utf-8").splitlines():
        record=json.loads(line);saved=record.pop("record_sha256")
        if object_hash(record)!=saved or record["previous_sha256"]!=head or record["identity_sha256"]!=identity:
            raise ValueError("Matched ledger changed")
        target=out/record["file"]
        if target.parent!=out or target.name!=record["file"] or file_hash(target)!=record["sha256"]:
            raise ValueError("Completed matched trial changed")
        row=strict_json(target);_validate_row(row,identity,plan,rules);key=(row["regime"],row["trial"])
        if key in seen:raise ValueError("Duplicate matched trial")
        seen.add(key);rows.append(row);head=saved
    return rows,head


def _worker(design,rules,plan,regime,trial,path):
    start=time.perf_counter()
    if source_hashes()!=plan["source_sha256"]:raise ValueError("Worker source changed before generation")
    row=matched.simulate_matched_trial(design,rules,regime,plan["seed"],trial)
    if source_hashes()!=plan["source_sha256"]:raise ValueError("Worker source changed during generation")
    row["worker_memory"]=process_peak_memory()
    row["worker_elapsed_seconds_before_serialization"]=time.perf_counter()-start
    atomic_bytes(path,canonical_json(row)+b"\n")


def process_peak_memory():
    """Read the actual current worker, never a venv launcher parent handle."""
    result=dict(pid=os.getpid(),parent_pid=os.getppid(),scope="process_lifetime_peak_at_completed_job_return",
        peak_resident_bytes=None,status="unavailable")
    try:
        if sys.platform=="win32":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                    *[(name,ctypes.c_size_t) for name in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]]
            kernel=ctypes.WinDLL("kernel32",use_last_error=True);psapi=ctypes.WinDLL("psapi",use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
            counters=Counters();counters.cb=ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            value=int(counters.PeakWorkingSetSize);method="GetProcessMemoryInfo(GetCurrentProcess).PeakWorkingSetSize"
        else:
            import resource
            value=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=="darwin" else 1024)
            method="getrusage(RUSAGE_SELF).ru_maxrss"
        if value<=0:raise ValueError("Nonpositive process peak resident memory")
        result.update(status="observed",peak_resident_bytes=value,method=method)
    except Exception as error:
        result["reason"]=type(error).__name__+": "+str(error)
    return result


def _append(out,row,identity,head):
    row["identity_sha256"]=identity
    name=f"trial_{row['regime']}_{row['trial']:07d}.json";path=out/name;content=canonical_json(row)+b"\n"
    if path.exists() and path.read_bytes()!=content:raise ValueError("Preserve differing uncommitted trial for review")
    atomic_bytes(path,content)
    record=dict(file=name,sha256=file_hash(path),previous_sha256=head,identity_sha256=identity);head=object_hash(record)
    with (out/"ledger.jsonl").open("ab") as stream:
        stream.write(canonical_json(dict(record,record_sha256=head))+b"\n");stream.flush();os.fsync(stream.fileno())
    return head


def run(design,rules,plan,out,*,resume=False):
    validate_plan(plan,design,rules);out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    identity=object_hash(dict(design=design,rules=rules,plan=plan));start=time.monotonic()
    with FileLock(str(out/"run.lock"),timeout=0):
        marker=out/"identity.json"
        if marker.exists():
            if not resume or strict_json(marker)["identity_sha256"]!=identity:raise ValueError("Resume requires identical frozen inputs")
        else:
            if resume or list(out.glob("trial_*.json")) or (out/"ledger.jsonl").exists():raise ValueError("Unexpected previous characterization")
            atomic_bytes(marker,canonical_json(dict(identity_sha256=identity,design=design,rules=rules,plan=plan))+b"\n")
        rows,head=_load_rows(out,identity,plan,rules);done={(r["regime"],r["trial"]) for r in rows}
        pending=[(r,t) for t in range(plan["trials_per_regime"]) for r in plan["regimes"] if (r,t) not in done]
        attempts=out/"attempts";attempts.mkdir(exist_ok=True);active=[];aborted=[];context=mp.get_context("spawn")
        timeout=False
        try:
            while pending or active:
                if source_hashes()!=plan["source_sha256"]:raise ValueError("Sources changed during characterization")
                # Harvest complete results before applying the deadline. A
                # completed atomic result is retained even at the boundary.
                for item in list(active):
                    process=item["process"]
                    if process.is_alive() and not item["path"].exists():continue
                    process.join(timeout=2)
                    if process.is_alive():process.terminate();process.join(timeout=2)
                    active.remove(item)
                    if process.exitcode!=0 and not item["path"].exists():raise RuntimeError(f"Matched worker failed: {item['regime']} trial {item['trial']}")
                    row=strict_json(item["path"]);row["identity_sha256"]=identity
                    _validate_row(row,identity,plan,rules)
                    key=(row["regime"],row["trial"])
                    if key!=(item["regime"],item["trial"]) or key in done:raise ValueError("Worker returned a different or duplicate scheduled trial")
                    head=_append(out,row,identity,head);rows.append(row);done.add(key)
                if time.monotonic()-start>=plan["wall_seconds"]:
                    timeout=bool(pending or active);break
                while pending and len(active)<plan["workers"]:
                    regime,trial=pending.pop(0);path=attempts/f"{regime}_{trial}_{uuid.uuid4().hex}.json"
                    process=context.Process(target=_worker,args=(design,rules,plan,regime,trial,path))
                    process.start();active.append(dict(process=process,regime=regime,trial=trial,path=path))
                if active:time.sleep(.05)
        finally:
            for item in active:
                process=item["process"]
                if process.is_alive():process.terminate()
                process.join(timeout=2)
                if process.is_alive():process.kill();process.join(timeout=2)
                aborted.append(dict(regime=item["regime"],trial=item["trial"],attempt_file=str(item["path"].relative_to(out)),
                    reason="operational_interruption_no_scientific_outcome",exitcode=process.exitcode))
            attempt_record=dict(identity_sha256=identity,source_sha256=source_hashes(),wall_limit_reached=timeout,
                elapsed_seconds=time.monotonic()-start,interrupted_trials=aborted,pending_pairs=len(pending))
            atomic_bytes(out/f"operation_{uuid.uuid4().hex}.json",canonical_json(attempt_record)+b"\n")
        if source_hashes()!=plan["source_sha256"]:raise ValueError("Sources changed before characterization completion")
        report=summarize(plan,rules,rows)
        report.update(identity_sha256=identity,source_sha256=source_hashes(),ledger_chain_sha256=head,
            ledger_sha256=file_hash(out/"ledger.jsonl") if (out/"ledger.jsonl").exists() else None,
            elapsed_seconds_this_run=time.monotonic()-start,wall_limit_reached=timeout,
            environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,blas_threads=1),
            mean_seconds_per_matched_pair=float(np.mean([r["elapsed_seconds"] for r in rows])) if rows else None)
        atomic_bytes(out/"characterization.json",canonical_json(report)+b"\n")
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("design","rules","plan"):
        parser.add_argument("--"+key,type=Path,required=True);parser.add_argument("--"+key+"-sha256",required=True)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();values={}
    for key in ("design","rules","plan"):
        path=getattr(args,key)
        if file_hash(path)!=getattr(args,key+"_sha256"):raise ValueError("Frozen file digest changed")
        values[key]=strict_json(path)
    report=run(values["design"],values["rules"],values["plan"],args.out,resume=args.resume)
    print(canonical_json(dict(status=report["status"],elapsed_seconds=report["elapsed_seconds_this_run"])).decode())


if __name__=="__main__":main()
