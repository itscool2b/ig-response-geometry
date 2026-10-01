"""Execute a frozen full pilot-rule simulation, without GPU or effect data.

All regimes and risk/resolution targets are declared before execution. A bounded
prefix is resumable but cannot be reported as approval. Empirical validation
applies only to the declared synthetic domain and the exact frozen procedure.
"""
from __future__ import annotations

import os
for _key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[_key]="1"

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import math
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
from scipy.stats import beta
from filelock import FileLock

import paired_study_analysis as study
import pilot_selection as selector
import pilot_selection_simulation as simulation
from experiment_io import atomic_bytes,canonical_json,file_hash,object_hash,strict_json


METRICS=("inspected_variance_bound_failure","scale_bound_failure","selected_variance_bound_failure",
    "completion_bound_failure","future_shortfall","future_precision_failure","future_population_precision_failure","future_mc_failure",
    "future_endpoint_mc_failure","selected_invalid")
SOURCES=("paired_study_analysis.py","pilot_selection.py","pilot_selection_simulation.py","pilot_scale_bounds.py",
    "scripts/validate_pilot_selection.py","experiment_io.py")


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    return {name:file_hash(root/name) for name in SOURCES}


def validate_plan(plan,design,protocol):
    selector.validate_protocol(protocol,design)
    if plan.get("schema_version")!=1 or plan.get("kind")!="full_pilot_rule_validation_plan":raise ValueError("Unknown full-rule validation plan")
    if plan.get("source_sha256")!=source_hashes():raise ValueError("Pilot validation sources changed")
    if plan.get("design_object_sha256")!=object_hash(design) or plan.get("selection_protocol_object_sha256")!=object_hash(protocol):
        raise ValueError("Full-rule validation changed its design or selection procedure")
    if plan.get("regimes")!=list(simulation.REGIMES) or plan.get("regime_definitions_sha256")!=object_hash(simulation.REGIMES):
        raise ValueError("Every prespecified regime must be retained")
    study.integer(plan["seed"],"independent validation seed",0);study.integer(plan["trials_per_regime"],"simulation trials",1)
    if not 0<study.finite(plan["simulation_confidence_alpha"],"simulation confidence")<1:raise ValueError("Invalid simulation confidence budget")
    if set(plan["maximum_risks"])!=set(METRICS):raise ValueError("All planning risk limits must be specified prospectively")
    if any(not 0<study.finite(value,"risk limit")<1 for value in plan["maximum_risks"].values()):raise ValueError("Invalid risk target")
    if set(plan["minimum_resolution_probability"])!=set(simulation.REGIMES):raise ValueError("Every regime needs an explicit resolution target")
    if any(not 0<=study.finite(p,"resolution target")<1 for p in plan["minimum_resolution_probability"].values()):raise ValueError("Invalid resolution target")
    if all(p==0 for p in plan["minimum_resolution_probability"].values()):raise ValueError("An always-inconclusive rule cannot be the entire approved design")
    if plan.get("acceptance")!="all_regime_simultaneous_risk_and_resolution_bounds":raise ValueError("No favorable-regime selection is allowed")
    settings=design["analysis"]
    for family in study.inferential_families(design):
        required=study.bootstrap_requirement(design["global_alpha"]*design["family_weights"][family],len(study.family_registry(design,family)),
            settings["min_tail_draws"],settings["tail_relative_mcse"])
        if settings["draws"]<required or settings["mc_repeats"]<2:raise ValueError("Future intervals must use the exact production bootstrap precision")
    if plan.get("domain_statement") is None or plan.get("arbitrary_tail_guarantee") is not False:
        raise ValueError("Synthetic-domain limitations must be explicit")
    return plan


def exact_bounds(successes,trials,alpha):
    if trials==0:return 0.,1.
    return (0. if successes==0 else float(beta.ppf(alpha,successes,trials-successes+1)),
        1. if successes==trials else float(beta.ppf(1-alpha,successes+1,trials-successes)))


def derive_trials(*,scenario_count,metric_count,confidence_alpha,risk_limit):
    """Minimum fixed trial count with a passing upper bound at zero failures.

    This is a simulation-precision lower bound, not a pilot episode count, a
    power guarantee, or evidence that the rule is likely to pass.
    """
    return math.ceil(math.log(confidence_alpha/(scenario_count*metric_count))/math.log(1-risk_limit))


def summarize(plan,rows):
    local_alpha=plan["simulation_confidence_alpha"]/(len(plan["regimes"])*(len(METRICS)+1))
    result={};all_pass=True;finished=True
    for regime in plan["regimes"]:
        values=[r for r in rows if r["regime"]==regime];n=len(values);selected=sum(r["selected"] for r in values)
        if n!=plan["trials_per_regime"]:finished=False
        metrics={}
        for key in METRICS:
            failures=sum(bool(r[key]) for r in values);lower,upper=exact_bounds(failures,n,local_alpha)
            conditional=sum(bool(r[key]) for r in values if r["selected"])
            # Conditional rates are exposed as diagnostics, not substituted for
            # the predeclared unconditional probability of an unsafe selection.
            conditional_bounds=exact_bounds(conditional,selected,local_alpha)
            metrics[key]=dict(failures=failures,trials=n,lower=lower,upper=upper,limit=plan["maximum_risks"][key],
                passed=upper<=plan["maximum_risks"][key],selected_trials=selected,
                conditional_selected_failures=conditional,conditional_selected_bounds=list(conditional_bounds))
        low,high=exact_bounds(selected,n,local_alpha);target=plan["minimum_resolution_probability"][regime]
        resolution_pass=low>=target
        passed=resolution_pass and all(m["passed"] for m in metrics.values())
        all_pass=all_pass and passed
        result[regime]=dict(trials=n,selected=selected,inconclusive=n-selected,risks=metrics,
            resolution=dict(lower=low,upper=high,minimum=target,passed=resolution_pass),passed=passed,
            interval_noncoverage_diagnostic=sum(r["future_interval_noncoverage"] for r in values),
            mean_seconds_per_trial=float(np.mean([r["elapsed_seconds"] for r in values])) if values else None)
    return dict(status="approved_for_declared_synthetic_domain" if finished and all_pass else "failed" if finished else "incomplete_not_approval",
        scenarios=result,simulation_local_alpha=local_alpha,all_planned_trials_complete=finished,
        guarantee="simultaneous_exact_binomial_bounds_for_declared_full_rule_metrics_in_each_synthetic_regime",
        limitation="no_arbitrary_tail_or_real_distribution_guarantee; observed_resolution_and_conditional_failure_rates_are_disclosed")


def _validated_rows(out,identity,plan):
    ledger=out/"ledger.jsonl";rows=[];head="0"*64;seen=set()
    if not ledger.exists():return rows,head
    import json
    for line in ledger.read_text(encoding="utf-8").splitlines():
        record=json.loads(line);saved_hash=record.pop("record_sha256")
        if object_hash(record)!=saved_hash or record["previous_sha256"]!=head or record["identity_sha256"]!=identity:
            raise ValueError("Pilot validation ledger changed")
        path=out/record["file"]
        if path.parent!=out or file_hash(path)!=record["sha256"]:raise ValueError("A completed synthetic trial changed")
        row=strict_json(path);key=(row["regime"],row["trial"])
        if key in seen or row["regime"] not in plan["regimes"] or not 0<=row["trial"]<plan["trials_per_regime"]:
            raise ValueError("Unexpected or repeated synthetic trial")
        expected=object_hash(dict(seed=plan["seed"],regime=row["regime"],trial=row["trial"]))
        if row["seed_sha256"]!=expected or row["identity_sha256"]!=identity:raise ValueError("Synthetic trial stream or scope changed")
        for field in (*METRICS,"selected","future_interval_noncoverage"):
            if type(row[field]) is not bool:raise ValueError("Invalid synthetic event accounting")
        seen.add(key);rows.append(row);head=saved_hash
    return rows,head


def run(design,protocol,plan,out, *, workers,max_trials,resume=False):
    validate_plan(plan,design,protocol);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    study.integer(workers,"CPU workers",1)
    if max_trials is not None:study.integer(max_trials,"operational trial bound",1)
    identity=object_hash(dict(design=design,protocol=protocol,plan=plan));lock=FileLock(str(out/"run.lock"),timeout=0)
    with lock:
        marker=out/"identity.json"
        if marker.exists():
            if not resume or strict_json(marker).get("identity_sha256")!=identity:raise ValueError("Use resume only for the same frozen rule and simulation plan")
        else:
            if resume or list(out.glob("*.jsonl")) or list(out.glob("trial_*.json")):raise ValueError("Unexpected existing simulation artifacts")
            atomic_bytes(marker,canonical_json(dict(identity_sha256=identity,design=design,selection_protocol=protocol,simulation_plan=plan))+b"\n")
        rows,head=_validated_rows(out,identity,plan);done={(r["regime"],r["trial"]) for r in rows}
        # Interleaving regimes avoids a bounded run exposing only easy cases.
        pending=[(regime,trial) for trial in range(plan["trials_per_regime"]) for regime in plan["regimes"] if (regime,trial) not in done]
        if max_trials is not None:pending=pending[:max_trials]
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tasks={pool.submit(simulation.simulate_rule_trial,design,protocol,regime,plan["seed"],trial):(regime,trial) for regime,trial in pending}
            for future in as_completed(tasks):
                if source_hashes()!=plan["source_sha256"]:raise ValueError("Pilot validation sources changed during execution")
                row=future.result();row["identity_sha256"]=identity
                regime,trial=tasks[future];name=f"trial_{regime}_{trial:07d}.json";path=out/name
                content=canonical_json(row)+b"\n"
                if path.exists() and path.read_bytes()!=content:
                    raise ValueError("Uncommitted trial artifact differs; preserve and review rather than overwrite")
                atomic_bytes(path,content)
                record=dict(file=name,sha256=file_hash(path),previous_sha256=head,identity_sha256=identity)
                head=object_hash(record)
                with (out/"ledger.jsonl").open("ab") as stream:
                    stream.write(canonical_json(dict(record,record_sha256=head))+b"\n");stream.flush();os.fsync(stream.fileno())
                rows.append(row)
        if source_hashes()!=plan["source_sha256"]:raise ValueError("Pilot validation sources changed before completion")
        report=summarize(plan,rows)
        report.update(kind="executed_full_pilot_rule_validation",identity_sha256=identity,source_sha256=source_hashes(),
            ledger_sha256=file_hash(out/"ledger.jsonl") if (out/"ledger.jsonl").exists() else None,ledger_chain_sha256=head,
            elapsed_seconds_this_run=time.perf_counter()-start,
            environment=dict(numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),blas_threads=1))
        atomic_bytes(out/"validation.json",canonical_json(report)+b"\n")
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("design","protocol","plan"):
        parser.add_argument("--"+name,type=Path,required=True);parser.add_argument("--"+name+"-sha256",required=True)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--workers",type=int,required=True)
    parser.add_argument("--max-trials",type=int);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();values={}
    for name in ("design","protocol","plan"):
        path=getattr(args,name)
        if file_hash(path)!=getattr(args,name+"_sha256"):raise ValueError("Frozen input hash mismatch")
        values[name]=strict_json(path)
    report=run(values["design"],values["protocol"],values["plan"],args.out,workers=args.workers,max_trials=args.max_trials,resume=args.resume)
    print(canonical_json(dict(status=report["status"],elapsed_seconds=report["elapsed_seconds_this_run"])).decode())


if __name__=="__main__":main()
