"""Read-only criterion audit of frozen E01 v4/v6 numerical cohorts.

Writes a fresh review directory. Never changes inputs, grants approval, selects
budgets, averages away failures, or treats missing raw artifacts as verified.
"""
from __future__ import annotations
import argparse
from collections import Counter
import itertools
import hashlib
import json
import math
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from experiment_io import canonical_json,file_hash,strict_json


def report_snapshot(path):
    """Bind parsed progress reports to the exact bytes read, including live jobs."""
    raw=Path(path).read_bytes()
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError("Duplicate report key")
            result[key]=value
        return result
    value=json.loads(raw,object_pairs_hook=unique)
    canonical_json(value)
    return value,hashlib.sha256(raw).hexdigest()


def extract_threshold(text,pattern):
    match=re.search(pattern,text)
    if not match: raise ValueError("Unsupported criterion wording: "+pattern)
    return float(match.group(1).rstrip("."))


def criteria_from_protocol(protocol):
    version=protocol["protocol_version"]
    if version==4:
        text=protocol["criteria"]
        return dict(coordinate_l1=extract_threshold(text,r"coordinate L1<=([0-9.]+)"),
            spearman=extract_threshold(text,r"Spearman>=([0-9.]+)"),
            top5=extract_threshold(text,r"top5 overlap>=([0-9.]+)"),
            auc=extract_threshold(text,r"AUC differences above([0-9.]+)"))
    if version!=6: raise ValueError("Only recorded E01 protocol versions 4 and 6 are supported")
    c=protocol["validation"]["criteria"]
    if "bitwise equal" not in c["repeatability"]: raise ValueError("Unsupported repeatability criterion")
    return dict(coordinate_l1=extract_threshold(c["coordinates"],r"L1 difference <=([0-9.]+)"),
        spearman=extract_threshold(c["group_order"],r"Spearman >=([0-9.]+)"),
        top5=extract_threshold(c["group_order"],r"top5 overlap >=([0-9.]+)"),
        residual=extract_threshold(c["completeness"],r"relative residual <=([0-9.]+)"),
        rms_curve_fraction=extract_threshold(c["primitive_response"],r"RMS curve difference <=([0-9.]+)"),
        rms_area_fraction=extract_threshold(c["primitive_response"],r"RMS area difference <=([0-9.]+)"),
        auc=extract_threshold(c["normalized_response"],r"area difference <=([0-9.]+)"))


def check(name,value=None,op=None,threshold=None,*,status=None,reason=None,**scope):
    if status is None:
        if value is None: status="undefined"
        elif op=="==": status="satisfies" if value==threshold else "violates"
        elif not isinstance(value,(int,float)) or not math.isfinite(value): raise ValueError("Nonfinite criterion value")
        elif op=="<=": status="satisfies" if value<=threshold else "violates"
        elif op==">=": status="satisfies" if value>=threshold else "violates"
        else: raise ValueError("Unknown criterion comparison")
    return dict(criterion=name,value=value,operator=op,threshold=threshold,status=status,reason=reason,**scope)


def coordinate_checks(value,criteria,*,v6=False,**scope):
    out=[]
    relative=value.get("relative_l1_difference")
    absolute=value.get("absolute_l1_difference")
    implied=None if relative in (None,0) or absolute is None else absolute/relative
    zero=(value.get("reference_zero_map") and value.get("candidate_zero_map") and value.get("absolute_l1_difference")==0)
    out.append(check("coordinate_relative_l1",relative,"<=",criteria["coordinate_l1"],
        status="degenerate" if relative is None and zero else None,reason="identical zero vectors" if zero else None,
        absolute_l1_difference=absolute,reference_l1_inferred_from_reported_ratio=implied,
        max_absolute_coordinate_difference=value.get("max_absolute_coordinate_difference"),
        reference_signed_sum=value.get("reference_signed_sum"),candidate_signed_sum=value.get("candidate_signed_sum"),
        denominator_qualification="Coordinate L1 denominator, distinct from the scalar endpoint gap; ratio inversion is undefined when reported relative difference is zero.",**scope))
    rho=value.get("group_spearman" if v6 else "absolute_group_spearman")
    out.append(check("group_spearman",rho,">=",criteria["spearman"],
        status="degenerate" if value.get("rank_status") in {"constant_group_scores","constant"} else None,**scope))
    n=value.get("top5_count" if v6 else "top_five_percent_count")
    overlap=value.get("top5_overlap" if v6 else "top_five_percent_overlap")
    out.append(check("top5_overlap",overlap,">=",criteria["top5"],
        status="not_applicable" if n==0 else None,reason="zero realized selected groups" if n==0 else None,
        realized_count=n,**scope))
    if v6 and n==0: out.append(check("top1_match",value.get("top1_match"),"==",True,**scope))
    return out


def area(x,y):
    if len(x)!=len(y) or len(x)<2 or any(a>b for a,b in zip(x,x[1:])): raise ValueError("Invalid curve grid")
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in [*x,*y]): raise ValueError("Invalid curve values")
    return math.fsum((b-a)*(u+v)/2 for a,b,u,v in zip(x,x[1:],y,y[1:]))


def verified_auc(common,arm,direction,response):
    ranking=common["rankings"][arm]
    curve=ranking["curves"][direction]["responses"][response]
    if curve["normalized"] is None:
        if curve["auc"] is not None: raise ValueError("Undefined curve has finite area")
        return None
    computed=area(ranking["realized_fractions"],curve["normalized"])
    if not math.isclose(computed,curve["auc"],rel_tol=1e-12,abs_tol=1e-12): raise ValueError("Saved normalized area differs from curve")
    return computed


def raw_response_difference(left,left_arm,right,right_arm,direction,response):
    a,b=(c["rankings"][arm] for c,arm in ((left,left_arm),(right,right_arm)))
    if a["realized_fractions"]!=b["realized_fractions"]: raise ValueError("Compared response grids differ")
    ca,cb=(v["curves"][direction] for v in (a,b))
    x=a["realized_fractions"];ya=ca["raw_scores"][response];yb=cb["raw_scores"][response]
    return dict(max_absolute_raw_curve_difference=max(abs(u-v) for u,v in zip(ya,yb)),
        raw_area_right_minus_left=area(x,yb)-area(x,ya),
        signed_endpoint_gaps=[v["responses"][response]["signed_gap"] for v in (ca,cb)],
        qualification="Raw differences are reported without inventing a V4 acceptance threshold.")


def v4_modality(report,modality,criteria,comparison):
    checks=[];observations=[]
    budgets=comparison["budgets"]
    for arm in ("bf16","fp32"):
        data=report.get("arms",{}).get(arm,{})
        endpoints=data.get("endpoint_repeats",{}).get(modality,{})
        for endpoint in ("actual","baseline"):
            value=endpoints.get(endpoint,{})
            checks.append(check("endpoint_repeatability",value.get("bitwise_repeatable"),"==",True,arm=arm,endpoint=endpoint))
            hashes=value.get("actions_sha256",[])
            checks.append(check("endpoint_hash_repeatability",bool(hashes) and len(set(hashes))==1,"==",True,arm=arm,endpoint=endpoint))
            if "endpoint_repeats" in comparison:
                checks.append(check("endpoint_repeat_count",len(hashes),"==",comparison["endpoint_repeats"],arm=arm,endpoint=endpoint))
            if arm=="bf16" and endpoint=="actual":
                checks.append(check("source_reference_exact_replay",value.get("frozen_reference_max_difference"),"==",0))
        for budget in budgets:
            record=data.get("maps",{}).get(f"{modality}_m{budget}")
            if record is None:
                checks.append(check("required_map",status="missing",arm=arm,m=budget));continue
            observations.append(dict(arm=arm,m=budget,**{key:record.get(key) for key in
                ("expected_gap","absolute_residual","relative_residual","gap_status","nonfinite_count")}))
            checks.append(check("finite_map",record.get("nonfinite_count"),"==",0,arm=arm,m=budget))
        for lo,hi in itertools.combinations(budgets,2):
            matches=[c for c in report.get("comparisons",[]) if c.get("arm")==arm and c.get("modality")==modality and c.get("candidate_m")==lo and c.get("reference_m")==hi]
            if len(matches)!=1: checks.append(check("budget_comparison",status="missing",arm=arm,candidate_m=lo,reference_m=hi));continue
            checks+=coordinate_checks(matches[0],criteria,kind="budget",arm=arm,candidate_m=lo,reference_m=hi)
    for budget in budgets:
        matches=[c for c in report.get("comparisons",[]) if c.get("modality")==modality and c.get("m")==budget and c.get("reference")=="fp32" and c.get("candidate")=="bf16"]
        if len(matches)!=1: checks.append(check("precision_comparison",status="missing",m=budget));continue
        checks+=coordinate_checks(matches[0],criteria,kind="precision",m=budget)
    common=report.get("common_response",{})
    for direction,response in itertools.product(("deletion","insertion"),("quadratic","l2")):
        for budget in budgets:
            values=common.get(f"{modality}_m{budget}")
            if values is None:
                checks.append(check("common_response",status="missing",m=budget));continue
            left,right=(verified_auc(values,arm,direction,response) for arm in ("bf16","fp32"))
            delta=None if left is None or right is None else abs(right-left)
            checks.append(check("normalized_auc_difference",delta,"<=",criteria["auc"],kind="precision",m=budget,direction=direction,response=response,
                raw_response=raw_response_difference(values,"bf16",values,"fp32",direction,response)))
        for arm,(lo,hi) in itertools.product(("bf16","fp32"),itertools.combinations(budgets,2)):
            if f"{modality}_m{lo}" not in common or f"{modality}_m{hi}" not in common: continue
            left=verified_auc(common[f"{modality}_m{lo}"],arm,direction,response)
            right=verified_auc(common[f"{modality}_m{hi}"],arm,direction,response)
            checks.append(check("normalized_auc_difference",None if left is None or right is None else abs(right-left),"<=",criteria["auc"],
                kind="budget",arm=arm,candidate_m=lo,reference_m=hi,direction=direction,response=response,
                raw_response=raw_response_difference(common[f"{modality}_m{lo}"],arm,common[f"{modality}_m{hi}"],arm,direction,response)))
    return checks,observations


def v6_target(report,criteria,decision):
    checks=[];observations=[]
    modality,target=report["modality"],report["target"]
    spec=decision["numerics"][modality][target]
    if report["specification"]!=spec: raise ValueError("Target report differs from declared numerical ladder")
    endpoints=report.get("endpoints",{})
    for endpoint in ("actual","baseline"):
        value=endpoints.get(endpoint,{})
        hashes=value.get("action_sha256",[])
        checks.append(check("endpoint_repeatability",value.get("bitwise_equal"),"==",True,endpoint=endpoint))
        checks.append(check("endpoint_hash_repeatability",bool(hashes) and len(set(hashes))==1,"==",True,endpoint=endpoint))
        checks.append(check("endpoint_repeat_count",len(hashes),"==",decision["endpoint_repeats"],endpoint=endpoint))
        if endpoint=="actual": checks.append(check("exact_new_reference",bool(hashes) and all(x==report["reference_sha256"] for x in hashes),"==",True))
    expected={(repeat,alpha) for repeat in range(decision["gradient_repeats"]) for alpha in decision["alphas"]}
    probes=report.get("gradient_probes",[])
    keys=[(x["repeat"],x["alpha"]) for x in probes]
    checks.append(check("gradient_probe_coverage",set(keys)==expected and len(keys)==len(expected),"==",True))
    for value in probes: checks.append(check("gradient_rng_unchanged",value.get("rng_unchanged"),"==",True,repeat=value["repeat"],alpha=value["alpha"]))
    audits=report.get("operation_audits",[])
    checks.append(check("fp32_operation_audit_coverage",len(audits)==len(decision["alphas"]) and {x["alpha"] for x in audits}==set(decision["alphas"]) and all(x["status"]=="passed" for x in audits),"==",True))
    for alpha in decision["alphas"]:
        value=report.get("gradient_repeatability",{}).get(str(alpha),{})
        checks.append(check("gradient_bitwise_repeatability",value.get("all_bitwise_equal"),"==",True,alpha=alpha))
        checks.append(check("gradient_repeat_pair_coverage",len(value.get("pairs",[])),"==",math.comb(decision["gradient_repeats"],2),alpha=alpha))
    expected={(m,r) for m in spec["budgets"] for r in range(decision["map_repeats"] if m in spec["repeat_budgets"] else 1)}
    maps=report.get("budgets",[]);keys=[(v["m"],v["repeat"]) for v in maps]
    checks.append(check("map_budget_repeat_coverage",set(keys)==expected and len(keys)==len(expected),"==",True))
    for value in maps:
        d=value["diagnostics"];gap=d.get("expected_gap");cutoff=decision["denominator_min"][target]
        if d.get("m") != value["m"]:
            raise ValueError("Map diagnostic budget differs from its declared budget")
        observations.append({**d,"m":value["m"],"repeat":value["repeat"]})
        checks.append(check("map_rng_unchanged",value.get("rng_unchanged"),"==",True,m=value["m"],repeat=value["repeat"]))
        checks.append(check("finite_map",d.get("nonfinite_count"),"==",0,m=value["m"],repeat=value["repeat"]))
        checks.append(check("candidate_completeness",d.get("relative_residual"),"<=",criteria["residual"],
            status="undefined" if gap is None or abs(gap)<=cutoff else None,
            reason="gap below recorded cutoff; raw-coordinate/response review required" if gap is None or abs(gap)<=cutoff else None,
            m=value["m"],repeat=value["repeat"],qualification="conditional check if this budget is selected; not a requirement that every budget is a candidate"))
    for m in spec["repeat_budgets"]:
        item=report.get("map_repeatability",{}).get(str(m)) or {}
        for ranking in ("IG","path_gradient"):
            checks.append(check("map_bitwise_repeatability",item.get(ranking,{}).get("all_bitwise_equal"),"==",True,m=m,ranking=ranking))
            checks.append(check("map_repeat_pair_coverage",len(item.get(ranking,{}).get("pairs",[])),"==",math.comb(decision["map_repeats"],2),m=m,ranking=ranking))
        checks.append(check("common_response_repeat_pair_coverage",len(item.get("common_responses",[])),"==",2*math.comb(decision["map_repeats"],2),m=m))
        for comparison in item.get("common_responses",[]):
            response_keys=[(v["direction"],v["response"]) for v in comparison["comparisons"]]
            checks.append(check("common_response_repeat_curve_coverage",len(response_keys)==6 and set(response_keys)=={
                (d,r) for d in ("deletion","insertion") for r in ("Q","L2","RMS")},"==",True,m=m,ranking=comparison["ranking"]))
            for response in comparison["comparisons"]:
                differences=response["raw_curve_candidate_minus_reference"]
                checks.append(check("reported_common_response_repeatability",bool(differences) and all(x==0 for x in differences),"==",True,
                    m=m,ranking=comparison["ranking"],direction=response["direction"],response=response["response"],
                    qualification="reported scalar equality; raw action/tensor files require artifact verification"))
    baseline_values=[v["RMS"] for v in endpoints.get("baseline",{}).get("scores",[])]
    baseline_rms=baseline_values[0] if baseline_values else None
    for lo,hi in itertools.combinations(spec["budgets"],2):
        for ranking in ("IG","path_gradient"):
            matches=[v for v in report.get("comparisons",[]) if v["candidate_m"]==lo and v["reference_m"]==hi and v["ranking"]==ranking]
            scope=dict(candidate_m=lo,reference_m=hi,ranking=ranking)
            if len(matches)!=1: checks.append(check("budget_comparison",status="missing",**scope));continue
            value=matches[0];checks+=coordinate_checks(value,criteria,v6=True,**scope)
            responses=value.get("common_responses",[])
            if {(r["direction"],r["response"]) for r in responses}!={(d,r) for d in ("deletion","insertion") for r in ("Q","L2","RMS")}:
                checks.append(check("common_response_coverage",status="missing",**scope))
            for response in responses:
                s={**scope,"direction":response["direction"],"response":response["response"]}
                delta=response["normalized_auc_candidate_minus_reference"]
                defined=response["reference_status"]==response["candidate_status"]=="defined"
                checks.append(check("normalized_auc_difference",None if delta is None else abs(delta),"<=",criteria["auc"],
                    status="not_applicable" if not defined and response["reference_status"] in {"zero_endpoint_gap","nearzero_endpoint_gap"} else None,
                    reason=None if defined else "shared endpoint gap undefined or below cutoff",**s))
                if response["response"]=="RMS":
                    for key,threshold,metric in (("max_absolute_raw_curve_difference",criteria["rms_curve_fraction"],"rms_curve_difference"),
                                               ("raw_auc_candidate_minus_reference",criteria["rms_area_fraction"],"rms_area_difference")):
                        raw=response[key]
                        checks.append(check(metric,None if raw is None else abs(raw),"<=",None if baseline_rms is None else threshold*baseline_rms,
                            status="undefined" if baseline_rms is None else None,baseline_rms=baseline_rms,fraction_threshold=threshold,**s))
                        if baseline_rms==0:checks[-1]["qualification"]="Exact zero baseline displacement: require exact equality, separately labeled degenerate baseline."
    return checks,observations


def summarize_checks(checks):
    counts=Counter(c["status"] for c in checks)
    return dict(counts=dict(counts),violations=[c for c in checks if c["status"]=="violates"],
        unresolved=[c for c in checks if c["status"] in {"missing","undefined","degenerate"}],
        status="recorded_threshold_violations" if counts["violates"] else "unresolved_criteria" if any(counts[k] for k in ("missing","undefined","degenerate"))
        else "reported_thresholds_satisfied_not_approval")


def relative_file(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError("Artifact escapes report directory")
    return path


def artifact_checks(report,report_path,verify):
    found={}
    def visit(value):
        if isinstance(value,dict):
            for file_key,hash_key in (("file","sha256"),("response_file","response_sha256")):
                if isinstance(value.get(file_key),str) and isinstance(value.get(hash_key),str):
                    name,digest=value[file_key],value[hash_key]
                    if name in found and found[name]!=digest: raise ValueError("Conflicting artifact hashes")
                    found[name]=digest
            for v in value.values(): visit(v)
        elif isinstance(value,list):
            for v in value: visit(v)
    visit(report)
    present=0
    for name,digest in found.items():
        path=relative_file(report_path.parent,name)
        if path.exists():
            present+=1
            if verify and file_hash(path)!=digest: raise ValueError("Raw artifact hash mismatch: "+name)
    return dict(referenced=len(found),present=present,verified=present if verify else 0,
                status="all_available_artifact_hashes_verified" if verify and present==len(found) else "partial_artifact_verification" if verify else "report_only_not_raw_verified")


def stage_artifact_checks(report_path,verify,expected_status="complete_diagnostics_not_approval"):
    completion_path=report_path.parent/"completion.json"
    if not completion_path.exists(): return dict(status="stage_incomplete_report_not_completion_bound")
    completion=strict_json(completion_path)
    if completion.get("status")!=expected_status: raise ValueError("Unexpected stage completion status")
    artifacts=completion["artifacts_sha256"]
    if artifacts.get("report.json")!=file_hash(report_path): raise ValueError("Candidate completion does not bind report bytes")
    present=0
    for name,digest in artifacts.items():
        path=relative_file(report_path.parent,name)
        if path.exists():
            present+=1
            if verify and file_hash(path)!=digest: raise ValueError("Completed candidate artifact changed: "+name)
    return dict(status="complete_artifacts_verified" if verify and present==len(artifacts) else "report_bound_raw_artifacts_not_fully_verified",
                referenced=len(artifacts),present=present,verified=present if verify else 0,
                completion_sha256=file_hash(completion_path))


def parse_job(job_id,stratum_ids):
    for stratum in sorted(stratum_ids,key=len,reverse=True):
        match=re.fullmatch(re.escape(stratum)+r"-e(\d+)-c(\d+)(?:-(logpi|l2))?",job_id)
        if match: return stratum,int(match[1]),int(match[2]),match[3]
    raise ValueError("Unknown queue job identity: "+job_id)


def planned_roster(protocol,queue,queue_root):
    strata=[s["id"] for s in protocol["strata"]]
    jobs={}
    for job in queue["jobs"]:
        identity=parse_job(job["job_id"],strata)
        if identity in jobs: raise ValueError("Duplicate queue job")
        jobs[identity]=job
    if protocol["protocol_version"]==4:
        absent={(v["stratum"],v["episode"],v["call"]):v for v in queue.get("unavailable",[])}
        roster=[]
        for stratum in protocol["strata"]:
            for episode,call in itertools.product(range(stratum["episodes"]),stratum["evaluate_calls"]):
                for target in protocol["comparison"]["targets"]:
                    key=(stratum["id"],episode,call,target)
                    missing=(stratum["id"],episode,call) in absent
                    roster.append(dict(stratum=stratum["id"],episode=episode,call=call,target=target,
                        planned_status="call_not_reached" if missing else "available",job=jobs.pop(key,None)))
    else:
        roster=[]
        for stratum in strata:
            bank_path=queue_root/"banks"/(stratum+".json")
            bank=strict_json(bank_path)
            for entry in bank["contexts"]:
                key=(stratum,entry["episode"],entry["policy_call_idx"],None)
                job=jobs.pop(key,None)
                if job and (file_hash(bank_path)!=job["bank_sha256"] or job["context_id"]!=entry["context_id"] or job["planned_status"]!=entry["status"]):
                    raise ValueError("Queue bank context identity mismatch")
                roster.append(dict(stratum=stratum,episode=entry["episode"],call=entry["policy_call_idx"],
                    target=None,planned_status=entry["status"],context_id=entry["context_id"],job=job))
    if jobs: raise ValueError("Queue contains jobs outside the planned roster")
    return roster


def audit_single_queue(protocol_path,queue_path,*,verify_artifacts=False,sealed_jobs=None):
    protocol_path,queue_path=Path(protocol_path),Path(queue_path)
    protocol,queue=strict_json(protocol_path),strict_json(queue_path)
    protocol_hash,queue_hash=file_hash(protocol_path),file_hash(queue_path)
    if queue["protocol_sha256"]!=protocol_hash: raise ValueError("Queue/protocol byte identity mismatch")
    criteria=criteria_from_protocol(protocol)
    version=protocol["protocol_version"]
    roster=planned_roster(protocol,queue,queue_path.parent)
    rows=[];sources={str(protocol_path):protocol_hash,str(queue_path):queue_hash}
    context_status={}
    for entry in roster:
        job=entry["job"];context_key=(entry["stratum"],entry["episode"],entry["call"])
        base={key:entry[key] for key in ("stratum","episode","call","planned_status")}
        targets=[entry["target"]] if version==4 else ["Q","L2"]
        combinations=list(itertools.product(("vision","language","state"),targets))
        def placeholders(status,reason=None):
            for modality,target in combinations: rows.append(dict(**base,modality=modality,target=target,status=status,reason=reason,checks=[]))
        if entry["planned_status"]!="available":
            context_status[context_key]="call_not_reached";placeholders("call_not_reached");continue
        if not job:
            context_status[context_key]="job_missing";placeholders("job_missing");continue
        if sealed_jobs is not None and job["job_id"] not in sealed_jobs:
            context_status[context_key]="unsealed_at_audit_start"
            placeholders("unsealed_at_audit_start","Live reports and artifacts were intentionally not read by the sealed-only audit")
            continue
        report_path=queue_path.parent/"results"/job["job_id"]/"report.json"
        completion_path=queue_path.parent/"completion"/(job["job_id"]+".json")
        rows_before=len(rows)
        if not report_path.exists():
            status="not_yet_reported"
            if completion_path.exists():
                completion=strict_json(completion_path)
                sources[str(completion_path)]=file_hash(completion_path)
                if completion["queue_sha256"]!=queue_hash: raise ValueError("Job completion queue identity mismatch")
                status="infrastructure_or_contract_failure" if completion["returncode"]!=0 else "invalid_or_incomplete_evidence"
            context_status[context_key]=status;placeholders(status);continue
        try:
            report,sources[str(report_path)]=report_snapshot(report_path)
            completion=strict_json(completion_path) if completion_path.exists() else None
            if completion:
                sources[str(completion_path)]=file_hash(completion_path)
                if completion["queue_sha256"]!=queue_hash: raise ValueError("Job completion queue identity mismatch")
            job_complete=completion is not None and completion["returncode"]==0
            job_failed=completion is not None and completion["returncode"]!=0
            if version==4:
                if report["decision_sha256"]!=protocol_hash or report["context_id"]!=job["context_id"] or report["attribution_target"]!=entry["target"]:
                    raise ValueError("V4 report context/target/protocol identity mismatch")
                if report["script_sha256"]!=queue["validator_sha256"]: raise ValueError("V4 validator byte identity differs")
                artifact=artifact_checks(report,report_path,verify_artifacts)
                finished=job_complete and report.get("execution_status")=="complete"
                context_status[context_key]="reported" if finished else "incomplete_job"
                for modality,target in combinations:
                    checks,observations=v4_modality(report,modality,criteria,protocol["comparison"])
                    rows.append(dict(**base,modality=modality,target=target,status="complete_diagnostics" if finished else "infrastructure_or_contract_failure" if job_failed else "incomplete_job",
                        context_id=report["context_id"],report_sha256=sources[str(report_path)],artifact_evidence=artifact,
                        checks=checks,observations=observations,criterion_audit=summarize_checks(checks),
                        qualification="V4 report hash is captured by this audit; the historical queue completion did not bind report bytes. Precision factor control is a different function from V6."))
            else:
                stage_evidence=stage_artifact_checks(report_path,verify_artifacts)
                decision_path=queue_path.parent/"decisions"/(job["job_id"]+".json")
                decision=strict_json(decision_path)
                if file_hash(decision_path)!=job["decision_sha256"] or report["decision_sha256"]!=job["decision_sha256"] or report["bank_sha256"]!=job["bank_sha256"]:
                    raise ValueError("V6 decision/bank report identity mismatch")
                if report["script_sha256"]!=queue["source_sha256"]["scripts/validate_fp32_probe.py"]:
                    raise ValueError("V6 validator byte identity differs")
                helper_files=dict(paired_comparison="paired_comparison.py",integrated_gradients="integrated_gradients.py",
                    precision="scripts/validate_downstream_precision.py",repeatability="scripts/validate_gradient_repeatability.py",
                    replay="faithfulness.py",cache_runtime="fp32_probe_cache.py")
                if report["helper_sha256"]!={k:queue["source_sha256"][v] for k,v in helper_files.items()}:
                    raise ValueError("V6 helper bytes differ from frozen queue")
                prepared_path=queue_path.parent/"prepared"/job["job_id"]/"report.json"
                preparation,sources[str(prepared_path)]=report_snapshot(prepared_path)
                prepared_evidence=stage_artifact_checks(prepared_path,verify_artifacts,"complete_source_preparation")
                if prepared_evidence.get("completion_sha256")!=report["preparation_completion_sha256"]:
                    raise ValueError("Candidate report does not bind source preparation completion")
                if any(preparation[k]!=report[k] for k in ("decision_sha256","bank_sha256","source_metrics_sha256")):
                    raise ValueError("Candidate and preparation source identities differ")
                sources[str(decision_path)]=file_hash(decision_path)
                contexts=report.get("contexts",[])
                matching=[v for v in contexts if (v["episode"],v["policy_call_idx"])==(entry["episode"],entry["call"])]
                if len(matching)!=1: raise ValueError("V6 report must identify its unique planned context")
                context=matching[0]
                if context.get("source_context_id")!=entry["context_id"]: raise ValueError("V6 source context differs")
                finished=job_complete and report.get("status")=="complete_diagnostics_not_approval"
                context_status[context_key]="reported" if finished else "incomplete_job"
                for modality,target in combinations:
                    outcomes=[v for v in context.get("numerics",[]) if (v["modality"],v["target"])==(modality,target)]
                    if len(outcomes)!=1:
                        rows.append(dict(**base,modality=modality,target=target,status="infrastructure_or_contract_failure" if job_failed else "target_not_yet_reported",checks=[]));continue
                    outcome=outcomes[0]
                    if outcome["status"]=="numerical_failure":
                        rows.append(dict(**base,modality=modality,target=target,status="numerical_failure",reason=outcome.get("reason"),
                            failure_kind=outcome.get("failure_kind"),checks=[],failure_artifact=context.get("failure_artifact")));continue
                    target_path=relative_file(report_path.parent,outcome["report_file"])
                    if file_hash(target_path)!=outcome["report_sha256"]: raise ValueError("Target report hash differs")
                    target_report,sources[str(target_path)]=report_snapshot(target_path)
                    if target_report["context_id"]!=entry["context_id"] or (target_report["modality"],target_report["target"])!=(modality,target): raise ValueError("Target identity differs")
                    if target_report["status"]!=outcome["status"]: raise ValueError("Target report status differs from parent outcome")
                    checks,observations=v6_target(target_report,criteria,decision)
                    runtime_expected=dict(cublas_workspace_config=":4096:8",deterministic_algorithms=True,
                        deterministic_warn_only=False,cudnn_benchmark=False,cudnn_deterministic=True,
                        float32_matmul_precision="highest",cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False)
                    checks.extend([
                        check("source_reference_authentication",context.get("source_replay_verified"),"==",True),
                        check("fresh_candidate_process",report.get("startup",{}).get("torch_not_preimported"),"==",True),
                        check("candidate_runtime_bundle",report.get("runtime",{}).get("settings")==runtime_expected,"==",True),
                        check("declared_source_runtime",report.get("settings_transition",{}).get("source_replay")==decision.get("source_replay_settings"),"==",True),
                        check("autocast_disabled",report.get("math_settings",{}).get("autocast"),"==","disabled")])
                    rows.append(dict(**base,modality=modality,target=target,status="complete_diagnostics" if finished else "infrastructure_or_contract_failure" if job_failed else "incomplete_job",
                        context_id=entry["context_id"],report_sha256=sources[str(target_path)],checks=checks,observations=observations,
                        artifact_evidence=artifact_checks(target_report,target_path,verify_artifacts),criterion_audit=summarize_checks(checks)))
                    rows[-1]["stage_evidence"]=stage_evidence
                    rows[-1]["preparation_evidence"]=prepared_evidence
        except (ValueError,KeyError,TypeError,OSError) as error:
            del rows[rows_before:]
            context_status[context_key]="invalid_or_incomplete_evidence"
            placeholders("invalid_or_incomplete_evidence",str(error))
    unique_contexts={(v["stratum"],v["episode"],v["call"]) for v in roster}
    if file_hash(protocol_path)!=protocol_hash or file_hash(queue_path)!=queue_hash:
        raise ValueError("Protocol or queue changed during read-only audit")
    if len(rows)!=len(unique_contexts)*6: raise ValueError("Audit lost or duplicated planned modality/target cells")
    for key in unique_contexts:
        statuses={r["status"] for r in rows if (r["stratum"],r["episode"],r["call"])==key}
        context_status[key]=("call_not_reached" if statuses=={"call_not_reached"} else "all_six_diagnostics_complete"
            if statuses=={"complete_diagnostics"} else "contains_numerical_failure" if "numerical_failure" in statuses
            else "contains_invalid_evidence" if "invalid_or_incomplete_evidence" in statuses
            else "contains_infrastructure_or_contract_failure" if "infrastructure_or_contract_failure" in statuses else "incomplete")
    return dict(schema_version=1,kind="numerical_cohort_criterion_audit",protocol_version=version,
        protocol_sha256=protocol_hash,queue_sha256=queue_hash,criteria=criteria,
        planned_contexts=len(unique_contexts),planned_context_target_modality_cells=len(unique_contexts)*6,
        unavailable_contexts=sum(v=="call_not_reached" for v in context_status.values()),
        context_status_counts=dict(Counter(context_status.values())),cell_status_counts=dict(Counter(v["status"] for v in rows)),
        criterion_status_counts=dict(Counter(c["status"] for row in rows for c in row["checks"])),
        rows=rows,source_files_sha256=sources,auditor_sha256=file_hash(__file__),
        conclusion="No approval or candidate selection. Every context and failure is retained; no cohort average substitutes for individual criteria.",
        qualification="Conditional candidate-completeness checks apply if that budget is selected. Missing raw tensors remain unverified even when report-only thresholds are satisfied.")


def attempt_record(queue_path,queue,job,*,sealed_jobs=None):
    """Keep failed and partial attempts even when an explicit repair supersedes them."""
    completion_path=queue_path.parent/"completion"/(job["job_id"]+".json")
    report_path=queue_path.parent/"results"/job["job_id"]/"report.json"
    record=dict(queue_sha256=file_hash(queue_path),job_id=job["job_id"],queue_path=str(queue_path),
                status="not_started_or_not_reported")
    if sealed_jobs is not None and job["job_id"] not in sealed_jobs:
        record.update(status="unsealed_at_audit_start",report_read_policy="not_read_unsealed_at_start",
                      partial_report_files_sha256={})
        return record
    if completion_path.exists():
        completion,digest=report_snapshot(completion_path)
        if completion["queue_sha256"]!=record["queue_sha256"] or completion.get("job_id",job["job_id"])!=job["job_id"]:
            raise ValueError("Attempt completion identity differs")
        record.update(completion_path=str(completion_path),completion_sha256=digest,
            returncode=completion["returncode"],stages=completion.get("stages"),
            status="completed" if completion["returncode"]==0 else "infrastructure_or_contract_failure")
    if report_path.exists():
        report,digest=report_snapshot(report_path)
        record.update(report_path=str(report_path),report_sha256=digest,
            report_status=report.get("status",report.get("execution_status")))
        if "returncode" not in record: record["status"]="partial_report_without_completion"
    claim_path=queue_path.parent/"claims"/(job["job_id"]+".json")
    if claim_path.exists():
        record["claim_sha256"]=file_hash(claim_path)
        if "returncode" not in record: record["status"]="started_without_completion"
    record["partial_report_files_sha256"]={str(p.relative_to(queue_path.parent)):file_hash(p)
        for p in sorted(report_path.parent.rglob("report.json")) if p!=report_path}
    return record


def _portable_name(value):
    return str(value).replace("\\","/").rstrip("/").split("/")[-1]


def retry_link(parent_path,retry_path,amendment_path,protocol_hash,version,*,parent_sealed_jobs=None):
    parent,retry=strict_json(parent_path),strict_json(retry_path)
    parent_hash=file_hash(parent_path)
    if retry["protocol_sha256"]!=protocol_hash or parent["protocol_sha256"]!=protocol_hash:
        raise ValueError("Retry changes the scientific protocol")
    amendment=strict_json(amendment_path);amendment_hash=file_hash(amendment_path)
    if amendment.get("recorded_before_retry") is not True: raise ValueError("Retry lacks a prospective amendment")
    if retry.get("parent_queue_sha256",parent_hash)!=parent_hash: raise ValueError("Retry parent queue hash differs")
    if retry.get("repair_amendment_sha256",amendment_hash)!=amendment_hash: raise ValueError("Retry amendment hash differs")
    section=amendment.get(f"v{version}",amendment)
    bound_parent=section.get("parent_queue_sha256",section.get("superseded_queue_sha256"))
    if bound_parent!=parent_hash: raise ValueError("Amendment does not bind the parent queue")
    if "superseded_queue_sha256" in section and any((parent_path.parent/"claims").glob("*.json")):
        raise ValueError("Superseded unlaunched queue contains execution claims")
    destination=section.get("retry_root",section.get("new_root",section.get("new_queue_root")))
    if destination is None or _portable_name(destination)!=retry_path.parent.name:
        raise ValueError("Amendment does not identify the retry destination")
    if amendment.get("scientific_protocol_sha256",protocol_hash)!=protocol_hash:
        raise ValueError("Amendment changes the scientific protocol")
    original={j["job_id"]:j for j in parent["jobs"]}
    replacement={j["job_id"]:j for j in retry["jobs"]}
    if len(original)!=len(parent["jobs"]) or len(replacement)!=len(retry["jobs"]): raise ValueError("Duplicate retry job")
    if not replacement.keys()<=original.keys(): raise ValueError("Retry introduces a new planned job")
    prior_records={}
    declared={v["job_id"]:v for v in retry.get("prior_attempts",[])}
    if "prior_attempts" in retry and (len(declared)!=len(retry["prior_attempts"]) or set(declared)!=set(replacement)):
        raise ValueError("Retry prior-attempt roster differs")
    for job_id,job in replacement.items():
        old=original[job_id]
        fields=("context_id","source_manifest_sha256","source_sidecar_sha256") if version==4 else ("context_id","planned_status","bank_sha256","decision_sha256")
        if any(job.get(k)!=old.get(k) for k in fields): raise ValueError("Retry changes source context or decision identity")
        prior=attempt_record(parent_path,parent,old,sealed_jobs=parent_sealed_jobs)
        if prior.get("returncode")==0: raise ValueError("Retry replaces a successful completed job")
        if job_id in declared:
            d=declared[job_id]
            if d["original_completion_sha256"]!=prior.get("completion_sha256"): raise ValueError("Retry prior completion hash differs")
            expected="failed_before_numerical_measurements" if prior.get("returncode") is not None else "not_started"
            if d["status_before_retry"]!=expected: raise ValueError("Retry declared prior status differs")
        prior_records[job_id]=prior
    if "retained_completed_jobs" in retry:
        retained=retry["retained_completed_jobs"]
        if len(retained)!=len(set(retained)) or set(retained)!=set(original)-set(replacement): raise ValueError("Retained job roster differs")
        for job_id in retained:
            if attempt_record(parent_path,parent,original[job_id],sealed_jobs=parent_sealed_jobs).get("returncode")!=0: raise ValueError("Retained job was not completed")
    if version==4 and retry["validator_sha256"]!=parent["validator_sha256"]:
        raise ValueError("V4 execution-only retry changes validator bytes")
    return dict(parent_queue_sha256=parent_hash,retry_queue_sha256=file_hash(retry_path),
        amendment_path=str(amendment_path),amendment_sha256=amendment_hash,amendment_id=amendment.get("amendment_id"),
        replaces_jobs=list(replacement),prior_attempts=prior_records)


def freeze_sealed_membership(paths,queues):
    """Freeze every queue's completion-file membership before reading job data.

    Both successful and failed completion records qualify as sealed attempts.
    Directory inventories happen first, so later job completions cannot enter
    through a slower report, history, or retry-lineage read.
    """
    inventories=[]
    for path,queue in zip(paths,queues):
        names={p.name for p in (path.parent/"completion").glob("*.json") if p.is_file()}
        inventories.append(frozenset(job["job_id"] for job in queue["jobs"] if job["job_id"]+".json" in names))
    return [{job_id:file_hash(path.parent/"completion"/(job_id+".json")) for job_id in sorted(ids)}
            for path,ids in zip(paths,inventories)]


def audit_cohort(protocol_path,queue_path,*,verify_artifacts=False,retry_queues=(),retry_amendments=(),sealed_only=False):
    """Apply explicit infrastructure repair lineage without increasing the population."""
    if len(retry_queues)!=len(retry_amendments): raise ValueError("Each retry queue requires its recorded amendment")
    paths=[Path(queue_path),*[Path(p) for p in retry_queues]]
    queues=[strict_json(p) for p in paths]
    sealed=freeze_sealed_membership(paths,queues) if sealed_only else [None]*len(paths)
    audits=[audit_single_queue(protocol_path,p,verify_artifacts=verify_artifacts,sealed_jobs=members)
            for p,members in zip(paths,sealed)]
    result=audits[0];protocol=strict_json(protocol_path);version=protocol["protocol_version"]
    if sealed_only:
        result["read_policy"]="sealed_only_membership_frozen_before_job_reads"
        result["sealed_membership"]=[dict(queue_path=str(path),queue_sha256=audit["queue_sha256"],
            sealed_job_ids=sorted(members),unsealed_job_ids=sorted(job["job_id"] for job in queue["jobs"] if job["job_id"] not in members),
            completion_sha256=members) for path,queue,audit,members in zip(paths,queues,audits,sealed)]
        for path,members in zip(paths,sealed):
            result["source_files_sha256"].update({str(path.parent/"completion"/(job_id+".json")):digest for job_id,digest in members.items()})
    def cell_key(row): return tuple(row[k] for k in ("stratum","episode","call","target","modality"))
    def job_key(job): return parse_job(job["job_id"],[s["id"] for s in protocol["strata"]])
    def row_job_key(row): return (row["stratum"],row["episode"],row["call"],row["target"] if version==4 else None)
    selected={cell_key(r):r for r in result["rows"]};history={};links=[]
    for index,(path,queue,audit) in enumerate(zip(paths,queues,audits)):
        if index:
            link=retry_link(paths[index-1],path,Path(retry_amendments[index-1]),result["protocol_sha256"],version,
                            parent_sealed_jobs=sealed[index-1])
            links.append(link)
            result["source_files_sha256"][str(retry_amendments[index-1])]=link["amendment_sha256"]
        present_jobs={job_key(j):j for j in queue["jobs"]}
        for identity,job in present_jobs.items():
            history.setdefault(identity,[]).append(attempt_record(path,queue,job,sealed_jobs=sealed[index]))
        for row in audit["rows"]:
            if index==0 or row_job_key(row) in present_jobs: selected[cell_key(row)]=row
        result["source_files_sha256"].update(audit["source_files_sha256"])
    result["rows"]=list(selected.values())
    for row in result["rows"]: row["attempts"]=history.get(row_job_key(row),[])
    contexts={}
    for row in result["rows"]: contexts.setdefault((row["stratum"],row["episode"],row["call"]),set()).add(row["status"])
    def context_status(statuses):
        if statuses=={"call_not_reached"}:return "call_not_reached"
        if statuses=={"complete_diagnostics"}:return "all_six_diagnostics_complete"
        for value in ("numerical_failure","invalid_or_incomplete_evidence","infrastructure_or_contract_failure"):
            if value in statuses:return "contains_"+value
        return "incomplete"
    result["context_status_counts"]=dict(Counter(context_status(v) for v in contexts.values()))
    result["cell_status_counts"]=dict(Counter(r["status"] for r in result["rows"]))
    result["criterion_status_counts"]=dict(Counter(c["status"] for r in result["rows"] for c in r["checks"]))
    result["attempt_status_counts"]=dict(Counter(a["status"] for records in history.values() for a in records))
    result["repair_lineage"]=links
    result["selected_queue_sha256"]=file_hash(paths[-1])
    groups={}
    scope_fields=("criterion","kind","arm","ranking","candidate_m","reference_m","m","direction","response")
    for row in result["rows"]:
        for item in row["checks"]:
            scope={"modality":row["modality"],"target":row["target"],**{k:item[k] for k in scope_fields if k in item}}
            key=canonical_json(scope)
            group=groups.setdefault(key,dict(scope=scope,status_counts=Counter(),observations=[],threshold=item.get("threshold")))
            group["status_counts"][item["status"]]+=1
            group["observations"].append(dict(stratum=row["stratum"],episode=row["episode"],call=row["call"],
                status=item["status"],value=item.get("value"),threshold=item.get("threshold")))
    result["criterion_groups"]=[]
    for group in groups.values():
        values=[v["value"] for v in group["observations"] if isinstance(v["value"],(int,float)) and not isinstance(v["value"],bool)]
        group["status_counts"]=dict(group["status_counts"])
        group["minimum_value"]=min(values) if values else None;group["maximum_value"]=max(values) if values else None
        result["criterion_groups"].append(group)
    if len(result["rows"])!=result["planned_context_target_modality_cells"]: raise ValueError("Retry changed planned cell count")
    if sealed_only:
        for path,members in zip(paths,sealed):
            for job_id,digest in members.items():
                if file_hash(path.parent/"completion"/(job_id+".json"))!=digest:
                    raise ValueError("Sealed completion changed after membership was frozen: "+job_id)
    for path,digest in result["source_files_sha256"].items():
        if file_hash(path)!=digest: raise ValueError("Source changed during cohort audit: "+path)
    return result


def markdown_report(audit):
    text=["# E01 numerical cohort criterion audit","",f"Protocol v{audit['protocol_version']}; {audit['planned_contexts']} planned contexts; {audit['unavailable_contexts']} unavailable contexts.","",audit["conclusion"],"",audit["qualification"],"",
        "| Stratum | Planned cells | Complete diagnostics | Unavailable | Other incomplete/failure cells |",
        "|---|---:|---:|---:|---:|"]
    if audit.get("sealed_membership") is not None:
        text[6:6]=["Sealed-only mode: completion-file membership was frozen for every queue before reading job reports. Active or newly completed jobs remain unsealed in this snapshot; their reports and artifacts were not read. Sealed failed attempts remain in the lineage.",""]
    for stratum in sorted({r["stratum"] for r in audit["rows"]}):
        rows=[r for r in audit["rows"] if r["stratum"]==stratum];counts=Counter(r["status"] for r in rows)
        text.append(f"| {stratum} | {len(rows)} | {counts['complete_diagnostics']} | {counts['call_not_reached']} | {len(rows)-counts['complete_diagnostics']-counts['call_not_reached']} |")
    text.extend(["","Attempt statuses retain superseded failures: "+", ".join(f"{k}={v}" for k,v in audit["attempt_status_counts"].items())+".","",
        "Counts below are comparisons, not independent samples. Values are extrema, never averages. Reported coordinate sensitivity does not establish a production setting. All criteria, undefined values, conditional completeness checks and individual context identities are in audit.json.","",
        "| Target/modality | Comparison | Largest relative coordinate L1 | Satisfies | Violates | Unresolved |",
        "|---|---|---:|---:|---:|---:|"])
    for group in audit["criterion_groups"]:
        s=group["scope"]
        if s["criterion"]!="coordinate_relative_l1":continue
        label=" ".join(str(s[k]) for k in ("kind","arm","ranking") if k in s)
        label+=" "+(f"m{s['candidate_m']} vs m{s['reference_m']}" if "candidate_m" in s else f"m{s.get('m','')}")
        counts=group["status_counts"];value=group["maximum_value"]
        text.append(f"| {s['target']}/{s['modality']} | {label.strip()} | {value:.6g}".rstrip() if value is not None else f"| {s['target']}/{s['modality']} | {label.strip()} | undefined")
        text[-1]+=f" | {counts.get('satisfies',0)} | {counts.get('violates',0)} | {sum(counts.get(k,0) for k in ('missing','undefined','degenerate'))} |"
    text.extend(["","Recorded criterion statuses: "+", ".join(f"{k}={v}" for k,v in audit["criterion_status_counts"].items())+".","",
        "Infrastructure retries are selected only by the named prospective amendments. Failed and partial attempt hashes remain attached to every affected logical cell. Raw artifact verification status is recorded separately from report criteria."])
    return "\n".join(text)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--queue",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--verify-artifacts",action="store_true")
    parser.add_argument("--sealed-only",action="store_true",help="Freeze completed-job membership before all report/history reads and skip unsealed reports/artifacts")
    parser.add_argument("--retry-queue",type=Path,action="append",default=[])
    parser.add_argument("--retry-amendment",type=Path,action="append",default=[])
    args=parser.parse_args()
    audit=audit_cohort(args.protocol,args.queue,verify_artifacts=args.verify_artifacts,
        retry_queues=args.retry_queue,retry_amendments=args.retry_amendment,sealed_only=args.sealed_only)
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"audit.json").write_bytes(canonical_json(audit)+b"\n")
    (args.out/"review.md").write_text(markdown_report(audit),encoding="utf-8")
    print({key:audit[key] for key in ("planned_contexts","unavailable_contexts","cell_status_counts","criterion_status_counts")})


if __name__=="__main__": main()
