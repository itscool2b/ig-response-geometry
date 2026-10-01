"""Cohort audits retain failures, numerical degeneracy and planned denominators."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiment_io import canonical_json,file_hash
from scripts.audit_numerical_cohort import (area,audit_cohort,coordinate_checks,criteria_from_protocol,
    retry_link,stage_artifact_checks,summarize_checks,v6_target,verified_auc)


CRITERIA=dict(coordinate_l1=.01,spearman=.99,top5=.95,residual=.01,rms_curve_fraction=.01,rms_area_fraction=.005,auc=.01)


def test_one_failure_cannot_be_hidden_by_many_favorable_comparisons():
    good=dict(relative_l1_difference=.001,absolute_group_spearman=1,top_five_percent_count=1,top_five_percent_overlap=1)
    checks=[]
    for _ in range(100): checks+=coordinate_checks(good,CRITERIA)
    bad={**good,"relative_l1_difference":.5}
    checks+=coordinate_checks(bad,CRITERIA)
    summary=summarize_checks(checks)
    assert summary["status"]=="recorded_threshold_violations"
    assert len(summary["violations"])==1 and summary["violations"][0]["value"]==.5


def test_coordinate_denominator_is_distinct_from_the_scalar_gap():
    values=dict(relative_l1_difference=.5,absolute_l1_difference=1.,absolute_group_spearman=1.,
                top_five_percent_count=1,top_five_percent_overlap=1.)
    item=coordinate_checks(values,CRITERIA)[0]
    assert item["reference_l1_inferred_from_reported_ratio"]==2.
    assert item["absolute_l1_difference"]==1.
    values.update(relative_l1_difference=0.,absolute_l1_difference=0.)
    assert coordinate_checks(values,CRITERIA)[0]["reference_l1_inferred_from_reported_ratio"] is None


def test_zero_map_and_zero_realized_top5_do_not_become_a_pass():
    values=dict(relative_l1_difference=None,absolute_l1_difference=0,reference_zero_map=True,candidate_zero_map=True,
        rank_status="constant_group_scores",group_spearman=None,top5_count=0,top5_overlap=None,top1_match=True)
    checks=coordinate_checks(values,CRITERIA,v6=True)
    assert [c["status"] for c in checks]==["degenerate","degenerate","not_applicable","satisfies"]
    assert summarize_checks(checks)["status"]=="unresolved_criteria"


def test_saved_auc_is_independently_recomputed_on_realized_grid():
    assert area([0,.25,1],[1,.5,0])==.375
    common={"rankings":{"fp32":{"realized_fractions":[0,.25,1],"curves":{"deletion":{"responses":{
        "quadratic":{"normalized":[1,.5,0],"auc":.375}}}}}}}
    assert verified_auc(common,"fp32","deletion","quadratic")==.375
    common["rankings"]["fp32"]["curves"]["deletion"]["responses"]["quadratic"]["auc"]+=.01
    with pytest.raises(ValueError,match="differs from curve"):verified_auc(common,"fp32","deletion","quadratic")


def protocol_v4():
    return dict(protocol_version=4,strata=[dict(id="fixture-170m",episodes=2,evaluate_calls=[0,12])],
        comparison=dict(targets=["logpi","l2"],budgets=[128,256]),
        criteria="relative coordinate L1<=0.01, group Spearman>=0.99, top5 overlap>=0.95. Common-response AUC differences above0.01 flag material sensitivity.")


def test_roster_counts_unavailable_not_started_and_failed_jobs_without_duplication(tmp_path):
    protocol=protocol_v4();protocol_path=tmp_path/"protocol.json";protocol_path.write_bytes(canonical_json(protocol))
    jobs=[]
    for ep,call in ((0,0),(0,12),(1,0)):
        for target in ("logpi","l2"):
            jobs.append(dict(job_id=f"fixture-170m-e{ep:03d}-c{call:03d}-{target}",context_id=f"context-{ep}-{call}"))
    queue=dict(protocol_sha256=file_hash(protocol_path),jobs=jobs,validator_sha256="script",
               unavailable=[dict(stratum="fixture-170m",episode=1,call=12)])
    queue_path=tmp_path/"queue.json";queue_path.write_bytes(canonical_json(queue))
    (tmp_path/"completion").mkdir()
    failed=tmp_path/"completion"/(jobs[0]["job_id"]+".json")
    failed.write_bytes(canonical_json(dict(queue_sha256=file_hash(queue_path),returncode=1)))
    audit=audit_cohort(protocol_path,queue_path)
    assert audit["planned_contexts"]==4 and len(audit["rows"])==24
    assert audit["unavailable_contexts"]==1
    assert audit["cell_status_counts"]==dict(infrastructure_or_contract_failure=3,not_yet_reported=15,call_not_reached=6)
    assert audit["context_status_counts"]["contains_infrastructure_or_contract_failure"]==1
    # Changed protocol cannot be silently interpreted under the old frozen queue.
    protocol_path.write_text("{}")
    with pytest.raises(ValueError,match="identity mismatch"):audit_cohort(protocol_path,queue_path)


def target_fixture():
    spec=dict(budgets=[2,4,8],repeat_budgets=[8],quadrature="trapezoid",arithmetic_dtype="float32")
    decision=dict(numerics={"state":{"Q":spec}},gradient_repeats=2,map_repeats=2,endpoint_repeats=2,alphas=[0,.9,1],
                  denominator_min={"Q":1e-12,"L2":1e-8,"RMS":1e-8})
    pair=dict(all_bitwise_equal=True,pairs=[{}])
    report=dict(modality="state",target="Q",specification=spec,reference_sha256="ref",
        endpoints={"actual":dict(bitwise_equal=True,action_sha256=["ref","ref"]),
                   "baseline":dict(bitwise_equal=True,action_sha256=["base","base"],scores=[{"RMS":.2},{"RMS":.2}])},
        gradient_probes=[dict(repeat=r,alpha=a,rng_unchanged=True) for r in range(2) for a in (0,.9,1)],
        operation_audits=[dict(alpha=a,status="passed") for a in (0,.9,1)],
        gradient_repeatability={str(a):deepcopy(pair) for a in (0,.9,1)},
        budgets=[dict(m=m,repeat=r,rng_unchanged=True,diagnostics=dict(expected_gap=.2,relative_residual=.001,nonfinite_count=0))
                 for m in (2,4,8) for r in range(2 if m==8 else 1)],
        map_repeatability={"8":dict(IG=deepcopy(pair),path_gradient=deepcopy(pair),common_responses=[])},comparisons=[])
    for lo,hi in ((2,4),(2,8),(4,8)):
        for ranking in ("IG","path_gradient"):
            responses=[dict(direction=d,response=r,reference_status="defined",candidate_status="defined",
                normalized_auc_candidate_minus_reference=0,max_absolute_raw_curve_difference=0,raw_auc_candidate_minus_reference=0)
                for d in ("deletion","insertion") for r in ("Q","L2","RMS")]
            report["comparisons"].append(dict(candidate_m=lo,reference_m=hi,ranking=ranking,relative_l1_difference=.001,
                group_spearman=1,rank_status="defined",top5_count=0,top5_overlap=None,top1_match=True,common_responses=responses))
    return report,decision


def test_rms_threshold_scales_by_baseline_and_does_not_average_directions():
    report,decision=target_fixture()
    response=report["comparisons"][0]["common_responses"][2]
    assert response["response"]=="RMS"
    response["max_absolute_raw_curve_difference"]=.003
    checks,_=v6_target(report,CRITERIA,decision)
    failed=[c for c in checks if c["criterion"]=="rms_curve_difference" and c["status"]=="violates"]
    assert len(failed)==1 and failed[0]["threshold"]==.002
    # Deliberately absent repeat-response comparisons are an explicit violation.
    assert any(c["criterion"]=="common_response_repeat_pair_coverage" and c["status"]=="violates" for c in checks)


def test_v6_completion_binds_report_and_distinguishes_missing_raw_artifacts(tmp_path):
    path=tmp_path/"report.json";path.write_text("{}")
    completion=dict(status="complete_diagnostics_not_approval",artifacts_sha256={"report.json":file_hash(path),"raw.pt":"missing"})
    (tmp_path/"completion.json").write_bytes(canonical_json(completion))
    audit=stage_artifact_checks(path,True)
    assert audit["referenced"]==2 and audit["verified"]==1
    assert audit["status"]=="report_bound_raw_artifacts_not_fully_verified"
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match="bind report bytes"):stage_artifact_checks(path,True)


def retry_fixture(tmp_path):
    original=tmp_path/"original";retry=tmp_path/"retry"
    original.mkdir();retry.mkdir()
    protocol=protocol_v4();protocol["strata"][0].update(episodes=1,evaluate_calls=[0])
    protocol_path=tmp_path/"protocol.json";protocol_path.write_bytes(canonical_json(protocol))
    jobs=[dict(job_id="fixture-170m-e000-c000-"+target,context_id="context",source_manifest_sha256="manifest",
        source_sidecar_sha256="sidecar") for target in ("logpi","l2")]
    queue=dict(protocol_sha256=file_hash(protocol_path),jobs=jobs,validator_sha256="validator",unavailable=[])
    original_path=original/"queue.json";original_path.write_bytes(canonical_json(queue))
    (original/"completion").mkdir()
    failed=original/"completion"/(jobs[0]["job_id"]+".json")
    failed.write_bytes(canonical_json(dict(queue_sha256=file_hash(original_path),job_id=jobs[0]["job_id"],returncode=1)))
    amendment=dict(recorded_before_retry=True,amendment_id="fixture_execution_repair",v4=dict(
        parent_queue_sha256=file_hash(original_path),retry_root=str(retry)))
    amendment_path=tmp_path/"amendment.json";amendment_path.write_bytes(canonical_json(amendment))
    retry_queue={**queue,"parent_queue_sha256":file_hash(original_path),"repair_amendment_sha256":file_hash(amendment_path),
        "prior_attempts":[dict(job_id=job["job_id"],original_completion_sha256=file_hash(failed) if index==0 else None,
            status_before_retry="failed_before_numerical_measurements" if index==0 else "not_started") for index,job in enumerate(jobs)]}
    retry_path=retry/"queue.json";retry_path.write_bytes(canonical_json(retry_queue))
    return protocol_path,original_path,retry_path,amendment_path


def test_linked_execution_retry_retains_failure_without_duplicating_population(tmp_path):
    protocol,original,retry,amendment=retry_fixture(tmp_path)
    audit=audit_cohort(protocol,original,retry_queues=[retry],retry_amendments=[amendment])
    assert audit["planned_contexts"]==1 and len(audit["rows"])==6
    assert audit["cell_status_counts"]=={"not_yet_reported":6}
    assert audit["attempt_status_counts"]["infrastructure_or_contract_failure"]==1
    row=next(r for r in audit["rows"] if r["target"]=="logpi")
    assert len(row["attempts"])==2 and row["attempts"][0]["returncode"]==1
    assert audit["repair_lineage"][0]["amendment_id"]=="fixture_execution_repair"


@pytest.mark.parametrize("corruption",["context","amendment","completed"])
def test_retry_cannot_replace_identity_or_successful_result(tmp_path,corruption):
    protocol,original,retry,amendment=retry_fixture(tmp_path)
    if corruption=="context":
        value=json.loads(retry.read_text());value["jobs"][0]["context_id"]="other";retry.write_bytes(canonical_json(value))
    elif corruption=="amendment":
        value=json.loads(amendment.read_text());value["v4"]["parent_queue_sha256"]="other";amendment.write_bytes(canonical_json(value))
    else:
        path=next((original.parent/"completion").glob("*.json"));value=json.loads(path.read_text());value["returncode"]=0;path.write_bytes(canonical_json(value))
    with pytest.raises(ValueError,match="identity|amendment|successful"):
        retry_link(original,retry,amendment,file_hash(protocol),4)


def test_complete_v6_cohort_authenticates_six_targets_and_preserves_partial_corruption(tmp_path):
    write=lambda p,v:p.write_bytes(canonical_json(v))
    criteria=dict(repeatability="bitwise equal",coordinates="L1 difference <=0.01",group_order="Spearman >=0.99 top5 overlap >=0.95",
        completeness="relative residual <=0.01",primitive_response="RMS curve difference <=0.01 RMS area difference <=0.005",
        normalized_response="area difference <=0.01")
    protocol=dict(protocol_version=6,strata=[dict(id="fixture")],validation=dict(criteria=criteria))
    protocol_path=tmp_path/"protocol.json";write(protocol_path,protocol)
    for name in ("banks","decisions","results","prepared","completion"):(tmp_path/name).mkdir()
    job_id="fixture-e000-c000"
    bank_path=tmp_path/"banks/fixture.json";write(bank_path,dict(contexts=[dict(episode=0,policy_call_idx=0,status="available",context_id="ctx")]))
    target,decision=target_fixture();spec=decision["numerics"]["state"]["Q"]
    decision["numerics"]={m:{t:spec for t in ("Q","L2")} for m in ("vision","language","state")}
    decision["source_replay_settings"]={"source":"declared"}
    decision_path=tmp_path/"decisions"/(job_id+".json");write(decision_path,decision)
    helpers=dict(paired_comparison="paired_comparison.py",integrated_gradients="integrated_gradients.py",
        precision="scripts/validate_downstream_precision.py",repeatability="scripts/validate_gradient_repeatability.py",
        replay="faithfulness.py",cache_runtime="fp32_probe_cache.py")
    source={v:"hash-"+k for k,v in helpers.items()};source["scripts/validate_fp32_probe.py"]="validator"
    queue=dict(protocol_sha256=file_hash(protocol_path),source_sha256=source,jobs=[dict(job_id=job_id,context_id="ctx",
        planned_status="available",bank_sha256=file_hash(bank_path),decision_sha256=file_hash(decision_path))])
    queue_path=tmp_path/"queue.json";write(queue_path,queue)
    prepared=tmp_path/"prepared"/job_id;prepared.mkdir()
    common=dict(decision_sha256=file_hash(decision_path),bank_sha256=file_hash(bank_path),source_metrics_sha256="metrics")
    write(prepared/"report.json",common)
    write(prepared/"completion.json",dict(status="complete_source_preparation",artifacts_sha256={"report.json":file_hash(prepared/"report.json")}))
    out=tmp_path/"results"/job_id;out.mkdir();outcomes=[]
    for modality in ("vision","language","state"):
        for label in ("Q","L2"):
            value=deepcopy(target);value.update(context_id="ctx",modality=modality,target=label,status="complete_diagnostics_not_approval")
            path=out/(modality+"-"+label+".json");write(path,value)
            outcomes.append(dict(modality=modality,target=label,status=value["status"],report_file=path.name,report_sha256=file_hash(path)))
    runtime=dict(cublas_workspace_config=":4096:8",deterministic_algorithms=True,deterministic_warn_only=False,cudnn_benchmark=False,
        cudnn_deterministic=True,float32_matmul_precision="highest",cuda_matmul_allow_tf32=False,cudnn_allow_tf32=False)
    report=dict(**common,script_sha256="validator",helper_sha256={k:source[v] for k,v in helpers.items()},
        preparation_completion_sha256=file_hash(prepared/"completion.json"),status="complete_diagnostics_not_approval",
        startup=dict(torch_not_preimported=True),runtime=dict(settings=runtime),math_settings=dict(autocast="disabled"),
        settings_transition=dict(source_replay=decision["source_replay_settings"]),contexts=[dict(episode=0,policy_call_idx=0,
            source_context_id="ctx",source_replay_verified=True,numerics=outcomes)])
    write(out/"report.json",report)
    write(out/"completion.json",dict(status=report["status"],artifacts_sha256={"report.json":file_hash(out/"report.json")}))
    write(tmp_path/"completion"/(job_id+".json"),dict(queue_sha256=file_hash(queue_path),job_id=job_id,returncode=0))
    audit=audit_cohort(protocol_path,queue_path)
    assert audit["cell_status_counts"]=={"complete_diagnostics":6}
    assert all(any(c["criterion"]=="candidate_runtime_bundle" and c["status"]=="satisfies" for c in r["checks"]) for r in audit["rows"])
    # A late corrupt report cannot retain earlier favorable rows or duplicate cells.
    (out/"state-L2.json").write_text("{}")
    audit=audit_cohort(protocol_path,queue_path)
    assert len(audit["rows"])==6 and audit["cell_status_counts"]=={"invalid_or_incomplete_evidence":6}
