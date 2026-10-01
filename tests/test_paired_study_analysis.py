"""Scientific estimand, provenance, uncertainty and planning tests; CPU only."""
from copy import deepcopy
import itertools
import json

import numpy as np
import pytest
from scipy.integrate import trapezoid

import paired_study_analysis as study
from experiment_io import file_hash,object_hash


def design(stage="variance_only_pilot"):
    groups=[dict(id="rdt170m_four_task",weights={f"a{i}":.25 for i in range(4)}),
            dict(id="authored1b_two_task",weights={f"b{i}":.5 for i in range(2)})]
    strata=[dict(id=f"{letter}{i}",task=f"task{i}",model=model,checkpoint_sha256=sha*64,
                 reset_seeds=[100*k+j for j in range(3)],calls_per_episode=2,random_permutations=3,e01_gate_sha256="d"*64)
            for k,(letter,i,model,sha) in enumerate([( "a",i,"170m","a") for i in range(4)]+
                                                   [("b",i,"1b","b") for i in range(2)],1)]
    value=dict(schema_version=1,kind="paired_global_design",stage=stage,groups=groups,strata=strata,
        modalities=list(study.MODALITIES),estimand=study.ESTIMAND,max_episode_steps=400,
        global_alpha=.05,family_weights=dict(primary_rms=1.),secondary_inference="descriptive_only_no_scale_effect",
        episode_dependence="disjoint_reset_streams_across_strata",failure_policy="joint_complete_episodes_no_replacement")
    value["primary_contrasts"]=study.primary_registry(value)
    return value


def result(factor=1.,exact=False):
    reference=[[[0.]*8]]
    table={}
    weights=np.array([.125,.375,.75])*factor
    for size in range(4):
        for subset in itertools.combinations(range(3),size):
            action=np.full((1,1,8),weights[list(subset)].sum(),dtype=np.float32).tolist()
            rms=float(np.asarray(action)[0,0,0])
            table[object_hash(list(subset))]=dict(status="finite",baseline_positions=list(subset),
                active_action=action,active_action_sha256=study.tensor_list_hash(action),RMS=rms)
    orders=dict(Q_IG=[2,1,0],L2_IG=[1,2,0],input_difference=[0,1,2])
    if exact:
        orders["random_exact"]=None
    else:
        orders.update(random_0000=[0,1,2],random_0001=[1,2,0],random_0002=[2,0,1])
    rankings={}
    for label,order in orders.items():
        curves={}
        for direction in study.DIRECTIONS:
            ids=[]
            for count in range(4):
                subsets=list(itertools.combinations(range(3),count)) if order is None else [sorted(order[:count])]
                if direction=="insertion":
                    subsets=[sorted(set(range(3))-set(s)) for s in subsets]
                ids.append([object_hash(list(s)) for s in subsets])
            values=[sum(table[k]["RMS"] for k in group)/len(group) for group in ids]
            curves[direction]=dict(response_ids=ids if exact and order is None else [g[0] for g in ids],
                responses=dict(RMS=dict(values=values,raw_auc=float(trapezoid(values,[0,1/3,2/3,1])),
                                         normalized_auc=None,status="defined")))
        rankings[label]=dict(status="defined",curves=curves)
        if order is None:
            rankings[label]["control_kind"]="exact_uniform_subset_expectation"
    return dict(active_reference_action=reference,active_reference_sha256=study.tensor_list_hash(reference),
        active_action_indices=list(range(8)),response_table=table,rankings=rankings,
        realized_fractions=[0,1/3,2/3,1],input_response_id=object_hash([]),baseline_response_id=object_hash([0,1,2]))


def bank_and_rows(lengths=(1,3,2),j=2,exact=False):
    contexts=[]; accounts=[]; rows=[]
    for ep,n in enumerate(lengths):
        selected=list(range(min(j,n)))
        accounts.append(dict(episode=ep,eligible_calls=list(range(n)),permutation=list(range(n)),selected_calls=selected,
                             inclusion_probability=len(selected)/n,within_episode_weight=1/len(selected),episode_weight=1/len(lengths)))
        for call in selected:
            eid=f"episode{ep}"; cid=f"context{ep}_{call}"
            contexts.append(dict(episode=ep,episode_id=eid,policy_call_idx=call,context_id=cid,
                                 reset_seed=100+ep,terminal_record_sha256="c"*64))
            for modality in study.MODALITIES:
                rows.append(dict(episode_id=eid,policy_call_idx=call,source_context_id=cid,modality=modality,
                                 status="evaluated",results=result(1+ep+.25*call,exact=exact)))
    return dict(estimand=study.ESTIMAND,selection=dict(rule="uniform_executed_calls_v1",calls_per_episode=j,max_episode_steps=400),
                selection_accounting=accounts,contexts=contexts),rows


def data_fixture(exact=False):
    bank,rows=bank_and_rows(exact=exact)
    item=study.stratum_data(rows,bank)
    return {s["id"]:deepcopy(item) for s in design()["strata"]}


def add_secondary(item):
    item=deepcopy(item)
    item["nominal_grid_percent"]=[0,100/3,200/3,100]
    item["denominator_min"]={r:0. for r in ("RMS","Q","L2")}
    for row in item["response_table"].values():
        row["Q"]=-row["RMS"]**2/2
        row["L2"]=-float(np.sqrt(8*row["RMS"]**2+1e-12))
    for ranking in item["rankings"].values():
        exact=ranking.get("control_kind")=="exact_uniform_subset_expectation"
        for curve in ranking["curves"].values():
            groups=curve["response_ids"] if exact else [[k] for k in curve["response_ids"]]
            for response in ("RMS","Q","L2"):
                values=[sum(item["response_table"][k][response] for k in keys)/len(keys) for keys in groups]
                actual=item["response_table"][item["input_response_id"]][response]
                baseline=item["response_table"][item["baseline_response_id"]][response]
                normalized=float(trapezoid((np.asarray(values)-baseline)/(actual-baseline),item["realized_fractions"])) if actual!=baseline else None
                curve["responses"][response]=dict(values=values,raw_auc=float(trapezoid(values,item["realized_fractions"])),
                    normalized_auc=normalized,status="defined" if normalized is not None else "zero_endpoint_gap")
    for target in ("Q","L2"):
        item["rankings"][f"{target}_path_gradient"]=deepcopy(item["rankings"][f"{target}_IG"])
    return item


def test_exact_sixty_family_and_fixed_weights_not_six_independent_tests():
    value=design()
    assert len(study.validate_design(value)["primary_contrasts"])==60
    assert study.bootstrap_requirement(.05,60)==240000
    assert study.bootstrap_requirement(.025,60)==480000
    bad=deepcopy(value); bad["groups"][0]["weights"]["a0"]=.5
    with pytest.raises(ValueError,match="equal-task"):
        study.validate_design(bad)
    bad=deepcopy(value); bad["primary_contrasts"].pop()
    with pytest.raises(ValueError,match="60"):
        study.validate_design(bad)


def test_future_roster_must_be_explicit_and_disjoint():
    value=design(); value["strata"][1]["reset_seeds"][0]=value["strata"][0]["reset_seeds"][0]
    with pytest.raises(ValueError,match="collision"):
        study.validate_design(value)
    value=design(); del value["strata"][0]["reset_seeds"]
    with pytest.raises(ValueError,match="roster"):
        study.validate_design(value)


def test_audit_detects_action_and_area_corruption_independently():
    item=result()
    assert study.audit_rms(item)==1.25
    corrupt=deepcopy(item)
    corrupt["response_table"][object_hash([0])]["active_action"][0][0][0]+=1
    with pytest.raises(ValueError,match="hash"):
        study.audit_rms(corrupt)
    corrupt=deepcopy(item)
    corrupt["rankings"]["Q_IG"]["curves"]["deletion"]["responses"]["RMS"]["raw_auc"]+=.01
    with pytest.raises(ValueError,match="area"):
        study.audit_rms(corrupt)


def test_rms_audit_does_not_require_numpy_two_trapezoid(monkeypatch):
    monkeypatch.delattr(np,"trapezoid",raising=False)
    assert study.audit_rms(result())==1.25


def test_per_episode_means_do_not_pool_calls_or_renormalize_task_weights():
    bank,rows=bank_and_rows()
    item=study.stratum_data(rows,bank)
    expected=[]; pooled=[]
    for ep in range(3):
        calls=[]
        for call in range(min(2,[1,3,2][ep])):
            vector=study.context_effects({m:next(r for r in rows if r["episode_id"]==f"episode{ep}" and r["policy_call_idx"]==call and r["modality"]==m) for m in study.MODALITIES})[0]
            calls.append(vector); pooled.append(vector)
        expected.append(np.mean(calls,axis=0))
    assert np.allclose(item["matrix"],expected)
    assert not np.allclose(item["matrix"].mean(axis=0),np.mean(pooled,axis=0))
    assert [m["conditional_episode_weight"] for m in item["membership"]]==[1/3]*3
    assert item["membership"][1]["contexts"][0]["inclusion_probability"]==2/3


def test_one_failed_selected_call_excludes_episode_jointly_without_replacement():
    bank,rows=bank_and_rows()
    row=next(r for r in rows if r["episode_id"]=="episode1" and r["modality"]=="language")
    row["results"]["rankings"]["Q_IG"]={"status":"numerical_failure"}
    item=study.stratum_data(rows,bank)
    assert len(item["matrix"])==2 and item["planned_episodes"]==3
    assert item["membership"][1]["joint_complete"] is False
    assert item["membership"][1]["conditional_episode_weight"]==0
    assert item["failure_counts"]["language:Q_IG:numerical_failure"]==1
    # Baseline scale does not inherit comparative-method exclusions.
    assert len(item["baseline_episodes"]["language"])==3


def test_membership_changes_when_sampled_call_changes_even_with_same_episodes():
    bank,rows=bank_and_rows()
    first=study.stratum_data(rows,bank)
    changed=deepcopy(bank)
    changed["selection_accounting"][1]["permutation"]=[2,1,0]
    changed["selection_accounting"][1]["selected_calls"]=[2,1]
    entry=next(e for e in changed["contexts"] if e["episode"]==1 and e["policy_call_idx"]==0)
    entry["policy_call_idx"]=2; entry["context_id"]="new_context"
    for r in rows:
        if r["episode_id"]=="episode1" and r["policy_call_idx"]==0:
            r["policy_call_idx"]=2; r["source_context_id"]="new_context"
    second=study.stratum_data(rows,changed)
    assert study.object_hash(first["membership"])!=study.object_hash(second["membership"])
    with pytest.raises(ValueError,match="membership"):
        study.stratum_data(rows,bank)


def test_shared_random_orders_retain_covariance_but_exact_control_has_zero_mc():
    rows={m:dict(status="evaluated",results=result()) for m in study.MODALITIES}
    values,mc,_,_,counts=study.context_effects(rows)
    assert mc[0,0]>0 and mc[0,1]==mc[0,0]
    assert mc[0,5]!=0  # deletion/insertion use the same random order
    assert mc[0,10]==0  # independent modality stream
    assert np.allclose(values[:5],values[5:10])
    exact={m:dict(status="evaluated",results=result(exact=True)) for m in study.MODALITIES}
    _,mc,_,_,counts=study.context_effects(exact)
    assert not np.any(mc) and set(counts.values())=={0}


def test_without_replacement_finite_population_and_j1_identifiability():
    data=data_fixture(exact=True)["a0"]
    components=study.variance_components(data)
    one=study.candidate_variance(components,1,2)
    all_calls=study.candidate_variance(components,25,2)
    assert np.all(one["non_mc_variance"]>=all_calls["non_mc_variance"])
    assert np.allclose(all_calls["non_mc_variance"],components["between"])
    bank,rows=bank_and_rows(j=1)
    with pytest.raises(ValueError,match="cannot identify"):
        study.variance_components(study.stratum_data(rows,bank))


def test_pilot_output_has_no_signed_efficacy_estimates_or_reversible_blinding_claim():
    value=design()
    report=study.pilot_report(value,data_fixture(exact=True),draws=25,seed=7,upper_quantile=.9,call_budgets=[1,2,25],permutations=3)
    text=json.dumps(report)
    for forbidden in ('"estimate":','"episode_effects":','"confidence_interval":','"anonymous_contrast_id":'):
        assert forbidden not in text
    assert report["disclosure"]=="labeled_variance_only_not_blinded"
    scale=report["scales"]["rdt170m_four_task:vision"]
    assert scale["halfwidth"]==.05*scale["value"]
    assert scale["substantial_difference"]==.10*scale["value"]
    assert report["strata"]["a0"]["frontier"][-1]["calls_per_episode"]==25


def test_baseline_scale_missingness_cannot_be_hidden_by_complete_case_analysis():
    data=data_fixture()
    data["a0"]["baseline_episodes"]["vision"].pop()
    with pytest.raises(ValueError,match="all planned episodes"):
        study.pilot_report(design(),data,draws=3,seed=7,upper_quantile=.9,call_budgets=[2],permutations=3)


def test_bootstrap_resamples_complete_contrast_vectors_and_fixed_task_weights():
    data=data_fixture(exact=True)
    for i,(sid,item) in enumerate(data.items()):
        item["matrix"]=np.full((3,30),i,dtype=float)
    draws=study._bootstrap(design(),data,29,1)
    assert np.allclose(draws[:,:30],1.5)
    assert np.allclose(draws[:,30:],4.5)


def test_confirmation_rejects_underresolved_global_tails_and_unmatched_calibration():
    value=design("confirmatory_locked")
    value["analysis"]=dict(method="stratified_episode_percentile_bootstrap",draws=10000,seed=1,
        mc_repeats=2,min_tail_draws=100,tail_relative_mcse=.1,endpoint_mc_fraction_h=.1)
    with pytest.raises(ValueError,match="240000"):
        study.confirmatory_report(value,data_fixture(),calibration={})
    value["analysis"]["draws"]=240000
    with pytest.raises(ValueError,match="calibration"):
        study.confirmatory_report(value,data_fixture(),calibration={"status":"approved"})


def test_confirmation_failure_population_and_mc_gate_are_explicit(monkeypatch):
    value=design("confirmatory_locked")
    value["analysis"]=dict(method="stratified_episode_percentile_bootstrap",draws=240000,seed=1,
        mc_repeats=2,min_tail_draws=100,tail_relative_mcse=.1,endpoint_mc_fraction_h=.1)
    value["frozen_scales"]={f"{g['id']}:{m}":2. for g in value["groups"] for m in study.MODALITIES}
    value["analysis"]["coverage_plan"]=dict(kind="prospective_exact_count_coverage_plan",schema_version=1,
        generator_source_sha256="a"*64,analysis_module_sha256=file_hash(study.__file__),regimes=["toy"],
        simulation_stages=[1000],stage_derivation=dict(simulation_alpha=.05,look_weights=[1.],limit=.05))
    # A deliberately noisy endpoint approximation must fail its own gate.
    monkeypatch.setattr(study,"_bootstrap",lambda d,x,n,s:np.full((4,60),s*.1))
    counts={s["id"]:3 for s in value["strata"]}
    calibration=dict(status="approved",scope_sha256=object_hash(study.calibration_scope(value,counts)),
        coverage_plan_sha256=object_hash(value["analysis"]["coverage_plan"]),planned_scenario_ids=["toy"],
        complete_episode_counts=counts,count_conditioning="exact_joint_complete_episode_counts",
        assumptions="toy test distribution only",generator_source_sha256="a"*64,simulation_family_alpha=.05,
        minimum_complete_episodes={s["id"]:2 for s in value["strata"]},
        scenarios=[dict(id="toy",description="toy covered scenario",simulations=1000,familywise_noncoverage=0)])
    report=study.confirmatory_report(value,data_fixture(),calibration=calibration)
    assert len(report["primary_results"])==60
    assert not any(r["endpoint_mc_gate"] for r in report["primary_results"])
    assert all(r["interval_status"]=="precision_gate_failed" for r in report["primary_results"])
    assert "conditional" in report["limitation"]
    assert report["task_weights"]["rdt170m_four_task"]=={f"a{i}":.25 for i in range(4)}


def allocation_fixture():
    value=design()
    value["planning"]=dict(failure_confidence_alpha=.05,recruitment_shortfall_alpha=.05,pilot_max_looks=3)
    pilot=dict(strata={},scales={},primary_contrasts=study.primary_registry(value),stage="variance_only_pilot")
    choices={s["id"]:2 for s in value["strata"]}
    costs={s["id"]:dict(collection_seconds=1.,fixed_probe_seconds_per_call=2.,random_seconds_per_order_call=0.) for s in value["strata"]}
    for sid in choices:
        pilot["strata"][sid]=dict(planned_episodes=55,complete_episodes=55,evaluated_calls_per_episode=2,
            evaluated_random_permutations=[3],zero_observed_variance_cells=[],frontier=[dict(calls_per_episode=2,
            expected_selected_calls=2.,non_mc_variance_upper=[.002]*30,mc_variance_per_permutation=[0.]*30,
            mc_variance_per_permutation_upper=[0.]*30)])
    for group in value["groups"]:
        for modality in study.MODALITIES:
            pilot["scales"][f"{group['id']}:{modality}"]=dict(conservative_planning_halfwidth=.05)
    return value,pilot,choices,costs


def test_allocation_uses_every_contrast_and_measured_cost_without_signed_means():
    value,pilot,choices,costs=allocation_fixture()
    result=study.allocate_episodes(value,pilot,choices,costs,max_episodes=100,permutations=3)
    assert result["precision_constraints_met"] and result["random_mc_constraints_met"]
    assert set(result["planned_episodes"])==set(choices)
    assert all(result["planned_episodes"][s]>result["required_complete_episodes"][s] for s in choices)
    assert result["maximum_halfwidth_to_target_ratio"]<=1
    assert result["approximate_cost_seconds"]==5*sum(result["planned_episodes"].values())
    assert "estimate" not in result
    constrained=study.allocate_episodes(value,pilot,choices,costs,max_episodes=2,permutations=3)
    assert not constrained["precision_constraints_met"]


def test_allocation_does_not_treat_zero_scale_or_observed_variance_as_certainty():
    value,pilot,choices,costs=allocation_fixture()
    pilot["scales"]["rdt170m_four_task:state"]["conservative_planning_halfwidth"]=0
    with pytest.raises(ValueError,match="epsilon"):
        study.allocate_episodes(value,pilot,choices,costs,max_episodes=10,permutations=3)
    value,pilot,choices,costs=allocation_fixture()
    pilot["strata"]["a0"]["zero_observed_variance_cells"]=[1]
    with pytest.raises(ValueError,match="Zero observed"):
        study.allocate_episodes(value,pilot,choices,costs,max_episodes=10,permutations=3)


def test_pilot_stages_are_derived_and_label_their_normal_reference_assumption():
    stages=study.normal_reference_pilot_stages([1.5,1.35,1.25],[.025,.015,.01])
    assert [s["complete_episodes_per_stratum"] for s in stages]==[55,99,178]
    assert all(s["reference_distribution"].startswith("normal") for s in stages)


def test_allocation_cannot_transport_failure_population_when_j_or_m_changes():
    value,pilot,choices,costs=allocation_fixture()
    pilot["strata"]["a0"]["evaluated_calls_per_episode"]=1
    with pytest.raises(ValueError,match="Changed J"):
        study.allocate_episodes(value,pilot,choices,costs,max_episodes=100,permutations=3)
    value,pilot,choices,costs=allocation_fixture()
    with pytest.raises(ValueError,match="Changed M"):
        study.allocate_episodes(value,pilot,choices,costs,max_episodes=100,permutations=6)
    for row in pilot["strata"].values():
        row["evaluated_random_permutations"]=[]
    result=study.allocate_episodes(value,pilot,choices,costs,max_episodes=100,permutations=6)
    assert result["precision_constraints_met"]


def test_secondary_reports_all_geometry_and_component_controls_with_shared_ratio_eligibility():
    bank,rows=bank_and_rows(exact=True)
    for row in rows:
        row["results"]=add_secondary(row["results"])
    output=study.secondary_summaries(rows,bank)
    assert len(output)==18
    raw=output["vision:RMS:raw_auc"]
    ratio=output["vision:RMS:normalized_auc"]
    assert raw["defined_episodes"]==ratio["defined_episodes"]==3
    assert raw["contrasts"][0]["estimate"]>0 and ratio["contrasts"][0]["estimate"]>0
    assert all(c["estimate"]==0 for c in raw["contrasts"] if c["control"].endswith("path_gradient"))
    assert all(r["inference"]=="descriptive_only" for r in output.values())
    for row in rows:
        row["results"]=add_secondary(result(0.,exact=True))
    output=study.secondary_summaries(rows,bank)
    assert output["vision:RMS:raw_auc"]["defined_episodes"]==3
    assert output["vision:RMS:normalized_auc"]["defined_episodes"]==0
    assert output["vision:RMS:normalized_auc"]["area_mean"] is None


def test_global_loader_binds_prospective_roster_and_sealed_output_registry(tmp_path,monkeypatch):
    import paired_comparison
    from experiment_io import file_hash
    value=design(); design_sha="e"*64
    registry=dict(global_design_sha256=design_sha,strata=[])
    loaded={}
    for spec in value["strata"]:
        bank,rows=bank_and_rows()
        sid=spec["id"]
        for entry in bank["contexts"]:
            entry["reset_seed"]=spec["reset_seeds"][entry["episode"]]
            entry["context_id"]=sid+entry["context_id"]
        for row in rows:
            row["source_context_id"]=sid+row["source_context_id"]
        bank.update(task=spec["task"],model=spec["model"],pipeline=dict(checkpoint=dict(sha256=spec["checkpoint_sha256"])))
        path=tmp_path/f"{sid}.jsonl"
        for suffix in ("",".manifest.json",".completion.json"):
            (tmp_path/(path.name+suffix)).write_text("{}")
        config=dict(protocol=dict(global_design_sha256=design_sha,stage=value["stage"],random_permutations=3),bank=bank,
                    bank_sha256="b"*64,protocol_sha256="c"*64,e01_gate_sha256=spec["e01_gate_sha256"])
        loaded[str(path)]=(rows,dict(configuration=config),{})
        registry["strata"].append(dict(id=sid,metrics=str(path),metrics_sha256=file_hash(path),
            manifest_sha256=file_hash(str(path)+".manifest.json"),completion_sha256=file_hash(str(path)+".completion.json"),
            bank_sha256="b"*64,protocol_sha256="c"*64))
    monkeypatch.setattr(paired_comparison,"load_completed_study",lambda p:loaded[str(p)])
    monkeypatch.setattr(study,"secondary_summaries",lambda rows,bank:{})
    assert set(study.load_study(value,registry,design_sha))=={s["id"] for s in value["strata"]}
    changed=deepcopy(value); changed["strata"][0]["reset_seeds"].append(99999)
    with pytest.raises(ValueError,match="full prospective"):
        study.load_study(changed,registry,design_sha)
    (tmp_path/"a0.jsonl").write_text('{"changed":true}')
    with pytest.raises(ValueError,match="artifact mismatch"):
        study.load_study(value,registry,design_sha)


def test_coverage_gate_requires_simulation_uncertainty_not_just_empirical_rate():
    value=design("confirmatory_locked")
    value["analysis"]={"method":"stratified_episode_percentile_bootstrap"}
    value["analysis"]["coverage_plan"]=dict(kind="prospective_exact_count_coverage_plan",schema_version=1,
        generator_source_sha256="a"*64,analysis_module_sha256=file_hash(study.__file__),regimes=["toy"],
        simulation_stages=[10,1000],stage_derivation=dict(simulation_alpha=.05,look_weights=[.5,.5],limit=.05))
    counts={s["id"]:3 for s in value["strata"]}
    calibration=dict(status="approved",scope_sha256=object_hash(study.calibration_scope(value,counts)),
        coverage_plan_sha256=object_hash(value["analysis"]["coverage_plan"]),planned_scenario_ids=["toy"],look_weights=[.5,.5],
        complete_episode_counts=counts,count_conditioning="exact_joint_complete_episode_counts",
        assumptions="toy only",generator_source_sha256="a"*64,simulation_family_alpha=.05,
        minimum_complete_episodes={s["id"]:2 for s in value["strata"]},
        scenarios=[dict(id="toy",description="not enough simulations",simulations=10,familywise_noncoverage=0,look_index=0)])
    with pytest.raises(ValueError,match="upper noncoverage"):
        study.validate_calibration(value,data_fixture(),calibration)
    calibration["scenarios"][0]["simulations"]=1000
    calibration["scenarios"][0]["look_index"]=1
    study.validate_calibration(value,data_fixture(),calibration)
    bad=deepcopy(calibration);bad["scenarios"][0]["simulations"]=1001
    with pytest.raises(ValueError,match="stage size"):
        study.validate_calibration(value,data_fixture(),bad)
    bad=deepcopy(calibration);bad["look_weights"]=[.1,.9]
    with pytest.raises(ValueError,match="probability budgets"):
        study.validate_calibration(value,data_fixture(),bad)
    bad=deepcopy(calibration);bad["planned_scenario_ids"]=["favorable"]
    with pytest.raises(ValueError,match="declared scenario"):
        study.validate_calibration(value,data_fixture(),bad)
    calibration["complete_episode_counts"]["a0"]=4
    with pytest.raises(ValueError,match="exact observed"):
        study.validate_calibration(value,data_fixture(),calibration)


def test_unequal_shards_pool_episodes_and_secondary_standard_errors_not_shard_means():
    bank,rows=bank_and_rows(exact=True)
    for row in rows:
        row["results"]=add_secondary(row["results"])
    whole=study.stratum_data(rows,bank,include_secondary=True)
    shards=[]
    for episodes in ({0},{1,2}):
        shard=deepcopy(bank)
        shard["contexts"]=[e for e in shard["contexts"] if e["episode"] in episodes]
        shard["selection_accounting"]=[e for e in shard["selection_accounting"] if e["episode"] in episodes]
        for account in shard["selection_accounting"]:
            account["episode_weight"]=1/len(episodes)
        subset=[r for r in rows if int(r["episode_id"].removeprefix("episode")) in episodes]
        shards.append(study.stratum_data(subset,shard,include_secondary=True))
    combined=study.combine_shards(shards)
    assert np.array_equal(combined["matrix"],whole["matrix"])
    assert [r["conditional_episode_weight"] for r in combined["membership"]]==[1/3]*3
    for key in whole["secondary"]:
        a,b=whole["secondary"][key],combined["secondary"][key]
        assert np.allclose(a["area_mean"],b["area_mean"])
        assert np.allclose(a["area_standard_error"],b["area_standard_error"])
        for x,y in zip(a["contrasts"],b["contrasts"]):
            assert np.isclose(x["estimate"],y["estimate"])
            assert np.isclose(x["standard_error"],y["standard_error"])
    with pytest.raises(ValueError,match="multiple shards"):
        study.combine_shards([shards[0],shards[0]])
