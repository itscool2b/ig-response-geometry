"""Pilot selection validity, coupled candidate populations and exact scale handoff."""
from copy import deepcopy

import numpy as np
import pytest

import paired_study_analysis as study
import pilot_scale_bounds as scale
import pilot_selection as selector
import pilot_selection_simulation as simulation
from scripts import validate_pilot_selection as validator
from experiment_io import canonical_json,object_hash
from test_weight_specificity_statistics import design as expanded_design


def fixture(n=24):
    design=expanded_design(n=n)
    design["pilot_stages"]=[dict(id="first",episodes_per_stratum={s["id"]:n//2 for s in design["strata"]}),
        dict(id="last",episodes_per_stratum={s["id"]:n for s in design["strata"]})]
    candidates=[dict(id="lower",calls_per_episode=1,random_permutations=3,parameter_draws=3),
        dict(id="higher",calls_per_episode=2,random_permutations=5,parameter_draws=5)]
    costs={c["id"]:{f:{s["id"]:dict(collection_seconds=1.,fixed_probe_seconds_per_call=1.,
        **({"random_seconds_per_order_call":.1} if f==study.FAMILY else {"parameter_seconds_per_draw_call":.1,"parameter_seconds_per_draw_episode":.2}))
        for s in design["strata"]} for f in study.inferential_families(design)} for c in candidates}
    protocol=dict(kind="paired_pilot_selection_protocol",schema_version=1,rule=selector.RULE,stages=deepcopy(design["pilot_stages"]),
        candidates=candidates,costs=costs,scale_look_weights=[.5,.5],scale_alpha=.05,
        failure_confidence_alpha=.05,recruitment_shortfall_alpha=.05,
        bounds=dict(method="empirical_episode_percentile_with_direct_total_guard_v1",draws=1000,upper_quantile=.9,
            minimum_expected_tail_draws=100,seed=192),maximum_sd_upper_to_point_ratio=2.,maximum_scale_relative_width=.9,
        maximum_confirmation_episodes_per_stratum=600,signed_means_allowed=False)
    return study.validate_design(design),selector.validate_protocol(protocol,design)


def plan_for(design,protocol):
    return dict(schema_version=1,kind="full_pilot_rule_validation_plan",source_sha256=validator.source_hashes(),
        design_object_sha256=object_hash(design),selection_protocol_object_sha256=object_hash(protocol),
        regimes=list(simulation.REGIMES),regime_definitions_sha256=object_hash(simulation.REGIMES),seed=291,
        trials_per_regime=1000,simulation_confidence_alpha=.05,
        maximum_risks={key:.1 for key in validator.METRICS},
        minimum_resolution_probability={key:(.5 if key=="gaussian" else 0.) for key in simulation.REGIMES},
        acceptance="all_regime_simultaneous_risk_and_resolution_bounds",domain_statement="synthetic test fixture only",
        arbitrary_tail_guarantee=False)


def test_no_reference_sample_count_or_unregistered_candidate_enters_rule():
    design,protocol=fixture()
    changed=deepcopy(protocol);changed["stages"][0]["episodes_per_stratum"]["a0"]+=1
    with pytest.raises(ValueError,match="entire stage"):selector.validate_protocol(changed,design)
    changed=deepcopy(protocol);changed["candidates"].append(deepcopy(changed["candidates"][0]))
    with pytest.raises(ValueError,match="candidate registry"):selector.validate_protocol(changed,design)
    changed=deepcopy(protocol);changed["bounds"]["draws"]=999
    with pytest.raises(ValueError,match="tail resolution"):selector.validate_protocol(changed,design)


def test_vectorized_variance_bootstrap_equals_direct_episode_resampling():
    rng=np.random.default_rng(871);matrix=rng.normal(size=(7,12));mc=np.linspace(.001,.007,7)
    data=dict(matrix=matrix,episodes=[dict(effects=np.zeros((1,12)),total_calls=2,parameter_mc_covariance=np.eye(12)*v) for v in mc])
    report=selector.moment_bounds(data,study.SPECIFICITY_FAMILY,budget=3,draws=123,seed=7,upper_quantile=.9)
    weights=np.random.default_rng(7).multinomial(7,np.full(7,1/7),size=123)
    variances=np.array([matrix[np.repeat(np.arange(7),w)].var(axis=0,ddof=1) for w in weights])
    assert np.allclose(report["total_upper"],np.maximum(matrix.var(axis=0,ddof=1),np.quantile(variances,.9,axis=0)))
    assert not report["component_diagnostics"]["identified"]
    assert np.all(np.array(report["allocation_non_mc_upper"])+np.array(report["mc_upper"])>=np.array(report["total_upper"])-1e-14)
    assert not {"estimate","mean","effects"}.intersection(report)


def test_all_call_scale_handoff_and_candidate_prefixes_retain_actual_rosters():
    design,protocol=fixture();counts=protocol["stages"][-1]["episodes_per_stratum"]
    data,bank=simulation.generate_panel(design,"gaussian",protocol["candidates"],counts,seed=31,trial=0,stream="pilot")
    first_data,first_bank=simulation.stage_prefix(data,bank,protocol["stages"][0]["episodes_per_stratum"])
    result=selector.evaluate_stage(design,protocol,"first",first_data,first_bank)
    assert result["stage_index"]==0 and result["selected_candidate"] is None
    expected=scale.simultaneous_scale_report(design,first_bank,"first",[.5,.5],.05)
    assert result["scale_report"]==expected
    for sid in first_data["lower"][study.FAMILY]:
        members=first_data["lower"][study.FAMILY][sid]["membership"]
        assert len(members)==12 and all(m["selected_calls"]==1 for m in members)
        assert expected["strata"][sid]["planned_calls"]==sum(m["executed_calls"] for m in members)
    bad=deepcopy(first_data);bad.pop("higher")
    with pytest.raises(ValueError,match="All frozen candidates"):selector.evaluate_stage(design,protocol,"first",bad,first_bank)
    bad=deepcopy(first_data);bad["lower"][study.FAMILY]["a0"]["membership"][0]["selected_calls"]=2
    with pytest.raises(ValueError,match="call budget"):selector.evaluate_stage(design,protocol,"first",bad,first_bank)
    bad=deepcopy(first_data);item=bad["lower"][study.FAMILY]["a0"]
    item["episodes"].append(deepcopy(item["episodes"][0]));item["matrix"]=np.concatenate([item["matrix"],item["matrix"][:1]])
    with pytest.raises(ValueError,match="complete population"):selector.evaluate_stage(design,protocol,"first",bad,first_bank)
    bad=deepcopy(first_data);bad["lower"][study.FAMILY]["a0"]["matrix"][[0,1]]=bad["lower"][study.FAMILY]["a0"]["matrix"][[1,0]]
    with pytest.raises(ValueError,match="ordered effect means"):selector.evaluate_stage(design,protocol,"first",bad,first_bank)
    bad=deepcopy(first_data);episode=bad["lower"][study.FAMILY]["a0"]["episodes"][0]
    episode["effects"]=np.tile(episode["effects"],(2,1))
    with pytest.raises(ValueError,match="effect shape"):selector.evaluate_stage(design,protocol,"first",bad,first_bank)
    bad=deepcopy(first_data);bad["lower"][study.FAMILY]["a0"]["episodes"][0]["effects"]=np.empty((0,30))
    with pytest.raises(ValueError,match="effect shape"):selector.evaluate_stage(design,protocol,"first",bad,first_bank)


def test_candidate_completion_truth_changes_with_j_m_r_not_just_mc_division():
    design,protocol=fixture();low,high=protocol["candidates"]
    a=simulation.truth(design,"quality_completion",low);b=simulation.truth(design,"quality_completion",high)
    for family in study.inferential_families(design):
        assert a[family]["a0"]["completion"]>b[family]["a0"]["completion"]
        assert not np.allclose(a[family]["a0"]["mean"],b[family]["a0"]["mean"])
        assert not np.allclose(a[family]["a0"]["non_mc"],b[family]["a0"]["non_mc"])


def test_generator_conditional_means_and_variance_match_analytic_truth():
    design,protocol=fixture(n=3000);design["strata"]=design["strata"][:1]
    candidates=[protocol["candidates"][1]];counts={"a0":3000}
    data,_=simulation.generate_panel(design,"quality_completion",candidates,counts,seed=103,trial=2,stream="pilot")
    known=simulation.truth(design,"quality_completion",candidates[0])
    for family in study.inferential_families(design):
        matrix=data["higher"][family]["a0"]["matrix"];truth=known[family]["a0"]
        se=np.sqrt(truth["total"]/len(matrix))
        assert np.max(abs(matrix.mean(axis=0)-truth["mean"])/se)<5
        assert np.max(abs(matrix.var(axis=0,ddof=1)/truth["total"]-1))<.13


def test_relative_mc_uses_lower_non_mc_bound_even_when_upper_would_pass():
    design,_=fixture();reports={};allocation=dict(family_allocations={},planned_episodes={s["id"]:100 for s in design["strata"]})
    for family in study.inferential_families(design):
        width=len(study.local_contrasts(family));reports[family]=dict(strata={},scales={})
        allocation["family_allocations"][family]=dict(required_complete_episodes={s["id"]:100 for s in design["strata"]})
        for s in design["strata"]:reports[family]["strata"][s["id"]]=dict(moment_bounds=dict(non_mc_lower=[.0001]*width,mc_upper=[.001]*width))
        for g in design["groups"]:
            for m in study.MODALITIES:reports[family]["scales"][f"{g['id']}:{m}"]=dict(conservative_planning_halfwidth=1.)
    passed,_=selector._relative_mc_pass(design,reports,allocation)
    assert not passed


def test_nonproportional_extra_complete_episodes_cannot_invalidate_mc_guard():
    design,_=fixture();reports={};allocation=dict(family_allocations={},planned_episodes={s["id"]:1000 for s in design["strata"]})
    for family in study.inferential_families(design):
        width=len(study.local_contrasts(family));reports[family]=dict(strata={},scales={})
        allocation["family_allocations"][family]=dict(required_complete_episodes={s["id"]:10 for s in design["strata"]})
        for i,s in enumerate(design["strata"]):
            low,mc=(100.,0.) if i%2==0 else (1.,1.)
            reports[family]["strata"][s["id"]]=dict(moment_bounds=dict(non_mc_lower=[low]*width,mc_upper=[mc]*width))
        for g in design["groups"]:
            for m in study.MODALITIES:reports[family]["scales"][f"{g['id']}:{m}"]=dict(conservative_planning_halfwidth=100.)
    # At both minima the two-task MC ratio is 1/101 < .01. Extra complete
    # low-MC episodes reduce its denominator and can make that ratio unsafe.
    assert 1/101<.01
    passed,diagnostics=selector._relative_mc_pass(design,reports,allocation)
    assert not passed
    assert all(v>0 for v in diagnostics[study.FAMILY+":authored1b_two_task"]["worst_case_relative_variance_excess"])


def test_candidate_completion_budget_covers_the_whole_inspected_grid(monkeypatch):
    design,protocol=fixture(n=24);counts=protocol["stages"][-1]["episodes_per_stratum"]
    data,bank=simulation.generate_panel(design,"gaussian",protocol["candidates"],counts,seed=31,trial=0,stream="pilot")
    scales=scale.simultaneous_scale_report(design,bank,"last",[.5,.5],.05)["scales"]
    captured=[]
    def allocate(d,reports,*args,**kwargs):
        captured.append(d["planning"])
        return dict(precision_constraints_met=True,planned_episodes=counts,family_allocations={f:dict(required_complete_episodes=counts) for f in study.inferential_families(d)})
    monkeypatch.setattr(study,"allocate_two_family_episodes",allocate)
    result=selector.assess_candidate(design,protocol,protocol["candidates"][1],data["higher"],scales,1)
    assert captured and captured[0]["failure_confidence_alpha"]==.025
    assert captured[0]["pilot_max_looks"]==2
    assert result["completion_bound_candidate_count"]==2


def test_cost_tie_break_is_frozen_and_uses_all_feasible_candidates(monkeypatch):
    design,protocol=fixture(n=24);counts=protocol["stages"][-1]["episodes_per_stratum"]
    data,bank=simulation.generate_panel(design,"gaussian",protocol["candidates"],counts,seed=31,trial=0,stream="pilot")
    visited=[]
    def assess(d,p,c,*args):
        visited.append(c["id"])
        return dict(candidate=c,feasible=True,allocation=dict(approximate_cost_seconds=12.))
    monkeypatch.setattr(selector,"assess_candidate",assess)
    result=selector.evaluate_stage(design,protocol,"last",data,bank)
    assert visited==["lower","higher"] and result["selected_candidate"]=="higher"


def test_degenerate_variance_report_is_finite_json_and_unresolved():
    values=np.zeros((3,12))
    data=dict(matrix=values,episodes=[dict(effects=np.zeros((1,12)),total_calls=2,parameter_mc_covariance=np.zeros((12,12))) for _ in range(3)])
    report=selector.moment_bounds(data,study.SPECIFICITY_FAMILY,budget=3,draws=20,seed=7,upper_quantile=.9)
    assert report["maximum_sd_upper_to_point_ratio"] is None
    assert report["zero_total_variance_cells"]==list(range(12))
    canonical_json(report)


def test_rule_stops_at_first_selected_stage_and_uses_independent_future_stream(monkeypatch):
    design,protocol=fixture();visited=[];streams=[]
    real_generate=simulation.generate_panel
    def generate(*args,**kwargs):streams.append(kwargs["stream"]);return real_generate(*args,**kwargs)
    monkeypatch.setattr(simulation,"generate_panel",generate)
    def stage(d,p,stage_id,*unused):
        visited.append(stage_id)
        allocation=dict(planned_episodes={s["id"]:2 for s in d["strata"]},family_allocations={f:dict(
            required_complete_episodes={s["id"]:2 for s in d["strata"]},completion_probability_lower={s["id"]:0. for s in d["strata"]}) for f in study.inferential_families(d)})
        scales={f"{g['id']}:{m}":dict(value=1.,lower=0.,upper=None) for g in d["groups"] for m in study.MODALITIES}
        return dict(stage_id=stage_id,status="selected",selected_candidate="lower",selected_allocation=allocation,
            scale_report=dict(scales=scales),scale_reasons=[],candidates=[])
    monkeypatch.setattr(selector,"evaluate_stage",stage)
    monkeypatch.setattr(simulation,"validate_future",lambda *args:{f:dict(shortfall=False,halfwidth_failure=False,population_halfwidth_failure=False,true_mc_failure=False,
        empirical_mc_failure=False,endpoint_mc_failure=False,interval_noncoverage=False) for f in study.inferential_families(design)})
    row=simulation.simulate_rule_trial(design,protocol,"gaussian",31,0)
    assert visited==["first"] and streams==["pilot","confirmation"] and row["selected"]


def test_exact_future_kernel_uses_production_streams_and_first_interval(monkeypatch):
    design,protocol=fixture(n=6);candidate=protocol["candidates"][0];counts={s["id"]:6 for s in design["strata"]}
    data,_=simulation.generate_panel(design,"gaussian",[candidate],counts,seed=31,trial=0,stream="confirmation")
    known=simulation.truth(design,"gaussian",candidate);calls=[]
    def bootstrap(d,x,n,seed,f=study.FAMILY):
        calls.append((f,seed));center=1. if seed==71 else 0.
        return np.full((4,len(study.family_registry(d,f))),center)
    monkeypatch.setattr(study,"_bootstrap",bootstrap);monkeypatch.setattr(study,"_bootstrap_family",bootstrap)
    allocation=dict(family_allocations={f:dict(required_complete_episodes=counts) for f in study.inferential_families(design)})
    scales={f"{g['id']}:{m}":dict(value=1.) for g in design["groups"] for m in study.MODALITIES}
    result=simulation.validate_future(design,candidate,allocation,scales,data["lower"],known)
    assert calls==[(study.FAMILY,71),(study.FAMILY,72),(study.SPECIFICITY_FAMILY,71),(study.SPECIFICITY_FAMILY,72)]
    assert all(r["interval_noncoverage"] and r["endpoint_mc_failure"] for r in result.values())


def test_validation_cannot_select_favorable_regimes_or_approve_always_inconclusive():
    design,protocol=fixture();plan=plan_for(design,protocol)
    validator.validate_plan(plan,design,protocol)
    changed=deepcopy(plan);changed["regimes"].pop()
    with pytest.raises(ValueError,match="Every prespecified regime"):validator.validate_plan(changed,design,protocol)
    changed=deepcopy(plan);changed["minimum_resolution_probability"]={key:0. for key in simulation.REGIMES}
    with pytest.raises(ValueError,match="always-inconclusive"):validator.validate_plan(changed,design,protocol)
    row=dict(selected=False,elapsed_seconds=0.,future_interval_noncoverage=False,**{key:False for key in validator.METRICS})
    rows=[dict(row,regime=r,trial=i) for r in simulation.REGIMES for i in range(1000)]
    assert validator.summarize(plan,rows)["status"]=="failed"
    assert validator.summarize(plan,rows[:1])["status"]=="incomplete_not_approval"


def test_missing_and_zero_scale_cannot_be_replaced_or_epsilon_repaired():
    design,protocol=fixture();counts=protocol["stages"][-1]["episodes_per_stratum"]
    for regime in ("zero_scale","missing_endpoint"):
        data,bank=simulation.generate_panel(design,regime,protocol["candidates"],counts,seed=31,trial=0,stream="pilot")
        result=selector.evaluate_stage(design,protocol,"last",data,bank)
        assert result["status"]=="inconclusive" and result["selected_candidate"] is None
        assert result["scale_reasons"]


def test_bounded_full_rule_runner_resumes_all_regimes_and_rejects_trial_mutation(tmp_path,monkeypatch):
    from concurrent.futures import Future
    design,protocol=fixture();plan=plan_for(design,protocol);calls=[]
    def trial(d,p,regime,seed,index):
        calls.append((regime,index))
        return dict(kind="test_fixture_no_simulation",regime=regime,trial=index,
            seed_sha256=object_hash(dict(seed=seed,regime=regime,trial=index)),selected=False,
            future_interval_noncoverage=False,elapsed_seconds=0.,**{key:False for key in validator.METRICS})
    class Pool:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,*args):
            future=Future();future.set_result(fn(*args));return future
    monkeypatch.setattr(validator,"ProcessPoolExecutor",Pool)
    monkeypatch.setattr(simulation,"simulate_rule_trial",trial)
    report=validator.run(design,protocol,plan,tmp_path,workers=1,max_trials=3)
    assert report["status"]=="incomplete_not_approval"
    assert calls==[(regime,0) for regime in list(simulation.REGIMES)[:3]]
    validator.run(design,protocol,plan,tmp_path,workers=1,max_trials=2,resume=True)
    assert calls==[(regime,0) for regime in list(simulation.REGIMES)[:5]]
    first=next(tmp_path.glob("trial_*.json"));first.write_bytes(first.read_bytes().replace(b'"selected":false',b'"selected":true'))
    with pytest.raises(ValueError,match="completed synthetic trial changed"):
        validator.run(design,protocol,plan,tmp_path,workers=1,max_trials=1,resume=True)
