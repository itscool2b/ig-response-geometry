"""Matched panels, direct-risk accounting and a non-approving operational cap."""
from copy import deepcopy

import pytest

import paired_study_analysis as study
import pilot_rule_characterization as matched
from scripts import characterize_pilot_rules as runner
from experiment_io import object_hash,strict_json
from test_pilot_selection import fixture


def setup():
    design,protocol=fixture()
    fixed=deepcopy(protocol);fixed["candidates"]=fixed["candidates"][1:]
    fixed["costs"]={"higher":fixed["costs"]["higher"]}
    rules=[dict(id="adaptive",protocol=protocol),dict(id="fixed",protocol=fixed)]
    plan=dict(schema_version=2,kind="matched_pilot_rule_characterization_plan",acceptance=runner.CONTRACT,
        can_approve=False,source_sha256=runner.source_hashes(),design_object_sha256=object_hash(design),
        rules_object_sha256=object_hash(rules),regimes=list(runner.simulation.REGIMES),
        regime_definitions_sha256=object_hash(runner.simulation.REGIMES),seed=132,trials_per_regime=2,
        workers=2,wall_seconds=5400,simulation_confidence_alpha=.05,
        risk_targets=dict(unconditional_direct_unsafe=.1,conditional_direct_unsafe_given_selected=.1),
        minimum_resolution_probability={key:(.8 if key=="gaussian" else 0.) for key in runner.simulation.REGIMES},
        exact_count_coverage_gate_required=True,arbitrary_tail_guarantee=False,domain_statement="test domain")
    return design,rules,plan


def fake_row(rules,regime="gaussian",trial=0,seed=132,selected=True,unsafe=False):
    values={}
    for rule in rules:
        value=dict(selected=selected,selected_candidate="higher" if selected else None,
            direct_unsafe_selection=unsafe,**{key:False for key in runner.DIAGNOSTICS})
        value["future_precision_failure"]=unsafe
        values[rule["id"]]=value
    return dict(regime=regime,trial=trial,rules=values,
        seed_sha256=object_hash(dict(seed=seed,regime=regime,trial=trial)),elapsed_seconds=0.)


def test_characterization_can_never_approve_even_perfect_complete_results():
    design,rules,plan=setup();runner.validate_plan(plan,design,rules)
    rows=[fake_row(rules,r,t) for r in plan["regimes"] for t in range(2)]
    result=runner.summarize(plan,rules,rows)
    assert result["status"]=="characterization_complete_not_approval" and not result["can_approve"]
    assert result["simulation_local_alpha"]==.05/(2*13*3)
    changed=deepcopy(plan);changed["can_approve"]=True
    with pytest.raises(ValueError,match="never approves"):runner.validate_plan(changed,design,rules)
    changed=deepcopy(plan);changed["exact_count_coverage_gate_required"]=False
    with pytest.raises(ValueError,match="Coverage"):runner.validate_plan(changed,design,rules)


def test_internal_bound_failures_and_undercoverage_remain_visible_without_becoming_precision_events():
    _,rules,plan=setup();row=fake_row(rules)
    for value in row["rules"].values():
        value["inspected_variance_bound_failure"]=True;value["future_interval_noncoverage"]=True
    result=runner.summarize(plan,rules,[row])["rules"]["adaptive"]["gaussian"]
    assert result["direct_unsafe_count"]==0
    assert result["diagnostic_counts"]["inspected_variance_bound_failure"]==1
    assert result["diagnostic_counts"]["future_interval_noncoverage"]==1
    assert "indispensable" in result["interval_noncoverage_role"]


def test_no_selection_does_not_imply_zero_conditional_risk():
    _,rules,plan=setup();row=fake_row(rules,selected=False)
    cell=runner.summarize(plan,rules,[row])["rules"]["fixed"]["gaussian"]
    risk=cell["conditional_direct_unsafe_given_selected"]
    assert risk["upper"]==1 and risk["status"]=="unresolved_no_selected_trials" and not risk["target_met"]
    assert not cell["resolution"]["target_met"]


def test_partial_regime_cannot_infer_risk_from_informatively_censored_completions():
    _,rules,plan=setup()
    row=fake_row(rules)
    partial=runner.summarize(plan,rules,[row])["rules"]["fixed"]["gaussian"]
    assert not partial["fixed_regime_roster_complete"]
    assert "runtime_censoring_can_be_informative" in partial["bound_interpretation"]
    assert all(partial[key]["target_met"] is None for key in
        ("unconditional_direct_unsafe","conditional_direct_unsafe_given_selected","resolution"))
    completed=runner.summarize(plan,rules,[row,fake_row(rules,trial=1)])["rules"]["fixed"]["gaussian"]
    assert completed["fixed_regime_roster_complete"]
    assert completed["bound_interpretation"]=="fixed_planned_binomial_denominator"
    assert type(completed["unconditional_direct_unsafe"]["target_met"]) is bool


def test_actual_shared_union_panel_is_consumed_by_both_rules(monkeypatch):
    design,rules,_=setup();seen=[]
    def assess(d,p,panel,endpoints,known):
        seen.append((id(panel),id(endpoints),id(panel["higher"]),set(panel)))
        return dict(selected=False,selected_candidate=None,inspected_variance_bound_failure=False,scale_bound_failure=False),None
    monkeypatch.setattr(matched,"assess_rule",assess)
    result=matched.simulate_matched_trial(design,rules,"gaussian",132,0)
    assert seen[0]==seen[1] and seen[0][-1]=={"higher","lower"}
    assert result["confirmation_endpoint_panel_sha256"] is None
    assert all(not r["direct_unsafe_selection"] for r in result["rules"].values())


def test_confirmation_is_shared_then_cut_to_each_preselected_attempt_roster(monkeypatch):
    design,rules,_=setup();generated=[];observed=[]
    original=matched.sim.generate_panel
    def generate(*args,**kwargs):
        generated.append((kwargs["stream"],args[3]));return original(*args,**kwargs)
    monkeypatch.setattr(matched.sim,"generate_panel",generate)
    def assess(d,p,*unused):
        cid=p["candidates"][0]["id"];n=3 if cid=="lower" else 5
        counts={s["id"]:n for s in d["strata"]}
        allocation=dict(planned_episodes=counts,approximate_cost_seconds=12.,family_allocations={f:dict(
            required_complete_episodes=counts,completion_probability_lower={s:0. for s in counts}) for f in study.inferential_families(d)})
        selected=dict(selected_candidate=cid,stage_id="first",selected_allocation=allocation,scale_report=dict(scales={}),candidates=[])
        return dict(selected=True,inspected_variance_bound_failure=True,scale_bound_failure=False),selected
    def future(d,c,a,scales,data,known):
        observed.append({s:[m["episode_id"] for m in v["membership"]] for s,v in data[study.FAMILY].items()})
        return {f:dict(shortfall=False,halfwidth_failure=False,population_halfwidth_failure=True,true_mc_failure=False,
            empirical_mc_failure=False,endpoint_mc_failure=False,interval_noncoverage=True) for f in study.inferential_families(d)}
    monkeypatch.setattr(matched,"assess_rule",assess);monkeypatch.setattr(matched.sim,"validate_future",future)
    result=matched.simulate_matched_trial(design,rules,"gaussian",132,0)
    assert [x[0] for x in generated]==["pilot","confirmation"]
    assert all(observed[0][s]==observed[1][s][:3] for s in observed[0])
    assert all(r["direct_unsafe_selection"] and not r["future_precision_failure"] and r["future_population_precision_failure"] for r in result["rules"].values())


def test_shared_candidate_identity_cannot_change_across_rules():
    _,rules,_=setup();rules[1]["protocol"]["candidates"][0]["parameter_draws"]+=1
    with pytest.raises(ValueError,match="different settings"):matched.union_candidates(rules)


def test_peak_memory_identity_is_the_current_actual_process():
    import os
    result=runner.process_peak_memory()
    assert result["pid"]==os.getpid() and result["parent_pid"]==os.getppid()
    assert result["scope"]=="process_lifetime_peak_at_completed_job_return"
    assert (result["status"]=="observed" and result["peak_resident_bytes"]>0) or result.get("reason")


def test_ledger_binds_complete_pairs_and_rejects_edits(tmp_path):
    _,rules,plan=setup();identity="a"*64;row=fake_row(rules)
    runner._append(tmp_path,row,identity,"0"*64)
    rows,_=runner._load_rows(tmp_path,identity,plan,rules);assert len(rows)==1
    path=next(tmp_path.glob("trial_*.json"));path.write_bytes(path.read_bytes().replace(b'"selected":true',b'"selected":false'))
    with pytest.raises(ValueError,match="Completed matched trial changed"):runner._load_rows(tmp_path,identity,plan,rules)


def test_deadline_stops_workers_preserves_denominator_and_never_approves(tmp_path,monkeypatch):
    design,rules,plan=setup();plan["wall_seconds"]=.03;clock=[0.];processes=[]
    def tick():clock[0]+=.005;return clock[0]
    class Process:
        def __init__(self,**kwargs):self.alive=False;self.exitcode=None;processes.append(self)
        def start(self):self.alive=True
        def is_alive(self):return self.alive
        def join(self,**kwargs):pass
        def terminate(self):self.alive=False;self.exitcode=-15
        def kill(self):self.alive=False;self.exitcode=-9
    class Context:pass
    Context.Process=Process
    monkeypatch.setattr(runner.mp,"get_context",lambda name:Context())
    monkeypatch.setattr(runner.time,"monotonic",tick);monkeypatch.setattr(runner.time,"sleep",lambda value:None)
    result=runner.run(design,rules,plan,tmp_path)
    assert result["wall_limit_reached"] and result["complete_matched_pairs"]==0 and not result["can_approve"]
    assert all(not p.alive for p in processes)
    operations=[strict_json(p) for p in tmp_path.glob("operation_*.json")]
    assert len(operations)==1 and len(operations[0]["interrupted_trials"])==2
    assert result["rules"]["adaptive"]["gaussian"]["planned_trials"]==2


def test_completed_worker_path_records_actual_pid_and_resumes_without_duplicate_trials(tmp_path,monkeypatch):
    import os
    design,rules,plan=setup();calls=[]
    def simulate(d,r,regime,seed,trial):
        calls.append((regime,trial));return fake_row(r,regime,trial,seed)
    class Process:
        def __init__(self,target,args):self.target=target;self.args=args;self.exitcode=None
        def start(self):self.target(*self.args);self.exitcode=0
        def is_alive(self):return False
        def join(self,**kwargs):pass
    class Context:pass
    Context.Process=Process
    monkeypatch.setattr(runner.mp,"get_context",lambda name:Context())
    monkeypatch.setattr(runner.matched,"simulate_matched_trial",simulate)
    report=runner.run(design,rules,plan,tmp_path)
    assert report["status"]=="characterization_complete_not_approval" and len(calls)==26
    row=strict_json(next(tmp_path.glob("trial_*.json")))
    assert row["worker_memory"]["pid"]==os.getpid()
    assert row["worker_elapsed_seconds_before_serialization"]>=0
    runner.run(design,rules,plan,tmp_path,resume=True)
    assert len(calls)==26
