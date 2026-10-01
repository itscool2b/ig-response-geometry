"""E05 conditional populations, parameter clusters and two-family inference."""
from copy import deepcopy
import itertools

import numpy as np
import pytest

import paired_study_analysis as study
from scripts import calibrate_paired_intervals as calibration
from experiment_io import file_hash,object_hash
from test_paired_study_analysis import design as base_design,bank_and_rows,data_fixture,allocation_fixture


def design(stage="variance_only_pilot",n=3):
    value=base_design(stage)
    value["e05"]=deepcopy(study.E05_SCOPE)
    value["e05_contrasts"]=study.family_registry(value,study.SPECIFICITY_FAMILY)
    value["family_weights"]={study.FAMILY:5/7,study.SPECIFICITY_FAMILY:2/7}
    for i,s in enumerate(value["strata"]):
        s.update(parameter_draws=3,e05_gate_sha256="e"*64,reset_seeds=list(range(100000*i,100000*i+n)))
    value["analysis"]=dict(method="stratified_episode_percentile_bootstrap",draws=336000,seed=71,mc_repeats=2,
        min_tail_draws=100,tail_relative_mcse=.1,endpoint_mc_fraction_h=.1,coverage_simulation_alpha=.05)
    value["frozen_scales"]={f"{g['id']}:{m}":2. for g in value["groups"] for m in study.MODALITIES}
    return study.validate_design(value)


def control_rows():
    bank,base=bank_and_rows();rows=[]
    for row in base:
        for r in range(3):
            result=deepcopy(row["results"]);old=result["rankings"]
            result["rankings"]={"trained_Q_IG":deepcopy(old["Q_IG"]),"trained_L2_IG":deepcopy(old["L2_IG"]),
                "permuted_Q_IG":deepcopy(old[f"random_{r:04d}"]),"permuted_L2_IG":deepcopy(old[f"random_{r:04d}"])}
            rows.append(dict(row,draw_index=r,results=result,
                parameter_draw_sha256=object_hash(dict(episode=row["episode_id"],draw=r)),
                randomized_reference_sha256=object_hash(dict(context=row["source_context_id"],draw=r))))
    return bank,base,rows


def test_selected_family_counts_and_equal_marginal_alpha_preserve_base_registry():
    value=design();original=base_design()
    assert value["primary_contrasts"]==original["primary_contrasts"]
    assert len(value["e05_contrasts"])==24
    for family,count in ((study.FAMILY,60),(study.SPECIFICITY_FAMILY,24)):
        alpha=value["global_alpha"]*value["family_weights"][family]
        assert alpha/count==pytest.approx(.05/84)
        assert study.bootstrap_requirement(alpha,count)==336000
    bad=deepcopy(value);bad["family_weights"][study.FAMILY]=1.
    with pytest.raises(ValueError,match="family weights|alpha allocation"):
        study.validate_design(bad)


def test_parameter_mc_uses_episode_vectors_and_preserves_cross_modality_covariance():
    bank,_,rows=control_rows();item=study.specificity_stratum_data(rows,bank,3)
    ep=item["episodes"][1];vectors=ep["draw_call_effects"].mean(axis=1)
    expected=np.cov(vectors,rowvar=False,ddof=1)/3
    assert np.allclose(ep["parameter_mc_covariance"],expected)
    naive=sum(np.cov(ep["draw_call_effects"][:,j],rowvar=False,ddof=1)/3 for j in range(2))/4
    assert expected[0,4]>0 and not np.allclose(expected,naive)
    assert np.allclose(item["matrix"][1],vectors.mean(axis=0))
    assert len(item["membership"][1]["parameter_draws"])==3


def test_one_failed_draw_excludes_whole_e05_episode_without_altering_base():
    bank,base,rows=control_rows();before=study.stratum_data(base,bank)
    row=next(r for r in rows if r["episode_id"]=="episode1" and r["draw_index"]==2)
    row.update(status="numerical_failure",failure_kind="nonfinite_gradient")
    after=study.specificity_stratum_data(rows,bank,3)
    assert before["matrix"].shape==(3,30) and after["matrix"].shape==(2,12)
    assert after["planned_episodes"]==before["planned_episodes"]==3
    assert after["membership"][1]["conditional_episode_weight"]==0
    assert after["membership"][0]["conditional_episode_weight"]==.5
    assert "nonfinite_gradient" in str(after["failure_counts"])
    assert np.array_equal(study.stratum_data(base,bank)["matrix"],before["matrix"])


@pytest.mark.parametrize("failure",[False,True])
def test_actual_completed_producer_loader_to_separate_e05_population(tmp_path,failure):
    """Use real tensor artifacts, writer, completion hashes and both strict loaders."""
    import paired_comparison as paired
    import weight_arrangement_control as control
    from test_weight_arrangement_control import completed_control_fixture
    output,base=completed_control_fixture(tmp_path,failure=failure)
    rows,manifest,completion=control.load_completed_control(output,base_path=base)
    base_rows,base_manifest,_=paired.load_completed_study(base)
    before=study.stratum_data(base_rows,base_manifest["configuration"]["bank"])
    item=study.specificity_stratum_data(rows,manifest["configuration"]["bank"],2)
    assert completion["expected_rows"]==6
    assert before["matrix"].shape==(1,30)
    assert item["matrix"].shape==(0 if failure else 1,12)
    assert item["planned_episodes"]==item["planned_contexts"]==1
    assert item["membership"][0]["joint_complete"] is (not failure)
    assert len(item["membership"][0]["parameter_draws"])==2
    if failure:
        assert any("randomized_self_reference" in reason for reason in item["failure_counts"])
        assert item["membership"][0]["conditional_episode_weight"]==0
    else:
        vectors=item["episodes"][0]["draw_episode_effects"]
        assert np.allclose(item["episodes"][0]["parameter_mc_covariance"],np.cov(vectors,rowvar=False,ddof=1)/2)
    assert np.array_equal(study.stratum_data(base_rows,base_manifest["configuration"]["bank"])["matrix"],before["matrix"])


def test_draw_identity_membership_and_trained_anchor_must_be_unchanged():
    bank,_,rows=control_rows()
    with pytest.raises(ValueError,match="Incomplete prospective"):
        study.specificity_stratum_data(rows[:-1],bank,3)
    bad=deepcopy(rows);bad[1]["parameter_draw_sha256"]="a"*64
    with pytest.raises(ValueError,match="changed within an episode"):
        study.specificity_stratum_data(bad,bank,3)
    bad=deepcopy(rows)
    bad[1]["results"]["rankings"]["trained_Q_IG"]=deepcopy(bad[1]["results"]["rankings"]["trained_L2_IG"])
    with pytest.raises(ValueError,match="changed across parameter draws"):
        study.specificity_stratum_data(bad,bank,3)


def test_episode_coherent_parameter_noise_does_not_disappear_by_adding_identical_calls():
    episodes=[]
    for e in range(5):
        values=np.tile(np.array([-1.,0.,1.])[:,None,None],(1,2,12))+e
        means=values.mean(axis=1)
        episodes.append(dict(total_calls=10,effects=values.mean(axis=0),draw_call_effects=values,
            draw_episode_effects=means,parameter_mc_covariance=np.cov(means,rowvar=False)/3))
    data=dict(episodes=episodes,matrix=np.stack([e["effects"].mean(axis=0) for e in episodes]))
    components=study.specificity_variance_components(data)
    one=study.candidate_specificity_variance(components,1,3)
    all_calls=study.candidate_specificity_variance(components,10,3)
    assert np.allclose(one["mc_variance_per_parameter_draw"],1.)
    assert np.array_equal(one["mc_variance_per_parameter_draw"],all_calls["mc_variance_per_parameter_draw"])


def test_finite_population_parameter_mc_matches_every_call_subset_expectation():
    """Independent enumeration checks both MC identities against all five calls."""
    rng=np.random.default_rng(817)
    r,n,j,width=7,5,3,12
    array=rng.normal(size=(r,n,width))+2*rng.normal(size=(r,1,width))+np.arange(n)[None,:,None]
    population_full_mc=np.var(array.mean(axis=1),axis=0,ddof=1)
    population_centered=array-array.mean(axis=1,keepdims=True)
    population_within_mc=np.var(population_centered,axis=0,ddof=1).sum(axis=0)/(n-1)
    full_estimates=[];within_estimates=[]
    for subset in itertools.combinations(range(n),j):
        values=array[:,subset,:];draw_means=values.mean(axis=1)
        episode=dict(total_calls=n,effects=values.mean(axis=0),draw_call_effects=values,
            draw_episode_effects=draw_means,parameter_mc_covariance=np.cov(draw_means,rowvar=False,ddof=1)/r)
        item=dict(episodes=[episode,deepcopy(episode)],matrix=np.tile(draw_means.mean(axis=0),(2,1)))
        components=study.specificity_variance_components(item)
        assert not np.any(components["parameter_full_mean_negative_moment_counts"])
        full_estimates.append(components["mc_full_mean"][0])
        within_estimates.append(components["mc_within"][0])
    assert len(full_estimates)==10
    assert np.allclose(np.mean(full_estimates,axis=0),population_full_mc,rtol=1e-13,atol=1e-13)
    assert np.allclose(np.mean(within_estimates,axis=0),population_within_mc,rtol=1e-13,atol=1e-13)


def toy_coverage(value,family,data):
    plan=dict(kind="prospective_exact_count_coverage_plan",schema_version=1,family=family,
        generator_source_sha256="a"*64,analysis_module_sha256=file_hash(study.__file__),regimes=["toy"],simulation_stages=[10000],
        stage_derivation=dict(simulation_alpha=.025,look_weights=[1.],limit=.05*value["family_weights"][family]))
    value["analysis"].setdefault("coverage_plans",{})[family]=plan
    counts={s:len(item["matrix"]) for s,item in data.items()}
    return dict(family=family,status="approved",scope_sha256=object_hash(study.calibration_scope(value,counts,family)),
        coverage_plan_sha256=object_hash(plan),complete_episode_counts=counts,count_conditioning="exact_joint_complete_episode_counts",
        assumptions="test-only distribution",generator_source_sha256="a"*64,simulation_family_alpha=.025,
        planned_scenario_ids=["toy"],look_weights=[1.],scenarios=[dict(id="toy",description="toy",simulations=10000,familywise_noncoverage=0)])


def test_family_specific_counts_and_population_intersection_survive_combined_inference(monkeypatch):
    value=design("confirmatory_locked");base=data_fixture();bank,_,rows=control_rows()
    rows[0].update(status="numerical_failure",failure_kind="test_failure")
    item=study.specificity_stratum_data(rows,bank,3);specificity={s:deepcopy(item) for s in base}
    families={study.FAMILY:base,study.SPECIFICITY_FAMILY:specificity}
    coverage={f:toy_coverage(value,f,data) for f,data in families.items()}
    monkeypatch.setattr(study,"_bootstrap",lambda d,x,n,s:np.zeros((4,60)))
    monkeypatch.setattr(study,"_bootstrap_family",lambda d,x,n,s,f:np.zeros((4,24)))
    report=study.confirmatory_study_report(value,families,coverage)
    assert len(report["family_reports"][study.FAMILY]["primary_results"])==60
    assert len(report["family_reports"][study.SPECIFICITY_FAMILY]["primary_results"])==24
    assert report["completion_overlap"]["a0"]["base_only_complete"]==1
    assert "no_cross_family_independence" in report["coverage_interpretation"]
    wrong=deepcopy(coverage);wrong[study.SPECIFICITY_FAMILY]=wrong[study.FAMILY]
    with pytest.raises(ValueError,match="exact observed|matching"):
        study.confirmatory_study_report(value,families,wrong)


def test_specificity_synthetic_truth_and_covariance_are_independent_of_base_completion():
    value=design("confirmatory_locked",n=1800)
    value["strata"][0]["calls_per_episode"]=4
    value["strata"][0]["parameter_draws"]=12
    counts={s["id"]:1800 for s in value["strata"]}
    data,truth,unconditional,diagnostics=calibration.generate_trial(value,"asymmetric_completion",8,0,counts,study.SPECIFICITY_FAMILY)
    estimate=np.concatenate([sum(w*data[s]["matrix"].mean(axis=0) for s,w in g["weights"].items()) for g in value["groups"]])
    se=np.concatenate([np.sqrt(sum(w*w*data[s]["matrix"].var(axis=0,ddof=1)/len(data[s]["matrix"]) for s,w in g["weights"].items())) for g in value["groups"]])
    assert np.max(abs(estimate-truth)/se)<5
    assert np.max(abs(truth-unconditional))>.001
    assert diagnostics["a0"]["parameter_seed_unit"]=="episode"
    assert np.mean(data["a0"]["parameter_mc_covariances"][:,0,4])>0


def test_two_family_prospective_plans_bind_different_counts_and_budget_spending():
    value=design("confirmatory_locked",n=100)
    value["analysis"]["coverage_plans"]={f:calibration.make_coverage_plan(value,seed=13,look_weights=[1.],
        reference_rate=.5*.05*value["family_weights"][f],joint_success_targets=[],family=f) for f in study.inferential_families(value)}
    frozen=deepcopy(value)
    for family,k in ((study.FAMILY,55),(study.SPECIFICITY_FAMILY,41)):
        counts={s["id"]:k for s in value["strata"]}
        protocol=calibration.instantiate_coverage_protocol(value,counts,family)
        assert protocol["family"]==family and protocol["stage_derivation"]["simulation_alpha"]==.025
        assert protocol["complete_episode_counts"]==counts
        protocol["regimes"].pop()
        assert value==frozen
    changed=deepcopy(value)
    changed["analysis"]["coverage_plans"][study.SPECIFICITY_FAMILY]["stage_derivation"]["simulation_alpha"]=.05
    with pytest.raises(ValueError,match="stage counts|confidence allocation"):
        calibration.instantiate_coverage_protocol(changed,{s["id"]:41 for s in value["strata"]},study.SPECIFICITY_FAMILY)


def test_e05_pilot_is_variance_only_and_uses_the_shared_base_scale_population():
    value=design();bank,_,rows=control_rows();item=study.specificity_stratum_data(rows,bank,3)
    data={s["id"]:deepcopy(item) for s in value["strata"]}
    report=study.specificity_pilot_report(value,data,data_fixture(),parameter_draws=3,
        draws=12,seed=19,upper_quantile=.9,call_budgets=[1,2,8])
    assert report["no_signed_efficacy_estimates"] and report["family"]==study.SPECIFICITY_FAMILY
    assert len(report["primary_contrasts"])==24
    assert report["strata"]["a0"]["evaluated_parameter_draws"]==3
    assert len(report["strata"]["a0"]["frontier"][0]["mc_variance_per_parameter_draw"])==12
    def walk(obj):
        if isinstance(obj,dict):
            assert not {"estimate","effects","draw_call_effects","draw_episode_effects"}.intersection(obj)
            for value in obj.values():walk(value)
        elif isinstance(obj,list):
            for value in obj:walk(value)
    walk(report)


def test_two_family_allocation_keeps_failure_budgets_and_counts_collector_once():
    value=design();_,base,choices,costs=allocation_fixture()
    value["planning"]=dict(failure_confidence_alpha=.05,recruitment_shortfall_alpha=.05,pilot_max_looks=3)
    second=deepcopy(base);second["primary_contrasts"]=value["e05_contrasts"]
    extra_costs={s:dict(collection_seconds=1.,fixed_probe_seconds_per_call=2.,parameter_seconds_per_draw_call=1.,
        parameter_seconds_per_draw_episode=.5) for s in choices}
    for row in second["strata"].values():
        row.pop("evaluated_random_permutations");row["evaluated_parameter_draws"]=3
        for frontier in row["frontier"]:
            frontier["non_mc_variance_upper"]=[.003]*12
            frontier.pop("mc_variance_per_permutation")
            frontier.pop("mc_variance_per_permutation_upper")
            frontier["mc_variance_per_parameter_draw_upper"]=[0.]*12
    pilots={study.FAMILY:base,study.SPECIFICITY_FAMILY:second};cost_by_family={study.FAMILY:costs,study.SPECIFICITY_FAMILY:extra_costs}
    result=study.allocate_two_family_episodes(value,pilots,choices,cost_by_family,max_episodes=100,permutations=3,parameter_draws=3)
    assert result["precision_constraints_met"]
    assert result["approximate_cost_seconds"]==16.5*sum(result["planned_episodes"].values())
    assert all(r["failure_planning"]["inferential_family_count"]==2 for r in result["family_allocations"].values())
    with pytest.raises(ValueError,match="Changed R"):
        study.allocate_two_family_episodes(value,pilots,choices,cost_by_family,max_episodes=100,permutations=3,parameter_draws=4)


def test_e05_completed_loader_requires_the_original_base_shard_and_full_membership(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sys
    value=design();design_sha="f"*64;registry=dict(global_design_sha256=design_sha,strata=[],e05_strata=[])
    loaded={};base_data={}
    for spec in value["strata"]:
        sid=spec["id"];bank,base,rows=control_rows()
        bank.update(task=spec["task"],model=spec["model"],pipeline=dict(checkpoint=dict(sha256=spec["checkpoint_sha256"])))
        for entry in bank["contexts"]:
            entry["context_id"]=sid+entry["context_id"];entry["reset_seed"]=spec["reset_seeds"][entry["episode"]]
        for row in (*base,*rows):row["source_context_id"]=sid+row["source_context_id"]
        for row in rows:row["parameter_draw_sha256"]=object_hash(dict(stratum=sid,episode=row["episode_id"],draw=row["draw_index"]))
        base_data[sid]=study.stratum_data(base,bank)
        paths=[tmp_path/f"{sid}-base.jsonl",tmp_path/f"{sid}-e05.jsonl"]
        entries=[]
        for path in paths:
            for suffix in ("",".manifest.json",".completion.json"):type(path)(str(path)+suffix).write_text("{}")
            entries.append(dict(id=sid,metrics=str(path),metrics_sha256=file_hash(path),manifest_sha256=file_hash(str(path)+".manifest.json"),
                completion_sha256=file_hash(str(path)+".completion.json"),bank_sha256="b"*64,protocol_sha256="c"*64))
        a,b=entries;registry["strata"].append(a)
        b.update(base_metrics=str(paths[0]),base_metrics_sha256=a["metrics_sha256"],base_manifest_sha256=a["manifest_sha256"],
            base_completion_sha256=a["completion_sha256"],base_protocol_sha256=a["protocol_sha256"])
        registry["e05_strata"].append(b)
        protocol=dict(study.E05_SCOPE,global_design_sha256=design_sha,stage=value["stage"],parameter_draws=3)
        loaded[str(paths[1])]=(rows,dict(configuration=dict(protocol=protocol,bank=bank,bank_sha256="b"*64,
            protocol_sha256="c"*64,e05_gate_sha256=spec["e05_gate_sha256"])),{})
    calls=[]
    def loader(path,*,base_path):calls.append(str(base_path));return loaded[str(path)]
    monkeypatch.setitem(sys.modules,"weight_arrangement_control",SimpleNamespace(load_completed_control=loader))
    result=study.load_specificity_study(value,registry,design_sha,base_data)
    assert len(calls)==6 and all(v["matrix"].shape==(3,12) for v in result.values())
    bad=deepcopy(registry);bad["e05_strata"][0]["base_metrics"]=registry["strata"][1]["metrics"]
    with pytest.raises(ValueError,match="outside its authenticated"):
        study.load_specificity_study(value,bad,design_sha,base_data)
    loaded[registry["e05_strata"][0]["metrics"]][1]["configuration"]["protocol"]["tie_rule"]="randomized_ties"
    with pytest.raises(ValueError,match="scientific null/reference/tie"):
        study.load_specificity_study(value,registry,design_sha,base_data)


def test_e05_simulation_uses_exact_production_bootstrap_seeds_and_first_interval(monkeypatch):
    value=design("confirmatory_locked",n=10);family=study.SPECIFICITY_FAMILY
    truth,_=calibration.analytic_truth(value,"gaussian_clustered",family);seeds=[]
    def bootstrap(d,data,draws,seed,f):
        assert f==family;seeds.append(seed)
        return np.tile(truth+(1. if len(seeds)==1 else 0.),(4,1))
    monkeypatch.setattr(study,"_bootstrap_family",bootstrap)
    result=calibration.simulate_one(value,"gaussian_clustered",13,0,{s["id"]:2 for s in value["strata"]},family=family)
    assert seeds==[71,72] and result["familywise_noncoverage"]
    assert result["family"]==family and len(result["confidence_interval"])==24
    assert result["seed_sha256"]!=calibration.trial_seed_hash(13,"gaussian_clustered",0)
