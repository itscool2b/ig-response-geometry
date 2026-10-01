"""Independent synthetic truth, multiplicity and simulation-protocol tests."""
from copy import deepcopy
from concurrent.futures import Future
import json

import numpy as np
import pytest

import paired_study_analysis as study
from scripts import calibrate_paired_intervals as calibration
from experiment_io import file_hash,object_hash


def design(n=6,j=2,m=8):
    value=dict(schema_version=1,kind="paired_global_design",stage="confirmatory_locked",
        groups=[dict(id="170m",weights={f"a{i}":.25 for i in range(4)}),dict(id="1b",weights={f"b{i}":.5 for i in range(2)})],
        strata=[dict(id=f"{letter}{i}",task=f"task{i}",model=model,checkpoint_sha256=sha*64,
            reset_seeds=list(range(k*n,(k+1)*n)),calls_per_episode=j,random_permutations=m,e01_gate_sha256="d"*64)
            for k,(letter,i,model,sha) in enumerate([("a",i,"170m","a") for i in range(4)]+[("b",i,"1b","b") for i in range(2)],1)],
        modalities=list(study.MODALITIES),estimand=study.ESTIMAND,max_episode_steps=400,global_alpha=.05,
        family_weights=dict(primary_rms=1.),secondary_inference="descriptive_only_no_scale_effect",
        episode_dependence="disjoint_reset_streams_across_strata",failure_policy="joint_complete_episodes_no_replacement",
        analysis=dict(method="stratified_episode_percentile_bootstrap",draws=240000,seed=11,mc_repeats=2,
            min_tail_draws=100,tail_relative_mcse=.1,endpoint_mc_fraction_h=.1))
    value["primary_contrasts"]=study.primary_registry(value)
    value["frozen_scales"]={f"{g['id']}:{modality}":1. for g in value["groups"] for modality in study.MODALITIES}
    return value


def protocol(value,mode="candidate_scope"):
    derivation=dict(look_weights=[1.],simulation_alpha=.05,limit=.05,reference_rate=.025,joint_success_targets=[])
    counts={s["id"]:len(s["reset_seeds"]) for s in value["strata"]} if mode=="candidate_scope" else None
    if mode=="candidate_scope":
        value["analysis"]["coverage_plan"]=calibration.make_coverage_plan(value,seed=129,
            look_weights=derivation["look_weights"],reference_rate=derivation["reference_rate"],joint_success_targets=[])
        return calibration.instantiate_coverage_protocol(value,counts)
    return dict(schema_version=1,kind="paired_interval_calibration_protocol",mode=mode,
        scope_sha256=object_hash(study.calibration_scope(value,counts)),analysis_module_sha256=file_hash(study.__file__),
        complete_episode_counts=counts,count_conditioning="exact_joint_complete_episode_counts" if counts else "planned_episode_sampling",
        generator_source_sha256=file_hash(calibration.__file__),regime_definitions_sha256=object_hash(calibration.REGIMES),
        regimes=list(calibration.REGIMES),stage_derivation=derivation,
        simulation_stages=calibration.derive_simulation_stages(scenario_count=len(calibration.REGIMES),**derivation),
        seed=129,minimum_complete_episodes={s["id"]:2 for s in value["strata"]})


def test_five_contrasts_obey_shared_method_identities_in_both_directions():
    areas=np.random.default_rng(1).normal(size=(30,3,2,4))
    contrasts=calibration.contrast_vector(areas).reshape(30,3,2,5)
    assert np.allclose(contrasts[...,4],contrasts[...,1]-contrasts[...,0])
    assert np.allclose(contrasts[...,4],contrasts[...,3]-contrasts[...,2])
    assert np.allclose(contrasts[:,0,0,0],areas[:,0,0,0]-areas[:,0,0,3])
    assert np.allclose(contrasts[:,0,1,0],areas[:,0,1,3]-areas[:,0,1,0])


def test_simulation_streams_are_reproducible_and_separate_regimes_and_trials():
    value=design(n=10)
    first=calibration.generate_trial(value,"gaussian_clustered",31,0)
    repeat=calibration.generate_trial(value,"gaussian_clustered",31,0)
    other=calibration.generate_trial(value,"gaussian_clustered",31,1)
    for sid in first[0]:
        assert np.array_equal(first[0][sid]["matrix"],repeat[0][sid]["matrix"])
    assert not np.array_equal(first[0]["a0"]["matrix"],other[0]["a0"]["matrix"])
    assert calibration.stream_seed(31,"gaussian_clustered",0,"data")!=calibration.stream_seed(31,"student_t3",0,"data")


def test_conditional_truth_accounts_for_length_call_budget_and_mask_exposure():
    probabilities=calibration.length_probabilities("asymmetric_completion",0)
    a=calibration.conditional_quality_mean("asymmetric_completion",probabilities,1,8)
    b=calibration.conditional_quality_mean("asymmetric_completion",probabilities,4,32)
    assert 0<a<b<1
    counts=np.minimum(4,np.arange(1,26))
    plus=np.dot(probabilities,.998**(counts*33)); minus=np.dot(probabilities,.988**(counts*33))
    assert b==(plus-minus)/(plus+minus)
    data,truth,unconditional,_=calibration.generate_trial(design(n=1600,j=4,m=32),"asymmetric_completion",13,0)
    estimate=np.concatenate([sum(w*data[s]["matrix"].mean(axis=0) for s,w in g["weights"].items()) for g in design()["groups"]])
    se=np.concatenate([np.sqrt(sum(w*w*data[s]["matrix"].var(axis=0,ddof=1)/len(data[s]["matrix"]) for s,w in g["weights"].items())) for g in design()["groups"]])
    assert np.max(abs(estimate-truth)/se)<5
    assert np.max(abs(truth-unconditional))>.025


def test_near_degenerate_target_difference_has_small_variance_without_epsilon_floor():
    data,truth,_,_=calibration.generate_trial(design(n=100),"near_degenerate",2,0)
    matrix=data["a0"]["matrix"]
    assert matrix[:,4].var()<1e-12
    assert matrix[:,0].var()>1e-4
    assert truth[4]==0


def test_simulation_repeats_do_not_choose_a_more_favorable_interval(monkeypatch):
    value=design(n=10)
    _,truth,_,_=calibration.generate_trial(value,"gaussian_clustered",3,0)
    calls=[]
    def fake_bootstrap(d,data,draws,seed):
        calls.append(seed)
        return np.tile(truth+(1. if len(calls)==1 else 0.),(4,1))
    monkeypatch.setattr(study,"_bootstrap",fake_bootstrap)
    record=calibration.simulate_one(value,"gaussian_clustered",3,0,{s["id"]:2 for s in value["strata"]})
    assert record["familywise_noncoverage"] and not record["endpoint_mc_gate"]
    assert calls==[value["analysis"]["seed"],value["analysis"]["seed"]+1]
    assert np.allclose(np.array(record["confidence_interval"])[:,0],truth+1)


def test_insufficient_conditional_population_is_not_dropped_from_coverage(monkeypatch):
    value=design()
    original=calibration.generate_trial(value,"gaussian_clustered",1,0)
    original[0]["a0"]["matrix"]=np.empty((0,30))
    monkeypatch.setattr(calibration,"generate_trial",lambda *args:original)
    result=calibration.simulate_one(value,"gaussian_clustered",1,0,{s["id"]:2 for s in value["strata"]})
    assert result["familywise_noncoverage"]
    assert result["status"]=="insufficient_conditional_population"


def test_binomial_stages_are_derived_and_spend_alpha_across_all_scenarios():
    value=design(); p=protocol(value)
    first=p["simulation_stages"][0]
    alpha=.05/len(calibration.REGIMES)
    assert calibration.binomial_bounds(0,first,alpha)[1]<=.05
    assert calibration.binomial_bounds(0,first-1,alpha)[1]>.05
    row=dict(familywise_noncoverage=False,status="evaluated",endpoint_mc_gate=True,halfwidth_targets_met=True)
    summary=calibration.stage_summary([row]*first,p,0)
    assert summary["decision"]=="passed" and summary["simulation_alpha"]==alpha


def test_protocol_cannot_drop_heavy_tail_regime_or_change_bootstrap_scope():
    value=design(); p=protocol(value)
    calibration.validate_protocol(p,value)
    bad=deepcopy(p); bad["regimes"].remove("student_t3")
    with pytest.raises(ValueError,match="prospective calibration plan"):
        calibration.validate_protocol(bad,value)
    changed=deepcopy(value); changed["analysis"]["draws"]*=2
    with pytest.raises(ValueError,match="scope"):
        calibration.validate_protocol(p,changed)
    changed=deepcopy(p); changed["simulation_stages"][0]-=1
    with pytest.raises(ValueError,match="prospective calibration plan"):
        calibration.validate_protocol(changed,value)


class InlinePool:
    def __init__(self,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def submit(self,function,*args):
        future=Future()
        future.set_result(function(*args))
        return future


def test_characterization_never_becomes_approval_and_all_regimes_are_saved(tmp_path,monkeypatch):
    value=design(); p=protocol(value,mode="characterization_only")
    monkeypatch.setattr(calibration,"ProcessPoolExecutor",InlinePool)
    def toy(design,regime,seed,trial,minimum,counts=None):
        truth,unconditional=calibration.analytic_truth(design,regime)
        return dict(regime=regime,trial=trial,status="evaluated",familywise_noncoverage=False,
            endpoint_mc_gate=True,halfwidth_targets_met=True,elapsed_seconds=0.,truth=truth.tolist(),
            unconditional_truth=unconditional.tolist(),confidence_interval=np.stack([truth-1,truth+1],axis=1).tolist(),
            noncovering_contrast_indices=[],seed_sha256=object_hash(dict(seed=seed,regime=regime,trial=trial)))
    monkeypatch.setattr(calibration,"simulate_one",toy)
    out=tmp_path/"calibration"
    report=calibration.run_calibration(value,p,out,workers=1,resume=False,design_sha256="a"*64,protocol_sha256="b"*64)
    assert report["status"]=="characterization_only" and report["statistical_status"]=="passed"
    assert set(report["planned_scenario_ids"])==set(calibration.REGIMES)
    assert len(report["trials"])==p["simulation_stages"][0]*len(calibration.REGIMES)
    # A missing export is recovered from its durable journal transaction.
    first=out/report["trials"][0]["path"]
    original=first.read_bytes();first.unlink()
    calibration.run_calibration(value,p,out,workers=1,resume=True,design_sha256="a"*64,protocol_sha256="b"*64)
    assert first.read_bytes()==original
    first.write_text('{"modified":true}')
    with pytest.raises(ValueError,match="export was modified"):
        calibration.run_calibration(value,p,out,workers=1,resume=True,design_sha256="a"*64,protocol_sha256="b"*64)
    first.write_bytes(original)
    changed=deepcopy(p);changed["seed"]+=1
    with pytest.raises(ValueError,match="Resume identity"):
        calibration.run_calibration(value,changed,out,workers=1,resume=True,design_sha256="a"*64,protocol_sha256="b"*64)
    ledger=out/"trial_ledger.jsonl"
    lines=ledger.read_text().splitlines();entry=json.loads(lines[0]);entry["entry_sha256"]="0"*64
    lines[0]=json.dumps(entry);ledger.write_text("\n".join(lines)+"\n")
    with pytest.raises(ValueError,match="hash chain"):
        calibration.run_calibration(value,p,out,workers=1,resume=True,design_sha256="a"*64,protocol_sha256="b"*64)


def test_operational_first_look_bound_preserves_the_full_protocol_and_unresolved_regimes(tmp_path,monkeypatch):
    value=design()
    p=calibration.characterization_protocol(value,seed=73,look_weights=[.5,.5],reference_rate=.025,joint_success_targets=[.8])
    original=deepcopy(p)
    monkeypatch.setattr(calibration,"ProcessPoolExecutor",InlinePool)
    def toy(design,regime,seed,trial,minimum,counts=None):
        truth,unconditional=calibration.analytic_truth(design,regime)
        miss=trial%20==0
        bounds=np.stack([truth+(1 if miss else -1),truth+2],axis=1)
        return dict(regime=regime,trial=trial,status="evaluated",familywise_noncoverage=miss,
            endpoint_mc_gate=True,halfwidth_targets_met=True,elapsed_seconds=0.,truth=truth.tolist(),
            unconditional_truth=unconditional.tolist(),confidence_interval=bounds.tolist(),
            noncovering_contrast_indices=list(range(60)) if miss else [],
            seed_sha256=object_hash(dict(seed=seed,regime=regime,trial=trial)))
    monkeypatch.setattr(calibration,"simulate_one",toy)
    out=tmp_path/"bounded"
    report=calibration.run_calibration(value,p,out,workers=1,resume=False,design_sha256="a"*64,protocol_sha256="b"*64,through_look=1)
    assert p==original and report["statistical_status"]=="inconclusive"
    assert report["unresolved_regimes"]==list(calibration.REGIMES)
    assert len(report["trials"])==len(calibration.REGIMES)*p["simulation_stages"][0]
    assert (out/"look_1.json").exists() and not (out/"look_2.json").exists()
    calibration.run_calibration(value,p,out,workers=1,resume=True,design_sha256="a"*64,protocol_sha256="b"*64,through_look=1)
    with pytest.raises(ValueError,match="execution look bound"):
        calibration.run_calibration(value,p,out,workers=1,resume=True,design_sha256="a"*64,protocol_sha256="b"*64,through_look=0)


def test_exact_complete_count_scope_samples_conditional_episodes_not_an_unverified_minimum():
    value=design(n=200,j=4,m=32)
    counts={s["id"]:37 for s in value["strata"]}
    data,truth,unconditional,diagnostics=calibration.generate_trial(value,"asymmetric_completion",7,0,counts)
    assert all(len(item["matrix"])==37 for item in data.values())
    assert all(d["count_conditioning"]=="exact_joint_complete_episode_counts" for d in diagnostics.values())
    assert np.max(abs(truth-unconditional))>.025
    protocol(value)
    p=calibration.instantiate_coverage_protocol(value,counts)
    calibration.validate_protocol(p,value)
    changed=deepcopy(p);changed["complete_episode_counts"]["a0"]=2
    with pytest.raises(ValueError,match="exact complete counts"):
        calibration.validate_protocol(changed,value)


def test_exact_count_instantiation_preserves_precollection_design_and_only_binds_counts():
    value=design(n=100)
    protocol(value)
    frozen=deepcopy(value);digest=object_hash(value)
    counts={s["id"]:37 for s in value["strata"]}
    candidate=calibration.instantiate_coverage_protocol(value,counts)
    assert value==frozen and object_hash(value)==digest
    assert candidate["coverage_plan_sha256"]==object_hash(value["analysis"]["coverage_plan"])
    assert candidate["minimum_complete_episodes"]==counts
    for key in ("seed","generator_source_sha256","simulation_stages"):
        changed=deepcopy(candidate)
        changed[key]=candidate[key]+1 if key=="seed" else "0"*64 if key.endswith("sha256") else [1]
        with pytest.raises(ValueError,match="prospective calibration plan"):
            calibration.validate_protocol(changed,value)


def test_source_and_saved_truth_or_seed_mutation_are_rejected():
    value=design();p=protocol(value)
    with pytest.raises(ValueError,match="source bytes changed"):
        calibration.verify_sources(dict(analysis_module_sha256="0"*64,generator_source_sha256=file_hash(calibration.__file__)))
    truth,unconditional=calibration.analytic_truth(value,"gaussian_clustered")
    row=dict(regime="gaussian_clustered",trial=0,protocol_sha256="a"*64,status="evaluated",
        seed_sha256=object_hash(dict(seed=p["seed"],regime="gaussian_clustered",trial=0)),
        truth=truth.tolist(),unconditional_truth=unconditional.tolist(),confidence_interval=np.stack([truth-1,truth+1],axis=1).tolist(),
        noncovering_contrast_indices=[],familywise_noncoverage=False,
        episode_accounting={s["id"]:dict(complete_episodes=len(s["reset_seeds"])) for s in value["strata"]})
    calibration.validate_trial_record(row,value,p,"a"*64,"gaussian_clustered",0)
    bad=deepcopy(row);bad["truth"][0]+=.1
    with pytest.raises(ValueError,match="truth differs"):
        calibration.validate_trial_record(bad,value,p,"a"*64,"gaussian_clustered",0)
    bad=deepcopy(row);bad["seed_sha256"]="b"*64
    with pytest.raises(ValueError,match="seed identity"):
        calibration.validate_trial_record(bad,value,p,"a"*64,"gaussian_clustered",0)
    bad=deepcopy(row);bad["familywise_noncoverage"]=True
    with pytest.raises(ValueError,match="coverage decision"):
        calibration.validate_trial_record(bad,value,p,"a"*64,"gaussian_clustered",0)
