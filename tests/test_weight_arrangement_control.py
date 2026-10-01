"""Conditional parameter null, restoration and two-function transfer; CPU only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import paired_comparison as paired
import weight_arrangement_control as control
from experiment_io import canonical_json, file_hash, object_hash, strict_json, tensor_hash


class Runner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(3,4),nn.Tanh(),nn.Linear(4,1))
        self.state_adaptor=nn.Linear(256,3)
        self.model.register_parameter("positions",nn.Parameter(torch.tensor([[.2,.4,.1]])))
        self.register_buffer("counter",torch.tensor(4))
        with torch.no_grad():
            for index,parameter in enumerate(self.parameters()):
                parameter.copy_((torch.arange(parameter.numel()).reshape(parameter.shape)%11-5)/20+index/100)
        self.requires_grad_(False)


def draw(index=0,episode="episode-a"):
    return dict(namespace="synthetic-e05",master_seed=701,stratum_id="fixture",episode_id=episode,draw_index=index)


def context(runner):
    ctx=dict(runner=runner,lang_adapted=torch.tensor([[[.2,.1,.3],[.4,.3,.1],[.1,.4,.5]]]),
        lang_adapted_bl=torch.zeros(1,3,3),lang_attn_mask=torch.ones(1,3,dtype=torch.bool),
        img_adapted=torch.zeros(1,4374,3),img_adapted_bl=torch.zeros(1,4374,3),
        state_input_actual=torch.zeros(1,1,128),
        state_input_baseline=torch.zeros(1,1,128),action_mask=torch.ones(1,1,128),
        initial_noise=torch.ones(1,64,128)*.2)
    ctx["state_input_actual"][...,paired.MANISKILL_INDICES]=.1
    ctx["state_traj_actual"]=runner.state_adaptor(torch.cat([ctx["state_input_actual"],ctx["action_mask"]],dim=2))
    def sample(language,image,state):
        value=language.sum(1)+image.sum(1)+state.sum(1)+runner.model.positions
        return runner.model(value).reshape(1,1,1).expand(1,64,128)+ctx["initial_noise"]
    ctx["seeded_conditional_sample"]=sample
    ctx["ref_action"]=sample(ctx["lang_adapted"],ctx["img_adapted"],ctx["state_traj_actual"]).detach()
    return ctx


SETTINGS={target:dict(m=32,quadrature="trapezoid",arithmetic_dtype="float32") for target in control.TARGETS}


def test_null_changes_only_backbone_linear_weights_and_preserves_bit_multisets():
    runner=Runner();session=control.ArrangementSession(runner,control.describe_structure(runner))
    rng=torch.get_rng_state().clone()
    with session.draw(draw()) as record:
        assert session.permuted==["model.0.weight","model.2.weight"]
        assert record["registry_sha256"]==object_hash(session.registry)
        assert record["parameter_draw_sha256"]==object_hash({k:v for k,v in record.items() if k!="parameter_draw_sha256"})
        assert any(item["changed"] for item in record["tensors"].values())
        for name,item in record["tensors"].items():
            assert item["parent_multiset_sha256"]==item["child_multiset_sha256"]
            assert control.multiset_hash(dict(runner.named_parameters())[name])==item["parent_multiset_sha256"]
        for name,expected in record["preserved_sha256"].items():
            assert tensor_hash(session.tensors[name])==expected
    assert torch.equal(rng,torch.get_rng_state())
    session.verify(session.hashes)


def test_draw_repeats_exactly_and_does_not_reuse_episode_or_parameter_streams():
    runner=Runner();session=control.ArrangementSession(runner,control.describe_structure(runner))
    with session.draw(draw()) as first: pass
    with session.draw(draw()) as again: pass
    with session.draw(draw(1)) as other_draw: pass
    with session.draw(draw(episode="episode-b")) as other_episode: pass
    assert first==again
    assert len({r["parameter_draw_sha256"] for r in (first,other_draw,other_episode)})==3
    seeds=[item["stream"]["seed"] for r in (first,other_draw,other_episode) for item in r["tensors"].values()]
    assert len(set(seeds))==6


def test_restores_after_forward_failure_and_rejects_unplanned_tensor_mutation():
    runner=Runner();session=control.ArrangementSession(runner,control.describe_structure(runner))
    with pytest.raises(FloatingPointError,match="synthetic numerical failure"):
        with session.draw(draw()):
            raise FloatingPointError("synthetic numerical failure")
    session.verify(session.hashes)
    with pytest.raises(ValueError,match="outside its declared"):
        with session.draw(draw()):
            runner.model[0].bias.add_(1)
    session.verify(session.hashes)


@pytest.mark.parametrize("corruption",["subclass","alias","storage","shape","dtype","nan","parametrization"])
def test_unknown_wrappers_aliases_dtype_and_nonfinite_trained_values_fail(corruption):
    runner=Runner();expected=control.describe_structure(runner)
    if corruption=="subclass":
        class Wrapped(nn.Linear): pass
        runner.model[0]=Wrapped(3,4)
    elif corruption=="alias": runner.model.alias=runner.model[0]
    elif corruption=="storage": runner.model.shared=nn.Parameter(runner.model[0].weight.detach().reshape(-1))
    elif corruption=="shape": runner.model[0]=nn.Linear(3,5)
    elif corruption=="dtype": runner.bfloat16()
    elif corruption=="nan": runner.model[0].weight[0,0]=float("nan")
    elif corruption=="parametrization": runner.model[0].weight_orig=torch.ones(4,3)
    with pytest.raises(ValueError): control.ArrangementSession(runner,expected)


def test_multiset_digest_preserves_signed_zero_bit_patterns():
    left=torch.tensor([0.,-0.,2.,-3.,2.])
    assert control.multiset_hash(left)==control.multiset_hash(left[[4,2,1,3,0]])
    changed=left.clone();changed[1]=0.
    assert control.multiset_hash(changed)!=control.multiset_hash(left)


def test_constant_weight_tensor_is_not_replaced_by_a_favorable_draw():
    runner=Runner();runner.model[2].weight.zero_()
    session=control.ArrangementSession(runner,control.describe_structure(runner))
    with session.draw(draw()) as item:
        assert item["tensors"]["model.2.weight"]["changed"] is False


def trained_row(ctx,modality="language"):
    actual,baseline,indices,axis,forward=paired.modality_inputs(ctx,modality)
    rankings=paired.build_rankings(forward,ctx["ref_action"],actual,baseline,indices,axis,SETTINGS,
        context_id="fixture",modality=modality,random_permutations=2,random_seed=1)
    results=paired.evaluate_rankings(forward,ctx["ref_action"],actual,baseline,indices,axis,rankings,
        [0,50,100],dict(Q=0,L2=0,RMS=0))
    return dict(results=results,input_sha256=tensor_hash(actual),baseline_sha256=tensor_hash(baseline),
        noise_sha256=tensor_hash(ctx["initial_noise"]),reference_sha256=tensor_hash(ctx["ref_action"]))


def test_own_reference_rankings_transfer_only_after_restoration_and_keep_trained_anchors():
    runner=Runner();trained=context(runner);base=trained_row(trained)
    session=control.ArrangementSession(runner,control.describe_structure(runner))
    with session.draw(draw()):
        randomized=control.own_reference_context(trained)
        assert not torch.equal(randomized["ref_action"],trained["ref_action"])
        ranks,raw,identity=control.own_rankings(randomized,"language",SETTINGS,context_id="fixture")
        assert identity["randomized_reference_sha256"]==tensor_hash(randomized["ref_action"])
        assert set(raw)=={"randomized_reference","Q_attribution","Q_path_gradient","L2_attribution","L2_path_gradient"}
        with pytest.raises(ValueError,match="Restored trained actual"):
            control.transfer_on_trained(trained,"language",ranks,base,[0,50,100],dict(Q=0,L2=0,RMS=0))
    results=control.transfer_on_trained(trained,"language",ranks,base,[0,50,100],dict(Q=0,L2=0,RMS=0))
    paired.authenticate_action_results(results)
    assert results["active_reference_sha256"]==base["results"]["active_reference_sha256"]
    assert set(results["rankings"])=={"trained_Q_IG","trained_L2_IG","permuted_Q_IG","permuted_L2_IG"}


def test_zero_native_head_diagnostic_is_all_tied_and_distinct_from_permutation():
    runner=Runner();runner.model[2].weight.zero_();runner.model[2].bias.zero_()
    ctx=context(runner)
    rankings,raw,_=control.own_rankings(ctx,"language",SETTINGS,context_id="zero-head")
    for target in control.TARGETS:
        ranking=rankings["permuted_"+target+"_IG"]
        assert ranking["status"]=="defined" and ranking["zero_map"] is True
        assert ranking["order"]==[0,1,2] and ranking["tied_positions"]==3
        assert torch.count_nonzero(raw[target+"_attribution"])==0


def test_nonfinite_reference_is_a_raw_numerical_outcome_not_zero_map():
    trained=context(Runner())
    trained["seeded_conditional_sample"]=lambda *args:torch.full_like(trained["ref_action"],float("nan"))
    with pytest.raises(FloatingPointError) as failure:
        control.own_reference_context(trained)
    assert torch.isnan(failure.value.raw_tensors["ref_action"]).all()


def test_reference_dtype_violation_is_contract_error():
    trained=context(Runner())
    trained["seeded_conditional_sample"]=lambda *args:trained["ref_action"].bfloat16()
    with pytest.raises(ValueError,match="dtype/shape"):
        control.own_reference_context(trained)


def completed_control_fixture(tmp_path,*,failure=False):
    runner=Runner();ctx=context(runner)
    source=tmp_path/"source.jsonl";source.write_text("synthetic source export")
    base=tmp_path/"base.jsonl";out=tmp_path/"e05.jsonl"
    source_manifest=dict(run_id="synthetic",configuration_sha256="source-config")
    bank=dict(task="fixture",model="170m",pipeline={},language={},
        estimand="equal_episode_mean_of_uniform_executed_calls_under_capped_behavior_policy",
        selection=dict(rule="uniform_executed_calls_v1",calls_per_episode=1,max_episode_steps=400),
        selection_accounting=[dict(episode=0,eligible_calls=[0],permutation=[0],selected_calls=[0],
            inclusion_probability=1.,within_episode_weight=1.,episode_weight=1.)],
        contexts=[dict(episode=0,episode_id="ep",policy_call_idx=0,context_id="context",reset_seed=31,terminal_record_sha256="e"*64)])
    base_protocol=dict(stage="variance_only_pilot",global_design_sha256="a"*64,forward_precision=paired.FP32_PROBE,
        numerics={modality:SETTINGS for modality in control.MODALITIES},grid_percent=[0,50,100],denominator_min=dict(Q=0,L2=0,RMS=0))
    base_config=dict(protocol=base_protocol,protocol_sha256="base-protocol",bank=bank,bank_sha256=object_hash(bank))
    source_row=dict(episode=0,seed=31,policy_call_idx=0,context_id="context",attr_sha256="source-sidecar")
    with paired.PairedEvaluationWriter(SimpleNamespace(out=base,metrics=source,limit=None),source_manifest,file_hash(source),"paired_comparison",base_config,3) as writer:
        writer.set_context(source_row)
        for modality in control.MODALITIES:
            writer.write(dict(episode_id="ep",modality=modality,status="evaluated",**trained_row(ctx,modality)))
    base_rows,base_manifest,_=paired.load_completed_study(base)
    session=control.ArrangementSession(runner,control.describe_structure(runner))
    protocol=dict(**control.SCOPE,stage="variance_only_pilot",parameter_draws=2,parameter_seed_namespace="synthetic-e05",
        parameter_master_seed=701,stratum_id="fixture",global_design_sha256="a"*64,numerics={modality:SETTINGS for modality in control.MODALITIES})
    config=dict(protocol=protocol,protocol_sha256="null-protocol",bank=bank,bank_sha256=object_hash(bank),parameter_registry=session.registry,
        base_metrics_sha256=file_hash(base),base_manifest_sha256=file_hash(str(base)+".manifest.json"),
        base_completion_sha256=file_hash(str(base)+".completion.json"),base_protocol_sha256="base-protocol",e05_gate_sha256="b"*64)
    with paired.PairedEvaluationWriter(SimpleNamespace(out=out,metrics=source,limit=None),source_manifest,file_hash(source),"weight_arrangement_control",config,6) as writer:
        writer.set_context(source_row)
        for index in range(2):
            identity={**draw(index),"episode_id":"ep"}
            pending={}
            with session.draw(identity) as record:
                name=f"parameter-draw-{index}.json"
                descriptor=dict(file=out.name+"."+name,sha256=writer.write_auxiliary(name,record))
                own=control.own_reference_context(ctx)
                for modality in control.MODALITIES:
                    ranks,raw,coordinates=control.own_rankings(own,modality,SETTINGS,context_id="context")
                    artifact=writer.write_tensor_auxiliary(f"coordinates-{modality}-{index}.pt",raw)
                    pending[modality]=(ranks,coordinates,artifact)
            for modality in control.MODALITIES:
                original=next(r for r in base_rows if r["modality"]==modality)
                common=dict(episode_id="ep",modality=modality,draw_index=index,base_row_sha256=object_hash(original),
                    parameter_draw_sha256=record["parameter_draw_sha256"],parameter_draw_artifact=descriptor)
                if failure and modality=="state" and index==1:
                    writer.write(dict(**common,status="numerical_failure",failure_kind="randomized_self_reference",results=None))
                    continue
                ranks,coordinates,artifact=pending[modality]
                results=control.transfer_on_trained(ctx,modality,ranks,original,[0,50,100],dict(Q=0,L2=0,RMS=0))
                writer.write(dict(**common,status="evaluated",results=results,coordinate_identity=coordinates,
                    randomized_reference_sha256=coordinates["randomized_reference_sha256"],coordinate_artifact=artifact))
            writer.write_auxiliary("restoration-"+object_hash(identity)+".json",dict(parameter_draw_sha256=record["parameter_draw_sha256"],
                registry_sha256=object_hash(session.registry),all_trained_tensor_hashes_verified=True,pre_intervention_reference_replay_verified=True))
    return out,base


def run_protocol_fixture(base_manifest,native_digest):
    import fp32_probe_cache as helper
    protocol={**control.SCOPE,"stage":"variance_only_pilot","parameter_draws":2,"global_design_sha256":"a"*64,
        "parameter_seed_namespace":"synthetic-e05","parameter_master_seed":701,"stratum_id":"fixture",
        "failure_policy":"retain_every_planned_draw_no_replacement","forward_precision":paired.FP32_PROBE,
        "runtime_settings":helper.strict_candidate_settings(),"native_registry_sha256":native_digest,
        "numerics":{modality:deepcopy(SETTINGS) for modality in control.MODALITIES}}
    base_manifest["configuration"]["protocol"]["runtime_settings"]=helper.strict_candidate_settings()
    gate=dict(status="approved_for_weight_arrangement_transfer",independent_holdout_complete=True,
        native_registry_sha256=native_digest,null_scope=control.SCOPE,weight_arrangement_control_sha256=file_hash(control.__file__),
        source_sha256=control.implementation_hashes(),torch_version=torch.__version__,
        selection_report_sha256=["b"*64],heldout_report_sha256=["c"*64],
        runtime_settings=helper.strict_candidate_settings(),approved_numerics=deepcopy(protocol["numerics"]),
        approved_strata=[dict(task="fixture",model="170m",pipeline_sha256=object_hash({}),language_sha256=object_hash({}))])
    return protocol,gate


@pytest.mark.parametrize("corruption",["unapproved","same_holdout","different_helpers","missing_target","wrong_R","wrong_scope"])
def test_independent_randomized_numerical_gate_is_mandatory(tmp_path,corruption):
    _,base=completed_control_fixture(tmp_path)
    _,manifest,_=paired.load_completed_study(base)
    protocol,gate=run_protocol_fixture(manifest,"d"*64)
    control.validate_run_protocol(protocol,gate,manifest,"d"*64)
    if corruption=="unapproved":gate["status"]="complete_diagnostics_not_approval"
    if corruption=="same_holdout":gate["heldout_report_sha256"]=gate["selection_report_sha256"]
    if corruption=="different_helpers":gate["source_sha256"]["paired_comparison.py"]="0"*64
    if corruption=="missing_target":del protocol["numerics"]["state"]["L2"]
    if corruption=="wrong_R":protocol["parameter_draws"]=1
    if corruption=="wrong_scope":protocol["reference"]="trained_reference_for_randomized_function"
    with pytest.raises(ValueError):control.validate_run_protocol(protocol,gate,manifest,"d"*64)


def test_cli_rejects_preimported_torch_before_model_or_artifact_access(tmp_path):
    script=Path(control.__file__).parent/"scripts/run_weight_arrangement_control.py"
    code="import torch,runpy,sys;sys.argv=[sys.argv[1],*sys.argv[2:]];runpy.run_path(sys.argv[0],run_name='__main__')"
    args=[sys.executable,"-c",code,str(script)]
    for name in ("metrics","base","protocol","e05-gate","native-registry","prepared","out","protocol-sha256","lang-dir"):
        args.extend(["--"+name,str(tmp_path/"unused")])
    result=subprocess.run(args,capture_output=True,text=True)
    assert result.returncode!=0 and "fresh process" in result.stderr


def test_producer_retains_six_outcomes_and_continues_after_randomized_reference_failure(tmp_path,monkeypatch):
    """Mock only model/cache/source bridges; exercise actual draw, writer, maps, transfer and loader."""
    import faithfulness
    import fp32_probe_cache as helper
    from scripts import validate_downstream_precision as precision
    _,base=completed_control_fixture(tmp_path)
    base_rows,manifest,_=paired.load_completed_study(base)
    source=tmp_path/"source.jsonl"
    native=tmp_path/"native.json";native.write_text('{"synthetic":true}')
    protocol,gate=run_protocol_fixture(manifest,file_hash(native))
    manifest["configuration"]["prepared_completion_sha256"]="f"*64
    Path(str(base)+".manifest.json").write_bytes(canonical_json(manifest))
    completion=strict_json(str(base)+".completion.json");completion["manifest_sha256"]=file_hash(str(base)+".manifest.json")
    Path(str(base)+".completion.json").write_bytes(canonical_json(completion))
    runner=Runner();structure=control.describe_structure(runner)
    gate["structure_sha256"]=object_hash(structure)
    gate_path=tmp_path/"gate.json";gate_path.write_bytes(canonical_json(gate))
    protocol.update(e05_gate_sha256=file_hash(gate_path),prepared_completion_sha256="f"*64,
        base_metrics_sha256=file_hash(base),base_manifest_sha256=file_hash(str(base)+".manifest.json"),
        base_completion_sha256=file_hash(str(base)+".completion.json"),base_protocol_sha256="base-protocol")
    protocol_path=tmp_path/"protocol.json";protocol_path.write_bytes(canonical_json(protocol))
    source_row=dict(episode=0,seed=31,policy_call_idx=0,context_id="context",attr_sha256="source-sidecar")
    source_manifest=dict(run_id="synthetic",configuration_sha256="source-config")
    monkeypatch.setattr(faithfulness,"authenticated_source",lambda *args:([source_row],source_manifest))
    monkeypatch.setattr(faithfulness,"load_sidecar",lambda *args:({},"source-sidecar"))
    monkeypatch.setattr(faithfulness,"replay_pipeline",lambda *args:({"runner":runner},{}))
    monkeypatch.setattr(paired,"make_bank",lambda *args:manifest["configuration"]["bank"])
    monkeypatch.setattr(paired,"bank_rows",lambda *args:{(0,0):source_row})
    monkeypatch.setattr(helper,"verify_preparation",lambda *args:{"synthetic":True})
    monkeypatch.setattr(helper,"load_prepared_context",lambda *args:({}, {"cache_file":{"file":"cache.pt","sha256":"f"*64}}))
    monkeypatch.setattr(precision,"convert_downstream_to_fp32",lambda *args:{"synthetic":"already FP32"})
    monkeypatch.setattr(control,"structure_from_native_registry",lambda *args:structure)
    original=tensor_hash(runner.model[0].weight)
    failed=[]
    def make_probe(source,loaded_runner):
        if tensor_hash(loaded_runner.model[0].weight)!=original and not failed:
            failed.append(True)
            error=FloatingPointError("synthetic randomized nonfinite reference")
            error.raw_tensors={"ref_action":torch.full((1,64,128),float("nan"))}
            raise error
        return context(loaded_runner),{}
    monkeypatch.setattr(paired,"make_fp32_probe_context",make_probe)
    out=tmp_path/"executed-e05.jsonl"
    args=SimpleNamespace(protocol=protocol_path,protocol_sha256=file_hash(protocol_path),e05_gate=gate_path,
        native_registry=native,base=base,metrics=source,prepared=tmp_path/"prepared",checkpoint_path=None,
        lang_dir="unused",out=out)
    previous=helper.runtime_settings()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    try:
        control.run(args,{"torch_not_preimported":True,"synthetic_fresh_process_test":True})
    finally:
        if previous["cublas_workspace_config"] is None:monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG",raising=False)
        else:monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",previous["cublas_workspace_config"])
        helper.apply_settings(previous)
    rows,_,_=control.load_completed_control(out,base_path=base)
    assert len(rows)==6
    assert all(r["status"]=="numerical_failure" and r["failure_kind"]=="randomized_self_reference" for r in rows[:3])
    assert all(r["status"]=="evaluated" for r in rows[3:])
    assert tensor_hash(runner.model[0].weight)==original


@pytest.mark.parametrize("failure",[False,True])
def test_completed_control_binds_every_draw_and_preserves_failed_outcomes(tmp_path,failure):
    output,base=completed_control_fixture(tmp_path,failure=failure)
    rows,manifest,completion=control.load_completed_control(output,base_path=base)
    assert len(rows)==completion["expected_rows"]==6
    assert sum(r["status"]=="numerical_failure" for r in rows)==int(failure)
    for index in range(2):
        assert len({r["parameter_draw_sha256"] for r in rows if r["draw_index"]==index})==1
    assert len({r["parameter_draw_sha256"] for r in rows})==2
    assert manifest["configuration"]["base_metrics_sha256"]==file_hash(base)


def rewrite_rows(path,rows):
    path.write_bytes(b"".join(canonical_json(row)+b"\n" for row in rows))
    completion=strict_json(str(path)+".completion.json")
    completion["output_sha256"]=file_hash(path)
    Path(str(path)+".completion.json").write_bytes(canonical_json(completion))


@pytest.mark.parametrize("corruption",["missing_draw","reused_draw","base_context","own_reference","coordinate_bytes"])
def test_semantic_loader_rejects_internally_resealed_invalid_artifacts(tmp_path,corruption):
    output,base=completed_control_fixture(tmp_path)
    rows,_,_=control.load_completed_control(output,base_path=base)
    if corruption=="missing_draw": rows.pop()
    elif corruption=="reused_draw": rows[3]["parameter_draw_sha256"]=rows[0]["parameter_draw_sha256"]
    elif corruption=="base_context": rows[0]["episode"]=42
    elif corruption=="own_reference": rows[0]["randomized_reference_sha256"]="0"*64
    elif corruption=="coordinate_bytes":
        descriptor=rows[0]["coordinate_artifact"]
        path=output.with_name(descriptor["file"])
        raw=torch.load(path,weights_only=True);raw["Q_attribution"].flatten()[0]+=1
        torch.save(raw,path)
        descriptor["sha256"]=file_hash(path)
        completion=strict_json(str(output)+".completion.json")
        completion["auxiliary_sha256"][path.name[len(output.name)+1:]]=file_hash(path)
        Path(str(output)+".completion.json").write_bytes(canonical_json(completion))
    rewrite_rows(output,rows)
    with pytest.raises(ValueError):control.load_completed_control(output,base_path=base)


def test_draw_ledger_cannot_change_retained_scope_or_learned_multiset():
    runner=Runner();session=control.ArrangementSession(runner,control.describe_structure(runner))
    with session.draw(draw()) as record: pass
    control.validate_draw_record(record,session.registry,draw())
    for mutation in ("preserved","multiset"):
        corrupt=deepcopy(record)
        if mutation=="preserved":corrupt["preserved_sha256"].pop("counter")
        else:corrupt["tensors"]["model.0.weight"]["child_multiset_sha256"]="0"*64
        corrupt["parameter_draw_sha256"]=object_hash({k:v for k,v in corrupt.items() if k!="parameter_draw_sha256"})
        with pytest.raises(ValueError):control.validate_draw_record(corrupt,session.registry,draw())
