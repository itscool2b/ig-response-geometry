"""Production-probe validation contracts exercised on exact CPU toy contexts."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import paired_comparison as paired
from experiment_io import file_hash, object_hash, tensor_hash
from fp32_probe_cache import verify_preparation, load_prepared_context
from scripts.validate_downstream_precision import CACHE_KEYS, cache_identity
from scripts.validate_fp32_probe import (compare_maps, compare_responses, endpoint_repeats,
    save_tensors, validate_decision, validate_target)


def decision_fixture():
    spec=dict(budgets=[2,4,8],repeat_budgets=[2,8],quadrature="trapezoid",arithmetic_dtype="float32")
    decision=dict(decision_id="E01",recorded_before_execution=True,forward_precision=paired.FP32_PROBE,
        bank_sha256="bank",phase="budget_selection",contexts=[dict(episode=0,policy_call_idx=0)],runtime_mode="deterministic",
        numerics={modality:{target:deepcopy(spec) for target in ("Q","L2")} for modality in ("vision","language","state")},
        gradient_repeats=2,map_repeats=2,endpoint_repeats=2,alphas=[0,.95,1],grid_percent=[0,5,50,100],
        denominator_min=dict(Q=0,L2=0,RMS=0),criteria="locked",failure_rule="retain",stopping_rule="complete",
        selection_rationale="prespecified engineering coverage")
    bank=dict(contexts=[dict(episode=0,policy_call_idx=0,episode_id="selected-episode",context_id="context",status="available")])
    return decision,bank


def test_requires_both_targets_all_modalities_and_two_budget_doublings():
    decision,bank=decision_fixture()
    assert len(validate_decision(decision,bank,"bank"))==1
    for mutation in ("target","modality","ladder"):
        bad=deepcopy(decision)
        if mutation=="target": del bad["numerics"]["state"]["L2"]
        elif mutation=="modality": del bad["numerics"]["state"]
        else: bad["numerics"]["state"]["L2"]["budgets"]=[2,4]
        with pytest.raises(ValueError): validate_decision(bad,bank,"bank")


def test_heldout_split_is_episode_level_and_binds_locked_candidate():
    decision,bank=decision_fixture()
    decision.update(phase="heldout_validation",selection_artifact_sha256="artifact",
                    calibration_episode_ids=["different-episode"],
                    candidate_numerics={modality:{target:dict(m=2,quadrature="trapezoid",arithmetic_dtype="float32")
                                                 for target in ("Q","L2")} for modality in ("vision","language","state")})
    validate_decision(decision,bank,"bank")
    decision["calibration_episode_ids"]=["selected-episode"]
    with pytest.raises(ValueError,match="share a calibration episode"):
        validate_decision(decision,bank,"bank")


def test_fresh_startup_sets_workspace_before_torch_is_imported():
    code="from scripts.validate_fp32_probe import process_startup; import sys,os,json; assert 'torch' not in sys.modules; print(json.dumps(process_startup('run'))); assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'"
    result=subprocess.run([sys.executable,"-c",code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)["torch_not_preimported"]


def test_top_five_percent_uses_exact_production_zero_count_for_state():
    a=torch.arange(8.).reshape(1,1,8)
    comparison=compare_maps(a,a,list(range(8)),2)
    assert comparison["top5_count"]==0 and comparison["top5_overlap"] is None
    assert comparison["top5_status"]=="zero_realized_count" and comparison["top1_match"]
    zero=compare_maps(torch.zeros_like(a),torch.zeros_like(a),list(range(8)),2)
    assert zero["relative_l1_difference"] is None and zero["group_spearman"] is None
    assert zero["reference_zero_map"] and zero["rank_status"]=="constant_group_scores"


def toy_probe():
    runner=torch.nn.Module()
    runner.state_adaptor=torch.nn.Linear(256,2,bias=False)
    torch.nn.init.constant_(runner.state_adaptor.weight,.01)
    for p in runner.parameters(): p.requires_grad_(False)
    image=torch.zeros(1,4374,2,dtype=torch.bfloat16)
    image[:,paired.EXT_CAM_START:paired.EXT_CAM_END]=1
    state=torch.zeros(1,1,128,dtype=torch.bfloat16)
    state[...,paired.MANISKILL_INDICES]=1
    language=torch.zeros(1,4,2,dtype=torch.bfloat16)
    language[:,:2]=1
    source=dict(img_adapted=image,img_adapted_bl=torch.zeros_like(image),
        lang_adapted=language,lang_adapted_bl=torch.zeros_like(language),
        state_input_actual=state,state_input_baseline=torch.zeros_like(state),
        state_traj_actual=torch.zeros(1,1,2,dtype=torch.bfloat16),
        action_mask=torch.ones(1,1,128,dtype=torch.bfloat16),ctrl_freqs=torch.tensor([25.],dtype=torch.bfloat16),
        initial_noise=torch.zeros(1,2,128,dtype=torch.bfloat16),ref_action=torch.zeros(1,2,128,dtype=torch.bfloat16),
        lang_attn_mask=torch.tensor([[True,True,False,False]]))
    def sample(runner,lang,langmask,image,state,actionmask,freq,noise):
        return (lang.sum()+image.sum()+state.sum()).reshape(1,1,1).expand_as(noise)
    probe,provenance=paired.make_fp32_probe_context(source,runner,sample_fn=sample)
    return probe,provenance


@pytest.mark.parametrize("target",["Q","L2"])
@pytest.mark.parametrize("modality",["vision","language","state"])
def test_exact_production_helpers_generate_complete_raw_numerical_artifacts(tmp_path,target,modality):
    probe,_=toy_probe()
    decision,_=decision_fixture()
    result=validate_target(probe,modality,target,decision["numerics"][modality][target],decision,tmp_path/"target","context")
    assert result["status"]=="complete_diagnostics_not_approval"
    assert len(result["operation_audits"])==3
    assert len(result["budgets"])==5
    assert len(result["comparisons"])==6
    assert result["endpoints"]["actual"]["bitwise_equal"]
    assert all(row["scores"][0]["Q"]==0 for name,row in result["endpoints"].items() if name=="actual")
    for artifact in result["budgets"]:
        payload=torch.load(tmp_path/"target"/artifact["file"],weights_only=True)
        assert set(payload)=={"attribution","path_gradient"}
        assert file_hash(tmp_path/"target"/artifact["file"])==artifact["sha256"]
    assert result["map_repeatability"]["8"]["IG"]["all_bitwise_equal"]
    assert all(row["rng_unchanged"] for row in result["gradient_probes"])


def test_corrupt_new_reference_rejected_before_gradient_maps(tmp_path):
    probe,_=toy_probe()
    probe["ref_action"]=probe["ref_action"]+1
    decision,_=decision_fixture()
    with pytest.raises(FloatingPointError,match="self-reference"):
        validate_target(probe,"language","Q",decision["numerics"]["language"]["Q"],decision,tmp_path/"bad","context")
    report=json.loads((tmp_path/"bad"/"report.json").read_text())
    assert report["status"]=="failed" and not report["budgets"]


def test_zero_gap_and_stabilized_l2_actual_endpoint_remain_finite(tmp_path):
    probe,_=toy_probe()
    probe["lang_adapted_bl"]=probe["lang_adapted"].clone()
    decision,_=decision_fixture()
    result=validate_target(probe,"language","L2",decision["numerics"]["language"]["L2"],decision,tmp_path/"zero","context")
    assert all(row["diagnostics"]["relative_residual"] is None for row in result["budgets"])
    assert result["endpoints"]["actual"]["scores"][0]["L2"]==pytest.approx(-1e-6)
    actual_gradients=[row for row in result["gradient_probes"] if row["alpha"]==1]
    assert all(row["gradient_l1"]==0 for row in actual_gradients)


def prepared_fixture(tmp_path):
    cache={key:torch.ones(1,dtype=torch.bfloat16) for key in CACHE_KEYS}
    cache["lang_attn_mask"]=torch.ones(1,dtype=torch.bool)
    payload={**cache,"context_id":"context"}
    source_row=dict(context_id="context",attr_sha256="sidecar")
    entry=dict(episode=0,policy_call_idx=0,episode_id="episode",reset_seed=7,context_id="context",row_sha256=object_hash(source_row))
    manifest=dict(run_id="run",configuration_sha256="configuration",
                  configuration=dict(pipeline={"weights":"hash"},language={"task":"hash"},source_sha256={"source":"hash"}))
    artifact=save_tensors(tmp_path/"cache.pt",cache)
    report=dict(kind="fp32_probe_source_preparation",status="complete_source_preparation",source_run_id="run",
        source_configuration_sha256="configuration",source_pipeline=manifest["configuration"]["pipeline"],
        source_language=manifest["configuration"]["language"],source_forward_sha256=manifest["configuration"]["source_sha256"],
        contexts=[dict(**entry,source_replay_verified=True,sidecar_sha256="sidecar",cached_tensors=cache_identity(cache),cache_file=artifact)])
    (tmp_path/"report.json").write_text(json.dumps(report))
    completion=dict(status="complete_source_preparation",artifacts_sha256={"report.json":file_hash(tmp_path/"report.json"),"cache.pt":file_hash(tmp_path/"cache.pt")})
    (tmp_path/"completion.json").write_text(json.dumps(completion))
    return report,entry,manifest,source_row,payload,file_hash(tmp_path/"completion.json")


def test_prepared_cache_authenticates_completion_context_and_original_noise(tmp_path):
    report,entry,manifest,row,payload,digest=prepared_fixture(tmp_path)
    verified=verify_preparation(tmp_path,digest)
    cache,record=load_prepared_context(tmp_path,verified,entry,manifest,row,payload)
    assert set(cache)==set(CACHE_KEYS) and record["source_replay_verified"]
    assert all(value.device.type=="cpu" for value in cache.values())
    payload["initial_noise"]=torch.zeros_like(payload["initial_noise"])
    with pytest.raises(ValueError,match="original sidecar"):
        load_prepared_context(tmp_path,verified,entry,manifest,row,payload)


@pytest.mark.parametrize("kind",["completion","cache","row","context","forward","missing_report"])
def test_prepared_cache_tampering_fails(tmp_path,kind):
    report,entry,manifest,row,payload,digest=prepared_fixture(tmp_path)
    if kind=="completion":
        with pytest.raises(ValueError,match="completion hash"):
            verify_preparation(tmp_path,"wrong")
    elif kind=="cache":
        with (tmp_path/"cache.pt").open("ab") as f:f.write(b"changed")
        with pytest.raises(ValueError,match="byte hash"):
            verify_preparation(tmp_path,digest)
    elif kind=="missing_report":
        path=tmp_path/"completion.json"
        path.write_text(json.dumps(dict(status="complete_source_preparation",artifacts_sha256={})))
        with pytest.raises(ValueError,match="bind its report"):
            verify_preparation(tmp_path,file_hash(path))
    else:
        if kind=="row":row["new_field"]="changed"
        if kind=="context":payload["context_id"]="other"
        if kind=="forward":manifest["configuration"]["source_sha256"]={"source":"changed"}
        with pytest.raises(ValueError):load_prepared_context(tmp_path,report,entry,manifest,row,payload)


def test_probe_numeric_failure_keeps_raw_tensor_but_dtype_violation_is_contract_error():
    source,_=toy_probe()
    runner=source["runner"]
    def invalid_reference(*args):
        return torch.full_like(args[-1],float("nan"))
    with pytest.raises(FloatingPointError) as failure:
        paired.make_fp32_probe_context(source,runner,sample_fn=invalid_reference)
    error=failure.value
    assert torch.isnan(error.raw_tensors["ref_action"]).all()
    assert error.diagnostics["tensors"]["ref_action"]["finite_count"]==0
    assert error.diagnostics["tensors"]["ref_action"]["dtype"]=="torch.float32"
    def wrong_dtype(*args):
        return torch.zeros_like(args[-1]).bfloat16()
    with pytest.raises(ValueError,match="dtype violates"):
        paired.make_fp32_probe_context(source,runner,sample_fn=wrong_dtype)


@pytest.mark.parametrize("first_failure",["numerical","identity","dtype"])
def test_run_keeps_six_failed_probe_arms_and_continues_but_contract_errors_abort(tmp_path,monkeypatch,first_failure):
    """Exercise the real run loop and artifact writer with model execution mocked."""
    from types import SimpleNamespace
    import faithfulness
    from scripts import validate_fp32_probe as validator
    from scripts import validate_downstream_precision as precision
    output=tmp_path/"results";output.mkdir()
    decision_path=tmp_path/"decision.json";decision_path.write_text("{}")
    bank_path=tmp_path/"bank.json";bank_path.write_text("{}")
    metrics=tmp_path/"metrics.jsonl";metrics.write_text("source")
    args=SimpleNamespace(decision_file=decision_path,bank=bank_path,metrics=metrics,
                         prepared=tmp_path/"prepared",prepared_sha256="completion")
    entries=[dict(episode=ep,policy_call_idx=0,episode_id="ep"+str(ep),context_id="ctx"+str(ep),status="available") for ep in range(2)]
    rows={(ep,0):dict(context_id="ctx"+str(ep)) for ep in range(2)}
    settings=validator.strict_candidate_settings()
    decision=dict(source_replay_settings=settings,numerics={m:{t:{} for t in ("Q","L2")} for m in ("vision","language","state")})
    preparation=dict(decision_sha256=file_hash(decision_path),bank_sha256=file_hash(bank_path),
        source_metrics_sha256=file_hash(metrics),replay_settings=settings,source_settings_authentication="fixture")
    monkeypatch.setattr(validator,"load_bank_inputs",lambda _: (decision,{},entries,None,{},file_hash(metrics),rows))
    monkeypatch.setattr(validator,"apply_settings",lambda _: settings)
    monkeypatch.setattr(validator,"runtime_settings",lambda: settings)
    monkeypatch.setattr(validator,"verify_preparation",lambda *a: preparation)
    monkeypatch.setattr(faithfulness,"load_sidecar",lambda *a: ({},"sidecar"))
    monkeypatch.setattr(faithfulness,"replay_pipeline",lambda *a: ({"runner":torch.nn.Linear(1,1)},None))
    monkeypatch.setattr(precision,"convert_downstream_to_fp32",lambda _: {"fixture":True})
    def cache(*args):
        if first_failure=="identity":
            raise ValueError("Source identity mismatch")
        return {"dummy":torch.ones(1)},{"cache_file":{"file":"cache.pt","sha256":"cache"}}
    monkeypatch.setattr(validator,"load_prepared_context",cache)
    constructed=[]
    def make_probe(*args):
        constructed.append(True)
        if len(constructed)==1:
            if first_failure=="dtype":
                raise ValueError("Probe dtype violates contract")
            error=FloatingPointError("Nonfinite new reference")
            raise paired.attach_probe_failure(error,{"ref_action":torch.full((1,2,128),float("nan"))},"self_reference")
        return {"ref_action":torch.zeros(1,2,128)},{"fixture":True}
    monkeypatch.setattr(paired,"make_fp32_probe_context",make_probe)
    executed=[]
    def validate(*arguments):
        _,modality,target,_,_,location,_=arguments
        location.mkdir();report={"status":"complete_diagnostics_not_approval"}
        (location/"report.json").write_text(json.dumps(report))
        executed.append((modality,target))
        return report
    monkeypatch.setattr(validator,"validate_target",validate)
    if first_failure!="numerical":
        with pytest.raises(ValueError): validator._run(args,output,{"fixture":True})
        assert not (output/"completion.json").exists()
        assert not executed
        return
    validator._run(args,output,{"fixture":True})
    report=json.loads((output/"report.json").read_text())
    assert report["planned_contexts"]==2 and report["available_contexts"]==2
    assert report["numerical_failures"]==6 and len(executed)==6
    first=report["contexts"][0]
    assert first["status"]=="probe_numerical_failure" and len(first["numerics"])==6
    raw=output/first["failure_artifact"]["file"]
    assert file_hash(raw)==first["failure_artifact"]["sha256"]
    assert torch.isnan(torch.load(raw,weights_only=True)["ref_action"]).all()
    completion=json.loads((output/"completion.json").read_text())
    assert completion["artifacts_sha256"][str(raw.relative_to(output))]==file_hash(raw)
