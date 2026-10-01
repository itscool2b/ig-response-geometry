"""E05 fixed-ladder two-function numerical validation, with CPU toy runners."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import paired_comparison as paired
import weight_arrangement_control as control
from experiment_io import file_hash, object_hash, strict_json, tensor_hash
from scripts import validate_weight_arrangement as validator
from test_fp32_probe_validation import decision_fixture
from test_weight_arrangement_control import Runner, context, draw


def decision():
    result,bank=decision_fixture()
    result.update(**control.SCOPE, native_registry_sha256="native", torch_version=torch.__version__,
        implementation_sha256=validator.implementation_hashes(), criterion_thresholds=deepcopy(validator.THRESHOLDS),
        parameter_draws=2, parameter_seed_namespace="numerical-e05-fixture", parameter_master_seed=8,
        stratum_id="toy", failure_policy="retain_every_planned_draw_no_replacement",
        checkpoint_identity={"sha256":"synthetic"},checkpoint_mode="pretrained")
    return result,bank


def test_entrypoint_does_not_import_torch_before_fresh_candidate_startup():
    code="from scripts.validate_weight_arrangement import process_startup; import sys,json; assert 'torch' not in sys.modules; print(json.dumps(process_startup('run')))"
    result=subprocess.run([sys.executable,"-c",code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)["torch_not_preimported"]


@pytest.mark.parametrize("field,value",[("parameter_draws",1),("runtime_mode","baseline"),
    ("reference","trained_reference"),("torch_version","other"),("implementation_sha256",{}),
    ("native_registry_sha256","other"),("failure_policy","replace_failed_draws"),("checkpoint_mode",None),
    ("criterion_thresholds",{**validator.THRESHOLDS,"coordinate_l1":.02})])
def test_protocol_rejects_scope_identity_threshold_or_failure_relaxation(field,value):
    value_before,bank=decision()
    validator.validate_decision(value_before,bank,"bank","native")
    value_before[field]=value
    with pytest.raises(ValueError): validator.validate_decision(value_before,bank,"bank","native")


@pytest.mark.parametrize("modality",["vision","language","state"])
@pytest.mark.parametrize("target",["Q","L2"])
def test_randomized_own_maps_use_restored_trained_response_and_distinct_gaps(tmp_path,modality,target):
    protocol,_=decision()
    runner=Runner(); trained=context(runner)
    session=control.ArrangementSession(runner,control.describe_structure(runner))
    with session.draw(draw()):
        own=context(runner)
        assert not torch.equal(own["ref_action"],trained["ref_action"])
        report=validator.collect_own_target(own,modality,target,protocol["numerics"][modality][target],protocol,tmp_path/"arm","context")
        assert report["status"]=="own_maps_complete_pending_trained_responses"
        assert all("response_file" not in row for row in report["budgets"])
        own_reference=tensor_hash(own["ref_action"])
    session.verify(session.hashes)
    report=validator.complete_trained_responses(trained,report,protocol,tmp_path/"arm")
    assert report["status"]=="complete_diagnostics_not_approval"
    assert report["reference_sha256"]==own_reference
    assert report["trained_reference_sha256"]==tensor_hash(trained["ref_action"])
    assert report["own_function_endpoints"]["actual"]["action_sha256"][0]==own_reference
    assert report["trained_response_endpoints"]["actual"]["action_sha256"][0]==report["trained_reference_sha256"]
    for row in report["budgets"]:
        assert row["diagnostics"]["signed_residual"]==row["diagnostics"]["ig_sum"]-row["diagnostics"]["expected_gap"]
        path=tmp_path/"arm"/row["response_file"]
        assert file_hash(path)==row["response_sha256"]
        responses=strict_json(path)
        paired.authenticate_action_results(responses)
        assert responses["active_reference_sha256"]==tensor_hash(trained["ref_action"][...,paired.MANISKILL_INDICES])
    assert len(report["comparisons"])==6
    assert all(len(item["common_responses"])==6 for item in report["comparisons"])
    assert report["map_repeatability"]["8"]["IG"]["all_bitwise_equal"]
    assert all(pair["bitwise_equal"] for pair in report["map_repeatability"]["8"]["trained_action_repeatability"])
    if modality=="vision":
        assert all(row["diagnostics"]["relative_residual"] is None for row in report["budgets"])
        assert any(check["status"]=="undefined" for check in report["criterion_audit"]["checks"])
    else:
        assert report["own_function_endpoints"]["baseline"]["scores"] != report["trained_response_endpoints"]["baseline"]["scores"]


def completed_arm(tmp_path):
    protocol,_=decision(); runner=Runner();trained=context(runner)
    session=control.ArrangementSession(runner,control.describe_structure(runner))
    with session.draw(draw()):
        own=context(runner)
        report=validator.collect_own_target(own,"language","Q",protocol["numerics"]["language"]["Q"],protocol,tmp_path/"arm","context")
    return protocol,trained,report


def test_response_threshold_uses_trained_gap_while_completeness_uses_own_gap(tmp_path):
    protocol,trained,report=completed_arm(tmp_path)
    report=validator.complete_trained_responses(trained,report,protocol,tmp_path/"arm")
    report["own_function_endpoints"]["baseline"]["scores"][0]["RMS"]=100.
    report["trained_response_endpoints"]["baseline"]["scores"][0]["RMS"]=2.
    report["budgets"][0]["diagnostics"].update(expected_gap=0.,relative_residual=None)
    checks=validator.criterion_audit(report,protocol)["checks"]
    raw=[check for check in checks if check["criterion"]=="rms_curve_difference"]
    assert raw and all(check["threshold"]==.02 and check["baseline_rms"]==2. for check in raw)
    own=[check for check in checks if check["criterion"]=="candidate_completeness" and check["m"]==2 and check["repeat"]==0]
    assert own[0]["status"]=="undefined"


def test_saved_map_tampering_is_a_contract_error_before_transfer(tmp_path):
    protocol,trained,report=completed_arm(tmp_path)
    with (tmp_path/"arm"/report["budgets"][0]["file"]).open("ab") as stream:stream.write(b"tampered")
    with pytest.raises(ValueError,match="bytes changed"):
        validator.complete_trained_responses(trained,report,protocol,tmp_path/"arm")


def test_trained_endpoint_dtype_violation_is_fatal_contract_error(tmp_path):
    protocol,trained,report=completed_arm(tmp_path)
    sample=trained["seeded_conditional_sample"]
    trained["seeded_conditional_sample"]=lambda *args:sample(*args).bfloat16()
    with pytest.raises(ValueError,match="Trained response action dtype or shape"):
        validator.complete_trained_responses(trained,report,protocol,tmp_path/"arm")
    assert strict_json(tmp_path/"arm"/"report.json")["status"]=="contract_failure"


def test_nonfinite_own_gradient_is_saved_and_not_coerced_to_zero(tmp_path):
    protocol,_=decision();ctx=context(Runner())
    ctx["lang_adapted_bl"]=ctx["lang_adapted"].clone()
    actual=ctx["lang_adapted"].detach().clone()
    def bad_sample(language,image,state):
        # Finite constant forward with an undefined sqrt derivative at its endpoint.
        return ((language-actual).abs().sqrt().sum()*0).expand_as(ctx["ref_action"])
    ctx["seeded_conditional_sample"]=bad_sample
    ctx["ref_action"]=torch.zeros_like(ctx["ref_action"])
    report=validator.collect_own_target(ctx,"language","Q",protocol["numerics"]["language"]["Q"],protocol,tmp_path/"arm","context")
    assert report["status"]=="numerical_failure"
    assert report["failure_kind"]=="randomized_numerical_failure"
    raw=torch.load(tmp_path/"arm"/report["failure_artifact"]["file"],weights_only=True)
    assert "gradient" in raw and not torch.isfinite(raw["gradient"]).all()


def entries():
    return [dict(episode=0,episode_id="episode",policy_call_idx=i,context_id=f"context{i}" if i<2 else None,
                 status="available" if i<2 else "call_not_reached") for i in range(3)]


def test_episode_loop_retains_failed_draw_all_six_arms_and_unavailable_membership(tmp_path):
    protocol,_=decision();runner=Runner()
    session=control.ArrangementSession(runner,control.describe_structure(runner))
    failed=False
    def load(entry):
        nonlocal failed
        if session.active and entry["policy_call_idx"]==0 and not failed:
            failed=True
            error=FloatingPointError("synthetic randomized reference failure")
            error.raw_tensors={"ref_action":torch.full((1,64,128),float("nan"))}
            raise error
        ctx=context(runner)
        return ctx,dict(probe_reference_sha256=tensor_hash(ctx["ref_action"]))
    progress=dict(contexts=[],draws=[])
    validator.run_episode(session,entries(),protocol,load,tmp_path,progress)
    assert len(progress["contexts"])==6
    cells=[arm for item in progress["contexts"] for arm in item["numerics"]]
    assert len(cells)==36
    assert sum(arm["status"]=="numerical_failure" for arm in cells)==6
    assert sum(arm["status"]=="source_unavailable" for arm in cells)==12
    assert sum(arm["status"]=="complete_diagnostics_not_approval" for arm in cells)==18
    first=progress["contexts"][0]
    assert first["failure"]["failure_kind"]=="randomized_self_reference"
    assert (tmp_path/first["failure"]["failure_artifact"]["file"]).exists()
    assert len({item["parameter_draw_sha256"] for item in progress["contexts"]})==2
    assert all(record["all_trained_tensor_hashes_verified"] for record in progress["draws"])
    session.verify(session.hashes)


def test_episode_contract_failure_stops_but_restores_original_weights(tmp_path):
    protocol,_=decision();runner=Runner()
    session=control.ArrangementSession(runner,control.describe_structure(runner))
    def load(entry):
        if session.active: raise ValueError("wrong dtype contract")
        ctx=context(runner)
        return ctx,dict(probe_reference_sha256=tensor_hash(ctx["ref_action"]))
    with pytest.raises(ValueError,match="wrong dtype"):
        validator.run_episode(session,entries()[:1],protocol,load,tmp_path,dict(contexts=[],draws=[]))
    session.verify(session.hashes)


def test_restored_probe_numerical_failure_keeps_own_maps_and_later_draws(tmp_path):
    protocol,_=decision();runner=Runner()
    session=control.ArrangementSession(runner,control.describe_structure(runner))
    calls=0
    def load(entry):
        nonlocal calls
        calls+=1
        if calls==3:
            error=FloatingPointError("synthetic restored reference nonfinite")
            error.raw_tensors={"ref_action":torch.full((1,64,128),float("nan"))}
            raise error
        ctx=context(runner)
        return ctx,dict(probe_reference_sha256=tensor_hash(ctx["ref_action"]))
    progress=dict(contexts=[],draws=[])
    validator.run_episode(session,entries()[:1],protocol,load,tmp_path,progress)
    assert all(arm["status"]=="numerical_failure" for arm in progress["contexts"][0]["numerics"])
    assert all(arm["status"]=="complete_diagnostics_not_approval" for arm in progress["contexts"][1]["numerics"])
    for arm in progress["contexts"][0]["numerics"]:
        report=strict_json(tmp_path/arm["report_file"])
        assert report["budgets"] and report["failure_kind"]=="trained_probe_repeatability"
        assert (tmp_path/arm["report_file"]).parent.joinpath(report["failure_artifact"]["file"]).exists()


def test_randomized_dtype_violation_is_fatal_contract_error(tmp_path):
    protocol,_=decision();ctx=context(Runner())
    sample=ctx["seeded_conditional_sample"]
    ctx["seeded_conditional_sample"]=lambda *args:sample(*args).bfloat16()
    with pytest.raises(ValueError,match="dtype or shape"):
        validator.collect_own_target(ctx,"language","Q",protocol["numerics"]["language"]["Q"],protocol,tmp_path/"arm","context")
    assert strict_json(tmp_path/"arm"/"report.json")["status"]=="contract_failure"


def test_full_run_completion_authenticates_all_planned_cells_and_raw_failures(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import faithfulness
    import fp32_probe_cache as cache_helper
    from scripts import validate_downstream_precision as precision
    protocol,_=decision();runner=Runner()
    output=tmp_path/"output";output.mkdir()
    paths={name:tmp_path/(name+".json") for name in ("metrics","bank","native_registry","decision_file")}
    for path in paths.values():path.write_text("{}")
    args=SimpleNamespace(**paths,prepared=tmp_path/"prepared",prepared_sha256="prepared",
                         decision_sha256=file_hash(paths["decision_file"]))
    roster=[dict(episode=i,episode_id=f"ep{i}",policy_call_idx=0,context_id=f"context{i}",
                 status="available" if i<2 else "call_not_reached") for i in range(3)]
    rows={(entry["episode"],0):dict(context_id=entry["context_id"]) for entry in roster}
    settings=cache_helper.strict_candidate_settings();protocol["source_replay_settings"]=settings
    preparation=dict(decision_sha256=args.decision_sha256,bank_sha256=file_hash(args.bank),
        source_metrics_sha256=file_hash(args.metrics),replay_settings=settings,source_settings_authentication="fixture",
        implementation_sha256=validator.implementation_hashes(),native_registry_sha256=file_hash(args.native_registry))
    monkeypatch.setattr(validator,"load_inputs",lambda _: (protocol,{"model":"170m"},roster,None,{},file_hash(args.metrics),rows,{}))
    monkeypatch.setattr(cache_helper,"apply_settings",lambda _:settings)
    monkeypatch.setattr(cache_helper,"runtime_settings",lambda:settings)
    monkeypatch.setattr(cache_helper,"verify_preparation",lambda *args:preparation)
    monkeypatch.setattr(cache_helper,"load_prepared_context",lambda directory,report,entry,*args:
        ({"fixture_id":torch.tensor(entry["episode"])},{"cache_file":{"file":"fixture","sha256":"fixture"}}))
    monkeypatch.setattr(faithfulness,"load_sidecar",lambda *args:({},"sidecar"))
    monkeypatch.setattr(faithfulness,"replay_pipeline",lambda *args:({"runner":runner},None))
    monkeypatch.setattr(precision,"convert_downstream_to_fp32",lambda _:dict(fixture=True))
    monkeypatch.setattr(control,"structure_from_native_registry",lambda *args:control.describe_structure(runner))
    def make(source,loaded_runner):
        if int(source["fixture_id"])==0:
            error=FloatingPointError("synthetic trained probe failure")
            error.raw_tensors={"ref_action":torch.full((1,64,128),float("nan"))}
            raise error
        ctx=context(loaded_runner)
        return ctx,dict(probe_reference_sha256=tensor_hash(ctx["ref_action"]))
    monkeypatch.setattr(paired,"make_fp32_probe_context",make)
    validator.run(args,output,dict(torch_not_preimported=True))
    report=strict_json(output/"report.json");completion=strict_json(output/"completion.json")
    assert report["planned_context_draw_target_modality_cells"]==36
    assert report["numerical_failure_cells"]==12 and report["unavailable_cells"]==12
    assert len(report["contexts"])==6 and len(report["draws"])==6
    assert report["status"]==completion["status"]=="complete_diagnostics_not_approval"
    for name,expected in completion["artifacts_sha256"].items(): assert file_hash(output/name)==expected
    first=report["contexts"][0]
    raw=output/first["failure"]["failure_artifact"]["file"]
    assert torch.isnan(torch.load(raw,weights_only=True)["ref_action"]).all()
    assert completion["artifacts_sha256"][str(raw.relative_to(output))]==file_hash(raw)


@pytest.mark.parametrize("mutate_source",[False,True])
def test_prepare_rechecks_sources_before_sealing_compatible_cache(tmp_path,monkeypatch,mutate_source):
    from types import SimpleNamespace
    import faithfulness
    import fp32_probe_cache as cache_helper
    from scripts import validate_downstream_precision as precision
    protocol,_=decision()
    paths={name:tmp_path/(name+".json") for name in ("metrics","bank","native_registry","decision_file")}
    for path in paths.values():path.write_text("{}")
    args=SimpleNamespace(**paths,decision_sha256=file_hash(paths["decision_file"]))
    output=tmp_path/"prepared";output.mkdir()
    settings=cache_helper.strict_candidate_settings();protocol["source_replay_settings"]=settings
    manifest=dict(run_id="run",configuration_sha256="config",configuration=dict(pipeline={},language={},
        environment={},source_sha256={},runtime_settings=settings))
    roster=[dict(episode=0,policy_call_idx=0,episode_id="episode",context_id="context",status="available")]
    monkeypatch.setattr(validator,"load_inputs",lambda _: (protocol,{},roster,None,manifest,file_hash(args.metrics),{(0,0):{}},{}))
    monkeypatch.setattr(cache_helper,"apply_settings",lambda _:settings)
    monkeypatch.setattr(cache_helper,"runtime_settings",lambda:settings)
    monkeypatch.setattr(faithfulness,"replay_pipeline",lambda *args:({},{}))
    monkeypatch.setattr(faithfulness,"load_sidecar",lambda *args:({},"sidecar"))
    cache={name:torch.ones(1) for name in precision.CACHE_KEYS}
    cache["lang_attn_mask"]=torch.ones(1,dtype=torch.bool)
    def replay(*arguments,**kwargs):
        assert kwargs==dict(verify_reference=True,strict=True)
        if mutate_source: args.metrics.write_text("changed source")
        return cache
    monkeypatch.setattr(faithfulness,"replay_context",replay)
    if mutate_source:
        with pytest.raises(ValueError,match="changed during replay"):
            validator.prepare(args,output,dict(torch_not_preimported=True))
        assert not (output/"completion.json").exists()
        return
    validator.prepare(args,output,dict(torch_not_preimported=True))
    verified=cache_helper.verify_preparation(output,file_hash(output/"completion.json"))
    assert verified["implementation_sha256"]==validator.implementation_hashes()
    assert verified["contexts"][0]["source_replay_verified"]
    assert verified["status"]=="complete_source_preparation"


def test_source_checkpoint_contract_reads_actual_pipeline_checkpoint_fields(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts import validate_fp32_probe as base_validator
    protocol,bank=decision()
    bank_path=tmp_path/"bank.json";bank_path.write_text(json.dumps(bank))
    native_path=tmp_path/"native.json";native_path.write_text("{}")
    protocol.update(bank_sha256=file_hash(bank_path),native_registry_sha256=file_hash(native_path))
    decision_path=tmp_path/"decision.json";decision_path.write_text(json.dumps(protocol))
    args=SimpleNamespace(bank=bank_path,native_registry=native_path,decision_file=decision_path,
                         decision_sha256=file_hash(decision_path))
    manifest=dict(configuration=dict(pipeline=dict(checkpoint=protocol["checkpoint_identity"],checkpoint_mode="pretrained")))
    monkeypatch.setattr(base_validator,"load_bank_inputs",lambda _: (protocol,bank,bank["contexts"],None,manifest,"metrics",{}))
    assert validator.load_inputs(args)[4]==manifest
    manifest["configuration"]["pipeline"]["checkpoint_mode"]="authors"
    with pytest.raises(ValueError,match="authenticated source checkpoint"):
        validator.load_inputs(args)
