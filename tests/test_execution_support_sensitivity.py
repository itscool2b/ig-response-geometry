"""Saved-action support, unchanged populations and source authentication; CPU only."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
from scipy.integrate import trapezoid
import torch

import execution_support_sensitivity as sensitivity
import paired_comparison as paired
import paired_study_analysis as study
from collect_contexts import collect_episode
from experiment_io import RunStore, canonical_json, file_hash, object_hash, strict_json, tensor_hash
from faithfulness import authenticated_source, load_sidecar


def protocol():
    return dict(kind="paired_execution_support_sensitivity_protocol", schema_version=1,
        recorded_before_comparative_inspection=True, stage="descriptive_saved_action_analysis_only",
        supported_collector_sha256=[sensitivity.COLLECTOR_SHA256],
        membership=dict(mode="authenticated_executed_source_positions_v1", predicted_chunk_length=64,
                        action_subsampling=4, active_indices=sensitivity.ACTIVE),
        drift=dict(action_ratio_minimum=1e-8, state_ratio_minimum=0.))


class Environment:
    def __init__(self, stop_after=None):
        self.stop_after = stop_after

    def reset(self, seed):
        self.steps = 0
        return {"agent": {"qpos": torch.zeros(1, 9)}}, {}

    def step(self, action):
        self.steps += 1
        assert action.shape == (1, 8)
        return {"agent": {"qpos": torch.ones(1, 9)*self.steps}}, 0., False, self.steps == self.stop_after, {}


def prepare(*args):
    return dict(initial_noise=torch.zeros(1,64,128), ref_action=torch.zeros(1,64,128,dtype=torch.bfloat16),
        lang_attn_mask=torch.ones(1,2,dtype=torch.bool), action_mask=torch.ones(1,1,128),
        ctrl_freqs=torch.tensor([25.]), sampler_metadata={})


def source_fixture(tmp_path, *, stops=(7,33), cap=400, max_calls=25):
    selection=dict(rule="uniform_executed_calls_v1", locked_before_collection=True,
        algorithm="numpy_pcg64_permutation_v1", seed_namespace="synthetic-support-test", seed=19,
        calls_per_episode=2, max_episode_steps=cap)
    config=dict(collector_type="context_only", control_mode="pd_joint_pos", action_subsampling=4,
        source_sha256={"collect_contexts.py":sensitivity.COLLECTOR_SHA256}, task="fixture", model="170m",
        pipeline={}, language={}, episodes=len(stops), seed_base=100, max_policy_calls=max_calls,
        max_episode_steps=cap, solver_steps=5, protocol_sha256="synthetic",
        protocol={"uniform_call_selection":selection},call_selection={"evaluate_calls":list(range(max_calls))})
    path=tmp_path/"source.jsonl"
    with RunStore(path,config).writer() as store:
        for ep, stop in enumerate(stops):
            spec={**config,"evaluate_calls":list(range(max_calls))}
            collect_episode(Environment(stop),store,ep,spec,prepare,render=lambda _:Image.new("RGB",(2,2)))
    rows,manifest=authenticated_source(path)
    terminals={d["episode"]:d["records"][-1] for d in
               (strict_json(p) for p in Path(str(path)+".run/episodes").glob("ep*.json"))}
    return path,rows,manifest,terminals,selection


def action(x, scale=1.):
    x=x.flatten()
    temporal=torch.ones(64)*3*x[1]
    temporal[::4]=x[0]
    temporal=scale*(temporal+.25*x[2])
    return temporal.reshape(1,64,1).expand(1,64,128)


def result(scale=1., exact=True):
    actual=torch.ones(1,3,1)
    orders={"Q_IG":[1.,3.,2.],"L2_IG":[3.,1.,2.],"input_difference":[1.,2.,3.],
            "Q_path_gradient":[2.,3.,1.],"L2_path_gradient":[3.,2.,1.]}
    rankings={name:paired.ranked(torch.tensor(values),[0,1,2]) for name,values in orders.items()}
    if exact:
        rankings["random_exact"]=dict(status="defined",control_kind="exact_uniform_subset_expectation")
    else:
        rankings.update({f"random_{n:04d}":paired.random_order([0,1,2],19,"fixture","state",n) for n in range(3)})
    return paired.evaluate_rankings(lambda x:action(x,scale),action(actual,scale),actual,torch.zeros_like(actual),
        [0,1,2],1,rankings,[0,1,50,100],dict(Q=0.,L2=0.,RMS=0.))


def test_mapping_uses_real_collector_step_counters_and_final_truncation(tmp_path):
    _,rows,manifest,terminals,_=source_fixture(tmp_path)
    mapping=sensitivity.executed_membership(rows,terminals,manifest["configuration"])
    assert mapping[0,0]["predicted_step_positions"]==list(range(0,28,4))
    assert mapping[1,0]["predicted_step_positions"]==list(range(0,64,4))
    assert mapping[1,1]["executed_commands"]==16
    assert mapping[1,2]["predicted_step_positions"]==[0]
    assert mapping[1,2]["terminal_record_sha256"]==object_hash(terminals[1])


def test_mapping_handles_explicit_step_cap_and_protocol_call_limit(tmp_path):
    cap_dir=tmp_path/"cap";cap_dir.mkdir()
    _,rows,manifest,ends,_=source_fixture(cap_dir,stops=(None,),cap=19)
    mapping=sensitivity.executed_membership(rows,ends,manifest["configuration"])
    assert mapping[0,1]["predicted_step_positions"]==[0,4,8]
    assert ends[0]["stop_reason"]=="collector_environment_step_cap"
    call_dir=tmp_path/"calls";call_dir.mkdir()
    _,rows,manifest,ends,_=source_fixture(call_dir,stops=(None,),max_calls=2)
    assert ends[0]["env_steps"]==32
    assert sensitivity.executed_membership(rows,ends,manifest["configuration"])[0,1]["executed_commands"]==16
    with pytest.raises(ValueError,match="complete trajectories"):
        paired.make_bank(call_dir/"source.jsonl",manifest["configuration"]["protocol"]["uniform_call_selection"])


@pytest.mark.parametrize("corruption", ["collector", "missing_call", "counter", "terminal", "bool_call", "cap", "hidden_cap"])
def test_mapping_never_guesses_when_membership_is_inconsistent(tmp_path,corruption):
    _,rows,manifest,ends,_=source_fixture(tmp_path,stops=(33,))
    config=manifest["configuration"]
    if corruption=="collector": config["source_sha256"]["collect_contexts.py"]="0"*64
    if corruption=="missing_call": rows.pop(1)
    if corruption=="counter": rows[1]["env_step_at_call"]=15
    if corruption=="terminal": ends[0]["policy_calls"]=2
    if corruption=="bool_call": rows[1]["policy_call_idx"]=True
    if corruption=="cap": ends[0].update(truncated=False,collector_step_cap=True,stop_reason="collector_environment_step_cap")
    if corruption=="hidden_cap": config["max_episode_steps"]=33
    with pytest.raises(ValueError): sensitivity.executed_membership(rows,ends,config)


def test_reprojection_keeps_grid_and_primary_and_separates_unexecuted_positions(monkeypatch):
    saved=result()
    monkeypatch.delattr(np,"trapezoid",raising=False)
    frozen=deepcopy(saved)
    out=sensitivity.reproject_results(saved,list(range(0,64,4)))
    assert saved==frozen
    assert out["full_chunk"]["realized_fractions"]==[0,0,2/3,1]
    outside_key=object_hash([1])
    assert out["full_chunk"]["response_rms"][outside_key]==pytest.approx(3*np.sqrt(.75))
    assert out["executed_source_positions"]["response_rms"][outside_key]==0
    for label,ranking in saved["rankings"].items():
        for direction in ("deletion","insertion"):
            curve=out["full_chunk"]["rankings"][label]["curves"][direction]
            assert curve["primary_saved_full_chunk_fp32_raw_auc"]==ranking["curves"][direction]["responses"]["RMS"]["raw_auc"]
            assert curve["raw_auc"]==pytest.approx(trapezoid(curve["values"],[0,0,2/3,1]))


def test_exact_expectation_takes_rms_before_averaging_opposite_actions():
    saved=result()
    keys=[object_hash([0]),object_hash([1]),object_hash([2])]
    reference=np.asarray(saved["active_reference_action"])
    for key,delta in zip(keys,[2.,-2.,0.]):
        saved["response_table"][key]["active_action"]=(reference+delta).tolist()
    # A primitive fixture isolates the nonlinearity at one exact-subset grid point.
    saved["rankings"]["random_exact"]["curves"]["deletion"]["response_ids"][2]=keys
    out=sensitivity.reproject_results(saved,[0])
    assert out["executed_source_positions"]["rankings"]["random_exact"]["curves"]["deletion"]["values"][2]==pytest.approx(4/3)
    assert np.sqrt(np.mean(np.mean([reference+2,reference-2,reference],axis=0)-reference)**2)==0


def test_nonfinite_full_response_remains_failed_on_every_support():
    saved=result()
    key=object_hash([1])
    saved["response_table"][key]=dict(status="numerical_failure",baseline_positions=[1],Q=None,L2=None,RMS=None)
    out=sensitivity.reproject_results(saved,[0])
    assert all(part["response_rms"][key] is None for part in out.values())
    assert all(part["rankings"]["random_exact"]["curves"]["insertion"]["raw_auc"] is None for part in out.values())


@pytest.mark.parametrize("denominator,status", [(None,"unavailable"),(0.,"zero_denominator"),(1e-9,"nearzero_denominator"),(1e-8,"nearzero_denominator"),(2e-8,"defined")])
def test_drift_ratios_keep_degenerate_denominators(denominator,status):
    out=sensitivity.guarded_ratio(2e-8,denominator,1e-8)
    assert out["status"]==status and out["numerator"]==2e-8 and out["denominator"]==denominator
    assert out["value"]==(1. if status=="defined" else None)


def completed_fixture(tmp_path, *, stage="confirmatory_locked", failed=False, prepared_bad_modality=None):
    source,source_rows,source_manifest,_,selection=source_fixture(tmp_path)
    bank=paired.make_bank(source,selection)
    protocol_path=tmp_path/"support-protocol.json";protocol_path.write_bytes(canonical_json(protocol()))
    output=tmp_path/"paired.jsonl"
    config=dict(protocol=dict(stage=stage,forward_precision=paired.FP32_PROBE,numerics={m:{} for m in study.MODALITIES}),
        protocol_sha256="protocol",bank=bank,bank_sha256=object_hash(bank),prepared_completion_sha256="a"*64)
    args=SimpleNamespace(out=output,metrics=source,limit=None)
    with paired.PairedEvaluationWriter(args,source_manifest,file_hash(source),"paired_comparison",config,3*len(bank["contexts"])) as writer:
        for entry in bank["contexts"]:
            row=next(r for r in source_rows if r["context_id"]==entry["context_id"])
            payload,_=load_sidecar(row,source,source_manifest)
            writer.set_context(row)
            for modality in study.MODALITIES:
                if failed and entry["episode"]==1 and modality=="language":
                    writer.write(dict(episode_id=entry["episode_id"],modality=modality,status="numerical_failure",failure_kind="probe_self_reference",reason="synthetic"))
                    continue
                saved=result(scale=1+entry["episode"]+.25*entry["policy_call_idx"],exact=modality=="state")
                provenance=dict(source_reference_sha256=tensor_hash(payload["ref_action"]),
                    probe_reference_active_values=saved["active_reference_action"],active_action_indices=sensitivity.ACTIVE,
                    reference_drift_l2=1.,reference_drift_rms=.1,reference_drift_max_abs=.2,state_token_drift_l2=.5,
                    cached_source_state_token_sha256=tensor_hash(torch.tensor([3.,4.])),prepared_cache={"file":"cache.pt","sha256":"b"*64})
                if modality==prepared_bad_modality: provenance["cached_source_state_token_sha256"]="d"*64
                writer.write(dict(episode_id=entry["episode_id"],modality=modality,status="evaluated",results=saved,probe_identity=provenance))
    kwargs=dict(study_sha256=file_hash(output),study_completion_sha256=file_hash(str(output)+".completion.json"),
                source_sha256=file_hash(source),protocol_sha256=file_hash(protocol_path))
    return output,source,protocol_path,kwargs


@pytest.mark.parametrize("failed",[False,True])
def test_completed_store_preserves_primary_joint_population_and_equal_episode_weights(tmp_path,failed):
    paths=completed_fixture(tmp_path,failed=failed)
    out=sensitivity.analyze(*paths[:3],**paths[3])
    assert out["planned_context_modality_rows"]==9
    assert out["summary"]["planned_episodes"]==2 and out["summary"]["planned_contexts"]==3
    assert len(out["summary"]["contrasts"])==30
    assert all(r.get("membership") for r in out["rows"])
    for contrast in out["summary"]["contrasts"]:
        assert contrast["conditional_episodes"]==(1 if failed else 2)
        for support in sensitivity.SUPPORTS:
            assert contrast["means"][support]==pytest.approx(np.mean([e[support] for e in contrast["episodes"]]))
        assert sorted(e["selected_calls"] for e in contrast["episodes"])==([1] if failed else [1,2])
    evaluated=next(r for r in out["rows"] if r["status"]=="evaluated")
    assert evaluated["drift"]["state_drift_over_source_norm"]["status"]=="unavailable"
    if failed:
        excluded=next(m for m in out["summary"]["membership"] if m["reset_seed"]==101)
        assert excluded["joint_complete"] is False
        assert excluded["conditional_episode_weight"]==0
        assert out["row_status_counts"]["numerical_failure"]==2


def test_support_contrast_changes_sign_and_uses_equal_episode_not_call_average(tmp_path):
    paths=completed_fixture(tmp_path)
    out=sensitivity.analyze(*paths[:3],**paths[3])
    rows,manifest,_=paired.load_completed_study(paths[0])
    bank=manifest["configuration"]["bank"]
    scales={}
    for entry in bank["contexts"]:
        scales.setdefault(entry["episode_id"],[]).append(1+entry["episode"]+.25*entry["policy_call_idx"])
    mean_scale=np.mean([np.mean(v) for v in scales.values()])
    pooled_scale=np.mean([v for values in scales.values() for v in values])
    assert mean_scale!=pooled_scale
    contrast=next(c for c in out["summary"]["contrasts"] if c["contrast"]["method"]=="L2_IG" and
        c["contrast"]["control"]=="Q_IG" and c["contrast"]["modality"]=="state" and c["contrast"]["direction"]=="deletion")
    # The sole interior point is at 2/3; its trapezoidal coefficient is 1/2.
    expected_full=.5*(np.sqrt(.25*1.25**2+.75*.25**2)-np.sqrt(.25*.25**2+.75*3.25**2))*mean_scale
    assert contrast["means"]["full_chunk"]==pytest.approx(expected_full)
    assert contrast["means"]["executed_source_positions"]==pytest.approx(.5*mean_scale)
    assert expected_full<0<contrast["means"]["executed_source_positions"]


def test_pilot_signed_output_and_changed_completion_hash_rejected(tmp_path):
    paths=completed_fixture(tmp_path,stage="variance_only_pilot")
    with pytest.raises(ValueError,match="variance-only pilot"):
        sensitivity.analyze(*paths[:3],**paths[3])
    paths[3]["study_completion_sha256"]="0"*64
    with pytest.raises(ValueError,match="Trusted input hash"):
        sensitivity.analyze(*paths[:3],**paths[3])


@pytest.mark.parametrize("bad_modality",[None,"state"])
def test_optional_state_denominator_checked_for_every_modality(tmp_path,monkeypatch,bad_modality):
    import fp32_probe_cache
    paths=completed_fixture(tmp_path,prepared_bad_modality=bad_modality)
    monkeypatch.setattr(fp32_probe_cache,"verify_preparation",lambda *a: {"test":"authenticated fixture"})
    calls=[]
    def load(*args):
        calls.append(args[2]["context_id"])
        return {"state_traj_actual":torch.tensor([3.,4.])},{"cache_file":{"file":"cache.pt","sha256":"b"*64}}
    monkeypatch.setattr(fp32_probe_cache,"load_prepared_context",load)
    if bad_modality:
        with pytest.raises(ValueError,match="State denominator differs"):
            sensitivity.analyze(*paths[:3],**paths[3],prepared=tmp_path/"prepared")
    else:
        out=sensitivity.analyze(*paths[:3],**paths[3],prepared=tmp_path/"prepared")
        assert len(calls)==3
        assert all(r["drift"]["state_drift_over_source_norm"]["value"]==.1 for r in out["rows"])


def test_source_sidecar_corruption_is_not_an_unavailable_sensitivity(tmp_path):
    paths=completed_fixture(tmp_path)
    sidecar=next(Path(str(paths[1])+".run").glob("episodes/ep*/*.pt"),None)
    if sidecar is None:
        sidecar=next(Path(str(paths[1])+".run").rglob("*.pt"))
    sidecar.write_bytes(b"changed")
    with pytest.raises(ValueError,match="hash|sidecar|Sidecar"):
        sensitivity.analyze(*paths[:3],**paths[3])
