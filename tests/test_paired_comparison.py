"""CPU tests for paired identities, numerical controls and episode estimands."""
from copy import deepcopy
import os
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

import paired_comparison as paired
from collect_contexts import collect_episode
from experiment_io import RunStore, file_hash
from faithfulness import authenticated_source


SETTINGS = dict(m=16, quadrature="trapezoid", arithmetic_dtype="float32")


def test_path_gradient_captures_zero_displacement_coordinates_without_division():
    actual, baseline = torch.tensor([2., 0.]), torch.zeros(2)
    result, gradient = paired.integrate_with_path_gradient(lambda x: -(x.sum()-2).square(), actual, baseline, SETTINGS)
    assert torch.equal(result.attributions, torch.tensor([4., 0.]))
    assert torch.equal(gradient, torch.tensor([2., 2.]))
    endpoint = actual.clone().requires_grad_()
    assert torch.equal(torch.autograd.grad(-(endpoint.sum()-2).square(), endpoint)[0], torch.zeros(2))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_same_stream_gradient_reconstructs_kernel_in_each_forward_dtype(dtype):
    actual = torch.tensor([1.3, -0.7], dtype=dtype)
    result, gradient = paired.integrate_with_path_gradient(lambda x: x.float().square().sum(), actual, torch.zeros_like(actual), SETTINGS)
    assert torch.equal(result.attributions, actual.float()*gradient)


def test_common_scores_cast_before_subtraction_and_share_action_geometry():
    action = torch.tensor([[[1000., 1.]]], dtype=torch.bfloat16)
    reference = torch.tensor([[[-1000., 0.]]], dtype=torch.bfloat16)
    scores = paired.common_scores(action, reference, active_indices=[0, 1])
    squared = ((action.float()-reference.float())**2).sum()
    assert scores["Q"] == -squared/4
    assert scores["L2"] == -torch.sqrt(squared+1e-12)


def toy_action(x):
    return x.sum().reshape(1, 1, 1).expand(1, 2, 128)


def toy_rankings():
    return {"Q_IG": paired.ranked(torch.tensor([2., 1., 1.]), [0, 1, 2]),
            "input_difference": paired.ranked(torch.tensor([1., 1., 2.]), [0, 1, 2]),
            "random_0000": paired.random_order([0, 1, 2], 11, "context", "language", 0),
            "random_0001": paired.random_order([0, 1, 2], 11, "context", "language", 1)}


def test_response_crossing_uses_one_action_per_unique_mask_and_realized_counts():
    actual, baseline = torch.ones(1, 3, 1), torch.zeros(1, 3, 1)
    calls = []
    def action(x):
        calls.append(x.clone())
        return toy_action(x)
    out = paired.evaluate_rankings(action, toy_action(actual), actual, baseline, [0, 1, 2], 1,
                                  toy_rankings(), [0, 1, 50, 100], dict(Q=0, L2=0))
    assert out["selected_counts"] == [0, 0, 2, 3]
    assert out["realized_fractions"] == [0, 0, 2/3, 1]
    assert len(calls) == len(out["response_table"])
    for result in out["response_table"].values():
        assert result["Q"] <= 0 and result["L2"] <= 0
    curve = out["rankings"]["Q_IG"]["curves"]["deletion"]
    assert curve["selected_positions"][0] == curve["selected_positions"][1] == []
    assert curve["response_ids"][0] == curve["response_ids"][1] == out["input_response_id"]
    assert out["rankings"]["Q_IG"]["order"] == [0, 1, 2]


@pytest.mark.parametrize("threshold,status", [(0, "zero_endpoint_gap"), (1, "zero_endpoint_gap")])
def test_zero_gap_keeps_raw_response_area_and_undefined_normalized_auc(threshold, status):
    actual = torch.zeros(1, 3, 1)
    out = paired.evaluate_rankings(toy_action, toy_action(actual), actual, actual, [0, 1, 2], 1,
        toy_rankings(), [0, 50, 100], dict(Q=threshold, L2=threshold))
    result = out["rankings"]["Q_IG"]["curves"]["deletion"]["responses"]["Q"]
    assert result["raw_auc"] == 0 and result["normalized_auc"] is None
    assert result["status"] == status


def test_nearzero_gap_uses_declared_threshold_without_dropping_raw_outcome():
    actual = torch.full((1, 3, 1), 1e-7)
    out = paired.evaluate_rankings(toy_action, toy_action(actual), actual, torch.zeros_like(actual), [0, 1, 2], 1,
        toy_rankings(), [0, 50, 100], dict(Q=1e-6, L2=1e-6))
    result = out["rankings"]["Q_IG"]["curves"]["deletion"]["responses"]["Q"]
    assert result["status"] == "nearzero_endpoint_gap"
    assert result["raw_auc"] is not None and result["normalized_auc"] is None


def test_nonfinite_response_is_explicit_and_not_zeroed():
    actual, baseline = torch.ones(1, 3, 1), torch.zeros(1, 3, 1)
    def action(x):
        return toy_action(x) if x.sum() in (0, 3) else torch.full((1, 2, 128), float("nan"))
    out = paired.evaluate_rankings(action, toy_action(actual), actual, baseline, [0, 1, 2], 1,
        toy_rankings(), [0, 50, 100], dict(Q=0, L2=0))
    assert any(r["status"] == "numerical_failure" for r in out["response_table"].values())
    result = out["rankings"]["Q_IG"]["curves"]["deletion"]["responses"]["Q"]
    assert result["status"] == "nonfinite_response" and result["raw_auc"] is None


def test_probe_reference_drift_is_retained_as_numerical_failure():
    actual = torch.ones(1, 3, 1)
    result = paired.evaluate_rankings(toy_action, toy_action(actual)+1, actual, torch.zeros_like(actual), [0, 1, 2], 1,
        toy_rankings(), [0, 100], dict(Q=0, L2=0, RMS=0))
    assert result["response_table"][result["input_response_id"]]["failure_kind"] == "probe_self_reference"
    assert result["rankings"]["Q_IG"]["curves"]["deletion"]["responses"]["RMS"]["raw_auc"] is None


def test_undeclared_population_differences_and_duplicate_ranks_fail():
    with pytest.raises(ValueError, match="outside"):
        paired.validate_population(torch.ones(1, 3, 1), torch.zeros(1, 3, 1), [0, 2], 1)
    actual = torch.ones(1, 3, 1)
    with pytest.raises(ValueError, match="permute"):
        paired.evaluate_rankings(toy_action, toy_action(actual), actual, torch.zeros_like(actual), [0, 1, 2], 1,
            {"bad": dict(status="defined", order=[0, 0, 2])}, [0, 100], dict(Q=0, L2=0))


def test_random_stream_is_local_reproducible_and_context_bound():
    before = np.random.get_state()
    first = paired.random_order(list(range(100)), 42, "first", "vision", 0)
    second = paired.random_order(list(range(100)), 42, "second", "vision", 0)
    assert first == paired.random_order(list(range(100)), 42, "first", "vision", 0)
    assert first["order"] != second["order"]
    assert np.array_equal(np.random.get_state()[1], before[1])


def summary_record(episode, call, treatment, controls=(0., 2.)):
    def ranking(value):
        return dict(status="defined", curves={"deletion": {"responses": {"Q": {"raw_auc": value}}}})
    return dict(episode_id=episode, policy_call_idx=call, modality="vision", status="evaluated",
                results={"rankings": {"Q_IG": ranking(treatment),
                         **{f"random_{i:04}": ranking(value) for i, value in enumerate(controls)}}})


def summarize(records, **kwargs):
    return paired.paired_episode_summary(records, method="Q_IG", control="random", response="Q",
        direction="deletion", metric="raw_auc", alpha=.05, draws=1000, seed=9, **kwargs)


def test_episode_equal_pairing_not_call_or_permutation_inflation():
    records = [summary_record("a", i, -1) for i in range(10)] + [summary_record("b", 0, -3)]
    result = summarize(records)
    assert result["planned_contexts"] == 11 and result["analysis_episodes"] == 2
    assert result["episode_effects"] == {"a": 2., "b": 4.}
    assert result["estimate"] == 3.
    assert result["point_population_sha256"] == result["ci_population_sha256"]
    assert result["random_monte_carlo_standard_error"] == pytest.approx(np.sqrt((.1+1)/4))


def test_missing_call_keeps_denominator_and_makes_primary_episode_incomplete():
    missing = dict(episode_id="a", policy_call_idx=1, modality="vision", status="call_not_reached")
    result = summarize([summary_record("a", 0, -1), missing, summary_record("b", 0, -3)])
    assert result["planned_contexts"] == 3 and result["valid_contexts"] == 2
    assert result["planned_episodes"] == 2 and result["analysis_episodes"] == 1
    assert result["estimate"] == 4 and result["confidence_interval"] is None
    assert result["incomplete_episode_ids"] == ["a"]


def test_duplicate_context_is_not_an_extra_sample():
    record = summary_record("a", 0, -1)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([record, deepcopy(record)])


class TinyEnvironment:
    def reset(self, seed):
        return {"agent": {"qpos": torch.zeros(1, 9)}}, {}
    def step(self, action):
        return {"agent": {"qpos": torch.zeros(1, 9)}}, 0., False, True, {}


def context_source(tmp_path):
    config = dict(collector_type="context_only", task="fixture", model="170m", pipeline={}, language={},
                  seed_base=12, episodes=1, solver_steps=5, protocol_sha256="protocol",
                  call_selection=dict(evaluate_calls=[0, 2]))
    stratum = dict(task="fixture", model="170m", seed_base=12, max_policy_calls=3, max_episode_steps=400, evaluate_calls=[0, 2])
    def prepare(*args):
        return dict(initial_noise=torch.zeros(1, 64, 128), ref_action=torch.zeros(1, 64, 128),
                    lang_attn_mask=torch.ones(1, 2, dtype=torch.bool), action_mask=torch.ones(1, 1, 128),
                    ctrl_freqs=torch.tensor([25.]), sampler_metadata={})
    output = tmp_path/"contexts.jsonl"
    with RunStore(output, config).writer() as store:
        collect_episode(TinyEnvironment(), store, 0, stratum, prepare, render=lambda _: Image.new("RGB", (2, 2)))
    return output


def test_bank_defaults_to_prespecified_calls_and_retains_missing_without_ig(tmp_path):
    source = context_source(tmp_path)
    bank = paired.make_bank(source)
    assert [entry["policy_call_idx"] for entry in bank["contexts"]] == [0, 2]
    assert [entry["status"] for entry in bank["contexts"]] == ["available", "call_not_reached"]
    rows, manifest = authenticated_source(source)
    selected = paired.bank_rows(bank, rows, manifest, file_hash(source))
    assert list(selected) == [(0, 0)]
    corrupted = deepcopy(bank)
    corrupted["contexts"][0]["row_sha256"] = "changed"
    with pytest.raises(ValueError, match="mismatch"):
        paired.bank_rows(corrupted, rows, manifest, file_hash(source))


def test_bank_cannot_retroactively_add_unselected_calls(tmp_path):
    selection = dict(rule="new", locked_before_collection=True, derivative_protocol_sha256="hash",
                     contexts=[dict(episode=0, policy_call_idx=1)])
    with pytest.raises(ValueError, match="not prespecified"):
        paired.make_bank(context_source(tmp_path), selection)


def protocol_fixture():
    numerics = {"vision": {target: SETTINGS.copy() for target in ("Q", "L2")}}
    gate = dict(status="approved_for_paired_study", approved_numerics=numerics,
                forward_precision="authenticated_source",
                integrated_gradients_sha256=file_hash(Path(paired.__file__).parent/"integrated_gradients.py"),
                paired_comparison_sha256=file_hash(paired.__file__))
    protocol = dict(stage="variance_only_pilot", bank_sha256="bank", e01_gate_sha256="gate",
        numerics=numerics, grid_percent=[0, 50, 100], random_permutations=2, random_seed=1,
        selection_filter="none", cluster_unit="reset_episode", failure_policy="retain_planned_denominators_no_replacement",
        denominator_min=dict(Q=0, L2=0, RMS=0), precision_plan="locked",
        contrast_family=[dict(modality="vision", method="Q_IG", control="random", response="Q", direction="deletion", metric="raw_auc")],
        analysis=dict(method="episode_percentile_bootstrap", family_alpha=.05, draws=1000, seed=9),
        sampling_plan="locked", monte_carlo_plan="locked", forward_precision="authenticated_source",
        random_control="exact_uniform_subsets_n_le_8_else_seeded_permutations")
    return protocol, gate


def test_no_implicit_integration_budget_or_unapproved_target():
    protocol, gate = protocol_fixture()
    paired.validate_protocol(protocol, "bank", gate, "gate")
    changed = deepcopy(protocol)
    changed["numerics"]["vision"]["L2"]["m"] = 64
    with pytest.raises(ValueError, match="differ"):
        paired.validate_protocol(changed, "bank", gate, "gate")
    gate["status"] = "engineering_only"
    with pytest.raises(ValueError, match="not approved"):
        paired.validate_protocol(protocol, "bank", gate, "gate")


def test_fp32_probe_uses_one_readapted_actual_state_and_new_self_reference():
    runner = torch.nn.Module()
    runner.state_adaptor = torch.nn.Linear(256, 2, bias=False)
    torch.nn.init.constant_(runner.state_adaptor.weight, .1)
    source = dict(lang_adapted=torch.ones(1, 2, 2, dtype=torch.bfloat16),
                  lang_adapted_bl=torch.zeros(1, 2, 2, dtype=torch.bfloat16),
                  img_adapted=torch.ones(1, 2, 2, dtype=torch.bfloat16),
                  img_adapted_bl=torch.zeros(1, 2, 2, dtype=torch.bfloat16),
                  state_input_actual=torch.ones(1, 1, 128, dtype=torch.bfloat16),
                  state_input_baseline=torch.zeros(1, 1, 128, dtype=torch.bfloat16),
                  state_traj_actual=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
                  action_mask=torch.ones(1, 1, 128, dtype=torch.bfloat16),
                  ctrl_freqs=torch.tensor([25.], dtype=torch.bfloat16),
                  initial_noise=torch.zeros(1, 2, 128, dtype=torch.bfloat16),
                  ref_action=torch.zeros(1, 2, 128, dtype=torch.bfloat16),
                  lang_attn_mask=torch.ones(1, 2, dtype=torch.bool))
    def sample(runner, lang, mask, image, state, actionmask, freq, noise):
        return (lang.sum()+image.sum()+state.sum()).reshape(1,1,1).expand_as(noise)
    ctx, provenance = paired.make_fp32_probe_context(source, runner, sample_fn=sample)
    state_endpoint = ctx["seeded_conditional_sample"](ctx["lang_adapted"], ctx["img_adapted"],
        runner.state_adaptor(torch.cat([ctx["state_input_actual"], ctx["action_mask"]], dim=2)))
    assert torch.equal(state_endpoint, ctx["ref_action"])
    assert paired.common_scores(state_endpoint, ctx["ref_action"])["Q"] == 0
    assert provenance["reference_drift_l2"] > 0 and provenance["state_token_drift_l2"] > 0
    assert torch.equal(source["state_traj_actual"], torch.zeros_like(source["state_traj_actual"]))


def test_separately_copied_script_reuses_canonical_helpers(tmp_path):
    copied = tmp_path/"paired_copy.py"
    copied.write_bytes(Path(paired.__file__).read_bytes())
    env = {**os.environ, "PYTHONPATH": str(Path(paired.__file__).parent)}
    result = subprocess.run([sys.executable, str(copied), "--help"], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "summarize" in result.stdout
    protocol, gate = protocol_fixture()
    config = tmp_path/"protocol.json"
    gate_path = tmp_path/"gate.json"
    config.write_text(json.dumps(protocol), encoding="utf-8")
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    code = "import importlib.util, json, sys; spec=importlib.util.spec_from_file_location('copied',sys.argv[1]); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); module.validate_protocol(json.load(open(sys.argv[2])), 'bank', json.load(open(sys.argv[3])), 'gate')"
    checked = subprocess.run([sys.executable, "-c", code, str(copied), str(config), str(gate_path)],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr


def test_numerical_gate_cannot_be_reused_for_an_unvalidated_model_or_task():
    bank = dict(task="fixture", model="1b", pipeline={"checkpoint": "weights"}, language={"task": "fixture"})
    with pytest.raises(ValueError, match="stratum"):
        paired.validate_gate_population({"approved_strata": []}, bank)
    identity = dict(task="fixture", model="1b", pipeline_sha256=paired.object_hash(bank["pipeline"]),
                    language_sha256=paired.object_hash(bank["language"]))
    paired.validate_gate_population({"approved_strata": [identity]}, bank)


def uniform_fixture():
    selection = dict(rule="uniform_executed_calls_v1", locked_before_collection=True,
        algorithm="numpy_pcg64_permutation_v1", seed_namespace="pilot-call-selection", seed=9941,
        calls_per_episode=2, max_episode_steps=400)
    config = dict(task="fixture", pipeline={"checkpoint":"identified"}, max_episode_steps=400,
                  protocol={"uniform_call_selection":deepcopy(selection)})
    terminals = {ep: dict(seed=42+ep,stop_reason="truncated",truncated=True,terminated=False,
                         env_steps=16*n,policy_calls=n) for ep,n in enumerate((7,1))}
    rows = [dict(episode=ep,policy_call_idx=c) for ep,t in terminals.items() for c in range(t["policy_calls"])]
    return selection,config,terminals,rows


def test_uniform_executed_selection_is_precommitted_local_and_keeps_short_episodes():
    selection,config,terminals,rows = uniform_fixture()
    state = np.random.get_state()
    chosen,accounting = paired.uniform_executed_selection(config,terminals,rows,selection)
    assert chosen == paired.uniform_executed_selection(config,terminals,rows,selection)[0]
    assert np.array_equal(state[1],np.random.get_state()[1])
    assert len(chosen)==3 and accounting[1]["selected_calls"]==[0]
    assert accounting[0]["inclusion_probability"]==2/7
    assert accounting[0]["within_episode_weight"]==.5 and accounting[1]["within_episode_weight"]==1
    assert all(item["episode_weight"]==.5 for item in accounting)
    # Success and numerical-quality values cannot affect call selection.
    terminals[0]["success"] = True
    for row in rows: row["ref_norm_maniskill"] = 0
    assert chosen == paired.uniform_executed_selection(config,terminals,rows,selection)[0]
    bad=deepcopy(selection);bad["seed"]+=1
    with pytest.raises(ValueError,match="original collection protocol"):
        paired.uniform_executed_selection(config,terminals,rows,bad)


def test_uniform_selection_rejects_incomplete_horizon_or_missing_executed_calls():
    selection,config,terminals,rows = uniform_fixture()
    terminals[0].update(stop_reason="protocol_call_limit",truncated=False)
    with pytest.raises(ValueError,match="complete trajectories"):
        paired.uniform_executed_selection(config,terminals,rows,selection)
    terminals[0].update(stop_reason="collector_environment_step_cap",collector_step_cap=True)
    with pytest.raises(ValueError,match="complete declared horizon"):
        paired.uniform_executed_selection(config,terminals,rows,selection)
    terminals[0]["env_steps"]=400
    assert paired.uniform_executed_selection(config,terminals,rows,selection)
    with pytest.raises(ValueError,match="every executed call"):
        paired.uniform_executed_selection(config,terminals,rows[1:],selection)


def weighted_action(x):
    weights=torch.arange(1,x.shape[1]+1,dtype=x.dtype).reshape(1,-1,1)
    return (x*weights).sum().reshape(1,1,1).expand(1,2,128)


def test_exact_subset_control_matches_all_permutations_and_reuses_masks():
    from itertools import permutations
    actual=torch.ones(1,3,1);baseline=torch.zeros_like(actual)
    rankings={"random_exact":dict(status="defined",control_kind="exact_uniform_subset_expectation")}
    out=paired.evaluate_rankings(weighted_action,weighted_action(actual),actual,baseline,[0,1,2],1,
                                 rankings,[0,100/3,200/3,100],dict(Q=0,L2=0,RMS=0))
    assert len(out["response_table"])==8
    paired.authenticate_action_results(out)
    for direction in ("deletion","insertion"):
        observed=out["rankings"]["random_exact"]["curves"][direction]["responses"]["RMS"]["values"]
        expected=[]
        for count in range(4):
            values=[]
            for order in permutations(range(3)):
                positions=order[:count] if direction=="deletion" else order[count:]
                changed=paired.replace_positions(actual,baseline,list(positions),1)
                values.append(float(paired.common_scores(weighted_action(changed),weighted_action(actual))["RMS"]))
            expected.append(np.mean(values))
        assert observed==pytest.approx(expected)
    corrupted=deepcopy(out)
    corrupted["response_table"][out["input_response_id"]]["active_action"][0][0][0]+=1
    with pytest.raises(ValueError,match="hash differs"):
        paired.authenticate_action_results(corrupted)


def test_state_exact_random_control_has_at_most_256_interventions():
    actual=torch.ones(1,8,1);baseline=torch.zeros_like(actual)
    rankings=paired.build_rankings(weighted_action,weighted_action(actual),actual,baseline,list(range(8)),1,
        {target:dict(m=2,quadrature="trapezoid",arithmetic_dtype="float32") for target in ("Q","L2")},
        context_id="fixture",modality="state",random_permutations=2,random_seed=5)
    assert "random_exact" in rankings and "random_0000" not in rankings
    out=paired.evaluate_rankings(weighted_action,weighted_action(actual),actual,baseline,list(range(8)),1,
                                 rankings,[12.5*k for k in range(9)],dict(Q=0,L2=0,RMS=0))
    assert len(out["response_table"])==256
    exact=out["rankings"]["random_exact"]
    assert exact["monte_carlo_variance"]==0
    assert exact["curves"]["deletion"]["subset_counts"]==[1,8,28,56,70,56,28,8,1]


def test_raw_rms_advantage_has_displacement_sign_and_exact_control_zero_mc_error():
    def ranking(value,exact=False):
        return dict(status="defined",control_kind="exact_uniform_subset_expectation" if exact else "IG",
                    curves={"deletion":{"responses":{"RMS":{"raw_auc":value}}}})
    rows=[dict(episode_id=str(i),policy_call_idx=0,modality="state",status="evaluated",
               results={"rankings":{"Q_IG":ranking(3),"random_exact":ranking(1,True)}}) for i in range(2)]
    summary=paired.paired_episode_summary(rows,method="Q_IG",control="random",response="RMS",
            direction="deletion",metric="raw_auc",alpha=.05,draws=100,seed=1)
    assert summary["estimate"]==2 and summary["random_monte_carlo_standard_error"]==0


def test_production_cli_refuses_torch_preimport_before_startup(tmp_path):
    protocol=tmp_path/"protocol.json"
    protocol.write_text(json.dumps(dict(forward_precision=paired.FP32_PROBE)))
    code="import torch,runpy,sys;sys.argv=[sys.argv[1],'run','--protocol',sys.argv[2]];runpy.run_path(sys.argv[0],run_name='__main__')"
    result=subprocess.run([sys.executable,"-c",code,paired.__file__,str(protocol)],capture_output=True,text=True)
    assert result.returncode!=0 and "fresh process" in result.stderr


def test_uniform_bank_uses_all_executed_contexts_under_prelocked_rule(tmp_path):
    selection,_,_,_=uniform_fixture()
    class ThreeCallEnvironment(TinyEnvironment):
        def reset(self,seed):
            self.steps=0
            return super().reset(seed)
        def step(self,action):
            self.steps+=1
            return {"agent":{"qpos":torch.zeros(1,9)}},0.,False,self.steps==33,{}
    config=dict(collector_type="context_only",task="fixture",model="170m",pipeline={},language={},
        seed_base=12,episodes=1,solver_steps=5,protocol_sha256="protocol",max_episode_steps=400,
        protocol={"uniform_call_selection":selection},call_selection=dict(evaluate_calls=[0]))
    stratum=dict(task="fixture",model="170m",seed_base=12,max_policy_calls=25,max_episode_steps=400,evaluate_calls=[0])
    def prepare(*args):
        return dict(initial_noise=torch.zeros(1,64,128),ref_action=torch.zeros(1,64,128),
            lang_attn_mask=torch.ones(1,2,dtype=torch.bool),action_mask=torch.ones(1,1,128),
            ctrl_freqs=torch.tensor([25.]),sampler_metadata={})
    source=tmp_path/"uniform.jsonl"
    with RunStore(source,config).writer() as store:
        collect_episode(ThreeCallEnvironment(),store,0,stratum,prepare,render=lambda _:Image.new("RGB",(2,2)))
    bank=paired.make_bank(source,selection)
    assert len(bank["contexts"])==2
    assert any(entry["policy_call_idx"]>0 for entry in bank["contexts"])
    assert bank["selection_accounting"][0]["eligible_calls"]==[0,1,2]
    assert bank==paired.make_bank(source,bank["selection"])


def completed_fixture(tmp_path):
    from types import SimpleNamespace
    source=tmp_path/"source.jsonl";source.write_text("source")
    output=tmp_path/"paired.jsonl"
    args=SimpleNamespace(out=output,metrics=source,limit=None)
    bank=dict(contexts=[dict(episode_id="ep",policy_call_idx=0,context_id="context")])
    config=dict(protocol={"numerics":{"vision":{}}},protocol_sha256="protocol",bank=bank,bank_sha256="bank")
    source_manifest=dict(run_id="source",configuration_sha256="config")
    actual=torch.ones(1,3,1)
    results=paired.evaluate_rankings(toy_action,toy_action(actual),actual,torch.zeros_like(actual),[0,1,2],1,
                                    toy_rankings(),[0,50,100],dict(Q=0,L2=0,RMS=0))
    with paired.PairedEvaluationWriter(args,source_manifest,file_hash(source),"paired_comparison",config,1) as writer:
        writer.set_context(dict(episode=0,seed=42,policy_call_idx=0,context_id="context",attr_sha256="sidecar"))
        writer.write_auxiliary("fixture.json",{"cpu":True})
        writer.write_tensor_auxiliary("failed-probe.pt",{"ref_action":torch.tensor([float("nan")])})
        writer.write(dict(episode_id="ep",modality="vision",status="evaluated",results=results))
    return output


def test_completed_shard_authenticates_manifest_membership_and_actions(tmp_path):
    output=completed_fixture(tmp_path)
    rows,manifest,completion=paired.load_completed_study(output)
    assert len(rows)==1 and completion["manifest_sha256"]==file_hash(str(output)+".manifest.json")
    assert completion["auxiliary_sha256"]["fixture.json"]==file_hash(str(output)+".fixture.json")
    assert torch.isnan(torch.load(str(output)+".failed-probe.pt",weights_only=True)["ref_action"]).all()
    path=Path(str(output)+".manifest.json")
    changed=json.loads(path.read_text());changed["configuration"]["protocol"]["analysis"]="changed"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="manifest"):
        paired.load_completed_study(output)


def test_population_hash_binds_calls_and_contexts_not_only_episode_ids():
    left=[summary_record("a",0,-1),summary_record("b",0,-2)]
    for i,row in enumerate(left): row["source_context_id"]="context-"+str(i)
    right=deepcopy(left);right[0]["policy_call_idx"]=2
    assert summarize(left)["point_population_sha256"] != summarize(right)["point_population_sha256"]
    missing=deepcopy(left)
    missing[0]["results"]["rankings"]["Q_IG"]={"status":"numerical_failure","failure_kind":"integration_numerical_failure"}
    result=summarize(missing)
    assert result["paired_validity_discordance"]["method_only_undefined"]==1
    assert result["method_outcome_counts"]["integration_numerical_failure"]==1
