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


def test_reference_corruption_is_rejected_not_reclassified_as_numerical_failure():
    actual = torch.ones(1, 3, 1)
    with pytest.raises(ValueError, match="reference replay"):
        paired.evaluate_rankings(toy_action, toy_action(actual)+1, actual, torch.zeros_like(actual), [0, 1, 2], 1,
            toy_rankings(), [0, 100], dict(Q=0, L2=0))


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
    stratum = dict(task="fixture", model="170m", seed_base=12, max_policy_calls=3, evaluate_calls=[0, 2])
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
    protocol = dict(stage="blinded_variance_pilot", bank_sha256="bank", e01_gate_sha256="gate",
        numerics=numerics, grid_percent=[0, 50, 100], random_permutations=2, random_seed=1,
        selection_filter="none", cluster_unit="reset_episode", failure_policy="retain_planned_denominators_no_replacement",
        denominator_min=dict(Q=0, L2=0), precision_plan="locked",
        contrast_family=[dict(modality="vision", method="Q_IG", control="random", response="Q", direction="deletion", metric="raw_auc")],
        analysis=dict(method="episode_percentile_bootstrap", family_alpha=.05, draws=1000, seed=9),
        sampling_plan="locked", monte_carlo_plan="locked", forward_precision="authenticated_source")
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
