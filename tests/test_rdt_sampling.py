"""CPU sampler contract tests, including the real pinned Diffusers scheduler."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from experiment_io import tensor_hash
from integrated_gradients import NonFiniteAttributionError, integrated_gradients
from rdt_sampling import (
    conditional_sample_with_noise, make_initial_noise, sampler_metadata,
)


class ToyDenoiser(torch.nn.Module):
    def forward(self, state_action, ctrl_freqs, timestep, language, image, lang_mask):
        language_mean = (language * lang_mask.unsqueeze(-1)).mean(dim=1, keepdim=True)
        return (
            state_action[:, 1:] * 0.25 + state_action[:, :1] * 0.125
            + language_mean * 0.5 + image.mean(dim=1, keepdim=True) * 0.25
        )


class KnownUpdateScheduler:
    """A mutable solver with an analytically checkable step."""
    def __init__(self, config=None):
        self.config = {"algorithm_type": "dpmsolver++"} if config is None else dict(config)
        self.history = 0

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def set_timesteps(self, count):
        self.timesteps = torch.arange(count, 0, -1)

    def step(self, prediction, timestep, sample):
        self.history += 1
        return SimpleNamespace(prev_sample=sample * 0.5 + prediction / self.history)


def make_fixture(dtype=torch.float32, scheduler=None):
    # Explicitly construct weights; keep fixture setup outside the RNG tests.
    adaptor = torch.nn.Linear(4, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        adaptor.weight.copy_(torch.tensor([[1, 0, 0.25, 0], [0, 1, 0, 0.25]], dtype=dtype))
    runner = SimpleNamespace(
        state_adaptor=adaptor, model=ToyDenoiser(), training=False,
        noise_scheduler_sample=KnownUpdateScheduler() if scheduler is None else scheduler,
        num_inference_timesteps=3, pred_horizon=3, action_dim=2,
    )
    language = torch.tensor([[[1, -1], [2, -2]]], dtype=dtype)
    image = torch.tensor([[[0.5, 1.0], [1.5, -1.0]]], dtype=dtype)
    state = torch.tensor([[[0.25, -0.5]]], dtype=dtype)
    mask = torch.tensor([[[1, 0]]], dtype=dtype)
    frequency = torch.tensor([25], dtype=dtype)
    language_mask = torch.tensor([[True, False]])
    args = (language, language_mask, image, state, mask, frequency)
    return runner, args


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_known_updates_match_independent_recurrence_with_exact_masking(dtype):
    runner, args = make_fixture(dtype)
    language, language_mask, image, state, mask, _ = args
    initial = torch.tensor([[[0.5, -0.25], [1.0, 0.0], [-0.5, 0.5]]], dtype=dtype).float()
    result = conditional_sample_with_noise(runner, *args, initial)
    expected = initial.to(dtype)
    for step in range(1, 4):
        action_tokens = expected + mask * 0.25
        prediction = (
            action_tokens * 0.25 + state * 0.125
            + (language * language_mask.unsqueeze(-1)).mean(dim=1, keepdim=True) * 0.5
            + image.mean(dim=1, keepdim=True) * 0.25
        )
        expected = (expected * 0.5 + prediction / step).to(dtype)
    expected = expected * mask
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert torch.equal(result[..., 1], torch.zeros_like(result[..., 1]))


def test_noise_draw_and_sampling_preserve_global_rng_and_stored_tensor():
    runner, args = make_fixture(torch.bfloat16)
    state = args[3]
    before_rng = torch.random.get_rng_state().clone()
    initial = make_initial_noise(runner, state, 42)
    initial_copy = initial.clone()
    initial_hash = tensor_hash(initial)
    result = conditional_sample_with_noise(runner, *args, initial)
    repeated = conditional_sample_with_noise(runner, *args, initial)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert torch.equal(initial, initial_copy)
    assert tensor_hash(initial) == initial_hash
    assert torch.equal(result, repeated)
    assert torch.equal(initial, make_initial_noise(runner, state, 42))
    assert not torch.equal(initial, make_initial_noise(runner, state, 43))
    assert initial.dtype == torch.float32
    assert torch.equal(initial, initial.to(torch.bfloat16).float())


def test_scheduler_history_is_local_and_survives_interleaved_contexts():
    runner, args = make_fixture()
    runner.noise_scheduler_sample.history = 19
    runner.noise_scheduler_sample.set_timesteps(7)
    initial = make_initial_noise(runner, args[3], 42)
    first = conditional_sample_with_noise(runner, *args, initial)
    _ = conditional_sample_with_noise(runner, *args, initial + 1)
    second = conditional_sample_with_noise(runner, *args, initial)
    assert torch.equal(first, second)
    assert runner.noise_scheduler_sample.history == 19
    assert runner.noise_scheduler_sample.timesteps.tolist() == list(range(7, 0, -1))


def test_gradients_reach_each_condition_and_pre_adaptor_state_coordinates():
    runner, args = make_fixture()
    language, language_mask, image, state, mask, frequency = args
    language = language.clone().requires_grad_(True)
    image = image.clone().requires_grad_(True)
    # A raw state has two features plus fixed mask entries, as in RDT state IG.
    raw_state = state.clone().requires_grad_(True)
    state_token = runner.state_adaptor(torch.cat([raw_state, mask], dim=2))
    initial = make_initial_noise(runner, state, 42)
    result = conditional_sample_with_noise(
        runner, language, language_mask, image, state_token, mask, frequency, initial,
    )
    gradients = torch.autograd.grad(result.sum(), (language, image, raw_state))
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    # Masked language positions cannot affect this fixture's outputs.
    assert torch.equal(gradients[0][:, 1], torch.zeros_like(gradients[0][:, 1]))
    assert not mask.requires_grad


def test_noise_metadata_is_content_bound_and_json_serializable():
    runner, args = make_fixture()
    initial = make_initial_noise(runner, args[3], 7)
    metadata = sampler_metadata(runner, initial)
    assert metadata["initial_noise_sha256"] == tensor_hash(initial)
    assert metadata["initial_noise_sha256"] != tensor_hash(initial + 1)
    assert metadata["solver_steps"] == 3
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize("algorithm", ["sde-dpmsolver++", "sde-dpmsolver", "unknown"])
def test_stochastic_or_unrecognized_solver_is_rejected(algorithm):
    runner, args = make_fixture()
    runner.noise_scheduler_sample.config["algorithm_type"] = algorithm
    with pytest.raises(ValueError, match="deterministic"):
        conditional_sample_with_noise(runner, *args, torch.zeros(1, 3, 2))


def test_noise_that_requires_rounding_is_rejected():
    runner, args = make_fixture(torch.bfloat16)
    with pytest.raises(ValueError, match="exactly representable"):
        conditional_sample_with_noise(runner, *args, torch.full((1, 3, 2), 1.00001))


def test_training_mode_is_rejected():
    runner, args = make_fixture()
    runner.training = True
    with pytest.raises(ValueError, match="evaluation mode"):
        conditional_sample_with_noise(runner, *args, torch.zeros(1, 3, 2))


def test_nonfinite_noise_is_rejected_with_diagnostics():
    runner, args = make_fixture()
    initial = torch.zeros(1, 3, 2)
    initial[0, 1, 0] = float("nan")
    with pytest.raises(NonFiniteAttributionError) as error:
        conditional_sample_with_noise(runner, *args, initial)
    assert error.value.diagnostics["stage"] == "initial_noise"


def test_sampler_failure_is_bound_to_ig_path_and_modality():
    runner, args = make_fixture()
    language, language_mask, image, state, mask, frequency = args
    runner.model = lambda *args, **kwargs: torch.full((1, 3, 2), float("nan"))

    def forward(interpolated_language):
        return conditional_sample_with_noise(
            runner, interpolated_language, language_mask, image, state, mask,
            frequency, torch.zeros(1, 3, 2),
        ).sum()

    with pytest.raises(NonFiniteAttributionError) as error:
        integrated_gradients(
            forward, language, torch.zeros_like(language), m=2,
            diagnostic_context={"modality": "language", "run_id": "fixture"},
        )
    diagnostic = error.value.diagnostics
    assert diagnostic["stage"] == "sampler_model_output"
    assert diagnostic["inner_step"] == 0
    assert diagnostic["alpha"] == 0
    assert diagnostic["context"] == {"modality": "language", "run_id": "fixture"}


def test_actual_dpmsolver_repeats_and_is_differentiable_without_rng_changes():
    scheduler_module = pytest.importorskip("diffusers.schedulers.scheduling_dpmsolver_multistep")
    scheduler = scheduler_module.DPMSolverMultistepScheduler(
        num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample",
    )
    runner, args = make_fixture(scheduler=scheduler)
    language, language_mask, image, state, mask, frequency = args
    language = language.clone().requires_grad_(True)
    initial = make_initial_noise(runner, state, 42)
    before_rng = torch.random.get_rng_state().clone()
    original_scheduler_state = deepcopy(scheduler.__dict__)
    result = conditional_sample_with_noise(
        runner, language, language_mask, image, state, mask, frequency, initial,
    )
    gradient = torch.autograd.grad(result.sum(), language)[0]
    repeated = conditional_sample_with_noise(
        runner, language, language_mask, image, state, mask, frequency, initial,
    )
    assert torch.equal(result, repeated)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert scheduler._step_index == original_scheduler_state["_step_index"]
    assert scheduler.model_outputs == original_scheduler_state["model_outputs"]
    assert torch.equal(scheduler.timesteps, original_scheduler_state["timesteps"])
    json.dumps(sampler_metadata(runner, initial), allow_nan=False)
