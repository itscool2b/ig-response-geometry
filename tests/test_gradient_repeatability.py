"""Forward determinism is insufficient: exercise independent backward variation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiment_io import file_hash
from integrated_gradients import NonFiniteAttributionError, integrated_gradients
from scripts.validate_downstream_precision import CACHE_KEYS, cache_identity
from scripts.validate_gradient_repeatability import (
    configure_startup, load_authenticated_cache, pairwise_variation,
    path_gradient, rng_identity,
)


def test_strict_environment_is_set_before_torch_import_in_a_fresh_process():
    code = "from scripts.validate_gradient_repeatability import configure_startup; import sys,os,json; assert 'torch' not in sys.modules; print(json.dumps(configure_startup('deterministic'))); assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'"
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            check=True, text=True, capture_output=True)
    assert json.loads(result.stdout)["torch_not_preimported"] is True
    with pytest.raises(RuntimeError, match="fresh process"):
        configure_startup("deterministic")


def test_master_path_uses_exact_endpoints_and_bf16_forward_cast():
    seen = []
    actual = torch.tensor([1., 256.], dtype=torch.bfloat16)
    baseline = torch.tensor([65536., 0.], dtype=torch.bfloat16)
    def forward(value):
        seen.append(value.detach().clone())
        assert value.dtype == torch.bfloat16
        return value.float().square().sum()
    for alpha, expected in [(0., baseline), (1., actual)]:
        score, gradient, point = path_gradient(forward, actual, baseline, alpha, torch.bfloat16)
        assert torch.equal(seen[-1], expected)
        assert torch.equal(point, expected.float())
        assert torch.equal(gradient, (2 * expected).float())
        assert score == expected.float().square().sum().item()
    assert pairwise_variation([actual.float(), actual.float()])["all_bitwise_equal"]


class VariableBackward(torch.autograd.Function):
    counter = 0

    @staticmethod
    def forward(ctx, value):
        return value.sum()

    @staticmethod
    def backward(ctx, gradient):
        VariableBackward.counter += 1
        return gradient.expand(2) * VariableBackward.counter


def test_identical_forward_scores_can_hide_gradient_and_map_variation():
    VariableBackward.counter = 0
    actual, baseline = torch.tensor([1., 2.]), torch.zeros(2)
    forward = VariableBackward.apply
    probes = [path_gradient(forward, actual, baseline, .5, torch.float32) for _ in range(3)]
    assert [row[0] for row in probes] == [1.5] * 3
    report = pairwise_variation([row[1] for row in probes])
    assert not report["all_bitwise_equal"]
    assert len(report["pairs"]) == 3
    assert report["max_absolute_coordinate_difference"] == 2
    maps = [integrated_gradients(forward, actual, baseline, m=2) for _ in range(3)]
    assert not pairwise_variation(maps)["all_bitwise_equal"]


def test_repeat_metric_handles_zero_gradient_without_invented_relative_error():
    result = pairwise_variation([torch.zeros(3), torch.ones(3)])
    assert result["max_relative_l1_difference"] is None
    assert result["max_absolute_l1_difference"] == 3
    assert result["pairs"][0]["relative_l1_difference"] is None


def test_fixed_linear_gradient_and_map_repeats_are_exact_without_rng_consumption():
    actual, baseline = torch.tensor([1., 2.]), torch.zeros(2)
    coefficient = torch.tensor([3., -4.])
    before = rng_identity()
    forward = lambda value: (coefficient * value).sum()
    grads = [path_gradient(forward, actual, baseline, .25, torch.float32)[1] for _ in range(3)]
    maps = [integrated_gradients(forward, actual, baseline, m=8) for _ in range(3)]
    assert pairwise_variation(grads)["all_bitwise_equal"]
    assert pairwise_variation(maps)["all_bitwise_equal"]
    assert rng_identity() == before
    torch.testing.assert_close(maps[0], actual * coefficient)


@pytest.mark.parametrize("alpha", [-.1, 1.1, float("nan")])
def test_invalid_alphas_fail(alpha):
    with pytest.raises(ValueError, match="alpha"):
        path_gradient(lambda x: x.sum(), torch.ones(2), torch.zeros(2), alpha, torch.float32)


def test_nonfinite_derivative_is_not_sanitized():
    with pytest.raises(NonFiniteAttributionError):
        path_gradient(lambda value: value.sqrt().sum(), torch.ones(2), torch.zeros(2), 0., torch.float32)


def cache_fixture(tmp_path):
    cache = {key: torch.ones(1, dtype=torch.bfloat16) for key in CACHE_KEYS}
    cache["initial_noise"] = torch.ones(1, dtype=torch.float32)
    cache["lang_attn_mask"] = torch.ones(1, dtype=torch.bool)
    cache_file = tmp_path / "prepared-bf16-context.pt"
    torch.save(cache, cache_file)
    row = {"context_id": "context", "attr_sha256": "sidecar"}
    manifest = {"run_id": "run", "configuration_sha256": "configuration", "configuration": {
        "pipeline": {"checkpoint": "hash"}, "source_sha256": {"code.py": "source"}}}
    source = {"run_id": "run", "context_id": "context", "configuration_sha256": "configuration",
        "sidecar_sha256": "sidecar", "checkpoint_identity": manifest["configuration"]["pipeline"],
        "source_sha256": manifest["configuration"]["source_sha256"],
        "cache_file": {"file": cache_file.name, "sha256": file_hash(cache_file)},
        "cached_tensors": cache_identity(cache)}
    report = tmp_path / "report.json"
    report.write_text(json.dumps(source))
    snapshot = tmp_path / "source-precision-report.json"
    snapshot.write_bytes(report.read_bytes())
    return report, snapshot, cache_file, source, manifest, row, cache


def test_prepared_cache_and_snapshot_are_bound_to_source_context(tmp_path):
    report, snapshot, cache_file, source, manifest, row, payload = cache_fixture(tmp_path)
    # A later progress report cannot alter the bytes this job authenticated.
    report.write_text(json.dumps({"later": "progress"}))
    cache, parsed, provenance = load_authenticated_cache(report, manifest, row, payload, report_snapshot=snapshot)
    assert cache_identity(cache) == source["cached_tensors"]
    assert provenance["report_sha256"] == file_hash(snapshot)
    assert parsed == source


@pytest.mark.parametrize("change", ["context", "file", "tensor", "noise", "escape"])
def test_cache_identity_tampering_fails(tmp_path, change):
    report, snapshot, cache_file, source, manifest, row, payload = cache_fixture(tmp_path)
    if change == "context":
        source["context_id"] = "other"
    elif change == "file":
        with cache_file.open("ab") as stream:
            stream.write(b"corruption")
    elif change == "tensor":
        source["cached_tensors"]["img_adapted"]["sha256"] = "other"
    elif change == "noise":
        payload["initial_noise"] = torch.zeros(1)
    else:
        source["cache_file"]["file"] = "../prepared-bf16-context.pt"
    snapshot.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        load_authenticated_cache(report, manifest, row, payload, report_snapshot=snapshot)
