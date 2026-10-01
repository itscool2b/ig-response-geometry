import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiment_io import RunStore, object_hash, tensor_hash
from faithfulness import (AUC_K_GRID, auc_normalized, authenticated_source,
                          compute_modality_metrics, load_sidecar, load_step_rows,
                          topk_mask)


def test_zero_percent_and_stable_ties_use_exact_eligible_counts():
    scores = torch.tensor([1., 4., 4., 2.])
    assert topk_mask(scores, 0, 4).tolist() == [False] * 4
    assert topk_mask(scores, 25, 4).tolist() == [False, True, False, False]
    assert topk_mask(scores, 50, 4).tolist() == [False, True, True, False]
    assert topk_mask(scores, 100, 4).all()
    assert not topk_mask(torch.ones(2), 1, 2).any()


@pytest.mark.parametrize("scores,pct,n", [(torch.ones(2), -1, 2), (torch.ones(2), 101, 2),
                                         (torch.ones(2), float("nan"), 2),
                                         (torch.tensor([1., float("inf")]), 5, 2),
                                         (torch.empty(0), 10, 0),
                                         (torch.ones(2), 5, 3)])
def test_invalid_mask_requests_fail(scores, pct, n):
    with pytest.raises(ValueError):
        topk_mask(scores, pct, n)


def language_metrics(**options):
    values = torch.ones((1, 2, 1))
    baseline = torch.zeros_like(values)
    attribution = torch.tensor([[[1.0], [1.001]]], dtype=torch.float32)
    return compute_modality_metrics(lambda x: -((x.sum()-2)**2), values,
                                    baseline, attribution, "language",
                                    real_mask=torch.ones(2, dtype=torch.bool), **options)


def test_short_language_uses_realized_axis_and_preserves_fp32_ranking():
    metrics = language_metrics()
    assert metrics["ranking_indices"] == [1, 0]
    expected_counts = [round(2*k/100) for k in AUC_K_GRID]
    assert metrics["selected_counts"] == expected_counts
    assert metrics["realized_grid_percent"] == [50*k for k in expected_counts]
    assert metrics["dlogp_k1"] == metrics["dlogp_k5"] == metrics["dlogp_k10"] == 0
    # Equal additive features: realized trapezoids on 0, .5, 1 give .625.
    assert metrics["deletion_auc"] == pytest.approx(.625)
    assert metrics["insertion_auc"] == pytest.approx(.625)
    assert metrics["legacy_nominal_deletion_auc"] != metrics["deletion_auc"]
    legacy = language_metrics(auc_grid="legacy_nominal")
    assert legacy["deletion_auc"] == metrics["legacy_nominal_deletion_auc"]
    assert legacy["auc_axis"] == "legacy_nominal"


def test_padding_outside_eligibility_cannot_change_unreported_endpoint():
    value = torch.ones((1, 3, 1))
    with pytest.raises(ValueError, match="outside"):
        compute_modality_metrics(lambda x: x.sum(), value, torch.zeros_like(value), value,
                                 "language", real_mask=torch.tensor([True, True, False]))


def test_degenerate_gap_has_explicit_undefined_status_and_serializes_strictly():
    value = torch.ones((1, 2, 1))
    result = compute_modality_metrics(lambda x: x.sum()*0 + 1, value,
                                     torch.zeros_like(value), value,
                                     "language", real_mask=torch.ones(2, dtype=torch.bool))
    assert result["auc_status"] == "undefined_endpoint_gap"
    assert result["insertion_auc"] is None and result["deletion_auc"] is None
    assert result["endpoint_gap"] == 0
    json.dumps(result, allow_nan=False)


def test_nonfinite_forward_and_attribution_fail_before_publishing_scores():
    value = torch.ones((1, 2, 1))
    for forward, attribution in [(lambda x: x.sum()*float("nan"), value),
                                 (lambda x: x.sum(), value*float("inf"))]:
        with pytest.raises(ValueError, match="finite"):
            compute_modality_metrics(forward, value, torch.zeros_like(value), attribution,
                                     "language", real_mask=torch.ones(2, dtype=torch.bool))


def test_auc_does_not_clip_negative_scores_or_silently_apply_old_gap_cutoff():
    assert auc_normalized([0,50,100], [0,-100,1], 1, 0) == -49.75
    assert auc_normalized([0,100], [0,1e-12], 1e-12, 0) == pytest.approx(.5)
    assert auc_normalized([0,100], [0,1e-12], 1e-12, 0, denominator_min=1e-9) is None
    with pytest.raises(ValueError, match="Nonfinite"):
        auc_normalized([0,100], [0,float("inf")], 1, 0)
    with pytest.raises(ValueError, match="nondecreasing"):
        auc_normalized([0,100,50], [0,1,0], 1, 0)


def basic_row():
    return dict(event="step", task="fixture", model="170m", episode=0,
                seed=42, policy_call_idx=0, attr_file="fixture.pt")


@pytest.mark.parametrize("damage", ["duplicate", "malformed", "nonfinite", "overflow", "missing_sidecar"])
def test_source_loader_rejects_invalid_or_ambiguous_populations(tmp_path, damage):
    path = tmp_path / "metrics.jsonl"
    row = basic_row()
    if damage == "missing_sidecar":
        del row["attr_file"]
    if damage == "nonfinite":
        row["value"] = float("nan")
    raw = json.dumps(row) + "\n"
    if damage == "duplicate":
        raw += raw
    if damage == "malformed":
        raw += "{bad}\n"
    if damage == "overflow":
        raw = raw.rstrip()[:-1] + ', "value":1e999}\n'
    path.write_text(raw)
    with pytest.raises(ValueError):
        load_step_rows(path)


def authenticated_fixture(tmp_path):
    path = tmp_path / "metrics.jsonl"
    store = RunStore(path, dict(target="logpi", m=64))
    with store.writer():
        payload = dict(obs_image=torch.zeros(2,2,3,dtype=torch.uint8), proprio=torch.zeros(8),
                       initial_noise=torch.ones(1,2,3), ref_action=torch.zeros(1,2,3),
                       vision_attr=torch.ones(2), lang_attr=torch.ones(2))
        noise_hash = tensor_hash(payload["initial_noise"])
        context_id = object_hash(dict(episode=0, policy_call=0, configuration=store.identity,
                                     observation=tensor_hash(payload["obs_image"]),
                                     proprio=tensor_hash(payload["proprio"]), noise=noise_hash,
                                     reference=tensor_hash(payload["ref_action"])))
        payload.update(initial_noise_sha256=noise_hash, context_id=context_id)
        transaction = store.begin_episode(0)
        row = transaction.save_step({**basic_row(), "context_id":context_id, "initial_noise_sha256":noise_hash}, payload)
        transaction.commit(dict(event="episode_end", episode=0, policy_calls=1))
    return path, row, store.manifest


def test_manifestless_source_requires_explicit_unverified_mode(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(json.dumps(basic_row())+"\n")
    with pytest.raises(ValueError, match="manifest"):
        authenticated_source(path)
    rows, manifest = authenticated_source(path, legacy=True)
    assert manifest is None and len(rows) == 1


def test_authenticated_source_checks_export_sidecar_and_context_bytes(tmp_path):
    path, row, manifest = authenticated_fixture(tmp_path)
    rows, observed = authenticated_source(path)
    assert observed == manifest and len(rows) == 1
    payload, digest = load_sidecar(row, path, manifest)
    assert digest == row["attr_sha256"]
    assert payload["context_id"] == row["context_id"]
    wrong = {**row, "context_id":"invented"}
    with pytest.raises(ValueError, match="context"):
        load_sidecar(wrong, path, manifest)
    with pytest.raises(ValueError, match="bypass"):
        authenticated_source(path, legacy=True)
    # Valid JSON but a changed source row cannot masquerade as committed data.
    path.write_bytes(path.read_bytes().replace(b'"seed":42',b'"seed":43'))
    with pytest.raises(ValueError, match="export"):
        authenticated_source(path)


def test_missing_or_changed_sidecar_fails_source_authentication(tmp_path):
    path, row, manifest = authenticated_fixture(tmp_path)
    sidecar = path.parent / row["attr_file"]
    sidecar.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        authenticated_source(path)
