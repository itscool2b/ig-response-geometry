import json
import math
from pathlib import Path

import pytest
import torch

import revision_numerics as rn
from experiment_io import file_hash
from paired_comparison import evaluate_rankings, random_order, ranked, grouped_scores
from per_step_attribution import MANISKILL_INDICES
from tests.revision_toy import DEFAULT, GRID, STRICT, stub_replay_layer, toy_context, toy_protocol, worker_args


def historical_count(n, k):
    """faithfulness.topk_mask at e4ad8bf, applied at interior grid points."""
    return min(n, max(1, int(round(n * k / 100.0))))


@pytest.mark.parametrize("n", [729, 30, 25, 20, 16, 8, 1])
def test_count_rules(n):
    counts = rn.count_schedule(n, GRID, "max1_round")
    assert counts[0] == 0 and counts[-1] == n
    assert counts[1:-1] == [historical_count(n, k) for k in GRID[1:-1]]
    assert rn.count_schedule(n, GRID, "round") == [round(n * k / 100) for k in GRID]
    with pytest.raises(ValueError):
        rn.count_schedule(n, GRID, "ceil")


def test_language_and_state_counts_differ_only_where_rounding_reaches_zero():
    assert rn.count_schedule(20, GRID, "round") == [0, 0, 1, 2, 4, 6, 10, 15, 20]
    assert rn.count_schedule(20, GRID, "max1_round") == [0, 1, 1, 2, 4, 6, 10, 15, 20]
    assert rn.count_schedule(8, GRID, "max1_round") == [0, 1, 1, 1, 2, 2, 4, 6, 8]
    assert rn.count_schedule(729, GRID, "max1_round") == rn.count_schedule(729, GRID, "round")


def test_fixed_orders_are_permutations_with_the_documented_geometry():
    vision = list(range(2187, 2916))
    orders = rn.fixed_orders("vision", vision)
    assert orders["fixed_raster"] == vision and orders["fixed_reverse_raster"] == vision[::-1]
    centre = orders["fixed_centre_out"]
    assert sorted(centre) == vision and centre[0] == 2187 + 13 * 27 + 13
    assert set(centre[1:5]) == {2187 + 12 * 27 + 13, 2187 + 14 * 27 + 13, 2187 + 13 * 27 + 12, 2187 + 13 * 27 + 14}
    assert rn.fixed_orders("state", MANISKILL_INDICES)["fixed_reverse_joint_order"] == MANISKILL_INDICES[::-1]
    with pytest.raises(ValueError):
        rn.fixed_orders("vision", list(range(10)))


@pytest.mark.parametrize("modality", ["vision", "language", "state"])
def test_round_rule_reproduces_evaluate_rankings_exactly(modality):
    from paired_comparison import modality_inputs
    ctx = toy_context()
    actual, baseline, indices, axis, forward = modality_inputs(ctx, modality)
    scores = grouped_scores(actual.float() - baseline.float(), indices, axis)
    rankings = {"input_difference": ranked(scores, indices)}
    if len(indices) <= 8:
        rankings["random_exact"] = dict(status="defined", control_kind="exact_uniform_subset_expectation",
                                        eligible_count=len(indices), monte_carlo_variance=0.0)
    else:
        rankings["random_0000"] = random_order(indices, 3, "ctx", modality, 0)
    minimum = dict(Q=1e-12, L2=1e-8, RMS=1e-8)
    expected = evaluate_rankings(forward, ctx["ref_action"], actual, baseline, indices, axis,
                                 json.loads(json.dumps(rankings)), GRID, minimum)
    observed = rn.evaluate_orders(forward, ctx["ref_action"], actual, baseline, indices, axis,
                                  json.loads(json.dumps(rankings)), GRID, minimum, count_rule="round")
    assert observed.pop("count_rule") == "round"
    for ranking in observed["rankings"].values():
        for curve in ranking["curves"].values():
            for response in curve["responses"].values():
                response.pop("nominal_normalized_auc")
    assert observed == expected


def test_nominal_axis_area_matches_the_historical_definition():
    from faithfulness import auc_normalized
    from paired_comparison import modality_inputs
    ctx = toy_context()
    actual, baseline, indices, axis, forward = modality_inputs(ctx, "language")
    rankings = {"fixed": rn.order_ranking(indices, "fixed_token_position")}
    out = rn.evaluate_orders(forward, ctx["ref_action"], actual, baseline, indices, axis, rankings, GRID,
                             dict(Q=0, L2=0, RMS=0), count_rule="max1_round")
    curve = out["rankings"]["fixed"]["curves"]["deletion"]["responses"]["Q"]
    table = out["response_table"]
    endpoints = table[out["input_response_id"]]["Q"], table[out["baseline_response_id"]]["Q"]
    assert curve["nominal_normalized_auc"] == auc_normalized(GRID, curve["values"], *endpoints)
    assert out["selected_counts"] == [0, 1, 1, 2, 4, 6, 10, 15, 20]


def run_phase(tmp_path, phase, body, binding=None):
    binding = binding or dict(protocol_sha256="p", bank_sha256="b", context_id="ctx")
    with rn.PhaseOutput(tmp_path, "P", "e0000-c000", phase, binding) as out:
        result = body(out)
    return out, result


def test_all_phases_on_a_toy_context(tmp_path):
    protocol = toy_protocol()
    ctx = toy_context()
    holder = {}

    def bf16(out):
        record, tensors, rankings = rn.bf16_phase(ctx, protocol, out, modalities=["vision", "language", "state"],
                                                  store_nodes={"language", "state"})
        out.write_tensors("maps.pt", tensors)
        out.write_json("rankings.json", rankings)
        out.write_json("record.json", record)
        holder.update(record=record, tensors=tensors, rankings=rankings)

    run_phase(tmp_path, "bf16", bf16)
    record = holder["record"]
    for modality in ("vision", "language", "state"):
        for target in ("Q", "N"):
            nested = record["modalities"][modality]["targets"][target]["nested"]
            assert nested["defined"] and len(nested["node_records"]) == 9
            assert set(nested["residuals"]) >= {"T8", "E8", "M4", "S8", "RT8", "RM4"}
            assert nested["versus_comparator"]["T8"]["relative_l1"] == 0
            assert "A0m4" in nested["versus_comparator"]
        assert {"IG_Q_A1T8", "IG_N_A1E4", "PG_Q_A1T8", "IG_Q_A0m4", "input_difference"} <= set(holder["rankings"][modality])
    assert "vision/Q/A1T8" in holder["tensors"] and "vision/Q/A1T64" not in holder["tensors"]
    assert "vision/Q/A1M4/group_scores" in holder["tensors"] and "language/N/A1RM4" in holder["tensors"]
    assert holder["tensors"]["vision/Q/A0m4"].dtype == torch.bfloat16
    nodes = torch.load(tmp_path / "P/e0000-c000/bf16/nodes-language-Q.pt", weights_only=True)
    assert nodes["node_gradients"].shape[0] == 9
    assert not (tmp_path / "P/e0000-c000/bf16/nodes-vision-Q.pt").exists()

    ctx32 = toy_context(dtype=torch.float32)
    def fp32(out):
        record, tensors, rankings = rn.fp32_phase(ctx32, protocol, out, modalities=["vision", "state"],
                                                  adaptive=True, store_nodes=set())
        out.write_tensors("maps.pt", tensors)
        out.write_json("rankings.json", rankings)
        out.write_json("record.json", record)
        holder.update(fp32=record, fp32_rankings=rankings, fp32_tensors=tensors)

    run_phase(tmp_path, "fp32", fp32)
    target = holder["fp32"]["modalities"]["vision"]["targets"]["Q"]
    assert set(target["gauss_legendre"]) == {"GL3", "GL4"} and "adaptive" in target
    assert target["reference"] in ("A4", "A2T8")
    assert set(target["qualification"]) == {"A4|A3GL4", "A4|A2T8", "A3GL4|A2T8"}
    assert "vision/Q/A4" in holder["fp32_tensors"] and "vision/Q/A2T64" not in holder["fp32_tensors"]
    assert holder["fp32_rankings"]["vision"]["IG_Q_A4"]["status"] == "defined"

    def scoring(out):
        phase_rankings = {"bf16": json.loads(json.dumps(holder["rankings"])),
                          "fp32": json.loads(json.dumps(holder["fp32_rankings"]))}
        record = rn.scoring_phase(ctx, protocol, phase_rankings, "ctx", modalities=["vision", "language", "state"])
        out.write_json("record.json", record)
        holder["scoring"] = record

    run_phase(tmp_path, "scoring", scoring)
    vision = holder["scoring"]["modalities"]["vision"]
    assert {"fp32:IG_Q_A4", "IG_Q_A1T8", "random_0001", "fixed_centre_out"} <= set(vision["rankings"])
    assert vision["count_rule"] == "max1_round"
    state = holder["scoring"]["modalities"]["state"]
    assert "random_exact" in state["rankings"] and state["selected_counts"] == [0, 1, 1, 1, 2, 2, 4, 6, 8]
    curve = vision["rankings"]["IG_Q_A1T8"]["curves"]["deletion"]["responses"]["Q"]
    assert curve["status"] == "defined" and curve["values"][0] == 0.0

    directory, completion = rn.load_phase(tmp_path, "P", "e0000-c000", "scoring",
                                          dict(protocol_sha256="p", bank_sha256="b", context_id="ctx"))
    assert completion["files_sha256"]["record.json"] == file_hash(directory / "record.json")
    with pytest.raises(ValueError):
        rn.load_phase(tmp_path, "P", "e0000-c000", "scoring", dict(protocol_sha256="other", bank_sha256="b",
                                                                   context_id="ctx"))
    (directory / "record.json").write_text("{}")
    with pytest.raises(ValueError):
        rn.load_phase(tmp_path, "P", "e0000-c000", "scoring", dict(protocol_sha256="p", bank_sha256="b",
                                                                   context_id="ctx"))


def test_failed_phase_is_kept_as_an_attempt_and_never_published(tmp_path):
    binding = dict(protocol_sha256="p", bank_sha256="b", context_id="ctx")
    with pytest.raises(RuntimeError):
        with rn.PhaseOutput(tmp_path, "P", "e0000-c000", "bf16", binding) as out:
            out.write_json("record.json", {"partial": True})
            raise RuntimeError("worker died")
    assert not (tmp_path / "P/e0000-c000/bf16").exists()
    attempts = list((tmp_path / "P/e0000-c000/attempts").iterdir())
    assert len(attempts) == 1
    failure = json.loads((attempts[0] / "failure.json").read_text())
    assert failure["status"] == "failed" and failure["error_type"] == "RuntimeError"
    with rn.PhaseOutput(tmp_path, "P", "e0000-c000", "bf16", binding) as out:
        out.write_json("record.json", {"value": float("nan")})
    assert json.loads((tmp_path / "P/e0000-c000/bf16/record.json").read_text()) == {"value": "NaN"}
    with pytest.raises(FileExistsError):
        with rn.PhaseOutput(tmp_path, "P", "e0000-c000", "bf16", binding):
            pass


def test_verify_phase_gates_pass_on_the_toy_context(capsys):
    checks = rn.verify_phase(toy_context(), toy_protocol())
    capsys.readouterr()
    for item in checks["targets"].values():
        assert item["repeats_bit_identical"]
        assert item["pinned_kernel"]["trapezoid_bit_identical"]
        assert item["pinned_kernel"]["outside_population_nonzero"] == 0
        assert item["archived_kernel"]["bit_identical"]
        assert item["archived_kernel"]["printed"].startswith("completeness:")


def test_settings_bundles_cannot_be_mixed():
    protocol = toy_protocol()
    assert rn.phase_settings(protocol, "P", ["bf16", "fp32", "scoring"]) == STRICT
    assert rn.phase_settings(protocol, "M", ["bf16", "scoring"]) == DEFAULT
    with pytest.raises(ValueError):
        rn.phase_settings(protocol, "M", ["bf16", "fp32"])


def test_subset_rules_and_partition():
    bank = dict(contexts=[dict(episode=e, policy_call_idx=c, status="available", context_id=f"{e}-{c}")
                          for e in range(3) for c in (0, 7)] + [dict(episode=3, policy_call_idx=1,
                                                                     status="call_not_reached", context_id=None)])
    assert rn.resolve_members("first_episodes:2", bank) == {"0-0", "0-7", "1-0", "1-7"}
    assert rn.resolve_members("contexts:e0002-c007", bank) == {"2-7"}
    assert rn.resolve_members("first_contexts:3", bank) == {"0-0", "0-7", "1-0"}
    assert rn.resolve_members("none", bank) == set() and len(rn.resolve_members("all", bank)) == 6
    shards = [rn.partition(bank["contexts"], w, 2) for w in range(2)]
    assert sorted(e["context_id"] for shard in shards for e in shard) == sorted(f"{e}-{c}" for e in range(3) for c in (0, 7))
    with pytest.raises(ValueError):
        rn.resolve_members("random:3", bank)


def test_jsonable_and_cli_validation():
    assert rn.jsonable({"a": [float("inf"), -float("inf"), 1.5], "b": torch.tensor(2.0)}) == \
        {"a": ["Infinity", "-Infinity", 1.5], "b": 2.0}
    with pytest.raises(TypeError):
        rn.jsonable(torch.zeros(2))
    args = rn.parse_args(["--protocol", "p", "--bank", "P", "--phases", "bf16,scoring", "--out", "o"])
    assert args.phases == ["bf16", "scoring"]
    for phases in ("scoring,bf16", "bf16,bf16", "other"):
        with pytest.raises(ValueError):
            rn.parse_args(["--protocol", "p", "--bank", "P", "--phases", phases, "--out", "o"])


def test_worker_runs_every_phase_once_and_resumes_by_skipping(monkeypatch, tmp_path, capsys):
    protocol_path, protocol = stub_replay_layer(monkeypatch, tmp_path)
    out = tmp_path / "out"
    assert rn.run_worker(worker_args(protocol_path, out), protocol, dict(settings=STRICT)) == 0
    root = out / "P"
    assert sorted(p.name for p in (root / "e0000-c003").iterdir() if p.name != "attempts") == ["bf16", "fp32", "scoring", "verify"]
    assert sorted(p.name for p in (root / "e0001-c003").iterdir() if p.name != "attempts") == ["bf16", "scoring"]
    assert (root / "e0000-c003/bf16/source-cache.pt").exists() and not (root / "e0001-c003/bf16/source-cache.pt").exists()
    scoring = json.loads((root / "e0000-c003/scoring/record.json").read_text())
    assert "fp32:IG_Q_A4" in scoring["modalities"]["vision"]["rankings"]
    assert scoring["context"]["source_replay_verified"] is True
    fp32 = json.loads((root / "e0000-c003/fp32/record.json").read_text())
    assert fp32["status"] == "evaluated" and fp32["context"]["probe_identity"] == {"probe": "toy"}
    assert (root / "e0000-c003/bf16/nodes-state-Q.pt").exists() and not (root / "e0000-c003/bf16/nodes-language-Q.pt").exists()
    logs = list((root / "workers").glob("*.jsonl"))
    events = [json.loads(line)["event"] for line in logs[0].read_text().splitlines()]
    assert events[0] == "worker_start" and events[-1] == "worker_end" and events.count("phase_complete") == 6
    before = {p: p.stat().st_mtime_ns for p in root.rglob("completion.json")}
    assert rn.run_worker(worker_args(protocol_path, out), protocol, dict(settings=STRICT)) == 0
    assert {p: p.stat().st_mtime_ns for p in root.rglob("completion.json")} == before
    capsys.readouterr()


def test_worker_stops_on_a_replay_mismatch_and_keeps_the_attempt(monkeypatch, tmp_path, capsys):
    protocol_path, protocol = stub_replay_layer(monkeypatch, tmp_path, fail_on="ctx1")
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="ref_action"):
        rn.run_worker(worker_args(protocol_path, out, "bf16,fp32,scoring"), protocol, dict(settings=STRICT))
    root = out / "P"
    assert (root / "e0000-c003/scoring/completion.json").exists()
    assert not (root / "e0001-c003/bf16").exists()
    failure = json.loads(next((root / "e0001-c003/attempts").glob("bf16-*/failure.json")).read_text())
    assert failure["error_type"] == "ValueError"
    events = [json.loads(line) for line in next((root / "workers").glob("*.jsonl")).read_text().splitlines()]
    assert events[-1]["event"] == "phase_failed" and events[-1]["context"] == "e0001-c003"
    capsys.readouterr()


def test_worker_refuses_a_changed_bank_or_scoring_before_maps(monkeypatch, tmp_path, capsys):
    protocol_path, protocol = stub_replay_layer(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="bf16 is not complete"):
        rn.run_worker(worker_args(protocol_path, tmp_path / "out", "scoring"), protocol, dict(settings=STRICT))
    with pytest.raises(ValueError, match="fp32 is not complete"):
        rn.run_worker(worker_args(protocol_path, tmp_path / "out1", "bf16,scoring"), protocol, dict(settings=STRICT))
    Path(protocol["banks"]["P"]["bank"]).write_text("{}")
    with pytest.raises(ValueError, match="Bank file differs"):
        rn.run_worker(worker_args(protocol_path, tmp_path / "out2", "bf16"), protocol, dict(settings=STRICT))
    capsys.readouterr()


def test_bank_overrides_touch_only_budgets_and_storage():
    protocol = toy_protocol()
    protocol["banks"]["M"]["numerics_override"] = dict(historical_budgets=[64, 128], store_rules=dict(vision="all"))
    effective = rn.bank_protocol(protocol, "M")
    assert effective["numerics"]["historical_budgets"] == [64, 128]
    assert effective["numerics"]["levels"] == protocol["numerics"]["levels"]
    assert protocol["numerics"]["historical_budgets"] == [4]
    assert rn.bank_protocol(protocol, "P")["numerics"] == protocol["numerics"]
    protocol["banks"]["M"]["numerics_override"] = dict(levels=4)
    with pytest.raises(ValueError, match="fixed numerics"):
        rn.bank_protocol(protocol, "M")
