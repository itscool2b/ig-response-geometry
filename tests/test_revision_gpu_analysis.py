import csv
import itertools
import json
import math

import numpy as np
import pytest

import revision_numerics as rn
from analysis.revision_gpu.v1 import methods, numerics
from tests.revision_toy import STRICT, stub_replay_layer, worker_args

VISION_METHODS = ["IG_Q_A1T8", "IG_N_A1T8", "PG_Q_A1T8", "PG_N_A1T8", "input_difference", "IG_Q_A0m4", "IG_N_A0m4"]
VISION_FIXED = ["fixed_raster", "fixed_reverse_raster", "fixed_centre_out"]


@pytest.fixture
def toy_run(monkeypatch, tmp_path, capsys):
    protocol_path, protocol = stub_replay_layer(monkeypatch, tmp_path, contexts=3)
    rn.run_worker(worker_args(protocol_path, tmp_path / "out", "bf16,fp32,scoring"), protocol, dict(settings=STRICT))
    capsys.readouterr()
    loaded = methods.load_bank(tmp_path / "out", protocol_path, tmp_path / "bank.json", "P")
    return protocol_path, loaded


def toy_plan():
    family = [
        dict(id="P1", contrast="method_minus_random", method="IG_Q_A1T8", response="Q"),
        dict(id="P5", contrast="method_minus_method", method="IG_Q_A1T8", control="IG_N_A1T8", response="L2"),
        dict(id="P7", contrast="matched_calibrated", method="IG_Q_A1T8", method_response="Q", control="IG_N_A1T8",
             control_response="L2")]
    for item in family:
        item.update(modality="vision", direction="deletion", metric="normalized_auc")
    return dict(methods_study=dict(primary_family=family, family_alpha=0.05,
                                   bootstrap=dict(draws=2000, seed=11, interval="percentile", resampling="episodes")))


def test_loading_authenticates_every_phase(toy_run, tmp_path):
    _, (protocol, cfg, bank, contexts) = toy_run
    assert [("fp32" in item) for item in contexts] == [True, False, False]
    assert all("scoring" in item and "bf16" in item for item in contexts)
    record = contexts[1]["scoring_dir"] / "record.json"
    record.write_text(record.read_text().replace('"Q"', '"q"', 1))
    with pytest.raises(ValueError, match="hash differs"):
        methods.load_bank(tmp_path / "out", tmp_path / "protocol.json", tmp_path / "bank.json", "P")


def test_exact_state_control_matches_the_mean_over_all_orders(toy_run):
    _, (_, _, _, contexts) = toy_run
    result = contexts[0]["scoring"]["modalities"]["state"]
    for direction in ("deletion", "insertion"):
        for response in ("Q", "L2"):
            values = methods.exact_order_values(result, direction, response, "normalized_auc")
            assert len(values) == math.factorial(8)
            exact = methods.value_of(result["rankings"], "random_exact", direction, response, "normalized_auc")
            assert math.fsum(values) / len(values) == pytest.approx(exact, rel=1e-12, abs=1e-12)
            mean, sd, kind = methods.random_reference(result, direction, response, "normalized_auc")
            assert kind == "exact_all_orders" and mean == exact and sd == pytest.approx(float(np.std(values)))


def test_advantage_signs_and_seeded_reference(toy_run):
    _, (_, _, _, contexts) = toy_run
    rows = methods.context_table(contexts, "vision", VISION_METHODS + VISION_FIXED)
    assert len(rows) == 3 * 2 * 2 * (len(VISION_METHODS) + len(VISION_FIXED))
    for row in rows:
        if row["direction"] == "deletion":
            assert row["advantage"] == pytest.approx(row["random_mean"] - row["value"])
        else:
            assert row["advantage"] == pytest.approx(row["value"] - row["random_mean"])
        assert row["random_kind"] == "seeded_2" and row["z"] == pytest.approx(row["advantage"] / row["random_sd"])


def test_joint_bootstrap_is_seeded_and_shares_episodes():
    series = {"a": {"e1": 1.0, "e2": 2.0, "e3": 3.0}, "b": {"e1": 0.0, "e2": 0.5, "e3": 1.0, "e4": 9.0}}
    one, _ = methods.joint_summary(series, ["e1", "e2", "e3", "e4"], draws=500, seed=3, unique_alpha=0.05 / 4)
    two, _ = methods.joint_summary(series, ["e1", "e2", "e3", "e4"], draws=500, seed=3, unique_alpha=0.05 / 4)
    assert one == two and one["episodes"] == 3 and one["best"] == "a"
    assert one["probability_best"]["a"] == 1.0
    assert one["best_minus_second"]["mean"] == pytest.approx(1.5)
    # Plan: method intervals are descriptive 95%; only best minus second is Bonferroni over the four cells.
    assert one["interval"]["a"]["alpha"] == 0.05 and one["best_minus_second"]["descriptive"]["alpha"] == 0.05
    assert one["best_minus_second"]["adjusted"]["alpha"] == 0.05 / 4 and one["unique"] in (True, False)
    assert set(one["best_minus_second"]["adjusted"]["sensitivity"]) == {"bca", "t"}


def test_exact_ties_split_credit_and_leave_the_preferred_method_undetermined():
    series = {"a": {"e1": 1.0, "e2": 2.0}, "b": {"e1": 1.0, "e2": 2.0}}
    summary, _ = methods.joint_summary(series, ["e1", "e2"], draws=200, seed=1, unique_alpha=0.0125)
    assert summary["probability_best"] == {"a": 0.5, "b": 0.5}
    assert summary["best"] is None and summary["tie"] == ["a", "b"] and summary["unique"] is None
    assert methods._differs(None, "a") is None and methods._differs("a", "a") is False


def test_extreme_episodes_count_calls_not_episode_means():
    rows = [dict(episode_id="e1", value=-3.0), dict(episode_id="e1", value=2.5), dict(episode_id="e2", value=0.1)]
    assert methods.extreme_episodes(rows, "value", ["e1", "e2"]) == 1


def test_primary_contrasts_and_decisions_run_on_scoring_records(toy_run):
    _, (_, _, _, contexts) = toy_run
    plan = toy_plan()
    contrasts = methods.primary_contrasts(contexts, plan)
    assert [c["id"] for c in contrasts] == ["P1", "P5", "P7"]
    for contrast in contrasts:
        assert contrast["episodes"] == 3 and contrast["alpha"] == pytest.approx(0.05 / 3)
        assert contrast["interval"][0] <= contrast["estimate"] <= contrast["interval"][1]
        assert set(contrast["sensitivity"]) == {"bca", "t"} and contrast["excluded_episodes"] == {}
    rows = methods.context_table(contexts, "vision", ["IG_Q_A1T8", "IG_N_A1T8"])
    by = {(r["episode_id"], r["ranking"], r["response"]): r["advantage"] for r in rows if r["direction"] == "deletion"}
    matched = [by[(e, "IG_Q_A1T8", "Q")] - by[(e, "IG_N_A1T8", "L2")] for e in ("ep0", "ep1", "ep2")]
    assert contrasts[2]["estimate"] == pytest.approx(sum(matched) / 3)
    result = methods.decisions(contexts, "vision", VISION_METHODS, VISION_FIXED, plan)
    assert set(result["cells"]) == {"Q/deletion", "Q/insertion", "L2/deletion", "L2/insertion",
                                    "population/deletion", "population/insertion"}
    cell = result["cells"]["Q/deletion"]
    assert cell["advantage"]["best"] in VISION_METHODS and set(cell["bars"]) == {"0.4", "0.5"}
    assert isinstance(result["changes"]["deletion"]["across_responses"], bool)
    assert set(result["changes"]["deletion"]["through_population"]) == {"Q", "L2"}
    replication = methods.primary_contrasts(contexts, plan, alpha=0.05)
    assert all(c["alpha"] == 0.05 for c in replication)


def test_primary_contrast_reproduces_paired_episode_summary(toy_run):
    from paired_comparison import paired_episode_summary
    _, (_, _, _, contexts) = toy_run
    plan = toy_plan()
    ours = methods.primary_contrasts(contexts, plan)[0]
    records = [dict(episode_id=c["entry"]["episode_id"], policy_call_idx=c["entry"]["policy_call_idx"], modality="vision",
                    status="evaluated", source_context_id=c["entry"]["context_id"],
                    results=c["scoring"]["modalities"]["vision"]) for c in contexts]
    boot = plan["methods_study"]["bootstrap"]
    theirs = paired_episode_summary(records, method="IG_Q_A1T8", control="random", response="Q", direction="deletion",
                                    metric="normalized_auc", alpha=ours["alpha"], draws=boot["draws"], seed=boot["seed"])
    assert ours["estimate"] == pytest.approx(theirs["estimate"], rel=1e-12)
    assert ours["interval"] == pytest.approx(theirs["confidence_interval"], rel=1e-12)


def test_complete_case_needs_both_responses_and_every_planned_call(toy_run):
    import copy
    _, (_, _, _, contexts) = toy_run
    plan = toy_plan()
    changed = copy.deepcopy(contexts)
    outcome = changed[0]["scoring"]["modalities"]["vision"]["rankings"]["IG_N_A1T8"]["curves"]["deletion"]["responses"]["L2"]
    outcome.update(normalized_auc=None, status="nearzero_endpoint_gap")
    unreached = dict(entry=dict(changed[1]["entry"], policy_call_idx=12, context_id="unreached", status="call_not_reached"))
    changed.append(unreached)
    result = methods.decisions(changed, "vision", VISION_METHODS, VISION_FIXED, plan)
    population = result["cells"]["population/deletion"]
    assert population["complete_case_episodes"] == 1
    assert population["excluded_episodes"]["ep0"].startswith("call 3: IG_N_A1T8 L2 nearzero")
    assert population["excluded_episodes"]["ep1"] == "call 12: call_not_reached"
    # One population for both responses of a direction; route 4 keeps every available call.
    assert result["cells"]["Q/deletion"]["advantage"]["episodes"] == result["cells"]["L2/deletion"]["advantage"]["episodes"]
    assert result["cells"]["Q/deletion"]["all_available"]["episodes"]["IG_Q_A1T8"] == 3
    assert result["changes"]["deletion"]["across_responses"] is None  # one complete episode: nothing preferred
    p1, p5 = methods.primary_contrasts(changed, plan)[:2]
    # P1 involves IG_Q_A1T8 and the random control only, so ep0 stays; P5 also needs IG_N_A1T8.
    assert p1["episodes"] == 2 and set(p1["excluded_episodes"]) == {"ep1"}
    assert p5["episodes"] == 1 and set(p5["excluded_episodes"]) == {"ep0", "ep1"}


def test_round_rule_area_matches_the_stored_area_where_the_rules_agree(toy_run):
    from faithfulness import auc_normalized
    _, (_, _, _, contexts) = toy_run
    for modality in ("vision", "language", "state"):
        result = contexts[0]["scoring"]["modalities"][modality]
        n = len(result["eligible_indices"])
        rounded = rn.count_schedule(n, result["nominal_grid_percent"], "round")
        outcome = result["rankings"]["IG_Q_A1T8"]["curves"]["deletion"]["responses"]["Q"]
        area = methods.round_rule_area(result, outcome["values"], "Q")
        # On the realized-fraction axis duplicate points add nothing, so the areas agree whenever
        # the round schedule reaches every count max1_round uses.
        if set(rounded) == set(result["selected_counts"]):
            assert area == pytest.approx(outcome["normalized_auc"], rel=1e-12, abs=1e-15)
        curve = [v if c == k else outcome["values"][0] for c, k, v in zip(rounded, result["selected_counts"], outcome["values"])]
        table = result["response_table"]
        expected = auc_normalized([100 * c / n for c in rounded], curve, table[result["input_response_id"]]["Q"],
                                  table[result["baseline_response_id"]]["Q"], denominator_min=result["denominator_min"]["Q"])
        assert area == expected
    # Thirty groups: round never gives one group (1.5 rounds to 2), so the max1_round point at 1/30 is extra.
    assert 1 not in rn.count_schedule(30, [0, 1, 5, 10, 20, 30, 50, 75, 100], "round")
    assert 1 in rn.count_schedule(8, [0, 1, 5, 10, 20, 30, 50, 75, 100], "round")


def test_response_shift_rows_survive_a_failed_baseline_response(toy_run):
    import copy
    _, (_, _, _, contexts) = toy_run
    changed = copy.deepcopy(contexts[:1])
    result = changed[0]["scoring"]["modalities"]["vision"]
    result["response_table"][result["baseline_response_id"]]["Q"] = None
    rows = methods.response_shift_rows(changed, "vision", ["IG_Q_A1T8"])
    assert rows and all(row["overshoot"] is None for row in rows)


def test_numerics_tables_cover_every_arm(toy_run):
    protocol_path, (protocol, _, _, contexts) = toy_run
    rows = numerics.residual_rows("P", contexts[0])
    arms = {row["arm"] for row in rows}
    assert arms == {"A0", "A1", "A2", "A3", "A4"}
    assert {(r["rule"], r["budget"]) for r in rows if r["arm"] == "A1"} >= {("T", 8), ("E", 8), ("M", 4), ("S", 8)}
    comparisons = numerics.comparison_rows("P", contexts[0])
    assert {r["family"] for r in comparisons} == {"bf16_vs_A1_comparator"}
    bank_protocol = rn.bank_protocol(protocol, "P")
    references = numerics.reference_rows("P", contexts[0], bank_protocol)
    families = {r["family"] for r in references}
    assert {"fp32_vs_A2T8", "bf16_vs_A2T8"} <= families
    assert {r["candidate"] for r in references if r["family"] == "bf16_vs_A2T8" and r["modality"] == "vision"} == {"A0m4", "A1T8", "A1E4"}
    assert not [r for r in references if r["family"].startswith("fp32") and r["candidate"] == r["reference"]]
    assert all(r["reference"] in ("A2T8", "A4") for r in references)
    a4 = {(m, t) for m, rec in contexts[0]["fp32"]["modalities"].items() for t, cell in rec["targets"].items()
          if numerics.converged_a4(cell)}
    assert {(r["modality"], r["target"]) for r in references if r["reference"] == "A4"} == a4
    assert numerics.reference_rows("P", contexts[1], bank_protocol) == []
    qualification = numerics.qualification_rows("P", contexts[0])
    for row in qualification:
        pairs = [v for k, v in row.items() if k.startswith("relative_l1[")]
        assert row["resolution_floor"] == max(pairs)
        assert row["qualified"] == (row["a4_converged"] and max(pairs) <= 1e-3)
        assert row["a4_error_space"] == "path_gradient_integral_l1"
    assert numerics.nonfinite_rows("P", contexts[0]) == []
    counts = numerics.node_count_rows("P", contexts[0])
    assert {r["arm"] for r in counts} >= {"A1", "A0m4", "A2", "A3GL3", "A4"}
    summary = numerics.nonfinite_summary(counts, [])
    assert summary and all(r["nodes"] > 0 and r["nodes_nonfinite_gradient"] == 0 for r in summary)
    changes = numerics.auc_change_rows("P", contexts[0], [("IG_Q_A0m4", "IG_Q_A1T8"), ("IG_Q_A1T8", "fp32:IG_Q_A4")])
    assert {r["reference"] for r in changes} == {"IG_Q_A1T8", "fp32:IG_Q_A4"}
    assert all(r["change"] == pytest.approx(r["candidate_auc"] - r["reference_auc"]) for r in changes)
    automatic = numerics.auc_change_rows("P", contexts[0], comparator="T8")
    references_used = {r["reference"] for r in automatic}
    assert {"IG_Q_A1T8", "fp32:IG_Q_A2T8"} <= references_used
    assert ("fp32:IG_Q_A4" in references_used) == numerics.converged_a4(contexts[0]["fp32"]["modalities"]["vision"]["targets"]["Q"])
    assert not [r for r in automatic if r["candidate"] == r["reference"]]


def test_analysis_driver_writes_deterministic_tables(toy_run, tmp_path):
    import subprocess, sys
    protocol_path, _ = toy_run
    inputs = dict(run_root=str(tmp_path / "out"), banks=[dict(id="P", protocol=str(protocol_path),
                                                             bank=str(tmp_path / "bank.json"), group="P")],
                  methods=VISION_METHODS, auc_pairs=[["IG_Q_A0m4", "IG_Q_A1T8"], ["IG_Q_A1T8", "fp32:IG_Q_A4"]])
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    plan = toy_plan()
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    outputs = []
    for run in ("a", "b"):
        out = tmp_path / f"analysis-{run}"
        subprocess.run([sys.executable, "-m", "analysis.revision_gpu.v1.analyze", "--inputs", str(tmp_path / "inputs.json"),
                        "--plan", str(tmp_path / "plan.json"), "--out", str(out)], check=True, capture_output=True)
        outputs.append(json.loads((out / "manifest.json").read_text())["outputs"])
    assert outputs[0] == outputs[1]
    assert {"numerics/residuals.csv", "numerics/auc_changes.csv", "numerics/references.csv", "numerics/qualification.csv",
            "numerics/nonfinite_summary.csv", "methods/P/contrasts.json", "methods/P/decisions.json",
            "methods/P/decisions_round_rule.json", "methods/P/decisions_nominal_axis.json", "methods/P/rms_units.csv",
            "methods/P/exclusions.csv", "shifts/P/summary_vision.csv"} <= set(outputs[0])
    contrasts = json.loads((tmp_path / "analysis-a/methods/P/contrasts.json").read_text())
    assert all(c["sensitivity"]["t"]["episodes"] == 3 for c in contrasts["normalized_auc"])
    assert all(c["simulated_coverage"] == "not simulated at 3 episodes" for c in contrasts["normalized_auc"])
    assert (tmp_path / "analysis-a/numerics/nonfinite.csv").read_text().startswith("bank,context,phase,")


def test_figures_render_from_the_analysis_tables(toy_run, tmp_path):
    import subprocess, sys
    protocol_path, _ = toy_run
    inputs = dict(run_root=str(tmp_path / "out"), banks=[dict(id="P", protocol=str(protocol_path),
                                                             bank=str(tmp_path / "bank.json"), group="P")],
                  methods=VISION_METHODS, auc_pairs=[["IG_Q_A0m4", "IG_Q_A1T8"]])
    (tmp_path / "inputs.json").write_text(json.dumps(inputs))
    (tmp_path / "plan.json").write_text(json.dumps(toy_plan()))
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(dict(event="episode_end", episode=e, success=s)) + "\n"
                                                    for e, s in ((0, True), (1, False))))
    subprocess.run([sys.executable, "-m", "analysis.revision_gpu.v1.analyze", "--inputs", str(tmp_path / "inputs.json"),
                    "--plan", str(tmp_path / "plan.json"), "--out", str(tmp_path / "analysis"),
                    "--metrics-copy", f"P={tmp_path / 'metrics.jsonl'}"], check=True, capture_output=True)
    with (tmp_path / "analysis/shifts/P/rows_vision.csv").open(newline="") as handle:
        assert {(row["episode"], row["success"]) for row in csv.DictReader(handle)} == {("0", "True"), ("1", "False"), ("2", "")}
    runs = []
    for name in ("figs-a", "figs-b"):
        subprocess.run([sys.executable, "-m", "analysis.revision_gpu.v1.figures", "--analysis", str(tmp_path / "analysis"),
                        "--out", str(tmp_path / name), "--min-budget", "1", "--ranking", "IG_Q_A1T8"],
                       check=True, capture_output=True)
        runs.append(json.loads((tmp_path / name / "figures.json").read_text())["files"])
    assert {"residual_by_budget_P.pdf", "residual_by_budget_P.png", "calibrated_advantage_P_vision.pdf",
            "response_shift_vision_insertion.png", "coordinate_error_by_budget_P.pdf"} <= set(runs[0])
    assert {k: v for k, v in runs[0].items() if k.endswith(".pdf")} == {k: v for k, v in runs[1].items() if k.endswith(".pdf")}


def test_node_check_rebuilds_every_stored_rule_from_the_saved_nodes(toy_run, tmp_path):
    from analysis.revision_gpu.v1 import node_check
    results = {}
    for phase, arm in node_check.PHASE_ARMS.items():
        for directory in sorted((tmp_path / "out").glob(f"*/e*-c*/{phase}")):
            if any(directory.glob("nodes-*-*.pt")):
                results[f"{directory.parts[-2]}/{phase}"] = node_check.check_phase(directory, arm)
    assert {key.split("/")[1] for key in results} == {"bf16", "fp32"}
    cells = [cell for cells in results.values() for cell in cells.values()]
    rules = [r for cell in cells for r in cell["rules"].values()]
    assert rules and all(r["fp32_replay"]["bit_identical"] for r in rules)
    assert max(r["float64"]["relative_l1"] for r in rules) < 1e-5
    # Every stored rule of the arm is rebuilt, Simpson, Romberg and the path gradient included.
    names = {name for cell in cells for name in cell["rules"]}
    assert {"S2", "RT8", "RM4", "PGT8"} <= names and not [c for c in cells if c["unchecked_stored_maps"]]
    identities = [v for cell in cells for v in cell["stored_identity_relative"].values()]
    assert identities and max(identities) < 1e-5


def test_population_counts_follow_the_complete_case_rule(toy_run, tmp_path):
    from analysis.revision_gpu.v1 import analyze
    protocol_path, _ = toy_run
    inputs = dict(run_root=str(tmp_path / "out"), banks=[dict(id="P", protocol=str(protocol_path),
                                                             bank=str(tmp_path / "bank.json"), group="P")])
    loaded = analyze.load_all(inputs)
    counts = analyze.population_counts(loaded, toy_plan(), "P", VISION_METHODS)
    assert counts["P/contrast/P1"] == 3 and counts["P/decisions/vision/deletion"] == 3
    assert {key.split("/")[2] for key in counts if "/decisions/" in key} == {"vision", "language", "state"}


def test_review_two_tables(toy_run, tmp_path, capsys):
    import copy
    from analysis.revision_gpu.v1 import analyze
    protocol_path, (protocol, _, _, contexts) = toy_run
    item = contexts[0]
    bank_protocol = rn.bank_protocol(protocol, "P")
    # A0 residual from the fp32 sum beside the input-dtype one
    for row in numerics.residual_rows("P", item):
        if row["arm"] == "A0":
            assert row["absolute_residual_fp32_sum"] == pytest.approx(abs(row["ig_sum_fp32"] - row["expected_gap"]))
    # a failed forward is its own row, without sign counts, and its own count
    broken = copy.deepcopy(item)
    records = broken["bf16"]["modalities"]["vision"]["targets"]["Q"]["nested"]["node_records"]
    records[1] = dict(node=1, alpha=records[1].get("alpha"), status="forward_nonfinite", stage="sampler",
                      inner_step=3, forward_nonfinite_count=5)
    rows = numerics.nonfinite_rows("P", broken)
    assert len(rows) == 1 and rows[0]["forward_failure"] and not rows[0]["sign_recorded"]
    assert rows[0]["nan"] is None and rows[0]["inner_step"] == 3 and rows[0]["forward_nonfinite_count"] == 5
    summary = numerics.nonfinite_summary(numerics.node_count_rows("P", broken), rows)
    hit = [r for r in summary if r["arm"] == "A1" and r["modality"] == "vision" and r["target"] == "Q"]
    assert hit[0]["nodes_forward_failure"] == 1 and hit[0]["nodes_nonfinite_gradient"] == 0
    # qualification needs a converged A4
    stalled = copy.deepcopy(item)
    for record in stalled["fp32"]["modalities"].values():
        for cell in record["targets"].values():
            cell["adaptive"]["converged"] = False
    assert all(r["qualified"] in (False, None) and r["qualification_reason"] in ("a4_not_converged", "pair_undefined")
               for r in numerics.qualification_rows("P", stalled))
    # fp32 rules the worker did not store come from its own comparisons; stored ones agree with them
    references = numerics.reference_rows("P", item, bank_protocol)
    worker = [r for r in references if r["source"] == "worker"]
    assert worker and all(r["family"] == f"fp32_vs_{r['reference']}" for r in worker)
    stored = [r for r in references if r["source"] == "recomputed" and r["worker_relative_l1"] is not None]
    assert stored and all(r["worker_relative_l1"] == pytest.approx(r["relative_l1"], rel=1e-9, abs=1e-15) for r in stored)
    vision = item["fp32"]["modalities"]["vision"]["targets"]["Q"]
    unstored = {name for name in vision["versus_reference"] if name != vision["reference"]} - \
        {r["candidate"] for r in references if r["source"] == "recomputed" and r["family"] == f"fp32_vs_{vision['reference']}"
         and r["modality"] == "vision" and r["target"] == "Q"}
    assert unstored == {r["candidate"] for r in worker if r["modality"] == "vision" and r["target"] == "Q"
                        and r["reference"] == vision["reference"]}
    # rank agreement on the scored orders
    ranks = numerics.ranking_rows("P", item, "T8")
    assert ranks and all(-1 <= r["order_spearman"] <= 1 for r in ranks if r["order_spearman"] is not None)
    result = item["scoring"]["modalities"]["vision"]
    counts = {c for c in result["selected_counts"] if c not in (0, len(result["eligible_indices"]))}
    for row in ranks:
        if row["deleted_set_jaccard"] is not None and row["modality"] == "vision":
            assert {c for c, _ in json.loads(row["deleted_set_jaccard"])} == counts
    flipped = copy.deepcopy(item)
    rankings = flipped["scoring"]["modalities"]["vision"]["rankings"]
    rankings["IG_Q_A0m4"]["order"] = list(reversed(rankings["IG_Q_A1T8"]["order"]))
    row = next(r for r in numerics.ranking_rows("P", flipped, "T8")
               if r["modality"] == "vision" and r["candidate"] == "IG_Q_A0m4" and r["reference"] == "IG_Q_A1T8")
    assert row["order_spearman"] == pytest.approx(-1.0)
    same = next(r for r in ranks if r["modality"] == "vision" and r["candidate"] == "IG_Q_A1T8"
                and r["reference"] == "fp32:IG_Q_A2T8")
    assert same["top5pct_size"] >= 1
    # verify gates come from the authenticated verify phase
    rn.run_worker(worker_args(protocol_path, tmp_path / "out", "verify"), protocol, dict(settings=STRICT))
    capsys.readouterr()
    inputs = dict(run_root=str(tmp_path / "out"), banks=[dict(id="P", protocol=str(protocol_path),
                                                             bank=str(tmp_path / "bank.json"), group="P")])
    gates = analyze.verify_gate_rows(inputs["run_root"], analyze.load_all(inputs))
    assert {(g["context"], g["target"]) for g in gates} == {("ctx0", "Q"), ("ctx0", "N")}
    assert all(g["status"] == "loaded" and g["all_gates_pass"] for g in gates)


def test_v6_check_counts_every_context_and_skipped_budget(tmp_path):
    import torch
    from analysis.revision_gpu.v1 import v6_check
    full = torch.arange(1 * 4 * 729 * 2, dtype=torch.float32).reshape(1, 4 * 729, 2)
    inner = tmp_path / "v6/results/pickcube-170m-e1-c2/ep1-call2"
    (inner / "vision-Q").mkdir(parents=True)
    (inner / "vision-Q/report.json").write_text("{}")
    for budget in (8, 16):
        torch.save(dict(attribution=full), inner / f"vision-Q/m{budget}-r0.pt")
    r1 = tmp_path / "r1/M8-pickcube-170m/e0001-c002/fp32"
    r1.mkdir(parents=True)
    torch.save({"vision/Q/A2T8": full[:, 3 * 729:4 * 729].clone()}, r1 / "maps.pt")
    (tmp_path / "v6/results/pickcube-170m-e3-c4/ep3-call4").mkdir(parents=True)
    out = tmp_path / "v6.json"
    import sys
    argv = sys.argv
    sys.argv = ["v6_check", "--v6", str(tmp_path / "v6"), "--r1", str(tmp_path / "r1"), "--out", str(out)]
    try:
        v6_check.main()
    finally:
        sys.argv = argv
    summary = json.loads(out.read_text())["summary"]
    assert summary["contexts"] == 2 and summary["by_status"] == {"compared": 1, "missing": 1}
    assert summary["maps_compared"] == 1 and summary["bit_identical"] == 1
    assert summary["budgets_compared"] == [8] and summary["budgets_skipped"] == [16]
    assert summary["maps_skipped_not_stored"] == 1 and summary["compared_contexts_with_no_map"] == []


def test_advantage_figure_skips_cells_and_figures_without_an_estimate(tmp_path):
    from analysis.revision_gpu.v1 import figures
    figures.style()
    empty = {"vision": {"cells": {key: {"with_fixed_orders": {"episodes": 1, "names": ["a"], "status": "no estimate"}}
                                  for key in ("Q/deletion", "Q/insertion", "L2/deletion", "L2/insertion")}}}
    assert figures.advantage_figure(empty, tmp_path, "none") is None and not list(tmp_path.iterdir())
    partial = json.loads(json.dumps(empty))
    partial["vision"]["cells"]["Q/deletion"]["with_fixed_orders"] = dict(
        episodes=2, names=["a", "b"], mean={"a": 0.1, "b": -0.2},
        interval={"a": {"interval": [0.0, 0.2]}, "b": {"interval": [-0.3, -0.1]}})
    written = figures.advantage_figure(partial, tmp_path, "partial")
    assert sorted(written) == ["partial.pdf", "partial.png"]


def test_budget_figure_panels_draw_every_point_over_the_same_contexts():
    from analysis.revision_gpu.v1 import figures
    rows = [dict(modality="vision", target="Q", arm="A1", rule="T", budget=b, context=c, value=1.0)
            for c in ("c1", "c2", "c3") for b in (16, 64)]
    rows += [dict(modality="vision", target="Q", arm="A0", rule="E", budget=64, context=c, value=2.0) for c in ("c1", "c2", "c3")]
    # An fp32 rule compared on the pod only where its reference matched: c3 lacks it, so c3 leaves the panel.
    rows += [dict(modality="vision", target="Q", arm="A2", rule="M", budget=32, context=c, value=3.0) for c in ("c1", "c2")]
    rows += [dict(modality="vision", target="Q", arm="A2", rule="T", budget=8, context="c3", value=4.0),
             dict(modality="language", target="Q", arm="A1", rule="T", budget=16, context="c3", value=float("nan"))]
    points, shared = figures.panel_population(rows, "vision", "Q", "value", 16)
    assert shared == ["c1", "c2"]
    assert set(points) == {("A1", "T", 16), ("A1", "T", 64), ("A0", None, 64), ("A2", "M", 32)}
    assert figures.panel_population(rows, "language", "Q", "value", 16)[1] == []
