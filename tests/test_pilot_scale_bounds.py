"""Distribution-free median bounds and complete prospective scale membership."""
from copy import deepcopy
from fractions import Fraction
import itertools
import math

import pytest

import pilot_scale_bounds as scale


def test_exact_rank_boundaries_are_not_interpolated_or_normal_approximations():
    result = scale.exact_median_bounds(range(1, 11), .025, .025)
    assert (result["lower_order_rank"], result["upper_order_rank"]) == (2, 9)
    assert (result["value"], result["lower"], result["upper"]) == (5.5, 2., 9.)
    assert result["lower_noncoverage_bound"] == 11/1024
    edge = scale.exact_median_bounds([1, 2, 3], .125, .125)
    assert (edge["lower"], edge["upper"]) == (1., 3.)
    below = scale.exact_median_bounds([1, 2, 3], math.nextafter(.125, 0), .125)
    assert below["lower"] == 0 and below["lower_order_rank"] == 0
    assert below["upper"] == 3


@pytest.mark.parametrize("p,medians", [(Fraction(1, 5), [0]), (Fraction(1, 2), [0, .5, 1]), (Fraction(4, 5), [1])])
def test_discrete_atoms_have_conservative_coverage_for_each_fixed_median(p, medians):
    n = 6
    for median in medians:
        lower_misses, upper_misses = Fraction(0), Fraction(0)
        for values in itertools.product((0, 1), repeat=n):
            result = scale.exact_median_bounds(values, .1, .1)
            probability = p**sum(values) * (1-p)**(n-sum(values))
            lower_misses += probability * (result["lower"] > median)
            upper_misses += probability * (result["upper"] is not None and result["upper"] < median)
        assert lower_misses <= Fraction.from_float(.1)
        assert upper_misses <= Fraction.from_float(.1)


def test_uninformative_and_zero_intervals_are_not_positive_resolution():
    small = scale.exact_median_bounds([3], .01, .01)
    assert small["value"] == 3 and small["lower"] == 0 and small["upper"] is None
    assert small["upper_unbounded"] and small["status"].startswith("unresolved")
    zeros = scale.exact_median_bounds([0] * 20, .025, .025)
    assert zeros["lower"] == zeros["upper"] == zeros["value"] == 0
    assert not zeros["upper_unbounded"] and zeros["status"].startswith("unresolved")
    large = scale.exact_median_bounds([1e308, 1e308], .1, .1)
    assert large["value"] == 1e308


@pytest.mark.parametrize("values", [[], [-1, 0], [float("nan")], [float("inf")], [True], [[1]]])
def test_invalid_observations_never_become_finite_scale(values):
    with pytest.raises(ValueError):
        scale.exact_median_bounds(values, .05, .05)


@pytest.mark.parametrize("lower,upper", [(0, .05), (-.1, .05), (.5, .5), (float("nan"), .05), (True, .05)])
def test_tail_probability_contract_is_explicit(lower, upper):
    with pytest.raises(ValueError):
        scale.exact_median_bounds([1, 2], lower, upper)


def fixture(stage="first"):
    design = dict(stage="variance_only_pilot", modalities=list(scale.MODALITIES),
        strata=[dict(id=s, reset_seeds=list(range(offset, offset+20))) for s,offset in (("a", 0), ("b", 100))],
        groups=[dict(id="fixed_group", weights={"a":.25, "b":.75})],
        pilot_stages=[dict(id="first", episodes_per_stratum={"a":12,"b":12}),
                      dict(id="second", episodes_per_stratum={"a":20,"b":20})])
    n = 12 if stage == "first" else 20
    bank = dict(schema_version=1, kind=scale.BANK_KIND, strata={})
    for spec in design["strata"]:
        episodes = []
        for number,seed in enumerate(spec["reset_seeds"][:n]):
            eid = f"{spec['id']}-episode-{number}"
            calls = []
            for call in range(1 + number % 3):
                cid = f"{eid}-call-{call}"
                calls.append(dict(policy_call_idx=call, context_id=cid, source_row_sha256=scale._hash(cid),
                    modalities={m:dict(status="finite", baseline_rms=float(1+number+2*call+i)*(1 if spec['id']=='a' else 3),
                        endpoint_identity_sha256=scale._hash([cid,m])) for i,m in enumerate(scale.MODALITIES)}))
            episodes.append(dict(episode_id=eid, reset_seed=seed, terminal_record_sha256=scale._hash([eid,"terminal"]),
                terminal_policy_calls=len(calls), executed_call_indices=list(range(len(calls))), calls=calls))
        bank["strata"][spec["id"]] = dict(episodes=episodes)
    return design, bank


def report(design, bank, stage="first"):
    return scale.simultaneous_scale_report(design, bank, stage, [.5, .5], .24)


def test_all_call_means_then_episode_medians_fixed_group_weights_and_exact_spending():
    design, bank = fixture()
    result = report(design, bank)
    first = bank["strata"]["a"]["episodes"]
    episode_means = [sum(c["modalities"]["vision"]["baseline_rms"] for c in e["calls"])/len(e["calls"]) for e in first]
    expected = (sorted(episode_means)[5] + sorted(episode_means)[6])/2
    assert result["strata"]["a"]["modalities"]["vision"]["value"] == expected
    assert result["scales"]["fixed_group:vision"]["value"] == 2.5*expected
    assert result["scales"]["fixed_group:vision"]["conservative_planning_halfwidth"] > 0
    assert result["strata"]["a"]["planned_calls"] == 24
    assert len(result["strata"]["a"]["membership"]) == 12
    alpha = result["spending"]["alpha_per_stratum_modality_tail"]
    assert Fraction.from_float(alpha)*24 <= Fraction.from_float(.24)
    for modality in scale.MODALITIES:
        intervals = [result["strata"][s]["modalities"][modality] for s in ("a","b")]
        group = result["scales"][f"fixed_group:{modality}"]
        assert group["lower"] == .25*intervals[0]["lower"] + .75*intervals[1]["lower"]
        assert group["upper"] == .25*intervals[0]["upper"] + .75*intervals[1]["upper"]


def test_unequal_episode_call_counts_do_not_turn_into_call_weighted_scale():
    design, bank = fixture()
    for sid in bank["strata"]:
        for i,episode in enumerate(bank["strata"][sid]["episodes"]):
            value = 100. if i % 3 == 0 else 0.
            for call in episode["calls"]:
                for endpoint in call["modalities"].values(): endpoint["baseline_rms"] = value
    result = report(design, bank)
    assert result["scales"]["fixed_group:vision"]["value"] == 0
    assert result["scales"]["fixed_group:vision"]["conservative_planning_halfwidth"] is None
    means = [e["episode_mean_baseline_rms"]["vision"] for e in result["strata"]["a"]["membership"]]
    assert means.count(100.) == 4 and means.count(0.) == 8


def test_explicit_failure_keeps_planned_population_and_only_affected_scale_unresolved():
    design, bank = fixture()
    endpoint = bank["strata"]["a"]["episodes"][0]["calls"][0]["modalities"]["vision"]
    endpoint.update(status="numerical_failure", baseline_rms=None, reason="nonfinite baseline action")
    result = report(design, bank)
    failed = result["scales"]["fixed_group:vision"]
    assert failed["status"] == "unresolved_endpoint_failure"
    assert failed["value"] is None and failed["conservative_planning_halfwidth"] is None
    assert not failed["confidence_bound_applicable"]
    assert result["scales"]["fixed_group:language"]["status"] == "positive_exact_scale"
    assert len(result["strata"]["a"]["membership"]) == 12
    assert result["strata"]["a"]["membership"][0]["planned_episode_weight"] == 1/12
    assert result["endpoint_status_counts"]["vision:numerical_failure"] == 1


@pytest.mark.parametrize("corruption", ["missing_episode","extra_episode","missing_call","truncated_call_list",
    "duplicate_call","duplicate_context","missing_modality","wrong_reset","failed_finite_value","nonfinite_finite_value"])
def test_missing_or_corrupt_membership_is_an_identity_error(corruption):
    design, bank = fixture()
    episodes = bank["strata"]["a"]["episodes"]
    call = episodes[1]["calls"][0]
    if corruption == "missing_episode": episodes.pop()
    if corruption == "extra_episode": episodes.append(deepcopy(episodes[0]))
    if corruption == "missing_call": episodes[1]["calls"].pop()
    if corruption == "truncated_call_list":
        episodes[1]["calls"].pop(); episodes[1]["executed_call_indices"].pop()
    if corruption == "duplicate_call": episodes[1]["calls"].append(deepcopy(call))
    if corruption == "duplicate_context": call["context_id"] = episodes[0]["calls"][0]["context_id"]
    if corruption == "missing_modality": del call["modalities"]["state"]
    if corruption == "wrong_reset": episodes[0]["reset_seed"] = 700
    if corruption == "failed_finite_value": call["modalities"]["vision"].update(status="failed", reason="broken")
    if corruption == "nonfinite_finite_value": call["modalities"]["vision"]["baseline_rms"] = float("nan")
    with pytest.raises(ValueError): report(design, bank)


def test_stage_prefix_and_probability_budgets_cannot_be_selected_after_seeing_data():
    design, bank = fixture("second")
    with pytest.raises(ValueError, match="exact planned stage prefix"): report(design, bank)
    second = report(design, bank, "second")
    assert second["look_index"] == 1 and second["strata"]["a"]["planned_episodes"] == 20
    for weights, alpha in (([1.,1.], .05), ([.5], .05), ([0.,1.], .05), ([.5,.5], 0), ([.5,.5], 1)):
        with pytest.raises(ValueError): scale.simultaneous_scale_report(design, bank, "second", weights, alpha)
    with pytest.raises(ValueError): report(design, bank, "undeclared")


def test_nonpositive_support_and_small_n_do_not_acquire_an_epsilon():
    design, bank = fixture()
    design["pilot_stages"][0]["episodes_per_stratum"] = {"a":1,"b":1}
    for value in bank["strata"].values(): value["episodes"] = value["episodes"][:1]
    result = report(design, bank)
    for group in result["scales"].values():
        assert group["value"] > 0 and group["lower"] == 0 and group["upper"] is None
        assert group["conservative_planning_halfwidth"] is None
        assert group["status"] == "unresolved_nonpositive_scale_lower_bound"


def zero_call_fixture():
    design, bank = fixture()
    bank["schema_version"] = 2
    episode = bank["strata"]["a"]["episodes"][0]
    episode.update(terminal_policy_calls=0, executed_call_indices=[], calls=[],
        collection_status="failed", collection_failure_reason="authenticated reset failed before first policy call")
    return design, bank


def test_explicit_zero_call_collection_failure_keeps_episode_and_blocks_every_modality():
    design, bank = zero_call_fixture()
    result = report(design, bank)
    assert result["endpoint_bank_schema_version"] == 2
    assert result["collection_status_counts"] == {"zero_call_failure": 1}
    assert result["strata"]["a"]["planned_calls"] == 23
    members = result["strata"]["a"]["membership"]
    assert len(members) == 12 and members[0]["planned_episode_weight"] == 1/12
    assert members[0]["contexts"] == []
    assert members[0]["terminal_record_sha256"] == bank["strata"]["a"]["episodes"][0]["terminal_record_sha256"]
    assert members[0]["episode_mean_baseline_rms"] == dict.fromkeys(scale.MODALITIES)
    for modality in scale.MODALITIES:
        cell = result["scales"][f"fixed_group:{modality}"]
        assert cell["status"] == "unresolved_collection_failure"
        assert cell["value"] is None and cell["conservative_planning_halfwidth"] is None
        assert not cell["confidence_bound_applicable"]
        assert result["strata"]["b"]["modalities"][modality]["confidence_bound_applicable"]
        assert sum(v for k,v in result["endpoint_status_counts"].items() if k.startswith(modality+":")) == 47


@pytest.mark.parametrize("corruption", ["old_schema", "no_status", "wrong_status", "no_reason", "empty_reason",
    "bad_terminal_hash", "extra_declared_call", "extra_call_row"])
def test_zero_call_failure_requires_versioned_explicit_authenticated_contract(corruption):
    design, bank = zero_call_fixture()
    episode = bank["strata"]["a"]["episodes"][0]
    if corruption == "old_schema": bank["schema_version"] = 1
    if corruption == "no_status": episode.pop("collection_status")
    if corruption == "wrong_status": episode["collection_status"] = "complete"
    if corruption == "no_reason": episode.pop("collection_failure_reason")
    if corruption == "empty_reason": episode["collection_failure_reason"] = " "
    if corruption == "bad_terminal_hash": episode["terminal_record_sha256"] = "unverified"
    if corruption == "extra_declared_call": episode["executed_call_indices"] = [0]
    if corruption == "extra_call_row": episode["calls"] = [deepcopy(bank["strata"]["a"]["episodes"][1]["calls"][0])]
    with pytest.raises(ValueError): report(design, bank)


def test_version_two_positive_call_bank_has_unchanged_statistical_results():
    design, bank = fixture()
    previous = report(design, bank)
    bank["schema_version"] = 2
    current = report(design, bank)
    for field in ("scales", "strata", "endpoint_status_counts", "spending"):
        assert current[field] == previous[field]
