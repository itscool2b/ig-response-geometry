"""Study-wide paired RMS analysis and honestly variance-only pilot planning.

This CPU module never collects contexts or runs a model. A prospective design
and an independently sealed artifact registry are separate inputs. There are
60 primary contrasts, not six independently corrected bank analyses. Bootstrap
replicates resample complete episode vectors within strata. Nominal Bonferroni
coverage is not a claim of small-sample bootstrap validity: coverage calibration
and bootstrap endpoint Monte Carlo precision are separate reported gates.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from experiment_io import canonical_json, file_hash, object_hash, strict_json

MODALITIES = ("vision", "language", "state")
DIRECTIONS = ("deletion", "insertion")
COMPARISONS = (("Q_IG", "random"), ("L2_IG", "random"),
               ("Q_IG", "input_difference"), ("L2_IG", "input_difference"),
               ("L2_IG", "Q_IG"))
LOCAL_CONTRASTS = [dict(modality=m, direction=d, method=a, control=b,
                        response="RMS", metric="raw_auc")
                   for m, d, (a, b) in itertools.product(MODALITIES, DIRECTIONS, COMPARISONS)]
FAMILY = "primary_rms"
SPECIFICITY_FAMILY = "weight_specificity_rms"
SPECIFICITY_CONTRASTS = [dict(modality=m,direction=d,method=t,control=f"permuted_{t}",response="RMS",metric="raw_auc")
    for m,d,t in itertools.product(MODALITIES,DIRECTIONS,("Q_IG","L2_IG"))]
E05_SCOPE = dict(null_kind="within_linear_weight_entry_permutation_v1",reference="own_function_self_reference",
    response="trained_probe_RMS",tie_rule="stable_descending_then_index",parameter_seed_unit="episode",
    completion_population="separate_joint_complete_episodes")
ESTIMAND = "equal_episode_mean_of_uniform_executed_calls_under_capped_behavior_policy"


def finite(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"Invalid finite {name}")
    return float(value)


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid integer {name}")
    return value


def digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Invalid SHA256 {name}")
    return value


def primary_registry(design):
    """Order is public and labeled. It is not an attempted blinding scheme."""
    return [dict(id=f"{g['id']}:{c['modality']}:{c['direction']}:{c['method']}-vs-{c['control']}",
                 group=g["id"], family=FAMILY, **c)
            for g in design["groups"] for c in LOCAL_CONTRASTS]


def local_contrasts(family=FAMILY):
    if family==FAMILY:return LOCAL_CONTRASTS
    if family==SPECIFICITY_FAMILY:return SPECIFICITY_CONTRASTS
    raise ValueError("Unknown prospective inferential family")


def family_registry(design,family=FAMILY):
    return [dict(id=f"{g['id']}:{c['modality']}:{c['direction']}:{c['method']}-vs-{c['control']}",
                 group=g["id"],family=family,**c) for g in design["groups"] for c in local_contrasts(family)]


def inferential_families(design):
    return (FAMILY,SPECIFICITY_FAMILY) if "e05" in design else (FAMILY,)


def prospective_coverage_plan(design,family=FAMILY):
    if "e05" in design:
        return design["analysis"].get("coverage_plans",{}).get(family)
    return design["analysis"].get("coverage_plan") if family==FAMILY else None


def validate_design(design):
    canonical_json(design)
    if design.get("schema_version") != 1 or design.get("kind") != "paired_global_design":
        raise ValueError("Unknown global design schema")
    if design.get("stage") not in {"variance_only_pilot", "confirmatory_locked"}:
        raise ValueError("A variance-only pilot or locked confirmation stage is required")
    if design.get("estimand") != ESTIMAND or design.get("modalities") != list(MODALITIES):
        raise ValueError("This primary design requires the declared per-call estimand and all three modalities")
    if design.get("max_episode_steps") != 400 or design.get("failure_policy") != "joint_complete_episodes_no_replacement":
        raise ValueError("Full capped trajectories and explicit joint conditional failure policy required")
    groups, strata = design.get("groups", []), design.get("strata", [])
    if len(groups) != 2 or len(strata) != 6 or sorted(len(g["weights"]) for g in groups) != [2, 4]:
        raise ValueError("This design has two fixed model groups and six task strata (four plus two)")
    ids = [s["id"] for s in strata]
    if len(set(ids)) != 6 or len({g["id"] for g in groups}) != 2:
        raise ValueError("Repeated stratum or model-group identifier")
    covered = []
    specs = {s["id"]:s for s in strata}
    for group in groups:
        weights = group["weights"]
        if any(not math.isclose(finite(w, "task weight", positive=True), 1/len(weights), rel_tol=0, abs_tol=1e-12)
               for w in weights.values()):
            raise ValueError("Model groups must retain their fixed equal-task weights")
        covered.extend(weights)
        if len({(specs[s]["model"],specs[s]["checkpoint_sha256"]) for s in weights}) != 1 or len({specs[s]["task"] for s in weights}) != len(weights):
            raise ValueError("Each model group requires one fixed checkpoint and distinct tasks")
    if sorted(covered) != sorted(ids):
        raise ValueError("Groups must partition the six declared strata")
    for stratum in strata:
        if not stratum.get("task") or not stratum.get("model"):
            raise ValueError("Every stratum needs a named task and model")
        digest(stratum["checkpoint_sha256"], "checkpoint")
        integer(stratum["calls_per_episode"], "locked call budget")
        integer(stratum["random_permutations"], "locked permutation budget", 2)
        digest(stratum["e01_gate_sha256"], "production numerical gate")
        seeds=stratum.get("reset_seeds",[])
        if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int or s<0 for s in seeds):
            raise ValueError("Every stratum requires the complete prospectively locked reset-seed roster")
    all_seeds=[seed for s in strata for seed in s["reset_seeds"]]
    if len(set(all_seeds))!=len(all_seeds):
        raise ValueError("Reset stream collision in the prospective roster")
    if not 0 < finite(design["global_alpha"], "global alpha") < 1:
        raise ValueError("Global alpha must lie in (0,1)")
    weights = design["family_weights"]
    if FAMILY not in weights or any(finite(w, "family weight", positive=True) > 1 for w in weights.values()) or sum(weights.values()) > 1+1e-12:
        raise ValueError("Positive family weights must consume at most the declared global alpha")
    if design.get("secondary_inference") != "descriptive_only_no_scale_effect":
        raise ValueError("Secondary task/geometry results are descriptive, not model-scale effects")
    if design.get("episode_dependence") != "disjoint_reset_streams_across_strata":
        raise ValueError("This implementation requires disjoint reset streams, not unmodeled shared blocks")
    if design.get("primary_contrasts") != primary_registry(design):
        raise ValueError("The exact 60 primary contrasts must be prospectively enumerated")
    if "e05" in design:
        if design["e05"]!=E05_SCOPE or design.get("e05_contrasts")!=family_registry(design,SPECIFICITY_FAMILY):
            raise ValueError("E05 scope and all 24 trained-response transfer contrasts must be explicit")
        expected={FAMILY:5/7,SPECIFICITY_FAMILY:2/7}
        if set(weights)!=set(expected) or any(not math.isclose(weights[k],v,rel_tol=0,abs_tol=1e-15) for k,v in expected.items()):
            raise ValueError("Expanded study requires the prospectively selected 60/84 and 24/84 alpha allocation")
        for stratum in strata:
            integer(stratum["parameter_draws"],"episode-coherent parameter draws",2)
            digest(stratum["e05_gate_sha256"],"randomized-function numerical gate")
    elif SPECIFICITY_FAMILY in weights:
        raise ValueError("A weight-specificity alpha allocation requires its full prospective scope")
    previous={s["id"]:0 for s in strata}
    stage_ids=set()
    for stage in design.get("pilot_stages",[]):
        if design["stage"]!="variance_only_pilot" or stage["id"] in stage_ids or set(stage["episodes_per_stratum"])!=set(ids):
            raise ValueError("Invalid prospective pilot stage registry")
        stage_ids.add(stage["id"])
        for sid,n in stage["episodes_per_stratum"].items():
            if not previous[sid]<integer(n,"stage episode count",2)<=len(specs[sid]["reset_seeds"]):
                raise ValueError("Pilot stages must be increasing prefixes of the locked reset roster")
            previous[sid]=n
    return design


def tensor_list_hash(values):
    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)) or not np.array_equal(array.astype(np.float64), np.asarray(values, dtype=np.float64)):
        raise ValueError("Saved active action is not an exact finite fp32 value")
    h = hashlib.sha256(canonical_json(dict(shape=list(array.shape), dtype="torch.float32")))
    h.update(array.astype("<f4", copy=False).tobytes(order="C"))
    return h.hexdigest()


def audit_rms(result):
    """Recompute positive RMS in float64 from exact saved fp32 active actions.

    GPU fp32 reduction and host float64 reduction differ by rounding. The
    32-epsilon relative audit tolerance is not a scientific IG tolerance.
    """
    reference = np.asarray(result["active_reference_action"], dtype=np.float64)
    if reference.ndim != 3 or reference.shape[-1] != len(result["active_action_indices"]) or reference.size == 0:
        raise ValueError("Invalid active action shape")
    if tensor_list_hash(result["active_reference_action"]) != result["active_reference_sha256"]:
        raise ValueError("Active reference hash mismatch")
    if len(set(result["active_action_indices"])) != reference.shape[-1]:
        raise ValueError("Active action coordinates repeat")
    scalars = {}
    for key, response in result["response_table"].items():
        if response["status"] != "finite":
            if response.get("RMS") is not None:
                raise ValueError("Failed response has a fabricated finite RMS")
            scalars[key] = None
            continue
        action = np.asarray(response["active_action"], dtype=np.float64)
        if action.shape != reference.shape or tensor_list_hash(response["active_action"]) != response["active_action_sha256"]:
            raise ValueError("Active action shape or hash mismatch")
        rms = float(np.sqrt(np.mean((action-reference)**2)))
        if not math.isclose(finite(response["RMS"], "RMS"), rms, rel_tol=32*np.finfo(np.float32).eps, abs_tol=np.finfo(np.float32).tiny):
            raise ValueError("Saved RMS does not reproduce from active actions")
        scalars[key] = response["RMS"]
    fractions = np.asarray(result["realized_fractions"], dtype=float)
    if fractions.ndim != 1 or len(fractions) < 2 or fractions[0] != 0 or fractions[-1] != 1 or np.any(np.diff(fractions) < 0):
        raise ValueError("Invalid realized intervention fraction grid")
    for ranking in result["rankings"].values():
        if ranking["status"] != "defined":
            continue
        exact = ranking.get("control_kind") == "exact_uniform_subset_expectation"
        for direction in DIRECTIONS:
            curve = ranking["curves"][direction]
            ids = curve["response_ids"]
            if len(ids) != len(fractions):
                raise ValueError("Curve grid/response membership mismatch")
            groups = ids if exact else [[key] for key in ids]
            if any(not group or len(set(group)) != len(group) for group in groups):
                raise ValueError("Empty or repeated subset response membership")
            values = [None if any(scalars[key] is None for key in group) else math.fsum(scalars[key] for key in group)/len(group) for group in groups]
            saved = curve["responses"]["RMS"]
            if saved["values"] != values:
                raise ValueError("RMS curve differs from its unique-mask responses")
            area = None if any(v is None for v in values) else float(trapezoid(values, fractions))
            if area != saved["raw_auc"]:
                raise ValueError("RMS area differs from the declared realized grid")
    return scalars[result["baseline_response_id"]]


def _ranking_area(rankings, label, direction):
    if label not in rankings:
        raise ValueError(f"Missing planned ranking {label}")
    item = rankings[label]
    if item["status"] != "defined":
        return None, f"{label}:{item['status']}"
    result = item["curves"][direction]["responses"]["RMS"]
    value = result["raw_auc"]
    if value is None:
        return None, f"{label}:{direction}:{result['status']}"
    return finite(value, "raw RMS area"), None


def context_effects(rows):
    """Paired vector and MC covariance; all directions share each random order."""
    values = np.full(len(LOCAL_CONTRASTS), np.nan)
    mc = np.zeros((len(values), len(values)))
    baselines, reasons, random_counts = {}, [], {}
    for modality in MODALITIES:
        row = rows[modality]
        if row["status"] != "evaluated":
            reasons.append(f"{modality}:{row.get('failure_kind', row['status'])}")
            baselines[modality] = None
            continue
        baselines[modality] = audit_rms(row["results"])
        rankings = row["results"]["rankings"]
        exact = "random_exact" in rankings
        labels = ["random_exact"] if exact else sorted(k for k in rankings if k.startswith("random_"))
        if not labels or (not exact and len(labels) < 2) or (exact and any(k.startswith("random_") and k != "random_exact" for k in rankings)):
            raise ValueError("Random control must be exact or have at least two MC permutations, never mixed")
        if exact and rankings["random_exact"].get("control_kind") != "exact_uniform_subset_expectation":
            raise ValueError("Unauthenticated exact-random label")
        random_counts[modality] = 0 if exact else len(labels)
        deviations = np.zeros((len(labels), len(values)))
        for i, contrast in enumerate(LOCAL_CONTRASTS):
            if contrast["modality"] != modality:
                continue
            treatment, problem = _ranking_area(rankings, contrast["method"], contrast["direction"])
            control_labels = labels if contrast["control"] == "random" else [contrast["control"]]
            controls = [_ranking_area(rankings, label, contrast["direction"]) for label in control_labels]
            failures = [x for x in [problem, *(item[1] for item in controls)] if x]
            if failures:
                reasons.extend(f"{modality}:{failure}" for failure in failures)
                continue
            sign = 1 if contrast["direction"] == "deletion" else -1
            replicates = sign*(treatment-np.asarray([item[0] for item in controls]))
            values[i] = replicates.mean()
            if contrast["control"] == "random" and not exact:
                deviations[:, i] = replicates-replicates.mean()
        if not exact:
            mc += deviations.T @ deviations / ((len(labels)-1)*len(labels))
    return values, mc, baselines, sorted(set(reasons)), random_counts


def _selection(bank):
    if bank.get("estimand") != ESTIMAND or bank.get("selection", {}).get("rule") != "uniform_executed_calls_v1":
        raise ValueError("Initial/fixed-call banks cannot stand in for the prospective per-call estimand")
    if bank["selection"].get("max_episode_steps") != 400:
        raise ValueError("Bank changed the locked 400-step trajectory cap")
    accounts = bank.get("selection_accounting", [])
    if not accounts:
        raise ValueError("Full executed/selected-call accounting is required")
    by_ep = {}
    for item in accounts:
        ep = item["episode"]
        eligible, selected, permutation = item["eligible_calls"], item["selected_calls"], item["permutation"]
        if ep in by_ep or not eligible or eligible != list(range(len(eligible))) or sorted(permutation) != eligible:
            raise ValueError("Invalid episode call population or permutation")
        if selected != permutation[:min(bank["selection"]["calls_per_episode"], len(eligible))]:
            raise ValueError("Selected calls differ from the locked random permutation prefix")
        for field, expected in (("inclusion_probability", len(selected)/len(eligible)),
                                ("within_episode_weight", 1/len(selected)), ("episode_weight", 1/len(accounts))):
            if item[field] != expected:
                raise ValueError("Incorrect selected-call inclusion probability or episode weight")
        by_ep[ep] = item
    return by_ep


def stratum_data(rows, bank, *, include_secondary=False):
    """Build one common complete-episode population for all 30 local contrasts."""
    accounts = _selection(bank)
    expected = {(e["episode_id"], e["policy_call_idx"], m): e["context_id"] for e in bank["contexts"] for m in MODALITIES}
    observed, lookup, diagnostics = {}, {}, Counter()
    for row in rows:
        key = row["episode_id"], row["policy_call_idx"], row["modality"]
        if key in observed:
            raise ValueError("Repeated episode/call/modality result")
        observed[key], lookup[key] = row["source_context_id"], row
        if row["status"]=="evaluated":
            for label,ranking in row["results"]["rankings"].items():
                if ranking["status"]=="defined":
                    prefix=f"{row['modality']}:{label}"
                    diagnostics[prefix+":defined"]+=1
                    diagnostics[prefix+":all_zero_map"]+=int(ranking.get("zero_map",False))
                    diagnostics[prefix+":tied_ranking"]+=int(ranking.get("tied_positions",0)>0)
                    # Endpoints are shared across rankings and directions. One
                    # record per context/modality/response avoids inflated rates.
            for response in ("RMS","Q","L2"):
                result=row["results"]
                a=result["response_table"][result["input_response_id"]].get(response)
                b=result["response_table"][result["baseline_response_id"]].get(response)
                if a is not None and b is not None:
                    diagnostics[f"{row['modality']}:{response}:finite_endpoints"]+=1
                    diagnostics[f"{row['modality']}:{response}:zero_endpoint_gap"]+=int(a==b)
                    cutoff=result.get("denominator_min",{}).get(response,0)
                    diagnostics[f"{row['modality']}:{response}:nearzero_endpoint_gap"]+=int(0<abs(a-b)<=cutoff)
    if observed != expected:
        raise ValueError("Full planned context membership does not match results")
    episodes = {}
    for entry in bank["contexts"]:
        episodes.setdefault(entry["episode_id"], []).append(entry)
    if len(episodes) != len(accounts):
        raise ValueError("Episode roster differs from selection accounting")
    retained, baseline_episodes, failure_counts, membership = [], {m: [] for m in MODALITIES}, Counter(), []
    for episode_id, entries in sorted(episodes.items()):
        entries = sorted(entries, key=lambda e: e["policy_call_idx"])
        account = accounts[entries[0]["episode"]]
        if sorted(e["policy_call_idx"] for e in entries) != sorted(account["selected_calls"]):
            raise ValueError("Bank selected calls differ from terminal selection accounting")
        outputs = [context_effects({m: lookup[(episode_id, e["policy_call_idx"], m)] for m in MODALITIES}) for e in entries]
        effects = np.stack([out[0] for out in outputs])
        mc = np.stack([out[1] for out in outputs])
        valid = bool(np.isfinite(effects).all())
        reasons = [reason for out in outputs for reason in out[3]]
        failure_counts.update(reasons)
        for modality in MODALITIES:
            endpoint_values = [out[2][modality] for out in outputs]
            if all(v is not None for v in endpoint_values):
                baseline_episodes[modality].append(dict(episode_id=episode_id, value=float(np.mean(endpoint_values))))
        membership.append(dict(episode_id=episode_id, reset_seed=entries[0]["reset_seed"],
            executed_calls=len(account["eligible_calls"]), selected_calls=len(entries),
            planned_episode_weight=1/len(episodes), joint_complete=valid, failure_reasons=sorted(set(reasons)),
            contexts=[dict(context_id=e["context_id"], call=e["policy_call_idx"],
                           within_episode_weight=1/len(entries), inclusion_probability=account["inclusion_probability"],
                           terminal_record_sha256=e["terminal_record_sha256"]) for e in entries]))
        if valid:
            counts = [out[4] for out in outputs]
            retained.append(dict(episode_id=episode_id, total_calls=len(account["eligible_calls"]),
                                 effects=effects, mc=mc, random_counts=counts))
    for item in membership:
        item["conditional_episode_weight"] = 1/len(retained) if item["joint_complete"] else 0
    return dict(episodes=retained, membership=membership, baseline_episodes=baseline_episodes,
                failure_counts=dict(failure_counts), planned_episodes=len(episodes),
                numerical_diagnostic_counts=dict(diagnostics),
                planned_contexts=len(bank["contexts"]),
                evaluated_calls_per_episode=bank["selection"]["calls_per_episode"],
                evaluated_random_permutations=sorted({count for ep in retained for call in ep["random_counts"] for count in call.values() if count}),
                conditioning="all_three_modalities_all_primary_methods_all_selected_calls_finite",
                secondary=secondary_summaries(rows,bank) if include_secondary else {},
                matrix=np.stack([e["effects"].mean(axis=0) for e in retained]) if retained else np.empty((0,30)))


def specificity_stratum_data(rows,bank,parameter_draws):
    """Separate E05 population; shared parameter draws remain episode clusters."""
    integer(parameter_draws,"parameter draws",2)
    accounts=_selection(bank)
    expected={(e["episode_id"],e["policy_call_idx"],m,r):e["context_id"]
        for e in bank["contexts"] for m in MODALITIES for r in range(parameter_draws)}
    lookup={};draw_ids={};seen_ids={};own_references={};diagnostics=Counter()
    for row in rows:
        key=(row["episode_id"],row["policy_call_idx"],row["modality"],row["draw_index"])
        if key in lookup:raise ValueError("Repeated E05 context/modality/parameter draw")
        if expected.get(key)!=row["source_context_id"]:raise ValueError("Unexpected E05 context membership")
        draw=(row["episode_id"],row["draw_index"])
        identity=digest(row["parameter_draw_sha256"],"parameter draw")
        if draw in draw_ids and draw_ids[draw]!=identity:raise ValueError("Parameter draw changed within an episode")
        if identity in seen_ids and seen_ids[identity]!=draw:raise ValueError("Parameter draw was reused across independent episode/draw units")
        draw_ids[draw]=identity;seen_ids[identity]=draw;lookup[key]=row
    if set(lookup)!=set(expected):raise ValueError("Incomplete prospective E05 membership; failed rows must be retained")
    by_ep={}
    for e in bank["contexts"]:by_ep.setdefault(e["episode_id"],[]).append(e)
    episodes=[];membership=[];failures=Counter();baselines={m:[] for m in MODALITIES}
    for eid,entries in sorted(by_ep.items()):
        entries=sorted(entries,key=lambda e:e["policy_call_idx"]);account=accounts[entries[0]["episode"]]
        if sorted(e["policy_call_idx"] for e in entries)!=sorted(account["selected_calls"]):raise ValueError("E05 selected calls differ from the bank")
        values=np.full((parameter_draws,len(entries),12),np.nan);reasons=[];trained={};ep_baselines={m:[] for m in MODALITIES}
        for j,e in enumerate(entries):
            for modality in MODALITIES:
                baseline=None
                for r in range(parameter_draws):
                    row=lookup[(eid,e["policy_call_idx"],modality,r)]
                    if row["status"]!="evaluated":
                        reasons.append(f"{modality}:draw{r}:{row.get('failure_kind',row['status'])}");continue
                    identity=digest(row["randomized_reference_sha256"],"own-function reference")
                    reference_key=(eid,e["policy_call_idx"],r)
                    if reference_key in own_references and own_references[reference_key]!=identity:raise ValueError("Own-function actual reference changed across modalities")
                    own_references[reference_key]=identity
                    result=row["results"];current_baseline=audit_rms(result);baseline=current_baseline
                    rankings=result["rankings"]
                    anchor=object_hash(dict(reference=result["active_reference_sha256"],
                        input=result["response_table"][result["input_response_id"]],baseline=result["response_table"][result["baseline_response_id"]],
                        fractions=result["realized_fractions"],trained={t:rankings[f"trained_{t}"] for t in ("Q_IG","L2_IG")}))
                    context_key=(e["policy_call_idx"],modality)
                    if context_key in trained and trained[context_key]!=anchor:raise ValueError("Trained response or trained ranking changed across parameter draws")
                    trained[context_key]=anchor
                    for label,ranking in rankings.items():
                        prefix=f"{modality}:{label}"
                        diagnostics[prefix+":"+ranking["status"]]+=1
                        if ranking["status"]=="defined":
                            diagnostics[prefix+":all_zero_map"]+=int(ranking.get("zero_map",False))
                            diagnostics[prefix+":tied_ranking"]+=int(ranking.get("tied_positions",0)>0)
                    for k,c in enumerate(SPECIFICITY_CONTRASTS):
                        if c["modality"]!=modality:continue
                        a,problem_a=_ranking_area(rankings,"trained_"+c["method"],c["direction"])
                        b,problem_b=_ranking_area(rankings,c["control"],c["direction"])
                        if problem_a or problem_b:
                            reasons.extend(f"{modality}:draw{r}:{problem}" for problem in (problem_a,problem_b) if problem)
                        else:values[r,j,k]=(1 if c["direction"]=="deletion" else -1)*(a-b)
                if baseline is not None:ep_baselines[modality].append(baseline)
        valid=bool(np.isfinite(values).all());failures.update(reasons)
        member=dict(episode_id=eid,reset_seed=entries[0]["reset_seed"],executed_calls=len(account["eligible_calls"]),
            selected_calls=len(entries),planned_episode_weight=1/len(by_ep),joint_complete=valid,failure_reasons=sorted(set(reasons)),
            parameter_draws=[dict(draw_index=r,parameter_draw_sha256=draw_ids[(eid,r)],within_episode_draw_weight=1/parameter_draws) for r in range(parameter_draws)],
            contexts=[dict(context_id=e["context_id"],call=e["policy_call_idx"],within_episode_weight=1/len(entries),
                inclusion_probability=account["inclusion_probability"],terminal_record_sha256=e["terminal_record_sha256"]) for e in entries])
        membership.append(member)
        for m in MODALITIES:
            if len(ep_baselines[m])==len(entries):baselines[m].append(dict(episode_id=eid,value=float(np.mean(ep_baselines[m]))))
        if valid:
            draw_vectors=values.mean(axis=1)
            episodes.append(dict(episode_id=eid,total_calls=len(account["eligible_calls"]),effects=values.mean(axis=0),
                draw_call_effects=values,draw_episode_effects=draw_vectors,parameter_mc_covariance=_cov(draw_vectors)/parameter_draws))
    for member in membership:member["conditional_episode_weight"]=1/len(episodes) if member["joint_complete"] else 0
    return dict(episodes=episodes,membership=membership,baseline_episodes=baselines,failure_counts=dict(failures),
        numerical_diagnostic_counts=dict(diagnostics),diagnostic_unit="context_modality_parameter_draw",planned_episodes=len(by_ep),planned_contexts=len(bank["contexts"]),
        evaluated_calls_per_episode=bank["selection"]["calls_per_episode"],evaluated_parameter_draws=parameter_draws,
        conditioning="all_selected_calls_modalities_targets_and_parameter_draws_finite; separate_from_base_family",
        secondary={},matrix=np.stack([e["draw_episode_effects"].mean(axis=0) for e in episodes]) if episodes else np.empty((0,12)))


def secondary_summaries(rows,bank):
    """All score conventions, component controls and ratio curves, descriptive.

    Each modality/response/metric uses one joint complete-episode population
    for all six ranking methods, including both path-gradient controls. Curves
    are indexed by requested intervention percent, with realized fractions
    disclosed, not silently interpolated onto a different estimand.
    """
    methods=("Q_IG","L2_IG","input_difference","Q_path_gradient","L2_path_gradient","random")
    by_ep={}
    for entry in bank["contexts"]:
        by_ep.setdefault(entry["episode_id"],[]).append(entry)
    lookup={(r["episode_id"],r["policy_call_idx"],r["modality"]):r for r in rows}
    output={}
    for modality,response,metric in itertools.product(MODALITIES,("RMS","Q","L2"),("raw_auc","normalized_auc")):
        groupid=f"{modality}:{response}:{metric}"
        episode_areas=[]; episode_curves=[]; retained=[]; excluded=[]; fractions=[]; grid=None
        for eid,entries in sorted(by_ep.items()):
            areas=[]; curves=[]; valid=True; ep_fractions=[]
            for entry in entries:
                row=lookup[(eid,entry["policy_call_idx"],modality)]
                if row["status"]!="evaluated":
                    valid=False; break
                result=row["results"]; rankings=result["rankings"]
                nominal=result["nominal_grid_percent"]
                if grid is None:
                    grid=nominal
                elif grid!=nominal:
                    raise ValueError("Secondary curves changed the locked nominal intervention grid")
                a=np.empty((2,6)); curve_values=np.empty((2,6,len(grid)))
                for mi,method in enumerate(methods):
                    labels=(["random_exact"] if "random_exact" in rankings else sorted(k for k in rankings if k.startswith("random_"))) if method=="random" else [method]
                    if not labels or any(k not in rankings for k in labels):
                        raise ValueError("A planned secondary ranking is absent")
                    for di,direction in enumerate(DIRECTIONS):
                        cells=[]; traces=[]
                        for label in labels:
                            ranking=rankings[label]
                            if ranking["status"]!="defined":
                                valid=False; break
                            cell=ranking["curves"][direction]["responses"][response]
                            if cell[metric] is None:
                                valid=False; break
                            cells.append(finite(cell[metric],"secondary area"))
                            values=np.asarray(cell["values"],dtype=float)
                            if metric=="normalized_auc":
                                actual=result["response_table"][result["input_response_id"]][response]
                                baseline=result["response_table"][result["baseline_response_id"]][response]
                                gap=actual-baseline
                                if abs(gap)<=result["denominator_min"][response]:
                                    raise ValueError("Normalized curve escaped its shared endpoint eligibility rule")
                                values=(values-baseline)/gap
                            if not np.isfinite(values).all():
                                raise ValueError("Nonfinite secondary curve escaped explicit failure handling")
                            traces.append(values)
                        if not valid:
                            break
                        a[di,mi]=np.mean(cells); curve_values[di,mi]=np.mean(traces,axis=0)
                    if not valid:
                        break
                if not valid:
                    break
                areas.append(a); curves.append(curve_values); ep_fractions.append(result["realized_fractions"])
            if valid:
                episode_areas.append(np.mean(areas,axis=0)); episode_curves.append(np.mean(curves,axis=0))
                retained.append(dict(episode_id=eid,contexts=[dict(context_id=e["context_id"],call=e["policy_call_idx"],weight=1/len(entries)) for e in entries]))
                fractions.extend(ep_fractions)
            else:
                excluded.append(eid)
        n=len(retained)
        for ep in retained:
            ep["episode_weight"]=1/n
        record=dict(modality=modality,response=response,metric=metric,methods=list(methods),directions=list(DIRECTIONS),
            inference="descriptive_only",interpretation="response_geometry_and_specific_component_controls; not_model_scale_effect",
            planned_episodes=len(by_ep),defined_episodes=n,excluded_episode_ids=excluded,population=retained,
            population_sha256=object_hash(retained),nominal_grid_percent=grid,
            realized_fraction_min=np.min(fractions,axis=0).tolist() if n else None,
            realized_fraction_max=np.max(fractions,axis=0).tolist() if n else None,
            area_mean=None,area_standard_error=None,curve_mean=None,contrasts=[])
        if n:
            array=np.stack(episode_areas)
            record.update(area_mean=array.mean(axis=0).tolist(),
                area_standard_error=(array.std(axis=0,ddof=1)/np.sqrt(n)).tolist() if n>1 else None,
                curve_mean=np.stack(episode_curves).mean(axis=0).tolist())
            for di,direction in enumerate(DIRECTIONS):
                for method,control in (*COMPARISONS,("Q_IG","Q_path_gradient"),("L2_IG","L2_path_gradient")):
                    sign=(1 if direction=="deletion" else -1) if response=="RMS" and metric=="raw_auc" else (-1 if direction=="deletion" else 1)
                    paired=sign*(array[:,di,methods.index(method)]-array[:,di,methods.index(control)])
                    record["contrasts"].append(dict(method=method,control=control,direction=direction,
                        estimate=float(paired.mean()),standard_error=float(paired.std(ddof=1)/np.sqrt(n)) if n>1 else None))
        output[groupid]=record
    return output


def load_study(design, registry, design_sha256):
    """Only authenticated completed producers enter analysis; hash pins are external.

    Registry creation time does not prove prospective registration. The producer
    protocol must already contain the trusted global design's file digest.
    """
    from paired_comparison import load_completed_study
    validate_design(design)
    if registry.get("global_design_sha256") != design_sha256:
        raise ValueError("Registry/global design identity mismatch")
    entries = registry["strata"]
    if len(entries) != 6 or {e["id"] for e in entries} != {s["id"] for s in design["strata"]}:
        raise ValueError("Artifact registry must cover all six strata exactly once")
    output, seen_contexts, seen_resets = {}, set(), set()
    specs = {s["id"]: s for s in design["strata"]}
    stages={s["id"]:s for s in design.get("pilot_stages",[])}
    if stages and registry.get("pilot_stage") not in stages:
        raise ValueError("Registry must name one prospectively locked pilot stage")
    flat_entries=[]
    for entry in entries:
        shards=entry.get("shards",[entry])
        if not shards:
            raise ValueError("A stratum has no completed artifacts")
        flat_entries.extend(dict(shard,id=entry["id"]) for shard in shards)
    for entry in flat_entries:
        path = Path(entry["metrics"])
        for suffix, field in (("", "metrics_sha256"), (".manifest.json", "manifest_sha256"), (".completion.json", "completion_sha256")):
            if file_hash(str(path)+suffix) != digest(entry[field], field):
                raise ValueError(f"Registry artifact mismatch: {field}")
        rows, manifest, completion = load_completed_study(path)
        config, spec = manifest["configuration"], specs[entry["id"]]
        protocol, bank = config["protocol"], config["bank"]
        if protocol.get("global_design_sha256") != design_sha256 or protocol["stage"] != design["stage"]:
            raise ValueError("Producer was not bound to this stage and prospective global design")
        if any(config[field] != entry[field] for field in ("bank_sha256", "protocol_sha256")):
            raise ValueError("Global registry changed the producer bank or protocol")
        if config["e01_gate_sha256"] != spec["e01_gate_sha256"]:
            raise ValueError("Producer changed the locked stratum numerical gate")
        if bank["selection"]["calls_per_episode"] != spec["calls_per_episode"] or protocol["random_permutations"] != spec["random_permutations"]:
            raise ValueError("Producer changed the locked call or random-order budget")
        if (bank["task"], bank["model"]) != (spec["task"], spec["model"]):
            raise ValueError("Task/model stratum mismatch")
        # Checkpoint digest must be authenticated by the collector's pipeline
        # identity, not merely copied into a post hoc analysis manifest.
        pipeline = bank["pipeline"]
        if pipeline.get("checkpoint",{}).get("sha256") != spec["checkpoint_sha256"]:
            raise ValueError("Declared checkpoint absent from authenticated pipeline identity")
        item = stratum_data(rows, bank, include_secondary=True)
        exact_only=all(r["status"]=="evaluated" and "random_exact" in r["results"]["rankings"] for r in rows)
        item["evaluated_random_permutations"]=[] if exact_only else [protocol["random_permutations"]]
        for ep in item["membership"]:
            if ep["reset_seed"] in seen_resets:
                raise ValueError("Reset stream collision across declared independent strata")
            seen_resets.add(ep["reset_seed"])
            for context in ep["contexts"]:
                if context["context_id"] in seen_contexts:
                    raise ValueError("Context reused across strata")
                seen_contexts.add(context["context_id"])
        output.setdefault(entry["id"],[]).append(item)
    output={sid:combine_shards(items) for sid,items in output.items()}
    for sid,item in output.items():
        roster=specs[sid]["reset_seeds"]
        if stages:
            roster=roster[:stages[registry["pilot_stage"]]["episodes_per_stratum"][sid]]
        if sorted(ep["reset_seed"] for ep in item["membership"])!=sorted(roster):
            raise ValueError("Completed bank differs from the full prospective reset-seed roster")
    return output


def load_specificity_study(design,registry,design_sha256,base_data):
    """Authenticate separate E05 shards against the already loaded base study."""
    from weight_arrangement_control import load_completed_control
    validate_design(design)
    if "e05" not in design or registry.get("global_design_sha256")!=design_sha256:raise ValueError("Missing prospective E05 design binding")
    entries=registry["e05_strata"];specs={s["id"]:s for s in design["strata"]}
    if len(entries)!=6 or {e["id"] for e in entries}!=set(specs):raise ValueError("E05 registry must cover every declared stratum")
    base_paths={}
    for entry in registry["strata"]:
        for shard in entry.get("shards",[entry]):
            base_paths[(entry["id"],str(Path(shard["metrics"]).resolve()))]=shard
    output={};all_draws=set()
    for entry in entries:
        sid=entry["id"];spec=specs[sid];items=[]
        for shard in entry.get("shards",[entry]):
            path=Path(shard["metrics"]);base=Path(shard["base_metrics"])
            original=base_paths.get((sid,str(base.resolve())))
            if original is None:raise ValueError("E05 base path is outside its authenticated study stratum")
            for prefix,source in (("",path),("base_",base)):
                for suffix,field in (("","metrics_sha256"),(".manifest.json","manifest_sha256"),(".completion.json","completion_sha256")):
                    if file_hash(str(source)+suffix)!=digest(shard[prefix+field],prefix+field):raise ValueError("E05 registry artifact mismatch")
                    if prefix and shard[prefix+field]!=original[field]:raise ValueError("E05 changed its original base identity")
            rows,manifest,_=load_completed_control(path,base_path=base)
            config=manifest["configuration"];protocol=config["protocol"];bank=config["bank"]
            if protocol.get("global_design_sha256")!=design_sha256 or protocol["stage"]!=design["stage"]:raise ValueError("E05 producer is not bound to the original prospective design")
            if any(config[field]!=shard[field] for field in ("bank_sha256","protocol_sha256")):raise ValueError("E05 bank/protocol identity mismatch")
            if shard["bank_sha256"]!=original["bank_sha256"] or shard["base_protocol_sha256"]!=original["protocol_sha256"]:raise ValueError("E05 bank or base protocol changed")
            if config["e05_gate_sha256"]!=spec["e05_gate_sha256"] or protocol["parameter_draws"]!=spec["parameter_draws"]:raise ValueError("E05 changed its numerical gate or parameter-draw budget")
            if (bank["task"],bank["model"],bank["pipeline"]["checkpoint"]["sha256"])!=(spec["task"],spec["model"],spec["checkpoint_sha256"]):raise ValueError("E05 task/checkpoint mismatch")
            if bank["selection"]["calls_per_episode"]!=spec["calls_per_episode"]:raise ValueError("E05 changed its call budget")
            if {key:protocol[key] for key in E05_SCOPE}!=design["e05"]:raise ValueError("E05 changed its scientific null/reference/tie scope")
            item=specificity_stratum_data(rows,bank,spec["parameter_draws"])
            for member in item["membership"]:
                for draw in member["parameter_draws"]:
                    identity=draw["parameter_draw_sha256"]
                    if identity in all_draws:raise ValueError("E05 parameter draw reused across independent study episodes")
                    all_draws.add(identity)
            items.append(item)
        if not items:raise ValueError("E05 stratum has no completed shards")
        output[sid]=combine_shards(items)
        def population(item):return {e["episode_id"]:(e["reset_seed"],e["executed_calls"],e["selected_calls"],e["contexts"]) for e in item["membership"]}
        if population(output[sid])!=population(base_data[sid]):raise ValueError("E05 planned episode/context membership differs from its base study")
    return output


def combine_shards(items):
    """Merge disjoint episode shards without treating shards as equal units."""
    from copy import deepcopy
    ids=[e["episode_id"] for item in items for e in item["membership"]]
    if len(ids)!=len(set(ids)):
        raise ValueError("An episode appears in multiple shards")
    result=deepcopy(items[0])
    for item in items[1:]:
        if any(item.get(k)!=result.get(k) for k in ("evaluated_calls_per_episode","evaluated_random_permutations","evaluated_parameter_draws")):
            raise ValueError("Pilot shards changed J, M or R and their conditional population")
        for key in ("episodes","membership"):
            result[key].extend(deepcopy(item[key]))
        result["matrix"]=np.concatenate([result["matrix"],item["matrix"]])
        for key in ("planned_episodes","planned_contexts"):
            result[key]+=item[key]
        for key in ("failure_counts","numerical_diagnostic_counts"):
            result[key]=dict(Counter(result[key])+Counter(item[key]))
        for modality in MODALITIES:
            result["baseline_episodes"][modality].extend(item["baseline_episodes"][modality])
        for key,right in item["secondary"].items():
            left=result["secondary"][key]
            n,m=left["defined_episodes"],right["defined_episodes"]
            if left["nominal_grid_percent"]!=right["nominal_grid_percent"]:
                raise ValueError("Secondary grid differs across study shards")
            def pool(mean_a,se_a,mean_b,se_b):
                if not n:return mean_b,se_b
                if not m:return mean_a,se_a
                a,b=np.asarray(mean_a),np.asarray(mean_b)
                mean=(n*a+m*b)/(n+m)
                ss=(np.zeros_like(a) if n<2 else np.asarray(se_a)**2*n*(n-1))+(np.zeros_like(b) if m<2 else np.asarray(se_b)**2*m*(m-1))
                variance=(ss+n*(a-mean)**2+m*(b-mean)**2)/(n+m-1)
                return mean.tolist(),np.sqrt(variance/(n+m)).tolist()
            left["area_mean"],left["area_standard_error"]=pool(left["area_mean"],left["area_standard_error"],right["area_mean"],right["area_standard_error"])
            if n and m:
                left["curve_mean"]=((n*np.array(left["curve_mean"])+m*np.array(right["curve_mean"]))/ (n+m)).tolist()
                left["realized_fraction_min"]=np.minimum(left["realized_fraction_min"],right["realized_fraction_min"]).tolist()
                left["realized_fraction_max"]=np.maximum(left["realized_fraction_max"],right["realized_fraction_max"]).tolist()
                for a,b in zip(left["contrasts"],right["contrasts"]):
                    a["estimate"],a["standard_error"]=pool(a["estimate"],a["standard_error"],b["estimate"],b["standard_error"])
            elif m:
                for field in ("curve_mean","realized_fraction_min","realized_fraction_max","contrasts"):
                    left[field]=deepcopy(right[field])
            left["defined_episodes"]+=m; left["planned_episodes"]+=right["planned_episodes"]
            left["population"].extend(deepcopy(right["population"]))
            left["excluded_episode_ids"].extend(right["excluded_episode_ids"])
    for ep in result["membership"]:
        ep["planned_episode_weight"]=1/result["planned_episodes"]
        ep["conditional_episode_weight"]=1/len(result["episodes"]) if ep["joint_complete"] else 0
    for record in result["secondary"].values():
        for ep in record["population"]:
            ep["episode_weight"]=1/record["defined_episodes"]
        record["population_sha256"]=object_hash(record["population"])
    return result


def _digest_leaves(value):
    if isinstance(value, dict):
        return set().union(*(_digest_leaves(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_digest_leaves(v) for v in value)) if value else set()
    return {value} if isinstance(value, str) else set()


def _cov(matrix):
    if len(matrix) < 2:
        raise ValueError("At least two independent complete episodes are required for variance")
    return np.atleast_2d(np.cov(matrix, rowvar=False, ddof=1))


def variance_components(data):
    """Moment estimates for design planning, not distribution-free bounds."""
    episodes, matrix = data["episodes"], data["matrix"]
    covariance = _cov(matrix)
    within, mc_mean, mc_per_order, total, counts, clipped_within = [], [], [], [], [], []
    for ep in episodes:
        j, n = len(ep["effects"]), ep["total_calls"]
        if j < 2 and n > 1:
            raise ValueError("J=1 pilot cannot identify within-episode call variance for a longer trajectory")
        noise = ep["mc"].mean(axis=0)
        unclipped_within=np.zeros(30) if n==1 else np.diag(_cov(ep["effects"]))-np.diag(noise)
        clipped_within.append(unclipped_within<0)
        call_variance=np.maximum(unclipped_within,0)
        within.append(call_variance)
        mc_mean.append(np.diag(noise)/j)
        # Independent context streams, exact small-group controls have M=0.
        raw_mc = []
        for call_mc, permutation_counts in zip(ep["mc"], ep["random_counts"]):
            multiplier = np.array([permutation_counts[c["modality"]] for c in LOCAL_CONTRASTS])
            raw_mc.append(np.diag(call_mc)*multiplier)
        mc_per_order.append(np.mean(raw_mc, axis=0))
        total.append(n)
        counts.append(j)
    within, mc_mean, mc_per_order = map(np.asarray, (within, mc_mean, mc_per_order))
    total, counts = np.asarray(total), np.asarray(counts)
    sampled = within*((1-counts/total)/counts)[:,None]
    unclipped = np.diag(covariance)-sampled.mean(axis=0)-mc_mean.mean(axis=0)
    return dict(covariance=covariance, between=np.maximum(unclipped, 0), between_unclipped=unclipped,
                within=within, mc_per_order=mc_per_order, observed_mc=mc_mean,
                within_negative_moment_counts=np.sum(clipped_within,axis=0),
                total_calls=total, sampled_calls=counts)


def candidate_variance(components, calls, permutations):
    integer(calls, "candidate call budget")
    integer(permutations, "candidate permutation budget", 2)
    j = np.minimum(calls, components["total_calls"])
    sampling = np.mean(components["within"]*((1-j/components["total_calls"])/j)[:,None], axis=0)
    mc_unit = np.mean(components["mc_per_order"]/j[:,None], axis=0)
    non_mc = components["between"]+sampling
    return dict(non_mc_variance=non_mc, mc_variance_per_permutation=mc_unit,
                total_variance=non_mc+mc_unit/permutations, expected_calls=float(j.mean()))


def specificity_variance_components(data):
    """Decompose episode-coherent parameter MC without treating calls as draws."""
    covariance=_cov(data["matrix"]);within=[];mc_within=[];mc_full=[];observed=[];totals=[];counts=[];negative=[];negative_mc=[]
    for ep in data["episodes"]:
        array=ep["draw_call_effects"];r,j,width=array.shape;n=ep["total_calls"]
        if r<2 or width!=12 or (j<2 and n>1):raise ValueError("E05 pilot needs repeated parameter draws and calls to identify components")
        mc=ep["parameter_mc_covariance"];observed.append(np.diag(mc));fpc=(1-j/n)/j
        centered=array-array.mean(axis=1,keepdims=True)
        within_mc=centered.var(axis=0,ddof=1).sum(axis=0)/(j-1) if j>1 else np.zeros(width)
        raw_within=ep["effects"].var(axis=0,ddof=1)-within_mc/r if j>1 else np.zeros(width)
        full=np.diag(mc)*r-fpc*within_mc
        within.append(np.maximum(raw_within,0));negative.append(raw_within<0)
        mc_within.append(within_mc);mc_full.append(np.maximum(full,0));negative_mc.append(full<0)
        totals.append(n);counts.append(j)
    totals=np.asarray(totals);counts=np.asarray(counts);within=np.asarray(within);observed=np.asarray(observed)
    between_unclipped=np.diag(covariance)-np.mean(within*((1-counts/totals)/counts)[:,None]+observed,axis=0)
    return dict(covariance=covariance,within=within,observed_mc=observed,between=np.maximum(between_unclipped,0),
        between_unclipped=between_unclipped,within_negative_moment_counts=np.sum(negative,axis=0),
        parameter_full_mean_negative_moment_counts=np.sum(negative_mc,axis=0),
        mc_within=np.asarray(mc_within),mc_full_mean=np.asarray(mc_full),total_calls=totals,sampled_calls=counts)


def candidate_specificity_variance(components,calls,parameter_draws):
    integer(calls,"candidate E05 calls");integer(parameter_draws,"candidate parameter draws",2)
    j=np.minimum(calls,components["total_calls"]);fpc=((1-j/components["total_calls"])/j)[:,None]
    non_mc=components["between"]+np.mean(components["within"]*fpc,axis=0)
    mc_unit=np.mean(components["mc_full_mean"]+components["mc_within"]*fpc,axis=0)
    return dict(non_mc_variance=non_mc,mc_variance_per_parameter_draw=mc_unit,
        total_variance=non_mc+mc_unit/parameter_draws,expected_calls=float(j.mean()))


def pilot_report(design, data, *, draws, seed, upper_quantile, call_budgets, permutations,family=FAMILY,baseline_data=None):
    """No means, signed effects, episode effects, efficacy curves or opaque IDs.

    Episode-bootstrap upper quantiles describe pilot uncertainty under the
    empirical resampling model. Recruitment and tail coverage require a
    separately locked/calibrated planning procedure; this function says so.
    """
    if design["stage"] != "variance_only_pilot":
        raise ValueError("Pilot output requires a variance-only design")
    integer(draws, "pilot bootstrap draws", 2)
    if not .5 < finite(upper_quantile, "planning upper quantile") < 1:
        raise ValueError("Planning upper quantile must exceed one half")
    if len(set(call_budgets)) != len(call_budgets):
        raise ValueError("Repeated call-budget candidate")
    for j in call_budgets:
        integer(j, "call budget")
    rng = np.random.Generator(np.random.PCG64(seed))
    width=len(local_contrasts(family));specificity=family==SPECIFICITY_FAMILY
    component_fn=specificity_variance_components if specificity else variance_components
    candidate_fn=candidate_specificity_variance if specificity else candidate_variance
    budget_field="parameter_draws" if specificity else "random_permutations"
    unit_field="mc_variance_per_parameter_draw" if specificity else "mc_variance_per_permutation"
    if specificity and baseline_data is None:raise ValueError("E05 planning must use the shared noncomparative base-study scale population")
    baseline_data=data if baseline_data is None else baseline_data
    report = dict(kind="variance_only_pilot_report", stage="variance_only_pilot",family=family, disclosure="labeled_variance_only_not_blinded",
        no_signed_efficacy_estimates=True, primary_contrasts=family_registry(design,family),
        uncertainty_status="empirical_bootstrap_planning_approximation_requires_recruitment_and_coverage_calibration",
        uncertainty=dict(draws=draws, seed=seed, upper_quantile=upper_quantile), strata={}, scales={})
    baseline_bootstrap = {}
    for stratum in design["strata"]:
        sid, item = stratum["id"], data[stratum["id"]]
        components = component_fn(item)
        n = len(item["matrix"])
        candidates = {j: candidate_fn(components, j, permutations) for j in call_budgets}
        boot = {j: np.empty((draws,width)) for j in call_budgets}
        boot_mc = {j: np.empty((draws,width)) for j in call_budgets}
        for b in range(draws):
            indices = rng.integers(0,n,n)
            sample = dict(episodes=[item["episodes"][k] for k in indices], matrix=item["matrix"][indices])
            comp = component_fn(sample)
            for j in call_budgets:
                candidate=candidate_fn(comp,j,permutations)
                boot[j][b] = candidate["non_mc_variance"]
                boot_mc[j][b] = candidate[unit_field]
        frontier = []
        for j, point in candidates.items():
            upper = np.maximum(point["non_mc_variance"], np.quantile(boot[j],upper_quantile,axis=0))
            frontier.append(dict(calls_per_episode=j, **{budget_field:permutations},
                expected_selected_calls=point["expected_calls"],
                non_mc_variance=point["non_mc_variance"].tolist(), non_mc_variance_upper=upper.tolist(),
                **{unit_field:point[unit_field].tolist(),unit_field+"_upper":np.maximum(point[unit_field],np.quantile(boot_mc[j],upper_quantile,axis=0)).tolist()},
                variance_upper_to_point_ratio=[float(u/v) if v>0 else None for u,v in zip(upper,point["non_mc_variance"])]))
        centered = item["matrix"]-item["matrix"].mean(axis=0)
        report["strata"][sid] = dict(planned_episodes=item["planned_episodes"], complete_episodes=n,
            planned_contexts=item["planned_contexts"], membership=item["membership"],
            evaluated_calls_per_episode=item["evaluated_calls_per_episode"],
            **({"evaluated_parameter_draws":item["evaluated_parameter_draws"]} if specificity else {"evaluated_random_permutations":item["evaluated_random_permutations"]}),
            membership_sha256=object_hash(item["membership"]), failure_counts=item["failure_counts"],
            numerical_diagnostic_counts=item["numerical_diagnostic_counts"],
            conditioning=item["conditioning"], covariance=components["covariance"].tolist(),
            observed_total_variance=np.diag(components["covariance"]).tolist(),
            observed_mc_variance_mean=components["observed_mc"].mean(axis=0).tolist(),
            between_variance=components["between"].tolist(), negative_moment_cells=np.flatnonzero(components["between_unclipped"]<0).tolist(),
            within_negative_moment_counts=components["within_negative_moment_counts"].tolist(),
            zero_observed_variance_cells=np.flatnonzero(np.diag(components["covariance"])==0).tolist(),
            centered_absolute_max=np.max(abs(centered),axis=0).tolist(),
            centered_absolute_q95=np.quantile(abs(centered),.95,axis=0).tolist(), frontier=frontier)
        if specificity:report["strata"][sid]["parameter_full_mean_negative_moment_counts"]=components["parameter_full_mean_negative_moment_counts"].tolist()
        baseline_bootstrap[sid] = {}
        for modality in MODALITIES:
            values = np.array([r["value"] for r in baseline_data[sid]["baseline_episodes"][modality]])
            if len(values) != baseline_data[sid]["planned_episodes"]:
                raise ValueError("Noncomparative scale requires all planned episodes' baseline endpoints; missingness must be resolved or redesigned")
            samples = np.empty(draws)
            for start in range(0,draws,512):
                end = min(start+512,draws)
                samples[start:end] = np.median(values[rng.integers(0,len(values),(end-start,len(values)))],axis=1)
            baseline_bootstrap[sid][modality] = dict(point=float(np.median(values)), samples=samples)
    for group in design["groups"]:
        for modality in MODALITIES:
            point = sum(w*baseline_bootstrap[s][modality]["point"] for s,w in group["weights"].items())
            samples = sum(w*baseline_bootstrap[s][modality]["samples"] for s,w in group["weights"].items())
            lower, upper = np.quantile(samples,[1-upper_quantile,upper_quantile])
            report["scales"][f"{group['id']}:{modality}"] = dict(value=point, lower=float(lower), upper=float(upper),
                definition="fixed_equal_task_weighted_median_of_episode_mean_all_baseline_RMS_displacement",
                halfwidth=.05*point, substantial_difference=.10*point, conservative_planning_halfwidth=.05*float(lower),
                status="positive_empirical_scale" if lower>0 else "unresolved_nonpositive_scale_lower_bound",
                interpretation="benchmark_resolution_not_physical_utility_or_task_success_threshold")
    return report


def specificity_pilot_report(design,data,baseline_data, *, parameter_draws,**kwargs):
    return pilot_report(design,data,permutations=parameter_draws,family=SPECIFICITY_FAMILY,baseline_data=baseline_data,**kwargs)


def bootstrap_requirement(alpha, comparisons, min_tail_draws=100, tail_relative_mcse=.1):
    p = finite(alpha,"family alpha",positive=True)/(2*integer(comparisons,"comparisons"))
    if not 0<p<.5 or not 0<tail_relative_mcse<1:
        raise ValueError("Invalid tail probability or MC precision")
    def ceil_arithmetic(value):
        # Rational family allocations can round one ulp above an integer.
        rounded=round(value)
        return rounded if math.isclose(value,rounded,rel_tol=1e-14,abs_tol=1e-10) else math.ceil(value)
    return max(ceil_arithmetic(min_tail_draws/p),ceil_arithmetic((1-p)/(p*tail_relative_mcse**2)))


def _bootstrap(design,data,draws,seed):
    return _bootstrap_family(design,data,draws,seed,FAMILY)


def _bootstrap_family(design,data,draws,seed,family):
    rng = np.random.Generator(np.random.PCG64(seed))
    width=len(local_contrasts(family))
    output = np.zeros((draws,len(design["groups"])*width))
    for g, group in enumerate(design["groups"]):
        for sid, weight in group["weights"].items():
            matrix = data[sid]["matrix"]
            if matrix.ndim!=2 or matrix.shape[1]!=width or not np.isfinite(matrix).all():
                raise ValueError("Episode matrix differs from its finite family contrast scope")
            n = len(matrix)
            if n<2:
                raise ValueError("Insufficient complete episodes in a declared task; task weights cannot be renormalized")
            for start in range(0,draws,256):
                stop=min(start+256,draws)
                counts=rng.multinomial(n,np.full(n,1/n),size=stop-start)
                output[start:stop,g*width:(g+1)*width] += weight*(counts@matrix/n)
    return output


def calibration_scope(design, complete_counts=None, family=FAMILY):
    return dict(method=design["analysis"]["method"],family=family,primary_family=family_registry(design,family),
        analysis_settings={k:v for k,v in design["analysis"].items() if not k.startswith("coverage_")},
        frozen_scales=design.get("frozen_scales"),analysis_module_sha256=file_hash(__file__),
        complete_episode_counts=complete_counts,
        global_alpha=design["global_alpha"],family_weights=design["family_weights"],
        strata=[dict(id=s["id"],planned_episodes=len(s["reset_seeds"]),calls_per_episode=s["calls_per_episode"],
                     random_permutations=s["random_permutations"],
                     **({"parameter_draws":s["parameter_draws"]} if family==SPECIFICITY_FAMILY else {})) for s in design["strata"]],
        groups=design["groups"],conditioning=design["failure_policy"],
        e05_scope=design.get("e05") if family==SPECIFICITY_FAMILY else None)


def validate_calibration(design,data,calibration,family=FAMILY):
    """Authenticate scope and binomial simulation uncertainty, not just a label."""
    from scipy.stats import beta
    observed_counts={sid:len(item["matrix"]) for sid,item in data.items()}
    if calibration.get("complete_episode_counts")!=observed_counts or calibration.get("count_conditioning")!="exact_joint_complete_episode_counts":
        raise ValueError("Coverage calibration must bind the exact observed complete-episode counts")
    if calibration.get("family",FAMILY)!=family or calibration.get("status") != "approved" or calibration.get("scope_sha256") != object_hash(calibration_scope(design,observed_counts,family)):
        raise ValueError("A matching, independently recorded coverage calibration is required")
    plan=prospective_coverage_plan(design,family)
    if not plan or plan.get("family",FAMILY)!=family or plan.get("kind")!="prospective_exact_count_coverage_plan" or plan.get("schema_version")!=1 or calibration.get("coverage_plan_sha256")!=object_hash(plan):
        raise ValueError("Calibration differs from the immutable prospective coverage plan")
    cases=calibration.get("scenarios",[])
    if not cases or not calibration.get("assumptions") or not calibration.get("generator_source_sha256"):
        raise ValueError("Coverage calibration needs scenarios, distributional assumptions and generator identity")
    digest(calibration["generator_source_sha256"],"coverage generator")
    if calibration["generator_source_sha256"]!=plan["generator_source_sha256"] or plan["analysis_module_sha256"]!=file_hash(__file__):
        raise ValueError("Coverage source identities differ from the prospective plan")
    confidence_alpha=finite(calibration["simulation_family_alpha"],"simulation family alpha",positive=True)
    if not confidence_alpha<1:
        raise ValueError("Invalid simulation confidence alpha")
    look_weights=calibration.get("look_weights",[1.])
    if not look_weights or any(finite(w,"simulation look weight",positive=True)>1 for w in look_weights) or sum(look_weights)>1+1e-12:
        raise ValueError("Simulation look weights exceed their probability budget")
    derivation=plan["stage_derivation"]
    if confidence_alpha!=derivation["simulation_alpha"] or look_weights!=derivation["look_weights"] or derivation["limit"]!=design["global_alpha"]*design["family_weights"][family]:
        raise ValueError("Coverage simulation changed its prospective probability budgets")
    if "e05" in design:
        plans=[prospective_coverage_plan(design,f) for f in inferential_families(design)]
        if any(p is None for p in plans) or sum(p["stage_derivation"]["simulation_alpha"] for p in plans)>design["analysis"].get("coverage_simulation_alpha",.05)+1e-15:
            raise ValueError("Family calibration confidence budgets exceed the prospective study-wide simulation budget")
    expected=plan["regimes"]
    if len(set(expected))!=len(expected) or calibration.get("planned_scenario_ids")!=expected or sorted(expected)!=sorted(c["id"] for c in cases):
        raise ValueError("Coverage simulation omitted a prospectively declared scenario")
    # Bonferroni confidence bounds across prespecified simulation scenarios.
    # They certify only the simulated distributions, never arbitrary tails.
    for case in cases:
        n=integer(case["simulations"],"coverage simulations")
        failures=integer(case["familywise_noncoverage"],"simulation noncoverage",0)
        if failures>n or not case.get("description"):
            raise ValueError("Invalid coverage simulation accounting")
        look=integer(case.get("look_index",0),"simulation look index",0)
        if look>=len(look_weights):
            raise ValueError("Coverage result uses an undeclared sequential look")
        if n!=plan["simulation_stages"][look]:
            raise ValueError("Coverage simulation changed its prospective stage size")
        upper=1. if failures==n else float(beta.ppf(1-confidence_alpha*look_weights[look]/len(cases),failures+1,n-failures))
        if upper>design["global_alpha"]*design["family_weights"][family]:
            raise ValueError("Simulation upper noncoverage bound exceeds the allocated family alpha")


def confirmatory_report(design,data, *, calibration,family=FAMILY):
    if design["stage"] != "confirmatory_locked":
        raise ValueError("Confirmation cannot reuse the variance-only pilot stage")
    settings=design["analysis"]
    if settings["method"] != "stratified_episode_percentile_bootstrap":
        raise ValueError("Unknown interval method")
    contrasts=local_contrasts(family);width=len(contrasts);registry=family_registry(design,family)
    alpha=design["global_alpha"]*design["family_weights"][family]
    required=bootstrap_requirement(alpha,len(registry),settings["min_tail_draws"],settings["tail_relative_mcse"])
    if integer(settings["draws"],"bootstrap draws")<required or integer(settings["mc_repeats"],"endpoint MC repeats",2)<2:
        raise ValueError(f"Global tail resolution requires at least {required} draws and independent endpoint repeats")
    validate_calibration(design,data,calibration,family)
    intervals=[]
    for repeat in range(settings["mc_repeats"]):
        samples=(_bootstrap(design,data,settings["draws"],settings["seed"]+repeat) if family==FAMILY else
                 _bootstrap_family(design,data,settings["draws"],settings["seed"]+repeat,family))
        tail=alpha/(2*len(registry))
        intervals.append(np.quantile(samples,[tail,1-tail],axis=0).T)
    intervals=np.asarray(intervals)
    endpoint_spread=np.ptp(intervals,axis=0).max(axis=1)
    output=[]
    planned_membership={sid:item["membership"] for sid,item in data.items()}
    for g,group in enumerate(design["groups"]):
        estimate=sum(w*data[s]["matrix"].mean(axis=0) for s,w in group["weights"].items())
        non_mc_var=np.zeros(width)
        mc_var=np.zeros(width)
        for sid,w in group["weights"].items():
            item=data[sid]; n=len(item["matrix"])
            episode_mc=np.stack([ep["mc"].sum(axis=0)/len(ep["effects"])**2 if family==FAMILY else ep["parameter_mc_covariance"] for ep in item["episodes"]])
            current_mc=np.diag(episode_mc.mean(axis=0))/n
            total=np.diag(_cov(item["matrix"]))/n
            mc_var+=w*w*current_mc
            non_mc_var+=w*w*np.maximum(total-current_mc,0)
        for local,c in enumerate(contrasts):
            index=g*width+local
            scale=design["frozen_scales"][f"{group['id']}:{c['modality']}"]
            h=.05*finite(scale,"frozen baseline scale",positive=True)
            mc_ok=mc_var[local] <= .01*non_mc_var[local] and mc_var[local] <= .01*h*h
            interval_ok=endpoint_spread[index] <= settings["endpoint_mc_fraction_h"]*h
            output.append(dict(**registry[index], estimate=float(estimate[local]),
                confidence_interval=intervals[0,index].tolist(), marginal_alpha=alpha/len(registry),
                halfwidth_target=h, substantial_difference=.10*scale,
                halfwidth_attained=float(np.diff(intervals[0,index])[0]/2)<=h,
                endpoint_mc_max_spread=float(endpoint_spread[index]), endpoint_mc_gate=bool(interval_ok),
                **({"random_mc_standard_error":float(np.sqrt(mc_var[local])),"random_mc_gate":bool(mc_ok)} if family==FAMILY else
                   {"parameter_mc_standard_error":float(np.sqrt(mc_var[local])),"parameter_mc_gate":bool(mc_ok)}),
                interval_status="calibrated_candidate" if mc_ok and interval_ok else "precision_gate_failed",
                population_sha256=object_hash({s:planned_membership[s] for s in group["weights"]})))
    # Total episode variance already contains finite random-control MC noise.
    # MC is audited, not added a second time to the bootstrap interval.
    return dict(kind="paired_global_confirmation",family=family, primary_results=output, global_alpha=design["global_alpha"],
        mc_source="independent_context_feature_order" if family==FAMILY else "episode_coherent_parameter_draw",
        family_weights=design["family_weights"], nominal_multiplicity="Bonferroni_within_declared_family",
        required_bootstrap_draws=required, bootstrap_settings=settings,
        point_and_ci_membership=planned_membership, membership_sha256=object_hash(planned_membership),
        task_weights={g["id"]:g["weights"] for g in design["groups"]},
        conditioning="joint_complete_episodes_within_each_task_no_task_weight_renormalization",
        failure_reporting={s:dict(planned_episodes=v["planned_episodes"], complete_episodes=len(v["matrix"]),
                                 planned_contexts=v["planned_contexts"], cause_counts=v["failure_counts"]) for s,v in data.items()},
        descriptive_stratum_results={s:dict(estimate=v["matrix"].mean(axis=0).tolist(),
            standard_error=np.sqrt(np.diag(_cov(v["matrix"]))/len(v["matrix"])).tolist(),
            contrast_order=contrasts, inference="descriptive_no_simultaneous_coverage_claim") for s,v in data.items()},
        descriptive_crossed_response_and_component_results={s:v.get("secondary",{}) for s,v in data.items()},
        limitation="finite_effects_conditional_on_joint_completion; unbounded_RMS_has_no_missing_outcome_bias_bound")


def confirmatory_study_report(design,data_by_family,calibrations):
    validate_design(design)
    families=inferential_families(design)
    if set(data_by_family)!=set(families) or set(calibrations)!=set(families):raise ValueError("Every prespecified inferential family must have data and executed coverage evidence")
    reports={f:confirmatory_report(design,data_by_family[f],calibration=calibrations[f],family=f) for f in families}
    overlap={}
    if SPECIFICITY_FAMILY in families:
        for sid in data_by_family[FAMILY]:
            maps={f:{e["episode_id"]:e for e in data_by_family[f][sid]["membership"]} for f in families}
            if set(maps[FAMILY])!=set(maps[SPECIFICITY_FAMILY]):raise ValueError("Family attempted episode populations differ")
            overlap[sid]=dict(both_complete=sum(maps[FAMILY][e]["joint_complete"] and maps[SPECIFICITY_FAMILY][e]["joint_complete"] for e in maps[FAMILY]),
                base_only_complete=sum(maps[FAMILY][e]["joint_complete"] and not maps[SPECIFICITY_FAMILY][e]["joint_complete"] for e in maps[FAMILY]),
                e05_only_complete=sum(not maps[FAMILY][e]["joint_complete"] and maps[SPECIFICITY_FAMILY][e]["joint_complete"] for e in maps[FAMILY]))
    return dict(kind="paired_two_family_confirmation" if len(families)==2 else "paired_single_family_confirmation",
        family_reports=reports,completion_overlap=overlap,global_alpha=design["global_alpha"],family_weights=design["family_weights"],
        coverage_interpretation="separate_family_noncoverage_bounds_sum_by_union_bound; no_cross_family_independence_assumption",
        limitation="family_specific_joint_completion_populations; no_distribution_free_coverage_or_training_causality_claim")


def allocate_episodes(design,pilot,call_choices,costs, *, max_episodes,permutations,family=FAMILY):
    """Greedy measured-cost allocation satisfying every approximate precision constraint.

    This is a feasible integer allocation, not a global-optimum proof. Welch t
    quantiles and empirical upper variance estimates remain planning assumptions.
    Return these assumptions and MC needs; never launch or authorize a study.
    """
    from scipy.stats import t, beta, binom
    integer(max_episodes,"maximum episodes per stratum",2)
    integer(permutations,"random permutations",2)
    contrasts=local_contrasts(family);width=len(contrasts);specificity=family==SPECIFICITY_FAMILY
    unit_field="mc_variance_per_parameter_draw" if specificity else "mc_variance_per_permutation"
    if pilot.get("primary_contrasts") != family_registry(design,family) or pilot.get("stage") != "variance_only_pilot":
        raise ValueError("Pilot contrast order or scientific scope differs from allocation design")
    planning=design["planning"]
    failure_alpha=finite(planning["failure_confidence_alpha"],"failure-bound alpha",positive=True)
    shortfall_alpha=finite(planning["recruitment_shortfall_alpha"],"recruitment-shortfall alpha",positive=True)
    looks=integer(planning["pilot_max_looks"],"maximum pilot looks")
    if max(failure_alpha,shortfall_alpha)>=1:
        raise ValueError("Planning probability budgets must lie in (0,1)")
    selected={}
    for sid in (s["id"] for s in design["strata"]):
        matches=[r for r in pilot["strata"][sid]["frontier"] if r["calls_per_episode"]==call_choices[sid]]
        if len(matches)!=1:
            raise ValueError("Call choice is absent from measured pilot frontier")
        if pilot["strata"][sid]["zero_observed_variance_cells"]:
            raise ValueError("Zero observed pilot variance cannot certify population precision; resolve degeneracy before allocation")
        if call_choices[sid]!=pilot["strata"][sid]["evaluated_calls_per_episode"]:
            raise ValueError("Changed J requires a pilot at that J: failure and conditional-variance populations do not transport automatically")
        observed_m=([pilot["strata"][sid]["evaluated_parameter_draws"]] if specificity else pilot["strata"][sid]["evaluated_random_permutations"])
        if observed_m and observed_m!=[permutations]:
            raise ValueError("Changed R requires matching parameter-draw completion/variance evidence" if specificity else "Changed M requires a pilot at that M: additional masks may alter joint completion; exact controls alone are exempt")
        selected[sid]=matches[0]
    alpha=design["global_alpha"]*design["family_weights"][family]/len(family_registry(design,family))
    counts={sid:2 for sid in selected}
    cost_field="parameter_seconds_per_draw_call" if specificity else "random_seconds_per_order_call"
    costs_per_episode={sid:finite(costs[sid]["collection_seconds"],"collection cost",positive=True)+
        selected[sid]["expected_selected_calls"]*(finite(costs[sid]["fixed_probe_seconds_per_call"],"probe cost",positive=True)+
        permutations*finite(costs[sid][cost_field],"MC cost"))+
        (permutations*finite(costs[sid].get("parameter_seconds_per_draw_episode",0.),"parameter setup cost") if specificity else 0.) for sid in selected}
    if any(x<=0 for x in costs_per_episode.values()):
        raise ValueError("Total measured cost must be positive")
    if any(costs[s][cost_field]<0 or costs[s].get("parameter_seconds_per_draw_episode",0)<0 for s in selected):
        raise ValueError("Random-control cost cannot be negative")

    def constraints(ns):
        results=[]
        for group in design["groups"]:
            terms=[]; mc_terms=[]
            for sid,w in group["weights"].items():
                r=selected[sid]
                terms.append(w*w*np.array(r["non_mc_variance_upper"])/ns[sid])
                mc_terms.append(w*w*np.array(r[unit_field+"_upper"])/ns[sid])
            non_mc=np.sum(terms,axis=0); unit_mc=np.sum(mc_terms,axis=0)
            variance=non_mc+unit_mc/permutations
            all_terms=np.array(terms)+np.array(mc_terms)/permutations
            denom=np.sum([v*v/(ns[sid]-1) for v,sid in zip(all_terms,group["weights"])],axis=0)
            df=np.divide(variance*variance,denom,out=np.full(width,np.inf),where=denom>0)
            q=t.ppf(1-alpha/2,df)
            h=np.array([pilot["scales"][f"{group['id']}:{c['modality']}"]["conservative_planning_halfwidth"] for c in contrasts])
            if np.any(h<=0):
                raise ValueError("No positive lower calibration scale; no epsilon replacement is permitted")
            ratios=q*np.sqrt(variance)/h
            relative_requirement=np.divide(unit_mc,.01*non_mc,out=np.zeros(width),where=non_mc>0)
            relative_requirement[(unit_mc>0)&(non_mc==0)]=np.inf
            needed=np.maximum(relative_requirement,unit_mc/(.01*h*h))
            results.append(dict(group=group["id"], ratios=ratios, required_permutations=needed))
        return results

    while True:
        current=constraints(counts)
        worst=max(float(r["ratios"].max()) for r in current)
        if worst<=1:
            break
        scores=[]
        for sid in counts:
            if counts[sid]>=max_episodes:
                continue
            proposal=dict(counts); proposal[sid]+=1
            new=constraints(proposal)
            # Sum squared violations avoids ignoring the second tied group.
            gain=sum(np.maximum(r["ratios"]**2-1,0).sum() for r in current)-sum(np.maximum(r["ratios"]**2-1,0).sum() for r in new)
            scores.append((gain/costs_per_episode[sid],sid))
        if not scores or max(scores)[0]<=0:
            break
        counts[max(scores)[1]]+=1
    constraints_final=constraints(counts)
    required_m=max(float(r["required_permutations"].max()) for r in constraints_final)
    feasible=all(np.all(r["ratios"]<=1) for r in constraints_final)
    rosters, completion_lower = {}, {}
    for sid,n in counts.items():
        record=pilot["strata"][sid]
        successes=integer(record["complete_episodes"],"pilot complete episodes",0)
        trials=integer(record["planned_episodes"],"pilot planned episodes")
        if successes>trials:
            raise ValueError("Pilot completion accounting is impossible")
        # Simultaneous Clopper-Pearson lower bounds over six strata and the
        # prospectively capped looks. Future quotas use a separate probability
        # budget and an iid completion model, never failed-episode replacement.
        family_count=len(inferential_families(design))
        lower=0. if successes==0 else float(beta.ppf(failure_alpha/(6*looks*family_count),successes,trials-successes+1))
        completion_lower[sid]=lower
        lo,hi=n,max_episodes
        if lower==0 or binom.sf(n-1,hi,lower)<1-shortfall_alpha/(6*family_count):
            rosters[sid]=None; feasible=False
            continue
        while lo<hi:
            mid=(lo+hi)//2
            if binom.sf(n-1,mid,lower)>=1-shortfall_alpha/(6*family_count):
                hi=mid
            else:
                lo=mid+1
        rosters[sid]=lo
    return dict(kind="variance_only_allocation_candidate",family=family, required_complete_episodes=counts,
        planned_episodes=rosters,completion_probability_lower=completion_lower,calls_per_episode=call_choices,
        **({"parameter_draws_evaluated":permutations,"required_parameter_draws":math.ceil(required_m) if math.isfinite(required_m) else None} if specificity else
           {"permutations_evaluated":permutations,"required_permutations":math.ceil(required_m) if math.isfinite(required_m) else None}),
        precision_constraints_met=bool(feasible), random_mc_constraints_met=bool(required_m<=permutations),
        approximate_cost_seconds=sum(rosters[s]*costs_per_episode[s] for s in counts) if all(rosters.values()) else None,
        measured_cost_seconds_per_episode=costs_per_episode,
        maximum_halfwidth_to_target_ratio=max(float(r["ratios"].max()) for r in constraints_final),
        interpretation="candidate_only; empirical_upper_variance_Welch_t_and_conditional_completion_require_calibration",
        failure_planning=dict(confidence_alpha=failure_alpha,shortfall_alpha=shortfall_alpha,pilot_max_looks=looks,inferential_family_count=len(inferential_families(design)),
                              assumption="independent_identically_distributed_episode_completion_within_stratum; no_replacement"),
        optimization="greedy_integer_cost_allocation_not_global_optimum", no_signed_efficacy_estimates=True)


def allocate_two_family_episodes(design,pilots,call_choices,costs, *, max_episodes,permutations,parameter_draws):
    validate_design(design)
    if set(pilots)!={FAMILY,SPECIFICITY_FAMILY} or set(costs)!=set(pilots):raise ValueError("Both families need matching pilot and measured cost evidence")
    budgets={FAMILY:permutations,SPECIFICITY_FAMILY:parameter_draws}
    results={f:allocate_episodes(design,pilots[f],call_choices,costs[f],max_episodes=max_episodes,permutations=budgets[f],family=f) for f in budgets}
    rosters={s:None if any(results[f]["planned_episodes"][s] is None for f in budgets) else max(results[f]["planned_episodes"][s] for f in budgets) for s in call_choices}
    per_episode={}
    for s in call_choices:
        if costs[FAMILY][s]["collection_seconds"]!=costs[SPECIFICITY_FAMILY][s]["collection_seconds"]:raise ValueError("Shared collector cost differs across families")
        per_episode[s]=sum(results[f]["measured_cost_seconds_per_episode"][s] for f in budgets)-costs[FAMILY][s]["collection_seconds"]
    return dict(kind="two_family_fixed_roster_planning_candidate",family_allocations=results,planned_episodes=rosters,
        approximate_cost_seconds=sum(rosters[s]*per_episode[s] for s in rosters) if all(rosters.values()) else None,
        precision_constraints_met=all(r["precision_constraints_met"] and r["random_mc_constraints_met"] for r in results.values()),
        optimization="maximum_of_two_feasible_family_rosters; conservative_not_global_cost_optimum",
        no_signed_efficacy_estimates=True,interpretation="empirical_pilot_bounds_and_conditional_completion_assumptions_require_separate_validation")


def normal_reference_pilot_stages(sd_upper_ratios,alpha_spending,variance_cells=180):
    """Derive stage sizes from an explicit normal-reference SD precision target.

    These counts are not valid SD guarantees for skewed/heavy-tailed episodes.
    They are a reproducible starting schedule for whole-rule simulation, not
    an automatic pilot authorization or imported conventional sample size.
    """
    from scipy.stats import chi2
    if len(sd_upper_ratios)!=len(alpha_spending) or not sd_upper_ratios:
        raise ValueError("Every pilot stage needs a ratio and probability allocation")
    integer(variance_cells,"variance cells")
    if not 0<sum(alpha_spending)<1:
        raise ValueError("Invalid pilot alpha-spending sum")
    stages=[]
    previous=2
    for ratio,alpha in zip(sd_upper_ratios,alpha_spending):
        if finite(ratio,"SD upper ratio")<=1 or not 0<finite(alpha,"stage alpha")<1:
            raise ValueError("SD ratio must exceed one and alpha must lie in (0,1)")
        n=previous
        while math.sqrt((n-1)/chi2.ppf(alpha/variance_cells,n-1))>ratio:
            n+=1
            if n>10**7:
                raise ValueError("Reference precision target implies an infeasible pilot schedule")
        stages.append(dict(complete_episodes_per_stratum=n,sd_upper_to_observed_target=ratio,
                           stage_alpha=alpha,reference_distribution="normal_independent_episode_estimates"))
        previous=n
    return stages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("pilot","confirm"))
    for name in ("design","registry","out"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--design-sha256",required=True)
    parser.add_argument("--registry-sha256",required=True)
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--calibration-sha256")
    args=parser.parse_args()
    for path,expected in ((args.design,args.design_sha256),(args.registry,args.registry_sha256)):
        if file_hash(path)!=digest(expected,str(path)):
            raise ValueError("Trusted input digest mismatch")
    design=validate_design(strict_json(args.design)); registry=strict_json(args.registry)
    data=load_study(design,registry,args.design_sha256)
    specificity=load_specificity_study(design,registry,args.design_sha256,data) if "e05" in design else None
    if args.command=="pilot":
        report=pilot_report(design,data,**design["pilot_analysis"])
        if specificity is not None:
            second=specificity_pilot_report(design,specificity,data,**design["e05_pilot_analysis"])
            second["scales"]=deepcopy(report["scales"])
            report=dict(kind="two_family_variance_only_pilot",family_reports={FAMILY:report,SPECIFICITY_FAMILY:second},
                disclosure="labeled_variance_only_not_blinded",no_signed_efficacy_estimates=True)
    else:
        if specificity is not None:
            if args.calibration is not None or args.calibration_sha256 is not None:raise ValueError("Expanded confirmation uses both sealed family calibration entries, not a single artifact")
            entries=registry["coverage_calibrations"]
            if set(entries)!=set(inferential_families(design)):raise ValueError("Missing a separately executed family calibration")
            calibrations={}
            for family,entry in entries.items():
                if file_hash(entry["path"])!=digest(entry["sha256"],"family calibration"):raise ValueError("Family calibration artifact differs from its sealed registry")
                value=strict_json(entry["path"])
                if value.get("protocol_sha256")!=digest(entry["protocol_sha256"],"family coverage protocol"):raise ValueError("Family calibration protocol differs from its sealed registry")
                calibrations[family]=value
            report=confirmatory_study_report(design,{FAMILY:data,SPECIFICITY_FAMILY:specificity},calibrations)
        else:
            if args.calibration is None or file_hash(args.calibration)!=args.calibration_sha256 or args.calibration_sha256!=registry.get("coverage_calibration_sha256"):
                raise ValueError("Coverage calibration must match the independently sealed post-run registry")
            calibration=strict_json(args.calibration)
            if calibration.get("protocol_sha256")!=registry.get("coverage_protocol_sha256"):
                raise ValueError("Exact-count calibration protocol differs from the sealed registry")
            report=confirmatory_report(design,data,calibration=calibration)
    report.update(global_design_sha256=args.design_sha256,registry_sha256=args.registry_sha256,
                  pilot_stage=registry.get("pilot_stage"),analysis_module_sha256=file_hash(__file__))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(canonical_json(report)+b"\n")


if __name__=="__main__":
    main()
