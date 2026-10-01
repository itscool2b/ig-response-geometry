"""Frozen, variance-only pilot selection; empirical variance bounds are not theorems.

This new interface deliberately leaves the separately frozen v4 inference code
unchanged. All candidates are evaluated at each completed stage. Selection uses
precision, completion and recorded costs, never signed comparative effects.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

import paired_study_analysis as study
from experiment_io import object_hash


RULE="all_candidates_each_stage_first_feasible_minimum_cost_then_id_v1"


def validate_protocol(protocol,design):
    study.validate_design(design)
    if design["stage"]!="variance_only_pilot" or "e05" not in design:
        raise ValueError("The pilot selector requires the expanded variance-only design")
    if protocol.get("kind")!="paired_pilot_selection_protocol" or protocol.get("schema_version")!=1 or protocol.get("rule")!=RULE:
        raise ValueError("Unknown pilot selection rule")
    stages=design.get("pilot_stages",[])
    if not stages or protocol.get("stages")!=stages:
        raise ValueError("The entire stage and stop schedule must be frozen in the design")
    candidates=protocol.get("candidates",[])
    if not candidates or len({c["id"] for c in candidates})!=len(candidates):
        raise ValueError("A finite uniquely named candidate registry is required")
    settings=[]
    for c in candidates:
        if set(c)!={"id","calls_per_episode","random_permutations","parameter_draws"} or not c["id"]:
            raise ValueError("Candidate settings must be explicit")
        for field,minimum in (("calls_per_episode",1),("random_permutations",2),("parameter_draws",2)):
            study.integer(c[field],field,minimum)
        settings.append(tuple(c[k] for k in ("calls_per_episode","random_permutations","parameter_draws")))
    if len(set(settings))!=len(settings):raise ValueError("Duplicate settings are not distinct candidates")
    look_weights=protocol["scale_look_weights"]
    if len(look_weights)!=len(stages) or any(not 0<w<=1 for w in look_weights) or sum(look_weights)>1+1e-12:
        raise ValueError("Scale spending must cover every declared look")
    for key in ("scale_alpha","failure_confidence_alpha","recruitment_shortfall_alpha"):
        if not 0<study.finite(protocol[key],key)<1:raise ValueError("Invalid planning probability budget")
    bounds=protocol["bounds"]
    if bounds.get("method")!="empirical_episode_percentile_with_direct_total_guard_v1":
        raise ValueError("Only the explicit empirical bound procedure is implemented")
    q=study.finite(bounds["upper_quantile"],"upper quantile")
    if not .5<q<1:raise ValueError("Upper quantile must exceed one half")
    draws=study.integer(bounds["draws"],"pilot bootstrap draws",2)
    minimum=study.integer(bounds["minimum_expected_tail_draws"],"pilot tail count",100)
    if draws*(1-q)<minimum-1e-10:raise ValueError("Pilot bound bootstrap has insufficient computational tail resolution")
    study.integer(bounds["seed"],"pilot bootstrap seed",0)
    if study.finite(protocol["maximum_sd_upper_to_point_ratio"],"SD precision target")<=1:
        raise ValueError("A finite variance-estimation precision target is required")
    if not 0<study.finite(protocol["maximum_scale_relative_width"],"scale precision target")<1:
        raise ValueError("A finite scale precision target is required")
    study.integer(protocol["maximum_confirmation_episodes_per_stratum"],"confirmation cap",2)
    if set(protocol["costs"])!={c["id"] for c in candidates}:raise ValueError("Every candidate needs measured or explicitly synthetic costs")
    if protocol.get("signed_means_allowed") is not False:raise ValueError("Signed pilot efficacy is outside the selection rule")
    return protocol


def candidate_design(design,candidate):
    value=deepcopy(design)
    for s in value["strata"]:
        for field in ("calls_per_episode","random_permutations","parameter_draws"):s[field]=candidate[field]
    return value


def _episode_mc(data,family):
    if family==study.SPECIFICITY_FAMILY:
        return np.stack([np.diag(e["parameter_mc_covariance"]) for e in data["episodes"]])
    return np.stack([np.diag(np.sum(e["mc"],axis=0))/len(e["effects"])**2 for e in data["episodes"]])


def moment_bounds(data,family, *, budget,draws,seed,upper_quantile):
    """Fast episode bootstrap of total, non-MC and MC variance at evaluated settings.

    Negative non-MC moments are exposed and truncated for allocation only.
    These are empirical percentile bounds. An unseen tail is not ruled out.
    Between/call components remain diagnostic when a sampled call alone cannot
    identify them. There is no transport to another J, M or R.
    """
    matrix=np.asarray(data["matrix"],float);n,width=matrix.shape
    if n<2 or not np.isfinite(matrix).all():raise ValueError("At least two finite independent episode vectors are required")
    mc=_episode_mc(data,family)
    if mc.shape!=(n,width) or not np.isfinite(mc).all() or np.any(mc<0):raise ValueError("Invalid episode Monte Carlo variance")
    centered=matrix-matrix.mean(axis=0);observed=np.var(centered,axis=0,ddof=1)
    mc_point=mc.mean(axis=0);non_mc_raw=observed-mc_point
    rng=np.random.Generator(np.random.PCG64(seed))
    sampled_total=np.empty((draws,width));sampled_mc=np.empty_like(sampled_total)
    for first in range(0,draws,256):
        last=min(first+256,draws);weights=rng.multinomial(n,np.full(n,1/n),size=last-first)
        means=weights@centered/n
        sampled_total[first:last]=np.maximum((weights@(centered*centered)-n*means*means)/(n-1),0)
        sampled_mc[first:last]=weights@mc/n
    sampled_non_mc=sampled_total-sampled_mc
    total_upper=np.maximum(observed,np.quantile(sampled_total,upper_quantile,axis=0))
    non_mc_upper=np.maximum(np.maximum(non_mc_raw,0),np.quantile(sampled_non_mc,upper_quantile,axis=0))
    non_mc_lower=np.maximum(0,np.minimum(non_mc_raw,np.quantile(sampled_non_mc,1-upper_quantile,axis=0)))
    mc_upper=np.maximum(mc_point,np.quantile(sampled_mc,upper_quantile,axis=0))
    # The existing allocator accepts non-MC plus per-draw MC terms. Requiring
    # that their sum also dominate the direct total bound prevents an unstable
    # decomposition from making total precision look easier.
    allocation_non_mc=np.maximum(non_mc_upper,total_upper-mc_upper)
    identifiable=all(len(e["effects"])>=2 or e["total_calls"]==1 for e in data["episodes"])
    component_diagnostics=dict(identified=identifiable)
    if identifiable:
        components=(study.specificity_variance_components if family==study.SPECIFICITY_FAMILY else study.variance_components)(data)
        component_diagnostics.update(between=components["between"].tolist(),
            between_negative_cells=np.flatnonzero(components["between_unclipped"]<0).tolist(),
            within_negative_moment_counts=components["within_negative_moment_counts"].tolist())
    ratio=float(np.max(np.sqrt(np.divide(total_upper,observed,out=np.full(width,np.inf),where=observed>0))))
    return dict(total_point=observed.tolist(),total_upper=total_upper.tolist(),
        non_mc_point=np.maximum(non_mc_raw,0).tolist(),non_mc_lower=non_mc_lower.tolist(),non_mc_upper=non_mc_upper.tolist(),
        allocation_non_mc_upper=allocation_non_mc.tolist(),mc_point=mc_point.tolist(),mc_upper=mc_upper.tolist(),
        mc_per_draw_upper=(budget*mc_upper).tolist(),negative_non_mc_cells=np.flatnonzero(non_mc_raw<0).tolist(),
        zero_total_variance_cells=np.flatnonzero(observed==0).tolist(),
        maximum_sd_upper_to_point_ratio=ratio if math.isfinite(ratio) else None,
        component_diagnostics=component_diagnostics,
        interpretation="empirical_percentile_bounds_require_full_rule_domain_specific_validation")


def _relative_mc_pass(design,reports,allocation):
    """Guard every complete-count vector between family quotas and fixed roster.

    Componentwise count increases need not preserve an MC/non-MC ratio. For
    each contrast, its inequality is a sum a_s/n_s <= 0. The rectangular worst
    case uses the minimum count for positive a_s and maximum for negative a_s.
    Absolute MC variance is maximized at the minimum counts in every stratum.
    """
    diagnostics={};passed=True
    rosters=allocation["planned_episodes"]
    if any(n is None for n in rosters.values()):return False,dict(unresolved="no_finite_fixed_roster")
    for family,report in reports.items():
        ns=allocation["family_allocations"][family]["required_complete_episodes"]
        for group in design["groups"]:
            upper=[];excess=[]
            for s,w in group["weights"].items():
                if rosters[s]<ns[s]:raise ValueError("Fixed roster is smaller than its required complete quota")
                bound=report["strata"][s]["moment_bounds"]
                a=w*w*(np.array(bound["mc_upper"])-.01*np.array(bound["non_mc_lower"]))
                excess.append(a/np.where(a>=0,ns[s],rosters[s]))
                upper.append(w*w*np.array(bound["mc_upper"])/ns[s])
            worst=np.sum(excess,axis=0);absolute=np.sum(upper,axis=0)
            h=np.array([report["scales"][f"{group['id']}:{c['modality']}"]["conservative_planning_halfwidth"] for c in study.local_contrasts(family)])
            ok=(worst<=0)&(absolute<=.01*h*h)
            diagnostics[f"{family}:{group['id']}"]=dict(passed=bool(ok.all()),failed_cells=np.flatnonzero(~ok).tolist(),
                worst_case_relative_variance_excess=worst.tolist(),absolute_mc_variance_upper=absolute.tolist(),
                count_domain={s:dict(minimum_complete=ns[s],maximum_complete=rosters[s]) for s in group["weights"]},
                assumption="actual_family_complete_counts_meet_quota; quota_shortfall_is_separately_reported")
            passed=passed and bool(ok.all())
    return passed,diagnostics


def assess_candidate(design,protocol,candidate,data_by_family,scales,stage_index):
    value=candidate_design(design,candidate);family_count=len(study.inferential_families(value));candidate_count=len(protocol["candidates"])
    # Completion bounds spend over ALL inspected candidates and looks, even
    # though the eventual confirmation uses only one candidate.
    value["planning"]=dict(failure_confidence_alpha=protocol["failure_confidence_alpha"]/candidate_count,
        recruitment_shortfall_alpha=protocol["recruitment_shortfall_alpha"],pilot_max_looks=len(protocol["stages"]))
    reports={};reasons=[]
    for family in study.inferential_families(value):
        budget=candidate["parameter_draws" if family==study.SPECIFICITY_FAMILY else "random_permutations"]
        unit="mc_variance_per_parameter_draw" if family==study.SPECIFICITY_FAMILY else "mc_variance_per_permutation"
        report=dict(stage="variance_only_pilot",family=family,primary_contrasts=study.family_registry(value,family),scales=scales,strata={})
        for si,spec in enumerate(value["strata"]):
            sid=spec["id"];data=data_by_family[family][sid]
            if data["evaluated_calls_per_episode"]!=candidate["calls_per_episode"] or (family==study.SPECIFICITY_FAMILY and data["evaluated_parameter_draws"]!=budget) or (family==study.FAMILY and data["evaluated_random_permutations"] not in ([],[budget])):
                raise ValueError("A candidate borrowed a different J/M/R completion population")
            if len(data["matrix"])<2:
                reasons.append(f"{family}:{sid}:insufficient_complete_episodes");continue
            seed=int(object_hash(dict(seed=protocol["bounds"]["seed"],stage=stage_index,candidate=candidate["id"],family=family,stratum=sid)),16)
            b=moment_bounds(data,family,budget=budget,draws=protocol["bounds"]["draws"],seed=seed,upper_quantile=protocol["bounds"]["upper_quantile"])
            if b["zero_total_variance_cells"]:reasons.append(f"{family}:{sid}:unresolved_zero_observed_variance")
            if b["maximum_sd_upper_to_point_ratio"] is None or b["maximum_sd_upper_to_point_ratio"]>protocol["maximum_sd_upper_to_point_ratio"]:reasons.append(f"{family}:{sid}:variance_bound_too_wide")
            mc_cells=range(12) if family==study.SPECIFICITY_FAMILY else [i for i,c in enumerate(study.LOCAL_CONTRASTS) if c["control"]=="random" and c["modality"]!="state"]
            if any(b["mc_point"][i]==0 for i in mc_cells):reasons.append(f"{family}:{sid}:unresolved_sampled_mc_degeneracy")
            expected_calls=np.mean([m["selected_calls"] for m in data["membership"]])
            report["strata"][sid]=dict(planned_episodes=data["planned_episodes"],complete_episodes=len(data["matrix"]),
                evaluated_calls_per_episode=candidate["calls_per_episode"],zero_observed_variance_cells=b["zero_total_variance_cells"],
                **({"evaluated_parameter_draws":budget} if family==study.SPECIFICITY_FAMILY else {"evaluated_random_permutations":data["evaluated_random_permutations"]}),
                moment_bounds=b,membership_sha256=object_hash(data["membership"]),
                frontier=[dict(calls_per_episode=candidate["calls_per_episode"],expected_selected_calls=float(expected_calls),
                    non_mc_variance_upper=b["allocation_non_mc_upper"],**{unit+"_upper":b["mc_per_draw_upper"]})])
        reports[family]=report
    result=dict(candidate=deepcopy(candidate),reasons=reasons,family_reports=reports,feasible=False,
        no_signed_efficacy_estimates=True,completion_bound_candidate_count=candidate_count)
    if reasons:return result
    allocation=study.allocate_two_family_episodes(value,reports,{s["id"]:candidate["calls_per_episode"] for s in value["strata"]},
        protocol["costs"][candidate["id"]],max_episodes=protocol["maximum_confirmation_episodes_per_stratum"],
        permutations=candidate["random_permutations"],parameter_draws=candidate["parameter_draws"])
    relative_ok,diagnostics=_relative_mc_pass(value,reports,allocation)
    result.update(allocation=allocation,mc_diagnostics=diagnostics,feasible=allocation["precision_constraints_met"] and relative_ok)
    if not allocation["precision_constraints_met"]:reasons.append("no_fixed_roster_meets_all_constraints_within_cap")
    if not relative_ok:reasons.append("mc_upper_exceeds_lower_non_mc_or_absolute_precision_target")
    return result


def evaluate_stage(design,protocol,stage_id,candidate_data,endpoint_bank):
    """One complete stage. A caller must stop at its first selected stage."""
    from pilot_scale_bounds import simultaneous_scale_report
    validate_protocol(protocol,design)
    ids=[s["id"] for s in protocol["stages"]]
    if stage_id not in ids:raise ValueError("Unknown prospectively registered stage")
    stage_index=ids.index(stage_id)
    if set(candidate_data)!={c["id"] for c in protocol["candidates"]}:raise ValueError("All frozen candidates must be evaluated; no favorable subset")
    stage=protocol["stages"][stage_index]
    for candidate in protocol["candidates"]:
        data=candidate_data[candidate["id"]]
        if set(data)!=set(study.inferential_families(design)):raise ValueError("Both conditional families must be present")
        for spec in design["strata"]:
            sid=spec["id"];expected=spec["reset_seeds"][:stage["episodes_per_stratum"][sid]]
            endpoint_members={e["reset_seed"]:(e["episode_id"],e["terminal_policy_calls"]) for e in endpoint_bank["strata"][sid]["episodes"]}
            for family in data:
                item=data[family][sid];members=item["membership"]
                if len(members)!=len(expected) or sorted(m["reset_seed"] for m in members)!=sorted(expected):
                    raise ValueError("Candidate pilot data do not match the complete locked stage roster")
                if item["planned_episodes"]!=len(expected):raise ValueError("Candidate attempted-episode count changed")
                for member in members:
                    if endpoint_members.get(member["reset_seed"])!=(member["episode_id"],member["executed_calls"]):
                        raise ValueError("Candidate and all-call scale populations differ")
                    if member["selected_calls"]!=min(candidate["calls_per_episode"],member["executed_calls"]):
                        raise ValueError("Candidate call budget differs from its evaluated membership")
                complete=[m["episode_id"] for m in members if m["joint_complete"]]
                actual=[e["episode_id"] for e in item["episodes"]]
                if len(set(complete))!=len(complete) or actual!=complete or len(item["episodes"])!=len(item["matrix"]):
                    raise ValueError("Candidate conditional episode vectors differ from their complete population")
                width=len(study.local_contrasts(family))
                if np.asarray(item["matrix"]).shape!=(len(complete),width):
                    raise ValueError("Candidate episode matrix has the wrong contrast shape")
                by_id={m["episode_id"]:m for m in members}
                for row,episode in zip(item["matrix"],item["episodes"]):
                    member=by_id[episode["episode_id"]];effects=np.asarray(episode["effects"],float)
                    if member["selected_calls"]<1 or effects.shape!=(member["selected_calls"],width) or episode["total_calls"]!=member["executed_calls"]:
                        raise ValueError("Candidate episode call counts or effect shape differ from membership")
                    expected_row=effects.mean(axis=0)
                    error=32*np.finfo(float).eps*np.abs(effects).mean(axis=0)
                    if not np.isfinite(effects).all() or not np.isfinite(expected_row).all() or not np.isfinite(row).all() or np.any(np.abs(row-expected_row)>error):
                        raise ValueError("Candidate episode matrix row differs from its ordered effect means")
    scale_report=simultaneous_scale_report(design,endpoint_bank,stage_id,protocol["scale_look_weights"],protocol["scale_alpha"])
    scales=scale_report["scales"];scale_reasons=[]
    for name,cell in scales.items():
        if cell.get("lower") is None or cell["lower"]<=0 or cell.get("upper") is None or cell.get("value") is None:
            scale_reasons.append(name+":unresolved_scale_support")
        elif (cell["upper"]-cell["lower"])/cell["value"]>protocol["maximum_scale_relative_width"]:
            scale_reasons.append(name+":scale_bound_too_wide")
    candidates=[]
    if not scale_reasons:
        for candidate in protocol["candidates"]:
            candidates.append(assess_candidate(design,protocol,candidate,candidate_data[candidate["id"]],scales,stage_index))
    feasible=[c for c in candidates if c["feasible"]]
    winner=min(feasible,key=lambda c:(c["allocation"]["approximate_cost_seconds"],c["candidate"]["id"])) if feasible else None
    return dict(kind="pilot_selection_stage",stage_id=stage_id,stage_index=stage_index,scale_report=scale_report,
        scale_reasons=scale_reasons,candidates=candidates,selected_candidate=None if winner is None else winner["candidate"]["id"],
        selected_allocation=None if winner is None else winner["allocation"],
        status="selected" if winner else "inconclusive" if stage_index==len(ids)-1 else "continue_to_next_frozen_stage",
        no_signed_efficacy_estimates=True,rule=RULE,
        interpretation="selection_requires_independent_full_rule_validation; variance_bounds_are_domain_and_simulation_limited")
