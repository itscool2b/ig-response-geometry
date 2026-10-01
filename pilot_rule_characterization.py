"""Matched full-rule characterization, separate from the frozen v1 procedure.

One union-candidate pilot panel and one independent confirmation panel are
shared by every declared rule. Internal bound failures remain diagnostics.
Direct planning success cannot replace exact-count interval coverage approval.
"""
from __future__ import annotations

import time

import paired_study_analysis as study
import pilot_selection as selector
import pilot_selection_simulation as sim
from experiment_io import object_hash


DIRECT_COMPONENTS=("future_shortfall","future_precision_failure",
    "future_population_precision_failure","future_mc_failure","future_endpoint_mc_failure")
INTERNAL_DIAGNOSTICS=("inspected_variance_bound_failure","scale_bound_failure",
    "selected_variance_bound_failure","completion_bound_failure")


def union_candidates(rules):
    union={}
    for rule in rules:
        for candidate in rule["protocol"]["candidates"]:
            cid=candidate["id"]
            if cid in union and union[cid]!=candidate:
                raise ValueError("Shared candidate ID has different settings")
            union[cid]=candidate
    return [union[cid] for cid in sorted(union)]


def assess_rule(design,protocol,full,endpoints,known):
    """Execute the unchanged v1 stage/selection rule against a shared panel."""
    inspected=[];scale_failures=[];history=[];selected=None
    candidates=protocol["candidates"]
    subset={c["id"]:full[c["id"]] for c in candidates}
    for stage in protocol["stages"]:
        data,bank=sim.stage_prefix(subset,endpoints,stage["episodes_per_stratum"])
        result=selector.evaluate_stage(design,protocol,stage["id"],data,bank)
        failures=sim.inspected_bound_failures(design,result,known);inspected.extend(failures)
        for group in design["groups"]:
            population=sum(w*known[candidates[0]["id"]][study.FAMILY][s]["baseline_median"] for s,w in group["weights"].items())
            for modality in study.MODALITIES:
                cell=result["scale_report"]["scales"][f"{group['id']}:{modality}"]
                if cell.get("lower") is not None and population<cell["lower"]-1e-14:
                    scale_failures.append(f"{stage['id']}:{group['id']}:{modality}:lower")
                if cell.get("upper") is not None and population>cell["upper"]+1e-14:
                    scale_failures.append(f"{stage['id']}:{group['id']}:{modality}:upper")
        history.append(dict(stage_id=stage["id"],status=result["status"],selected_candidate=result["selected_candidate"],
            inspected_bound_failure=bool(failures),scale_reasons=result["scale_reasons"],
            candidate_reasons={c["candidate"]["id"]:c["reasons"] for c in result["candidates"]}))
        if result["selected_candidate"] is not None:selected=result;break
    row=dict(stages=history,selected=selected is not None,selected_candidate=None,
        inspected_variance_bound_failure=bool(inspected),scale_bound_failure=bool(scale_failures),
        inspected_bound_failures=inspected,scale_bound_failures=scale_failures)
    return row,selected


def simulate_matched_trial(design,rules,regime,seed,trial):
    """Two or more rules, one genuinely shared latent pilot/future population."""
    start=time.perf_counter()
    for rule in rules:selector.validate_protocol(rule["protocol"],design)
    candidates=union_candidates(rules)
    counts=design["pilot_stages"][-1]["episodes_per_stratum"]
    full,endpoints=sim.generate_panel(design,regime,candidates,counts,seed=seed,trial=trial,stream="pilot")
    known={c["id"]:sim.truth(design,regime,c) for c in candidates}
    panel_hash=object_hash(endpoints);decisions={};outputs={}
    for rule in rules:
        before=time.perf_counter()
        row,selected=assess_rule(design,rule["protocol"],full,endpoints,known)
        row["selection_seconds"]=time.perf_counter()-before
        outputs[rule["id"]]=row;decisions[rule["id"]]=selected
    chosen=[v for v in decisions.values() if v is not None]
    future=None;future_bank=None
    if chosen:
        # A single maximum roster is generated, then each rule uses its own
        # preselected attempted-episode prefix and conditional population.
        max_counts={s["id"]:max(v["selected_allocation"]["planned_episodes"][s["id"]] for v in chosen) for s in design["strata"]}
        selected_ids={v["selected_candidate"] for v in chosen}
        selected_candidates=[c for c in candidates if c["id"] in selected_ids]
        future,future_bank=sim.generate_panel(design,regime,selected_candidates,max_counts,seed=seed,trial=trial,stream="confirmation")
    for rule in rules:
        row=outputs[rule["id"]];selected=decisions[rule["id"]]
        if selected is None:
            row.update(status="inconclusive",selected_variance_bound_failure=False,completion_bound_failure=False,
                future_interval_noncoverage=False,**{key:False for key in DIRECT_COMPONENTS})
        else:
            cid=selected["selected_candidate"];candidate=next(c for c in candidates if c["id"]==cid)
            allocation=selected["selected_allocation"]
            data,_=sim.stage_prefix({cid:future[cid]},future_bank,allocation["planned_episodes"])
            before=time.perf_counter()
            diagnostics=sim.validate_future(design,candidate,allocation,selected["scale_report"]["scales"],data[cid],known[cid])
            row.update(status="selected",selected_candidate=cid,selected_stage=selected["stage_id"],
                planned_confirmation_episodes=allocation["planned_episodes"],
                assumed_confirmation_cost_seconds=allocation["approximate_cost_seconds"],
                future_seconds=time.perf_counter()-before,future=diagnostics,
                selected_variance_bound_failure=any(f["candidate"]==cid for f in sim.inspected_bound_failures(design,selected,known)),
                completion_bound_failure=any(record["completion_probability_lower"][sid]>known[cid][family][sid]["completion"]+1e-14
                    for family,record in allocation["family_allocations"].items() for sid in record["completion_probability_lower"]),
                future_shortfall=any(r["shortfall"] for r in diagnostics.values()),
                future_precision_failure=any(r["halfwidth_failure"] for r in diagnostics.values()),
                future_population_precision_failure=any(r["population_halfwidth_failure"] for r in diagnostics.values()),
                future_mc_failure=any(r["true_mc_failure"] or r["empirical_mc_failure"] for r in diagnostics.values()),
                future_endpoint_mc_failure=any(r["endpoint_mc_failure"] for r in diagnostics.values()),
                future_interval_noncoverage=any(r["interval_noncoverage"] for r in diagnostics.values()))
        row["direct_unsafe_selection"]=bool(row["selected"] and any(row[k] for k in DIRECT_COMPONENTS))
        row["legacy_internal_or_direct_failure"]=bool(row["selected"] and
            (row["direct_unsafe_selection"] or any(row[k] for k in
                ("selected_variance_bound_failure","scale_bound_failure","completion_bound_failure"))))
        row["undercoverage_is_not_precision_success"]=True
    return dict(kind="matched_pilot_rule_trial_v2",regime=regime,trial=trial,
        seed_sha256=object_hash(dict(seed=seed,regime=regime,trial=trial)),rules=outputs,
        pilot_endpoint_panel_sha256=panel_hash,
        confirmation_endpoint_panel_sha256=None if future_bank is None else object_hash(future_bank),
        shared_panel_contract="single_union_candidate_pilot_and_independent_shared_confirmation_roster_prefixes",
        elapsed_seconds=time.perf_counter()-start,no_real_efficacy_data=True,
        exact_count_interval_coverage_approval_required=True)
