"""Independent synthetic panels and known planning truth, never RDT results.

Candidate prefixes share episodes, selected-call permutations and Monte Carlo
streams. Each candidate nevertheless retains its own completion population.
Analytic variances concern the finite synthetic domain declared below.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

import paired_study_analysis as study
from experiment_io import object_hash


REGIMES={
    "gaussian":dict(noise="normal",description="Correlated finite-variance episode and call effects."),
    "skewed_gamma":dict(noise="gamma",description="Centered Gamma shape .5, finite fourth moment and pronounced skew."),
    "student_t3":dict(noise="t3",description="Finite variance, infinite fourth moment; empirical variance bounds need not work."),
    "heteroskedastic_tasks":dict(noise="normal",description="Six different task noise multipliers and trajectory length mixtures."),
    "rare_episode_tail":dict(noise="rare",description="Episode/call noise is 99.9% N(0,1), .1% N(0,100^2), standardized to known unit variance."),
    "rare_parameter_mc":dict(noise="normal",description="A parameter draw has a shared zero-or-rare signed shock; unseen parameter-tail stress."),
    "dominant_mc":dict(noise="normal",description="Large shared parameter/random-control noise with small non-MC variance."),
    "quality_completion":dict(noise="normal",description="Quality shifts effects, noise scale and J/M/R-dependent completion; conditional truth changes with settings."),
    "length_completion":dict(noise="normal",description="Quality-dependent failure compounds over calls and random/parameter draws; long episodes are selected differently."),
    "near_degenerate":dict(noise="normal",description="Algebra-preserving Q/L2 near equality in base contrasts and tiny E05 target variance."),
    "zero_scale":dict(noise="normal",description="Every baseline endpoint is zero; a positive benchmark scale is unavailable."),
    "missing_endpoint":dict(noise="normal",description="An explicitly failed all-call baseline endpoint blocks the unchanged scale population."),
    "collection_failure":dict(noise="normal",description="An iid technical collection failure leaves an attempted episode with zero observed calls, explicit failure metadata and unresolved scales.")}


def seed_for(seed,*parts):return int(object_hash(dict(seed=seed,parts=list(parts))),16)


def noise(rng,shape,kind):
    if kind=="normal":return rng.normal(size=shape)
    if kind=="gamma":return (rng.gamma(.5,size=shape)-.5)/math.sqrt(.5)
    if kind=="t3":return rng.standard_t(3,size=shape)/math.sqrt(3)
    if kind=="rare":return rng.normal(size=shape)*np.where(rng.random(shape)<.001,100.,1.)/math.sqrt(10.999)
    raise ValueError("Unknown synthetic noise")


def base_loadings(regime):
    """Thirty contrasts retain the exact five-comparison algebra."""
    matrix=[]
    method={"Q_IG":0,"L2_IG":1,"input_difference":2,"random":3}
    for c in study.LOCAL_CONTRASTS:
        row=np.zeros((3,2,4));m=study.MODALITIES.index(c["modality"]);d=study.DIRECTIONS.index(c["direction"])
        sign=1 if c["direction"]=="deletion" else -1
        row[m,d,method[c["method"]]]=sign;row[m,d,method[c["control"]]]=-sign
        matrix.append(row.reshape(-1))
    contrast=np.array(matrix);transform=np.eye(24)
    if regime=="near_degenerate":
        for m in range(3):
            for d in range(2):
                q=(m*2+d)*4;transform[q+1]=transform[q]+1e-7*transform[q+1]
    intrinsic=contrast@transform
    common=intrinsic@np.tile(np.array([1.,.8,-.4,.2]),6)
    random_columns=[(m*2+d)*4+3 for m in range(3) for d in range(2)]
    covariance=np.kron(np.eye(3),np.array([[1.,.64],[.64,1.]]))
    random=contrast[:,random_columns]@np.linalg.cholesky(covariance)
    random[:,4:]=0.  # Exact state control, independently of nominal M.
    return intrinsic,common,random


def length_probabilities(regime,index):
    short=[.1,.3,.5,.7,.2,.6][index] if regime=="heteroskedastic_tasks" else .25
    return np.array([short]+[(1-short)/7]*7)


def parameters(regime,index):
    scale=[.25,.5,1.,2.,.75,1.5][index] if regime=="heteroskedastic_tasks" else 1.
    return dict(episode_sd=(.015 if regime=="dominant_mc" else .15)*scale,
        call_sd=(.015 if regime=="dominant_mc" else .10)*scale,
        random_sd=.3 if regime=="dominant_mc" else .04,
        parameter_sd=.3 if regime=="dominant_mc" else .04,
        baseline=1+.1*index)


def completion_probability(regime,quality,n,candidate,family):
    if regime not in {"quality_completion","length_completion"}:return 1.
    budget=candidate["parameter_draws" if family==study.SPECIFICITY_FAMILY else "random_permutations"]
    exposure=min(n,candidate["calls_per_episode"])*(budget+1)
    if regime=="length_completion":exposure*=n/4
    return (.999 if quality>0 else .985)**exposure


def truth(design,regime,candidate):
    """Known conditional total/non-MC/MC variances for each evaluated setting."""
    intrinsic,common,random=base_loadings(regime)
    base_diag=np.sum(intrinsic**2,axis=1);between_diag=base_diag+.09*common**2
    random_diag=np.sum(random**2,axis=1)
    e05_scale=np.ones(12)
    if regime=="near_degenerate":e05_scale[::2]=1e-7
    base_shift=intrinsic@np.tile(np.array([.06,-.04,.02,0.]),6)
    e05_shift=np.tile(np.array([.04,-.03,.02,-.01]),3)*e05_scale
    output={family:{} for family in study.inferential_families(design)}
    for index,spec in enumerate(design["strata"]):
        sid=spec["id"];p=parameters(regime,index);lengths=np.arange(1,9);j=np.minimum(lengths,candidate["calls_per_episode"])
        probs=length_probabilities(regime,index)
        for family in output:
            raw=np.array([[.5*prob*completion_probability(regime,q,int(n),candidate,family) for n,prob in zip(lengths,probs)] for q in (-1,1)])
            completion=float(raw.sum());weights=raw/completion
            if regime=="collection_failure":completion*=.99
            mean_quality=float(np.sum(weights*np.array([-1,1])[:,None]));var_quality=1-mean_quality**2
            quality_scale=np.array([.6,1.4]) if regime=="quality_completion" else np.ones(2)
            scale2=float(np.sum(weights*quality_scale[:,None]**2))
            inv_j=float(np.sum(weights/j));scaled_inv_j=float(np.sum(weights*quality_scale[:,None]**2/j))
            active=regime in {"quality_completion","length_completion"}
            if family==study.FAMILY:
                non_mc=p["episode_sd"]**2*between_diag*scale2+p["call_sd"]**2*base_diag*scaled_inv_j
                if active:non_mc+=base_shift**2*var_quality
                mc=p["random_sd"]**2*random_diag*inv_j/candidate["random_permutations"]
                mean=base_shift*mean_quality if active else np.zeros(30)
            else:
                non_mc=e05_scale**2*(1.49*p["episode_sd"]**2*scale2+p["call_sd"]**2*scaled_inv_j)
                if active:non_mc+=e05_shift**2*var_quality
                mc=e05_scale**2*p["parameter_sd"]**2*(.74+.16*inv_j)/candidate["parameter_draws"]
                mean=e05_shift*mean_quality if active else np.zeros(12)
            output[family][sid]=dict(non_mc=non_mc,total=non_mc+mc,mc=mc,mean=mean,completion=completion,
                baseline_median=0. if regime=="zero_scale" else p["baseline"])
    return output


def _empty_data(candidate,family):
    return dict(episodes=[],matrix=[],membership=[],planned_episodes=0,planned_contexts=0,
        evaluated_calls_per_episode=candidate["calls_per_episode"],
        **({"evaluated_parameter_draws":candidate["parameter_draws"]} if family==study.SPECIFICITY_FAMILY else {"evaluated_random_permutations":[candidate["random_permutations"]]}))


def generate_panel(design,regime,candidates,counts, *, seed,trial,stream):
    """Generate all attempted episodes once, then coherent candidate prefixes."""
    if regime not in REGIMES:raise ValueError("A mandatory synthetic regime is unknown")
    families=study.inferential_families(design)
    result={c["id"]:{f:{} for f in families} for c in candidates}
    endpoints=dict(schema_version=2 if regime=="collection_failure" else 1,kind="complete_executed_call_baseline_endpoints",strata={})
    max_m=max(c["random_permutations"] for c in candidates);max_r=max(c["parameter_draws"] for c in candidates)
    intrinsic,common,random=base_loadings(regime);base_shift=intrinsic@np.tile(np.array([.06,-.04,.02,0.]),6)
    e05_scale=np.ones(12)
    if regime=="near_degenerate":e05_scale[::2]=1e-7
    e05_shift=np.tile(np.array([.04,-.03,.02,-.01]),3)*e05_scale
    for index,spec in enumerate(design["strata"]):
        sid=spec["id"];p=parameters(regime,index);rng=np.random.default_rng(seed_for(seed,regime,trial,stream,sid))
        for c in candidates:
            for f in families:result[c["id"]][f][sid]=_empty_data(c,f)
        endpoint_episodes=[]
        for e in range(counts[sid]):
            eid=f"{stream}:{sid}:episode{e}"
            reset=spec["reset_seeds"][e] if stream=="pilot" else seed_for(seed,regime,trial,stream,sid,e)%2**63
            if regime=="collection_failure" and rng.random()<.01:
                for c in candidates:
                    for f in families:
                        item=result[c["id"]][f][sid];item["planned_episodes"]+=1
                        item["membership"].append(dict(episode_id=eid,reset_seed=reset,executed_calls=0,selected_calls=0,joint_complete=False,
                            failure_kind="synthetic_collection_failure"))
                endpoint_episodes.append(dict(episode_id=eid,reset_seed=reset,terminal_record_sha256=object_hash([eid,"failed_terminal"]),
                    terminal_policy_calls=0,executed_call_indices=[],calls=[],collection_status="failed",collection_failure_reason="synthetic_collection_failure"))
                continue
            n=int(rng.choice(np.arange(1,9),p=length_probabilities(regime,index)));quality=int(rng.choice([-1,1]))
            selected=rng.permutation(n);kind=REGIMES[regime]["noise"]
            qscale=(.6 if quality<0 else 1.4) if regime=="quality_completion" else 1.
            between=p["episode_sd"]*qscale*(noise(rng,(24,),kind)@intrinsic.T+.3*float(noise(rng,(),kind))*common)
            calls=p["call_sd"]*qscale*(noise(rng,(n,24),kind)@intrinsic.T)
            base_non_mc=between+calls
            e05_between=p["episode_sd"]*qscale*(noise(rng,(12,),kind)+.7*float(noise(rng,(),kind)))*e05_scale
            e05_calls=p["call_sd"]*qscale*noise(rng,(n,12),kind)*e05_scale
            if regime in {"quality_completion","length_completion"}:
                base_non_mc+=quality*base_shift;e05_between+=quality*e05_shift
            random_draws=p["random_sd"]*(rng.normal(size=(max_m,n,6))@random.T)
            shared=rng.normal(size=(max_r,1,1))
            if regime=="rare_parameter_mc":
                shared=rng.choice([-1.,1.],size=(max_r,1,1))*np.where(rng.random((max_r,1,1))<.001,math.sqrt(1000),0.)
            coherent=p["parameter_sd"]*(.7*shared+.5*rng.normal(size=(max_r,1,12)))*e05_scale
            interaction=.4*p["parameter_sd"]*rng.normal(size=(max_r,n,12))*e05_scale
            e05_draws=e05_between+e05_calls+coherent+interaction
            survival={f:float(rng.random()) for f in families}
            for c in candidates:
                indices=selected[:min(n,c["calls_per_episode"])];j=len(indices)
                for f in families:
                    item=result[c["id"]][f][sid];complete=survival[f]<=completion_probability(regime,quality,n,c,f)
                    item["planned_episodes"]+=1;item["planned_contexts"]+=j
                    item["membership"].append(dict(episode_id=eid,reset_seed=reset,executed_calls=n,selected_calls=j,joint_complete=bool(complete)))
                    if not complete:continue
                    if f==study.FAMILY:
                        m=c["random_permutations"];replicates=random_draws[:m,indices]
                        effects=base_non_mc[indices]+replicates.mean(axis=0)
                        covariance=np.stack([study._cov(replicates[:,call])/m for call in range(j)])
                        episode=dict(total_calls=n,effects=effects,mc=covariance,
                            random_counts=[dict(vision=m,language=m,state=0) for call in range(j)])
                    else:
                        r=c["parameter_draws"];array=e05_draws[:r,indices];vectors=array.mean(axis=1)
                        effects=array.mean(axis=0)
                        episode=dict(total_calls=n,effects=effects,draw_call_effects=array,draw_episode_effects=vectors,
                            parameter_mc_covariance=study._cov(vectors)/r)
                    episode["episode_id"]=eid;item["episodes"].append(episode);item["matrix"].append(effects.mean(axis=0))
            baseline=0. if regime=="zero_scale" else p["baseline"]*math.exp(.12*rng.normal())
            endpoint_calls=[]
            for call in range(n):
                # Call heterogeneity sums to zero, so the FULL-call episode
                # mean has exactly the declared lognormal median.
                endpoint=baseline*(1+.2*(call-(n-1)/2)/n)
                modalities={}
                for modality in study.MODALITIES:
                    fail=regime=="missing_endpoint" and rng.random()<.01
                    modalities[modality]=dict(status="numerical_failure" if fail else "finite",baseline_rms=None if fail else float(endpoint),
                        endpoint_identity_sha256=object_hash([regime,trial,stream,sid,e,call,modality]),
                        **({"reason":"synthetic_nonfinite_endpoint"} if fail else {}))
                endpoint_calls.append(dict(policy_call_idx=call,context_id=f"{eid}:call{call}",source_row_sha256=object_hash([eid,call,"source"]),modalities=modalities))
            endpoint_episodes.append(dict(episode_id=eid,reset_seed=reset,terminal_record_sha256=object_hash([eid,"terminal"]),
                terminal_policy_calls=n,executed_call_indices=list(range(n)),calls=endpoint_calls))
        endpoints["strata"][sid]=dict(episodes=endpoint_episodes)
        for c in candidates:
            for f in families:
                item=result[c["id"]][f][sid]
                item["matrix"]=np.stack(item["matrix"]) if item["matrix"] else np.empty((0,len(study.local_contrasts(f))))
    return result,endpoints


def stage_prefix(data,endpoints,counts):
    output={}
    for cid,families in data.items():
        output[cid]={}
        for family,strata in families.items():
            output[cid][family]={}
            for sid,item in strata.items():
                members=item["membership"][:counts[sid]];allowed={m["episode_id"] for m in members}
                indices=[i for i,e in enumerate(item["episodes"]) if e["episode_id"] in allowed]
                clone={k:deepcopy(v) for k,v in item.items() if k not in {"matrix","episodes","membership"}}
                clone.update(episodes=[item["episodes"][i] for i in indices],matrix=item["matrix"][indices],membership=members,
                    planned_episodes=len(members),planned_contexts=sum(m["selected_calls"] for m in members))
                output[cid][family][sid]=clone
    bank=deepcopy(endpoints)
    for sid,item in bank["strata"].items():item["episodes"]=item["episodes"][:counts[sid]]
    return output,bank


def inspected_bound_failures(design,stage_result,truth_by_candidate):
    failures=[]
    for candidate in stage_result["candidates"]:
        cid=candidate["candidate"]["id"]
        for family,report in candidate["family_reports"].items():
            for sid,row in report["strata"].items():
                b=row["moment_bounds"];known=truth_by_candidate[cid][family][sid]
                for field,source in (("total_upper","total"),("non_mc_upper","non_mc"),("mc_upper","mc")):
                    bad=np.flatnonzero(known[source]>np.array(b[field])*(1+1e-12)+1e-15)
                    if len(bad):failures.append(dict(candidate=cid,family=family,stratum=sid,bound=field,cells=bad.tolist()))
                bad=np.flatnonzero(known["non_mc"]+1e-15<np.array(b["non_mc_lower"])*(1-1e-12))
                if len(bad):failures.append(dict(candidate=cid,family=family,stratum=sid,bound="non_mc_lower",cells=bad.tolist()))
    return failures


def validate_future(design,candidate,allocation,scales,data,known):
    """Use the exact frozen v4 bootstrap kernel and seeds on new episodes.

    This function reports simulated precision and coverage diagnostics. It does
    not manufacture the separate exact-K calibration artifact required by v4.
    """
    output={};settings=design["analysis"]
    for family in study.inferential_families(design):
        registry=study.family_registry(design,family);items=data[family]
        counts={sid:len(item["matrix"]) for sid,item in items.items()}
        required=allocation["family_allocations"][family]["required_complete_episodes"]
        shortfall=any(counts[sid]<required[sid] for sid in counts)
        if any(n<2 for n in counts.values()):
            output[family]=dict(status="insufficient_complete_episodes",counts=counts,shortfall=shortfall,
                halfwidth_failure=True,population_halfwidth_failure=True,true_mc_failure=True,empirical_mc_failure=True,endpoint_mc_failure=True,interval_noncoverage=True)
            continue
        alpha=design["global_alpha"]*design["family_weights"][family];tail=alpha/(2*len(registry));intervals=[]
        for repeat in range(settings["mc_repeats"]):
            samples=(study._bootstrap(design,items,settings["draws"],settings["seed"]+repeat) if family==study.FAMILY else
                study._bootstrap_family(design,items,settings["draws"],settings["seed"]+repeat,family))
            intervals.append(np.quantile(samples,[tail,1-tail],axis=0).T)
        intervals=np.asarray(intervals);ci=intervals[0];spread=np.ptp(intervals,axis=0).max(axis=1)
        truth_vector=np.concatenate([sum(w*known[family][s]["mean"] for s,w in g["weights"].items()) for g in design["groups"]])
        h=np.array([.05*scales[f"{c['group']}:{c['modality']}"]["value"] for c in registry])
        population_scales={g["id"]:sum(w*known[family][s]["baseline_median"] for s,w in g["weights"].items()) for g in design["groups"]}
        population_h=np.array([.05*population_scales[c["group"]] for c in registry])
        width=(ci[:,1]-ci[:,0])/2
        true_mc=[];true_non_mc=[];estimated_mc=[];estimated_non_mc=[]
        from pilot_selection import _episode_mc
        for group in design["groups"]:
            true_mc.extend(sum(w*w*known[family][s]["mc"]/counts[s] for s,w in group["weights"].items()))
            true_non_mc.extend(sum(w*w*known[family][s]["non_mc"]/counts[s] for s,w in group["weights"].items()))
            em={s:_episode_mc(items[s],family).mean(axis=0) for s in group["weights"]}
            estimated_mc.extend(sum(w*w*em[s]/counts[s] for s,w in group["weights"].items()))
            estimated_non_mc.extend(sum(w*w*np.maximum(items[s]["matrix"].var(axis=0,ddof=1)-em[s],0)/counts[s] for s,w in group["weights"].items()))
        true_mc=np.asarray(true_mc);true_non_mc=np.asarray(true_non_mc);estimated_mc=np.asarray(estimated_mc);estimated_non_mc=np.asarray(estimated_non_mc)
        output[family]=dict(status="evaluated",counts=counts,shortfall=shortfall,
            halfwidth_failure=bool(np.any(width>h)),maximum_halfwidth_ratio=float(np.max(width/h)),
            population_halfwidth_failure=bool(np.any(width>population_h)),
            maximum_population_halfwidth_ratio=float(np.max(width/population_h)) if np.all(population_h>0) else None,
            true_mc_failure=bool(np.any((true_mc>.01*true_non_mc)|(true_mc>.01*h*h))),
            empirical_mc_failure=bool(np.any((estimated_mc>.01*estimated_non_mc)|(estimated_mc>.01*h*h))),
            endpoint_mc_failure=bool(np.any(spread>settings["endpoint_mc_fraction_h"]*h)),
            interval_noncoverage=bool(np.any((truth_vector<ci[:,0])|(truth_vector>ci[:,1]))),
            bootstrap_seed=settings["seed"],bootstrap_repeats=settings["mc_repeats"],first_interval_prespecified=True)
    return output


def simulate_rule_trial(design,protocol,regime,seed,trial):
    import time
    import pilot_selection as selector
    start=time.perf_counter();selector.validate_protocol(protocol,design)
    candidates=protocol["candidates"];counts=protocol["stages"][-1]["episodes_per_stratum"]
    full,endpoints=generate_panel(design,regime,candidates,counts,seed=seed,trial=trial,stream="pilot")
    known={c["id"]:truth(design,regime,c) for c in candidates}
    inspected=[];scale_failures=[];history=[];selected=None
    for stage in protocol["stages"]:
        data,bank=stage_prefix(full,endpoints,stage["episodes_per_stratum"])
        result=selector.evaluate_stage(design,protocol,stage["id"],data,bank)
        failures=inspected_bound_failures(design,result,known);inspected.extend(failures)
        for group in design["groups"]:
            population=sum(w*known[candidates[0]["id"]][study.FAMILY][s]["baseline_median"] for s,w in group["weights"].items())
            for modality in study.MODALITIES:
                cell=result["scale_report"]["scales"][f"{group['id']}:{modality}"]
                if cell.get("lower") is not None and population<cell["lower"]-1e-14:scale_failures.append(f"{stage['id']}:{group['id']}:{modality}:lower")
                if cell.get("upper") is not None and population>cell["upper"]+1e-14:scale_failures.append(f"{stage['id']}:{group['id']}:{modality}:upper")
        history.append(dict(stage_id=stage["id"],status=result["status"],selected_candidate=result["selected_candidate"],
            inspected_bound_failure=bool(failures),scale_reasons=result["scale_reasons"],
            candidate_reasons={c["candidate"]["id"]:c["reasons"] for c in result["candidates"]}))
        if result["selected_candidate"] is not None:selected=result;break
    row=dict(kind="pilot_rule_synthetic_trial",regime=regime,trial=trial,seed_sha256=object_hash(dict(seed=seed,regime=regime,trial=trial)),
        stages=history,selected=selected is not None,selected_candidate=None,
        inspected_variance_bound_failure=bool(inspected),scale_bound_failure=bool(scale_failures),
        inspected_bound_failures=inspected,scale_bound_failures=scale_failures,
        no_real_efficacy_data=True,interpretation="finite_synthetic_domain_validation_not_distribution_free")
    if selected is None:
        row.update(status="inconclusive",selected_variance_bound_failure=False,completion_bound_failure=False,
            future_shortfall=False,future_precision_failure=False,future_mc_failure=False,future_interval_noncoverage=False,
            future_population_precision_failure=False,future_endpoint_mc_failure=False,selected_invalid=False)
    else:
        cid=selected["selected_candidate"];candidate=next(c for c in candidates if c["id"]==cid);allocation=selected["selected_allocation"]
        completion_bad=any(record["completion_probability_lower"][sid]>known[cid][family][sid]["completion"]+1e-14
            for family,record in allocation["family_allocations"].items() for sid in record["completion_probability_lower"])
        future,_=generate_panel(design,regime,[candidate],allocation["planned_episodes"],seed=seed,trial=trial,stream="confirmation")
        diagnostics=validate_future(design,candidate,allocation,selected["scale_report"]["scales"],future[cid],known[cid])
        selected_bounds=any(f["candidate"]==cid for f in inspected_bound_failures(design,selected,known))
        row.update(status="selected",selected_candidate=cid,selected_stage=selected["stage_id"],planned_confirmation_episodes=allocation["planned_episodes"],
            selected_variance_bound_failure=selected_bounds,completion_bound_failure=completion_bad,future=diagnostics,
            future_shortfall=any(r["shortfall"] for r in diagnostics.values()),
            future_precision_failure=any(r["halfwidth_failure"] for r in diagnostics.values()),
            future_population_precision_failure=any(r["population_halfwidth_failure"] for r in diagnostics.values()),
            future_mc_failure=any(r["true_mc_failure"] or r["empirical_mc_failure"] for r in diagnostics.values()),
            future_interval_noncoverage=any(r["interval_noncoverage"] for r in diagnostics.values()),
            future_endpoint_mc_failure=any(r["endpoint_mc_failure"] for r in diagnostics.values()))
        row["selected_invalid"]=bool(selected_bounds or scale_failures or completion_bad or row["future_shortfall"] or row["future_precision_failure"] or row["future_population_precision_failure"] or row["future_mc_failure"] or row["future_endpoint_mc_failure"])
    row["elapsed_seconds"]=time.perf_counter()-start
    return row
