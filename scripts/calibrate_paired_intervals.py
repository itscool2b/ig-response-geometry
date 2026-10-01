"""CPU-only, prospectively specified coverage simulation for the global study.

Synthetic data are independent of efficacy outcomes. Every declared regime is
retained. The exact production episode-bootstrap kernel is used, including its
Bonferroni tails and repeated endpoint-MC audit. Stagewise exact binomial bounds
spend a fixed simulation probability budget across regimes and looks. No GPU,
model loading, empirical signed pilot means, or favorable-regime selection.
"""
from __future__ import annotations

# Set before importing NumPy/BLAS in parent and spawned CPU workers.
import os
for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name]="1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import math
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy
from scipy.stats import beta, binom
from filelock import FileLock

import paired_study_analysis as study
from experiment_io import canonical_json,file_hash,object_hash,strict_json,atomic_bytes

REGIMES={
    "gaussian_clustered":dict(noise="normal",description="Correlated Gaussian episode and call effects with unequal trajectory lengths."),
    "right_skewed_gamma":dict(noise="gamma",description="Centered unit-variance Gamma(shape=.5) effects; finite moments and pronounced right skew."),
    "student_t3":dict(noise="t3",description="Centered unit-variance Student t with three degrees of freedom; finite variance and infinite fourth moment."),
    "heteroskedastic_tasks":dict(noise="normal",description="Fixed task SD multipliers .25,.5,1,2,.75,1.5 and different task length mixtures."),
    "rare_outlier_mixture":dict(noise="mixture",description="99% normal SD1 and 1% normal SD10, rescaled to unit variance; unseen-tail stress."),
    "near_degenerate":dict(noise="normal",description="Q/L2 method differences have 1e-7 residual noise; other contrasts retain ordinary variance."),
    "correlated_mc":dict(noise="normal",description="Small episode/call variance and dominant correlated random-order noise shared by both IG comparisons and directions."),
    "symmetric_completion":dict(noise="normal",description="Call survival decreases with absolute episode shock; even selection leaves centered conditional means unchanged."),
    "asymmetric_completion":dict(noise="normal",description="Latent quality shifts paired effects and per-mask survival; exact length/J/M-dependent conditional truth differs from the unconditional truth.")}


def stream_seed(seed,regime,trial,stream):
    return int(object_hash(dict(master_seed=seed,regime=regime,trial=trial,stream=stream)),16)


def noise(rng,shape,kind):
    if kind=="normal":return rng.normal(size=shape)
    if kind=="gamma":return (rng.gamma(.5,size=shape)-.5)/math.sqrt(.5)
    if kind=="t3":return rng.standard_t(3,size=shape)/math.sqrt(3)
    if kind=="mixture":
        return rng.normal(size=shape)*np.where(rng.random(shape)<.01,10.,1.)/math.sqrt(1.99)
    raise ValueError("Unknown synthetic noise distribution")


def length_probabilities(regime,stratum_index):
    short=[.1,.3,.5,.7,.2,.6][stratum_index] if regime=="heteroskedastic_tasks" else .5
    return np.array([short]+[(1-short)/24]*24)


def area_means(regime,stratum_index):
    # Method axis is Q-IG,L2-IG,input-difference,random. Synthetic means are
    # declared generator constants, never empirical efficacy estimates.
    means=np.array([[.12,.15,.10,.08],[.05,.03,.09,.10]])
    means=np.tile(means[None,:,:],(3,1,1))
    means+=.01*stratum_index
    if regime=="near_degenerate":means[:,:,1]=means[:,:,0]
    return means


def contrast_vector(areas):
    output=[]
    index={"Q_IG":0,"L2_IG":1,"input_difference":2,"random":3}
    for c in study.LOCAL_CONTRASTS:
        m=study.MODALITIES.index(c["modality"]); d=study.DIRECTIONS.index(c["direction"])
        sign=1 if c["direction"]=="deletion" else -1
        output.append(sign*(areas[...,m,d,index[c["method"]]]-areas[...,m,d,index[c["control"]]]))
    return np.stack(output,axis=-1)


def conditional_quality_mean(regime,probabilities,j,m):
    if regime!="asymmetric_completion":return 0.
    counts=np.minimum(j,np.arange(1,26))
    # One mandatory mask block plus M sampled random blocks per selected call.
    # State exact control is not used to increase this failure exposure.
    plus=np.dot(probabilities,.998**(counts*(m+1)))
    minus=np.dot(probabilities,.988**(counts*(m+1)))
    return float((plus-minus)/(plus+minus))


def analytic_truth(design,regime):
    conditional={}; unconditional={}
    shift=np.tile(np.array([.06,-.04,.02,0.])[None,None,:],(3,2,1));shift[:,1,:]*=-1
    for index,spec in enumerate(design["strata"]):
        means=area_means(regime,index)
        tilt=conditional_quality_mean(regime,length_probabilities(regime,index),spec["calls_per_episode"],spec["random_permutations"])
        conditional[spec["id"]]=contrast_vector(means+shift*tilt)
        unconditional[spec["id"]]=contrast_vector(means)
    return tuple(np.concatenate([sum(w*values[s] for s,w in group["weights"].items()) for group in design["groups"]]) for values in (conditional,unconditional))


def generate_trial(design,regime,seed,trial,complete_counts=None):
    """Generate independently reset episode vectors with an analytic truth.

    Four method areas define the five linearly dependent paired contrasts.
    Adding one common context offset would make every synthetic RMS area
    nonnegative without changing any contrast. Thus the generator concerns
    the paired estimator, not fabricated neural-network intervention outputs.
    """
    if regime not in REGIMES:raise ValueError("Unknown calibration regime")
    rng=np.random.Generator(np.random.PCG64(stream_seed(seed,regime,trial,"data")))
    data={}; truths={}; unconditional={}; diagnostics={}
    loading=np.array([1.,.8,-.4])[None,None,None,:]
    quality_shift=np.tile(np.array([.06,-.04,.02,0.])[None,None,:],(3,2,1))
    quality_shift[:,1,:]*=-1
    for si,spec in enumerate(design["strata"]):
        planned_n=len(spec["reset_seeds"])
        n=planned_n if complete_counts is None else complete_counts[spec["id"]]
        j=spec["calls_per_episode"]; m=spec["random_permutations"]
        probs=length_probabilities(regime,si)
        lengths=[]
        means=area_means(regime,si)
        truth=means+quality_shift*conditional_quality_mean(regime,probs,j,m)
        truths[spec["id"]]=contrast_vector(truth)
        unconditional[spec["id"]]=contrast_vector(means)
        matrices=[]; completed_lengths=[]; failed=0
        task_scale=[.25,.5,1.,2.,.75,1.5][si] if regime=="heteroskedastic_tasks" else 1.
        episode_sd=.015 if regime=="correlated_mc" else .15
        call_sd=.015 if regime=="correlated_mc" else .10
        random_sd=.30 if regime=="correlated_mc" else .06
        attempts=0
        while (attempts<n if complete_counts is None else len(matrices)<n):
            attempts+=1
            if attempts>10000*n:
                raise RuntimeError("Conditional synthetic sampler failed to reach its fixed count; no silent replacement of the regime")
            if complete_counts is not None and regime=="asymmetric_completion":
                sizes=np.minimum(j,np.arange(1,26))
                joint=np.array([.5*probs*.988**(sizes*(m+1)),.5*probs*.998**(sizes*(m+1))])
                draw=rng.choice(50,p=(joint/joint.sum()).reshape(-1))
                quality=-1. if draw<25 else 1.
                length=int(draw%25+1)
            else:
                length=int(rng.choice(np.arange(1,26),p=probs))
                quality=float(rng.choice([-1.,1.]))
            lengths.append(length)
            kind=REGIMES[regime]["noise"]
            common=noise(rng,(1,1,1,1),kind)
            modality=noise(rng,(1,3,1,1),kind)
            method=noise(rng,(1,3,1,3),kind)
            specific=noise(rng,(1,3,2,3),kind)
            between=(.4*common*loading+.3*modality*loading+.3*method+.7*specific)*episode_sd*task_scale
            within=noise(rng,(length,3,2,3),kind)*call_sd*task_scale
            all_noise=between+within
            if regime=="near_degenerate":all_noise[:,:,:,1]=all_noise[:,:,:,0]+1e-7*noise(rng,(length,3,2),kind)
            areas=np.broadcast_to(means,(length,3,2,4)).copy()
            areas[:,:,:,:3]+=all_noise
            selected=rng.permutation(length)[:min(j,length)]
            if regime=="asymmetric_completion":areas+=quality*quality_shift
            selected_areas=areas[selected].copy()
            # Random orders are independent across contexts and modalities,
            # but their same noisy control is used in both IG comparisons.
            for call_index in range(len(selected)):
                for modality_index in (0,1):
                    if modality_index==1 and selected[call_index]%2==0:continue
                    shared=rng.normal(size=(m,1)); independent=rng.normal(size=(m,2))
                    random_orders=random_sd*(.8*shared+.6*independent)
                    selected_areas[call_index,modality_index,:,3]+=random_orders.mean(axis=0)
            if regime=="asymmetric_completion":
                probability=1. if complete_counts is not None else (.998 if quality>0 else .988)**(len(selected)*(m+1))
            elif regime=="symmetric_completion":
                probability=math.exp(-.12*len(selected)*float(abs(common.item())))
            else:probability=1.
            if rng.random()>probability:
                failed+=1;continue
            matrices.append(contrast_vector(selected_areas).mean(axis=0))
            completed_lengths.append(int(length))
        matrix=np.stack(matrices) if matrices else np.empty((0,30))
        data[spec["id"]]=dict(matrix=matrix)
        diagnostics[spec["id"]]=dict(design_planned_episodes=planned_n,complete_episodes=len(matrix),
            synthetic_episode_proposals=attempts,synthetic_rejections=failed,
            count_conditioning="planned_episode_sampling" if complete_counts is None else "exact_joint_complete_episode_counts",
            selected_calls_mean=float(np.minimum(j,lengths).mean()),
            completed_trajectory_length_mean=float(np.mean(completed_lengths)) if completed_lengths else None,
            conditional_quality_mean=conditional_quality_mean(regime,probs,j,m))
    truth=np.concatenate([sum(w*truths[s] for s,w in group["weights"].items()) for group in design["groups"]])
    unconditional_truth=np.concatenate([sum(w*unconditional[s] for s,w in group["weights"].items()) for group in design["groups"]])
    return data,truth,unconditional_truth,diagnostics


def simulate_one(design,regime,seed,trial,minimum_complete,complete_counts=None):
    started=time.perf_counter()
    data,truth,unconditional,diagnostics=generate_trial(design,regime,seed,trial,complete_counts)
    base=dict(regime=regime,trial=trial,truth=truth.tolist(),unconditional_truth=unconditional.tolist(),
              episode_accounting=diagnostics,seed_sha256=object_hash(dict(seed=seed,regime=regime,trial=trial)))
    if any(len(data[s]["matrix"])<max(2,minimum_complete[s]) for s in data):
        return dict(**base,status="insufficient_conditional_population",familywise_noncoverage=True,
                    endpoint_mc_gate=False,elapsed_seconds=time.perf_counter()-started)
    settings=design["analysis"]
    alpha=design["global_alpha"]*design["family_weights"][study.FAMILY]
    intervals=[]
    # Production uses these fixed weights/streams on random datasets. Varying
    # the bootstrap seed by synthetic trial would calibrate another procedure.
    bootstrap_seed=settings["seed"]
    for repeat in range(settings["mc_repeats"]):
        draws=study._bootstrap(design,data,settings["draws"],bootstrap_seed+repeat)
        intervals.append(np.quantile(draws,[alpha/120,1-alpha/120],axis=0).T)
    intervals=np.array(intervals)
    primary=intervals[0]
    misses=(truth<primary[:,0])|(truth>primary[:,1])
    widths=(primary[:,1]-primary[:,0])/2
    h=np.array([.05*design["frozen_scales"][f"{c['group']}:{c['modality']}"] for c in study.primary_registry(design)])
    spreads=np.ptp(intervals,axis=0).max(axis=1)
    estimates=np.concatenate([sum(w*data[s]["matrix"].mean(axis=0) for s,w in g["weights"].items()) for g in design["groups"]])
    return dict(**base,status="evaluated",familywise_noncoverage=bool(misses.any()),
        noncovering_contrast_indices=np.flatnonzero(misses).tolist(),estimate=estimates.tolist(),
        confidence_interval=primary.tolist(),endpoint_mc_spread=spreads.tolist(),
        endpoint_mc_gate=bool(np.all(spreads<=settings["endpoint_mc_fraction_h"]*h)),
        halfwidth_targets_met=bool(np.all(widths<=h)),maximum_halfwidth_ratio=float(np.max(widths/h)),
        elapsed_seconds=time.perf_counter()-started)


def binomial_bounds(failures,trials,alpha):
    if not 0<=failures<=trials or trials<1 or not 0<alpha<1:raise ValueError("Invalid simulation binomial accounting")
    lower=0. if failures==0 else float(beta.ppf(alpha,failures,trials-failures+1))
    upper=1. if failures==trials else float(beta.ppf(1-alpha,failures+1,trials-failures))
    return lower,upper


def derive_simulation_stages(*,scenario_count,look_weights,simulation_alpha,limit,reference_rate,joint_success_targets):
    """Counts derived from binomial precision/power, never a customary default.

    The first look is the minimum giving a passing upper bound at zero errors.
    Later sizes attain declared joint powers under the explicit reference rate.
    That planning reference is not an assumption used in the actual bounds.
    """
    if len(look_weights)!=len(joint_success_targets)+1 or not 0<reference_rate<limit<1:
        raise ValueError("Invalid simulation-stage planning assumptions")
    if not 0<sum(look_weights)<=1 or any(w<=0 for w in look_weights):raise ValueError("Invalid look weights")
    first_alpha=simulation_alpha*look_weights[0]/scenario_count
    stages=[math.ceil(math.log(first_alpha)/math.log(1-limit))]
    for weight,target in zip(look_weights[1:],joint_success_targets):
        local_alpha=simulation_alpha*weight/scenario_count
        local_target=target**(1/scenario_count)
        if not 0<target<1:raise ValueError("Invalid prospective joint success probability")
        def power(n):
            # Largest error count whose exact one-sided upper bound passes.
            lo,hi=-1,n
            while lo+1<hi:
                mid=(lo+hi)//2
                if binomial_bounds(mid,n,local_alpha)[1]<=limit:lo=mid
                else:hi=mid
            return 0. if lo<0 else float(binom.cdf(lo,n,reference_rate))
        n=stages[-1]+1
        # Acceptance power has small sawtooth jumps as the integer rejection
        # threshold changes. Search increasing n, not an invalid monotone bisection.
        while power(n)<local_target:
            n+=1
            if n>1000000:raise ValueError("Requested simulation precision exceeds supported planning search")
        stages.append(n)
    return stages


def validate_protocol(protocol,design):
    study.validate_design(design)
    if design["stage"]!="confirmatory_locked":raise ValueError("Coverage scenarios require a fixed candidate confirmation design")
    if protocol.get("kind")!="paired_interval_calibration_protocol" or protocol.get("schema_version")!=1:
        raise ValueError("Unknown calibration protocol")
    if protocol.get("mode") not in {"candidate_scope","characterization_only"}:raise ValueError("Unknown calibration mode")
    counts=protocol.get("complete_episode_counts")
    if protocol["mode"]=="candidate_scope" and (counts is None or protocol.get("count_conditioning")!="exact_joint_complete_episode_counts"):
        raise ValueError("Candidate coverage must bind the exact observed complete counts")
    if protocol["mode"]=="candidate_scope":
        plan=design["analysis"].get("coverage_plan")
        if not plan or plan.get("kind")!="prospective_exact_count_coverage_plan" or plan.get("schema_version")!=1 or protocol.get("coverage_plan_sha256")!=object_hash(plan):
            raise ValueError("Candidate calibration requires its immutable prospective plan")
        keys=("analysis_module_sha256","generator_source_sha256","regime_definitions_sha256","regimes",
              "stage_derivation","simulation_stages","seed","count_conditioning")
        if any(protocol.get(key)!=plan.get(key) for key in keys):
            raise ValueError("Exact-count instantiation changed the prospective calibration plan")
        if protocol.get("minimum_complete_episodes")!=counts:
            raise ValueError("Candidate calibration domain must equal its exact complete counts")
    if counts is not None:
        if set(counts)!={s["id"] for s in design["strata"]} or any(type(counts[s["id"]]) is not int or not 2<=counts[s["id"]]<=len(s["reset_seeds"]) for s in design["strata"]):
            raise ValueError("Invalid complete-count scope")
    if protocol["scope_sha256"]!=object_hash(study.calibration_scope(design,counts)):
        raise ValueError("Calibration changed the planned inference scope")
    if protocol["analysis_module_sha256"]!=file_hash(study.__file__) or protocol["generator_source_sha256"]!=file_hash(__file__):
        raise ValueError("Calibration code differs from the prospective source identity")
    if protocol["regime_definitions_sha256"]!=object_hash(REGIMES) or protocol["regimes"]!=list(REGIMES):
        raise ValueError("The complete fixed regime registry must be preserved without favorable selection")
    if protocol["simulation_stages"]!=derive_simulation_stages(scenario_count=len(REGIMES),**protocol["stage_derivation"]):
        raise ValueError("Simulation stage counts differ from their recorded precision derivation")
    alpha=design["global_alpha"]*design["family_weights"][study.FAMILY]
    if protocol["stage_derivation"]["limit"]!=alpha:
        raise ValueError("Simulation acceptance changed the allocated inferential alpha")
    settings=design["analysis"]
    minimum=study.bootstrap_requirement(alpha,60,settings["min_tail_draws"],settings["tail_relative_mcse"])
    if settings["draws"]<minimum or settings["mc_repeats"]<2:
        raise ValueError("Coverage must simulate the fully resolved production interval procedure")
    if set(protocol["minimum_complete_episodes"])!={s["id"] for s in design["strata"]}:
        raise ValueError("Coverage needs each stratum's explicit complete-episode domain")
    for spec in design["strata"]:
        minimum=protocol["minimum_complete_episodes"][spec["id"]]
        if type(minimum) is not int or not 2<=minimum<=len(spec["reset_seeds"]):
            raise ValueError("Invalid conditional population domain")
    study.integer(protocol["seed"],"simulation seed",0)
    return protocol


def stage_summary(records,protocol,look):
    alpha=protocol["stage_derivation"]["simulation_alpha"]*protocol["stage_derivation"]["look_weights"][look]/len(REGIMES)
    failures=sum(r["familywise_noncoverage"] for r in records)
    lower,upper=binomial_bounds(failures,len(records),alpha)
    limit=protocol["stage_derivation"]["limit"]
    decision="passed" if upper<=limit else "failed" if lower>limit else "inconclusive"
    return dict(simulations=len(records),familywise_noncoverage=failures,look_index=look,
        simulation_alpha=alpha,lower_noncoverage_bound=lower,upper_noncoverage_bound=upper,decision=decision,
        insufficient_population_trials=sum(r["status"]!="evaluated" for r in records),
        endpoint_mc_gate_failures=sum(not r["endpoint_mc_gate"] for r in records),
        halfwidth_goal_failures=sum(not r.get("halfwidth_targets_met",False) for r in records))


def verify_sources(manifest):
    if file_hash(study.__file__)!=manifest["analysis_module_sha256"] or file_hash(__file__)!=manifest["generator_source_sha256"]:
        raise ValueError("Calibration source bytes changed during the run; partial evidence is preserved without approval")


def validate_trial_record(record,design,protocol,protocol_sha256,regime,trial):
    if (record.get("regime"),record.get("trial"),record.get("protocol_sha256"))!=(regime,trial,protocol_sha256):
        raise ValueError("Stored synthetic trial identity mismatch")
    expected_seed=object_hash(dict(seed=protocol["seed"],regime=regime,trial=trial))
    if record.get("seed_sha256")!=expected_seed:
        raise ValueError("Synthetic trial seed identity mismatch")
    truth,unconditional=analytic_truth(design,regime)
    if not np.array_equal(record["truth"],truth) or not np.array_equal(record["unconditional_truth"],unconditional):
        raise ValueError("Synthetic trial truth differs from the declared generator")
    if record["status"]=="evaluated":
        bounds=np.asarray(record["confidence_interval"])
        if bounds.shape!=(60,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>bounds[:,1]):
            raise ValueError("Invalid saved synthetic confidence interval")
        misses=(truth<bounds[:,0])|(truth>bounds[:,1])
        if record["familywise_noncoverage"] is not bool(misses.any()) or record["noncovering_contrast_indices"]!=np.flatnonzero(misses).tolist():
            raise ValueError("Saved coverage decision differs from saved truth and interval")
    elif record["status"]!="insufficient_conditional_population" or record["familywise_noncoverage"] is not True:
        raise ValueError("Unknown or favorably omitted synthetic outcome")
    if protocol.get("complete_episode_counts") is not None:
        actual={s:r["complete_episodes"] for s,r in record["episode_accounting"].items()}
        if actual!=protocol["complete_episode_counts"]:
            raise ValueError("Candidate synthetic trial changed its exact complete counts")


def _ledger_path(relative,protocol):
    parts=Path(relative).parts
    if len(parts)!=2 or parts[0] not in REGIMES:
        raise ValueError("Invalid trial ledger path")
    name=parts[1]
    try:trial=int(name[6:-5])
    except ValueError:raise ValueError("Invalid trial ledger filename") from None
    if name!=f"trial_{trial:06d}.json" or not 0<=trial<max(protocol["simulation_stages"]):
        raise ValueError("Trial ledger names an unplanned replicate")
    return parts[0],trial


def read_trial_ledger(out,manifest,design,protocol,protocol_sha256):
    """The fsynced ledger is authoritative; per-trial JSON files are exports."""
    ledger=out/"trial_ledger.jsonl"
    head=file_hash(out/"manifest.json"); entries={}
    if ledger.exists():
        payload=ledger.read_bytes()
        if payload and not payload.endswith(b"\n"):
            raise ValueError("Incomplete trailing ledger transaction; preserve it for explicit recovery")
        def unique(pairs):
            result={}
            for key,value in pairs:
                if key in result:raise ValueError("Repeated ledger JSON key")
                result[key]=value
            return result
        for line in payload.splitlines():
            item=json.loads(line,object_pairs_hook=unique);canonical_json(item)
            entry_hash=item.pop("entry_sha256")
            if item["previous_chain_sha256"]!=head or object_hash(item)!=entry_hash or item["path"] in entries:
                raise ValueError("Trial ledger hash chain changed or a trial was repeated")
            regime,trial=_ledger_path(item["path"],protocol)
            record=item["record"]
            validate_trial_record(record,design,protocol,protocol_sha256,regime,trial)
            content=canonical_json(record)+b"\n"
            import hashlib
            if hashlib.sha256(content).hexdigest()!=item["file_sha256"]:
                raise ValueError("Ledger payload differs from its export digest")
            path=out/item["path"]
            if path.exists():
                if file_hash(path)!=item["file_sha256"]:
                    raise ValueError("A committed synthetic trial export was modified")
            else:
                path.parent.mkdir(exist_ok=True)
                atomic_bytes(path,content)
            entries[item["path"]]=item["file_sha256"]
            head=entry_hash
    actual={str(p.relative_to(out)) for p in out.glob("*/trial_*.json")}
    if actual!=set(entries):
        raise ValueError("Uncommitted trial file outside the authoritative ledger; do not silently resume it")
    verify_sources(manifest)
    return entries,head


def commit_trial(out,path,record,head):
    import hashlib
    content=canonical_json(record)+b"\n"
    relative=str(path.relative_to(out))
    item=dict(path=relative,file_sha256=hashlib.sha256(content).hexdigest(),record=record,previous_chain_sha256=head)
    item["entry_sha256"]=object_hash(item)
    # Journal first, then derived export. A crash between these operations is
    # recovered from the authenticated journal without rerunning a simulation.
    with (out/"trial_ledger.jsonl").open("ab") as stream:
        stream.write(canonical_json(item)+b"\n");stream.flush();os.fsync(stream.fileno())
    if path.exists():raise ValueError("Refusing to overwrite a trial export")
    atomic_bytes(path,content)
    return item["entry_sha256"],item["file_sha256"]


def run_calibration(design,protocol,out, *, workers,resume,design_sha256,protocol_sha256,through_look=None):
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(out)+".lock",timeout=0):
        return _run_calibration(design,protocol,out,workers=workers,resume=resume,design_sha256=design_sha256,protocol_sha256=protocol_sha256,through_look=through_look)


def _run_calibration(design,protocol,out, *, workers,resume,design_sha256,protocol_sha256,through_look):
    validate_protocol(protocol,design)
    study.integer(workers,"CPU workers")
    through_look=len(protocol["simulation_stages"]) if through_look is None else study.integer(through_look,"execution look bound")
    if through_look>len(protocol["simulation_stages"]):raise ValueError("Execution look bound exceeds the fixed protocol")
    out=Path(out); manifest_path=out/"manifest.json"
    manifest=dict(schema_version=1,kind="paired_interval_calibration_run",design_sha256=design_sha256,
        protocol_sha256=protocol_sha256,analysis_module_sha256=file_hash(study.__file__),generator_source_sha256=file_hash(__file__),
        numpy_version=np.__version__,scipy_version=scipy.__version__,python_version=platform.python_version(),
        platform=platform.platform(),cpu_workers=workers,blas_threads_per_worker=1,protocol=protocol)
    if resume:
        if strict_json(manifest_path)!=manifest:raise ValueError("Resume identity differs; preserve the old run and record an amendment")
        previous_looks=[int(p.stem.removeprefix("look_")) for p in out.glob("look_*.json")]
        if previous_looks and through_look<max(previous_looks):raise ValueError("Cannot roll back a completed simulation look")
    else:
        out.mkdir(parents=True,exist_ok=False)
        with manifest_path.open("xb") as stream:stream.write(canonical_json(manifest)+b"\n")
    committed,head=read_trial_ledger(out,manifest,design,protocol,protocol_sha256)
    summaries={}; started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for look,count in enumerate(protocol["simulation_stages"]):
            if look>=through_look:break
            verify_sources(manifest)
            pending={}; active=[]
            for regime in REGIMES:
                if summaries.get(regime,{}).get("decision") in {"passed","failed"}:continue
                active.append(regime)
                directory=out/regime; directory.mkdir(exist_ok=True)
                for trial in range(count):
                    path=directory/f"trial_{trial:06d}.json"
                    if str(path.relative_to(out)) in committed:
                        continue
                    job=pool.submit(simulate_one,design,regime,protocol["seed"],trial,protocol["minimum_complete_episodes"],protocol.get("complete_episode_counts"))
                    pending[job]=(path,regime,trial)
            finished=0
            for job in as_completed(pending):
                verify_sources(manifest)
                record=job.result(); record["protocol_sha256"]=protocol_sha256
                path,regime,trial=pending[job]
                validate_trial_record(record,design,protocol,protocol_sha256,regime,trial)
                head,export_hash=commit_trial(out,path,record,head)
                committed[str(path.relative_to(out))]=export_hash
                finished+=1
                if finished%max(1,workers*8)==0:
                    print(f"look {look+1}: saved {finished}/{len(pending)} new synthetic trials",flush=True)
            for regime in active:
                records=[strict_json(out/regime/f"trial_{trial:06d}.json") for trial in range(count)]
                summaries[regime]=stage_summary(records,protocol,look)
            atomic_bytes(out/f"look_{look+1}.json",canonical_json(dict(summaries=summaries,elapsed_seconds=time.perf_counter()-started))+b"\n")
            print(f"look {look+1}: "+str({k:v["decision"] for k,v in summaries.items()}),flush=True)
            if all(v["decision"] in {"passed","failed"} for v in summaries.values()):break
    statistical="passed" if all(v["decision"]=="passed" for v in summaries.values()) else "failed" if any(v["decision"]=="failed" for v in summaries.values()) else "inconclusive"
    status="approved" if statistical=="passed" and protocol["mode"]=="candidate_scope" else "characterization_only" if protocol["mode"]=="characterization_only" else statistical
    verify_sources(manifest)
    result=dict(status=status,statistical_status=statistical,scope_sha256=protocol["scope_sha256"],
        protocol_sha256=protocol_sha256,generator_source_sha256=file_hash(__file__),analysis_module_sha256=file_hash(study.__file__),
        coverage_plan_sha256=protocol.get("coverage_plan_sha256"),
        assumptions="Coverage evidence for the complete recorded synthetic regimes only; no distribution-free guarantee. Insufficient conditional populations count as noncoverage, not favorable exclusions.",
        complete_episode_counts=protocol.get("complete_episode_counts"),count_conditioning=protocol.get("count_conditioning","planned_episode_sampling"),
        minimum_complete_episodes=protocol["minimum_complete_episodes"],simulation_family_alpha=protocol["stage_derivation"]["simulation_alpha"],
        look_weights=protocol["stage_derivation"]["look_weights"],planned_scenario_ids=list(REGIMES),
        scenarios=[dict(id=r,description=REGIMES[r]["description"],**summaries[r]) for r in REGIMES],
        endpoint_mc_interpretation="The first fixed bootstrap stream defines coverage; independently repeated endpoints are a separately reported computational precision gate, never selected for favorable coverage.",
        execution_limit_look=through_look,
        unresolved_regimes=[r for r in REGIMES if summaries[r]["decision"]=="inconclusive"],
        elapsed_seconds=time.perf_counter()-started,manifest_sha256=file_hash(manifest_path),
        trial_ledger_sha256=file_hash(out/"trial_ledger.jsonl"),trial_ledger_chain_sha256=head,
        trials=[dict(path=str(path.relative_to(out)),sha256=file_hash(path)) for path in sorted(out.glob("*/trial_*.json"))])
    if status=="approved":study.validate_calibration(design,{sid:{"matrix":np.empty((n,30))} for sid,n in protocol["complete_episode_counts"].items()},result)
    atomic_bytes(out/"calibration.json",canonical_json(result)+b"\n")
    return result


def benchmark(design, *, seed,trials,out,complete_counts=None):
    """All regimes, fixed trial count, never an approval artifact."""
    records=[]
    minimum={s["id"]:2 for s in design["strata"]}
    for regime in REGIMES:
        for trial in range(trials):
            row=simulate_one(design,regime,seed,trial,minimum,complete_counts)
            records.append(row)
            print(f"{regime} trial {trial}: {row['elapsed_seconds']:.3f}s; noncoverage={row['familywise_noncoverage']}",flush=True)
    times=np.array([r["elapsed_seconds"] for r in records])
    report=dict(kind="CPU_runtime_benchmark_not_coverage_approval",analysis_module_sha256=file_hash(study.__file__),
        generator_source_sha256=file_hash(__file__),scope=study.calibration_scope(design,complete_counts),
        complete_episode_counts=complete_counts,
        seed=seed,trials_per_regime=trials,records=records,mean_seconds_per_trial=float(times.mean()),
        maximum_seconds_per_trial=float(times.max()),total_serial_seconds=float(times.sum()),
        bootstrap_array_bytes=design["analysis"]["draws"]*60*8,
        environment=dict(numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),platform=platform.platform(),blas_threads=1))
    with Path(out).open("xb") as stream:stream.write(canonical_json(report)+b"\n")
    return report


def characterization_design(*,episodes,calls,permutations,draws,repeats):
    """Explicit hypothetical scope for timing/characterization, never a pilot lock."""
    for value,name,minimum in ((episodes,"episodes",2),(calls,"calls",1),(permutations,"permutations",2),
                               (draws,"draws",1),(repeats,"repeats",2)):
        study.integer(value,name,minimum)
    tasks=("PickCube","StackCube","PickSingleYCB","PegInsertionSide")
    groups=[dict(id="rdt170m_four_task",weights={f"a{i}":.25 for i in range(4)}),
            dict(id="authored1b_two_task",weights={f"b{i}":.5 for i in range(2)})]
    specifications=[(f"a{i}",tasks[i],"synthetic_170m_group") for i in range(4)]+[(f"b{i}",tasks[i],"synthetic_authored_1b_group") for i in range(2)]
    design=dict(schema_version=1,kind="paired_global_design",stage="confirmatory_locked",
        purpose="synthetic_hypothetical_scope_only; no_real_checkpoint_gate_or_efficacy_launch_authorization",
        groups=groups,strata=[dict(id=sid,task=task,model=model,
            checkpoint_sha256=object_hash(dict(synthetic_checkpoint_label=model)),
            e01_gate_sha256=object_hash("synthetic_simulation_has_no_G2_approval"),
            reset_seeds=list(range(index*episodes,(index+1)*episodes)),calls_per_episode=calls,random_permutations=permutations)
            for index,(sid,task,model) in enumerate(specifications,1)],
        modalities=list(study.MODALITIES),estimand=study.ESTIMAND,max_episode_steps=400,global_alpha=.05,
        family_weights=dict(primary_rms=1.),secondary_inference="descriptive_only_no_scale_effect",
        episode_dependence="disjoint_reset_streams_across_strata",failure_policy="joint_complete_episodes_no_replacement",
        analysis=dict(method="stratified_episode_percentile_bootstrap",draws=draws,seed=29092026,mc_repeats=repeats,
                      min_tail_draws=100,tail_relative_mcse=.1,endpoint_mc_fraction_h=.1),
        frozen_scales={f"{group['id']}:{modality}":1. for group in groups for modality in study.MODALITIES})
    design["primary_contrasts"]=study.primary_registry(design)
    return study.validate_design(design)


def characterization_protocol(design, *, seed,look_weights,reference_rate,joint_success_targets,complete_counts=None):
    derivation=dict(look_weights=look_weights,simulation_alpha=.05,
        limit=design["global_alpha"]*design["family_weights"][study.FAMILY],
        reference_rate=reference_rate,joint_success_targets=joint_success_targets)
    return dict(schema_version=1,kind="paired_interval_calibration_protocol",mode="characterization_only",
        scope_sha256=object_hash(study.calibration_scope(design,complete_counts)),analysis_module_sha256=file_hash(study.__file__),
        complete_episode_counts=complete_counts,
        count_conditioning="exact_joint_complete_episode_counts" if complete_counts is not None else "planned_episode_sampling",
        generator_source_sha256=file_hash(__file__),regime_definitions_sha256=object_hash(REGIMES),regimes=list(REGIMES),
        stage_derivation=derivation,simulation_stages=derive_simulation_stages(scenario_count=len(REGIMES),**derivation),
        seed=seed,minimum_complete_episodes={s["id"]:2 for s in design["strata"]})


def make_coverage_plan(design, *, seed,look_weights,reference_rate,joint_success_targets):
    """Embed this plan in the global design before collecting confirmation data."""
    template=characterization_protocol(design,seed=seed,look_weights=look_weights,
        reference_rate=reference_rate,joint_success_targets=joint_success_targets)
    keys=("analysis_module_sha256","generator_source_sha256","regime_definitions_sha256","regimes",
          "stage_derivation","simulation_stages","seed")
    return dict(kind="prospective_exact_count_coverage_plan",schema_version=1,
        count_conditioning="exact_joint_complete_episode_counts",**{key:template[key] for key in keys})


def instantiate_coverage_protocol(design,complete_counts):
    """After an effects-free completion audit, bind exact counts to the old plan.

    The global design is not amended. Future calibration/protocol file hashes
    belong in the separately sealed post-run artifact registry.
    """
    plan=design["analysis"]["coverage_plan"]
    protocol=dict(plan,kind="paired_interval_calibration_protocol",mode="candidate_scope",
        scope_sha256=object_hash(study.calibration_scope(design,complete_counts)),
        complete_episode_counts=complete_counts,coverage_plan_sha256=object_hash(plan),
        minimum_complete_episodes=dict(complete_counts))
    return validate_protocol(protocol,design)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run","benchmark"))
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--design-sha256",required=True)
    parser.add_argument("--protocol",type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--workers",type=int)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--through-look",type=int,help="Operational bound, one-based; resume the unchanged protocol for later looks.")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--trials",type=int)
    args=parser.parse_args()
    if file_hash(args.design)!=args.design_sha256:raise ValueError("Design identity mismatch")
    design=study.validate_design(strict_json(args.design))
    if args.command=="benchmark":
        study.integer(args.trials,"benchmark trials"); study.integer(args.seed,"benchmark seed",0)
        counts=None
        if args.protocol is not None:
            if file_hash(args.protocol)!=args.protocol_sha256:raise ValueError("Benchmark protocol identity mismatch")
            protocol=validate_protocol(strict_json(args.protocol),design)
            if args.seed!=protocol["seed"]:raise ValueError("Benchmark changed its fixed data-generation seed")
            counts=protocol.get("complete_episode_counts")
        report=benchmark(design,seed=args.seed,trials=args.trials,out=args.out,complete_counts=counts)
        print(canonical_json({k:report[k] for k in ("mean_seconds_per_trial","maximum_seconds_per_trial","total_serial_seconds")}).decode())
    else:
        if args.protocol is None or file_hash(args.protocol)!=args.protocol_sha256:raise ValueError("Prospectively frozen protocol identity required")
        report=run_calibration(design,strict_json(args.protocol),args.out,workers=args.workers,resume=args.resume,
                               design_sha256=args.design_sha256,protocol_sha256=args.protocol_sha256,through_look=args.through_look)
        print(canonical_json(dict(status=report["status"],elapsed_seconds=report["elapsed_seconds"])).decode())


if __name__=="__main__":main()
