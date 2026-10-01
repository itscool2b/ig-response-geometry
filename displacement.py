"""Conditional solver-depth sensitivity of saved attribution rankings.

The selected positions are fixed by each source attribution's recorded solver
depth. References are recomputed at each evaluated depth with the same saved
initial noise. This measures solver-resolution sensitivity, not denoiser causality
or a universal contraction mechanism. All valid calls are eligible by default.
"""
import argparse
from collections import OrderedDict
import math
import time

import torch

from experiment_io import file_hash, object_hash, tensor_hash
from faithfulness import (EXT_CAM_START, EXT_CAM_END, EvaluationWriter,
                          add_replay_arguments, load_sidecar, perturb_image,
                          perturb_lang, replay_context, replay_inputs,
                          replay_pipeline, topk_mask, preflight_source_replays)
from per_step_attribution import MANISKILL_INDICES


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    add_replay_arguments(p)
    p.add_argument("--solver-steps", required=True, help="Explicit planned comma-separated positive solver depths")
    p.add_argument("--del-grid", default="0,1,5,10,20")
    p.add_argument("--modality", choices=["vision","language","both"], default="vision")
    p.add_argument("--signal-filter", choices=["all","legacy_norm_ge15"], default="all")
    p.add_argument("--no-signal-filter", action="store_true", help="Compatibility alias for --signal-filter all")
    p.add_argument("--selection", choices=["prefix","round_robin_episodes"], default="round_robin_episodes")
    p.add_argument("--reference-norm-min", type=float, default=0.0)
    return p.parse_args()


def displacement(a_pert,a_orig,*,reference_norm_min=0.0):
    """Cast before subtraction; zero/small relative denominators are explicit."""
    if a_pert.shape != a_orig.shape or a_orig.ndim != 3 or a_orig.shape[-1] <= max(MANISKILL_INDICES):
        raise ValueError("Action shapes must agree and contain the declared active coordinates")
    if not math.isfinite(reference_norm_min) or reference_norm_min < 0:
        raise ValueError("Reference norm cutoff must be finite and nonnegative")
    if not torch.isfinite(a_pert).all() or not torch.isfinite(a_orig).all():
        raise ValueError("Nonfinite action tensor")
    perturbed,reference = a_pert.float(),a_orig.float()
    active = perturbed[...,MANISKILL_INDICES]-reference[...,MANISKILL_INDICES]
    full = perturbed-reference
    norm = float(reference[...,MANISKILL_INDICES].norm().item())
    active_norm = float(active.norm().item())
    result=dict(l2_active=active_norm,rms_active=float(active.square().mean().sqrt().item()),
                l2_full=float(full.norm().item()),
                rel_active=active_norm/norm if norm > reference_norm_min else None,
                relative_status="defined" if norm > reference_norm_min else "undefined_reference_norm",
                ref_action_norm_active=norm,active_dimensions=active.numel())
    if any(isinstance(value,float) and not math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite displacement arithmetic")
    return result


def select_calls(rows,limit,method):
    if limit is None or len(rows) <= limit:
        return rows
    if limit < 1:
        raise ValueError("Selection limit must be positive")
    if method == "prefix":
        return rows[:limit]
    if method != "round_robin_episodes":
        raise ValueError("Unknown selection rule")
    groups=OrderedDict()
    for row in rows:
        key=tuple(row[k] for k in ("task","model","seed","episode"))
        groups.setdefault(key,[]).append(row)
    selected,index=[],0
    while len(selected) < limit:
        for group in groups.values():
            if index < len(group):
                selected.append(group[index])
                if len(selected) == limit:
                    break
        index+=1
    return selected


def parse_grid(text,*,solver=False):
    grid=[int(value) for value in text.split(",")]
    if not grid or len(grid) != len(set(grid)) or grid != sorted(grid):
        raise ValueError("Grid must be nonempty, unique, and increasing")
    if solver and min(grid) < 1:
        raise ValueError("Solver depths must be positive")
    if not solver and (grid[0] != 0 or grid[-1] > 100):
        raise ValueError("Deletion grid must start at zero and remain within [0,100]")
    return grid


def main():
    args=parse_args()
    solvers,grid=parse_grid(args.solver_steps,solver=True),parse_grid(args.del_grid)
    if not math.isfinite(args.reference_norm_min) or args.reference_norm_min < 0:
        raise ValueError("Reference norm threshold must be finite and nonnegative")
    if args.no_signal_filter and args.signal_filter != "all":
        raise ValueError("Conflicting signal-filter choices")
    modalities=["vision","language"] if args.modality == "both" else [args.modality]
    rows,source,digest=replay_inputs(args,apply_limit=False)
    n_source=len(rows)
    if args.signal_filter == "legacy_norm_ge15":
        rows=[r for r in rows if r["ref_norm_maniskill"] >= 15]
    n_filter=len(rows)
    rows=select_calls(rows,args.limit,args.selection)
    if not rows:
        raise ValueError("Selected displacement population is empty")
    config=dict(target=args.target,source_pipeline=source["configuration"]["pipeline"] if source else None,
                solver_grid=solvers,nominal_deletion_percent=grid,modalities=modalities,
                signal_filter=args.signal_filter,selection=args.selection,source_rows=n_source,
                rows_after_filter=n_filter,selected_rows=len(rows),
                selected_source_keys=[[r.get("run_id"),r["seed"],r["episode"],r["policy_call_idx"]] for r in rows],
                ranking_source_solver_steps=sorted({r.get("solver_steps") for r in rows},key=str),
                reference_norm_min=args.reference_norm_min,source_code_sha256=file_hash(__file__),
                interpretation="Solver-resolution sensitivity with fixed ranking and saved initial noise; no denoiser-causality claim")
    with EvaluationWriter(args,source,digest,"displacement",config,len(rows)*len(solvers)*len(modalities)) as writer:
        # The source depth need not appear in the study grid. Always authenticate
        # its original action before loading any alternative solver configuration.
        original_pipe,original_lang=replay_pipeline(args,source)
        preflight=preflight_source_replays(args,rows,source,original_pipe,original_lang,writer=writer)
        preflight_hash=writer.write_auxiliary("source_replay.json",preflight)
        del original_pipe,original_lang
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for steps in solvers:
            writer.context={"solver_steps":steps}
            pipe,lang=replay_pipeline(args,source,solver_steps=steps,allow_solver_change=True)
            for row in rows:
                writer.set_context(row,solver_steps=steps)
                started=time.time()
                sidecar,sidecar_hash=load_sidecar(row,args.metrics,source)
                writer.set_context(row,solver_steps=steps,source_attr_sha256=sidecar_hash)
                rank_steps=row.get("solver_steps")
                if source is not None and rank_steps != source["configuration"]["solver_steps"]:
                    raise ValueError("Source row solver depth differs from source manifest")
                ctx=replay_context(args,row,sidecar,pipe,lang,verify_reference=steps == rank_steps,strict=source is not None)
                reference=ctx["ref_action"]
                device=reference.device
                for modality in modalities:
                    writer.set_context(row,solver_steps=steps,modality=modality,source_attr_sha256=sidecar_hash)
                    attribution=sidecar["vision_attr" if modality == "vision" else "lang_attr"].to(device)
                    scores=attribution.float().abs().sum(dim=-1).squeeze(0)
                    if modality == "vision":
                        indices=torch.arange(EXT_CAM_START,EXT_CAM_END,device=device)
                        real_mask=None
                    else:
                        real_mask=ctx["lang_attn_mask"].squeeze(0)
                        indices=torch.nonzero(real_mask,as_tuple=False).flatten()
                    ranked=scores[indices]
                    count=len(indices)
                    order=indices[torch.argsort(ranked,descending=True,stable=True)].cpu().tolist()
                    values={key:[] for key in ("l2_active","rms_active","l2_full","rel_active","relative_status")}
                    counts,mask_hashes=[],[]
                    cached={}
                    with torch.no_grad():
                        for percent in grid:
                            mask=topk_mask(ranked,percent,count)
                            selected=int(mask.sum().item())
                            counts.append(selected)
                            mask_hashes.append(object_hash(sorted(order[:selected])))
                            if selected not in cached:
                                if modality == "vision":
                                    condition=perturb_image(ctx["img_adapted"],ctx["img_adapted_bl"],mask)
                                    perturbed=ctx["seeded_conditional_sample"](ctx["lang_adapted"],condition,ctx["state_traj_actual"]).detach()
                                else:
                                    full_mask=torch.zeros(ctx["lang_adapted"].shape[1],dtype=torch.bool,device=device)
                                    full_mask[indices]=mask
                                    condition=perturb_lang(ctx["lang_adapted"],ctx["lang_adapted_bl"],real_mask,full_mask)
                                    perturbed=ctx["seeded_conditional_sample"](condition,ctx["img_adapted"],ctx["state_traj_actual"]).detach()
                                if selected == 0 and not torch.equal(perturbed,reference):
                                    raise ValueError("Zero-intervention replay differs from its reference action")
                                cached[selected]=displacement(perturbed,reference,reference_norm_min=args.reference_norm_min)
                            measured=cached[selected]
                            for key in values:
                                values[key].append(measured[key])
                    writer.write(dict(event="displacement",task=args.task,model=args.model,target=args.target,
                                      source_replay_verified=preflight["verified"],source_replay_sha256=preflight_hash,
                                      rank_T=rank_steps,del_grid_nominal_percent=grid,selected_counts=counts,
                                      realized_deletion_fractions=[n/count for n in counts],eligible_count=count,
                                      eligible_indices=indices.cpu().tolist(),ranking_indices=order,mask_sha256=mask_hashes,
                                      tie_rule="stable decreasing score, increasing eligible index",
                                      **values,ref_action_norm_active=measured["ref_action_norm_active"],
                                      active_dimensions=measured["active_dimensions"],reference_norm_min=args.reference_norm_min,
                                      reference_sha256=tensor_hash(reference),initial_noise_sha256=tensor_hash(ctx["initial_noise"]),
                                      evaluated_pipeline_identity=pipe["identity"],wall_seconds=time.time()-started))
                del ctx,sidecar
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del pipe,lang
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
