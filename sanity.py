"""Controlled perturbations of authenticated attribution contexts.

Correlations describe sensitivity to the named intervention, not a guarantee of
learned-model grounding. C1_cascade is simultaneous whole-backbone randomization,
not a sequence of layerwise tests. C2 encoding mode and EOS policy are explicit
and never chosen from cache-file availability.
"""
import argparse
from pathlib import Path
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

from experiment_io import file_hash, tensor_hash
from faithfulness import (EXT_CAM_START, EXT_CAM_END, EvaluationWriter,
                          add_replay_arguments, load_sidecar, replay_context,
                          replay_inputs, replay_pipeline, sidecar_image,
                          preflight_source_replays)
from integrated_gradients import integrated_gradients
from per_step_attribution import build_forward_fns, MANISKILL_INDICES


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_replay_arguments(p)
    p.add_argument("--phase", choices=["C1", "C1_frozen", "C1_cascade", "C2"], required=True)
    p.add_argument("--m", type=int, default=None, help="Defaults to source budget; mismatched comparisons are refused")
    p.add_argument("--randomization-seed", type=int, default=12345)
    p.add_argument("--shuffle-seed", type=int, default=None)
    p.add_argument("--c2-mode", choices=["pre_encoding", "post_encoding"], default=None)
    p.add_argument("--eos-policy", choices=["fixed", "shuffle"], default=None)
    p.add_argument("--c2-modalities", choices=["vision", "language", "both"], default="both")
    p.add_argument("--shuffled-language", type=Path, default=None)
    return p.parse_args()


def reinit_last_action_layer(runner, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layer = runner.model.final_layer.ffn_final.fc2
    weight = torch.empty_like(layer.weight, device="cpu", dtype=torch.float32)
    torch.nn.init.xavier_uniform_(weight, generator=generator)
    with torch.no_grad():
        layer.weight.copy_(weight.to(device=layer.weight.device, dtype=layer.weight.dtype))
        if layer.bias is not None:
            layer.bias.zero_()
    return dict(scope="last_action_layer_xavier_weight_zero_bias", seed=seed,
                tensors={"final_layer.ffn_final.fc2."+name:tensor_hash(value)
                         for name,value in layer.named_parameters()})


def reinit_full_backbone(runner, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    changed, retained = {}, []
    with torch.no_grad():
        for name, parameter in runner.model.named_parameters():
            if parameter.ndim >= 2:
                weight = torch.empty_like(parameter, device="cpu", dtype=torch.float32)
                torch.nn.init.xavier_uniform_(weight, generator=generator)
                parameter.copy_(weight.to(device=parameter.device, dtype=parameter.dtype))
                changed[name] = tensor_hash(parameter)
            else:
                retained.append(name)
    return dict(scope="simultaneous_all_backbone_matrices_xavier", seed=seed,
                changed_tensors=changed, retained_one_dimensional_parameters=retained,
                adaptors="unchanged", layerwise_cascade=False)


def pixel_shuffle_image(obs_image, shuffle_seed):
    array = np.asarray(obs_image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise ValueError("Pixel permutation requires an H x W x 3 uint8 image")
    permutation = np.random.default_rng(shuffle_seed).permutation(array.shape[0]*array.shape[1])
    return Image.fromarray(array.reshape(-1,3)[permutation].reshape(array.shape))


def shuffled_language(base, source_artifact, *, mode, seed, eos_policy, shuffled_path=None):
    """Return an explicit language intervention and its reproducible identity."""
    if mode not in {"pre_encoding","post_encoding"} or eos_policy not in {"fixed","shuffle"}:
        raise ValueError("C2 requires an explicit encoding mode and EOS policy")
    if seed is None:
        raise ValueError("C2 requires a shuffle seed")
    original = torch.load(source_artifact, weights_only=True, map_location="cpu")
    if file_hash(source_artifact) != base["identity"]["task_embedding_sha256"]:
        raise ValueError("Language artifact changed after source pipeline loading")
    indices = torch.nonzero(base["lang_attn_mask"].squeeze(0).cpu(), as_tuple=False).flatten()
    if len(indices) < 2:
        raise ValueError("C2 language permutation requires at least two real positions")
    token_ids = original.get("token_ids")
    if not isinstance(token_ids,list) or len(token_ids) != len(indices) or token_ids[-1] != 1:
        raise ValueError("Verified original token IDs with terminal T5 EOS are required")
    if mode == "pre_encoding":
        if shuffled_path is None:
            raise ValueError("pre_encoding mode requires an explicit shuffled-language artifact")
        if eos_policy != "fixed":
            raise ValueError("The current pre-encoding producer supports fixed EOS only")
        if not isinstance(original.get("identity"),dict) or not original["identity"].get("model_revision") or not original["identity"].get("tokenizer_sha256"):
            raise ValueError("Pre-encoding comparison requires original encoder/tokenizer identity")
        shuffled = torch.load(shuffled_path, weights_only=True, map_location="cpu")
        if (shuffled.get("shuffle_mode") != "pre_encoding_content_tokens_eos_fixed"
                or shuffled.get("shuffle_seed") != seed
                or shuffled.get("identity") != original.get("identity")
                or shuffled.get("task") != original.get("task")):
            raise ValueError("Pre-encoding artifact protocol/encoder/task identity mismatch")
        changed_ids = shuffled.get("token_ids")
        if not isinstance(changed_ids,list) or len(changed_ids) != len(token_ids) or changed_ids[-1] != 1 or sorted(changed_ids[:-1]) != sorted(token_ids[:-1]):
            raise ValueError("Pre-encoding tokens do not preserve content multiset and EOS")
        expected_ids = token_ids[:-1].copy()
        random.Random(seed).shuffle(expected_ids)
        if changed_ids != expected_ids+[1]:
            raise ValueError("Pre-encoding tokens do not match the declared shuffle seed")
        if shuffled["embeds"].shape != base["lang_tokens"].shape or not torch.isfinite(shuffled["embeds"]).all():
            raise ValueError("Invalid pre-encoded embedding shape or values")
        if not torch.equal(shuffled["attn_mask"].bool(),base["lang_attn_mask"].cpu()):
            raise ValueError("Pre-encoding permutation changed the attention mask")
        tokens = shuffled["embeds"].to(base["lang_tokens"])
        identity = dict(mode=mode,eos_policy=eos_policy,seed=seed,
                        source_file_sha256=file_hash(source_artifact),intervention_file_sha256=file_hash(shuffled_path),
                        token_ids=changed_ids,correlation_coordinates="sequence_position")
    else:
        if shuffled_path is not None:
            raise ValueError("post_encoding mode does not consume a cached pre-encoding artifact")
        eligible = indices[:-1] if eos_policy == "fixed" else indices
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = eligible[torch.randperm(len(eligible),generator=generator)]
        tokens = base["lang_tokens"].clone()
        tokens[0,eligible.to(tokens.device)] = base["lang_tokens"][0,permutation.to(tokens.device)]
        identity = dict(mode=mode,eos_policy=eos_policy,seed=seed,
                        source_file_sha256=file_hash(source_artifact),
                        destination_indices=eligible.tolist(),source_indices=permutation.tolist(),
                        correlation_coordinates="sequence_position")
    identity["intervention_embedding_sha256"] = tensor_hash(tokens)
    return {**base,"lang_tokens":tokens},identity


def reduce_vision_for_spearman(attribution):
    return attribution.float().abs().sum(dim=-1).squeeze(0)[EXT_CAM_START:EXT_CAM_END].cpu().numpy()


def reduce_language_for_spearman(attribution,real_mask):
    scores = attribution.float().abs().sum(dim=-1).squeeze(0).cpu()
    return scores[real_mask.cpu().bool()].numpy()


def reduce_state_for_spearman(attribution):
    return attribution.float().abs().reshape(-1)[MANISKILL_INDICES].cpu().numpy()


def safe_spearman(a,b):
    """Undefined constant/short vectors return None; nonfinite values are errors."""
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
        raise ValueError("Spearman vectors must have identical one-dimensional shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite Spearman vector")
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    result = float(spearmanr(a,b).statistic)
    if not np.isfinite(result):
        raise ValueError("Unexpected nonfinite Spearman result")
    return result


def main():
    args = parse_args()
    if args.phase == "C2" and (args.shuffle_seed is None or args.c2_mode is None or args.eos_policy is None):
        raise ValueError("C2 requires --shuffle-seed, --c2-mode, and --eos-policy")
    if args.phase != "C2" and any(x is not None for x in (args.shuffle_seed,args.c2_mode,args.eos_policy,args.shuffled_language)):
        raise ValueError("C2 options cannot be silently ignored by a C1 experiment")
    rows,source,digest = replay_inputs(args)
    source_config = source["configuration"] if source else {}
    args.m = args.m if args.m is not None else source_config.get("m",64)
    quadrature = source_config.get("quadrature","trapezoid")
    if args.m < 1 or (source and args.m != source_config["m"]):
        raise ValueError("Sanity comparison requires the same positive quadrature budget as its source")
    pipe,base_language = replay_pipeline(args,source)
    language = base_language
    config = dict(pipeline=pipe["identity"],language=base_language["identity"],target=args.target,
                  m=args.m,quadrature=quadrature,source_replay_required=True,
                  requested_intervention=dict(phase=args.phase,randomization_seed=args.randomization_seed,
                                              c2_mode=args.c2_mode,eos_policy=args.eos_policy,
                                              shuffle_seed=args.shuffle_seed,modalities=args.c2_modalities),
                  source_code_sha256=file_hash(__file__),
                  reference_policy="source_frozen" if args.phase in {"C1_frozen","C1_cascade"} else "recomputed_after_intervention",
                  interpretation="Sensitivity to the specified intervention; no universal rho cutoff or grounding guarantee")
    with EvaluationWriter(args,source,digest,"sanity",config,len(rows)) as writer:
        preflight=preflight_source_replays(args,rows,source,pipe,base_language,writer=writer)
        preflight_hash=writer.write_auxiliary("source_replay.json",preflight)
        intervention = dict(phase=args.phase,randomization_seed=args.randomization_seed)
        if args.phase in {"C1","C1_frozen"}:
            intervention.update(reinit_last_action_layer(pipe["runner"],args.randomization_seed))
        elif args.phase == "C1_cascade":
            intervention.update(reinit_full_backbone(pipe["runner"],args.randomization_seed))
        else:
            intervention.update(c2_mode=args.c2_mode,eos_policy=args.eos_policy,
                                shuffle_seed=args.shuffle_seed,modalities=args.c2_modalities)
            if args.c2_modalities in {"language","both"}:
                language,detail = shuffled_language(base_language,Path(args.lang_dir)/f"{args.task}.pt",
                                                   mode=args.c2_mode,seed=args.shuffle_seed,
                                                   eos_policy=args.eos_policy,shuffled_path=args.shuffled_language)
                intervention["language"] = detail
        intervention_hash=writer.write_auxiliary("intervention.json",intervention)
        for row in rows:
            writer.set_context(row)
            started = time.time()
            sidecar,sidecar_hash = load_sidecar(row,args.metrics,source)
            writer.set_context(row,source_attr_sha256=sidecar_hash)
            observation = sidecar_image(sidecar)
            if args.phase == "C2" and args.c2_modalities in {"vision","both"}:
                observation = pixel_shuffle_image(observation,args.shuffle_seed)
            frozen = sidecar["ref_action"] if args.phase in {"C1_frozen","C1_cascade"} else None
            ctx = replay_context(args,row,sidecar,pipe,language,verify_reference=False,
                                 obs_image=observation,frozen_reference=frozen,strict=source is not None)
            forwards = build_forward_fns(ctx)
            pairs = [(ctx["img_adapted"],ctx["img_adapted_bl"]),
                     (ctx["lang_adapted"],ctx["lang_adapted_bl"]),
                     (ctx["state_input_actual"],ctx["state_input_baseline"])]
            new = [integrated_gradients(f,x,b,m=args.m,quadrature=quadrature) for f,(x,b) in zip(forwards,pairs)]
            correlations = dict(
                vision=safe_spearman(reduce_vision_for_spearman(sidecar["vision_attr"]),reduce_vision_for_spearman(new[0])),
                language=safe_spearman(reduce_language_for_spearman(sidecar["lang_attr"],ctx["lang_attn_mask"].squeeze(0)),
                                       reduce_language_for_spearman(new[1],ctx["lang_attn_mask"].squeeze(0))),
                state=safe_spearman(reduce_state_for_spearman(sidecar["state_attr"]),reduce_state_for_spearman(new[2])))
            writer.write(dict(event=f"sanity_{args.phase}",task=args.task,model=args.model,
                              target=args.target,m=args.m,quadrature=quadrature,
                              source_replay_verified=preflight["verified"],source_replay_sha256=preflight_hash,
                              intervention_sha256=intervention_hash,
                              **{f"spearman_{name}":rho for name,rho in correlations.items()},
                              correlation_status={name:"defined" if rho is not None else "undefined_constant_or_short" for name,rho in correlations.items()},
                              correlation_coordinates="sequence_position",initial_noise_sha256=tensor_hash(ctx["initial_noise"]),
                              perturbed_reference_sha256=tensor_hash(ctx["ref_action"]),wall_seconds=time.time()-started))
            del ctx,sidecar,new
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
