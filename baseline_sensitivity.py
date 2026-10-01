"""Vision-baseline sensitivity on authenticated saved contexts.

Black, gray, and Gaussian-blurred baselines are distinct interventions. Pairwise
correlation describes their observed agreement and does not prove interchangeability
or isolate an attribution mechanism. All arms use the same stored noise/reference.
"""
import argparse
import math
import time

import torch
from PIL import Image, ImageFilter

from experiment_io import file_hash, tensor_hash
from faithfulness import (EvaluationWriter, add_replay_arguments, load_sidecar,
                          replay_context, replay_inputs, replay_pipeline, sidecar_image)
from integrated_gradients import integrated_gradients
from per_step_attribution import build_forward_fns
from sanity import reduce_vision_for_spearman, safe_spearman


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_replay_arguments(p)
    p.add_argument("--m", type=int, default=None)
    p.add_argument("--blur-radius", type=float, default=30.0)
    return p.parse_args()


def encode_6slot_baseline(vision_model, baseline_image, bg_image_encoded):
    pixels = vision_model.image_processor.preprocess(baseline_image,return_tensors="pt")["pixel_values"][0]
    pixels = pixels.unsqueeze(0).to(device=bg_image_encoded.device,dtype=bg_image_encoded.dtype)
    with torch.no_grad():
        tokens = vision_model(pixels).detach()
    if not torch.isfinite(tokens).all():
        raise ValueError("Nonfinite encoded vision baseline")
    bg,baseline = bg_image_encoded.squeeze(0),tokens.squeeze(0)
    return torch.stack([bg,bg,bg,baseline,bg,bg]).reshape(1,-1,vision_model.hidden_size)


def reduce_vision(attribution):
    return reduce_vision_for_spearman(attribution)


def main():
    args = parse_args()
    if not math.isfinite(args.blur_radius) or args.blur_radius < 0:
        raise ValueError("Blur radius must be finite and nonnegative")
    rows,source,digest = replay_inputs(args)
    source_config = source["configuration"] if source else {}
    budget = args.m if args.m is not None else source_config.get("m",64)
    if budget < 1:
        raise ValueError("Quadrature budget must be positive")
    quadrature = source_config.get("quadrature","trapezoid")
    pipe,lang = replay_pipeline(args,source)
    config = dict(pipeline=pipe["identity"],language=lang["identity"],target=args.target,
                  m=budget,quadrature=quadrature,source_code_sha256=file_hash(__file__),
                  baselines=dict(black=[0,0,0],gray=[128,128,128],
                                 blur=dict(operation="PIL GaussianBlur",radius=args.blur_radius)),
                  reference_policy="fixed_source_reference",error_units="relative_fraction")
    with EvaluationWriter(args,source,digest,"baseline_sensitivity",config,len(rows)) as writer:
        for row in rows:
            writer.set_context(row)
            started = time.time()
            sidecar,sidecar_hash = load_sidecar(row,args.metrics,source)
            writer.set_context(row,source_attr_sha256=sidecar_hash)
            image = sidecar_image(sidecar)
            baselines = dict(black=Image.new("RGB",image.size,(0,0,0)),
                             gray=Image.new("RGB",image.size,(128,128,128)),
                             blur=image.filter(ImageFilter.GaussianBlur(radius=args.blur_radius)))
            ctx = replay_context(args,row,sidecar,pipe,lang,strict=source is not None)
            forward,_,_ = build_forward_fns(ctx)
            attrs,diagnostics,baseline_hashes = {},{},{}
            for name,baseline_image in baselines.items():
                baseline = encode_6slot_baseline(pipe["vision_model"],baseline_image,pipe["bg_image_encoded"])
                with torch.no_grad():
                    adapted = pipe["runner"].img_adaptor(baseline)
                result = integrated_gradients(forward,ctx["img_adapted"],adapted,m=budget,
                                              quadrature=quadrature,return_result=True,
                                              diagnostic_context=dict(study="baseline_sensitivity",baseline=name,
                                                                      episode=row["episode"],policy_call=row["policy_call_idx"]))
                attrs[name] = reduce_vision(result.attributions)
                diagnostics[name] = result.diagnostics()
                baseline_hashes[name] = tensor_hash(adapted)
                del result,adapted,baseline
            correlations = {f"rho_{a}_{b}":safe_spearman(attrs[a],attrs[b])
                            for a,b in [("black","gray"),("black","blur"),("gray","blur")]}
            writer.write(dict(event="baseline_sensitivity",task=args.task,model=args.model,
                              target=args.target,m=budget,quadrature=quadrature,
                              **correlations,
                              correlation_status={key:"defined" if value is not None else "undefined_constant_or_short"
                                                  for key,value in correlations.items()},
                              baseline_adapted_sha256=baseline_hashes,baseline_diagnostics=diagnostics,
                              error_units="relative_fraction",
                              initial_noise_sha256=tensor_hash(ctx["initial_noise"]),
                              reference_sha256=tensor_hash(ctx["ref_action"]),wall_seconds=time.time()-started))
            del ctx,sidecar,attrs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
