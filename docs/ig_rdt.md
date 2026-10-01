# RDT attribution: scalar target and representation

Updated October 1, 2026. The active path uses `pipeline.py`, `per_step_attribution.py`, `rdt_sampling.py` and `per_step_ig.py`. Execution of the historical one-observation demonstration `ig_rdt.py` is retired. Its exact source is preserved in `legacy/2026-10-01/ig_rdt.py.txt`, with a separate dated manifest. The root entrypoint's `--status` verifies that preserved source without loading a model. Its original plots and timings do not validate the revised runtime. Notebook cells that call the old demonstration remain historical examples.

## What is attributed

RDT produces a normalized 64-step action chunk through a selected denoising solver. Attribution differentiates through that trajectory with the initial latent tensor fixed. One noise draw is not generally the policy's conditional expectation. Upstream `prediction_type="sample"` means the denoiser predicts a clean sample rather than noise; it does not calculate a policy density.

Let a_ref be the detached sample at the actual observation and a(z) the sample at a perturbed representation using identical noise. The historical name `logpi` denotes the auxiliary score

```
Q(z) = -sum_j (a_j(z)-a_ref,j)**2 / (2*sigma_sq*D).
```

There are 64 horizon positions times eight selected joint coordinates, so D=512 here. The default `sigma_sq=1` is a score scale. Q is a Gaussian log-kernel per action entry, not the diffusion policy's log likelihood. Score changes are not measured changes in policy log probability. Actions are cast to fp32 before subtraction and reduction.

Other targets are negative stabilized L2, negative squared L2, negative maximum absolute deviation and cosine similarity. Stabilized L2 is `-sqrt(sum(diff**2)+1e-12)` and equals -1e-6 at its reference. Squared L2 is a positive constant multiple of Q, giving identical exact-IG rankings for fixed conditions. The nonlinear L2 transform can change rankings; whether it improves them requires a common-response comparison. Rescoring a fixed ranking with another response demonstrates score sensitivity.

## Representation boundaries

The selected observation pipeline uses one current external-camera image in slot 3. The other five slots are constant background. It does not implement a full six-camera/history wrapper.

| Modality | Free variable | Reference / fixed conditions |
|---|---|---|
| Vision | Post-image-adaptor embeddings, 4374 tokens | Gray image in slot 3; five background slots match actual input |
| Language | Post-language-adaptor embeddings | Encoded `[PAD, EOS]` outputs padded to task length; actual task attention mask fixed |
| State | Raw normalized 128-coordinate state | Zero raw state, fixed embodiment mask, state adaptor inside the closure |

Vision IG is not pixel IG through SigLIP, and language IG is not IG through T5 or token IDs. State indices `[0,1,2,3,4,5,6,10]` identify seven joints and the gripper. Inactive coordinates have zero attribution because displacement is zero, even if the partial derivative is nonzero.

A straight path before a nonlinear adaptor generally maps to a curve after it. A separate straight path between adapted endpoints answers a different question. Equal completeness totals do not imply equal decompositions. Raw-state IG is the chosen per-joint definition, not a theorem excluding all other valid constructions. Implementation invariance does not equate different representation paths.

## Language reference and masks

T5 has no BOS token. `baseline_bos_eos.pt` is a legacy name for `[PAD, EOS]`, IDs `[0,1]`. Both positions are encoded with attention enabled; subsequent positions are padded. These tokens have learned semantics and are not guaranteed information-free.

The real task attention mask is held fixed for both endpoints and all path points. Padded baseline positions beyond its two tokens but within the task's attended span remain attended. The baseline's separately saved mask is not substituted. Holding structural metadata fixed defines this study's scalar function; it is not a universal restriction on every possible IG study.

Fresh language encoding creates identified new conditions. It does not recover missing historical embeddings or converter outputs.

## Noise and differentiability

`rdt_sampling.py` adapts [the pinned official RDT sampler](https://github.com/thu-ml/RoboticsDiffusionTransformer/blob/cd79363a1387e8f81c7724d070ef7e45fd23150f/models/rdt_runner.py), preserving its MIT attribution. It accepts a stored latent tensor, creates a fresh scheduler per call and rejects unsupported stochastic solver variants. Dedicated noise generation avoids reseeding the global RNG during IG. Saved values and their hash establish identity; a seed alone cannot establish identical noise across runtimes or dtypes.

The episode driver reuses the episode seed when generating each context's saved draw and records that policy. This differs from drawing new policy noise on every action chunk. Noise-sensitivity studies require separately specified draws and interpretation.

The reference sample is detached. The actual state token stays frozen during vision and language IG as well as reference generation. During state IG, its adaptor runs anew at each path point. Non-reentrant checkpointing trades computation for activation memory; actual memory use and gradients need validation in the chosen runtime.

## Numerical and historical interpretation

The revised core defaults to fp32 master arithmetic and trapezoidal quadrature while model forwards may remain bf16. See [the numerical contract](integrated_gradients.md). Larger m and a larger model are not guaranteed remedies, and completeness does not prove map accuracy.

The old m=300 one-observation example recorded:

| Modality | Relative residual | Signed endpoint gap |
|---|---:|---:|
| Vision | 9.06% | 0.0481 |
| Language | 5.11% | 0.0019 |
| State | 6.12% | 0.0749 |

Its active-action norm was 9.4697. A preceding placeholder-image example recorded residuals 8.93%, 6.97%, 7.93% and norm 8.6013. Language baseline development also reports 13.71% before a reference change. These remain historical observations with different conditions; they do not isolate integration error, forward precision, reference geometry or solver mechanism. They have not been regenerated by this documentation edit.

The active loader requires explicit checkpoint mode and records validated key/shape, model, embedding and upstream identities. See [per_step_ig.md](per_step_ig.md). Historical training/replay/evaluation scripts have separate unresolved contracts. This page does not certify their recorded training losses, success rates, memory claims or current compatibility. Missing historical patched source and artifacts cannot be replaced by calling a fresh run a recovery.
