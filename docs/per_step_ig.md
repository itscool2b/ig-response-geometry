# Per-policy-call attribution and stored runs

Updated September 30, 2026. This page describes revised collection. Historical `data/metrics_*.jsonl` files and shared sidecar names remain archival inputs and cannot be resumed by the new writer.

## Components and execution

`pipeline.py` loads explicitly identified RDT, SigLIP and language artifacts. RDT-1B requires a selected `pretrained`, `authors` or `lora` mode rather than file-presence fallback. `per_step_attribution.py` prepares contexts and scalar closures, returning fp32 maps, numerical metadata, reference actions and noise. `rdt_sampling.py` runs the explicit-noise differentiable solver. `experiment_io.py` manages run manifests and episode transactions. `per_step_ig.py` connects them to ManiSkill.

Prepare the environment, model and embeddings using the current README. Specify experimental selection, budgets and acceptance criteria before scientific collection. This is a command example, not a claim that this budget is sufficient:

```bash
python per_step_ig.py --task PickCube-v1 --model 170m \
  --episodes 1 --max-policy-calls 1 --m 16 --quadrature trapezoid \
  --out runs/engineering-example/metrics.jsonl
```

Use a new output path. Resume by repeating the same identified settings with `--resume`. A changed checkpoint, seed, target, budget, quadrature, source or other identified setting needs a new run. For 1B, supply `--checkpoint-mode` and, in authors/LoRA mode, an explicit `--checkpoint-path`. `--help` defines the current CLI.

The automatic destination is `runs/<unique-run-name>/metrics.jsonl`. Its sibling `metrics.jsonl.run/` contains the manifest, immutable committed episode documents, and attempt directories with sidecars, progress and failures. JSONL is a derived export. Interrupted attempts remain preserved without becoming duplicate successful records. Resume verifies committed artifacts before skipping episodes and reconstructs the derived export.

## Observation and action scope

Collection uses `obs_mode="state_dict"`, `render_mode="rgb_array"`, `control_mode="pd_joint_pos"` and a 400-step maximum. Proprioception is `obs["agent"]["qpos"][0,:8]`. The rendered current image is resized for SigLIP; five image slots remain constant background even though the loop maintains a deque.

The reference action has shape `(1,64,128)`. Eight coordinates are selected, converted with the recorded controller bounds, and subsampled with `[::4]` to at most 16 environment steps. This is subsampling, not interpolation. Interpolating a chunk attribution would not create a valid per-environment-step explanation without defining that new target.

The local 170M configuration uses hidden size 1024, depth 14 and 32 heads. The early 16-head specification did not match the checkpoint. Available 170M and 1B configurations differ in scale and training, so their comparison cannot isolate either cause.

## Stored evidence

Step records include episode/seed/call, context identity, target, m, quadrature, solver steps, noise hash, modality endpoint gaps, sums, relative residuals and numerical diagnostics. Sidecars preserve maps, observation, proprioception, reference sample, noise, masks and sampler metadata. File hashes bind records to sidecars; the manifest binds code/model/language/environment identity.

Maps stay fp32. A zero gap has null relative residual with its absolute residual retained. Nonfinite failures are recorded as failed attempts. Small scalar residuals do not establish accurate rankings. See [the core contract](integrated_gradients.md) and [RDT target semantics](ig_rdt.md).

A `(1,4374,2048)` fp32 vision map occupies about 35.8 MB before serialization overhead. This is why tensors stay outside the JSONL. Runtime and peak memory depend on the selected model, checkpointing, dtype and hardware; old timing estimates are not current benchmarks.

## Historical distinctions

The old writer relied on episode-end presence, retained partial rows and could overwrite sidecars shared by campaigns. The new contract neither authenticates old pairings nor repairs missing old sidecars.

Some original m64 episodes used shorter task-default limits; later 400-step episodes and first-four-call m128 subsets are different populations. Unmatched differences cannot be attributed to budget alone. Earlier controller-bugged records were discarded. TurnFaucet was excluded before the controller correction and the preserved record documents no corrected-controller rerun. The omission limits scope; it does not establish corrected-task impossibility.

The old claims that low initial action norm defines a numerical noise floor, that the IG implementation had no bug, or that larger m/1B necessarily repairs completeness are withdrawn. The audit established bf16 coordinate error analytically. The effect on particular historical RDT maps requires recovered matched inputs or explicitly identified new validation.
