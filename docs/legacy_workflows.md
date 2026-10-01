# Legacy workflow disposition

Updated September 30, 2026. The retained research workflow uses explicitly identified published pretrained or authors' checkpoints. Project retraining is not required for that scope. Earlier evaluation, replay and local LoRA results do not acquire valid provenance because current loading and storage were repaired.

## Exact source preservation

In the full repository, `legacy/2026-09-30/manifest.json` records 27 already-public source files, 151,241 bytes, their original paths, source commit and SHA-256 hashes. The copies end in `.txt` to distinguish preserved source from current commands. No private original was added to the public archive. Old data, notebooks, photographs, figures and report artifacts remain historical assets. Rights exceptions in `NOTICE` still apply.

The anonymous CPU research supplement excludes this historical archive and the retired workflow entrypoints. This page records their disposition; the commands and archive paths below describe the full repository, not additional files bundled in that supplement. Use the supplement's reproduction guide for its supported saved-data and CPU commands.

| Files | Current disposition |
|---|---|
| `eval_maniskill.py`, `finetune_rdt.py`, `generate_demos.py` | Execution retired; exact sources preserved. The evaluator did not apply the intended PEFT adapters, replay provenance is incomplete, and training checkpoint/accounting support was insufficient. |
| `ig_llava.py`, `verify_models.py` | Execution retired. The quantized image-feature API and checkpointing contract were not validated; a model-loading smoke is not research verification. |
| `analyze_month4.py`, `bootstrap_ci.py` | Historical entrypoints retired. Use the strict `analysis.revision` analyzer and verifier. |
| `make_month4_figs.py`, `make_month5_figs.py`, `make_paper_figs.py` | Historical figure generators retired. Their fixed destinations and historical populations/constants are preserved; current assets use `scripts/build_revision_assets.py`. |
| `patch_nb_alttarget.py` | One-off historical notebook mutation retired. |
| `scripts/run_all.sh`, `scripts/run_paper_experiments.sh`, `scripts/run_target_ablation.sh`, `scripts/run_verification.sh` | Historical campaigns retired. They must not launch append jobs or return apparent execution success. |
| `scripts/run_full_pass.sh`, `scripts/run_faithfulness.sh`, `scripts/run_sanity.sh`, `scripts/run_displacement.sh`, `scripts/run_overlays.sh` | Replaced with explicit single-job wrappers around current entrypoints. The old full-pass name no longer chooses a campaign. |
| `generate_overlays.py`, `overlays.py`, `make_annotations.py`, `audit.py` | Repaired for authenticated current inputs, identifiable fresh outputs and failing integrity checks. |
| `ig_resnet.py`, `ig_vit.py`, `ig_tinyllama.py` | Optional primitive demonstrations retained with explicit inputs, budgets and fresh output paths. Current GPU runs are not certified by historical figures. |

Retired entrypoints print a reason and exit 2. An explicit `--status` only verifies their preserved source hash and prints the disposition, exiting 0 without running a workflow:

```bash
python eval_maniskill.py --status
bash scripts/run_all.sh --status
```

## Explicit current jobs

Each wrapper requires a protocol JSON with a nonempty `decision_id`, and records its hash, entrypoint hash, command and destination in a fresh invocation document. This checks protocol identity, not scientific adequacy or external registration. The caller must set selection, budgets, acceptance criteria and amendments before running. No wrapper chooses episode counts or an IG budget.

The following is an engineering command example, not a recommended scientific budget:

```bash
bash scripts/run_full_pass.sh --decision-file /path/to/protocol.json --dry-run -- \
  --task PickCube-v1 --model 170m --episodes 1 --max-policy-calls 1 \
  --m 64 --seed-base 42 --target logpi --quadrature trapezoid \
  --lang-dir /path/to/identified-embeddings --out /fresh/run/metrics.jsonl
```

Review the emitted command, then remove `--dry-run` to execute the declared job. Relative file arguments resolve from the caller's directory. Paths containing spaces are preserved. The destination must be new and outside preserved `data`, `out`, `output`, `paper`, `legacy` and revision-result directories. Only collection permits explicit `--resume`, subject to its exact configuration/artifact checks. A requested 1B job must select checkpoint mode, plus an explicit path in authors/LoRA mode. An identified LoRA loader does not authenticate the old local adapter.

Faithfulness and sanity wrappers require explicit task, metrics and output. Sanity also requires phase/randomization seed; C2 requires perturbation mode, EOS policy and shuffle seed. Displacement additionally requires solver budgets, deletion grid, modality, selection and signal filter. Use the underlying current entrypoint's `--help` for valid values. These are new observations and do not reconstruct lost historical contexts.

## Display and annotations

```bash
python generate_overlays.py --metrics /path/to/metrics.jsonl --task PickCube-v1 \
  --out /fresh/figures --three-panel
python make_annotations.py --metrics /path/to/metrics.jsonl --out /fresh/annotations.csv
python audit.py analysis/revision/results/2026-09-30-v2
```

Overlays require authenticated sidecars and the recorded one-current-camera/five-background-slot observation pipeline. They display the stored 384 by 384 observation and its 27 by 27 patch map in the same extent. The display convention is absolute value after summing signed hidden coordinates, separately max-normalized; cancellation and loss of sign are explicit. This is a representation map, not pixel IG. Language masks may have holes; labels default to exact sequence positions unless an explicitly supplied embedding file matches its recorded hash.

Paths include run and context identity. The manifest retains model/task/seed/target/budget/configuration identity, source and sidecar hashes, rendering code hashes, selection rule, PNG hashes and failure status. A `--limit` selects an explicitly recorded input prefix. Existing figures are never overwritten. Episode summaries use actual policy-call indices; selected calls are not new environment-step explanations.

Annotations export every supplied authenticated call with run/context/seed/model/checkpoint/configuration identity. Their labels describe relative completeness residuals only. A null relative residual is undefined, and a largest residual is not a behavioral failure or attribution-quality verdict. The old `out/annotations.csv` and its categories remain historical. The audit now delegates to the strict reconciled-analysis verifier and exits nonzero on missing or inconsistent integrity artifacts.

## Training and evaluator corrections

The archived training loop counts microbatches as steps. With `MAX_STEPS=5000` and `GRAD_ACCUM=8`, a complete uninterrupted loop performs 625 optimizer updates; recorded step 3000 corresponds to 375 updates. The logger accumulates the loss after dividing by 8, then averages those scaled values. Thus the recorded loss around 0.0001 corresponds approximately to 0.0008 on the unscaled microbatch-mean convention, subject to printed rounding. This is an accounting reinterpretation, not a new training measurement or evidence of useful task performance.

Saved training files contain adapter tensors and the step number, without a sufficient base/configuration identity, optimizer state, RNG state or complete resume contract. The old evaluator constructs the backbone with `pretrained=None` and inserts adapter keys into an ordinary state dictionary without constructing/applying PEFT adapters. A printed checkpoint step does not prove that the intended learned function ran. Missing-checkpoint text claiming pretrained fallback was also unsupported by that construction. Historical poor rollout outcomes cannot therefore diagnose adapter quality from those messages alone.

The replay generator copied success from the source trajectory rather than verifying the rendered replay's terminal success; it did not bind the controller/state trajectory and immutable HDF5 identity. Missing original demonstration data and local checkpoint identity remain unresolved. The current checkpoint contract and newly collected runs do not retroactively validate them. Future training/replay support requires an identified base and adapter configuration, source data and controller/state checks, correct update/loss accounting, saved optimizer/RNG/configuration state, and separately declared validation. No new fine-tuning campaign was run for these corrections.

## Optional primitive demonstrations and verification limits

```bash
python ig_resnet.py --image /path/to/authorized-image.jpg --m 64 --out /fresh/resnet.png
python ig_vit.py --image /path/to/authorized-image.jpg --m 64 --out /fresh/vit.png
python ig_tinyllama.py --prompt "The capital of France is" --m 64 --out /fresh/language.png
```

These programs use the revised IG core and its trapezoidal default. Vision figures now display the actual normalized model input after inversion, preserving the model's resize/crop geometry; ViT labels describe pixel coordinates. The language example uses explicit evaluation mode and checks the PAD reference. Historical residuals and figure files were not regenerated. The original photograph's rights provenance remains unresolved; supply an image you are authorized to use.

Eleven CPU tests cover archive verification and exit semantics, coordinate geometry, masked positions, signed cancellation/nonfinite display failures, synthetic authenticated rendering, annotation identity, malformed inputs, overwrite protection, protocol arguments and paths with spaces. The read-only audit of `analysis/revision/results/2026-09-30-v2` checked 10 artifact hashes, 284 results, 57 populations and 22,966 valid physical records. These checks do not establish real-model primitive runtime, new training quality, recovered historical provenance, behavioral faithfulness or completion of every research gate.
