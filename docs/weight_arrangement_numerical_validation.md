# Numerical validation for the conditional weight-arrangement control

`scripts/validate_weight_arrangement.py` validates the exact two-function E05 procedure before a paired weight-arrangement campaign. It produces diagnostics, not an approval. It does not need completed paired-study output, avoiding a circular dependency between a numerical gate and the study it gates.

The randomized function uses the declared within-Linear-weight entry permutation and its own actual-input reference. Its endpoint scores, fixed-alpha gradients, IG maps, same-stream mean path-gradient maps, and completeness residuals all belong to that function. After the runner tensors are restored and every trained tensor hash is verified, each saved ranking is evaluated under the unchanged trained FP32 probe and the trained reference. RMS and normalized-response criteria use the trained response endpoints. They do not use the randomized function's gap.

Both functions use the same authenticated behavior-policy source cache, lifted BF16-valued image/language representations and weights, readapted FP32 state, explicit noise, strict deterministic runtime, eligible input groups, and intervention grid. This remains an offline probe on contexts visited by the BF16 behavior policy. It is not a native FP32 behavior-policy experiment or an untrained-model comparison.

## Prospective decision and commands

The decision extends the fixed-ladder decision consumed by `validate_fp32_probe.py`: explicit selected episode/call pairs, all three modalities, both Q/L2 targets, at least three consecutively doubled budgets, declared repeated budgets including the finest, fixed-alpha/endpoint/map repeats, source replay flags, grid, gap cutoffs, and selection or independent held-out phase. Held-out episodes must be disjoint from calibration episodes and bind the frozen selection artifact and candidate budgets. No adaptive stopping rule is implemented.

Additional required fields are:

- The six flat scope fields in `weight_arrangement_control.SCOPE`.
- `native_registry_sha256`, `torch_version`, and `implementation_sha256` equal to this wrapper's `implementation_hashes()` mapping.
- `checkpoint_mode` and `checkpoint_identity` matching the source pipeline's mode and complete `checkpoint` identity object, including its byte hash and filename.
- An explicit engineering `parameter_draws >= 2`, `parameter_seed_namespace`, `parameter_master_seed`, and `stratum_id`. Draw identities are fixed per episode and shared across all its selected calls, modalities, and targets. There is no implicit campaign draw count.
- `failure_policy: retain_every_planned_draw_no_replacement` and exact `criterion_thresholds` equal to the wrapper's `THRESHOLDS` mapping: coordinate L1 .01, Spearman .99, top-five overlap .95, relative completeness .01, RMS point difference .01 times trained baseline RMS, RMS area difference .005 times trained baseline RMS, and defined normalized-area difference .01.

The decision is recorded before either process, then passed with its trusted SHA256. Preparation runs under the declared source replay settings. Candidate execution starts in a separate process with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before importing Torch.

```bash
python scripts/validate_weight_arrangement.py prepare \
  --metrics SOURCE_METRICS --bank CONTEXT_BANK \
  --decision-file RECORDED_DECISION --decision-sha256 DECISION_SHA256 \
  --native-registry LOCKED_NATIVE_REGISTRY --lang-dir LANGUAGE_DIRECTORY \
  --checkpoint-path LEXICAL_CHECKPOINT_PATH --out NEW_PREPARATION_DIRECTORY

python scripts/validate_weight_arrangement.py run \
  --metrics SOURCE_METRICS --bank CONTEXT_BANK \
  --decision-file RECORDED_DECISION --decision-sha256 DECISION_SHA256 \
  --native-registry LOCKED_NATIVE_REGISTRY --lang-dir LANGUAGE_DIRECTORY \
  --checkpoint-path LEXICAL_CHECKPOINT_PATH \
  --prepared COMPLETED_PREPARATION_DIRECTORY \
  --prepared-sha256 PREPARATION_COMPLETION_SHA256 --out NEW_VALIDATION_DIRECTORY
```

For pretrained mode the loader may use its pinned repository artifact instead of an explicit checkpoint path. Preserve the authenticated lexical checkpoint filename when a path is supplied. Every output directory must be new. Commands do not choose a draw count, seed, numerical ladder, holdout roster, or approval on the user's behalf.

## Evidence and failure accounting

The top report records the entire planned roster, source/preparation/checkpoint/native structure identities, runtime, conversion, actual parameter registry, every draw descriptor, every restoration record, and every context-by-draw-by-modality-by-target outcome. The native registry has a specifically qualified structural-shell root; all other loaded module classes, names, shapes, tensor identities, and alias restrictions must match the approved control contract.

Each arm saves its own-function reference and endpoint repeats, fixed-alpha gradient tensors, all declared IG/path-gradient tensors, exact diagnostics, cross-budget comparisons, and independent repeat comparisons. After restoration it saves trained endpoint repeats and the existing production response schema, including active actions and hashes for every unique intervention. Own-function completeness and trained-response gaps have distinct field names. Raw nonfinite gradients, actions, or references are saved when available. Maps already computed before a later response failure remain preserved.

Nonfinite numerical outcomes continue to later planned arms, contexts, and draws. A failed randomized reference retains all six arms of that context and draw. Unavailable source calls retain all planned cells and are never replaced. Dtype, shape, source, structure, cache, and restoration identity violations are contract errors: they stop the attempt and leave failure artifacts. A changed new trained reference after verified restoration is a numerical repeatability failure, while a changed source/cache/checkpoint identity remains a contract error.

All-zero maps, tied group scores, zero gaps, and poorly conditioned normalized outcomes remain explicit. A zero gradient is not imputed for a failed gradient. No cohort average can erase a failed cell. The criterion audit reuses the existing v6 numerical calculations with an explicitly trained-endpoint response view and separate randomized-endpoint checks. It never writes an approved gate. A completed report can contain numerical failures or criterion violations.

At completion the wrapper rechecks all declared implementation and input hashes, the prepared-cache completion, runtime settings, full roster accounting, and restored tensors. `completion.json` authenticates every saved file. No GPU campaign is authorized merely by creating this wrapper; a recorded selection protocol, source freeze, review, and separate independent held-out numerical gate are required before authoritative comparative use.
