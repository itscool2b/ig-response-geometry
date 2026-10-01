# Saved-action sensitivity to executed source positions

`execution_support_sensitivity.py` is a CPU-only descriptive analysis of completed paired study artifacts. It asks whether the same fixed rankings give different perturbation responses when the output includes only predicted positions submitted to `env.step` during the source rollout. It does not run a model, change the rankings, replace the full-chunk RMS primary outcome, or add inferential claims.

The calculation protocol must be recorded before inspecting these comparisons. The initial revision protocol is `execution_support_sensitivity_protocol_v1.json`, SHA-256 `f3f00ebb188012920baf9092dd37ed1ee12bf78591ce7a836b34290b75de0ecb`, recorded at `2026-10-01T01:45:39Z` in the private execution evidence. The public implementation and synthetic tests are available independently of those private records. Its hash authenticates a calculation plan, not completed scientific results. No paired empirical results were used to choose the implementation or its synthetic fixtures.

## Authenticated execution membership

The supported collector source has SHA-256 `9bd7a95b3fd48b624f2a79a35c89eef46fddeeab023e0514674efc575fad9bf7`. It stores each call before executing that call's actions. A source reference has shape `[1,64,128]`; the eight active dimensions are `[0,1,2,3,4,5,6,10]`. After affine denormalization, the collector sends predicted positions `0,4,...,60` to the environment, stopping when the environment terminates or truncates, or the collector reaches its declared step cap.

For each selected call, the next call's `env_step_at_call` minus the current counter gives the number of submitted commands. The final call instead uses the committed terminal `env_steps`. Every nonfinal call must contain 16 commands; the final call may contain 1 through 16. For example, a final call lasting three environment steps uses positions `[0,4,8]`. The analysis requires all consecutive source call records, committed terminals, exact counter/stop-flag agreement, source-reference shape, source/sidecar hashes, and the matching frozen context bank. Unknown collector code or inconsistent provenance stops the analysis. It never guesses a prefix.

These positions identify commands submitted to `env.step`. They do not measure actual robot motion or rule out controller or simulator clipping. The saved FP32 probe outputs were not the original BF16 rollout commands. Responses remain in the same normalized action coordinates as the primary analysis, rather than physical joint units. Both successful and unsuccessful tasks are retained when their source episode transactions completed.

## Response and population rules

The module preserves the producer's saved full-chunk FP32 RMS areas. Separately, it recomputes two descriptive response curves from the saved FP32-valued active actions and reference: all 64 predicted positions, and the authenticated executed source positions. Both use float64 subtraction and reduction, with RMS equal to the square root of mean squared residual over the selected positions and eight active dimensions. Comparing those two curves therefore isolates output support under common arithmetic.

Every curve uses its original intervention masks and realized fraction grid, including repeated fractions. Areas use the trapezoidal raw integral. There is no endpoint normalization or clipping. Exact random controls average RMS responses across subsets after computing each subset's RMS. Sampled controls retain each permutation curve and average its area within context. Taking RMS after averaging actions would answer a different question and is not used.

The same five core contrasts, three modalities, and two intervention directions are summarized within each stratum. Positive effects use method minus control for deletion and control minus method for insertion. Selected calls receive equal weight within each episode; episodes receive equal weight. The summary uses exactly the primary study's joint-complete episode population across all selected calls, primary methods and three modalities. Removing unexecuted positions cannot rescue a primary numerical failure. Failed rows, planned memberships, conditioning and exclusion causes remain visible. No cross-stratum pooling, intervals, p-values or new significance labels are produced.

Signed outputs are refused for `variance_only_pilot` artifacts. This prevents this descriptive analysis from exposing comparative pilot means. A completed `confirmatory_locked` artifact and uniform executed-call bank are required; that stage label alone does not establish approval of the underlying numerical or statistical protocol.

## Descriptive drift

For both supports, the module authenticates the source reference and stored probe reference, computes their RMS displacement in float64, and divides it by the same support's all-baseline perturbation RMS. That baseline response is modality-specific. The output retains numerator and denominator. An exactly zero denominator is `zero_denominator`; a positive denominator at or below `1e-8` is `nearzero_denominator`. Both produce a null ratio. A failed baseline produces `unavailable`. No epsilon is inserted.

The producer-reported action and state drift scalars are also preserved. If the exact authenticated prepared cache is supplied, the state ratio divides the producer-reported FP32 state-token L2 drift by the float64 L2 norm of the cached source token. This mixed arithmetic is labeled. An exactly zero norm makes the ratio undefined. Without the prepared cache, absolute state drift remains available and the ratio remains explicitly unavailable. Small ratios do not prove derivative, ranking, or behavioral equivalence.

## Invocation and integrity

Provide SHA-256 values from the trusted evidence registry, including the paired completion hash that binds its manifest and auxiliary files. Do not infer trust merely by hashing unknown input files at analysis time.

```powershell
python execution_support_sensitivity.py `
  --study path/to/paired.jsonl --study-sha256 TRUSTED_OUTPUT_SHA256 `
  --study-completion-sha256 TRUSTED_COMPLETION_SHA256 `
  --source path/to/metrics.jsonl --source-sha256 TRUSTED_SOURCE_SHA256 `
  --protocol path/to/execution_support_sensitivity_protocol_v1.json `
  --protocol-sha256 TRUSTED_PROTOCOL_SHA256 `
  --prepared path/to/authenticated-preparation `
  --out path/to/new-sensitivity.json
```

`--prepared` is optional. The source export must retain its adjacent `.run` store; the paired output must retain its manifest, completion and registered auxiliaries. The analysis authenticates and rechecks them, records analysis/helper hashes and each original paired-row hash, and writes a fresh exclusive output. It cannot overwrite a prior result.

The CPU tests use the actual collector transaction and paired-action artifact machinery with synthetic actions. They cover early truncation, step and call caps, invalid provenance, repeated grids, exact subset expectations, nonfinite outcomes, zero/nearzero ratios, equal episode weighting, pilot refusal, completion/sidecar corruption, and per-modality prepared-cache identity. These are engineering validation, not empirical robustness evidence.
