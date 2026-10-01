# E05 weight-arrangement control

`weight_arrangement_control.py` implements one declared conditional parameter null and a separate paired artifact producer. It permutes the flattened entries of every exact `torch.nn.Linear.weight` under `runner.model`, independently by tensor. It retains each changed tensor's exact learned FP32 bit-value multiset, shape and dtype. Biases, norms, positions, buffers, adaptors and encoders remain fixed. This tests checkpoint-specific linear-weight arrangement conditional on the retained learned quantities. It is not native initialization or a fully untrained policy.

The function remains the offline FP32 probe obtained by lifting the recorded BF16-valued downstream weights and cached image/language representations, readapting state in FP32, and using fixed source noise on BF16 behavior-policy visited contexts. No full native-FP32 checkpoint, encoder or rollout equivalence is implied.

## Two distinct functions

Each parameter draw generates its own actual self-reference and Q-IG/L2-IG maps. Using the trained action as the randomized objective's reference would change the question and is rejected by this contract. Both targets share the same randomized reference, noise, cached inputs, baseline operator and parameter draw.

After map construction, the complete original runner state is restored and hashed. The unchanged trained probe then evaluates every mask selected by either trained or permuted rankings, using the trained reference and the base study's intervention grid. Trained actual-action replay is checked before each parameter intervention and after restoration. Shared trained masks must reproduce the completed base artifact's exact action records, scores and areas. Positive deletion advantage is trained area minus permuted area; insertion reverses that subtraction. Randomized native response magnitudes are not compared with trained response magnitudes as an efficacy test.

The stable tie rule is symmetric: decreasing grouped absolute attribution, then increasing eligible index. Completely zero or tied maps remain finite, degenerate outcomes with that arbitrary index order. They are not replaced by favorable seeds or described as informative attribution. Nonfinite maps have no ranking and remain numerical failures.

The pinned upstream native RDT constructor zeroes the final output Linear weight and bias. For finite internal computations, fixed noise and a conditioning-independent zero output head, the action function is conditioning-independent and own-reference IG is zero. This structural diagnostic is distinct from the selected permutation null. The CPU tests exercise that condition in a small network; they do not claim a new full RDT sampler experiment.

## Parameter identity and draws

The producer requires an independently reviewed exact structural registry. It rejects unexpected module or parameter names, classes, shapes, special Linear wrappers, parameter subclasses, aliases, shared storage and unconverted floating tensors. Root-only registry adaptation is explicit: the inspected native CPU structural shell uses `torch.nn.Module`, whereas the runtime root must be the pinned `models.rdt_runner.RDTRunner`. No other class difference is accepted. The structural inspection proves no checkpoint values; the actual loaded FP32 checkpoint supplies those hashes at runtime.

Each draw is independent by episode, with one coherent backbone shared across that episode's selected calls, modalities and targets. A tensor stream hashes the declared namespace, master seed, stratum, episode, draw index and parameter name. It seeds a local CPU `torch.Generator` for `torch.randperm`; Torch version and stream identity are recorded, and distinct-stream seed collisions stop the run. The intervention does not reseed the global RNG. Draw count is explicit and at least two, with its adopted value justified by variance and cost planning.

The draw ledger records parent/child tensor hashes, sorted FP32 bit-pattern multiset hashes, permutation hashes, every retained tensor hash and the complete registry. Signed zero is distinguished in the multiset digest. Identical-valued weights may be unchanged by permutation and are retained as such. Original tensor values are restored even if a forward calculation raises. Unexpected structural or tensor-object replacement is a fatal contract violation; the module does not pretend it can repair arbitrary external model surgery.

An FP32 CPU copy of every runner tensor is retained for restoration. Sorting changed tensors to verify their exact multisets, hashing state and copying weights also consume CPU time and memory. Measure these costs for planning. There is no unrecorded cheaper permutation rule or outcome-dependent draw replacement.

## Production gate and artifacts

Run `scripts/run_weight_arrangement_control.py` in a fresh process. It sets the required CUBLAS configuration before Torch import and applies the same strict runtime settings as the trained probe. It requires a trusted protocol hash, the completed base artifact, its source run and prepared cache, the native structure registry, and a separate E05 numerical gate. No default budget, draw count or automatic experiment launch is supplied.

```powershell
python scripts/run_weight_arrangement_control.py `
  --metrics path/to/source.jsonl --base path/to/base-paired.jsonl `
  --protocol path/to/e05-protocol.json --protocol-sha256 TRUSTED_PROTOCOL_SHA256 `
  --e05-gate path/to/separately-approved-e05-gate.json `
  --native-registry path/to/reviewed-native-registry.json `
  --prepared path/to/base-preparation --lang-dir path/to/language-cache `
  --out path/to/fresh-e05.jsonl
```

The protocol uses the six flat scope fields exported as `SCOPE`, `parameter_draws`, `parameter_seed_namespace`, `parameter_master_seed`, `stratum_id`, the base-compatible `stage` and `global_design_sha256`, explicit `numerics` for both targets in all three modalities, `runtime_settings`, and `failure_policy=retain_every_planned_draw_no_replacement`. It binds the base output/manifest/completion/protocol, prepared completion, native registry and E05 gate hashes.

The E05 gate must explicitly approve `approved_for_weight_arrangement_transfer`, bind the exact null scope, native structure, helper sources, Torch version and runtime, and list distinct selection and held-out report hashes with independent holdout complete. Its `approved_numerics` must match the production protocol. The current trained-probe E01 gate is insufficient. These checks authenticate a separately reviewed decision; they do not manufacture scientific approval from completed diagnostic files.

Output has one row per planned context, modality and parameter draw. Evaluated rows use the existing paired active-action/response schema, with `trained_Q_IG`, `trained_L2_IG`, `permuted_Q_IG` and `permuted_L2_IG`. Coordinate auxiliaries retain complete own-reference attribution and path-gradient tensors, reference values and hashes. Parameter and restoration auxiliaries bind each coherent draw. Response artifacts retain each unique trained intervention's active action once within the row. There is no change to the base artifact.

`load_completed_control(path, base_path=...)` authenticates both completed artifacts, every registered auxiliary, the entire planned context/modality/draw roster, the registry's changed/retained scope, draw stream identities, coordinate-to-ranking agreement, cross-modality own-reference agreement, and all shared trained anchors. Missing or corrupted records stop analysis. Numerical failures remain planned rows with causes and available raw failure tensors. An unavailable base outcome is labeled explicitly; it is never regenerated into a substitute outcome. A failed randomized reference produces failures for all three modalities of that context/draw and later planned work continues.

The E05 family has its own joint-complete episode population across all selected calls, three modalities, two targets and all fixed draws. It does not delete episodes from the base family or average only successful draws. Draw covariance must be computed from each draw's episode-average contrast vector before dividing by draw count. Global family weights, intervals and variance-only planning belong to the separately locked study analysis. The producer does not infer them from observed efficacy.

## Separate randomized-function numerical validation

Reuse the authenticated cache/runtime boundary, exact IG kernel, endpoint/map/repeat comparisons and saved-action evaluator. Do not call the existing `validate_fp32_probe.validate_target` unchanged: it uses one action function for both gradients and intervention responses. E05 requires the randomized own-reference function for gradients and the restored trained function for response curves.

A separate wrapper must preserve that distinction, all planned context/draw/modality/target outcomes, tied/zero-map diagnostics, and selection versus independent holdout membership. It must assess repeatability, coordinate/rank stability across budgets, own-reference agreement and common trained-response stability. Finite numerical caps and failures must remain visible. The CPU fixtures validate implementation and accounting only. They provide no RDT randomized-function numerical acceptance or empirical transfer claim.
