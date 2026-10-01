# Saved numerical case study

This bundle reports the preserved sealed FP32-probe audit labeled 2026-10-01 03:36 UTC. Its 12 completed contexts are a subset of 21 source-available and 24 planned contexts. It is not a final census at the later compute stop. Later completion is unknown and scientifically unassessed by this bundle.

`inputs/sealed_v6.json` is a deterministic projection of authenticated saved evidence, not a new model run. It retains all 144 planned target/modality rows, all recorded checks and observations, bank membership, protocol and queue identities, retry receipts and earlier overlapping audit identities. Workspace prefixes are replaced by the literal `artifact-root/`. No numerical values, source availability or attempt status are changed. Runtime values declared for source replay remain qualified as exact-reference replay, rather than reconstructed historical settings.

The complete raw artifact sets for all 72 completed arms were verified by the original sealed auditor. They are not all included locally or in this projection. The public calculation therefore reproduces the reported audit counts and metrics, not the full original raw-model audit. Eight locally retained YCB reports and 100 saved state-gradient/map artifacts were independently hash-checked during extraction. Their FP32 tensor values are included with original tensor hashes; the reproducer checks those hashes directly without Torch. The two state-input projections match tensor hashes in their sealed full-context receipts. The full context files themselves are not copied.

From this directory, reproduce all JSON, CSV, Markdown and TeX outputs with Python standard library only:

```text
python summarize.py --input inputs/sealed_v6.json --output reproduced
python -m pytest test_summary.py -q
```

`extract_local.py` is the optional provenance bridge from the original preserved evidence. It requires CPU Torch to decode already-saved tensors, hides CUDA, and neither imports a model nor accesses a network. Re-extraction requires the original archive and protocol files; public table regeneration does not.

```text
python extract_local.py --evidence PATH_TO_PRESERVED_EVIDENCE --source-repo PATH_TO_SOURCE_GIT --output inputs/sealed_v6.json
```

`outputs/manifest.json` binds the input, reproducer source and every generated output. `numerical_roster.tex` is generated from `strata.csv`; `numerical_diagnostics.tex` is generated from `modality_target.csv`. Their comments include the exact CSV hashes. The roster table's last column is complete/planned cells; every context has six target/modality cells. The diagnostics table uses all 12 completed arms per modality/target, and displays maxima as percentages. Its violation column counts arms with at least one violation comparing the two finest budgets. Residual is finest-budget relative completeness; RMS is the maximum absolute curve change divided by baseline RMS across both IG/path-gradient rankings and deletion/insertion. These maxima can occur in different contexts.

The source collection protocol selected calls 0 and 12 prospectively, with caps of 13 policy calls and 400 environment steps. This is not uniform sampling over a full completed trajectory. The vision/language ladder is 128, 256, 512 intervals; state is 256, 512, 1024. A trapezoidal budget of m intervals evaluates m+1 nodes. All original threshold values and undefined cases remain in the input. No setting is selected or certified by these summaries.

The descriptive gradient-geometry comparison uses the saved active-reference shape `[1,64,8]`, giving 512 active entries. Unit sigma squared is verified from the exact `common_scores` function in the pinned Git blob `d9e8901:paired_comparison.py`, whose SHA256 matches the completed queue source manifest. The shape, active indices, full-reference hash and inspected source snippet are retained in the projection. No shape or scale constant is inferred from a favorable result.

The earlier factor-control v4 audit is included only as provenance with its own hash and coverage counts. It is a different numerical comparison contract and is not pooled with the exact FP32-probe observations. Earlier v6 snapshots overlap the present roster; their already-complete report hashes are checked unchanged rather than counted again.

The retained scope supports response-geometry theory plus descriptive numerical findings. It does not establish attribution superiority, learned-weight specificity, policy efficacy, scale generalization or native FP32-policy behavior. `outputs/review.md` states the observed failures and limits in detail.
