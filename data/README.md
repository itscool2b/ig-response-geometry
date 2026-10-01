# Preserved records and current analysis

The 99 JSONL files in this directory are immutable historical evidence. Their
current statistical interpretation is the September 30, 2026 revision in
[`analysis/revision`](../analysis/revision/README.md), with canonical outputs in
[`2026-09-30-v2`](../analysis/revision/results/2026-09-30-v2/). No historical raw
record was changed to obtain those results.

The original filename inventory is retained in
[`README-historical-2026-09-30.md`](README-historical-2026-09-30.md). Its scientific
interpretations and former table references are historical and superseded by
this guide, the current analysis registry, and the revised manuscript. In
particular, matching a scalar fingerprint does not authenticate a context or
checkpoint; nominally matched filenames do not establish paired experiments;
and the old AUC thresholds are descriptive conventions, not universal
faithfulness criteria. Old campaign wrappers have been retired. They cannot
recover the missing original tensors by rerunning an environment.

The current paper also uses the separate [paired saved-curve rescoring](../analysis/paired_rescoring/README.md)
and [exploratory episode-influence supplement](../analysis/paired_rescoring/influence_results/2026-10-01-v1/methods_results.md).
The pairing holds each saved ranking and its interventions fixed while changing
the response. It is not a prospectively collected comparison between attribution
targets on authenticated identical contexts.

## Authenticity, duplicates, and missing evidence

The input manifest authenticates each complete original file against the
recorded Git bytes, with an explicitly enumerated Windows CRLF representation.
The ledger accounts for 22,967 physical lines: 22,966 valid records and one
known NUL-corrupted line quarantined by its exact hash. An unexpected parse or
hash failure stops analysis. There are 129 repeated-key groups: 71 differ only
in wall time, and 58 differ in scientific fields. Every occurrence is retained.

The primary retrospective view selects the last occurrence of each repeated key
within its source file. It is a documented reconciliation rule, not proof that
the selected attempt is correct. Conflict-excluded and raw-inclusive
sensitivities have separate population identities. Episode bootstrap intervals
use the same population as the point estimate, preserve within-episode
dependence, and do not estimate checkpoint variability. Call-weighted and
episode-weighted estimands remain distinct.

Original attribution sidecars, observations, action tensors, diffusion noise,
model identities and several main-pass raw records were not recovered. This
limits historical attribution and causal claims. Numeric agreement can support
a consistency check but cannot supply those missing identities. Old filenames,
seed labels and source code are provenance clues, not substitutes for raw
context authentication. Preserved historical derived tables may use different
duplicate rules or unrecovered constants; they are not current authority.

## Current interpretation

Saved curves permit Q/L2 rescoring with the same historical ranking and
interventions held fixed. This establishes sensitivity to the response
transformation. It does not measure whether one attribution target produces a
better ranking. The current registry records exact membership, filters,
denominator behavior, units, source hashes, and explicit result identifiers.
Zero endpoint gaps are undefined; negative AUCs are retained. The equal-feature
quadratic example has continuous AUC 2/3 and sampled-grid AUC 0.65976, so 0.5 is
not a general random-order null. Empirical random rankings require measurement.

The released solver results remain descriptive comparisons under their
recorded keys. Unrecovered contexts and varying reference actions prevent them
from establishing denoiser causality or equivalence. The current manuscript
uses the canonical v2 results and generated figure/table lineage rather than
the former release's unrecovered primary constants.

## Reproduction

From the repository root, using the pinned CPU requirements:

```console
python -m analysis.revision.analyze --output <new-directory> --draws 10000 --seed 0
python -m analysis.revision.verify <new-directory>
```

Existing output directories are refused. The complete analysis produces 284
result rows with 57 population identities and checks ten artifact hashes.
See the analysis guide for definitions and uncertainty limitations. Current
GPU collection writes new, explicitly identified transactional run directories
outside this historical inventory. It never masquerades as recovery of these
old runs. Production numerical validation and a prospective paired
ranking-quality study on authenticated contexts remain unresolved. Successful
CPU reproduction alone does not certify either.
