# Round-three manuscript reporting, October 3, 2026

This offline layer generates the current manuscript's tables, figures and prose
numbers from immutable historical results. It does not run a policy, recover
missing contexts, pair the two ranking cohorts, or change the original analyses.
The current displays use Q for the quadratic function and N for the norm
function; immutable source records retain the historical labels Q and L2.

The original paired protocol named the equal-episode mean primary. The current
retrospective presentation leads with the paired-call median and positive-change
fraction. It preserves the original mean, descriptive intervals, all eight cases,
extreme negative values, undefined-case accounting and episode-omission results.

From the repository root, in the documented CPU environment:

```text
python scripts/build_revision_assets.py --reproduce
python -m pytest tests/test_manuscript_reporting.py -q
```

`--reproduce` repeats the canonical, paired, influence and portable numerical
analyses into fresh temporary directories and checks every scientific output
against the original hashes. It also verifies the aliasing construction. The
receipt is saved in `results/2026-10-03-v2/reproduction.json`. Subsequent runs
without this flag authenticate the same inputs and refresh the manuscript assets
and lineage, while keeping the successful receipt tied to the evidence hashes.

The versioned report includes the source and population identities, estimator
history, exact unrounded paired values, task success linked to individual
episode-end records, and numerical coordinate checks with IG and path-gradient
rankings kept separate. `paired_summary.csv`, `task_success.csv`, and
`numerical_by_context.csv` are convenient projections of `report.json`.

All eight cases retain 750 calls. The current data have no undefined paired
effects or undershoots. Across all cases, overshoot prevalence is 16.0% to 74.0%.
The numerical subset covers 12 contexts from six episodes. A twelve-context
matched Q/L ranking comparison is not possible from the locally retained raw
maps and common-response curves. The two retained YCB state cases were selected
for numerical diagnosis and are not substituted as a representative pilot.

The complete figure shows every call, including the full negative tail, on a
symmetric logarithmic axis. A separately labeled central-range panel has its
outside-range count printed for each row. No statistic uses this visual zoom as
a filter. Figure point jitter is deterministic visual spacing only.

Old generated displays remain in the repository as historical artifacts; the
current manuscript and current lineage identify only the seven retained displays.

The current report preserves `results/2026-10-02-v1/` and `results/2026-10-03-v1/`.
It retains these descriptive projections:

- `call_index_summary.csv` separates first calls from later calls without excluding
  either group. Contributions use the full case's equal-episode weights; group
  means and medians use defined calls within each group. The observed equal-length
  episodes make each contribution a sum divided by 750, while a later-call mean
  divides its sum by 720. Empty groups and zero tail denominators are undefined,
  represented by null in JSON and empty cells in CSV.
- `baseline_gap_summary.csv` reports baseline squared residual and response-gap
  ranges, zero counts, and the counts below 0.1 and 0.5. It applies no gap filter.
- `grid_diagnostics.csv` records nominal-grid weights, equal 1%/5% residual
  ordinates, and the universal 0.2175 AUC-difference upper bound. Equal ordinates
  do not authenticate missing masks, and changing integration fractions can
  change the paired difference. The upper bound is not an attainable-policy claim.

These descriptive breakdowns were chosen after inspecting the saved data. They
do not revise the original estimator or supply independent replications. The
numerical report distinguishes 1,800 equality checks from 2,880 combined
repeat-condition and coverage checks. Its roster counts describe the audited
collection, not a recovered final compute census. Positive-epsilon norm targets do not have an intrinsic gradient jump;
observed finite-budget behavior is not an asymptotic convergence guarantee.

## Round-three additions

Reporting version `2026-10-03-v2` adds fully-in-range call counts, positive shares
within each geometric category, in-range difference quartiles and baseline
residual quantiles. Nonnegative differences are guaranteed for fully-in-range
curves; strict positivity additionally requires a positively weighted node with
residual strictly between zero and the all-baseline residual. The saved calls'
positive counts are measured separately.

The `cohort_relationship` object joins recorded task, model, seed, episode and
call labels, rejects duplicate or missing keys, and reports baseline-residual
correlations and shared extreme episodes. Extreme means vision-insertion
`AUC_Q - AUC_N < -100`. Shared labels do not authenticate missing observations,
noise or checkpoints and do not establish a matched ranking comparison.

`language_schedule_sensitivity.csv` removes only the nominal 1% node and
reintegrates both responses on the same reduced schedule. It reports both
medians, their difference and the in-range per-call bound. Equal saved ordinates
do not establish identical intervention masks or realized feature fractions.

`numerical_evaluation_checks.csv` projects saved rank correlation, normalized
AUC difference and RMS-curve difference checks, preserving context, ranking
type, target, response, direction, both budgets and source-report hash.
`numerical_violation_summary.csv` counts arms with any finest-pair violation.
The 302 full-ladder violations count dependent criteria, including both IG and
path-gradient checks, rather than independent failing contexts.

No additional cohort is rescored and no policy is executed. Undefined metrics
remain null, all finite effects remain included, and every reporting artifact is
bound to its producer and authenticated input hashes.
