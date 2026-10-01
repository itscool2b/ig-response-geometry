# Saved-data revision, September 30, 2026

This analysis authenticates preserved raw records and makes their statistical
populations explicit. It does not repair missing historical identities, perform
model inference, recover missing evidence, or establish submission readiness.
The active current artifact is `results/2026-09-30-v2/`. The earlier
`results/2026-09-30/` is a preserved preliminary computation; its solver analysis
used k=20 on the initial seed files. The current artifact uses the historical
paper's k=5 across both released solver evaluation seeds. Do not mix their rows.

Run from the repository root with Python 3.12 and NumPy:

```console
python -m analysis.revision.analyze --output <new-output-directory> --draws 10000 --seed 0
python -m analysis.revision.verify <new-output-directory>
python -m pytest tests/test_revision_analysis.py -q
```

Existing output directories are refused. The parser checks the complete 99-file
inventory against `input_manifest.json`, rejects changed input bytes, duplicate
JSON keys, malformed records, invalid identity types, nulls, and nonfinite
numbers. The one previously documented NUL-corrupt maxdev line is quarantined
only by its exact source and physical-line SHA256. New parse failures stop the
analysis. Original data and historical `out/` products are never written.

The manifest authenticates original Git-blob bytes to the September 30 audit of
public commit `8062d9300922420b0ee055766d246e825d9df2f9`. Windows Git expands
98 of these text files to CRLF in the current checkout. The manifest separately
declares the SHA256 of the exact CRLF expansion of each original file. Only one
of these two exact byte representations is accepted. Conversion to the original
bytes happens in memory after hash authentication. The line ledger hashes the
original physical bytes, including their LF, and provenance separately records
the actual checkout hashes. This preserves stable evidence identities across
Git checkout conventions without modifying or loosely normalizing the data.

## Identity and duplicate rules

The ledger covers 22,967 physical lines: 22,966 valid records and one quarantined
line. Record identity includes the source path, original line number, and exact
line hash. A within-file legacy key includes event, task, model, recorded seed,
episode, policy call, solver steps, and modality. Files remain separate run
namespaces. Neither this key, a sidecar path, nor matching scalar norms proves
observation, reference-action tensor, noise, or checkpoint identity. Those
identities remain explicitly unknown.

There are 129 repeated-key groups. Of these, 71 differ only in wall time and
58 differ in other recorded fields. `duplicates.json` retains every occurrence,
all differing fields, and selection decisions. The principal retrospective
view keeps the last occurrence to reconcile the historical interval convention.
This is an occurrence rule, not an assertion that a conflicting attempt is
correct. A separately named sensitivity removes the entire key of every
conflicting group. Changed result populations are reported together. Faithfulness
curve repeats differ only in wall time, so the Q/L2 rescoring is unaffected by
scientific-payload conflict exclusions. Raw-inclusive comparisons have no new
confidence interval and are labeled historical replay sensitivities.

## Estimands, intervals, and units

Each `results.csv` row records a result ID, occurrence view, units, estimand,
population filter, source-record count, episode-group count, and equal point/CI
population hashes. The compressed membership artifact resolves those hashes to
physical-line identities. `verify.py` checks generated hashes and every result's
membership and point/CI contract. Gzip outputs have fixed headers and no dates.

The default estimand is the median over retained policy calls. Entire recorded
episodes are resampled, preserving within-episode dependence, with B=10,000 and
seed 0. Episode groups use `(task, model, recorded_seed, episode)`. This estimates
uncertainty conditional on the released empirical mixture, allowing its seed and
task proportions to vary. It does not estimate training-checkpoint variability.
Long episodes receive more call weight. The 1B sensitivity instead takes the
median of episode medians, using the same source records and equally weighted
episode values. The distinction is part of the estimand, not a correction to a
miscomputed median. One episode receives no interval; two-episode intervals are
explicitly descriptive and do not claim calibrated 95% coverage. Frozen/cascade
shared task-episode grouping is a conditional dependence sensitivity, not newly
authenticated pairing.

`Q_score` means the historical auxiliary score `-d^2/(2*512)` in the recorded
action coordinate system. `action_L2` means negative stabilized distance
`-sqrt(d^2+1e-12)` in that coordinate system; `action_coordinate` is the maxdev
coordinate response. These are not physical robot distances. AUC and relative
completeness errors are dimensionless, correlations are Spearman rho, and
completeness pass rates are percentages of calls with relative error <=0.03.
Thresholds remain study-specific descriptive conventions.

All-valid and historical action-norm >=15 populations are separate. Norm <15
is also reported for verification data. Actual endpoint gaps, negative AUC
counts, and score ranges are recorded. A zero endpoint gap is explicitly
undefined. No clipping, nonfinite deletion, or action-norm proxy substitutes
for denominator evaluation. The preserved native AUCs all reproduce on their
stored grids, including negative extremes.

## Scientific interpretation

Within-ranking Q/L2 rescoring changes only the recorded curve ordinates, using
`Q=-d^2/(2D)` and `L2=-sqrt(d^2+epsilon)` with D=512 and epsilon=1e-12. It keeps
the ranking and all stored interventions fixed. Q-ranking vision deletion AUC
changes from 0.447903 to 0.290759 under this response transformation. This
demonstrates score-geometry sensitivity, not improved ranking. The later Q arm
has zero exact reference-norm matches among 750 nominal keys against the earlier
regeneration population; L2 and maxdev each match all 750 norms. Exact raw
contexts cannot be inferred from either agreement or disagreement.

The analytic random-order control uses a finite policy with `n` equal additive
features. Whole-feature prefixes realize only fractions `q=k/n`; every ordering
gives the same normalized quadratic insertion/deletion values,
`1-(1-q)^2` and `1-q^2`. Trapezoidal integration over all `n+1` prefixes gives
`2/3 - 1/(6n^2)`. The polynomial extensions have continuous integral 2/3,
which is also the limit as `n` increases, not the exact finite-grid area.
Choosing `n=100` realizes every fraction on the released nine-point grid exactly;
that grid gives trapezoidal AUC 0.65976 for every ordering. This finite example
refutes a universal 0.5 reference. It is not an empirical RDT random-ranking
baseline. The frozen artifact's `quadratic_continuous_integral` field records
the polynomial extension, while its grid fields record the finite-grid areas.

The saved solver endpoint analysis reports T20/T2 relative-displacement median
ratios at k=5, with paired nominal-episode resampling. Reference actions vary
across solver configurations. Pairing is conditional on historical keys, and
raw context/noise identity remains unknown. Neither these descriptive ratios
nor intervals identify denoiser causality or establish equivalence.

Remaining material studies include same-context, same-noise target/ranking
comparisons under a common response, empirical random-ranking controls,
high-precision real-model attribution checks, and adequate independent sampling
for any retained mechanism or absence claim. Their design must follow the active
claim/evidence assessment; this saved-data analysis does not prescribe a blanket
rerun or claim that additional experiments are unnecessary.
