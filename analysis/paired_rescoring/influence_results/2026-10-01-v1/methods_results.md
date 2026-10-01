# Exploratory episode influence, 2026-10-01-v1

This supplement follows an earlier exploratory audit; the saved protocol is not an outcome-blind preregistration. It preserves all eight paired cases and all original untrimmed means, medians and intervals. Each omission removes one entire recorded episode and then gives every remaining episode equal weight. All 240 omissions retain physical-record membership and hashes. Omission ranges are deterministic influence diagnostics, not confidence intervals or calibration of general-population coverage.

`case_summary.csv` reports the full mean, all-omission extrema, exact sign agreement, and concentration of negative episode means. The negative episode-mean denominator is the sum of magnitudes of negative within-episode means, not the signed net mean or raw call total. `episode_omissions.csv` identifies every omitted episode and its records and hashes the retained population. `tail_diagnostics.json` describes the minimum paired call difference in every case, with ties broken by physical record ID, and top negative-call concentration using the sum of magnitudes of negative call differences. A zero negative-mass denominator yields null, not zero. Weak empirical percentiles are 100 times the fraction of within-case values less than or equal to the example.

The finite squared residuals and normalization gaps retain the historical action-coordinate convention and imply no physical importance threshold. The historical 1e-9 gap cutoff is only a descriptive comparison; no new eligibility rule is applied. All examples remain within their own ranking cohort. Matching nominal labels cannot authenticate a cross-arm context, and recorded episodes do not establish independent randomized or training replications. Saved nodes do not identify unobserved continuous curves or realized masks. These diagnostics establish no ranking efficacy or task-success result.

From the repository root, using the same NumPy dependency as the paired analysis:

```text
python -m analysis.paired_rescoring.influence --output <new-empty-directory>
python -m analysis.paired_rescoring.influence --verify <result-directory>
python -m pytest tests/test_paired_influence.py -q
```

The producer authenticates the frozen paired and canonical source/output records and raw inputs before using saved calls, refuses an existing destination, and records input, source, protocol and output hashes. Reproduction requires no model, simulator, GPU or private file. The new package is nested so the frozen paired producer's top-level source-file enumeration is unchanged.
