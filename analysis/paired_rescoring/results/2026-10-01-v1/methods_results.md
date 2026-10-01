# Paired response rescoring: 2026-10-01-v1

This is a retrospective response-only analysis of authenticated saved curves. The ranking, saved intervention ordinates and endpoints are fixed within each call. The signed difference is normalized AUC_Q minus normalized AUC_stabilized_L2. Q-ranking and L2-ranking cohorts are never paired with each other.

The primary statistic gives equal weight to each recorded episode after averaging its paired call differences. This new estimand is distinct from the canonical marginal call medians; neither weighting is universally required by episode resampling. The median of paired call differences is retained as a prespecified descriptive sensitivity. Both intervals resample whole recorded episodes, B=10000, PCG64 seed 0, percentile 95%. All 16 intervals across eight cases are unadjusted, conditional empirical descriptions, without calibrated coverage or a global significance claim.

| Ranking | Modality | Curve | Calls / episodes | Mean paired difference [95% interval] | Paired-call median [95% interval] |
|---|---|---|---|---|---|
| Q | vision | insertion | 750 / 30 | -17.096621 [-33.821047, -3.974597] | 0.104156 [0.096065, 0.112259] |
| Q | vision | deletion | 750 / 30 | 0.122724 [0.105902, 0.138168] | 0.158758 [0.148116, 0.168694] |
| Q | lang | insertion | 750 / 30 | -0.181926 [-0.405141, -0.027740] | 0.116940 [0.066691, 0.137381] |
| Q | lang | deletion | 750 / 30 | -0.009473 [-0.102242, 0.063100] | 0.118929 [0.091170, 0.137958] |
| L2 | vision | insertion | 750 / 30 | -18.561981 [-37.867211, -3.173110] | 0.104140 [0.094683, 0.112615] |
| L2 | vision | deletion | 750 / 30 | 0.018423 [-0.203690, 0.137787] | 0.158958 [0.146034, 0.169022] |
| L2 | lang | insertion | 750 / 30 | -0.032907 [-0.096332, 0.026553] | 0.127875 [0.110635, 0.144525] |
| L2 | lang | deletion | 750 / 30 | -1.370999 [-4.208132, 0.077980] | 0.120459 [0.099098, 0.131048] |

## Saved-point geometry

Classification compares squared discrepancy with both stored endpoint discrepancies. In range means inside that closed interval; high-distance overshoot is above both endpoints; low-distance undershoot is below both. If the baseline is nearer than the actual endpoint, the usual in-range sign reverses. We never assume exact self-reference. Reported node percentages use all saved grid ordinates, including endpoints, with interior-only counts also retained in geometry.csv. Calls are counted as having an overshoot if at least one recorded ordinate is above both endpoints. These are pooled descriptive counts, not independent observations or additional inference units.

| Ranking | Modality | Curve | In-range nodes | High overshoot nodes | Low undershoot nodes | Calls with high overshoot |
|---|---|---|---|---|---|---|
| Q | vision | insertion | 90.12% | 9.88% | 0.00% | 312/750 |
| Q | vision | deletion | 95.11% | 4.89% | 0.00% | 127/750 |
| Q | lang | insertion | 73.76% | 26.24% | 0.00% | 555/750 |
| Q | lang | deletion | 92.34% | 7.66% | 0.00% | 281/750 |
| L2 | vision | insertion | 90.47% | 9.53% | 0.00% | 292/750 |
| L2 | vision | deletion | 95.59% | 4.41% | 0.00% | 120/750 |
| L2 | lang | insertion | 78.81% | 21.19% | 0.00% | 491/750 |
| L2 | lang | deletion | 91.60% | 8.40% | 0.00% | 312/750 |

The exact trapezoid difference is decomposed by saved-node category in summary.csv and calls.csv.gz. Segment labels refer only to their two sampled endpoints. No continuous crossing or unseen intervention is inferred. The prespecified 1e-6 endpoint-scale tolerance changes labels only; it cannot change an AUC, endpoint gap, defined population or inference result.

## Accounting and limits

The input audit retains 1524 physical faithfulness occurrences and selects 1500 by the canonical last-occurrence rule. The conflict-excluded population comparison is recorded for each ranking arm. Every input source and physical line is hash-bound to the canonical v2 manifest and ledger, and every native stored AUC is checked before transformation.

Undefined zero-gap calls remain in raw accounting and are excluded only from the explicit paired-defined estimand. Negative gaps and negative effects are retained. There is no action-norm filter. Native L2 conversion roundoff, actual endpoint residuals, orientation and finite extremes remain visible in the outputs. Source-file namespaces are checked against legacy episode group counts to detect unintended regrouping.

The historical grid is nominal percentage, not authenticated realized mask fraction. Checkpoint, observation, noise and reference action tensors are unavailable for authentication. This analysis changes the response readout only, and supplies no ranking efficacy, task success, causal effect or empirical random-control claim. The saved-grid statistic also differs from a continuous path integral.

## Reproduction

From the repository root, with NumPy installed:

```text
python -m analysis.paired_rescoring.analyze --output <new-empty-directory>
```

The command refuses an existing directory. Compare the generated scientific files against artifacts_sha256 in provenance.json. The environment fields and exact new source/protocol hashes document execution; they are not silently normalized. Canonical revision code and outputs remain untouched.
