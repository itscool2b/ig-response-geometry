# Paired response rescoring

This separate CPU analysis asks how much normalized response AUC changes when the same historical saved ranking curve is expressed as Q or stabilized L2. The signed effect is `AUC_Q - AUC_L2`, paired within the same physical record. Q-ranking and L2-ranking cohorts remain separate. It does not compare their efficacy or authenticate a shared cross-arm context.

The protocol in `protocol.json` was recorded before this analysis computed outcome values or bootstrap intervals. All eight ranking, modality and insertion/deletion cases are retained. The primary statistic averages call differences within each recorded episode, then weights those episode averages equally. This targets an average response-only change for an equally weighted recorded episode. It differs intentionally from the canonical marginal call medians. The median of paired call differences is always reported as a descriptive sensitivity, including when it disagrees with the mean. It is not the difference of two marginal medians.

Both intervals resample complete recorded episodes with 10,000 draws, PCG64 seed 0, and 2.5/97.5 percentile endpoints. They are conditional empirical descriptions with no multiplicity adjustment across the eight cases and two statistics. They are not calibrated confirmatory intervals, and historical episode labels do not establish independently randomized episodes. There is no action-norm selection.

The input manifest, raw line hashes, duplicate resolution and native stored AUC arithmetic are authenticated. A zero endpoint gap in either response leaves the paired effect explicitly undefined. Nonfinite or malformed evidence fails instead of being repaired. The original inputs, authenticated revision code and canonical v2 results are preserved.

Geometry uses each stored actual and baseline endpoint. It never assumes that the actual endpoint is an exact zero-residual self-reference. A saved point is in range if its implied squared discrepancy lies between the two endpoint discrepancies, a high-distance overshoot if above both, or a low-distance undershoot if below both. Baseline-nearer curves reverse the familiar in-range sign. The exact trapezoid difference is decomposed by saved-node category; sampled segment endpoint categories do not assert the behavior of unseen intermediate interventions. A fixed endpoint-scale tolerance is reported only as a label sensitivity and never changes values or membership.

The resulting output is versioned separately from `analysis/revision/results/2026-09-30-v2`. Results and an exact reproduction command are recorded in the generated `methods_results.md` and provenance. No current or historical model is executed.

## Results and their limits

The [versioned results](results/2026-10-01-v1/methods_results.md) show an important disagreement between summaries. All eight paired-call medians are positive, ranging from 0.1041 to 0.1590, but six of the eight mean paired differences are negative. In particular, vision insertion has means of -17.0966 for the Q-ranking cohort and -18.5620 for the separate L2-ranking cohort, even though both paired medians are about +0.1041. The effects compare response readouts within a saved call, never the two ranking cohorts with each other.

This sign disagreement is explained arithmetically by high-distance overshooting ordinates. For Q-ranking vision insertion, the mean in-range node contribution is +0.1144 and the mean overshoot contribution is -17.2111. For the separate L2-ranking vision insertion cohort the corresponding terms are +0.1138 and -18.6758. These terms sum to the respective mean differences. The most negative individual paired differences are -4438.84 and -5242.87. Such values are possible because endpoint-normalized response AUC is not bounded when interventions move beyond the endpoint discrepancy range. No trimming, clipping, revised eligibility threshold or estimator substitution is used.

High-distance overshoots occur at 4.41% to 26.24% of saved ordinates across cases, including endpoint ordinates in that denominator. At least one such ordinate occurs in 120 to 555 of the 750 calls, depending on the case. Thus the positive paired medians describe a typical saved call while the means remain highly sensitive to the magnitude of negative tail effects. Four of the eight episode-mean intervals include zero. The other intervals remain unadjusted descriptive intervals and do not turn this historical analysis into a confirmatory superiority result.

Each ranking cohort contains 750 defined calls in 30 recorded episodes, each with 25 calls. Equal-episode and equal-call means therefore coincide in these particular cohorts. Adding the source-file namespace does not change the 30 groups. There are no undefined paired effects, reversed endpoint orientations, low-distance undershoots or tolerance-induced label changes in these data. The method and tests nevertheless retain those cases explicitly. The Q actual endpoints imply zero squared residual. Native L2 actual values are the saved FP32 representation of -1e-6 and imply a tiny negative squared residual before the canonical, declared domain-roundoff clamp; this is recorded rather than claimed to be an exact real-number self-reference. The maximum generalized-identity discrepancy on stored floats is approximately 7.28e-12.

All 1,524 physical faithfulness occurrences are authenticated; the canonical last-occurrence rule retains 1,500 calls across the two cohorts. Conflict-excluded membership is identical. All 16 canonical marginal rescoring medians and their physical populations are reproduced. Neither checkpoint/context authentication nor historical numerical qualification is supplied by this successful arithmetic check.

## Reproduce

From the repository root:

```text
python -m analysis.paired_rescoring.analyze --output <new-empty-directory>
python -m pytest tests/test_paired_rescoring.py -q
```

The callable API is `analysis.paired_rescoring.analyze.run(output, root=repository_root)`. It writes eight scientific artifacts plus `provenance.json`, refuses an existing output directory, and returns the provenance object. `artifacts_sha256` authenticates every scientific artifact. The compact `paired_rescoring.tex` fragment reports the prespecified mean and interval; `summary.csv` additionally retains the paired median, its interval, raw/defined denominators, extrema, negative-effect counts and exact geometry contributions. Tests reproduce every scientific artifact byte for byte against the versioned manifest.
