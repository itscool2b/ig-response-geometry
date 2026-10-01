# Preserved FP32 probe diagnostics

Source snapshot: 2026-10-01 03:36 UTC sealed audit snapshot. This is the last fully audited local snapshot used here, not a final census at the later compute stop.

The full roster has 24 contexts and 144 target/modality cells. Of 21 available contexts, 12 have all six diagnostics complete, representing six episodes from four 170M tasks. Three planned late calls were never reached. Nine available contexts remain unsealed at this cutoff; their later completion status is unknown here.

| Stratum | Planned contexts | Complete | Source unavailable | Unsealed | Complete/planned cells |
|---|---:|---:|---:|---:|---:|
| peginsertionside-170m | 4 | 2 | 0 | 2 | 12/24 |
| pickcube-170m | 4 | 4 | 0 | 0 | 24/24 |
| pickcube-1b | 4 | 0 | 2 | 2 | 0/24 |
| picksingleycb-170m | 4 | 2 | 0 | 2 | 12/24 |
| stackcube-170m | 4 | 4 | 0 | 0 | 24/24 |
| stackcube-1b | 4 | 0 | 1 | 3 | 0/24 |

All 2,880 recorded repeat-related equality and coverage checks satisfy their recorded conditions, including 1,800 equality checks. The 72 completed arms contain seven fixed-alpha probes repeated three times and finest maps repeated twice. These are dependent diagnostics on the same contexts, not 2,880 independent replicates. Repeatability does not establish quadrature accuracy.

Source collection was prospectively capped at 13 policy calls and 400 environment steps, with calls 0 and 12 selected before collection and unavailable calls retained without replacement. This is not uniform sampling from a completed 400-step trajectory. The input binds the source collection protocol, all six banks, initial queue and both repaired queues. Earlier overlapping snapshots are not added to the sample count.

The full ladder contains 302 threshold violations, 11,146 satisfied checks and 144 inapplicable checks. These counts combine dependent comparisons and conditional candidate-completeness checks; they are not numbers of failed experiments. Inapplicable checks are the recorded zero-realized-count top-5-percent comparisons for small groups. Every original check is retained in the input.

The following extrema and counts use only the two finest recorded budgets: 256 versus 512 for vision/language, and 512 versus 1024 for state. The IG coordinate column uses IG only. RMS extrema include both IG and path-gradient rankings and both deletion/insertion; extrema in different columns may arise in different contexts. The finer map is a comparator, not ground truth. No budget or subset is approved here.

| Modality | Target | Complete arms | Any finer-pair violation | Finest completeness >1% | Max IG relative L1 | Max finest residual | Max RMS curve change / baseline RMS |
|---|---|---:|---:|---:|---:|---:|---:|
| language | L2 | 12 | 0 | 0 | 0.00401978 | 0.00315173 | 0 |
| language | Q | 12 | 0 | 0 | 5.6813e-05 | 1.76156e-05 | 0 |
| state | L2 | 12 | 2 | 2 | 0.207783 | 0.0665499 | 0.185733 |
| state | Q | 12 | 2 | 2 | 0.321192 | 0.114357 | 0.0164115 |
| vision | L2 | 12 | 4 | 0 | 0.0123367 | 0.00288473 | 0.0736083 |
| vision | Q | 12 | 1 | 0 | 0.0204948 | 0.00123669 | 0 |

All four finest-budget completeness violations occur in the two YCB episode-1 state contexts, for both targets. The table below preserves every budget for those four arms; the apparent improvement with increasing budget does not make the finest result accurate.

| Call | Target | m | Scalar gap | IG sum | Absolute residual | Relative residual |
|---:|---|---:|---:|---:|---:|---:|
| 0 | Q | 256 | 0.47357374 | 1.0501218 | 0.57654804 | 1.21744 |
| 0 | Q | 512 | 0.47357374 | 0.64124721 | 0.16767347 | 0.35406 |
| 0 | Q | 1024 | 0.47357374 | 0.42953128 | 0.044042468 | 0.0930002 |
| 0 | L2 | 256 | 22.021339 | 40.660217 | 18.638878 | 0.846401 |
| 0 | L2 | 512 | 22.021339 | 28.03299 | 6.0116501 | 0.272992 |
| 0 | L2 | 1024 | 22.021339 | 21.464672 | 0.55666733 | 0.0252785 |
| 12 | Q | 256 | 0.68219489 | 1.5319726 | 0.84977776 | 1.24565 |
| 12 | Q | 512 | 0.68219489 | 1.0090075 | 0.32681257 | 0.47906 |
| 12 | Q | 1024 | 0.68219489 | 0.7602089 | 0.078014016 | 0.114357 |
| 12 | L2 | 256 | 26.430428 | 42.06266 | 15.632233 | 0.591448 |
| 12 | L2 | 512 | 26.430428 | 34.232857 | 7.8024292 | 0.295206 |
| 12 | L2 | 1024 | 26.430428 | 28.189369 | 1.7589417 | 0.0665499 |

These state discrepancies are not merely division by an almost-zero scalar gap: the Q gaps are about 0.47 and 0.68, and the L2 gaps about 22.0 and 26.4. The corresponding coordinate L1 denominators and absolute map changes are retained in per_cell.csv. YCB call 12 Q has a 32.1% IG coordinate difference between 512 and 1024 despite identical transferred deletion/insertion response curves at those budgets. Stable ranks or response curves therefore do not repair its unresolved coordinate integral.

Saved probes show large interior changes in action distance and gradients, including nonmonotonic distance along the call-0 state path. This is consistent with difficult path integration. Only seven fixed-alpha gradients and aggregate maps are available for diagnosis, so neither a narrow peak nor an exclusive numerical cause is established. Both Q and L2 fail, which rules out attributing all failures to L2 smoothing at the self-reference endpoint. The common-distance gradient identity is checked descriptively from saved tensors; it is not a new model evaluation.

The estimand is an offline FP32 downstream probe on contexts visited by a BF16 behavior policy. BF16-valued weights and cached adapted image/language values are lifted to FP32, raw state is readapted in FP32, recorded noise values are held fixed, and a new FP32 self-reference is used. This is neither a native FP32-weight/encoder rollout nor the exact derivative of the original BF16 policy. The strict recorded backend bundle gives repeatability on tested points; its individual switches were not causally isolated.

The narrow paper can report the probe contract, observed repeatability, heterogeneous budget sensitivity and explicit finite-grid failures with the full roster. These diagnostics cannot support attribution superiority, learned-weight dependence, scale generalization, policy efficacy, native-FP32-policy equivalence, or cohort-wide numerical certification. Earlier v3/v4 precision controls use different comparison contracts and are provenance context, not pooled replication of this probe.

Attempt accounting preserves two sealed historical infrastructure/contract failures and their retries. Thirteen sealed successful jobs include one source-unavailable call, leaving twelve numerical contexts. Unsealed historical or current jobs are not labeled numerical failures. Budget-stopped work after this snapshot remains outside its assessed evidence.
