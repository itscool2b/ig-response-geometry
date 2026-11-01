# November 1 audit reporting

This is a retrospective descriptive analysis of existing, authenticated records.
It is not a new experiment or an amendment to the preregistered analysis.
The frozen campaign outputs and all prior deliveries remain unchanged.

The vision-deletion stability analysis uses the registered common complete-case
episodes and seven candidate attribution methods in the three new experiment
groups. The 10,000 whole-episode bootstrap draws use PCG64 seed 930011. Both
responses and every method receive the same draw. Raw mean and scaled mean
statistics average calls within episodes, then episodes equally. Scaling occurs
per context using the archived random-order standard deviation. The median
criterion is the median of the per-episode mean raw advantages, not a pooled-call
median. Exact ties remain undetermined. The three outcome shares sum to one:
same unique winner, different unique winners, or at least one tied winner.
These shares describe stability under resampling the observed episodes, not the
probability of an underlying population winner. The mean and scaled-mean point
results are checked against the registered outputs before a report is written.

The successful-policy split uses Q-target trapezoidal IG at 512 intervals. The
31 successful and 9 failed episodes are resampled separately, keeping both calls
of each episode together. Intervals are descriptive 95% percentile intervals
for the pooled-call median, with 10,000 draws and seed 930011. They do not support
a causal effect of task success or inference across checkpoints or tasks.

Reproduce into a new directory:

```bash
python -m analysis.manuscript_revision.nhsjs_audit_2026_11_01.report --output runs/nhsjs-audit-report
```

The report records SHA-256 hashes of every input and its analysis code. A fresh
run must match `results/2026-11-01-v1/report.json` byte for byte in the documented
CPU environment. No GPU, model download, or original numerical run volume is
required. Figure and table rendering is handled by
`scripts/build_nhsjs_revision_assets.py`.
