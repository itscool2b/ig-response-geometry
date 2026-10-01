# TinyLlama attribution and normalization

Updated September 30, 2026. `ig_tinyllama.py` is a historical TinyLlama-1.1B-Chat next-token demonstration. It attributes a selected log-softmax score to continuous input embeddings, not discrete token IDs. The recorded prompt was “The capital of France is” and selected continuation was “Paris” (recorded token ID 3681).

The script loads fp32 weights, looks up input embeddings, and uses learned PAD/EOS embeddings as its reference. PAD/EOS has semantics; it is not guaranteed information-free. Approximately 4.4 GB of model weights does not establish total IG memory usage.

## Positive-epsilon normalization

Ignoring learned gain for clarity,

```
RMSNorm(x) = x / sqrt(mean(x**2)+eps)
J_RMSNorm(0) = I / sqrt(eps)
J_LayerNorm(0) = (I-11^T/n) / sqrt(eps).
```

Both Jacobians are finite for positive epsilon. At eps=1e-5 their spectral norms are about 316.23 for n>1. LayerNorm centering does not eliminate amplification. Learned gain multiplies the relevant Jacobian rows.

A local factor is not a full-transformer derivative estimate. Residual paths, attention, other Jacobians, activation values and dtype all matter. Multiplying 316 across every block does not prove overflow or identify the cause of a particular NaN. A zero baseline may be numerically difficult without the regularized layer having a literal singularity.

The revised core fails on nonfinite values with diagnostics. It does not hide derivatives with `nan_to_num`. PAD can be tested as another path but is not universally safe or neutral.

## Historical observations

| m | Recorded relative residual, PAD reference |
|---|---:|
| 300 | 25.91% |
| 500 | 9.96% |
| 1000 | 0.55% |

This historical endpoint-average sweep improved scalar accounting at the measured budgets. It does not isolate SiLU or dimensionality as the cause, guarantee behavior on other prompts, or prove map convergence. The claim that m=1000 was needed applies only to the tested candidates and historical criterion.

Per-token bars sum signed embedding-coordinate attribution. Sign describes the selected scalar difference relative to the chosen reference. Historical output: `output/ig_tinyllama.png`. See [the current numerical contract](integrated_gradients.md) for revised precision, quadrature and failure semantics.
