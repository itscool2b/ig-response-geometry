# Integrated gradients: current numerical contract

Updated September 30, 2026. This page describes the revised public implementation. Updating the code does not regenerate or validate historical outputs.

## Definition and scope

For scalar F, input x, reference b and straight path z(a)=b+a(x-b),

```
IG_i(x;b) = (x_i-b_i) integral_0^1 partial_i F(z(a)) da.
```

A baseline is a chosen reference, not necessarily an absence of information. Under the appropriate differentiability/integrability assumptions, summing the exact integrals gives F(x)-F(b) by the chain rule and the fundamental theorem along the complete path. This is not a separate endpoint-difference theorem for each coordinate.

Implementation invariance concerns equivalent functions in the same coordinates, with the same baseline and path. It does not identify different representations, baselines or targets. An affine target transformation cF+d scales IG by c; a nonlinear transformation changes the path gradient weights and may change rankings. Sensitivity(a) addresses the specific case where input and baseline differ in one feature and their outputs differ. See [Sundararajan, Taly and Yan (2017)](https://proceedings.mlr.press/v70/sundararajan17a.html).

## API and precision

```python
result = integrated_gradients(
    forward_fn, input_tensor, baseline_tensor, m=64,
    quadrature="trapezoid", arithmetic_dtype=torch.float32,
    return_result=True,
)
attr = result.attributions
metadata = result.diagnostics()
```

Inputs must be finite floating tensors with identical shapes and devices. The scalar callable must retain its autograd connection. The default Tensor return remains available for existing callers.

Detached fp32 master tensors supply displacement and interpolation. Gradient accumulation, division, attribution products and accounting reductions also use fp32. Exact endpoint tensors are used at alpha 0 and 1. Each master leaf is differentiably cast immediately before the forward to `forward_dtype`, which defaults to the original input dtype. This keeps bf16 models compatible while separating model precision from IG arithmetic. It does not remove rounding inside the forward/backward computation.

For small mathematical diagnostics, explicitly select `arithmetic_dtype=torch.float64` and a compatible `forward_dtype`. Precision comparisons must hold the model, inputs, baseline, target and actual latent-noise values fixed.

## Quadrature

`m` is a positive integer interval count. Both rules evaluate m+1 points:

```
trapezoid:
 (x_i-b_i)/m * [g_i(0)/2 + sum(k=1..m-1) g_i(k/m) + g_i(1)/2]
legacy_endpoint_average:
 (x_i-b_i)/(m+1) * sum(k=0..m) g_i(k/m)
```

Composite trapezoidal quadrature is the revised default. The named legacy option preserves the old quadrature, not its bf16 arithmetic or nonfinite replacement. Under sufficient regularity these rules generally have first-order and second-order error, respectively for legacy and trapezoid. Such asymptotic statements do not guarantee monotone improvement at successive budgets.

Softmax, SiLU and GELU are smooth at finite inputs. Softmax has no argmax discontinuity. Smooth functions can still be poorly resolved by a finite grid; Gauss-Legendre is not generally invalid for nonsmooth functions. The old ResNet/ViT/LLM examples cannot establish a universally superior rule or isolate activation choice.

## Nonfinite values and completeness

Any nonfinite input, path value, score, derivative, running sum or returned arithmetic result fails the computation. `NonFiniteAttributionError.diagnostics` records the stage, path alpha/index where applicable, NaN/infinity counts, bounded coordinate samples and caller identity. There is no silent derivative replacement.

The structured result records evaluated endpoint scores, signed gap, attribution sum, absolute residual and relative residual. A zero gap has relative residual `None`, serialized as JSON null. A small nonzero gap can still be ill conditioned; no universal denominator threshold is established by the code.

Completeness checks one signed sum. Coordinate errors can cancel. The preserved linear bf16 audit fixture had 6.25% maximum coordinate error at m=64 while reporting 0.26% scalar error. A corrupted-backward example lost canceling coordinates while reporting exact completeness. Revised tests check known coordinates directly. The historical 3% residual criterion is an accounting convention, not a map-quality certificate.

## Autograd and memory

`autograd.grad` returns requested derivatives without accumulating them into those tensors' `.grad` buffers. `.backward()` uses the same chain rule but normally accumulates leaf gradients. Existing parameter-gradient buffers do not feed into a fresh input derivative. The old API switch coincided with other changes and did not establish parameter-gradient interference as the cause of the reported error. See [PyTorch's API contract](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html).

The loop releases each graph after its gradient. Increasing m adds sequential work rather than storing m graphs simultaneously. Unused CUDA allocator blocks are reusable by PyTorch; `empty_cache()` releases cached blocks to other applications and does not free live tensors. It is no longer called after every interpolation. See [PyTorch memory management](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-memory-management).

## Evidence boundaries

`tests/test_integrated_gradients.py` checks linear/nonlinear analytical coordinates, bf16 inputs, float64 diagnostics, endpoints and failures. `tests/test_rdt_sampling.py` checks explicit-noise sampler contracts. CPU tests are not real-model accuracy evidence. `scripts/validate_rdt_numerics.py` compares selected authenticated contexts with numerical factors separated.

The following old single-example values are retained for traceability. They were not produced by the revised default:

| Model / modality | m | Recorded relative residual |
|---|---:|---:|
| ViT-B/16 | 300 | 0.14% |
| TinyLlama 1.1B | 1000 | 0.55% |
| RDT-1B vision | 300 | 9.06% |
| RDT-1B language | 300 | 5.11% |
| RDT-1B state | 300 | 6.12% |
| ResNet50 | 300 | 13.13% |
| LLaVA 1.5-7B, 4-bit | 300 | 28.26% |

These numbers do not rank attribution quality or establish the mechanism causing their differences.
