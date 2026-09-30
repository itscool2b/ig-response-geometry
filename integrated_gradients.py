"""Integrated gradients with explicit quadrature and arithmetic precision.

The default keeps the path, displacement, gradient sum, product and reductions
in float32. Model inputs may still be lower precision. This prevents the old
bf16 accumulation error; it does not establish that a low precision forward or
a particular integration budget gives accurate coordinates.
"""

from dataclasses import dataclass
import math
import numbers
import time

import torch


QUADRATURE_RULES = ("trapezoid", "legacy_endpoint_average")


class NonFiniteAttributionError(FloatingPointError):
    """A failed IG computation, with JSON-compatible diagnostic information."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        location = "" if diagnostics["alpha"] is None else f" at alpha={diagnostics['alpha']}"
        super().__init__(
            f"Nonfinite values in {diagnostics['stage']}{location}: "
            f"{diagnostics['nonfinite_count']} of {diagnostics['numel']}"
        )


def _require_finite(tensor, stage, *, step=None, alpha=None, context=None):
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    # Only failures copy coordinates to the host. Bound the diagnostic size even
    # when a large model produces an entirely invalid gradient tensor.
    indices = (~finite).nonzero(as_tuple=False)[:8].detach().cpu().tolist()
    raise NonFiniteAttributionError({
        "stage": stage,
        "step": step,
        "alpha": alpha,
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "nonfinite_count": int((~finite).sum().item()),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "posinf_count": int(torch.isposinf(tensor).sum().item()),
        "neginf_count": int(torch.isneginf(tensor).sum().item()),
        "sample_indices": indices,
        "context": dict(context or {}),
    })


@dataclass(frozen=True)
class IntegratedGradientsResult:
    attributions: torch.Tensor
    input_score: float
    baseline_score: float
    expected_gap: float
    attribution_sum: float
    absolute_residual: float
    relative_residual: float | None
    quadrature: str
    m: int
    input_dtype: str
    forward_dtype: str
    arithmetic_dtype: str
    elapsed_seconds: float

    def diagnostics(self):
        """Return scalar metadata without copying the attribution tensor."""
        return {
            "input_score": self.input_score,
            "baseline_score": self.baseline_score,
            "expected_gap": self.expected_gap,
            "ig_sum": self.attribution_sum,
            "absolute_residual": self.absolute_residual,
            "relative_residual": self.relative_residual,
            "gap_status": "zero_gap" if self.expected_gap == 0 else "nonzero_gap",
            "quadrature": self.quadrature,
            "m": self.m,
            "forward_evaluations": self.m + 1,
            "input_dtype": self.input_dtype,
            "forward_dtype": self.forward_dtype,
            "arithmetic_dtype": self.arithmetic_dtype,
            "nonfinite_count": 0,
            "elapsed_seconds": self.elapsed_seconds,
        }


def integrated_gradients(
    forward_fn,
    input_tensor,
    baseline_tensor,
    m=300,
    *,
    quadrature="trapezoid",
    arithmetic_dtype=torch.float32,
    forward_dtype=None,
    return_result=False,
    diagnostic_context=None,
):
    """Integrate a scalar target along a straight input-baseline path.

    ``m`` is the positive number of intervals, giving ``m + 1`` evaluations.
    ``trapezoid`` assigns half weight to the endpoints, divided by ``m``.
    ``legacy_endpoint_average`` assigns equal weights, divided by ``m + 1``;
    it reproduces the historical *quadrature*, not the old bf16 arithmetic or
    silent nonfinite replacement.

    ``forward_dtype`` defaults to the input dtype for existing model closures.
    Each master leaf is cast immediately before the call without detaching,
    so the gradient is pulled back to the master path. Float64 diagnostic work
    must explicitly set ``arithmetic_dtype=torch.float64`` and, if its inputs
    are not already float64, ``forward_dtype=torch.float64``.

    By default return a Tensor, preserving the original API. ``return_result``
    returns endpoint scores and numerical diagnostics as well. The endpoints
    are evaluated once each and reused for accounting. A zero signed gap has
    an undefined relative residual (None), with the absolute residual retained.
    A nonzero gap is not automatically well conditioned, and completeness is
    not a coordinate accuracy or feature-ranking guarantee.

    All inputs, scores, gradients and arithmetic results must be finite. Any
    failure raises NonFiniteAttributionError; no failed map is returned.
    """
    started = time.perf_counter()
    if isinstance(m, bool) or not isinstance(m, numbers.Integral) or m < 1:
        raise ValueError("m must be a positive integer number of intervals")
    m = int(m)
    if quadrature not in QUADRATURE_RULES:
        raise ValueError(f"quadrature must be one of {QUADRATURE_RULES}")
    if arithmetic_dtype not in (torch.float32, torch.float64):
        raise ValueError("arithmetic_dtype must be torch.float32 or torch.float64")
    for name, tensor in (("input", input_tensor), ("baseline", baseline_tensor)):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating point Tensor")
        if tensor.numel() == 0:
            raise ValueError(f"{name} must not be empty")
        _require_finite(tensor, name, context=diagnostic_context)
    if input_tensor.shape != baseline_tensor.shape:
        raise ValueError("input and baseline must have identical shapes")
    if input_tensor.device != baseline_tensor.device:
        raise ValueError("input and baseline must be on the same device")
    model_dtype = input_tensor.dtype if forward_dtype is None else forward_dtype
    if model_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError("forward_dtype must be a supported floating point dtype")

    actual = input_tensor.detach().to(dtype=arithmetic_dtype)
    baseline = baseline_tensor.detach().to(dtype=arithmetic_dtype)
    _require_finite(actual, "master_input", context=diagnostic_context)
    _require_finite(baseline, "master_baseline", context=diagnostic_context)
    displacement = actual - baseline
    _require_finite(displacement, "displacement", context=diagnostic_context)
    sum_gradients = torch.zeros_like(actual)
    endpoint_scores = {}

    for k in range(m + 1):
        alpha = k / m
        check = {"step": k, "alpha": alpha, "context": diagnostic_context}
        # Avoid endpoint reconstruction error, including subtract/add cancellation.
        point = baseline if k == 0 else actual if k == m else baseline + alpha * displacement
        _require_finite(point, "interpolation", **check)
        with torch.enable_grad():
            interpolated = point.detach().requires_grad_(True)
            model_input = interpolated.to(dtype=model_dtype)
            _require_finite(model_input, "model_input", **check)
            try:
                output = forward_fn(model_input)
            except NonFiniteAttributionError as exc:
                # Retain a sampler's denoising-step diagnosis while binding it
                # to the failed IG path point and caller's modality/run identity.
                details = dict(exc.diagnostics)
                details["inner_step"] = details["step"]
                details["step"] = k
                details["alpha"] = alpha
                details["context"] = {**details["context"], **(diagnostic_context or {})}
                raise NonFiniteAttributionError(details) from exc
            if not isinstance(output, torch.Tensor) or not output.is_floating_point() or output.numel() != 1:
                raise ValueError("forward_fn must return a single floating point scalar Tensor")
            _require_finite(output, "score", **check)
            if not output.requires_grad:
                raise ValueError("forward_fn must keep its scalar connected to the input for autograd")
            grad = torch.autograd.grad(output, interpolated)[0]
        _require_finite(grad, "gradient", **check)
        weight = 0.5 if quadrature == "trapezoid" and k in (0, m) else 1.0
        sum_gradients.add_(grad, alpha=weight)
        _require_finite(sum_gradients, "gradient_sum", **check)
        if k in (0, m):
            endpoint_scores[k] = output.detach().reshape(()).to(dtype=arithmetic_dtype)
            _require_finite(endpoint_scores[k], "endpoint_score", **check)
        del interpolated, model_input, output, grad, point

    denominator = m if quadrature == "trapezoid" else m + 1
    average_gradient = sum_gradients / denominator
    _require_finite(average_gradient, "average_gradient", context=diagnostic_context)
    attributions = displacement * average_gradient
    _require_finite(attributions, "attribution", context=diagnostic_context)
    gap = endpoint_scores[m] - endpoint_scores[0]
    total = attributions.sum()
    _require_finite(gap, "endpoint_gap", context=diagnostic_context)
    _require_finite(total, "attribution_sum", context=diagnostic_context)
    residual = (gap - total).abs()
    _require_finite(residual, "absolute_residual", context=diagnostic_context)
    expected = gap.item()
    error = residual.item()
    relative = error / abs(expected) if expected != 0 else None
    if relative is not None and not math.isfinite(relative):
        raise FloatingPointError("The relative completeness residual overflowed")
    result = IntegratedGradientsResult(
        attributions=attributions,
        input_score=endpoint_scores[m].item(),
        baseline_score=endpoint_scores[0].item(),
        expected_gap=expected,
        attribution_sum=total.item(),
        absolute_residual=error,
        relative_residual=relative,
        quadrature=quadrature,
        m=m,
        input_dtype=str(input_tensor.dtype),
        forward_dtype=str(model_dtype),
        arithmetic_dtype=str(arithmetic_dtype),
        elapsed_seconds=time.perf_counter() - started,
    )
    if return_result:
        return result
    relative_text = "undefined (zero gap)" if relative is None else f"{relative:.2%}"
    print(f"completeness: expected={expected:.4f}, actual={result.attribution_sum:.4f}, "
          f"error={error:.4f} ({relative_text}); rule={quadrature}, m={m}")
    return result.attributions
