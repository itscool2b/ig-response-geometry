# Response transformations, path weights and IG ranks

This note gives elementary derivations for the scalar responses used in the revision. It claims neither a new general metric principle nor validation of unidentified historical RDT contexts. The accompanying Python module is an independent CPU analysis and does not modify the frozen model runtime.

## Assumptions and production definitions

Hold the action reference, active-coordinate membership, noise, model, other modalities and interpolation path fixed. Let the differentiable predicted active action be \(\mu(x)\), the detached reference be \(a\), and

\[
r(x)=\|a-\mu(x)\|_2^2,\qquad
Q(x)=-\frac{r(x)}{2D\sigma^2},\qquad
L(x)=-\sqrt{r(x)+\epsilon}.
\]

Here \(D>0\) counts active predicted-action entries, \(\sigma^2>0\) is constant and \(\epsilon>0\). The production common-score implementation uses \(\sigma^2=1\), \(\epsilon=10^{-12}\) and `difference.numel()` for D. For one 64-step chunk and eight active dimensions, D=512. The two-feature constructive example below intentionally uses D=1 because its action is scalar.

The statements apply wherever the relevant gradients exist, and to IG when its path integrands are integrable. They compare the same function inputs and reference. A changed reference, state mapping, model precision or noise is an additional intervention.

## Exact relation between path gradients

By the chain rule,

\[
\nabla Q=-\frac{\nabla r}{2D\sigma^2},\qquad
\nabla L=-\frac{\nabla r}{2\sqrt{r+\epsilon}},\qquad
\boxed{\nabla Q=\frac{\sqrt{r+\epsilon}}{D\sigma^2}\nabla L}.
\]

The scalar multiplier is positive. Thus nonzero pointwise gradient vectors have the same direction. At exact self-reference both gradients are zero; stabilized L has no residual-space kink there.

For \(x(\alpha)=x'+\alpha(x-x')\), this gives

\[
\mathrm{IG}^{Q}_i=(x_i-x_i')\int_0^1
\frac{\sqrt{r(x(\alpha))+\epsilon}}{D\sigma^2}
\partial_i L(x(\alpha))\,d\alpha.
\]

The weight generally varies along the path. It can change feature ordering when coordinate contributions vary differently along the path or cancel. A positive pointwise scalar factor does not imply that the integrated vectors are proportional. Conversely, rank changes are not guaranteed: an effectively constant weight or path gradients with one fixed coordinate direction can preserve ranks. None of this violates IG implementation invariance, which concerns implementations of the same scalar function.

## Normalized-response ordering and its exact exceptions

Assume an exact self-reference actual endpoint, so its squared residual is zero, and an all-baseline squared residual \(R>0\). For any intervention residual \(r\ge0\), actual-to-baseline normalization gives

\[
N_Q(r)=1-\frac rR,\qquad
N_L(r)=\frac{\sqrt{R+\epsilon}-\sqrt{r+\epsilon}}
{\sqrt{R+\epsilon}-\sqrt\epsilon}.
\]

Set \(a_0=\sqrt\epsilon\), \(b=\sqrt{R+\epsilon}\), and \(z=\sqrt{r+\epsilon}\). Rationalization gives the exact identity

\[
\boxed{N_Q(r)-N_L(r)=\frac{(z-a_0)(b-z)}{R}}.
\]

Consequently:

- For \(0<r<R\), \(N_Q>N_L\); the endpoints r=0 and r=R give equality.
- For r>R, \(N_Q<N_L\). Negative normalized outcomes remain valid overshoots and must not be clipped.
- If R=0, both endpoint gaps are zero and both normalized quantities are undefined. The stabilizer does not repair normalization.
- If every point of a fixed intervention curve remains in range, integration with nonnegative weights preserves the ordering. This includes continuous integration and trapezoidal area on a nondecreasing realized-fraction grid. The intervention curve need not be monotone.
- Mixed in-range and overshooting points can produce either area ordering; the theorem does not assign their net sign.

In the unstabilized limit, writing \(u=\sqrt{r/R}\) gives \(N_Q-N_L=u(1-u)\). The stabilized identity remains exact and requires no small-epsilon approximation. For in-range residuals its maximum is \((b-a_0)^2/(4R)\le1/4\).

This compares evaluation responses while holding each ranking's interventions fixed. It does not compare different ranking populations, assign random order an AUC of 0.5, prove causal denoiser exclusion, or turn area changes into task-success improvements. The prospective study's raw positive RMS primary response remains a separately declared common outcome.

## A constructive rank reversal with the same path

Consider the smooth scalar action \(\mu(x_1,x_2)=x_1^2+cx_2\), baseline (0,0), actual (1,1), reference 1+c, and D=\(\sigma^2=1\). Along the straight path (t,t),

\[
d(t)=1+c-t^2-ct=(1-t)(1+t+c)\ge0.
\]

For the quadratic target, the coordinate path integrands are \(2td(t)\) and \(cd(t)\). Exact integration gives

\[
\mathrm{IG}^{Q}=\left(\frac12+\frac c3,\frac{2c}3+\frac{c^2}2\right).
\]

For the unstabilized negative norm, the path gradient equals \((2t,c)\) for t<1, and its limiting integral is (1,c). The endpoint at t=1 has measure zero in this limiting calculation; the production target instead uses positive epsilon and is smooth everywhere.

At c=4/5,

\[
\mathrm{IG}^{Q}=\left(\frac{23}{30},\frac{64}{75}\right),\qquad
\lim_{\epsilon\downarrow0}\mathrm{IG}^{L}=\left(1,\frac45\right).
\]

All coordinates are positive. Q ranks feature 2 first with exact margin 13/150. L ranks feature 1 first. To establish the reversal for epsilon=\(10^{-12}\), not merely its limit, use \(d(t)\ge c(1-t)\) and the decreasing function \(1-d/\sqrt{d^2+\epsilon}\):

\[
\int_0^1\left(1-\frac{d(t)}{\sqrt{d(t)^2+\epsilon}}\right)dt
\le1-\frac{\sqrt{c^2+\epsilon}-\sqrt\epsilon}{c}
\le\frac{\sqrt\epsilon}{c}.
\]

Multiplying by the coordinate derivative bounds gives

\[
0\le1-\mathrm{IG}^{L}_1\le\frac{2\sqrt\epsilon}{c}=2.5\cdot10^{-6},\qquad
0\le c-\mathrm{IG}^{L}_2\le\sqrt\epsilon=10^{-6}.
\]

Therefore the stabilized L ranking margin exceeds 0.1999975, so the reversal is robust. This is a constructive nonlinear policy example, not an RDT measurement or a claim that every response transformation reverses ranks.

## Reproduction

Run `python -m pytest tests/test_response_geometry_theory.py -q`. Tests independently differentiate a nonlinear vector action with nonunit sigma squared; check the normalization identity, endpoints, overshoots, undefined gaps and irregular-grid areas; integrate the polynomial Q gradients; and verify the stabilized L bounds and completeness. A second member of the same policy family preserves the ranking, checking the scope of the existence claim.

Run `python -m analysis.revision.response_geometry_theory --out NEW_REPORT.json` for the exact rational example, stabilized quadrature values, analytical error bounds and numerical integration diagnostics. The output path must be fresh. No GPU, model checkpoint or historical input is accessed.
