"""Coverage of percentile episode-bootstrap intervals, checked before unblinding (analysis plan R1).

Episode means are drawn from distributions with a known mean, including heavy
tails and rare extreme episodes like the overshoot tails seen in saved curves.
The interval is the one the methods study uses: resample episodes with
replacement, take the mean, report percentiles.

Usage: python -m analysis.revision_gpu.v1.coverage --episodes 60 --out FILE
"""
import argparse
import json
from pathlib import Path

import numpy as np


SCENARIOS = ("normal", "student_t3", "lognormal_skew", "rare_extreme_negative", "two_point_outlier")


def draw(name, rng, n):
    if name == "normal":
        return rng.normal(0.1, 0.05, n), 0.1
    if name == "student_t3":
        return 0.1 + 0.05 * rng.standard_t(3, n), 0.1
    if name == "lognormal_skew":
        sigma = 1.0
        values = rng.lognormal(0.0, sigma, n)
        return 0.05 * values, 0.05 * np.exp(sigma ** 2 / 2)
    if name == "rare_extreme_negative":
        # 5% of episodes carry a large negative tail, as in the saved vision-insertion means.
        extreme = rng.random(n) < 0.05
        values = np.where(extreme, rng.normal(-3.0, 1.0, n), rng.normal(0.1, 0.05, n))
        return values, 0.95 * 0.1 + 0.05 * -3.0
    if name == "two_point_outlier":
        # Amendment 2: the plan's two-point outlier distribution, point masses at 0.1 and -3.0 (5%).
        return np.where(rng.random(n) < 0.05, -3.0, 0.1), 0.95 * 0.1 + 0.05 * -3.0
    raise ValueError(name)


def bca_interval(values, boot, alpha):
    """Bias-corrected and accelerated percentile interval for the mean."""
    from scipy.stats import norm
    n = len(values)
    theta = values.mean()
    below = np.mean(boot < theta)
    z0 = norm.ppf(min(max(below, 1 / (2 * len(boot))), 1 - 1 / (2 * len(boot))))
    jackknife = (values.sum() - values) / (n - 1)
    centred = jackknife.mean() - jackknife
    denominator = 6 * (np.sum(centred ** 2) ** 1.5)
    a = np.sum(centred ** 3) / denominator if denominator > 0 else 0.0
    out = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        level = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(np.quantile(boot, level))
    return out


def t_interval(values, alpha):
    from scipy.stats import t
    n = len(values)
    se = values.std(ddof=1) / np.sqrt(n)
    half = t.ppf(1 - alpha / 2, n - 1) * se
    return [values.mean() - half, values.mean() + half]


def coverage(name, n, alphas, sims, draws, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    methods = ("percentile", "bca", "t")
    hits = {(m, a): 0 for m in methods for a in alphas}
    widths = {(m, a): [] for m in methods for a in alphas}
    for _ in range(sims):
        values, mean = draw(name, rng, n)
        index = rng.integers(0, n, size=(draws, n))
        boot = values[index].mean(axis=1)
        for alpha in alphas:
            intervals = dict(percentile=np.quantile(boot, [alpha / 2, 1 - alpha / 2]),
                             bca=bca_interval(values, boot, alpha), t=t_interval(values, alpha))
            for method, (low, high) in intervals.items():
                hits[(method, alpha)] += int(low <= mean <= high)
                widths[(method, alpha)].append(high - low)
    return {f"{method}/{alpha:.6f}": dict(nominal=1 - alpha, coverage=hits[(method, alpha)] / sims,
                                          mean_width=float(np.mean(widths[(method, alpha)])))
            for method in methods for alpha in alphas}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+", default=[30, 40, 60])
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=930013)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    alphas = [0.05, 0.05 / 7, 0.05 / 4]
    result = dict(kind="bootstrap_coverage_simulation", sims=args.sims, draws=args.draws, seed=args.seed,
                  interval="percentile, episodes resampled with replacement", results={})
    for k, n in enumerate(args.episodes):
        for j, name in enumerate(SCENARIOS):
            result["results"][f"{name}/n{n}"] = coverage(name, n, alphas, args.sims, args.draws, args.seed + 100 * k + j)
    Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    for key, value in result["results"].items():
        print(key, " ".join(f"{a.split('/')[0][:4]}@{1 - float(a.split('/')[1]):.4f}={v['coverage']:.3f}" for a, v in value.items()))


if __name__ == "__main__":
    main()
