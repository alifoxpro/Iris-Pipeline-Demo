"""
Wilson score 95 % confidence interval (Chapter 3 §3.7.2).

For a binomial proportion p_hat = k / n the Wilson interval
centres on a shifted estimator and is far more accurate than the
classical normal interval in the tails -- the right tool when
reporting rank-1 accuracy with small sample sizes or proportions
near 0 or 1.
"""

import numpy as np
from scipy import stats


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Return (lo, hi) interval in PERCENT units."""
    if n == 0:
        return 0.0, 0.0
    z      = stats.norm.ppf(1 - alpha / 2)
    p_hat  = k / n
    denom  = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half   = (z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))
              / denom)
    return float((centre - half) * 100), float((centre + half) * 100)
