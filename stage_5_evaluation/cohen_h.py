"""
Cohen's effect size h between two proportions (Chapter 3 §3.7.4).

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

Conventional cut-offs:
    |h| < 0.2  trivial
    |h| < 0.5  small
    |h| < 0.8  medium
    |h| >= 0.8 large

Used to quantify the practical magnitude of an accuracy gap on top
of McNemar's p-value (which only tests existence, not size).
"""

import numpy as np


def cohen_h(p1: float, p2: float) -> float:
    """p1, p2 are accuracies in [0, 1]."""
    p1 = float(np.clip(p1, 0, 1))
    p2 = float(np.clip(p2, 0, 1))
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def cohen_h_label(h: float) -> str:
    a = abs(h)
    if a < 0.2: return "trivial"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"
