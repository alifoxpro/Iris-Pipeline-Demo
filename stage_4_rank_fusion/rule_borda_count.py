"""
Borda Count rule (Chapter 3 §3.6.2).

    fused_rank(c) = rank_CNN(c) + rank_DBN(c)

A class needs to be ranked well by BOTH matchers to win, so single
spurious top-1 mistakes are softened.  The corresponding score-level
operation is the mean of cosines.
"""

import numpy as np


def borda_count(ranks_A, ranks_B):
    """Element-wise sum of two rank matrices (lower sum = better)."""
    return ranks_A + ranks_B


def sims_for_borda_count(sims_A, sims_B):
    """Score version: mean cosine across the two matchers."""
    return (sims_A + sims_B) / 2.0
