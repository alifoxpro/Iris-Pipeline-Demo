"""
Highest-Rank rule (Chapter 3 §3.6.1).

    fused_rank(c) = min( rank_CNN(c), rank_DBN(c) )

For ROC analysis the rule's *score* version takes the higher cosine
of the two matchers.
"""

import numpy as np


def highest_rank(ranks_A, ranks_B):
    """Element-wise minimum of two rank matrices (lower rank = better)."""
    return np.minimum(ranks_A, ranks_B)


def sims_for_highest_rank(sims_A, sims_B):
    """Score version: maximum cosine across the two matchers."""
    return np.maximum(sims_A, sims_B)
