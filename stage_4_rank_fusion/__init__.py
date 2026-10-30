"""Stage 4 — Rank-level fusion (Chapter 3 §3.6).

Three fusion rules combine the per-class scores of IrisDeepNet and
the DBN-style network:

    highest_rank        — pick the smaller (better) rank of the two
    borda_count         — sum of ranks (lower = better)
    logistic_regression — class score = LR( cosCNN, cosDBN ) trained
                          on half the probes
"""
from .rule_highest_rank        import highest_rank, sims_for_highest_rank
from .rule_borda_count         import borda_count,   sims_for_borda_count
from .rule_logistic_regression import logistic_regression_fusion
