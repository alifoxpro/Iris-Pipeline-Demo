"""Stage 5 — Statistical significance + reporting (Chapter 3 §3.7).

Four standalone test modules:
    mcnemar_test     — McNemar's exact paired test
    wilson_ci        — Wilson 95 % CI on a Bernoulli proportion
    bootstrap_ci     — non-parametric CI on AUC / EER
    cohen_h          — effect size h between two proportions

The driver `evaluate.py` runs all of them on the fused results and
writes outputs/results/significance.json.
"""
from .mcnemar_test  import mcnemar_test
from .wilson_ci     import wilson_ci
from .bootstrap_ci  import bootstrap_auc_ci
from .cohen_h       import cohen_h
