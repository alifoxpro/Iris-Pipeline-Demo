"""
Stratified bootstrap 95 % CI on AUC (Chapter 3 §3.7.3).

We resample genuine and imposter score arrays SEPARATELY (stratified
bootstrap) B times, recompute the ROC AUC for each replicate, and
report the 2.5/97.5 percentile interval.  No parametric assumption
on the score distribution is needed.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


def bootstrap_auc_ci(gen: np.ndarray, imp: np.ndarray,
                     B: int = None, seed: int = None):
    """Return (auc_point, lo, hi, std) on the AUC distribution."""
    B    = B    if B    is not None else cfg.BOOTSTRAP_B
    seed = seed if seed is not None else cfg.SEED
    rng  = np.random.RandomState(seed)

    aucs = np.empty(B)
    for b in range(B):
        g_b = gen[rng.randint(0, len(gen), len(gen))]
        i_b = imp[rng.randint(0, len(imp), len(imp))]
        y   = np.concatenate([np.ones_like(g_b), np.zeros_like(i_b)])
        s   = np.concatenate([g_b, i_b])
        fpr, tpr, _ = roc_curve(y, s)
        aucs[b] = auc(fpr, tpr)

    # Point AUC computed on the unshuffled data (matches Stage 4)
    y0 = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    s0 = np.concatenate([gen, imp])
    fpr0, tpr0, _ = roc_curve(y0, s0)
    auc0 = auc(fpr0, tpr0)

    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(auc0), float(lo), float(hi), float(aucs.std())
