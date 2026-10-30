"""
Logistic-Regression rank fusion (Chapter 3 §3.6.3).

Train a single logistic-regression classifier on (sim_CNN, sim_DBN)
pairs labelled 1 if the candidate class matches the probe's ground
truth and 0 otherwise.  At inference time the LR posterior is used
as the fused class score.

Half the probes are reserved for training the LR; the remaining
half provides an unbiased estimate of fused performance (see
Stage 5 evaluation).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


def _build_features(sims_A, sims_B, y_true, probe_idx):
    n_cls = sims_A.shape[1]
    X, Y  = [], []
    for i in probe_idx:
        for c in range(n_cls):
            X.append([sims_A[i, c], sims_B[i, c]])
            Y.append(1 if c == y_true[i] else 0)
    return np.array(X), np.array(Y)


def logistic_regression_fusion(sims_A, sims_B, y_true,
                               train_frac=None, seed=None):
    """Return fused per-class scores (N, C) + held-out probe indices."""
    train_frac = train_frac if train_frac is not None else cfg.LR_TRAIN_FRAC
    seed       = seed       if seed       is not None else cfg.SEED

    rng     = np.random.RandomState(seed)
    idx     = np.arange(len(y_true))
    rng.shuffle(idx)
    n_train = int(train_frac * len(idx))
    tr_idx, te_idx = idx[:n_train], idx[n_train:]

    Xtr, Ytr = _build_features(sims_A, sims_B, y_true, tr_idx)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(Xtr, Ytr)

    fused = np.zeros_like(sims_A)
    for i in range(len(y_true)):
        feats = np.stack([sims_A[i], sims_B[i]], 1)
        fused[i] = lr.predict_proba(feats)[:, 1]
    return fused, te_idx
