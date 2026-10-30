"""
Iris template dataset + gallery/probe splitter.

Augmentation (training only):
    - random circular shift along the angular axis (0..256 px) — this
      exploits the rotational equivariance of the polar template
    - random horizontal flip
    - brightness / contrast jitter
    - additive Gaussian noise
    - random radial crop + resize back to 64 x 256 (drop-out radial
      bands simulate eyelash occlusion)
"""

from __future__ import annotations
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


class IrisDataset(Dataset):
    """Wraps an (X, y) array pair; replicates grayscale to 3 channels."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = X
        self.y       = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].astype(np.float32) / 255.0           # (64, 256)
        if self.augment:
            x = np.roll(x, np.random.randint(0, 256), axis=1)
            if np.random.rand() < 0.5:
                x = x[:, ::-1].copy()
            x = np.clip(x * np.random.uniform(0.8, 1.2)
                          + np.random.uniform(-0.05, 0.05), 0, 1)
            x += np.random.normal(0, 0.025, x.shape).astype(np.float32)
            if np.random.rand() < 0.4:
                start = np.random.randint(0, 8)
                end   = 64 - np.random.randint(0, 8)
                x     = cv2.resize(x[start:end], (256, 64))
            x = np.clip(x, 0, 1).astype(np.float32)
        x = np.stack([x, x, x], 0)                          # (3, 64, 256)
        return torch.from_numpy(x), int(self.y[i])


def make_gallery_probe_split(X: np.ndarray,
                             subjects: np.ndarray,
                             eyes: np.ndarray,
                             eye_filter: int,
                             n_gallery: int = None,
                             seed: int = None):
    """Split CASIA into gallery (n_gallery per subject) + probe (rest).

    Returns ((Xg, yg), (Xp, yp), classes) where labels are remapped to
    contiguous integers 0..C-1 for the classifier head.
    """
    n_gallery = n_gallery or cfg.N_GALLERY
    seed      = seed      or cfg.SEED

    mask     = eyes == eye_filter
    Xe, se   = X[mask], subjects[mask]
    rng      = np.random.RandomState(seed)
    idx_g, idx_p = [], []
    for s in np.unique(se):
        idx = np.where(se == s)[0]
        rng.shuffle(idx)
        idx_g += list(idx[:n_gallery])
        idx_p += list(idx[n_gallery:])

    classes  = sorted(np.unique(se).tolist())
    cls2idx  = {c: i for i, c in enumerate(classes)}
    yg = np.array([cls2idx[c] for c in se[idx_g]])
    yp = np.array([cls2idx[c] for c in se[idx_p]])
    return (Xe[idx_g], yg), (Xe[idx_p], yp), classes
