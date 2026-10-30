"""
DBN-style CNN — 4 stride-2 conv blocks, trained from scratch
(Chapter 3 §3.5.2, Fig. 3.6).

Although the thesis names this branch "DBN-style", it is implemented
here as a small fully-convolutional network trained from scratch on
the left-eye gallery.  The architectural divergence from the ResNet
backbone (no pretraining, different receptive field, different
parameter scale) is what allows fusion (stage 4) to recover errors
where the two matchers disagree (Chapter 3 §3.6 motivation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


class DBNStyleCNN(nn.Module):
    """4-layer custom CNN — distinct from ResNet18, trained from scratch.

    Layer order matches the training script (real_experiments/
    03c_train_v2.py) exactly, so saved checkpoints load directly:
    flat nn.Sequential of (Conv, BN, ReLU) × 4 followed by AvgPool.
    """

    def __init__(self, n_classes: int, emb_dim: int = None):
        super().__init__()
        emb_dim = emb_dim or cfg.EMB_DIM
        self.features = nn.Sequential(
            nn.Conv2d(3,   32, 3, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.Conv2d(32,  64, 3, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding  = nn.Linear(256, emb_dim)
        self.bn         = nn.BatchNorm1d(emb_dim)
        self.dropout    = nn.Dropout(cfg.DROPOUT_DBN)
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x, return_embedding: bool = False):
        z   = self.features(x).flatten(1)
        e   = self.bn(self.embedding(z))
        out = self.classifier(self.dropout(F.relu(e)))
        if return_embedding:
            return out, F.normalize(e, dim=1)
        return out
