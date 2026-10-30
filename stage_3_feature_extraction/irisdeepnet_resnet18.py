"""
IrisDeepNet — ResNet-18 backbone + 256-D embedding head
(Chapter 3 §3.5.1, Fig. 3.5).

ImageNet-pretrained ResNet-18 is used as a transferable feature
extractor.  Polar templates are replicated to 3 channels and fed
directly to the network; the final fully-connected layer is
replaced with a 256-D embedding + BatchNorm + Dropout + classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


class IrisDeepNetResNet18(nn.Module):
    """ResNet-18 feature extractor + 256-D L2-normalised embedding."""

    def __init__(self, n_classes: int, emb_dim: int = None,
                 pretrained: bool = True):
        super().__init__()
        emb_dim = emb_dim or cfg.EMB_DIM
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb      = resnet18(weights=weights)
        self.features  = nn.Sequential(*list(bb.children())[:-1])    # ->(B,512,1,1)
        self.embedding = nn.Linear(512, emb_dim)
        self.bn        = nn.BatchNorm1d(emb_dim)
        self.dropout   = nn.Dropout(cfg.DROPOUT_CNN)
        self.classifier= nn.Linear(emb_dim, n_classes)

    def forward(self, x, return_embedding: bool = False):
        z   = self.features(x).flatten(1)
        e   = self.bn(self.embedding(z))
        out = self.classifier(self.dropout(F.relu(e)))
        if return_embedding:
            return out, F.normalize(e, dim=1)         # unit-norm embedding
        return out
