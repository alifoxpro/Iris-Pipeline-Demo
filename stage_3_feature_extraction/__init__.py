"""Stage 3 — Feature extraction networks (Chapter 3 §3.5).

Two architecturally distinct networks share a 256-D L2-normalised
embedding layer:

    IrisDeepNetResNet18 — ResNet-18 backbone, ImageNet pretrained
    DBNStyleCNN         — 4-layer custom CNN, trained from scratch

Both networks are trained with cross-entropy + label-smoothing on
the right eye (IrisDeepNet) or left eye (DBN-style) gallery split
and produce L2-normalised embeddings used by stage 4 for fusion.
"""
from .iris_dataset            import IrisDataset, make_gallery_probe_split
from .irisdeepnet_resnet18    import IrisDeepNetResNet18
from .dbn_style_cnn           import DBNStyleCNN
