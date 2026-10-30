"""
Sub-stage 1.1 — Specular reflection removal (Chapter 3 §3.3.1, Eqs. 3.1-3.2).

Detects bright NIR reflections via intensity threshold T = 240, dilates
the mask with a 5x5 elliptical structuring element, then fills the
masked region with Telea fast-marching inpainting (radius = 3 px).
"""

import numpy as np
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


def remove_specular(gray):
    """Eq. 3.1 + 3.2: build R_m, dilate to M_d, inpaint to I_1."""
    # Eq. 3.1 — binary reflection mask
    _, R_m = cv2.threshold(gray, cfg.SPECULAR_THRESHOLD, 255, cv2.THRESH_BINARY)

    # Morphological dilation to cover halo around each reflection
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg.SPECULAR_DILATE_SE)
    M_d = cv2.dilate(R_m, se, iterations=1)

    if M_d.sum() == 0:
        return gray.copy(), M_d

    # Eq. 3.2 — Telea fast-marching inpainting
    I_1 = cv2.inpaint(gray, M_d, cfg.TELEA_RADIUS, cv2.INPAINT_TELEA)
    return I_1, M_d


if __name__ == "__main__":
    # Smoke test on the first CASIA image
    sample = next(cfg.DATASET.glob("*/L/*.jpg"))
    img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))
    I_1, M_d = remove_specular(img)
    print(f"sample        : {sample.name}")
    print(f"reflection px : {(M_d > 0).sum()}")
    print(f"max change    : {np.abs(img.astype(int) - I_1.astype(int)).max()}")
