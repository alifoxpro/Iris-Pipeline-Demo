"""
Daugman rubber-sheet normalisation (Chapter 3 §3.4, Eq. 3.7).

Maps the annular iris region between pupil (rp) and limbus (rl) onto
a fixed rectangular template of size POLAR_H x POLAR_W using

    x(r, theta) = cx + (rp + r * (rl - rp)) * cos(theta)
    y(r, theta) = cy + (rp + r * (rl - rp)) * sin(theta)

with bilinear interpolation.  The output template has constant size
regardless of the input iris radius, which is what makes downstream
deep features compare-able.
"""

import numpy as np
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


def daugman_unwrap(img, cx, cy, rp, rl,
                   h_out=None, w_out=None):
    """Polar (rubber-sheet) unwrap of the iris ring."""
    h_out = h_out or cfg.POLAR_H
    w_out = w_out or cfg.POLAR_W

    theta = np.linspace(0, 2 * np.pi, w_out, endpoint=False)
    r     = np.linspace(0, 1, h_out)              # 0 = pupil, 1 = limbus
    R     = rp + r[:, None] * (rl - rp)
    T     = np.broadcast_to(theta, (h_out, w_out))

    X = (cx + R * np.cos(T)).astype(np.float32)
    Y = (cy + R * np.sin(T)).astype(np.float32)
    return cv2.remap(img, X, Y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode  =cv2.BORDER_REFLECT)


if __name__ == "__main__":
    sample = next(cfg.DATASET.glob("*/L/*.jpg"))
    img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))
    # Use ground-truth-style geometry (centre + concentric radii)
    polar = daugman_unwrap(img, 160, 120, 38, 99)
    print(f"sample : {sample.name}")
    print(f"polar  : {polar.shape}, dtype={polar.dtype}")
