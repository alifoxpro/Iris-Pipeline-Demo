"""
Sub-stage 1.4 — Eyelid suppression (Chapter 3 §3.3.4, Eqs. 3.5-3.6).

Approach:
    1. Perona-Malik anisotropic diffusion (Eq. 3.5) smooths eyelash
       texture while preserving the strong eyelid edge.
    2. Modified Canny restricted to mostly-horizontal gradients
       (60 deg <= |angle| <= 120 deg) extracts eyelid arcs only.
    3. Strongest horizontal edge above the pupil = upper eyelid;
       below the pupil = lower eyelid.

A full Radon-transform line fit (Eq. 3.6) is illustrated in
Figure 3.13 of the thesis; here we use the simpler row-energy
maximum which is what was actually used in the empirical pipeline
that produced Chapter 4's numbers.
"""

import numpy as np
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


def perona_malik(img, kappa=None, n_iter=None, dt=None):
    """Eq. 3.5 — exponential edge-stop diffusion."""
    kappa  = kappa  if kappa  is not None else cfg.PM_KAPPA
    n_iter = n_iter if n_iter is not None else cfg.PM_N_ITER
    dt     = dt     if dt     is not None else cfg.PM_DT
    out = img.astype(np.float64)
    for _ in range(n_iter):
        gN = np.roll(out, -1, axis=0) - out
        gS = np.roll(out,  1, axis=0) - out
        gE = np.roll(out, -1, axis=1) - out
        gW = np.roll(out,  1, axis=1) - out
        cN = np.exp(-(gN / kappa) ** 2)
        cS = np.exp(-(gS / kappa) ** 2)
        cE = np.exp(-(gE / kappa) ** 2)
        cW = np.exp(-(gW / kappa) ** 2)
        out = out + dt * (cN * gN + cS * gS + cE * gE + cW * gW)
    return np.clip(out, 0, 255).astype(np.uint8)


def modified_canny_horizontal(gray):
    """Canny edges restricted to roughly horizontal orientations."""
    blurred = cv2.GaussianBlur(gray, (5, 5), cfg.EYELID_GAUSS_SIGMA)
    edges   = cv2.Canny(blurred, cfg.EYELID_CANNY_LOW, cfg.EYELID_CANNY_HIGH)

    sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.abs(np.arctan2(sy, sx) * 180 / np.pi)
    lo, hi = cfg.EYELID_HORIZ_BAND
    horizontal = (angle > lo) & (angle < hi)
    return edges * horizontal.astype(np.uint8)


def detect_eyelids(gray, pupil, limbus):
    """Return (upper_y, lower_y) — row indices of the two eyelid lines."""
    cx, cy, rp = pupil
    _, _,  rl  = limbus
    h, w       = gray.shape

    # Anisotropic diffusion + horizontal Sobel energy per row
    diff = perona_malik(gray)
    sy   = np.abs(cv2.Sobel(diff, cv2.CV_32F, 0, 1, ksize=3))

    # Search bands: above the pupil for the upper eyelid,
    # below the pupil for the lower eyelid
    upper_top = max(cy - rl - 30, 0)
    upper_band = sy[upper_top:cy - rp,
                    max(cx - rl, 0):min(cx + rl, w)]
    lower_band = sy[cy + rp:min(cy + rl + 30, h),
                    max(cx - rl, 0):min(cx + rl, w)]

    upper_y = upper_top
    if upper_band.size > 0:
        upper_y += int(np.argmax(upper_band.sum(axis=1)))
    lower_y = cy + rp
    if lower_band.size > 0:
        lower_y += int(np.argmax(lower_band.sum(axis=1)))
    return upper_y, lower_y


if __name__ == "__main__":
    from s1_specular_removal import remove_specular
    from s2_pupil_detection  import detect_pupil
    from s3_limbus_detection import detect_limbus
    sample = next(cfg.DATASET.glob("*/L/*.jpg"))
    img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))
    I_1, _ = remove_specular(img)
    pupil  = detect_pupil(I_1)
    limbus = detect_limbus(I_1, pupil)
    upper, lower = detect_eyelids(I_1, pupil, limbus)
    print(f"sample : {sample.name}")
    print(f"upper  : y = {upper}")
    print(f"lower  : y = {lower}")
