"""
Sub-stage 1.3 — Limbus (outer iris boundary) detection
   (Chapter 3 §3.3.3, Eq. 3.4 and Fig. 3.9).

Pipeline:
    1. AHE (CLAHE clipLimit = 2.0, tile 8x8) to lift iris contrast
    2. Gaussian blur to suppress eyelash texture
    3. Hough-Gradient with radius window (1.7 rp, 3.5 rp), where rp
       is the pupil radius from sub-stage 1.2
    4. Pick the candidate whose centre lies closest to the pupil
       centre and snap the centre back to the pupil centre for
       geometric stability (small concentric assumption).
"""

import numpy as np
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


def _apply_ahe(gray):
    """Adaptive Histogram Equalisation — Fig. 3.9 of Chapter 3."""
    clahe = cv2.createCLAHE(clipLimit=cfg.AHE_CLIP_LIMIT,
                            tileGridSize=cfg.AHE_TILE_GRID)
    return clahe.apply(gray)


def detect_limbus(gray, pupil):
    """Return (cx, cy, rl) — limbus circle anchored on the pupil."""
    cx, cy, rp = pupil
    enh  = _apply_ahe(gray)
    blur = cv2.GaussianBlur(enh, cfg.PUPIL_BLUR_KSIZE, cfg.PUPIL_BLUR_SIGMA)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp        = cfg.LIMBUS_DP,
        minDist   = cfg.LIMBUS_MIN_DIST,
        param1    = cfg.LIMBUS_CANNY_HIGH,
        param2    = cfg.LIMBUS_ACC_THRESH,
        minRadius = int(rp * cfg.LIMBUS_R_MIN_MULT),
        maxRadius = int(rp * cfg.LIMBUS_R_MAX_MULT),
    )
    if circles is not None:
        # Eq. 3.4 — choose limbus closest to the pupil centre
        best, best_d = None, 1e9
        for c in circles[0]:
            ccx, ccy, rl = c.astype(int)
            d = (ccx - cx) ** 2 + (ccy - cy) ** 2
            if d < best_d and rl > rp + 10:
                best_d, best = d, (ccx, ccy, rl)
        if best is not None:
            # Snap centre to pupil centre (concentric assumption)
            return (cx, cy, best[2])

    # Fallback: rl = 2.5 * rp (empirical CASIA prior)
    return (cx, cy, int(rp * 2.5))


if __name__ == "__main__":
    from s1_specular_removal import remove_specular
    from s2_pupil_detection  import detect_pupil
    sample = next(cfg.DATASET.glob("*/L/*.jpg"))
    img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))
    I_1, _   = remove_specular(img)
    pupil    = detect_pupil(I_1)
    limbus   = detect_limbus(I_1, pupil)
    print(f"sample : {sample.name}")
    print(f"pupil  : {pupil}")
    print(f"limbus : {limbus}")
