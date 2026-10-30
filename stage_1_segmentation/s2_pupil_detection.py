"""
Sub-stage 1.2 — Pupil detection (Chapter 3 §3.3.2, Eq. 3.3).

Pupil is the darkest near-circular blob inside the iris.  We apply
a Gaussian blur (sigma = 1.5) and run the Hough-Gradient circle
detector with a radius window (20, 70) px appropriate for the
240 x 320 input.  Among the candidate circles we pick the one whose
centre is darkest — this avoids picking up reflections that survive
the inpainter or the limbus arc.
"""

import numpy as np
import cv2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg


def detect_pupil(gray):
    """Return (cx, cy, rp) — best pupil circle or a safe fallback."""
    blur = cv2.GaussianBlur(gray, cfg.PUPIL_BLUR_KSIZE, cfg.PUPIL_BLUR_SIGMA)

    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp        = cfg.PUPIL_DP,
        minDist   = cfg.PUPIL_MIN_DIST,
        param1    = cfg.PUPIL_CANNY_HIGH,
        param2    = cfg.PUPIL_ACC_THRESH,
        minRadius = cfg.PUPIL_R_MIN,
        maxRadius = cfg.PUPIL_R_MAX,
    )
    if circles is not None:
        # Eq. 3.3 — pick candidate with the darkest centre
        best, best_score = None, 1e9
        for c in circles[0]:
            cx, cy, r = c.astype(int)
            if 0 <= cy < gray.shape[0] and 0 <= cx < gray.shape[1]:
                score = float(blur[cy, cx])           # darker = better
                if score < best_score:
                    best_score = score
                    best = (cx, cy, r)
        if best is not None:
            return best

    # Fallback: centroid of the dark-region mask
    _, dark = cv2.threshold(blur, cfg.PUPIL_DARK_T, 255, cv2.THRESH_BINARY_INV)
    dark    = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    M       = cv2.moments(dark)
    if M["m00"] > 0:
        return (int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"]),
                35)
    h, w = gray.shape
    return (w // 2, h // 2, 35)


if __name__ == "__main__":
    from s1_specular_removal import remove_specular
    sample = next(cfg.DATASET.glob("*/L/*.jpg"))
    img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))
    I_1, _ = remove_specular(img)
    cx, cy, rp = detect_pupil(I_1)
    print(f"sample        : {sample.name}")
    print(f"pupil (cx,cy,rp) = ({cx}, {cy}, {rp})")
