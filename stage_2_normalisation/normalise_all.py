"""
Stage 2 driver — load segmentation_meta.json from stage 1 and unwrap
every iris into a 64 x 256 polar template.

Outputs:
    outputs/normalised_templates/normalised.npz   (N, 64, 256) uint8
    outputs/normalised_templates/labels.npz       subjects, eyes, srcs
    outputs/normalised_templates/sample_*.png     8 visual samples
"""

import json, time
from pathlib import Path
import numpy as np
import cv2

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg
from stage_2_normalisation.daugman_unwrap import daugman_unwrap


def main():
    print("[Stage 2] Daugman rubber-sheet normalisation")
    print(f"  Template : {cfg.POLAR_H} x {cfg.POLAR_W}")

    meta = json.loads(cfg.SEG_META.read_text())
    keys = sorted(meta.keys())
    print(f"  Inputs   : {len(keys)}")

    arr      = np.zeros((len(keys), cfg.POLAR_H, cfg.POLAR_W), dtype=np.uint8)
    subjects = np.zeros(len(keys), dtype=np.int32)
    eyes     = np.zeros(len(keys), dtype=np.int32)
    srcs     = []

    t0 = time.time()
    for i, k in enumerate(keys):
        m   = meta[k]
        img = cv2.imread(m["src"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img        = cv2.resize(img, (m["w"], m["h"]))
        arr[i]     = daugman_unwrap(img, m["cx"], m["cy"], m["rp"], m["rl"])
        subjects[i] = m["subject"]
        eyes[i]     = 0 if m["eye"] == "L" else 1
        srcs.append(m["src"])
        if (i + 1) % 1000 == 0:
            print(f"    {i+1:5d}/{len(keys)}  ({time.time()-t0:.0f}s)")

    np.savez_compressed(cfg.NORM_NPZ, templates=arr)
    np.savez(cfg.LBL_NPZ, subjects=subjects, eyes=eyes,
             srcs=np.array(srcs))
    print(f"  Saved    : {cfg.NORM_NPZ}  ({arr.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  Saved    : {cfg.LBL_NPZ}")

    # Save 8 visual samples for the appendix
    for i in range(min(8, len(keys))):
        cv2.imwrite(str(cfg.OUT_NORM / f"sample_{i:02d}.png"), arr[i])
    print(f"  Samples  : 8 PNGs in {cfg.OUT_NORM}")
    print(f"  Time     : {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
