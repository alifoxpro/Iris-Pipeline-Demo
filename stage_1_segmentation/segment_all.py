"""
Stage 1 driver — runs the four segmentation sub-stages on every
CASIA-Iris-Thousand image and writes:

    outputs/segmentation/segmentation_meta.json   (per-image geometry)
    outputs/segmentation/<key>.png                (every 50th overlay)

Each entry in segmentation_meta.json has the form:

    "000_L_S5000L00": {
        "cx": 160, "cy": 122, "rp": 38, "rl": 99,
        "upper": 22, "lower": 215,
        "h": 240, "w": 320,
        "subject": 0, "eye": "L",
        "src": "<absolute path to .jpg>"
    }
"""

import json, time
from pathlib import Path
import numpy as np
import cv2

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg
from stage_1_segmentation.s1_specular_removal import remove_specular
from stage_1_segmentation.s2_pupil_detection  import detect_pupil
from stage_1_segmentation.s3_limbus_detection import detect_limbus
from stage_1_segmentation.s4_eyelid_detection import detect_eyelids


def segment_one(path):
    """Run all four sub-stages on a single image."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    img = cv2.resize(img, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))

    I_1, _       = remove_specular(img)
    pupil        = detect_pupil(I_1)
    limbus       = detect_limbus(I_1, pupil)
    upper, lower = detect_eyelids(I_1, pupil, limbus)

    cx, cy, rp = pupil
    _, _, rl   = limbus

    meta = dict(cx=int(cx), cy=int(cy), rp=int(rp), rl=int(rl),
                upper=int(upper), lower=int(lower),
                h=int(cfg.TARGET_HW[0]), w=int(cfg.TARGET_HW[1]))

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.circle(vis, (cx, cy), rp, (0, 255, 0),   2)
    cv2.circle(vis, (cx, cy), rl, (255, 0, 0),   2)
    cv2.line  (vis, (0, upper), (vis.shape[1], upper), (0, 200, 255), 1)
    cv2.line  (vis, (0, lower), (vis.shape[1], lower), (0, 200, 255), 1)
    return meta, vis


def main():
    print("[Stage 1] Iris Segmentation")
    print(f"  Dataset  : {cfg.DATASET}")
    print(f"  Subjects : {cfg.N_SUBJECTS}")

    # Collect image paths
    paths = []
    for sid in range(cfg.N_SUBJECTS):
        for eye in ("L", "R"):
            d = cfg.DATASET / f"{sid:03d}" / eye
            if d.exists():
                for jpg in sorted(d.glob("*.jpg")):
                    paths.append((sid, eye, jpg))
    print(f"  Images   : {len(paths)}")

    meta_all = {}
    success, failed = 0, 0
    t0 = time.time()
    for i, (sid, eye, p) in enumerate(paths):
        key = f"{sid:03d}_{eye}_{p.stem}"
        try:
            meta, vis = segment_one(p)
            if meta is None:
                failed += 1
                continue
            meta_all[key] = {**meta,
                             "src":     str(p),
                             "subject": sid,
                             "eye":     eye}
            if i % 50 == 0:
                cv2.imwrite(str(cfg.OUT_SEG / f"{key}.png"), vis)
            success += 1
        except Exception as e:
            print(f"    [FAIL] {key}: {e}")
            failed += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(paths) - i - 1)
            print(f"    {i+1:5d}/{len(paths)}  "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s ETA)")

    cfg.SEG_META.write_text(json.dumps(meta_all, indent=1))
    print()
    print(f"  Success : {success}  | Failed: {failed}")
    print(f"  Time    : {time.time() - t0:.1f}s")
    print(f"  Meta    : {cfg.SEG_META}")
    print(f"  Vis     : {cfg.OUT_SEG}  (every 50th image)")


if __name__ == "__main__":
    main()
