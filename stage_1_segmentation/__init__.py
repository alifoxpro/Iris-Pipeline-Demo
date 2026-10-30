"""Stage 1 — Iris segmentation (Chapter 3 §3.3).

Four sequential sub-stages:
    s1_specular_removal — Telea inpainting           (Eqs. 3.1-3.2)
    s2_pupil_detection  — Hough Gradient on dark blob (Eq. 3.3)
    s3_limbus_detection — CLAHE + Hough on outer ring (Eq. 3.4)
    s4_eyelid_detection — Perona-Malik + Canny+Radon  (Eqs. 3.5-3.6)

The driver `segment_all` runs all four in order on every CASIA image
and writes `outputs/segmentation/segmentation_meta.json`.
"""
from .s1_specular_removal import remove_specular
from .s2_pupil_detection  import detect_pupil
from .s3_limbus_detection import detect_limbus
from .s4_eyelid_detection import detect_eyelids
