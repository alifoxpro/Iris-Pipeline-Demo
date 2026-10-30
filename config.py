"""
Global configuration for the Chapter 3 methodology implementation.

All paths, hyperparameters and constants used across stages 1-5
live here. Every stage module imports `cfg` from this file rather
than hard-coding values, so the implementation matches the values
documented in Chapter 3 of the thesis (UNITEN Iris Segmentation
and Recognition Using Deep Learning Approach).
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent

# Primary dataset (training + within-dataset evaluation)
DATASET    = ROOT / "dataset" / "CASIA-Iris-Thousand"

# Cross-dataset (validation) — see dataset/IITD/README.md for download
IITD_ROOT  = ROOT / "dataset" / "IITD"

OUTPUTS    = ROOT / "outputs"
OUT_SEG    = OUTPUTS / "segmentation"
OUT_NORM   = OUTPUTS / "normalised_templates"
OUT_CKPT   = OUTPUTS / "checkpoints"
OUT_FEAT   = OUTPUTS / "features"
OUT_RES    = OUTPUTS / "results"
OUT_IITD   = OUTPUTS / "iitd"                # cross-dataset artefacts

for d in (OUT_SEG, OUT_NORM, OUT_CKPT, OUT_FEAT, OUT_RES, OUT_IITD):
    d.mkdir(parents=True, exist_ok=True)

# Files
SEG_META   = OUT_SEG  / "segmentation_meta.json"
NORM_NPZ   = OUT_NORM / "normalised.npz"
LBL_NPZ    = OUT_NORM / "labels.npz"

# IITD-specific output files (parallel to the CASIA artefacts)
IITD_SEG_META = OUT_IITD / "segmentation_meta.json"
IITD_NORM_NPZ = OUT_IITD / "normalised.npz"
IITD_LBL_NPZ  = OUT_IITD / "labels.npz"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
N_SUBJECTS = 1000                 # subjects 000..999 (full CASIA-Iris-Thousand)
TARGET_HW  = (240, 320)           # (h, w) — input resize used throughout


# ---------------------------------------------------------------------------
# Stage 1 — Segmentation (Chapter 3, §3.3 and Table 3.4)
# ---------------------------------------------------------------------------
# §3.3.1 Specular reflection removal (Eqs. 3.1-3.2)
SPECULAR_THRESHOLD = 240          # T in R_m = (I >= T)
SPECULAR_DILATE_SE = (5, 5)       # elliptical SE for mask dilation
TELEA_RADIUS       = 3            # inpaint radius (px)

# §3.3.2 Pupil detection (Hough Gradient, Eq. 3.3)
PUPIL_BLUR_KSIZE   = (7, 7)
PUPIL_BLUR_SIGMA   = 1.5
PUPIL_DARK_T       = 70
PUPIL_DP           = 1
PUPIL_MIN_DIST     = 200
PUPIL_CANNY_HIGH   = 50
PUPIL_ACC_THRESH   = 18
PUPIL_R_MIN        = 20
PUPIL_R_MAX        = 70

# §3.3.3 Limbus localisation (CLAHE + Hough, Eq. 3.4)
AHE_CLIP_LIMIT     = 2.0          # CLAHE clipLimit
AHE_TILE_GRID      = (8, 8)
LIMBUS_DP          = 1
LIMBUS_MIN_DIST    = 200
LIMBUS_CANNY_HIGH  = 50
LIMBUS_ACC_THRESH  = 15
LIMBUS_R_MIN_MULT  = 1.7          # rl_min = 1.7 * rp
LIMBUS_R_MAX_MULT  = 3.5          # rl_max = 3.5 * rp

# §3.3.4 Eyelid detection (Perona-Malik + Canny + Radon, Eq. 3.5-3.6)
PM_KAPPA           = 30           # Perona-Malik edge-stop parameter
PM_N_ITER          = 50
PM_DT              = 0.14
EYELID_CANNY_LOW   = 30
EYELID_CANNY_HIGH  = 90
EYELID_GAUSS_SIGMA = 1.0
EYELID_HORIZ_BAND  = (60, 120)    # only keep ±30° around horizontal


# ---------------------------------------------------------------------------
# Stage 2 — Daugman rubber-sheet normalisation (Chapter 3, §3.4)
# ---------------------------------------------------------------------------
POLAR_H = 64                      # radial samples
POLAR_W = 256                     # angular samples


# ---------------------------------------------------------------------------
# Stage 3 — Feature extraction networks (Chapter 3, §3.5)
# ---------------------------------------------------------------------------
EMB_DIM        = 256              # L2-normalised embedding dimension
N_GALLERY      = 6                # gallery samples per class (rest = probe)
SEED           = 42

# Training
BATCH          = 64
EPOCHS         = 80
LR             = 1e-3
WEIGHT_DECAY   = 5e-4
LABEL_SMOOTH   = 0.1
DROPOUT_CNN    = 0.4
DROPOUT_DBN    = 0.3

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Stage 4 — Rank-level fusion (Chapter 3, §3.6)
# ---------------------------------------------------------------------------
LR_TRAIN_FRAC  = 0.5              # half of probes used to fit logistic reg.
IMP_SUBSAMPLE  = 100_000          # cap imposter pairs (memory)


# ---------------------------------------------------------------------------
# Stage 5 — Statistical evaluation (Chapter 3, §3.7 and Table 3.6)
# ---------------------------------------------------------------------------
ALPHA          = 0.05             # significance level (95 % CI)
BOOTSTRAP_B    = 1000             # bootstrap resamples
CMC_K          = 20               # CMC truncation rank


def show():
    """Print a human-readable summary (useful for run_pipeline.py)."""
    print("=" * 64)
    print("Chapter 3 methodology — implementation configuration")
    print("=" * 64)
    print(f"  Dataset           : {DATASET}")
    print(f"  Subjects          : {N_SUBJECTS}")
    print(f"  Input resize      : {TARGET_HW}")
    print(f"  Polar template    : {POLAR_H} x {POLAR_W}")
    print(f"  Embedding dim     : {EMB_DIM}")
    print(f"  Epochs / batch    : {EPOCHS} / {BATCH}")
    print(f"  Device            : {DEVICE}")
    print(f"  Outputs           : {OUTPUTS}")
    print("=" * 64)


if __name__ == "__main__":
    show()
