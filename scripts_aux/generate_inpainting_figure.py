"""
Generate Figure 3.10 — Telea Fast-Marching Inpainting demonstration
for Stage 1 of the iris segmentation pipeline.

The figure shows the four-step specular-reflection removal process
on a real CASIA-Iris-Thousand sample with prominent reflections.

Output:
  thesis_chapters/chapter_3_figures/figure_3_10.png  (300 DPI)
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

OUT = Path("E:/Strudin/Jamal Nasir Al-khafaji/Correct_theiss/thesis_chapters/chapter_3_figures")
OUT.mkdir(parents=True, exist_ok=True)

CASIA_PATH = Path("E:/Strudin/Jamal Nasir Al-khafaji/innovatorsoasis-iris_hind/CASIA-Iris-Thousand")

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "figure.dpi":       100,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        False,
})


def find_image_with_reflections():
    """Find a CASIA image with visible specular reflections.
    Uses T=200 threshold for search (visualization), but the actual
    pipeline still uses T=240."""
    # Search across many subjects
    candidates = list(CASIA_PATH.glob("*/L/*.jpg"))[:300]
    if not candidates:
        candidates = list(CASIA_PATH.glob("**/*.jpg"))[:300]

    best = None
    best_score = 0
    for p in candidates:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (320, 280))
        # Use T=200 for search to find images with brightish patches
        _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        n_reflection_pixels = (mask > 0).sum()
        if 800 < n_reflection_pixels < 8000 and n_reflection_pixels > best_score:
            best_score = n_reflection_pixels
            best = img
    if best is None:
        best = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
        best = cv2.resize(best, (320, 280))
    print(f"Selected image with {best_score} bright pixels (T=200 search)")
    return best


def main():
    img = find_image_with_reflections()

    # Step 1: Reflection mask R_m via threshold T=220 (slightly relaxed
    # from the methodology default of T=240 so that the demonstration
    # captures the more subtle reflections typical of CASIA-Iris-Thousand
    # acquisition; methodology values are unchanged).
    T_demo = 220
    _, R_m = cv2.threshold(img, T_demo, 255, cv2.THRESH_BINARY)

    # Step 2: Dilate with 5x5 elliptical SE
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    M_d = cv2.dilate(R_m, se, iterations=1)

    # Step 3: Telea inpainting
    I_1 = cv2.inpaint(img, M_d, 3, cv2.INPAINT_TELEA)

    # Create RGB overlays for visualization
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay = img_rgb.copy()
    overlay[M_d > 0] = [255, 0, 0]   # Red for dilated mask

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))

    # Top row
    ax = axes[0, 0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("(a) Original eye image $I$", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    ax.imshow(R_m, cmap="gray")
    ax.set_title(r"(b) Reflection mask $R_m$ (Eq. 3.1, demo $T = 220$)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.98,
            f"Pixels: {(R_m > 0).sum()}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=2))

    ax = axes[0, 2]
    ax.imshow(M_d, cmap="gray")
    ax.set_title(r"(c) Dilated mask $M_d$ (SE $5 \times 5$ ellipse)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.98,
            f"Pixels: {(M_d > 0).sum()}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.6, pad=2))

    # Bottom row
    ax = axes[1, 0]
    ax.imshow(overlay)
    ax.set_title(r"(d) Mask overlay on $I$ (red = $M_d$)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]
    ax.imshow(I_1, cmap="gray", vmin=0, vmax=255)
    ax.set_title(r"(e) After Telea inpainting $I_1$ (Eq. 3.2)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 2]
    diff = cv2.absdiff(img.astype(np.int16), I_1.astype(np.int16)).astype(np.uint8)
    im = ax.imshow(diff, cmap="hot")
    ax.set_title(r"(f) Pixel change $|I - I_1|$", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="Intensity")

    plt.suptitle("Telea Fast-Marching Inpainting — Specular Reflection Removal (Stage 1)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()

    out_path = OUT / "figure_3_10.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
