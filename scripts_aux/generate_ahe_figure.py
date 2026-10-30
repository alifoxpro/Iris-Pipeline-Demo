"""
Generate an illustrative figure demonstrating Adaptive Histogram
Equalisation (AHE) for the iris outer-boundary localisation stage.

Output:
  thesis_chapters/chapter_3_figures/figure_3_9.png  (300 DPI)

The figure has 6 panels:
  (a) Original iris image
  (b) AHE-enhanced image
  (c) Histogram of original
  (d) Histogram of AHE-enhanced
  (e) Canny edges on original
  (f) Canny edges on AHE-enhanced

This shows visually why AHE is the right pre-processing step before
the Canny edge detector that drives Stage 3 of the segmentation
pipeline (Iris Outer Boundary Localisation).
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = Path("E:/Strudin/Jamal Nasir Al-khafaji/Correct_theiss/thesis_chapters/chapter_3_figures")
OUT.mkdir(parents=True, exist_ok=True)

CASIA_PATH = Path("E:/Strudin/Jamal Nasir Al-khafaji/innovatorsoasis-iris_hind/CASIA-Iris-Thousand")

# Academic style
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
    "axes.spines.top":   True,
    "axes.spines.right": True,
})


def load_sample_image():
    """Load a representative CASIA-Iris-Thousand sample with good iris visibility."""
    candidates = list(CASIA_PATH.glob("00*/L/S*L00.jpg"))[:5]
    if not candidates:
        candidates = list(CASIA_PATH.glob("**/*.jpg"))[:5]
    img = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
    # Resize to methodology resolution (320x280)
    img = cv2.resize(img, (320, 280))
    return img


def apply_ahe(img, clip_limit=2.0, tile_grid=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalisation."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)


def canny_edges(img, sigma=1.0, low=40, high=120):
    """Apply Canny edge detection."""
    blurred = cv2.GaussianBlur(img, (5, 5), sigma)
    return cv2.Canny(blurred, low, high)


def main():
    img = load_sample_image()
    img_ahe = apply_ahe(img)
    edges_orig = canny_edges(img)
    edges_ahe  = canny_edges(img_ahe)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))

    # ----- Top row: images -----
    ax = axes[0, 0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("(a) Original iris image", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    ax.imshow(img_ahe, cmap="gray", vmin=0, vmax=255)
    ax.set_title("(b) After AHE", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 2]
    # Show difference map
    diff = cv2.absdiff(img_ahe.astype(np.int16), img.astype(np.int16)).astype(np.uint8)
    im = ax.imshow(diff, cmap="hot")
    ax.set_title("(c) Absolute difference $|I_{AHE} - I|$", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="Pixel intensity")

    # ----- Bottom row: histograms and Canny -----
    ax = axes[1, 0]
    ax.hist(img.ravel(), bins=64, range=(0, 256),
            color="#1f77b4", alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.set_title("(d) Histogram — original", fontweight="bold")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 255)
    ax.grid(True, alpha=0.3, ls=":")

    ax = axes[1, 1]
    ax.hist(img_ahe.ravel(), bins=64, range=(0, 256),
            color="#c00000", alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.set_title("(e) Histogram — after AHE", fontweight="bold")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 255)
    ax.grid(True, alpha=0.3, ls=":")

    ax = axes[1, 2]
    # Show edges side-by-side
    edges_panel = np.hstack([edges_orig, edges_ahe])
    ax.imshow(edges_panel, cmap="gray")
    h, w = edges_orig.shape
    ax.axvline(w, color="yellow", lw=1.5)
    ax.text(w * 0.5, -10, "Canny on (a)", ha="center", fontsize=9, fontweight="bold")
    ax.text(w * 1.5, -10, "Canny on (b)", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("(f) Canny edges — original vs AHE", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Adaptive Histogram Equalisation (AHE) — Effect on Iris Edge Detection",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()

    out_path = OUT / "figure_3_9.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
