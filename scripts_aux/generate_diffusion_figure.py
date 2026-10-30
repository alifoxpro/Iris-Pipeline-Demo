"""
Generate Figure 3.11 — Perona–Malik Anisotropic Diffusion demonstration
for Stage 4 of the iris segmentation pipeline (eyelid detection).

The figure shows how anisotropic diffusion smooths fine eyelash
texture while preserving the strong eyelid edge, which is essential
for the modified Canny + Radon transform pipeline that follows.

Output:
  thesis_chapters/chapter_3_figures/figure_3_11.png  (300 DPI)
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


def perona_malik(img, kappa=30, n_iter=50, dt=0.14):
    """Perona-Malik anisotropic diffusion with exponential edge-stop function.
    Eq. 3.5 from Chapter 3."""
    img = img.astype(np.float64)
    for _ in range(n_iter):
        # Compute gradients in 4 directions
        gN = np.roll(img, -1, axis=0) - img
        gS = np.roll(img,  1, axis=0) - img
        gE = np.roll(img, -1, axis=1) - img
        gW = np.roll(img,  1, axis=1) - img
        # Edge-stop function c(s) = exp(-(s/kappa)^2)
        cN = np.exp(-(gN / kappa) ** 2)
        cS = np.exp(-(gS / kappa) ** 2)
        cE = np.exp(-(gE / kappa) ** 2)
        cW = np.exp(-(gW / kappa) ** 2)
        # Update
        img = img + dt * (cN * gN + cS * gS + cE * gE + cW * gW)
    return np.clip(img, 0, 255).astype(np.uint8)


def find_image_with_eyelashes():
    """Find a CASIA image with visible eyelashes."""
    candidates = list(CASIA_PATH.glob("00*/L/S*L00.jpg"))[:20]
    if not candidates:
        candidates = list(CASIA_PATH.glob("**/*.jpg"))[:20]
    img = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (320, 280))
    return img


def modified_canny_horizontal(img, sigma=1.0, low=30, high=90):
    """Canny restricted to mostly-horizontal edges."""
    blurred = cv2.GaussianBlur(img, (5, 5), sigma)
    edges = cv2.Canny(blurred, low, high)
    # Compute gradient orientation
    sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    angle = np.arctan2(sy, sx) * 180 / np.pi
    # Keep only horizontal-ish edges (±30° from horizontal)
    horizontal_mask = (np.abs(angle) > 60) & (np.abs(angle) < 120)
    return edges * horizontal_mask.astype(np.uint8)


def main():
    img = find_image_with_eyelashes()

    # Apply anisotropic diffusion
    img_diff = perona_malik(img, kappa=30, n_iter=50)

    # Apply modified Canny on both
    edges_orig = modified_canny_horizontal(img)
    edges_diff = modified_canny_horizontal(img_diff)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))

    # Top row
    ax = axes[0, 0]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title("(a) Original $I_1$", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    ax.imshow(img_diff, cmap="gray", vmin=0, vmax=255)
    ax.set_title(r"(b) After Perona–Malik diffusion ($\kappa = 30$, $50$ iter.)",
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 2]
    diff = cv2.absdiff(img.astype(np.int16), img_diff.astype(np.int16)).astype(np.uint8)
    im = ax.imshow(diff, cmap="hot")
    ax.set_title(r"(c) Change $|I_1 - I_{\text{diff}}|$", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="Intensity")

    # Bottom row
    ax = axes[1, 0]
    ax.imshow(edges_orig, cmap="gray")
    ax.set_title("(d) Modified Canny on $I_1$ (cluttered)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    n_edges_orig = (edges_orig > 0).sum()
    ax.text(0.02, 0.98, f"Edge pixels: {n_edges_orig}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.7, pad=2))

    ax = axes[1, 1]
    ax.imshow(edges_diff, cmap="gray")
    ax.set_title("(e) Modified Canny on $I_{\\text{diff}}$ (clean eyelid arcs)",
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    n_edges_diff = (edges_diff > 0).sum()
    ax.text(0.02, 0.98, f"Edge pixels: {n_edges_diff}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color="white",
            bbox=dict(facecolor="black", alpha=0.7, pad=2))

    ax = axes[1, 2]
    # Plot edge-pixel count reduction
    ax.bar(["Before\ndiffusion", "After\ndiffusion"],
           [n_edges_orig, n_edges_diff],
           color=["#888888", "#c00000"], edgecolor="black",
           linewidth=1.2, alpha=0.85)
    ax.set_ylabel("Edge-pixel count (horizontal edges)")
    ax.set_title(f"(f) Edge reduction: ${100*(1 - n_edges_diff/max(n_edges_orig,1)):.1f}\\%$",
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3, ls=":")
    for i, v in enumerate([n_edges_orig, n_edges_diff]):
        ax.text(i, v + max(n_edges_orig, n_edges_diff) * 0.02,
                f"{v:,}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Perona–Malik Anisotropic Diffusion — Eyelash Suppression (Stage 4, Eq. 3.5)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()

    out_path = OUT / "figure_3_11.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
