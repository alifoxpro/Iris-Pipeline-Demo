"""
Generate Figures 3.12 (CHT voting), 3.13 (Radon eyelid), 3.14 (Daugman)
all on real CASIA-Iris-Thousand samples.

Outputs:
  thesis_chapters/chapter_3_figures/figure_3_12.png  (CHT Hough voting)
  thesis_chapters/chapter_3_figures/figure_3_13.png  (Radon transform eyelids)
  thesis_chapters/chapter_3_figures/figure_3_14.png  (Daugman normalisation real)
"""

from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks, radon
from skimage.feature import canny

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


def load_sample():
    candidates = list(CASIA_PATH.glob("00*/L/S*L00.jpg"))[:5]
    if not candidates:
        candidates = list(CASIA_PATH.glob("**/*.jpg"))[:5]
    img = cv2.imread(str(candidates[0]), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (320, 280))


# =========================================================================
# Figure 3.12 — CHT Hough Voting
# =========================================================================
def figure_3_12():
    img = load_sample()
    blurred = cv2.GaussianBlur(img, (7, 7), 1.5)

    # Edge map for visualization
    edges = canny(blurred, sigma=2.0, low_threshold=20, high_threshold=80)

    # Hough circles over pupil radius range
    radii = np.arange(25, 71, 2)
    accum = hough_circle(edges, radii)

    # 3D accumulator: (n_radii, H, W). Sum across radii for visualization
    accum_2d = accum.max(axis=0)

    # Peak detection
    accums, cx_list, cy_list, radii_picks = hough_circle_peaks(
        accum, radii, total_num_peaks=5)

    # ----- Plot -----
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))

    ax = axes[0, 0]
    ax.imshow(img, cmap="gray")
    ax.set_title("(a) Input $I_1$ (after specular removal)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    ax.imshow(edges, cmap="gray")
    ax.set_title("(b) Canny edges (input to CHT)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.98, f"Edges: {edges.sum()}",
            transform=ax.transAxes, va="top", color="white",
            bbox=dict(facecolor="black", alpha=0.7, pad=2), fontsize=8)

    ax = axes[0, 2]
    im = ax.imshow(accum_2d, cmap="hot")
    ax.set_title(r"(c) Hough accumulator (max over $r$)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="Vote count")
    # Mark the top peak
    top_cy, top_cx = cy_list[0], cx_list[0]
    ax.plot(top_cx, top_cy, "c+", markersize=18, markeredgewidth=2)

    ax = axes[1, 0]
    # 3D-like view: radius vs vote count for the best (cy, cx)
    best_radii_votes = accum[:, top_cy, top_cx]
    ax.plot(radii, best_radii_votes, "o-", color="#c00000",
            markersize=5, linewidth=1.6)
    ax.set_xlabel("Candidate radius $r$ (pixels)")
    ax.set_ylabel("Vote count at peak centre")
    ax.set_title(f"(d) Vote vs radius at peak (cx={top_cx}, cy={top_cy})",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, ls=":")
    best_r_idx = np.argmax(best_radii_votes)
    ax.axvline(radii[best_r_idx], color="green", ls="--", lw=1.4,
               label=f"Best $r = {radii[best_r_idx]}$ px")
    ax.legend(loc="best")

    ax = axes[1, 1]
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for i, (cx_, cy_, r_) in enumerate(zip(cx_list, cy_list, radii_picks)):
        color = (0, 255, 0) if i == 0 else (255, 200, 0)
        thickness = 2 if i == 0 else 1
        cv2.circle(overlay, (int(cx_), int(cy_)), int(r_), color, thickness)
    ax.imshow(overlay)
    ax.set_title("(e) Top-5 candidate circles (green = best)",
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 2]
    ax.bar(range(1, 6), accums, color="#1f4e79",
           edgecolor="black", linewidth=1, alpha=0.85)
    ax.set_xlabel("Candidate rank")
    ax.set_ylabel("Accumulator vote (normalised)")
    ax.set_title("(f) Top-5 vote scores", fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(True, axis="y", alpha=0.3, ls=":")
    for i, v in enumerate(accums):
        ax.text(i + 1, v + accums.max() * 0.02, f"{v:.3f}",
                ha="center", fontsize=8.5, fontweight="bold")

    plt.suptitle("Circular Hough Transform (CHT) — Pupil Voting Mechanism",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / "figure_3_12.png")
    plt.close()
    print(f"Saved: figure_3_12.png ({(OUT / 'figure_3_12.png').stat().st_size/1024:.0f} KB)")


# =========================================================================
# Figure 3.13 — Radon Transform for Eyelid Line Detection
# =========================================================================
def figure_3_13():
    img = load_sample()

    # Step 1: mild edge-preserving smoothing (single bilateral pass)
    img_diff = cv2.bilateralFilter(img, 7, 50, 50)

    # Step 2: horizontal-only Canny (lower thresholds for visibility)
    blurred = cv2.GaussianBlur(img_diff, (3, 3), 1.0)
    edges = cv2.Canny(blurred, 15, 50)
    sy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    sx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    angle_deg = np.arctan2(sy, sx) * 180 / np.pi
    # Horizontal edges: gradient direction perpendicular to x-axis,
    # i.e. dominant vertical gradient (gy strong, gx small)
    horizontal_mask = (np.abs(angle_deg) > 60) & (np.abs(angle_deg) < 120)
    h_edges = (edges > 0) & horizontal_mask
    h_edges = h_edges.astype(np.uint8) * 255

    # Step 3: Radon transform near horizontal angles
    theta = np.linspace(75, 105, 31)
    sinogram = radon(h_edges.astype(float), theta=theta, circle=False)

    # Step 4: find top-2 peaks (upper and lower eyelid)
    flat_idx = np.argsort(sinogram.flatten())[-2:]
    peaks = []
    for idx in flat_idx[::-1]:
        rho_idx, theta_idx = np.unravel_index(idx, sinogram.shape)
        peaks.append((theta[theta_idx], rho_idx, sinogram[rho_idx, theta_idx]))

    # ----- Plot -----
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))

    ax = axes[0, 0]
    ax.imshow(img, cmap="gray")
    ax.set_title("(a) Original $I_1$", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 1]
    ax.imshow(img_diff, cmap="gray")
    ax.set_title("(b) After edge-preserving smoothing", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[0, 2]
    ax.imshow(h_edges, cmap="gray")
    ax.set_title("(c) Horizontal-edge Canny (input to Radon)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.02, 0.98, f"Edges: {(h_edges > 0).sum()}",
            transform=ax.transAxes, va="top", color="white",
            bbox=dict(facecolor="black", alpha=0.7, pad=2), fontsize=8)

    ax = axes[1, 0]
    im = ax.imshow(sinogram, cmap="hot", aspect="auto",
                   extent=[theta[0], theta[-1], sinogram.shape[0], 0])
    ax.set_xlabel(r"Angle $\theta$ (degrees)")
    ax.set_ylabel(r"Distance $\rho$ (pixels)")
    ax.set_title(r"(d) Radon transform $\mathcal{R}(\rho, \theta)$",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04, label="Vote sum")
    # Mark top-2 peaks
    for i, (t, r, v) in enumerate(peaks):
        ax.plot(t, r, "co", markersize=10, markeredgecolor="white",
                markeredgewidth=2)

    ax = axes[1, 1]
    # Show fitted eyelid lines on original image
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    h, w = img.shape
    cy, cx = h // 2, w // 2
    for i, (t_deg, rho_px, v) in enumerate(peaks):
        # rho here is along radon axis; recompute line endpoints
        t_rad = np.deg2rad(t_deg)
        # The radon transform parameterizes lines as rho = x*cos(t) + y*sin(t)
        # rho_idx is offset from center
        rho_offset = rho_px - sinogram.shape[0] / 2
        # Two endpoints
        if abs(np.sin(t_rad)) > 0.01:
            x_pts = np.array([0, w - 1])
            y_pts = (rho_offset + cx * np.cos(t_rad) + cy * np.sin(t_rad)
                     - x_pts * np.cos(t_rad)) / np.sin(t_rad)
            y_pts = np.clip(y_pts, 0, h - 1).astype(int)
            color = (0, 200, 255) if i == 0 else (255, 200, 0)
            cv2.line(overlay, (int(x_pts[0]), int(y_pts[0])),
                     (int(x_pts[1]), int(y_pts[1])), color, 2)
    ax.imshow(overlay)
    ax.set_title("(e) Top-2 Radon peaks → eyelid lines",
                 fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 2]
    ax.bar(range(1, len(peaks) + 1), [v for _, _, v in peaks],
           color=["#0099cc", "#e8a33d"], edgecolor="black",
           linewidth=1, alpha=0.85)
    ax.set_xlabel("Peak rank")
    ax.set_ylabel("Radon vote magnitude")
    ax.set_title("(f) Top-2 peak votes (upper & lower eyelid)",
                 fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Upper eyelid", "Lower eyelid"])
    ax.grid(True, axis="y", alpha=0.3, ls=":")
    for i, (_, _, v) in enumerate(peaks):
        ax.text(i + 1, v + max(p[2] for p in peaks) * 0.02,
                f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Radon Transform — Eyelid Line Detection (Stage 4)",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / "figure_3_13.png")
    plt.close()
    print(f"Saved: figure_3_13.png ({(OUT / 'figure_3_13.png').stat().st_size/1024:.0f} KB)")


# =========================================================================
# Figure 3.14 — Daugman Normalisation on Real CASIA Data
# =========================================================================
def figure_3_14():
    img = load_sample()
    h, w = img.shape

    # Quick Hough-based segmentation
    blurred = cv2.GaussianBlur(img, (7, 7), 1.5)
    # Pupil
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=200,
                                param1=50, param2=18, minRadius=25, maxRadius=70)
    if circles is not None:
        cx, cy, rp = circles[0, 0].astype(int)
    else:
        cx, cy, rp = w // 2, h // 2, 30
    # Limbus
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=200,
                                param1=50, param2=15,
                                minRadius=int(rp * 1.7), maxRadius=int(rp * 3.2))
    if circles is not None:
        rl = int(circles[0, 0, 2])
        rl = max(rl, rp + 25)
    else:
        rl = int(rp * 2.5)

    # Daugman rubber-sheet mapping
    H_out, W_out = 64, 256
    theta = np.linspace(0, 2 * np.pi, W_out, endpoint=False)
    r_norm = np.linspace(0, 1, H_out)
    R = rp + r_norm[:, None] * (rl - rp)
    T = np.broadcast_to(theta, (H_out, W_out))
    X = (cx + R * np.cos(T)).astype(np.float32)
    Y = (cy + R * np.sin(T)).astype(np.float32)
    polar = cv2.remap(img, X, Y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)

    # ----- Plot -----
    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    # Top-left: original eye with circles
    ax = fig.add_subplot(gs[0, 0])
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.circle(overlay, (cx, cy), rp, (0, 255, 0), 2)
    cv2.circle(overlay, (cx, cy), rl, (255, 0, 0), 2)
    # Sample radial line
    t_demo = np.pi / 3
    x1, y1 = int(cx + rp * np.cos(t_demo)), int(cy + rp * np.sin(t_demo))
    x2, y2 = int(cx + rl * np.cos(t_demo)), int(cy + rl * np.sin(t_demo))
    cv2.line(overlay, (x1, y1), (x2, y2), (255, 165, 0), 2)
    # Angular reference
    cv2.line(overlay, (cx, cy), (cx + rl, cy), (0, 255, 255), 1)
    ax.imshow(overlay)
    ax.set_title("(a) Iris ROI in Cartesian $(x, y)$", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.annotate(r"Pupil ($r_p={}$)".format(rp), xy=(cx, cy - rp), xytext=(cx - 50, cy - rp - 20),
                fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=0.8))
    ax.annotate(r"Limbus ($r_l={}$)".format(rl), xy=(cx + rl, cy),
                xytext=(cx + rl + 5, cy + 5),
                fontsize=8, color="red")

    # Top-middle: zoomed iris ring
    ax = fig.add_subplot(gs[0, 1])
    # Crop around iris
    y_lo = max(cy - rl - 10, 0); y_hi = min(cy + rl + 10, h)
    x_lo = max(cx - rl - 10, 0); x_hi = min(cx + rl + 10, w)
    crop = overlay[y_lo:y_hi, x_lo:x_hi]
    ax.imshow(crop)
    ax.set_title("(b) Zoomed iris annulus", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    # Top-right: schematic of mapping
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.85, 0.5), xytext=(0.15, 0.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
    ax.text(0.5, 0.65, r"$T : (x, y) \rightarrow (r, \theta)$",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(0.5, 0.32,
            "Daugman\nRubber-Sheet\nMapping\n(Eq. 3.3)",
            ha="center", fontsize=10, fontstyle="italic")
    ax.text(0.5, 0.05,
            f"Input: annular ROI\nOutput: ${W_out} \\times {H_out}$ polar template",
            ha="center", fontsize=8.5,
            bbox=dict(facecolor="#fafafa", edgecolor="gray", boxstyle="round,pad=0.4"))

    # Bottom row: polar template
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(polar, cmap="gray", aspect="auto",
              extent=[0, 2 * np.pi, 1, 0])
    ax.set_xlabel(r"Angular coordinate $\theta$ (radians)")
    ax.set_ylabel(r"Normalised radial $r$")
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_title(f"(c) Resulting polar template ${W_out} \\times {H_out}$ pixels — input to feature extractors",
                 fontweight="bold")
    # Annotate the angular reference
    ax.axvline(0, color="cyan", lw=0.8, ls="--", alpha=0.7)
    ax.axvline(t_demo, color="orange", lw=0.8, ls="--", alpha=0.7)
    ax.text(0.05, 0.05, "Sample radial line at\n$\\theta = \\pi/3$ shown in orange",
            transform=ax.transAxes, color="white", fontsize=8.5,
            bbox=dict(facecolor="black", alpha=0.6, pad=3))

    plt.suptitle("Daugman Rubber-Sheet Normalisation on a Real CASIA-Iris-Thousand Sample",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / "figure_3_14.png")
    plt.close()
    print(f"Saved: figure_3_14.png ({(OUT / 'figure_3_14.png').stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    figure_3_12()
    figure_3_13()
    figure_3_14()
    print("\nAll 3 figures generated.")
