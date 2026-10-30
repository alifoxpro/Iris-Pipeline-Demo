"""
Generate the proposed system diagram for the thesis.
Reflects the ACTUAL implemented pipeline — uses real architecture
details, real dataset, real numbers from the experiments.

Output:
  thesis_chapters/chapter_3_figures/figure_3_1.png    (300 DPI)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.lines as mlines

OUT = Path("E:/Strudin/Jamal Nasir Al-khafaji/Correct_theiss/thesis_chapters/chapter_3_figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":   10,
})

# ----- color palette -----
C_EXISTING = "#cfe2f3"   # light blue
C_MODIFIED = "#fff2cc"   # light yellow
C_NOVEL    = "#fad4d4"   # light pink
C_BORDER_NOVEL    = "#c00000"
C_BORDER_EXISTING = "#3d6da6"
C_BORDER_MODIFIED = "#bf9000"
C_PHASE_BG = "#f3f3f3"

fig, ax = plt.subplots(figsize=(11.5, 14))
ax.set_xlim(0, 12); ax.set_ylim(0, 18)
ax.axis("off")

# =========================================================================
# Helper: draw a labeled box
# =========================================================================
def box(x, y, w, h, text, color, border, lw=1.4, ls="-", fontsize=9,
        bold=False, italic=False):
    p = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=border, facecolor=color, linestyle=ls,
    )
    ax.add_patch(p)
    style = "italic" if italic else "normal"
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color="black",
            fontstyle=style, fontweight=weight,
            wrap=True)
    return p


def arrow(x1, y1, x2, y2, color="black", lw=1.5, ls="-",
          arrowsize=10, dash=False):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=f"-|>,head_width=0.18,head_length=0.30",
        linewidth=lw, color=color,
        mutation_scale=arrowsize,
        linestyle="--" if dash else ls,
    )
    ax.add_patch(a)


def phase_label(x, y, text):
    ax.text(x, y, text, ha="left", va="center",
            fontsize=10, color="#444",
            fontweight="bold", fontstyle="italic")


# =========================================================================
# Title
# =========================================================================
ax.text(6, 17.5,
        "PROPOSED MULTI-BIOMETRIC IRIS RECOGNITION SYSTEM",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color=C_BORDER_NOVEL,
        fontstyle="italic")
ax.text(6, 17.05,
        "with three explicitly marked research contributions (RC1, RC2, RC3)",
        ha="center", va="center", fontsize=9.5, color="#444",
        fontstyle="italic")

# =========================================================================
# Phase 1 — Acquisition (split into right & left iris)
# =========================================================================
phase_label(0.05, 16.0, "PHASE 1\nAcquisition")
box(3, 16.0, 3.0, 0.85,
    "Right Iris\n640×480 px · NIR\nCASIA-Iris-Thousand",
    C_EXISTING, C_BORDER_EXISTING, fontsize=8.5)
box(9, 16.0, 3.0, 0.85,
    "Left Iris\n640×480 px · NIR\nCASIA-Iris-Thousand",
    C_EXISTING, C_BORDER_EXISTING, fontsize=8.5)
ax.text(6, 16.0, "PIPELINE 1", ha="center", va="center",
        fontsize=7, color=C_BORDER_NOVEL, fontstyle="italic", fontweight="bold")
ax.text(6, 15.65, "PIPELINE 2", ha="center", va="center",
        fontsize=7, color=C_BORDER_NOVEL, fontstyle="italic", fontweight="bold")

# =========================================================================
# Phase 2 — Segmentation (RC1, novel) — both pipelines
# =========================================================================
phase_label(0.05, 14.5, "PHASE 2\nSegmentation\n[RC1]")
box(3, 14.5, 3.0, 1.10,
    "RC1: Four-Stage Segmentation\nStage 1: Specular inpainting\n"
    "Stage 2: Pupil Hough\nStage 3: Limbus Hough · Stage 4: Eyelid",
    C_NOVEL, C_BORDER_NOVEL, lw=2.2, fontsize=7.7, bold=False)
box(9, 14.5, 3.0, 1.10,
    "RC1: Four-Stage Segmentation\nStage 1: Specular inpainting\n"
    "Stage 2: Pupil Hough\nStage 3: Limbus Hough · Stage 4: Eyelid",
    C_NOVEL, C_BORDER_NOVEL, lw=2.2, fontsize=7.7, bold=False)

# Arrows from phase 1 to phase 2
arrow(3, 15.55, 3, 15.10)
arrow(9, 15.55, 9, 15.10)

# =========================================================================
# Phase 3 — Normalization
# =========================================================================
phase_label(0.05, 12.95, "PHASE 3\nNormalization")
box(3, 12.95, 3.0, 0.85,
    "Daugman Rubber-Sheet\n64 × 256 polar template\nbilinear remap",
    C_EXISTING, C_BORDER_EXISTING, fontsize=8.5)
box(9, 12.95, 3.0, 0.85,
    "Daugman Rubber-Sheet\n64 × 256 polar template\nbilinear remap",
    C_EXISTING, C_BORDER_EXISTING, fontsize=8.5)
arrow(3, 13.95, 3, 13.40)
arrow(9, 13.95, 9, 13.40)

# =========================================================================
# Phase 4 — Feature Extraction (architectural diversity)
# =========================================================================
phase_label(0.05, 11.30, "PHASE 4\nFeature Extr.")
box(3, 11.30, 3.0, 1.20,
    "IrisDeepNet (CNN)\nResNet-18 backbone\n"
    "ImageNet pre-trained\n11.36 M params · 256-D emb",
    C_MODIFIED, C_BORDER_MODIFIED, fontsize=8.2)
box(9, 11.30, 3.0, 1.20,
    "DBN-Style Network\n4-layer custom CNN\n"
    "trained from scratch\n0.71 M params · 256-D emb",
    C_MODIFIED, C_BORDER_MODIFIED, fontsize=8.2)
arrow(3, 12.50, 3, 11.95)
arrow(9, 12.50, 9, 11.95)

# =========================================================================
# Phase 5 — Embedding & cosine similarity matching
# =========================================================================
phase_label(0.05, 9.65, "PHASE 5\nMatching")
box(3, 9.65, 3.0, 0.85,
    "L2-normalised embedding\nCosine similarity\nvs 6,000-image gallery",
    C_EXISTING, C_BORDER_EXISTING, fontsize=8.5)
box(9, 9.65, 3.0, 0.85,
    "L2-normalised embedding\nCosine similarity\nvs 6,000-image gallery",
    C_EXISTING, C_BORDER_EXISTING, fontsize=8.5)
arrow(3, 10.65, 3, 10.10)
arrow(9, 10.65, 9, 10.10)

# Rank-vector labels
ax.text(3, 9.10, r"Rank Vector $\mathbf{r}_1 \in \mathbb{R}^{1000}$",
        ha="center", va="center", fontsize=9, fontstyle="italic")
ax.text(9, 9.10, r"Rank Vector $\mathbf{r}_2 \in \mathbb{R}^{1000}$",
        ha="center", va="center", fontsize=9, fontstyle="italic")

# Convergence arrows -> fusion
arrow(3, 8.85, 4.6, 7.85)
arrow(9, 8.85, 7.4, 7.85)

# =========================================================================
# Phase 6 — Rank-Level Fusion (RC2, novel)
# =========================================================================
phase_label(0.05, 7.40, "PHASE 6\nRank Fusion\n[RC2]")

# Big fusion box
box(6, 7.40, 8.0, 1.4,
    "RC2: PROPOSED RANK-LEVEL FUSION MODULE\n\n"
    r"Highest Rank: $HR(i)=\min(r_1(i),r_2(i))$" + "\n"
    r"Borda Count: $BC(i)=\sum_{k=1}^{2}r_k(i)$" + "\n"
    r"Logistic Regression: $LR(i)=\sigma(\beta_0+\sum_k\beta_k r_k(i))$",
    C_NOVEL, C_BORDER_NOVEL, lw=2.2, fontsize=8.0)

# =========================================================================
# Phase 7 — Decision
# =========================================================================
arrow(6, 6.70, 6, 6.10)
phase_label(0.05, 5.75, "PHASE 7\nDecision")
box(6, 5.75, 4.5, 0.7,
    r"User Identity:  $ID \in \{1, 2, \ldots, 1000\}$",
    "#d4edda", "#28a745", fontsize=10, bold=False)

arrow(6, 5.40, 6, 4.90)

# =========================================================================
# Phase 8 — Cross-dataset evaluation (RC3, novel — future work in this thesis)
# =========================================================================
phase_label(0.05, 4.40, "PHASE 8\nCross-Dataset\n[RC3]")
box(6, 4.40, 9.5, 1.0,
    "RC3: CROSS-DATASET GENERALISATION PROTOCOL\n"
    "Train: CASIA-Iris-Thousand\n"
    "Test (future work): IITD · MMU · BATH · IITK",
    C_NOVEL, C_BORDER_NOVEL, lw=2.2, ls="--", fontsize=8.5)

# =========================================================================
# Real measured performance summary (right side info box)
# =========================================================================
box(10.5, 3.10, 2.6, 1.55,
    "REAL RESULTS\n"
    "(this thesis, CASIA-Iris-\n"
    "Thousand, 1000 subj)\n\n"
    "Rank-1: 80.33%\n"
    "EER:    5.20%\n"
    "AUC:    0.9877\n"
    "Time:   12 ms / query",
    "#fff7e6", "#bf9000", lw=1.0, fontsize=7.8)

# =========================================================================
# Legend (bottom)
# =========================================================================
legend_x = 1.5
legend_y = 1.4

ax.add_patch(FancyBboxPatch((legend_x, legend_y - 1.0), 8.5, 1.30,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             facecolor="#fafafa", edgecolor="#888", lw=0.7))
ax.text(legend_x + 0.15, legend_y + 0.10, "Legend:",
        fontsize=10, fontweight="bold", va="center")

# Existing component
ax.add_patch(Rectangle((legend_x + 1.4, legend_y - 0.20), 0.5, 0.3,
                        facecolor=C_EXISTING, edgecolor=C_BORDER_EXISTING))
ax.text(legend_x + 2.05, legend_y - 0.05,
        "Existing component (from prior literature)",
        fontsize=8.5, va="center")

# Modified
ax.add_patch(Rectangle((legend_x + 1.4, legend_y - 0.55), 0.5, 0.3,
                        facecolor=C_MODIFIED, edgecolor=C_BORDER_MODIFIED))
ax.text(legend_x + 2.05, legend_y - 0.40,
        "Modified / adapted component",
        fontsize=8.5, va="center")

# Novel contribution
ax.add_patch(Rectangle((legend_x + 1.4, legend_y - 0.90), 0.5, 0.3,
                        facecolor=C_NOVEL, edgecolor=C_BORDER_NOVEL, lw=2.2))
ax.text(legend_x + 2.05, legend_y - 0.75,
        "Novel research contribution of this thesis (red border, RC1 / RC2 / RC3)",
        fontsize=8.5, va="center", fontweight="bold")

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "figure_3_1.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved {OUT / 'figure_3_1.png'}")
