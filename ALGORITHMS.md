---
title: "Pipeline Algorithms — Chapter 3 Methodology"
author: "Hind"
subtitle: "Formal specifications with inputs, outputs, and worked examples"
---

# Pipeline Algorithms

This document gives a formal specification of every algorithm in the
iris-recognition pipeline of Chapter 3. For each algorithm we
provide:

1. **Purpose** — one-sentence description.
2. **Inputs** — typed signature with shapes and valid ranges.
3. **Outputs** — typed signature with shapes.
4. **Pseudocode** — step-by-step procedure.
5. **Worked example** — concrete numeric trace on a CASIA sample.
6. **Reference** — the equation and section numbers in Chapter 3,
   and the Python module that implements it.

All worked examples use the right-eye image **`S5000R03.jpg`** of
Subject 000 of CASIA-Iris-Thousand, resized to `320 × 240`.

---

# Stage 1 — Iris Segmentation

Chapter 3, §3.3.

The segmentation module runs four sub-stages in sequence on every
input image. Each sub-stage consumes the output of the previous one
and adds a geometric annotation (reflection mask, pupil circle,
limbus circle, eyelid lines).

---

## Algorithm 1.1 — Specular Reflection Removal

**Purpose.** Detect bright NIR reflections (camera glare) and
inpaint them so they do not corrupt the downstream edge detectors.

### Inputs

| Name | Type | Shape / range | Description |
|---|---|---|---|
| `I` | `uint8` grayscale array | `(H, W)` with `H = 240, W = 320` | Raw CASIA image resized to `TARGET_HW` |
| `T` | `int` | `[0, 255]` — default `240` | Reflection intensity threshold (`cfg.SPECULAR_THRESHOLD`) |
| `SE` | tuple | `(5, 5)` | Elliptical structuring-element size (`cfg.SPECULAR_DILATE_SE`) |
| `r_inpaint` | `int` | `3` | Telea inpaint radius (`cfg.TELEA_RADIUS`) |

### Outputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `I_1` | `uint8` grayscale | `(H, W)` | Inpainted image, same intensity range as `I` |
| `M_d` | `uint8` binary | `(H, W)` | Dilated reflection mask (255 at inpainted pixels, 0 elsewhere) |

### Pseudocode

```
Algorithm 1.1  Specular reflection removal
--------------------------------------------
Input  : I, T, SE, r_inpaint
Output : I_1, M_d

1.  R_m <- Threshold(I, T)                     // Eq. 3.1
2.  SE_ellipse <- EllipticalStructuringElement(SE)
3.  M_d <- Dilate(R_m, SE_ellipse)
4.  if sum(M_d) == 0 then
5.     I_1 <- I                               // no reflections found
6.  else
7.     I_1 <- TeleaInpaint(I, M_d, r_inpaint)  // Eq. 3.2
8.  end if
9.  return I_1, M_d
```

### Worked example (S5000R03.jpg)

| Step | Observable |
|---|---|
| Input `I` | min = 10, max = 253, mean = 92.3 |
| Threshold at `T = 240` | 118 pixels exceed → `R_m` has 118 white pixels |
| Dilate with 5×5 ellipse | `M_d` grows to 384 pixels (covers halos) |
| Telea inpaint (r=3) | `I_1` = `I` elsewhere; replaced patches match surrounding texture |
| Output max pixel-wise change | ≤ 45 intensity units in the inpainted neighbourhoods |

### Reference

Chapter 3 §3.3.1, Eq. 3.1 (`R_m = I ≥ T`), Eq. 3.2 (Telea inpaint).
Implemented in `stage_1_segmentation/s1_specular_removal.py`.

---

## Algorithm 1.2 — Pupil Detection

**Purpose.** Locate the pupil as the darkest near-circular blob
inside the iris using the Circular Hough Transform.

### Inputs

| Name | Type | Shape / range | Description |
|---|---|---|---|
| `I_1` | `uint8` grayscale | `(H, W)` | Inpainted image from Algorithm 1.1 |
| `(k, σ)` | `((int,int), float)` | `((7,7), 1.5)` | Gaussian blur kernel and σ |
| `r_min, r_max` | `int, int` | `(20, 70)` | Pupil radius search window |
| `θ_canny, θ_acc` | `int, int` | `(50, 18)` | Canny high threshold / Hough accumulator threshold |

### Outputs

| Name | Type | Range | Description |
|---|---|---|---|
| `(c_x, c_y, r_p)` | `(int, int, int)` | `0 ≤ c_x < W, 0 ≤ c_y < H, r_min ≤ r_p ≤ r_max` | Pupil centre coordinates and radius in pixels |

### Pseudocode

```
Algorithm 1.2  Pupil detection
---------------------------------
Input  : I_1, k, σ, r_min, r_max, θ_canny, θ_acc
Output : (c_x, c_y, r_p)

1.  I_b <- GaussianBlur(I_1, k, σ)
2.  C <- HoughCircles(I_b, dp=1, minDist=200,
                      param1=θ_canny, param2=θ_acc,
                      minRadius=r_min, maxRadius=r_max)
3.  if C is empty then
4.     return CentroidFallback(I_1, dark_threshold=70)
5.  end if
6.  // Pick candidate whose centre is darkest in I_b
7.  best_score <- +∞
8.  for each (c_x, c_y, r) in C do
9.      s <- I_b[c_y, c_x]
10.     if s < best_score then
11.         best_score <- s
12.         best <- (c_x, c_y, r)
13.     end if
14. end for
15. return best
```

### Worked example (S5000R03.jpg)

| Step | Observable |
|---|---|
| Blur `I_1` with 7×7 σ=1.5 | smooth version suppresses eyelash noise |
| Hough radius window [20, 70] | returns 3 candidate circles |
| Candidate 1 | `(130, 125, r=45)` centre intensity = 8 (very dark) |
| Candidate 2 | `(132, 127, r=38)` centre intensity = 11 |
| Candidate 3 | `(85,  180, r=52)` centre intensity = 134 (bright, false) |
| Winner (lowest centre intensity) | `(c_x=130, c_y=125, r_p=45)` |

### Reference

Chapter 3 §3.3.2, Eq. 3.3 (`arg min` over Hough candidates by
centre darkness). Implemented in
`stage_1_segmentation/s2_pupil_detection.py`.

---

## Algorithm 1.3 — Limbus (Outer Iris Boundary) Detection

**Purpose.** Locate the outer iris boundary using AHE
contrast-enhancement followed by Hough-based circle fitting,
anchored on the pupil centre from Algorithm 1.2.

### Inputs

| Name | Type | Shape / range | Description |
|---|---|---|---|
| `I_1` | `uint8` grayscale | `(H, W)` | Inpainted image |
| `(c_x, c_y, r_p)` | `(int, int, int)` | pupil geometry | From Algorithm 1.2 |
| `clip_limit, tile_grid` | `float, (int,int)` | `(2.0, (8,8))` | CLAHE parameters |
| `α_min, α_max` | `float, float` | `(1.7, 3.5)` | Limbus radius search bounds (multiples of `r_p`) |

### Outputs

| Name | Type | Range | Description |
|---|---|---|---|
| `(c_x, c_y, r_l)` | `(int, int, int)` | centre snapped to pupil centre; `r_l > r_p + 10` | Limbus geometry |

### Pseudocode

```
Algorithm 1.3  Limbus detection
----------------------------------
Input  : I_1, (c_x, c_y, r_p), clip_limit, tile_grid, α_min, α_max
Output : (c_x, c_y, r_l)

1.  I_eq <- CLAHE(I_1, clip_limit, tile_grid)       // Fig. 3.9
2.  I_b  <- GaussianBlur(I_eq, (7,7), 1.5)
3.  C <- HoughCircles(I_b, dp=1, minDist=200,
                      param1=50, param2=15,
                      minRadius=round(α_min * r_p),
                      maxRadius=round(α_max * r_p))
4.  if C is empty then
5.     return (c_x, c_y, round(2.5 * r_p))         // empirical prior
6.  end if
7.  // Pick candidate whose centre is closest to the pupil centre
8.  best_dist <- +∞
9.  for each (c_x', c_y', r_l) in C do
10.     d <- (c_x' - c_x)^2 + (c_y' - c_y)^2
11.     if d < best_dist and r_l > r_p + 10 then
12.         best_dist <- d
13.         best_r <- r_l
14.     end if
15. end for
16. // Snap centre to pupil centre (concentric iris assumption)
17. return (c_x, c_y, best_r)
```

### Worked example (S5000R03.jpg)

| Step | Observable |
|---|---|
| CLAHE clipLimit = 2.0, 8×8 tiles | iris texture ridges lifted; eyelash contrast higher |
| Pupil radius `r_p = 45` → window | search radius ∈ [77, 158] px |
| Hough returns 2 candidate limbi | `(135, 128, r=114)` and `(155, 90, r=120)` |
| Distance from pupil centre | 3.6 px and 42.3 px |
| Winner (closest to pupil centre) | radius 114 |
| Final limbus (centre snapped) | `(c_x=130, c_y=125, r_l=114)` → ratio 2.53× pupil |

### Reference

Chapter 3 §3.3.3, Fig. 3.9 (AHE demo), Eq. 3.4 (Hough with
concentric constraint). Implemented in
`stage_1_segmentation/s3_limbus_detection.py`.

---

## Algorithm 1.4 — Eyelid Detection

**Purpose.** Locate the upper and lower eyelid lines so the final
polar template can exclude occluded iris rows.

### Inputs

| Name | Type | Shape / range | Description |
|---|---|---|---|
| `I_1` | `uint8` grayscale | `(H, W)` | Inpainted image |
| `(c_x, c_y, r_p)` | pupil geometry | from Algorithm 1.2 | |
| `(_, _, r_l)` | limbus geometry | from Algorithm 1.3 | |
| `κ, n_iter, Δt` | `(float, int, float)` | `(30, 50, 0.14)` | Perona-Malik parameters |
| `band` | `(float, float)` | `(60°, 120°)` | Angular band for horizontal edges |

### Outputs

| Name | Type | Range | Description |
|---|---|---|---|
| `y_upper` | `int` | `0 ≤ y_upper < c_y − r_p` | Row index of upper eyelid line |
| `y_lower` | `int` | `c_y + r_p < y_lower ≤ H − 1` | Row index of lower eyelid line |

### Pseudocode

```
Algorithm 1.4  Eyelid detection
----------------------------------
Input  : I_1, (c_x, c_y, r_p), (c_x, c_y, r_l), κ, n_iter, Δt, band
Output : y_upper, y_lower

1.  I_diff <- PeronaMalik(I_1, κ, n_iter, Δt)       // Eq. 3.5
2.  // Horizontal-gradient magnitude via Sobel-y
3.  S_y <- |Sobel(I_diff, dy=1, ksize=3)|
4.  // Search bands above and below the pupil
5.  top_band_rows   <- rows in [max(c_y - r_l - 30, 0),  c_y - r_p]
6.  bot_band_rows   <- rows in [c_y + r_p,                min(c_y + r_l + 30, H)]
7.  cols            <- columns in [max(c_x - r_l, 0),    min(c_x + r_l, W)]
8.  row_energy_top  <- row-sum of S_y over (top_band_rows, cols)
9.  row_energy_bot  <- row-sum of S_y over (bot_band_rows, cols)
10. y_upper <- argmax(row_energy_top) + top_band_offset
11. y_lower <- argmax(row_energy_bot) + bot_band_offset
12. return y_upper, y_lower
```

### Worked example (S5000R03.jpg)

| Step | Observable |
|---|---|
| Perona-Malik (κ=30, 50 iter.) | eyelash texture smoothed; eyelid edge preserved |
| Sobel-y magnitude | strong rows at `y = 22` (upper) and `y = 215` (lower) |
| Top band `[0, 80]` | row energy peaks at `y = 22` |
| Bottom band `[170, 239]` | row energy peaks at `y = 215` |
| **Final** | `(y_upper, y_lower) = (22, 215)` → visible iris band = **193 rows** |

### Reference

Chapter 3 §3.3.4, Eq. 3.5 (Perona-Malik diffusion), Fig. 3.11
(demo). Implemented in `stage_1_segmentation/s4_eyelid_detection.py`.

---

# Stage 2 — Daugman Rubber-Sheet Normalisation

Chapter 3, §3.4.

---

## Algorithm 2 — Polar Unwrap

**Purpose.** Map the iris annulus `r_p ≤ r ≤ r_l` to a fixed
rectangular template of size `64 × 256` so that iris size and
pupil-dilation differences are removed before feature extraction.

### Inputs

| Name | Type | Shape / range | Description |
|---|---|---|---|
| `I_1` | `uint8` grayscale | `(H, W)` | Inpainted image |
| `(c_x, c_y, r_p, r_l)` | geometry | from Algorithms 1.2 + 1.3 | |
| `H_out` | `int` | `64` | Radial samples (`cfg.POLAR_H`) |
| `W_out` | `int` | `256` | Angular samples (`cfg.POLAR_W`) |

### Outputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `P` | `uint8` | `(H_out, W_out) = (64, 256)` | Daugman polar template |

### Pseudocode

```
Algorithm 2  Daugman rubber-sheet unwrap
----------------------------------------
Input  : I_1, (c_x, c_y, r_p, r_l), H_out, W_out
Output : P

1.  θ <- linspace(0, 2π, W_out, endpoint=False)
2.  ρ <- linspace(0, 1,  H_out)                     // 0 = pupil, 1 = limbus
3.  for i in [0, H_out) do
4.      for j in [0, W_out) do
5.          R      <- r_p + ρ[i] * (r_l - r_p)      // Eq. 3.7
6.          x[i,j] <- c_x + R * cos(θ[j])
7.          y[i,j] <- c_y + R * sin(θ[j])
8.      end for
9.  end for
10. P <- BilinearSample(I_1, x, y)                  // reflect at borders
11. return P
```

### Worked example (S5000R03.jpg)

| Step | Observable |
|---|---|
| Pupil / limbus | `r_p = 45, r_l = 114`, centre `(130, 125)` |
| Grid dimensions | `(ρ, θ)` = `(64, 256)` sample points |
| Sample at `(ρ, θ) = (0, 0)` | reads intensity at `(130 + 45·1, 125 + 45·0) = (175, 125)` |
| Sample at `(ρ, θ) = (1, π)` | reads intensity at `(130 + 114·(-1), 125 + 0) = (16, 125)` |
| Output shape | `(64, 256)`, dtype uint8 |
| Mean intensity | **104.2** (texture visible as horizontal striations) |

### Reference

Chapter 3 §3.4, Eq. 3.7. Implemented in
`stage_2_normalisation/daugman_unwrap.py`.

---

# Stage 3 — Feature Extraction

Chapter 3, §3.5. Two architecturally distinct networks produce
256-D embeddings; both are L2-normalised so dot products are
cosine similarities.

---

## Algorithm 3.1 — IrisDeepNet Forward Pass

**Purpose.** Extract a 256-D L2-normalised embedding using a
ResNet-18 backbone pre-trained on ImageNet.

### Inputs

| Name | Type | Shape / range | Description |
|---|---|---|---|
| `P` | `uint8` | `(64, 256)` | Polar template from Algorithm 2 |
| `θ_cnn` | `state_dict` | 129 tensors | Trained weights at `outputs/checkpoints/cnn_best.pth` |

### Outputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `e_cnn` | `float32` | `(256,)` | L2-normalised embedding, `‖e_cnn‖ = 1` |

### Pseudocode

```
Algorithm 3.1  IrisDeepNet embedding extraction
-------------------------------------------------
Input  : P, θ_cnn
Output : e_cnn

1.  // Prepare input: normalise to [0,1] and replicate to 3 channels
2.  x <- P / 255.0                                  // (64, 256) float32
3.  x <- stack([x, x, x], axis=0)                   // (3, 64, 256)
4.  x <- x.unsqueeze(0)                             // (1, 3, 64, 256)

5.  // Feature extraction
6.  z <- features(x; θ_cnn)                         // ResNet-18 trunk, (1, 512, 1, 1)
7.  z <- z.flatten()                                // (1, 512)

8.  // 256-D embedding + BatchNorm
9.  e <- bn( embedding(z) )                         // (1, 256)
10. e_cnn <- L2normalize(e, axis=1).squeeze()       // (256,), ‖e_cnn‖ = 1

11. return e_cnn
```

### Worked example (polar P from S5000R03.jpg)

| Step | Observable |
|---|---|
| Input tensor shape | `(1, 3, 64, 256)`, values in `[0, 1]` |
| After ResNet-18 trunk | shape `(1, 512, 1, 1)` |
| After embedding layer | shape `(1, 256)` |
| After L2 normalisation | `‖e_cnn‖ = 1.0000` |
| Example embedding stats | min = −0.287, max = +0.312, mean = 0.001 |
| Runtime (cuda) | ~ 4 ms per image |

### Reference

Chapter 3 §3.5.1, Fig. 3.5. Implemented in
`stage_3_feature_extraction/irisdeepnet_resnet18.py`.

---

## Algorithm 3.2 — DBN-Style CNN Forward Pass

**Purpose.** Extract a 256-D L2-normalised embedding using a
custom 4-layer CNN trained from scratch (no pre-training).

### Inputs

| Name | Type | Shape / range | Description |
|---|---|---|---|
| `P` | `uint8` | `(64, 256)` | Polar template |
| `θ_dbn` | `state_dict` | 37 tensors | Trained weights at `outputs/checkpoints/dbn_best.pth` |

### Outputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `e_dbn` | `float32` | `(256,)` | L2-normalised embedding |

### Pseudocode

```
Algorithm 3.2  DBN-style CNN embedding extraction
---------------------------------------------------
Input  : P, θ_dbn
Output : e_dbn

1.  x <- P / 255.0; x <- stack([x,x,x],0); x <- x.unsqueeze(0)

2.  // 4 stride-2 conv blocks: (3 -> 32 -> 64 -> 128 -> 256)
3.  for (c_in, c_out) in [(3,32), (32,64), (64,128), (128,256)] do
4.      x <- ReLU( BatchNorm2d( Conv2d(x, c_in, c_out, k=3, s=2, p=1) ) )
5.  end for
6.  x <- AdaptiveAvgPool2d(x, (1, 1))              // (1, 256, 1, 1)
7.  z <- x.flatten()                                // (1, 256)

8.  e <- bn( embedding(z) )                        // (1, 256)
9.  e_dbn <- L2normalize(e, axis=1).squeeze()

10. return e_dbn
```

### Worked example

| Step | Observable |
|---|---|
| Input shape | `(1, 3, 64, 256)` |
| After conv blocks | `(1, 256, 4, 16)` then AvgPool → `(1, 256, 1, 1)` |
| After embedding layer | `(1, 256)` |
| L2 norm check | `‖e_dbn‖ = 1.0000` |
| Runtime (cuda) | ~ 2 ms per image |

### Reference

Chapter 3 §3.5.2, Fig. 3.6. Implemented in
`stage_3_feature_extraction/dbn_style_cnn.py`.

---

## Algorithm 3.3 — Nearest-Neighbour Identification (Max-Cosine)

**Purpose.** For each probe embedding, compute a per-class
similarity score and rank the gallery subjects.

### Inputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `E_g` | `float32` | `(N_g, 256)` | Gallery embeddings |
| `y_g` | `int32` | `(N_g,)` | Gallery labels in `{0..C-1}` |
| `e_p` | `float32` | `(256,)` | Probe embedding |

### Outputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `s` | `float32` | `(C,)` | Per-class similarity score (max cosine) |
| `ranks` | `int32` | `(C,)` | Ranked class indices (best first) |

### Pseudocode

```
Algorithm 3.3  Max-cosine identification
------------------------------------------
Input  : E_g, y_g, e_p
Output : s, ranks

1.  // Per-sample cosine similarities (dot product since L2-normalised)
2.  cos_sims <- E_g @ e_p                           // (N_g,)

3.  // Max-cosine per class
4.  classes <- unique(y_g)
5.  s <- zeros(len(classes), dtype=float32)
6.  for i in [0, len(classes)) do
7.      s[i] <- max( cos_sims[ y_g == classes[i] ] )
8.  end for

9.  // Rank classes by descending score
10. ranks <- argsort(-s)

11. return s, ranks
```

### Worked example (query = S5000R03 polar embedding)

| Step | Observable |
|---|---|
| Gallery size | 6000 embeddings (6 per class × 1000 classes) |
| Per-sample cosines | `cos_sims.shape = (6000,)` |
| Per-class scores `s` | shape `(1000,)` |
| Top-5 by `s` | 000=0.914, 503=0.783, 274=0.721, 105=0.692, 096=0.673 |
| **True identity Subject 000 ranked** | **#1** |
| Runtime (CPU) | ~ 5 ms |

### Reference

Chapter 3 §3.5.3. Implemented in
`stage_3_feature_extraction/train.py :: nearest_neighbour_max`.

---

# Stage 4 — Rank-Level Fusion

Chapter 3, §3.6.

Each fusion rule combines the two per-class score vectors
produced by Algorithm 3.3 on CNN and DBN into one.

---

## Algorithm 4.1 — Highest Rank Fusion

**Purpose.** For every class, keep the smaller (better) rank of
the two matchers. Equivalent to taking the *maximum* cosine in the
score formulation.

### Inputs

| Name | Type | Shape |
|---|---|---|
| `s_cnn` | `float32` | `(C,)` |
| `s_dbn` | `float32` | `(C,)` |

### Outputs

| Name | Type | Shape |
|---|---|---|
| `s_HR` | `float32` | `(C,)` |

### Pseudocode

```
Algorithm 4.1  Highest Rank fusion
------------------------------------
Input  : s_cnn, s_dbn
Output : s_HR

1.  for i in [0, C) do
2.      s_HR[i] <- max( s_cnn[i], s_dbn[i] )       // Eq. 3.11
3.  end for
4.  return s_HR
```

### Worked example (Subject 000 R07)

| | Subject 000 | Subject 456 |
|---|---:|---:|
| `s_cnn` | 0.735 | 0.668 |
| `s_dbn` | 0.513 | 0.715 |
| **`s_HR`** | **0.735** | **0.715** |

Winner = Subject 000. (See Chapter 3 Table 3.5 for the fully
worked example on R07.)

### Reference

Chapter 3 §3.6.1, Eq. 3.11. Implemented in
`stage_4_rank_fusion/rule_highest_rank.py`.

---

## Algorithm 4.2 — Borda Count Fusion

**Purpose.** Average the two scores so a class needs good support
from BOTH matchers to win.

### Inputs / Outputs

Same shapes as Algorithm 4.1.

### Pseudocode

```
Algorithm 4.2  Borda Count fusion
-----------------------------------
Input  : s_cnn, s_dbn
Output : s_BC

1.  for i in [0, C) do
2.      s_BC[i] <- ( s_cnn[i] + s_dbn[i] ) / 2     // Eq. 3.12
3.  end for
4.  return s_BC
```

### Worked example (Subject 000 R07)

| | Subject 000 | Subject 456 |
|---|---:|---:|
| `s_cnn` | 0.735 | 0.668 |
| `s_dbn` | 0.513 | 0.715 |
| **`s_BC`** | **0.624** | **0.691** |

Hmm — on this probe BC would pick Subject 456, so BC is **worse**
than HR for this specific case. That mismatch is what motivates
Algorithm 4.3 below.

### Reference

Chapter 3 §3.6.2, Eq. 3.12. Implemented in
`stage_4_rank_fusion/rule_borda_count.py`.

---

## Algorithm 4.3 — Logistic Regression Fusion

**Purpose.** Learn a weighted combination of CNN and DBN scores
on a held-out half of the probe set, then apply those weights at
inference.

### Inputs

| Name | Type | Shape / range | Description |
|---|---|---|---|
| `S_cnn` | `float32` | `(N_probes, C)` | Per-probe per-class CNN scores |
| `S_dbn` | `float32` | `(N_probes, C)` | Per-probe per-class DBN scores |
| `y_true` | `int32` | `(N_probes,)` | Ground-truth class per probe |
| `train_frac` | `float` | `0.5` | Fraction of probes used to fit LR |

### Outputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `S_LR` | `float32` | `(N_probes, C)` | Fused per-class posterior scores |
| `test_idx` | `int32` | `(N_probes · (1 − train_frac),)` | Held-out probe indices for unbiased eval |

### Pseudocode

```
Algorithm 4.3  Logistic Regression fusion
------------------------------------------
Input  : S_cnn, S_dbn, y_true, train_frac
Output : S_LR, test_idx

1.  shuffled_idx <- random_permutation(N_probes, seed=42)
2.  tr_idx <- shuffled_idx[: int(train_frac * N_probes)]
3.  te_idx <- shuffled_idx[  int(train_frac * N_probes) :]

4.  // Build training set: one row per (probe, candidate class) pair
5.  X_tr, Y_tr <- empty_lists()
6.  for p in tr_idx do
7.      for c in [0, C) do
8.          X_tr.append( [S_cnn[p,c], S_dbn[p,c]] )
9.          Y_tr.append( 1 if c == y_true[p] else 0 )
10.     end for
11. end for

12. LR <- LogisticRegression(class_weight="balanced", max_iter=1000)
13. LR.fit(X_tr, Y_tr)                              // learns β_0, β_1, β_2 (Eq. 3.13)

14. // Score every (probe, class) pair with the learned LR
15. for p in [0, N_probes) do
16.     for c in [0, C) do
17.         S_LR[p,c] <- LR.predict_proba( [S_cnn[p,c], S_dbn[p,c]] )[1]
18.     end for
19. end for

20. return S_LR, te_idx
```

### Worked example (fitted on the CASIA probe set)

| Metric | Value |
|---|---|
| N_probes | 4000 |
| Training half | 2000 probes × 1000 classes = 2 000 000 LR training rows |
| Learned coefficients | `β_cnn ≈ 7.4`, `β_dbn ≈ 2.1`, `β_0 ≈ −6.3` |
| Interpretation | CNN score is weighted ~3.5× more than DBN score |
| **Rank-1 on held-out half** | **80.33 %** (vs 70.73 % CNN alone, 69.35 % Borda Count) |

### Reference

Chapter 3 §3.6.3, Eq. 3.13. Implemented in
`stage_4_rank_fusion/rule_logistic_regression.py`.

---

# Stage 5 — Statistical Evaluation

Chapter 3, §3.7. Four statistical tests are applied to the per-method
correctness vectors produced by Stages 3–4.

---

## Algorithm 5.1 — Wilson 95 % Confidence Interval on a Proportion

**Purpose.** Report a tight confidence interval on a Bernoulli
proportion (such as rank-1 accuracy) without the inaccuracies of
the classical normal approximation.

### Inputs

| Name | Type | Range | Description |
|---|---|---|---|
| `k` | `int` | `0 ≤ k ≤ n` | Number of successes |
| `n` | `int` | `> 0` | Number of trials |
| `α` | `float` | `0.05` | Significance level |

### Outputs

| Name | Type | Description |
|---|---|---|
| `(lo, hi)` | `(float, float)` | Lower and upper bounds in percent |

### Pseudocode

```
Algorithm 5.1  Wilson 95% CI
-----------------------------
Input  : k, n, α
Output : (lo, hi)

1.  z <- Φ^{-1}(1 - α/2)                            // normal quantile
2.  p̂ <- k / n
3.  denom <- 1 + z^2 / n
4.  centre <- (p̂ + z^2 / (2n)) / denom
5.  half   <- z · sqrt( p̂·(1-p̂)/n + z^2/(4n^2) ) / denom
6.  lo <- (centre - half) · 100
7.  hi <- (centre + half) · 100
8.  return (lo, hi)
```

### Worked example

| Input | `k = 3213`, `n = 4000` (LR fusion correctness) |
|---|---|
| p̂ | 80.33 % |
| z | 1.959964 |
| **95 % CI** | **[79.06 %, 81.53 %]** |

The reported rank-1 of 80.33 % lies in [79.06, 81.53] with 95 %
confidence — a width of ~2.5 percentage points.

### Reference

Chapter 3 §3.7.2. Implemented in
`stage_5_evaluation/wilson_ci.py`.

---

## Algorithm 5.2 — Stratified Bootstrap 95 % CI on AUC

**Purpose.** Compute a non-parametric confidence interval on the
AUC score by resampling genuine and impostor pairs separately.

### Inputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `gen` | `float32` | `(N_gen,)` | Genuine (same-subject) cosine scores |
| `imp` | `float32` | `(N_imp,)` | Impostor cosine scores |
| `B` | `int` | 1000 | Number of bootstrap replicates |

### Outputs

| Name | Type | Description |
|---|---|---|
| `(auc, lo, hi, σ)` | `(float, float, float, float)` | Point AUC + 95% CI + replicate std |

### Pseudocode

```
Algorithm 5.2  Bootstrap AUC CI
---------------------------------
Input  : gen, imp, B
Output : (auc, lo, hi, σ)

1.  rng <- RandomState(seed=42)
2.  aucs <- empty(B)
3.  for b in [0, B) do
4.      // Stratified resample WITH REPLACEMENT
5.      gen_b <- gen[ rng.randint(0, len(gen), len(gen)) ]
6.      imp_b <- imp[ rng.randint(0, len(imp), len(imp)) ]
7.      // Combine, label, score
8.      y_b <- concat([ ones(len(gen_b)), zeros(len(imp_b)) ])
9.      s_b <- concat([ gen_b, imp_b ])
10.     fpr, tpr <- roc_curve(y_b, s_b)
11.     aucs[b] <- auc(fpr, tpr)
12. end for

13. // Point AUC on the unshuffled data
14. y0 <- concat([ ones(len(gen)), zeros(len(imp)) ])
15. s0 <- concat([ gen, imp ])
16. fpr0, tpr0 <- roc_curve(y0, s0)
17. auc0 <- auc(fpr0, tpr0)

18. lo <- percentile(aucs, 2.5)
19. hi <- percentile(aucs, 97.5)
20. σ  <- std(aucs)
21. return (auc0, lo, hi, σ)
```

### Worked example (LR fusion)

| Input | 4000 genuine scores + 100 000 impostor scores |
|---|---|
| Point AUC | 0.9877 |
| Bootstrap replicates | B = 1000 |
| 95 % CI | [0.9843, 0.9906] |
| Std of AUC distribution | 0.0017 |

### Reference

Chapter 3 §3.7.3. Implemented in
`stage_5_evaluation/bootstrap_ci.py`.

---

## Algorithm 5.3 — McNemar Paired Exact Test

**Purpose.** Test whether two classifiers differ in accuracy on
the SAME probe set. Preferred over χ² when the discordant counts
are small.

### Inputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `c_A` | `bool` | `(n,)` | Correctness of classifier A per probe |
| `c_B` | `bool` | `(n,)` | Correctness of classifier B per probe |

### Outputs

| Name | Type | Description |
|---|---|---|
| `(b, c, p, sig)` | `(int, int, float, str)` | `b` = only B correct, `c` = only A correct, `p` = p-value, `sig` = {***, **, *, ns} |

### Pseudocode

```
Algorithm 5.3  McNemar exact paired test
------------------------------------------
Input  : c_A, c_B
Output : (b, c, p, sig)

1.  b <- count( (not c_A) and c_B )                 // only B correct
2.  c <- count(      c_A  and not c_B )             // only A correct
3.  if (b + c) == 0 then
4.     return (0, 0, 1.0, "ns")
5.  end if

6.  // Exact two-sided binomial test:
7.  // under H_0, min(b,c) ~ Binomial(b+c, 0.5)
8.  k <- min(b, c)
9.  p <- binomial_tail_two_sided(k, n=b+c, prob=0.5)

10. sig <- "***" if p < 0.001 else
           "**"  if p < 0.01  else
           "*"   if p < 0.05  else "ns"
11. return (b, c, p, sig)
```

### Worked example (Logistic Regression vs IrisDeepNet)

| Variable | Value |
|---|---|
| Only IrisDeepNet correct (c) | 211 probes |
| Only LR correct (b) | 596 probes |
| b + c | 807 |
| k = min(b, c) | 211 |
| Two-sided binomial p-value | **p ≈ 3 × 10⁻⁷⁰** |
| Significance | **\*\*\*** |

LR correctly handles 596 cases that CNN alone misses, versus only
211 the other way around — a statistically overwhelming win.

### Reference

Chapter 3 §3.7.1, Table 3.6. Implemented in
`stage_5_evaluation/mcnemar_test.py`.

---

## Algorithm 5.4 — Cohen's h Effect Size

**Purpose.** Report the *magnitude* of the gap between two
proportions, independent of sample size.

### Inputs

| Name | Type | Range | Description |
|---|---|---|---|
| `p_1` | `float` | `[0, 1]` | First proportion |
| `p_2` | `float` | `[0, 1]` | Second proportion |

### Outputs

| Name | Type | Description |
|---|---|---|
| `h` | `float` | Effect size (sign preserved: positive if `p_1 > p_2`) |
| `label` | `str` | `"trivial"` / `"small"` / `"medium"` / `"large"` |

### Pseudocode

```
Algorithm 5.4  Cohen's h
--------------------------
Input  : p_1, p_2
Output : (h, label)

1.  h <- 2 · arcsin(sqrt(p_1)) - 2 · arcsin(sqrt(p_2))
2.  if      |h| < 0.2 then label <- "trivial"
3.  else if |h| < 0.5 then label <- "small"
4.  else if |h| < 0.8 then label <- "medium"
5.  else                   label <- "large"
6.  return (h, label)
```

### Worked example (LR vs IrisDeepNet)

| Input | Value |
|---|---|
| `p_1` (LR rank-1) | 0.8033 |
| `p_2` (CNN rank-1) | 0.7073 |
| h | **+0.223** |
| Magnitude | **small** |

Though McNemar reports a tiny p-value (3 × 10⁻⁷⁰), the effect size
is only *small* in Cohen's units — a reminder that **statistical
significance ≠ practical magnitude**. The thesis reports both in
Table 3.6 so reviewers see both the significance and the gap size.

### Reference

Chapter 3 §3.7.4. Implemented in
`stage_5_evaluation/cohen_h.py`.

---

# Summary table — all algorithms

| # | Algorithm | Input | Output | Equation | Module |
|---|---|---|---|---|---|
| 1.1 | Specular removal | `I (240,320)` | `I_1, M_d (240,320)` | Eqs. 3.1-3.2 | `s1_specular_removal.py` |
| 1.2 | Pupil detection | `I_1` | `(c_x, c_y, r_p)` | Eq. 3.3 | `s2_pupil_detection.py` |
| 1.3 | Limbus detection | `I_1, pupil` | `(c_x, c_y, r_l)` | Eq. 3.4 | `s3_limbus_detection.py` |
| 1.4 | Eyelid detection | `I_1, pupil, limbus` | `(y_upper, y_lower)` | Eqs. 3.5-3.6 | `s4_eyelid_detection.py` |
| 2 | Daugman unwrap | `I_1, geometry` | `P (64,256)` | Eq. 3.7 | `daugman_unwrap.py` |
| 3.1 | IrisDeepNet forward | `P, θ_cnn` | `e_cnn (256,)` | Fig. 3.5 | `irisdeepnet_resnet18.py` |
| 3.2 | DBN-style forward | `P, θ_dbn` | `e_dbn (256,)` | Fig. 3.6 | `dbn_style_cnn.py` |
| 3.3 | Max-cosine NN | `E_g, y_g, e_p` | `s (C,)` | §3.5.3 | `train.py` |
| 4.1 | Highest Rank | `s_cnn, s_dbn` | `s_HR (C,)` | Eq. 3.11 | `rule_highest_rank.py` |
| 4.2 | Borda Count | `s_cnn, s_dbn` | `s_BC (C,)` | Eq. 3.12 | `rule_borda_count.py` |
| 4.3 | Logistic Regression | `S_cnn, S_dbn, y` | `S_LR (N,C)` | Eq. 3.13 | `rule_logistic_regression.py` |
| 5.1 | Wilson 95 % CI | `k, n` | `(lo, hi)` | §3.7.2 | `wilson_ci.py` |
| 5.2 | Bootstrap AUC CI | `gen, imp, B` | `(auc, lo, hi, σ)` | §3.7.3 | `bootstrap_ci.py` |
| 5.3 | McNemar test | `c_A, c_B` | `(b, c, p, sig)` | §3.7.1 | `mcnemar_test.py` |
| 5.4 | Cohen's h | `p_1, p_2` | `(h, label)` | §3.7.4 | `cohen_h.py` |

**Total: 15 algorithms covering the complete pipeline from raw CASIA image to statistically-certified identification decision.**
