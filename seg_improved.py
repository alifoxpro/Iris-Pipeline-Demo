"""
Stage 1 (improved) — rigorous iris localisation + mask-aware normalisation.

Two upgrades over the thesis classical pipeline:
  (A) Limbus radius by a Daugman-style integro-differential radial search
      over the LEFT/RIGHT iris-sclera sectors only (avoids eyelid/lash
      edges that fool the wide Hough search), replacing the 1.7-3.5*rp
      Hough window that over-estimated the limbus.
  (B) Mask-aware Daugman unwrap: a noise mask (specular + eyelash) is
      propagated into the polar domain so occluded pixels are flagged,
      and the valid-iris fraction is reported.

Reuses the thesis sub-stages s1 (specular) and s2 (pupil) unchanged.
Nothing in the thesis code is modified; outputs go to a separate dir.
"""
import numpy as np
import cv2
from pathlib import Path
import config as cfg
from stage_1_segmentation.s1_specular_removal import remove_specular
from stage_1_segmentation.s2_pupil_detection import detect_pupil
from stage_1_segmentation.s3_limbus_detection import detect_limbus as detect_limbus_hough
from stage_1_segmentation.s4_eyelid_detection import detect_eyelids
from stage_2_normalisation.daugman_unwrap import daugman_unwrap


def detect_pupil_robust(gray):
    """Reflection-robust pupil: darkest compact circular blob, else Hough.

    Finds the darkest connected region (the pupil is the darkest large
    blob), keeps it only if it is reasonably circular and dark, which
    rejects bright spectacle reflections that fool the Hough+darkest-
    centre rule on eyeglass images.
    """
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    t = max(int(np.percentile(blur, 4)), 15)
    _, dark = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1e9
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 150:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if not (cfg.PUPIL_R_MIN <= r <= cfg.PUPIL_R_MAX * 1.4):
            continue
        circ = area / (np.pi * r * r + 1e-6)          # 1 = perfect circle
        if circ < 0.55:
            continue
        m = np.zeros(gray.shape, np.uint8)
        cv2.circle(m, (int(x), int(y)), int(r), 255, -1)
        mean_i = cv2.mean(gray, m)[0]
        score = circ - mean_i / 255.0                  # circular AND dark
        if score > best_score:
            best_score, best = score, (int(x), int(y), int(r))
    if best is not None:
        return best
    return detect_pupil(gray)


# ---------------------------------------------------------------- (A)
def detect_limbus_ido(gray, cx, cy, rp,
                      r_lo_mult=1.8, r_hi_mult=2.8, sector_deg=35):
    """Integro-differential limbus radius, concentric with the pupil.

    The iris is darker than the surrounding sclera, so moving outward the
    radial intensity profile rises sharply at the iris-sclera boundary.
    We sample mean intensity on circular arcs in the two horizontal
    sectors (around 0 deg and 180 deg), where the boundary is least
    occluded by eyelids/lashes, then pick the radius of strongest
    positive gradient of a Gaussian-smoothed profile (Daugman's operator
    restricted to the visible sectors).
    """
    h, w = gray.shape
    enh = cv2.createCLAHE(clipLimit=cfg.AHE_CLIP_LIMIT,
                          tileGridSize=cfg.AHE_TILE_GRID).apply(gray)
    enh = cv2.GaussianBlur(enh, (5, 5), 1.0).astype(np.float32)

    a = np.deg2rad(np.r_[np.arange(-sector_deg, sector_deg + 1),
                         np.arange(180 - sector_deg, 180 + sector_deg + 1)])
    ca, sa = np.cos(a), np.sin(a)

    r_lo, r_hi = int(r_lo_mult * rp), int(r_hi_mult * rp)
    r_hi = min(r_hi, int(0.95 * min(cx, w - cx, cy, h - cy)) + rp)
    if r_hi <= r_lo + 4:
        return (cx, cy, int(rp * 2.2)), 0.0
    radii = np.arange(r_lo, r_hi)
    prof = np.full(len(radii), np.nan, np.float32)
    for i, r in enumerate(radii):
        px = np.round(cx + r * ca).astype(int)
        py = np.round(cy + r * sa).astype(int)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        if ok.sum() >= 0.6 * len(a):
            prof[i] = enh[py[ok], px[ok]].mean()
    # fill gaps, smooth, differentiate
    valid = ~np.isnan(prof)
    if valid.sum() < 5:
        return (cx, cy, int(rp * 2.2)), 0.0
    prof[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), prof[valid])
    k = cv2.getGaussianKernel(9, 2).ravel()
    prof_s = np.convolve(prof, k, mode="same")
    grad = np.gradient(prof_s)
    gmax = float(np.max(grad))
    # iris->sclera is the OUTERMOST strong dark->bright rise (inner peaks
    # are collarette/lash artefacts); take the largest radius whose
    # gradient still exceeds 55% of the global peak.
    strong = np.flatnonzero(grad >= 0.55 * gmax)
    idx = int(strong[-1]) if len(strong) else int(np.argmax(grad))
    rl = max(int(radii[idx]), rp + 12)
    return (cx, cy, rl), gmax


# ---------------------------------------------------------------- (B)
def unwrap_mask_aware(img, occ_mask, cx, cy, rp, rl,
                      h_out=None, w_out=None, lash_sigma=2.5):
    """Daugman unwrap returning (template, noise_mask, valid_fraction).

    occ_mask (Cartesian, uint8) flags specular + eyelid-occluded pixels;
    it is remapped into the polar domain and combined with an eyelash
    mask (pixels much darker than their column's robust mean). Valid
    fraction = share of unmasked polar pixels.
    """
    h_out = h_out or cfg.POLAR_H
    w_out = w_out or cfg.POLAR_W
    theta = np.linspace(0, 2 * np.pi, w_out, endpoint=False)
    r = np.linspace(0, 1, h_out)
    R = rp + r[:, None] * (rl - rp)
    T = np.broadcast_to(theta, (h_out, w_out))
    X = (cx + R * np.cos(T)).astype(np.float32)
    Y = (cy + R * np.sin(T)).astype(np.float32)

    templ = cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    occ_p = cv2.remap(occ_mask, X, Y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    col_med = np.median(templ, axis=0, keepdims=True)
    col_mad = np.median(np.abs(templ - col_med), axis=0, keepdims=True) + 1e-3
    lash = ((col_med - templ) > (lash_sigma * 1.4826 * col_mad)) & (templ < 90)
    noise = ((occ_p > 0) | lash).astype(np.uint8)
    valid_frac = float(1.0 - noise.mean())
    return templ, noise * 255, valid_frac


def eyelid_mask_cartesian(img, pupil, limbus):
    """Binary mask (255=occluded) above the upper and below the lower eyelid."""
    h, w = img.shape
    cx, cy, rp = pupil
    try:
        upper_y, lower_y = detect_eyelids(img, pupil, limbus)
    except Exception:
        return np.zeros(img.shape, np.uint8)
    m = np.zeros(img.shape, np.uint8)
    # trust an eyelid line only if it sits between the pupil edge and the limbus
    if 0 < upper_y < cy - rp + 2:
        m[:max(upper_y, 0), :] = 255
    if cy + rp - 2 < lower_y < h:
        m[min(lower_y, h):, :] = 255
    return m


# ---------------------------------------------------------------- driver
def segment_normalise(gray_raw):
    """Original vs improved pipeline on a raw grayscale eye image."""
    g = cv2.resize(gray_raw, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))
    I1, spec = remove_specular(g)

    # ---- ORIGINAL thesis pipeline (for comparison) ----
    p_old = detect_pupil(I1)
    l_old = detect_limbus_hough(I1, p_old)
    t_old = daugman_unwrap(I1, p_old[0], p_old[1], p_old[2], l_old[2])

    # ---- IMPROVED pipeline ----
    cx, cy, rp = detect_pupil_robust(I1)
    (cx, cy, rl_new), _ = detect_limbus_ido(I1, cx, cy, rp)
    eyel = eyelid_mask_cartesian(I1, (cx, cy, rp), (cx, cy, rl_new))
    occ = cv2.bitwise_or(spec, eyel)
    t_new, mask_new, vf = unwrap_mask_aware(I1, occ, cx, cy, rp, rl_new)

    return dict(img=g, I1=I1,
                p_old=p_old, rl_old=l_old[2],
                p_new=(cx, cy, rp), rl_new=rl_new,
                t_old=t_old, t_new=t_new, mask_new=mask_new, valid_frac=vf)


if __name__ == "__main__":
    import sys
    print("improved segmentation module loaded OK")
