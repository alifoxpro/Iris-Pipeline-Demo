"""
Streamlit demo for the Chapter 3 iris recognition pipeline.

Launch with:

    streamlit run app.py

Features:
    - Upload an iris image, or pick one from CASIA-Iris-Thousand.
    - Watch the full 4-stage segmentation pipeline run live.
    - See the 64 x 256 Daugman polar template.
    - Extract 256-D IrisDeepNet + DBN-style embeddings.
    - Rank the probe against the 1,000-subject gallery with four
      methods: IrisDeepNet, DBN, Highest Rank, Borda Count.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

sys.path.append(str(Path(__file__).resolve().parent))
import config as cfg
from datasets import casia_thousand
from stage_1_segmentation import (
    detect_eyelids, detect_limbus, detect_pupil, remove_specular,
)
from stage_2_normalisation import daugman_unwrap
from stage_3_feature_extraction import DBNStyleCNN, IrisDeepNetResNet18


# =====================================================================
# Page config
# =====================================================================
st.set_page_config(
    page_title="Iris Recognition Demo — UNITEN",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# Cached loaders — survive across Streamlit reruns
# =====================================================================
# The training script (real_experiments/03c_train_v2.py) used shorter
# layer names: feat/embed/cls/drop. Our architecture classes use the
# longer names features/embedding/classifier/dropout. We remap on load
# so the checkpoints populate the correct tensors.
_KEY_RENAMES = {
    "feat.":  "features.",
    "embed.": "embedding.",
    "cls.":   "classifier.",
    "drop.":  "dropout.",
}


def _remap_state_dict(sd):
    out = {}
    for k, v in sd.items():
        for old, new in _KEY_RENAMES.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        out[k] = v
    return out


# ---------------------------------------------------------------------
# Remote asset bootstrap — download trained weights + gallery on first
# launch (used by the Streamlit Cloud deployment where the local
# outputs/ artefacts are not bundled). Hosted as a GitHub Release so
# the git repository itself stays small.
# ---------------------------------------------------------------------
_RELEASE_BASE = (
    "https://github.com/alifoxpro/Iris-Pipeline-Demo/"
    "releases/download/v1.0-weights"
)
_REMOTE_ASSETS = {
    cfg.OUT_CKPT / "cnn_best.pth": f"{_RELEASE_BASE}/cnn_best.pth",
    cfg.OUT_CKPT / "dbn_best.pth": f"{_RELEASE_BASE}/dbn_best.pth",
    cfg.OUT_FEAT / "cnn_eval.npz": f"{_RELEASE_BASE}/cnn_eval.npz",
    cfg.OUT_FEAT / "dbn_eval.npz": f"{_RELEASE_BASE}/dbn_eval.npz",
}


@st.cache_resource(show_spinner=False)
def ensure_remote_assets() -> bool:
    """Download trained weights + gallery from the GitHub Release if
    they are not already on disk.

    Returns True if every expected file is now present, False otherwise.
    Runs once per Streamlit session (cached).
    """
    import urllib.request
    import shutil

    pending = [(p, u) for p, u in _REMOTE_ASSETS.items() if not p.exists()]
    if not pending:
        return True

    with st.status(
        f"Fetching {len(pending)} model asset(s) from GitHub Release "
        "(first launch only)...",
        expanded=True,
    ) as status:
        for local_path, url in pending:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = local_path.with_suffix(local_path.suffix + ".part")
            st.write(f"Downloading **{local_path.name}** ...")
            try:
                with urllib.request.urlopen(url, timeout=120) as resp:
                    with open(tmp, "wb") as out:
                        shutil.copyfileobj(resp, out, length=1 << 20)
                tmp.replace(local_path)
            except Exception as e:
                status.update(label=f"Download failed: {e}",
                              state="error")
                return False
        status.update(label="All model assets ready.",
                      state="complete", expanded=False)
    return True


@st.cache_resource(show_spinner="Loading IrisDeepNet + DBN-style weights...")
def load_models():
    """Load both networks with their CASIA-trained weights (keys remapped).

    Returns (None, None) if the checkpoints are not available (e.g.
    the GitHub Release download failed).
    """
    ensure_remote_assets()
    cnn_ckpt = cfg.OUT_CKPT / "cnn_best.pth"
    dbn_ckpt = cfg.OUT_CKPT / "dbn_best.pth"
    if not cnn_ckpt.exists() or not dbn_ckpt.exists():
        return None, None

    cnn = IrisDeepNetResNet18(n_classes=1000).to(cfg.DEVICE).eval()
    sd = torch.load(cnn_ckpt, map_location=cfg.DEVICE)
    missing, unexpected = cnn.load_state_dict(_remap_state_dict(sd),
                                              strict=False)
    assert not missing,    f"CNN missing: {missing[:5]}"
    assert not unexpected, f"CNN unexpected: {unexpected[:5]}"

    dbn = DBNStyleCNN(n_classes=1000).to(cfg.DEVICE).eval()
    sd = torch.load(dbn_ckpt, map_location=cfg.DEVICE)
    missing, unexpected = dbn.load_state_dict(_remap_state_dict(sd),
                                              strict=False)
    assert not missing,    f"DBN missing: {missing[:5]}"
    assert not unexpected, f"DBN unexpected: {unexpected[:5]}"

    return cnn, dbn


@st.cache_data(show_spinner="Loading 1,000-subject gallery embeddings...")
def load_gallery():
    """Load the pre-computed CASIA gallery embeddings.

    Returns (None, None, None, None) when the gallery .npz files are
    not present (and the GitHub Release download also failed).
    """
    ensure_remote_assets()
    cnn_npz = cfg.OUT_FEAT / "cnn_eval.npz"
    dbn_npz = cfg.OUT_FEAT / "dbn_eval.npz"
    if not cnn_npz.exists() or not dbn_npz.exists():
        return None, None, None, None
    cnn_e = np.load(cnn_npz, allow_pickle=True)
    dbn_e = np.load(dbn_npz, allow_pickle=True)
    return (
        np.asarray(cnn_e["g_emb"]), np.asarray(cnn_e["g_y"]),
        np.asarray(dbn_e["g_emb"]), np.asarray(dbn_e["g_y"]),
    )


@st.cache_data(show_spinner="Indexing CASIA image paths...")
def load_casia_catalog():
    """Return deterministic list of (subject, eye, path, key) records.

    Returns [] when the CASIA dataset is not mounted (cloud deployments
    do not carry it — it is licensed CRIPAC research data).
    """
    try:
        return casia_thousand.enumerate_images()
    except Exception:
        return []


# =====================================================================
# Algorithms tab — formal specs loaded from ALGORITHMS.md
# =====================================================================
def show_algorithms_for_stage(stage_label: str, keys: list[str]):
    """Render an expander with the specific algorithms this stage uses.

    `keys` is a list of algorithm numbers like ["1.1", "1.2"] matched
    against the parsed sections of ALGORITHMS.md.
    """
    _, sections = load_algorithms_doc()
    if not sections:
        return
    with st.expander(f"Algorithms used in this stage "
                     f"({len(keys)} algorithms — click to expand)"):
        for key in keys:
            # Match titles that contain "Algorithm {key} "
            matches = [t for t in sections
                       if t.startswith(f"Algorithm {key} ")
                       or t.startswith(f"Algorithm {key} ")
                       or t.startswith(f"Algorithm {key}")]
            if not matches:
                st.warning(f"Algorithm {key} not found in ALGORITHMS.md")
                continue
            title = matches[0]
            st.markdown(sections[title])
            st.markdown("---")


@st.cache_data(show_spinner="Loading algorithm reference...")
def load_algorithms_doc():
    """Parse ALGORITHMS.md into {algorithm_title: markdown_body}."""
    import re
    path = Path(__file__).resolve().parent / "ALGORITHMS.md"
    if not path.exists():
        return None, {}
    text = path.read_text(encoding="utf-8")

    # Strip YAML frontmatter so it does not render as text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end > 0:
            text = text[end + 5:].lstrip()

    # Split by H2 headings ("## Algorithm ...") preserving the heading
    chunks = re.split(r"(?=^##\s+Algorithm\s)", text, flags=re.MULTILINE)
    sections = {}
    for chunk in chunks:
        m = re.match(r"^##\s+(Algorithm[^\n]+)", chunk)
        if m:
            # Clean the title and stop at the next H1 so we don't bleed
            # into the next stage's intro
            title = m.group(1).strip()
            body = re.split(r"^#\s", chunk, maxsplit=1,
                            flags=re.MULTILINE)[0]
            sections[title] = body
    return text, sections


# =====================================================================
# Pipeline functions
# =====================================================================
def run_segmentation(img):
    """Run the four segmentation sub-stages and build an overlay."""
    I_1, M_d = remove_specular(img)
    pupil    = detect_pupil(I_1)
    limbus   = detect_limbus(I_1, pupil)
    upper, lower = detect_eyelids(I_1, pupil, limbus)

    cx, cy, rp = pupil
    _, _, rl   = limbus

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.circle(overlay, (cx, cy), rp, (0, 255, 0), 2)       # pupil
    cv2.circle(overlay, (cx, cy), rl, (0, 0, 255), 2)       # limbus
    cv2.line(overlay, (0, upper), (overlay.shape[1], upper),
             (255, 165, 0), 1)
    cv2.line(overlay, (0, lower), (overlay.shape[1], lower),
             (255, 165, 0), 1)

    return {
        "inpainted": I_1, "specular_mask": M_d,
        "pupil": pupil, "limbus": limbus,
        "upper": upper, "lower": lower,
        "overlay": overlay,
    }


def extract_embeddings(polar, cnn, dbn):
    """Push a polar template through both networks."""
    x = polar.astype(np.float32) / 255.0
    x = np.stack([x, x, x], 0)
    x = torch.from_numpy(x).unsqueeze(0).to(cfg.DEVICE)
    with torch.no_grad():
        _, cnn_emb = cnn(x, return_embedding=True)
        _, dbn_emb = dbn(x, return_embedding=True)
    return cnn_emb.cpu().numpy()[0], dbn_emb.cpu().numpy()[0]


def identify(cnn_emb, dbn_emb,
             cnn_gal, cnn_y, dbn_gal, dbn_y, top_k=5):
    """Compare the probe embeddings against the gallery and fuse."""
    # Per-sample cosine similarity (embeddings are L2-normalised)
    cnn_sims = cnn_gal @ cnn_emb
    dbn_sims = dbn_gal @ dbn_emb

    # Max-cosine per class (identification score per subject)
    classes = np.unique(cnn_y)
    cnn_cs = np.array([cnn_sims[cnn_y == c].max() for c in classes])
    dbn_cs = np.array([dbn_sims[dbn_y == c].max() for c in classes])

    # Fusion
    hr_cs = np.maximum(cnn_cs, dbn_cs)
    bc_cs = (cnn_cs + dbn_cs) / 2.0

    def topk(scores):
        idx = np.argsort(-scores)[:top_k]
        return [(int(classes[i]), float(scores[i])) for i in idx]

    return {
        "classes": classes,
        "cnn": {"scores": cnn_cs, "top": topk(cnn_cs)},
        "dbn": {"scores": dbn_cs, "top": topk(dbn_cs)},
        "hr":  {"scores": hr_cs,  "top": topk(hr_cs)},
        "bc":  {"scores": bc_cs,  "top": topk(bc_cs)},
    }


# =====================================================================
# Sidebar — input selection
# =====================================================================
st.sidebar.title("Iris Pipeline Demo")
st.sidebar.caption("UNITEN PhD Thesis — Chapter 3 Methodology")
st.sidebar.markdown("---")

st.sidebar.header("Input source")

# Detect whether CASIA is mounted. If not (cloud deployment), hide the
# "Pick from CASIA" option entirely and force the upload path so the
# user gets a working experience instead of an empty dropdown.
_catalog_cached = load_casia_catalog()
_casia_available = len(_catalog_cached) > 0

if _casia_available:
    mode = st.sidebar.radio(
        "Choose source:",
        ("Pick from CASIA-Iris-Thousand", "Upload my own image"),
    )
else:
    mode = "Upload my own image"
    st.sidebar.info(
        "CASIA-Iris-Thousand is a licensed CRIPAC research dataset and "
        "is not bundled with the cloud demo. Upload any iris image "
        "below to run Stages 1–2 (segmentation + Daugman unwrap) live."
    )

img = None
src_label = ""
ground_truth_subject = None

if mode == "Pick from CASIA-Iris-Thousand":
    catalog = _catalog_cached
    subject_ids = sorted({r["subject"] for r in catalog})
    subject = st.sidebar.selectbox(
        f"Subject (0 – {max(subject_ids)})",
        subject_ids,
        format_func=lambda s: f"Subject {s:03d}",
    )
    subj_recs = [r for r in catalog if r["subject"] == subject]
    rec = st.sidebar.selectbox(
        "Image",
        subj_recs,
        format_func=lambda r: f"{r['eye']} / {r['path'].name}",
    )
    img = cv2.imread(str(rec["path"]), cv2.IMREAD_GRAYSCALE)
    src_label = f"CASIA / {rec['key']}"
    ground_truth_subject = int(rec["subject"])
else:
    uploaded = st.sidebar.file_uploader(
        "Upload a grayscale iris image",
        type=["jpg", "jpeg", "png", "bmp"],
    )
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), dtype=np.uint8)
        img  = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        src_label = uploaded.name

st.sidebar.markdown("---")
run_identification = st.sidebar.checkbox(
    "Run identification against gallery",
    value=True,
    help="Compute cosine similarity against the 1,000-subject CASIA "
         "gallery with 4 methods: IrisDeepNet, DBN, Highest Rank, "
         "Borda Count.",
)
top_k = st.sidebar.slider("Top-K predictions", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.caption(f"Device: `{cfg.DEVICE}`  |  "
                   f"Input resize: `{cfg.TARGET_HW[1]} × {cfg.TARGET_HW[0]}`")


# =====================================================================
# Main panel
# =====================================================================
st.title("Iris Segmentation and Recognition Demo")
st.caption(
    "Live demo of the Chapter 3 pipeline — "
    "Specular → Pupil → Limbus → Eyelid → Daugman unwrap → "
    "IrisDeepNet + DBN embeddings → Rank-level fusion → Identity."
)

if img is None:
    st.info("Pick an image from the sidebar to start.")
    st.stop()

# Normalise input geometry
img = cv2.resize(img, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))

# -------- Tabs for each stage ----------------------------------------
(tab_overview, tab_in, tab_seg, tab_norm, tab_feat, tab_id, tab_results,
 tab_algo) = st.tabs([
    "Overview",
    "Input",
    "Stage 1 — Segmentation",
    "Stage 2 — Daugman unwrap",
    "Stage 3 — Features",
    "Stages 4-5 — Fusion + Identity",
    "Results and Tables",
    "Algorithms",
])

# ---- 0) Overview tab -------------------------------------------------
with tab_overview:
    st.subheader("Thesis pipeline — end-to-end overview")
    st.caption(
        "Complete methodology from raw CASIA-Iris-Thousand JPEG to "
        "identity decision and statistical validation. "
        "Each box corresponds to a thesis section and one or more "
        "algorithms from the Algorithms tab."
    )

    st.graphviz_chart(r"""
    digraph pipeline {
        rankdir=TB;
        nodesep=0.35;
        ranksep=0.45;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled",
              fontname="Helvetica", fontsize=11, margin="0.2,0.1"];
        edge [fontname="Helvetica", fontsize=9, color="#555555"];

        input [label=<<B>Input</B><BR/>CASIA-Iris-Thousand<BR/>1,000 subjects &times; 20 images<BR/>640 &times; 480 JPEG &rarr; 320 &times; 240>,
               fillcolor="#E3F2FD"];

        stage1 [label=<<B>Stage 1 &mdash; Segmentation</B> (&sect;3.3)<BR/>Specular &rarr; Pupil &rarr; Limbus &rarr; Eyelid<BR/>Algorithms 1.1 &ndash; 1.4>,
                fillcolor="#FFF3E0"];

        stage2 [label=<<B>Stage 2 &mdash; Daugman unwrap</B> (&sect;3.4)<BR/>Rubber-sheet polar transform<BR/>Output: 64 &times; 256 polar image<BR/>Algorithm 2>,
                fillcolor="#FFF3E0"];

        cnn [label=<<B>IrisDeepNet</B> (&sect;3.5.1)<BR/>ResNet-18 + ImageNet pretrain<BR/>256-D L2-normalised embedding<BR/>Algorithm 3.1>,
             fillcolor="#E8F5E9"];
        dbn [label=<<B>DBN-style CNN</B> (&sect;3.5.2)<BR/>4-block CNN trained from scratch<BR/>256-D L2-normalised embedding<BR/>Algorithm 3.2>,
             fillcolor="#E8F5E9"];

        scores [label=<<B>Max-cosine NN search</B><BR/>s<SUB>CNN</SUB> &isin; &real;<SUP>1,000</SUP> &nbsp;&nbsp; s<SUB>DBN</SUB> &isin; &real;<SUP>1,000</SUP><BR/>Algorithm 3.3>,
                fillcolor="#F3E5F5"];

        stage4 [label=<<B>Stage 4 &mdash; Rank-level fusion</B> (&sect;3.6)<BR/>Highest Rank (Eq. 3.11)<BR/>Borda Count (Eq. 3.12)<BR/>Logistic Regression (Eq. 3.13)<BR/>Algorithms 4.1 &ndash; 4.3>,
                fillcolor="#FCE4EC"];

        stage5 [label=<<B>Stage 5 &mdash; Identity decision</B><BR/>&#375; = argmax<SUB>c</SUB> s<SUB>fused</SUB>(c)>,
                fillcolor="#FCE4EC"];

        stats [label=<<B>Chapter 4 &mdash; Statistical validation</B><BR/>Wilson CI &middot; Bootstrap AUC &middot; McNemar &middot; Cohen's h<BR/>Algorithms 5.1 &ndash; 5.4>,
               fillcolor="#FFEBEE"];

        results [label=<<B>Headline results</B><BR/>LR rank-1 = <B>80.33%</B> &nbsp;&nbsp; EER = <B>5.20%</B><BR/>AUC = <B>0.9877</B> &nbsp;&nbsp; McNemar p &lt; 10<SUP>-70</SUP>>,
                 fillcolor="#FFECB3", penwidth=2];

        input -> stage1;
        stage1 -> stage2;
        stage2 -> cnn;
        stage2 -> dbn;
        cnn -> scores;
        dbn -> scores;
        scores -> stage4;
        stage4 -> stage5;
        stage5 -> stats;
        stats -> results;
    }
    """)

    st.markdown("#### How to read this diagram")
    st.markdown(
        "- **Orange** boxes are the four geometric segmentation stages "
        "(Chapter 3 §3.3) plus the Daugman unwrap (§3.4).\n"
        "- **Green** boxes are the two parallel deep-feature extractors "
        "(§3.5). They produce two independent 256-D embeddings per probe.\n"
        "- **Purple** box is the max-cosine nearest-neighbour search that "
        "turns each 256-D embedding into a 1,000-D score vector against "
        "the gallery (Algorithm 3.3).\n"
        "- **Pink** boxes are rank-level fusion (§3.6) and the final "
        "argmax identity decision — this is where the thesis's main "
        "contribution lives.\n"
        "- **Red** box is Chapter 4's statistical significance testing — "
        "how the thesis proves the fusion improvement is not noise.\n"
        "- **Yellow** box lists the headline numbers reported in the "
        "abstract and conclusion."
    )

    st.info(
        "Click the other tabs above to drill into each stage and run it "
        "live on a probe image. The **Algorithms** tab contains the full "
        "pseudocode, inputs, outputs, and worked examples for every "
        "algorithm referenced here."
    )

# ---- 1) Input tab ----------------------------------------------------
with tab_in:
    st.subheader("Input image")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(img, caption=src_label, use_container_width=True,
                 clamp=True)
    with c2:
        st.metric("Width",  f"{img.shape[1]} px")
        st.metric("Height", f"{img.shape[0]} px")
        st.metric("Mean intensity", f"{img.mean():.1f}")
        if ground_truth_subject is not None:
            st.metric("Ground truth",
                      f"Subject {ground_truth_subject:03d}")

# ---- 2) Segmentation tab --------------------------------------------
t_seg_start = time.time()
seg = run_segmentation(img)
t_seg = time.time() - t_seg_start

with tab_seg:
    st.subheader("Stage 1 — Four-stage iris segmentation")
    st.caption(f"Total segmentation runtime: **{t_seg*1000:.0f} ms**")

    st.markdown("##### §3.3.1 — Specular reflection removal "
                "(Telea inpainting, Eqs. 3.1-3.2)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img, caption="(a) Original $I$",
                 use_container_width=True, clamp=True)
    with c2:
        st.image(seg["specular_mask"],
                 caption=f"(b) Reflection mask $M_d$ "
                         f"(T = {cfg.SPECULAR_THRESHOLD})",
                 use_container_width=True, clamp=True)
    with c3:
        st.image(seg["inpainted"],
                 caption="(c) After Telea inpainting $I_1$",
                 use_container_width=True, clamp=True)

    st.markdown("##### §3.3.2 – §3.3.4 — Pupil, limbus, and eyelid "
                "detection")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.image(seg["overlay"],
                 caption="Pupil (green) · Limbus (blue) · "
                         "Eyelids (orange)",
                 use_container_width=True)
    with c2:
        cx, cy, rp = seg["pupil"]
        _, _,  rl  = seg["limbus"]
        up, lo     = seg["upper"], seg["lower"]
        st.markdown(f"""
**Pupil (Eq. 3.3):**
- Centre: `({cx}, {cy})`
- Radius: `{rp}` px

**Limbus (Eq. 3.4):**
- Radius: `{rl}` px
- Ratio: `{rl/rp:.2f}× r_p`

**Eyelids (Eqs. 3.5-3.6):**
- Upper row: `{up}`
- Lower row: `{lo}`
- Visible iris band: `{lo - up}` px
""")

    # Algorithm references for this stage
    show_algorithms_for_stage("Stage 1 — Segmentation",
                              ["1.1", "1.2", "1.3", "1.4"])

# ---- 3) Normalization tab -------------------------------------------
cx, cy, rp = seg["pupil"]
_, _,  rl  = seg["limbus"]
polar = daugman_unwrap(seg["inpainted"], cx, cy, rp, rl)

with tab_norm:
    st.subheader("Stage 2 — Daugman rubber-sheet normalisation "
                 "(Eq. 3.7)")
    st.markdown(f"""
The iris annulus between $r = r_p$ ({rp} px) and $r = r_l$ ({rl} px)
is mapped to a fixed rectangular template of size
`{cfg.POLAR_H} × {cfg.POLAR_W}`:

$$x(r,\\theta) = c_x + (r_p + r\\,(r_l - r_p))\\cos\\theta,\\quad
   y(r,\\theta) = c_y + (r_p + r\\,(r_l - r_p))\\sin\\theta.$$
""")
    st.image(polar,
             caption=f"Polar template — {cfg.POLAR_H} × {cfg.POLAR_W}, "
                     f"mean = {polar.mean():.1f}",
             use_container_width=True, clamp=True)

    show_algorithms_for_stage("Stage 2 — Daugman unwrap", ["2"])

# ---- 4) Features tab -------------------------------------------------
cnn, dbn = load_models()
cnn_emb = dbn_emb = None
if cnn is not None and dbn is not None:
    cnn_emb, dbn_emb = extract_embeddings(polar, cnn, dbn)

with tab_feat:
    st.subheader("Stage 3 — Feature extraction")
    st.caption("Both networks produce a 256-D L2-normalised embedding.")

    if cnn is None or dbn is None:
        st.warning(
            "**Trained checkpoints could not be loaded.** They are "
            "normally fetched automatically from the "
            "[v1.0-weights GitHub Release]"
            "(https://github.com/alifoxpro/Iris-Pipeline-Demo/"
            "releases/tag/v1.0-weights) on first launch. The download "
            "appears to have failed — check the app logs for details, "
            "or clone the repository and run `run_pipeline.py` locally."
        )
    else:
        st.caption(f"Embedding device: `{cfg.DEVICE}`")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**IrisDeepNet** (ResNet-18 backbone, "
                        "ImageNet-pretrained)")
            fig, ax = plt.subplots(figsize=(8, 1.6))
            ax.bar(range(len(cnn_emb)), cnn_emb,
                   color="#1f77b4", width=1.0)
            ax.set_xlim(0, len(cnn_emb))
            ax.set_ylim(-0.3, 0.3)
            ax.set_xticks([]); ax.axhline(0, color="k", lw=0.3)
            ax.set_title("256-D embedding", fontsize=9)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)
            st.caption(f"‖e‖ = {np.linalg.norm(cnn_emb):.3f}  "
                       f"(should ≈ 1.0)")
        with c2:
            st.markdown("**DBN-style CNN** (4-layer, from scratch)")
            fig, ax = plt.subplots(figsize=(8, 1.6))
            ax.bar(range(len(dbn_emb)), dbn_emb,
                   color="#d62728", width=1.0)
            ax.set_xlim(0, len(dbn_emb))
            ax.set_ylim(-0.3, 0.3)
            ax.set_xticks([]); ax.axhline(0, color="k", lw=0.3)
            ax.set_title("256-D embedding", fontsize=9)
            fig.tight_layout(); st.pyplot(fig, use_container_width=True)
            st.caption(f"‖e‖ = {np.linalg.norm(dbn_emb):.3f}  "
                       f"(should ≈ 1.0)")

    show_algorithms_for_stage("Stage 3 — Features",
                              ["3.1", "3.2", "3.3"])

# ---- 5) Identification tab ------------------------------------------
with tab_id:
    if not run_identification:
        st.info("Identification is disabled in the sidebar — enable the "
                "checkbox to run it.")
        st.stop()

    st.subheader("Stages 4–5 — Rank-level fusion + identity decision")

    # ---- Explanatory introduction --------------------------------
    st.markdown("#### What this stage does")
    st.markdown(
        "Stage 4 takes the per-class similarity scores produced by "
        "the two networks of Stage 3 and combines them into a single "
        "ranked list. Stage 5 then picks the top-ranked class as "
        "the predicted identity."
    )
    st.markdown(
        "Each of the two networks produces a vector of **1,000 cosine "
        "similarities** — one per CASIA subject — using *max-cosine "
        "nearest-neighbour* identification against the gallery "
        "(Algorithm 3.3). Those two 1,000-D vectors are what the "
        "fusion module receives:"
    )
    st.latex(r"s_{\text{CNN}} \in \mathbb{R}^{1{,}000}, \qquad "
             r"s_{\text{DBN}} \in \mathbb{R}^{1{,}000}.")

    st.markdown("#### Why fuse at all?")
    st.markdown(
        "Chapter 4 shows that for roughly **40 %** of probes the two "
        "networks disagree — one is correct, the other is not (see "
        "the *Disagreement profile* on the Results tab). Fusion is "
        "the mechanism that recovers the probes where **only one** "
        "classifier is correct."
    )

    st.markdown("#### Three fusion rules")
    st.markdown(
        "The thesis implements and compares three rules, each "
        "operating on the two score vectors:"
    )

    st.markdown("**1. Highest Rank (HR)** — Eq. 3.11:")
    st.latex(r"s_{\text{HR}}(c) \;=\; "
             r"\max\!\bigl(\,s_{\text{CNN}}(c),\; s_{\text{DBN}}(c)\,\bigr)")
    st.caption("Trust whichever classifier is most confident per "
               "class — one strong vote is enough.")

    st.markdown("**2. Borda Count (BC)** — Eq. 3.12:")
    st.latex(r"s_{\text{BC}}(c) \;=\; "
             r"\tfrac{1}{2}\bigl(\,s_{\text{CNN}}(c) + s_{\text{DBN}}(c)\,\bigr)")
    st.caption("Reward classes that both classifiers agree on — "
               "consensus matters.")

    st.markdown("**3. Logistic Regression (LR)** — Eq. 3.13:")
    st.latex(r"s_{\text{LR}}(c) \;=\; "
             r"\sigma\!\bigl(\,\beta_0 + \beta_1\,s_{\text{CNN}}(c) "
             r"+ \beta_2\,s_{\text{DBN}}(c)\,\bigr)")
    st.caption("Learn optimal weights from a held-out subset of "
               "probes. Chapter 4 reports LR with "
               "β₁ ≈ 7.4 (CNN) and β₂ ≈ 2.1 (DBN) — CNN is weighted "
               "about 3.5× more than DBN. LR rank-1 = **80.33 %** on "
               "the full probe set.")

    st.markdown("#### Stage 5 — Identity decision")
    st.markdown("After fusion, the predicted identity is simply the "
                "class with the highest fused score:")
    st.latex(r"\hat{y} \;=\; "
             r"\arg\max_{c \in \{0,\,1,\,\ldots,\,999\}} "
             r"s_{\text{fused}}(c)")

    st.info(
        "The live demo below runs **CNN, DBN, HR, and BC** on the "
        "probe image you selected and shows the top-K predicted "
        "subjects for each rule. **LR is omitted from the live demo** "
        "because it needs the pre-fitted β coefficients from the "
        "batch run — see the Results and Tables tab for the LR "
        "numbers."
    )
    st.markdown("---")

    cnn_gal, cnn_y, dbn_gal, dbn_y = load_gallery()

    if (cnn_emb is None or dbn_emb is None or
            cnn_gal is None or dbn_gal is None):
        st.warning(
            "**Live identification is unavailable.** The trained "
            "checkpoints and 6,000-vector CASIA gallery are normally "
            "fetched automatically from the "
            "[v1.0-weights GitHub Release]"
            "(https://github.com/alifoxpro/Iris-Pipeline-Demo/"
            "releases/tag/v1.0-weights) on first launch. The download "
            "appears to have failed — see the app logs. The precomputed "
            "batch results are still available on the **Results and "
            "Tables** tab."
        )
        st.stop()

    t0 = time.time()
    ident = identify(cnn_emb, dbn_emb, cnn_gal, cnn_y, dbn_gal, dbn_y,
                     top_k=top_k)
    t_id = time.time() - t0
    st.caption(f"Gallery search over {cnn_gal.shape[0]} embeddings: "
               f"**{t_id*1000:.0f} ms**")

    # Top-K per method side-by-side
    labels = {
        "cnn": "IrisDeepNet",
        "dbn": "DBN-style",
        "hr":  "Highest Rank",
        "bc":  "Borda Count",
    }
    cols = st.columns(4)
    for col, (key, name) in zip(cols, labels.items()):
        with col:
            st.markdown(f"**{name}**")
            for rank, (cls_id, score) in enumerate(ident[key]["top"], 1):
                prefix = "**" if rank == 1 else ""
                hit    = (ground_truth_subject is not None and
                          cls_id == ground_truth_subject)
                tag    = "  — match" if hit else ""
                st.markdown(
                    f"{prefix}#{rank}{prefix}  Subject "
                    f"`{cls_id:03d}`  —  cos = `{score:.4f}`{tag}"
                )

    # Winner callout
    winner = ident["bc"]["top"][0]
    st.markdown("---")
    if ground_truth_subject is not None:
        correct = winner[0] == ground_truth_subject
        if correct:
            st.success(
                f"**Predicted identity (Borda Count): "
                f"Subject {winner[0]:03d}**  —  "
                f"cosine = {winner[1]:.4f}  —  "
                f"**CORRECT** (ground truth was "
                f"Subject {ground_truth_subject:03d})"
            )
        else:
            st.error(
                f"**MISMATCH — predicted Subject {winner[0]:03d}** "
                f"(cosine = {winner[1]:.4f})  "
                f"vs ground truth Subject {ground_truth_subject:03d}."
            )
    else:
        st.success(
            f"**Predicted identity (Borda Count): "
            f"Subject {winner[0]:03d}**  —  cosine = {winner[1]:.4f}"
        )

    # Rank distribution plot
    st.markdown("---")
    st.markdown("##### Full cosine-similarity distribution per method")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    for ax, (key, name) in zip(axes, labels.items()):
        scores = np.sort(ident[key]["scores"])[::-1]     # desc
        ax.plot(scores, lw=1, color="steelblue")
        ax.axhline(scores[0], color="red", ls="--", lw=0.6,
                   alpha=0.6, label=f"max = {scores[0]:.3f}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Rank (0 = top-1)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Cosine similarity")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    show_algorithms_for_stage(
        "Stages 4–5 — Rank-level fusion + identity decision",
        ["4.1", "4.2", "4.3", "5.1", "5.2", "5.3", "5.4"],
    )

# =====================================================================
# Results and Tables tab — thesis metrics from outputs/results/
# =====================================================================
@st.cache_data(show_spinner="Loading thesis metrics...")
def load_thesis_metrics():
    import json
    fm = cfg.OUT_RES / "final_metrics.json"
    sg = cfg.OUT_RES / "significance.json"
    if not fm.exists() or not sg.exists():
        return None, None
    final = json.loads(fm.read_text())
    sig   = json.loads(sg.read_text())
    return final, sig


with tab_results:
    st.subheader("Thesis-reported metrics on CASIA-Iris-Thousand")
    st.caption(
        "The numbers below come from the batch evaluation "
        "(1,000 subjects, 20,000 images, 4,000 probes). They match the "
        "tables printed in Chapter 4 of the thesis."
    )

    final, sig = load_thesis_metrics()
    if final is None or sig is None:
        st.warning(
            "Thesis metrics JSON files are not present in this "
            "deployment. Run `run_pipeline.py` locally against "
            "CASIA-Iris-Thousand to generate `outputs/results/"
            "final_metrics.json` and `significance.json`."
        )
        st.stop()

    # -------- Table 1: identification + verification per method -----
    st.markdown("##### Table 1 — Identification and verification "
                "accuracy per method")
    methods_order = ["IrisDeepNet", "DBN", "Highest Rank",
                     "Borda Count", "Logistic Regression"]
    rows = []
    for m in methods_order:
        if m not in final:
            continue
        r = final[m]
        rows.append({
            "Method":   m,
            "Rank-1 (%)":  f"{r['rank1']:.2f}",
            "Rank-5 (%)":  f"{r['rank5']:.2f}",
            "Rank-20 (%)": f"{r['rank20']:.2f}",
            "EER (%)":  f"{r['eer']:.2f}",
            "AUC":      f"{r['auc']:.4f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Winner: **Logistic Regression** — rank-1 = 80.33 %, "
               "EER = 5.20 %, AUC = 0.9877.")

    # -------- Table 2: Wilson 95 % CI on rank-1 ---------------------
    st.markdown("##### Table 2 — Wilson 95 % confidence interval on "
                "rank-1 (n = 4,000 probes)")
    rows = []
    for m in methods_order:
        if m not in sig["wilson_ci"]:
            continue
        w = sig["wilson_ci"][m]
        rows.append({
            "Method":    m,
            "Rank-1 (%)": f"{w['rank1']:.2f}",
            "CI lower":  f"{w['ci95_low']:.2f}",
            "CI upper":  f"{w['ci95_high']:.2f}",
            "CI width":  f"{w['ci95_high'] - w['ci95_low']:.2f}",
            "n":         w["n"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # -------- Table 3: Bootstrap 95 % CI on AUC ---------------------
    st.markdown("##### Table 3 — Bootstrap 95 % confidence interval "
                "on AUC (B = 1,000 resamples)")
    rows = []
    for m in methods_order:
        if m not in sig["bootstrap_ci"]:
            continue
        b = sig["bootstrap_ci"][m]
        rows.append({
            "Method":   m,
            "AUC":      f"{b['auc']:.4f}",
            "CI lower": f"{b['auc_ci_low']:.4f}",
            "CI upper": f"{b['auc_ci_high']:.4f}",
            "std":      f"{b['auc_std']:.4f}",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    # -------- Table 4: McNemar paired tests -------------------------
    st.markdown("##### Table 4 — McNemar paired exact test "
                "(α = 0.05)")
    rows = []
    for test in sig["mcnemar"]:
        rows.append({
            "Method A":        test["method_A"],
            "Method B":        test["method_B"],
            "A-only correct":  test["A_only_correct"],
            "B-only correct":  test["B_only_correct"],
            "p-value":         f"{test['p_value']:.3g}",
            "Significance":    test["significance"],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Significance code: *** p < 0.001 — ** p < 0.01 — "
               "* p < 0.05 — ns = not significant.")

    # -------- Table 5: disagreement profile -------------------------
    st.markdown("##### Table 5 — Disagreement profile "
                "(CNN vs DBN on the probe set)")
    dis = final.get("disagreement", {})
    if dis:
        total = dis.get("total", 1)
        rows = [
            ("Both correct",         dis.get("both_correct", 0)),
            ("Only CNN correct",     dis.get("cnn_only", 0)),
            ("Only DBN correct",     dis.get("dbn_only", 0)),
            ("Both wrong",           dis.get("both_wrong", 0)),
            ("Total probes",         total),
        ]
        df_rows = [{
            "Outcome": name,
            "Count":   count,
            "Share (%)": f"{count/total*100:.2f}" if name != "Total probes" else "100.00",
        } for name, count in rows]
        st.dataframe(df_rows, use_container_width=True, hide_index=True)
        st.caption(
            f"The {dis.get('cnn_only',0) + dis.get('dbn_only',0)} probes "
            f"where only one classifier is correct are exactly the ones "
            f"rank-level fusion can rescue."
        )

    # -------- CMC curve plot ----------------------------------------
    st.markdown("##### CMC curves — rank 1 through rank 20")
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in methods_order:
        if m not in final or "cmc" not in final[m]:
            continue
        cmc = final[m]["cmc"]
        ax.plot(range(1, len(cmc) + 1), cmc, lw=1.4, marker="o",
                markersize=3, label=m)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Identification rate (%)")
    ax.set_xlim(1, 20)
    ax.set_ylim(50, 100)
    ax.set_xticks(range(1, 21, 2))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Cumulative Match Characteristic (CMC) curves — "
                 "CASIA-Iris-Thousand probe set",
                 fontsize=10)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

with tab_algo:
    st.subheader("Pipeline algorithms — formal specifications")
    st.caption(
        "Pseudocode, input/output tables, and worked examples for every "
        "stage of the Chapter 3 methodology. Loaded live from "
        "`ALGORITHMS.md` — edits to that file appear here on reload."
    )

    full_text, sections = load_algorithms_doc()

    if full_text is None:
        st.error("ALGORITHMS.md not found next to app.py.")
    else:
        titles = list(sections.keys())
        # Group selector by stage prefix for readability
        choice = st.selectbox(
            "Pick an algorithm to inspect:",
            options=["(Show all 15 algorithms)"] + titles,
            index=0,
        )
        st.markdown("---")
        if choice == "(Show all 15 algorithms)":
            st.markdown(full_text, unsafe_allow_html=False)
        else:
            st.markdown(sections[choice], unsafe_allow_html=False)

        # Download button so the user can grab the raw markdown
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download ALGORITHMS.md",
                data=full_text.encode("utf-8"),
                file_name="ALGORITHMS.md",
                mime="text/markdown",
                use_container_width=True,
            )
        docx_path = Path(__file__).resolve().parent / "ALGORITHMS.docx"
        if docx_path.exists():
            with c2:
                st.download_button(
                    "Download ALGORITHMS.docx (UNITEN format)",
                    data=docx_path.read_bytes(),
                    file_name="ALGORITHMS.docx",
                    mime="application/vnd.openxmlformats-officedocument."
                         "wordprocessingml.document",
                    use_container_width=True,
                )

st.markdown("---")
st.caption(
    "UNITEN PhD thesis — *Iris Segmentation and Recognition Using Deep "
    "Learning Approach* — "
    "Pipeline figures in Chapter 3; empirical validation on "
    "CASIA-Iris-Thousand (1,000 subjects, 20,000 images) in Chapter 4."
)
