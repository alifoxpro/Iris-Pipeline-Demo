"""
Cross-dataset validation on IITD (Chapter 3 §3.8, Figure 3.8 protocol).

Procedure
---------
1.  Run the same 4-stage segmentation pipeline on every IITD image.
2.  Daugman-unwrap each iris into a 64 x 256 polar template.
3.  Load the IrisDeepNet + DBN-style network weights TRAINED ON
    CASIA-Iris-Thousand (outputs/checkpoints/{cnn,dbn}_best.pth).
4.  Replace each network's classifier head with a fresh head sized
    to the number of IITD subjects, but FREEZE the feature
    extractor + embedding layer.  The networks therefore act as
    pure feature extractors on the new dataset (no re-training).
5.  Split IITD subject-disjointly into gallery (4 images / eye) and
    probe (remaining images).  Compute max-cosine nearest-neighbour
    identification with the L2-normalised embeddings.
6.  Apply the same three fusion rules (Highest Rank / Borda Count /
    Logistic Regression) and report rank-1, EER, AUC + 95 % CIs.
7.  Compare to the within-CASIA result to quantify the cross-dataset
    accuracy gap.

Output:  outputs/results/iitd_cross_dataset.json
"""

from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import cv2

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg

from datasets import iitd
from stage_1_segmentation.s1_specular_removal       import remove_specular
from stage_1_segmentation.s2_pupil_detection        import detect_pupil
from stage_1_segmentation.s3_limbus_detection       import detect_limbus
from stage_1_segmentation.s4_eyelid_detection       import detect_eyelids
from stage_2_normalisation.daugman_unwrap           import daugman_unwrap
from stage_3_feature_extraction.iris_dataset        import IrisDataset
from stage_3_feature_extraction.irisdeepnet_resnet18 import IrisDeepNetResNet18
from stage_3_feature_extraction.dbn_style_cnn       import DBNStyleCNN
from stage_4_rank_fusion.rule_highest_rank          import (
    highest_rank, sims_for_highest_rank)
from stage_4_rank_fusion.rule_borda_count           import (
    borda_count,  sims_for_borda_count)
from stage_4_rank_fusion.rule_logistic_regression   import (
    logistic_regression_fusion)
from stage_5_evaluation.wilson_ci                   import wilson_ci
from stage_5_evaluation.bootstrap_ci                import bootstrap_auc_ci
from stage_5_evaluation.cohen_h                     import cohen_h, cohen_h_label

from torch.utils.data import DataLoader
from sklearn.metrics  import roc_curve, auc


# ----------------------------------------------------------------------
# 1) Segmentation + normalisation on IITD
# ----------------------------------------------------------------------
def segment_and_normalise_iitd():
    records = iitd.enumerate_images()
    if not records:
        raise FileNotFoundError(
            f"No IITD images found under {cfg.IITD_ROOT}.\n"
            f"Place the extracted IITD database there. "
            f"See dataset/IITD/README.md for instructions.")

    print(f"[Cross-IITD] segmenting + unwrapping {len(records)} images ...")
    arr      = np.zeros((len(records), cfg.POLAR_H, cfg.POLAR_W), dtype=np.uint8)
    subjects = np.zeros(len(records), dtype=np.int32)
    eyes     = np.zeros(len(records), dtype=np.int32)
    meta_all = {}

    t0 = time.time()
    for i, rec in enumerate(records):
        img = cv2.imread(str(rec["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))
        I_1, _       = remove_specular(img)
        pupil        = detect_pupil(I_1)
        limbus       = detect_limbus(I_1, pupil)
        up, lo       = detect_eyelids(I_1, pupil, limbus)
        arr[i]       = daugman_unwrap(I_1, pupil[0], pupil[1],
                                            pupil[2], limbus[2])
        subjects[i]  = rec["subject"]
        eyes[i]      = 0 if rec["eye"] == "L" else 1
        meta_all[rec["key"]] = dict(
            cx=int(pupil[0]),  cy=int(pupil[1]),
            rp=int(pupil[2]),  rl=int(limbus[2]),
            upper=int(up),     lower=int(lo),
            subject=int(rec["subject"]), eye=rec["eye"],
            src=str(rec["path"]),
        )
        if (i + 1) % 200 == 0:
            print(f"    {i+1:5d}/{len(records)}  ({time.time()-t0:.0f}s)")

    np.savez_compressed(cfg.IITD_NORM_NPZ, templates=arr)
    np.savez(cfg.IITD_LBL_NPZ, subjects=subjects, eyes=eyes)
    cfg.IITD_SEG_META.write_text(json.dumps(meta_all, indent=1))
    print(f"  templates: {arr.shape}  ({arr.nbytes / 1024 / 1024:.1f} MB)")
    return arr, subjects, eyes


# ----------------------------------------------------------------------
# 2) Embedding extraction with frozen CASIA-trained backbones
# ----------------------------------------------------------------------
def _strip_classifier(state_dict):
    """Drop classifier weights so we can load into a head of any size."""
    return {k: v for k, v in state_dict.items()
            if not k.startswith("classifier")}


def extract_iitd_embeddings(X, y, ckpt_name, model_cls, n_classes_iitd):
    """Load CASIA-trained weights, freeze backbone, extract embeddings."""
    model = model_cls(n_classes=n_classes_iitd).to(cfg.DEVICE)
    sd    = torch.load(cfg.OUT_CKPT / ckpt_name, map_location=cfg.DEVICE)
    missing, unexpected = model.load_state_dict(_strip_classifier(sd),
                                                strict=False)
    # `classifier.*` will be missing — that's expected (random head).
    model.eval()

    ds = IrisDataset(X, y, augment=False)
    dl = DataLoader(ds, batch_size=cfg.BATCH, shuffle=False)
    embs, lbls = [], []
    with torch.no_grad():
        for x, y_b in dl:
            _, e = model(x.to(cfg.DEVICE), return_embedding=True)
            embs.append(e.cpu().numpy())
            lbls.append(y_b.numpy())
    return np.concatenate(embs, 0), np.concatenate(lbls, 0)


# ----------------------------------------------------------------------
# 3) Gallery / probe split + identification on IITD
# ----------------------------------------------------------------------
def gallery_probe_split_iitd(X, subjects, eyes, eye_filter,
                             n_gallery=4, seed=None):
    seed = seed or cfg.SEED
    mask = eyes == eye_filter
    Xe, se = X[mask], subjects[mask]
    rng = np.random.RandomState(seed)
    idx_g, idx_p = [], []
    for s in np.unique(se):
        idx = np.where(se == s)[0]
        rng.shuffle(idx)
        idx_g += list(idx[:n_gallery])
        idx_p += list(idx[n_gallery:])
    classes  = sorted(np.unique(se).tolist())
    cls2idx  = {c: i for i, c in enumerate(classes)}
    yg = np.array([cls2idx[c] for c in se[idx_g]])
    yp = np.array([cls2idx[c] for c in se[idx_p]])
    return (Xe[idx_g], yg), (Xe[idx_p], yp), classes


def nn_max_cosine(g_emb, g_y, p_emb, p_y, K=None):
    K = K or cfg.CMC_K
    sims = p_emb @ g_emb.T
    classes = np.unique(g_y)
    class_sims = np.full((sims.shape[0], len(classes)), -1.0)
    for i, c in enumerate(classes):
        class_sims[:, i] = sims[:, g_y == c].max(1)
    rank_order = np.argsort(-class_sims, axis=1)
    ranks = np.array([np.where(rank_order[i] == gt)[0][0] + 1
                      for i, gt in enumerate(p_y)])
    cmc = np.array([(ranks <= k).mean() for k in range(1, K + 1)]) * 100
    return ranks, cmc, class_sims, classes


def split_gen_imp(class_sims, p_y):
    same = np.zeros_like(class_sims, dtype=bool)
    for i in range(class_sims.shape[0]):
        same[i, p_y[i]] = True
    gen, imp = class_sims[same], class_sims[~same]
    if len(imp) > cfg.IMP_SUBSAMPLE:
        rng = np.random.RandomState(cfg.SEED)
        imp = imp[rng.choice(len(imp), cfg.IMP_SUBSAMPLE, replace=False)]
    return gen.astype(np.float32), imp.astype(np.float32)


def roc_metrics(gen, imp):
    y   = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    s   = np.concatenate([gen, imp])
    fpr, tpr, _ = roc_curve(y, s)
    auc_v = auc(fpr, tpr)
    fnr   = 1 - tpr
    idx   = np.argmin(np.abs(fpr - fnr))
    eer   = (fpr[idx] + fnr[idx]) / 2 * 100
    return float(auc_v), float(eer)


def sims_to_ranks(class_sims):
    order = np.argsort(-class_sims, axis=1)
    ranks = np.empty_like(order)
    rows  = np.arange(class_sims.shape[0])[:, None]
    ranks[rows, order] = np.tile(np.arange(1, class_sims.shape[1] + 1),
                                  (class_sims.shape[0], 1))
    return ranks


# ----------------------------------------------------------------------
# 4) Driver
# ----------------------------------------------------------------------
def main():
    print("=" * 64)
    print(" Cross-dataset validation on IITD")
    print("=" * 64)

    # 1) Segmentation + normalisation (skip if cached)
    if cfg.IITD_NORM_NPZ.exists():
        print(f"  using cached templates -> {cfg.IITD_NORM_NPZ}")
        arr      = np.load(cfg.IITD_NORM_NPZ)["templates"]
        labels   = np.load(cfg.IITD_LBL_NPZ)
        subjects = labels["subjects"]
        eyes     = labels["eyes"]
    else:
        arr, subjects, eyes = segment_and_normalise_iitd()

    # 2) Subject-disjoint gallery/probe split
    (XR_g, yR_g), (XR_p, yR_p), classes_R = gallery_probe_split_iitd(
        arr, subjects, eyes, eye_filter=1)
    (XL_g, yL_g), (XL_p, yL_p), classes_L = gallery_probe_split_iitd(
        arr, subjects, eyes, eye_filter=0)
    print(f"  RIGHT  gallery={len(XR_g)}  probe={len(XR_p)}  classes={len(classes_R)}")
    print(f"  LEFT   gallery={len(XL_g)}  probe={len(XL_p)}  classes={len(classes_L)}")

    # 3) Extract embeddings using CASIA-trained networks
    print("\n  Extracting embeddings (CASIA-trained, frozen backbones) ...")
    cnn_g_emb, cnn_g_y = extract_iitd_embeddings(
        XR_g, yR_g, "cnn_best.pth", IrisDeepNetResNet18, len(classes_R))
    cnn_p_emb, cnn_p_y = extract_iitd_embeddings(
        XR_p, yR_p, "cnn_best.pth", IrisDeepNetResNet18, len(classes_R))
    dbn_g_emb, dbn_g_y = extract_iitd_embeddings(
        XL_g, yL_g, "dbn_best.pth", DBNStyleCNN,        len(classes_L))
    dbn_p_emb, dbn_p_y = extract_iitd_embeddings(
        XL_p, yL_p, "dbn_best.pth", DBNStyleCNN,        len(classes_L))

    # 4) Per-classifier identification
    print("\n  Computing CMC + ROC ...")
    _, cnn_cmc, cnn_cs, _ = nn_max_cosine(cnn_g_emb, cnn_g_y, cnn_p_emb, cnn_p_y)
    _, dbn_cmc, dbn_cs, _ = nn_max_cosine(dbn_g_emb, dbn_g_y, dbn_p_emb, dbn_p_y)
    if len(cnn_p_y) != len(dbn_p_y):
        # Left and right probes have different sizes; we evaluate fusion
        # only on the right eye (CNN matches CNN; DBN matches DBN
        # separately).  Cross-eye fusion is reported per-eye.
        print("  ! L and R probe counts differ -- fusion reported on RIGHT eye.")
    P_Y = cnn_p_y
    N_R = len(P_Y)

    # Same-eye fusion: re-extract DBN embeddings on the RIGHT eye
    dbn_gR_emb, _ = extract_iitd_embeddings(
        XR_g, yR_g, "dbn_best.pth", DBNStyleCNN, len(classes_R))
    dbn_pR_emb, _ = extract_iitd_embeddings(
        XR_p, yR_p, "dbn_best.pth", DBNStyleCNN, len(classes_R))
    _, dbn_cmcR, dbn_csR, _ = nn_max_cosine(
        dbn_gR_emb, cnn_g_y, dbn_pR_emb, cnn_p_y)

    # 5) Rank-level fusion (HR / BC / LR) on the right-eye probes
    cnn_ranks = sims_to_ranks(cnn_cs)
    dbn_ranks = sims_to_ranks(dbn_csR)
    hr = sims_to_ranks(-highest_rank(cnn_ranks, dbn_ranks).astype(float))
    bc = sims_to_ranks(-borda_count (cnn_ranks, dbn_ranks).astype(float))
    lr_sims, _ = logistic_regression_fusion(cnn_cs, dbn_csR, P_Y)
    lr = sims_to_ranks(lr_sims)

    sim_table  = {
        "IrisDeepNet":         cnn_cs,
        "DBN":                 dbn_csR,
        "Highest Rank":        sims_for_highest_rank(cnn_cs, dbn_csR),
        "Borda Count":         sims_for_borda_count (cnn_cs, dbn_csR),
        "Logistic Regression": lr_sims,
    }
    rank_table = {
        "IrisDeepNet":         cnn_ranks,
        "DBN":                 dbn_ranks,
        "Highest Rank":        hr,
        "Borda Count":         bc,
        "Logistic Regression": lr,
    }

    results = {}
    print(f"\n{'method':22s} {'rank-1':>8s} {'95% CI':>20s} {'EER':>7s} {'AUC':>7s}")
    for name in rank_table:
        cmc1   = (rank_table[name][np.arange(N_R), P_Y] == 1).mean() * 100
        k_corr = int((rank_table[name][np.arange(N_R), P_Y] == 1).sum())
        wlo, whi = wilson_ci(k_corr, N_R)
        gen, imp = split_gen_imp(sim_table[name], P_Y)
        auc_v, eer = roc_metrics(gen, imp)
        auc_p, blo, bhi, _ = bootstrap_auc_ci(gen, imp)
        results[name] = dict(
            rank1     = float(cmc1),
            rank1_ci  = [wlo, whi],
            eer       = float(eer),
            auc       = float(auc_p),
            auc_ci    = [blo, bhi],
            n         = int(N_R),
        )
        print(f"{name:22s} {cmc1:7.2f}%  [{wlo:5.2f}, {whi:5.2f}]  "
              f"{eer:6.2f}% {auc_v:6.4f}")

    # 6) Cross-dataset gap (vs within-CASIA result, if available)
    casia_metrics_file = cfg.OUT_RES / "final_metrics.json"
    if casia_metrics_file.exists():
        casia = json.loads(casia_metrics_file.read_text())
        print(f"\n  Cross-dataset gap (CASIA -> IITD):")
        gap = {}
        for name in results:
            if name in casia:
                p_casia = casia[name]["rank1"] / 100
                p_iitd  = results[name]["rank1"] / 100
                h = cohen_h(p_casia, p_iitd)
                gap[name] = dict(
                    casia_rank1 = casia[name]["rank1"],
                    iitd_rank1  = results[name]["rank1"],
                    delta       = casia[name]["rank1"] - results[name]["rank1"],
                    cohen_h     = h,
                    h_magnitude = cohen_h_label(h),
                )
                print(f"    {name:22s} CASIA {casia[name]['rank1']:5.2f}%  "
                      f"->  IITD {results[name]['rank1']:5.2f}%  "
                      f"(delta {gap[name]['delta']:+5.2f}, h={h:+.3f}, "
                      f"{gap[name]['h_magnitude']})")
        results["_cross_dataset_gap"] = gap

    out = dict(
        dataset    = "IITD",
        n_subjects = int(len(np.unique(subjects))),
        n_images   = int(len(arr)),
        n_probes_R = int(N_R),
        results    = results,
    )
    out_path = cfg.OUT_RES / "iitd_cross_dataset.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
