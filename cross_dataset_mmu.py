"""
Cross-dataset generalisation: train on CASIA-Iris-Thousand, test on MMU
(no re-training). Frozen CASIA feature extractors are used as encoders;
identification is nearest-neighbour cosine within MMU's own gallery/probe
split. Reports Rank-1 / EER / AUC for CNN (right), DBN (left), and the
three rank-fusion rules, plus the within->cross drop and robustness score.

MMU is segmented/normalised by the SAME original pipeline that produced
the CASIA training templates, so the measured gap reflects sensor/
population shift, not a segmentation-method mismatch.
"""
import json, time
from pathlib import Path
import numpy as np, cv2, torch
from sklearn.metrics import roc_curve, auc as sk_auc

import config as cfg
from stage_1_segmentation.s1_specular_removal import remove_specular
from stage_1_segmentation.s2_pupil_detection import detect_pupil
from stage_1_segmentation.s3_limbus_detection import detect_limbus
from stage_2_normalisation.daugman_unwrap import daugman_unwrap
from stage_3_feature_extraction.iris_dataset import make_gallery_probe_split
from stage_3_feature_extraction.irisdeepnet_resnet18 import IrisDeepNetResNet18
from stage_3_feature_extraction.dbn_style_cnn import DBNStyleCNN
import exp_multiseed_pairs as E
from stage_4_rank_fusion.rule_highest_rank import highest_rank
from stage_4_rank_fusion.rule_borda_count import borda_count
from stage_4_rank_fusion.rule_logistic_regression import logistic_regression_fusion

DEV = cfg.DEVICE
OUT = Path(r"E:\Strudin\Jamal Nasir Al-khafaji\Paper3_Complementarity_PR\experiments")
MMU = Path(r"E:\Strudin\Jamal Nasir Al-khafaji\Correct_theiss\methodology_implementation\dataset\MMU-Iris-Database")
SEED = 42


def mmu_template(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = cv2.resize(g, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))
    I1, _ = remove_specular(g)
    p = detect_pupil(I1)
    l = detect_limbus(I1, p)
    t = daugman_unwrap(I1, p[0], p[1], p[2], l[2])        # 64x256 uint8
    return t.astype(np.uint8)


def load_mmu(eye):  # eye in {"left","right"}
    subs = sorted([d for d in MMU.iterdir() if d.is_dir() and d.name.isdigit()],
                  key=lambda p: int(p.name))
    X, y = [], []
    for si, s in enumerate(subs):
        imgs = sorted((s / eye).glob("*.bmp"))
        for p in imgs:
            t = mmu_template(p)
            if t is not None:
                X.append(t); y.append(si)
    return np.array(X), np.array(y), len(subs)


def gallery_probe(X, y, n_gal=3):
    gi, pi = [], []
    for s in np.unique(y):
        idx = np.where(y == s)[0]
        gi += list(idx[:n_gal]); pi += list(idx[n_gal:])
    return (X[gi], y[gi]), (X[pi], y[pi])


def embed_templates(model, X, y):
    return E.embed(model, X, y)   # reuse (resizes grayscale->3ch, L2-norm)


def class_sims(g_emb, g_y, p_emb):
    sims = p_emb @ g_emb.T
    classes = np.unique(g_y)
    cs = np.full((sims.shape[0], len(classes)), -1.0)
    for i, c in enumerate(classes):
        cs[:, i] = sims[:, g_y == c].max(1)
    return cs, classes


def eer_auc(cs, py):
    N, C = cs.shape
    gen = cs[np.arange(N), py]
    mask = np.ones_like(cs, bool); mask[np.arange(N), py] = False
    imp = cs[mask]
    scores = np.r_[gen, imp]; labels = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))] * 100
    return float(eer), float(sk_auc(fpr, tpr))


def rank1_of(cs, classes, py):
    pred = classes[np.argmax(cs, 1)]
    return float((pred == py).mean() * 100)


def main():
    log = []
    t0 = time.time()
    print(f"[xd] device={DEV}", flush=True)

    # 1) train CASIA encoders (current code), seed 42
    tmpl = np.load(cfg.NORM_NPZ)["templates"]; L = np.load(cfg.LBL_NPZ)
    subj, eyes = L["subjects"], L["eyes"]
    E.set_seed(SEED)
    (XRg, yRg), (XRp, yRp), clsR = make_gallery_probe_split(tmpl, subj, eyes, eye_filter=1, seed=SEED)
    (XLg, yLg), (XLp, yLp), clsL = make_gallery_probe_split(tmpl, subj, eyes, eye_filter=0, seed=SEED)
    E.set_seed(SEED)
    cnn = E.train_net(IrisDeepNetResNet18(n_classes=len(clsR)).to(DEV), XRg, yRg, XRp, yRp, "cnnR(CASIA)")
    E.set_seed(SEED + 1000)
    dbn = E.train_net(DBNStyleCNN(n_classes=len(clsL)).to(DEV), XLg, yLg, XLp, yLp, "dbnL(CASIA)")

    # CASIA within-domain reference (right=CNN, left=DBN)
    csR_ca, clsR_ca = class_sims(E.embed(cnn, XRg, yRg), yRg, E.embed(cnn, XRp, yRp))
    casia_cnn = rank1_of(csR_ca, clsR_ca, yRp)
    print(f"[xd] CASIA within-domain CNN rank-1 = {casia_cnn:.2f}", flush=True)

    # 2) MMU templates (original pipeline), per eye
    print("[xd] segmenting MMU ...", flush=True)
    XR, yR, nR = load_mmu("right")
    XL, yL, nL = load_mmu("left")
    print(f"[xd] MMU right={len(XR)} imgs/{nR} subj  left={len(XL)} imgs/{nL} subj", flush=True)
    (gR, gRy), (pR, pRy) = gallery_probe(XR, yR)
    (gL, gLy), (pL, pLy) = gallery_probe(XL, yL)

    # 3) extract embeddings with FROZEN CASIA encoders, NN-identify within MMU
    csR, clsR2 = class_sims(embed_templates(cnn, gR, gRy), gRy, embed_templates(cnn, pR, pRy))
    csL, clsL2 = class_sims(embed_templates(dbn, gL, gLy), gLy, embed_templates(dbn, pL, pLy))

    # align right/left probes by subject for fusion (both 2 probes/subject, subject order)
    # trim to subjects present in both and equal counts
    rR1, rL1 = rank1_of(csR, clsR2, pRy), rank1_of(csL, clsL2, pLy)
    eerR, aucR = eer_auc(csR, pRy); eerL, aucL = eer_auc(csL, pLy)
    print(f"[xd] MMU  CNN(right) rank-1={rR1:.2f} EER={eerR:.2f} AUC={aucR:.3f}", flush=True)
    print(f"[xd] MMU  DBN(left)  rank-1={rL1:.2f} EER={eerL:.2f} AUC={aucL:.3f}", flush=True)

    # fusion: pair probes by subject (requires same probe count/subject both eyes)
    res = dict(casia_within_cnn=casia_cnn,
               mmu_cnn=dict(rank1=rR1, eer=eerR, auc=aucR),
               mmu_dbn=dict(rank1=rL1, eer=eerL, auc=aucL))
    if np.array_equal(pRy, pLy):
        rks_R = E.sims_to_ranks(csR); rks_L = E.sims_to_ranks(csL)
        hr = rank1_of(-highest_rank(rks_R, rks_L).astype(float), clsR2, pRy)
        bc = rank1_of(-borda_count(rks_R, rks_L).astype(float), clsR2, pRy)
        lr_sims, _ = logistic_regression_fusion(csR, csL, pRy)
        lr = rank1_of(lr_sims, clsR2, pRy)
        print(f"[xd] MMU  fusion  HR={hr:.2f}  BC={bc:.2f}  LR={lr:.2f}", flush=True)
        res["mmu_fusion"] = dict(HR=hr, BC=bc, LR=lr)
        res["drop_cnn_rank1"] = casia_cnn - rR1
        res["robustness_cnn"] = 1 - (eerR - eerL) / max(eerL, 1e-6)  # placeholder
    else:
        print("[xd] right/left probe labels not aligned; fusion skipped", flush=True)

    (OUT / "cross_dataset_mmu.json").write_text(json.dumps(res, indent=2))
    print(f"[xd] done in {time.time()-t0:.0f}s -> cross_dataset_mmu.json", flush=True)


if __name__ == "__main__":
    main()
