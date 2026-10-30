"""
Fair cross-dataset comparison. Train CASIA encoders once (seed 42),
then evaluate the SAME frozen encoders on:
  (i)  MMU  (45 subjects, cross-sensor/population) -- the real target
  (ii) CASIA, averaged over 20 random 45-subject subsets -- the matched
       within-domain reference at the SAME 45-way difficulty.

The gap (ii) - (i) is the honest cross-dataset drop, free of the
45-vs-1000 gallery-size artefact.
"""
import json, time
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import roc_curve, auc as sk_auc

import config as cfg
from stage_3_feature_extraction.iris_dataset import make_gallery_probe_split
from stage_3_feature_extraction.irisdeepnet_resnet18 import IrisDeepNetResNet18
from stage_3_feature_extraction.dbn_style_cnn import DBNStyleCNN
import exp_multiseed_pairs as E
import cross_dataset_mmu as X
from stage_4_rank_fusion.rule_highest_rank import highest_rank
from stage_4_rank_fusion.rule_borda_count import borda_count
from stage_4_rank_fusion.rule_logistic_regression import logistic_regression_fusion

DEV = cfg.DEVICE
OUT = Path(r"E:\Strudin\Jamal Nasir Al-khafaji\Paper3_Complementarity_PR\experiments")
SEED = 42
N_SUBSETS = 20
SUBSET_SIZE = 45


def class_sims_subset(g_emb, g_y, p_emb, p_y, subj_set):
    """Return (class_sims, probe_labels) with labels REMAPPED to 0..n-1
    so they equal the column index (required by the LR-fusion feature
    builder, which compares the column index to the probe label)."""
    gm = np.isin(g_y, subj_set); pm = np.isin(p_y, subj_set)
    ge, gy = g_emb[gm], g_y[gm]
    pe, py = p_emb[pm], p_y[pm]
    sims = pe @ ge.T
    classes = np.unique(gy)
    remap = {c: i for i, c in enumerate(classes)}
    py_r = np.array([remap[c] for c in py])
    cs = np.full((sims.shape[0], len(classes)), -1.0)
    for i, c in enumerate(classes):
        cs[:, i] = sims[:, gy == c].max(1)
    return cs, py_r


def r1(cs, py):
    """Rank-1 with labels already equal to column indices (0..C-1)."""
    return float((np.argmax(cs, 1) == py).mean() * 100)


def fuse_r1(csA, csB, py):
    rA, rB = E.sims_to_ranks(csA), E.sims_to_ranks(csB)
    hr = r1(-highest_rank(rA, rB).astype(float), py)
    bc = r1(-borda_count(rA, rB).astype(float), py)
    lr_s, _ = logistic_regression_fusion(csA, csB, py)
    lr = r1(lr_s, py)
    return hr, bc, lr


def main():
    t0 = time.time()
    print(f"[xm] device={DEV}", flush=True)
    cache = OUT / "xm_embeddings.npz"
    if cache.exists():
        print("[xm] loading cached embeddings (skip training)", flush=True)
        z = np.load(cache)
        gR, gRy, pR, pRy = z["gR"], z["gRy"], z["pR"], z["pRy"]
        gL, gLy, pL, pLy = z["gL"], z["gLy"], z["pL"], z["pLy"]
        gRm, gRym, pRm, pRym = z["gRm"], z["gRym"], z["pRm"], z["pRym"]
        gLm, gLym, pLm, pLym = z["gLm"], z["gLym"], z["pLm"], z["pLym"]
    else:
        tmpl = np.load(cfg.NORM_NPZ)["templates"]; L = np.load(cfg.LBL_NPZ)
        subj, eyes = L["subjects"], L["eyes"]
        E.set_seed(SEED)
        (XRg, yRg), (XRp, yRp), clsR = make_gallery_probe_split(tmpl, subj, eyes, 1, seed=SEED)
        (XLg, yLg), (XLp, yLp), clsL = make_gallery_probe_split(tmpl, subj, eyes, 0, seed=SEED)
        E.set_seed(SEED)
        cnn = E.train_net(IrisDeepNetResNet18(n_classes=len(clsR)).to(DEV), XRg, yRg, XRp, yRp, "cnnR")
        E.set_seed(SEED + 1000)
        dbn = E.train_net(DBNStyleCNN(n_classes=len(clsL)).to(DEV), XLg, yLg, XLp, yLp, "dbnL")
        gR, gRy, pR, pRy = E.embed(cnn, XRg, yRg), yRg, E.embed(cnn, XRp, yRp), yRp
        gL, gLy, pL, pLy = E.embed(dbn, XLg, yLg), yLg, E.embed(dbn, XLp, yLp), yLp
        XRm, yRm, _ = X.load_mmu("right"); XLm, yLm, _ = X.load_mmu("left")
        (gRm_i, gRym), (pRm_i, pRym) = X.gallery_probe(XRm, yRm)
        (gLm_i, gLym), (pLm_i, pLym) = X.gallery_probe(XLm, yLm)
        gRm, pRm = E.embed(cnn, gRm_i, gRym), E.embed(cnn, pRm_i, pRym)
        gLm, pLm = E.embed(dbn, gLm_i, gLym), E.embed(dbn, pLm_i, pLym)
        np.savez(cache, gR=gR, gRy=gRy, pR=pR, pRy=pRy, gL=gL, gLy=gLy, pL=pL, pLy=pLy,
                 gRm=gRm, gRym=gRym, pRm=pRm, pRym=pRym, gLm=gLm, gLym=gLym, pLm=pLm, pLym=pLym)
        print("[xm] embeddings cached", flush=True)

    # (ii) CASIA matched 45-way, averaged over random subsets
    rng = np.random.RandomState(123)
    allS = np.unique(gRy)
    acc = {k: [] for k in ["CNN", "DBN", "HR", "BC", "LR"]}
    for _ in range(N_SUBSETS):
        S = rng.choice(allS, SUBSET_SIZE, replace=False)
        csR, pyR = class_sims_subset(gR, gRy, pR, pRy, S)
        csL, pyL = class_sims_subset(gL, gLy, pL, pLy, S)
        acc["CNN"].append(r1(csR, pyR))
        acc["DBN"].append(r1(csL, pyL))
        if np.array_equal(pyR, pyL):
            hr, bc, lr = fuse_r1(csR, csL, pyR)
            acc["HR"].append(hr); acc["BC"].append(bc); acc["LR"].append(lr)
    casia45 = {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items() if v}
    print("[xm] CASIA 45-way (mean+/-sd over %d subsets):" % N_SUBSETS, flush=True)
    for k, (m, s) in casia45.items():
        print(f"      {k}: {m:.2f} +/- {s:.2f}", flush=True)

    # (i) MMU 45-way (same frozen encoders) -- labels already 0..44
    def cs_full(ge, gy, pe):
        sims = pe @ ge.T; classes = np.unique(gy)
        cs = np.full((sims.shape[0], len(classes)), -1.0)
        for i, c in enumerate(classes):
            cs[:, i] = sims[:, gy == c].max(1)
        return cs
    csRm = cs_full(gRm, gRym, pRm); csLm = cs_full(gLm, gLym, pLm)
    mmu = dict(CNN=r1(csRm, pRym), DBN=r1(csLm, pLym))
    if np.array_equal(pRym, pLym):
        hr, bc, lr = fuse_r1(csRm, csLm, pRym)
        mmu.update(HR=hr, BC=bc, LR=lr)
    print("[xm] MMU 45-way:", {k: round(v, 2) for k, v in mmu.items()}, flush=True)

    drop = {k: round(casia45[k][0] - mmu[k], 2) for k in mmu if k in casia45}
    out = dict(casia_45way=casia45, mmu_45way=mmu, rank1_drop=drop)
    (OUT / "cross_matched.json").write_text(json.dumps(out, indent=2))
    print("[xm] Rank-1 drop (CASIA45 - MMU):", drop, flush=True)
    print(f"[xm] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
