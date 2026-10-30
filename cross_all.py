"""
Cross-dataset generalisation on all VALID test databases:
  CASIA-Iris-Interval, CASIA-Iris-Lamp (same institution, different
  sensor/conditions), and MMU (independent lab). CASIA-Iris-Syn is
  excluded (synthetic). IITD is excluded here (flat/partial structure
  on disk -- needs re-extraction into per-subject folders).

Train CASIA-Thousand encoders once (seed 42, cached weights), then for
each test DB: segment+normalise (original pipeline), extract embeddings
with the FROZEN encoders, nearest-neighbour cosine identification within
the test DB, and compare against CASIA at MATCHED N-way difficulty
(20 random N-subject subsets). Reports Rank-1/EER/AUC for CNN, DBN,
HR, BC, LR and the within->cross drop.
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
DS = Path(r"E:\Strudin\Jamal Nasir Al-khafaji\Correct_theiss\methodology_implementation\dataset")
SEED = 42
N_SUBSETS = 20


def make_template(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = cv2.resize(g, (cfg.TARGET_HW[1], cfg.TARGET_HW[0]))
    I1, _ = remove_specular(g)
    p = detect_pupil(I1)
    l = detect_limbus(I1, p)
    return daugman_unwrap(I1, p[0], p[1], p[2], l[2]).astype(np.uint8)


def load_db_bysubject(root, eyemap, ext):
    """eyemap: {'R': right_subfolder, 'L': left_subfolder}. Returns
    {subject_id: {'R': [templates], 'L': [templates]}} over numeric
    subject folders that exist on disk."""
    by = {}
    subs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()],
                  key=lambda p: int(p.name))
    for s in subs:
        sid = int(s.name)
        rec = {}
        for eye, sub in eyemap.items():
            folder = s / sub
            imgs = []
            if folder.exists():
                for p in sorted(folder.glob(ext)):
                    t = make_template(p)
                    if t is not None:
                        imgs.append(t)
            rec[eye] = imgs
        by[sid] = rec
    return by


def _split(imgs):
    ng = max(1, len(imgs) // 2)
    return imgs[:ng], imgs[ng:]


def build_eval(by_subj):
    """From a by-subject dict, build right/left gallery+probe arrays and a
    subject-aligned probe pairing. Only subjects with >=2 images in BOTH
    eyes are used (so each yields gallery+probe for both eyes and can be
    fused)."""
    valid = [s for s, d in by_subj.items() if len(d["R"]) >= 2 and len(d["L"]) >= 2]
    valid.sort()
    gR, gRy, gL, gLy = [], [], [], []
    pairsR, pairsL, pairY = [], [], []   # subject-aligned probe pairs
    for s in valid:
        Rg, Rp = _split(by_subj[s]["R"])
        Lg, Lp = _split(by_subj[s]["L"])
        gR += Rg; gRy += [s] * len(Rg)
        gL += Lg; gLy += [s] * len(Lg)
        k = min(len(Rp), len(Lp))
        pairsR += Rp[:k]; pairsL += Lp[:k]; pairY += [s] * k
    return (np.array(gR), np.array(gRy), np.array(gL), np.array(gLy),
            np.array(pairsR), np.array(pairsL), np.array(pairY), len(valid))


def class_sims(ge, gy, pe):
    sims = pe @ ge.T
    classes = np.unique(gy)
    cs = np.full((sims.shape[0], len(classes)), -1.0)
    for i, c in enumerate(classes):
        cs[:, i] = sims[:, gy == c].max(1)
    return cs, classes


def r1(cs, py):
    return float((np.argmax(cs, 1) == py).mean() * 100)


def remap(py, classes):
    m = {c: i for i, c in enumerate(classes)}
    return np.array([m[c] for c in py])


def eer_auc(cs, py):
    N = cs.shape[0]
    gen = cs[np.arange(N), py]
    mask = np.ones_like(cs, bool); mask[np.arange(N), py] = False
    imp = cs[mask]
    s = np.r_[gen, imp]; lab = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    fpr, tpr, _ = roc_curve(lab, s); fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fpr - fnr))] * 100), float(sk_auc(fpr, tpr))


def fuse(csR, csL, py):
    rA, rB = E.sims_to_ranks(csR), E.sims_to_ranks(csL)
    hr = r1(-highest_rank(rA, rB).astype(float), py)
    bc = r1(-borda_count(rA, rB).astype(float), py)
    lr_s, _ = logistic_regression_fusion(csR, csL, py)
    lr = r1(lr_s, py)
    return hr, bc, lr


def eval_db(name, cnn, dbn, by_subj):
    gR, gRy, gL, gLy, pR, pL, pY, nvalid = build_eval(by_subj)
    # embeddings
    geR = E.embed(cnn, gR, gRy); peR = E.embed(cnn, pR, pY)
    geL = E.embed(dbn, gL, gLy); peL = E.embed(dbn, pL, pY)
    csR, clsR = class_sims(geR, gRy, peR)
    csL, clsL = class_sims(geL, gLy, peL)
    pyc = remap(pY, clsR)                 # classes identical (same valid subjects)
    out = dict(subjects=int(nvalid), probes=int(len(pY)),
               CNN=r1(csR, pyc), DBN=r1(csL, pyc))
    out["CNN_eer"], out["CNN_auc"] = eer_auc(csR, pyc)
    out["DBN_eer"], out["DBN_auc"] = eer_auc(csL, pyc)
    hr, bc, lr = fuse(csR, csL, pyc)
    out.update(HR=hr, BC=bc, LR=lr)
    print(f"[{name}] valid-subj={nvalid} probes={len(pY)}  CNN={out['CNN']:.2f} "
          f"DBN={out['DBN']:.2f} HR={hr:.2f} BC={bc:.2f} LR={lr:.2f}", flush=True)
    return out


def casia_matched(gR, gRy, pR, pRy, gL, gLy, pL, pLy, n_subj):
    """CASIA at matched N-way difficulty, with the SAME subject-aligned
    probe pairing as eval_db (right probe paired with left probe of the
    same subject)."""
    rng = np.random.RandomState(7)
    allS = np.unique(gRy)
    acc = {k: [] for k in ["CNN", "DBN", "HR", "BC", "LR"]}
    for _ in range(N_SUBSETS):
        S = rng.choice(allS, min(n_subj, len(allS)), replace=False)
        geR, gyR = gR[np.isin(gRy, S)], gRy[np.isin(gRy, S)]
        geL, gyL = gL[np.isin(gLy, S)], gLy[np.isin(gLy, S)]
        iR, iL, pyp = [], [], []
        for s in S:
            ri = np.where(pRy == s)[0]; li = np.where(pLy == s)[0]
            k = min(len(ri), len(li))
            iR += list(ri[:k]); iL += list(li[:k]); pyp += [s] * k
        iR, iL = np.array(iR), np.array(iL)
        csR, clsR = class_sims(geR, gyR, pR[iR])
        csL, clsL = class_sims(geL, gyL, pL[iL])
        pyc = remap(np.array(pyp), clsR)
        acc["CNN"].append(r1(csR, pyc)); acc["DBN"].append(r1(csL, pyc))
        hr, bc, lr = fuse(csR, csL, pyc)
        acc["HR"].append(hr); acc["BC"].append(bc); acc["LR"].append(lr)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items() if v}


def main():
    t0 = time.time(); print(f"[all] device={DEV}", flush=True)
    ckpt = OUT / "casia_encoders.pt"
    tmpl = np.load(cfg.NORM_NPZ)["templates"]; L = np.load(cfg.LBL_NPZ)
    subj, eyes = L["subjects"], L["eyes"]
    E.set_seed(SEED)
    (XRg, yRg), (XRp, yRp), clsR = make_gallery_probe_split(tmpl, subj, eyes, 1, seed=SEED)
    (XLg, yLg), (XLp, yLp), clsL = make_gallery_probe_split(tmpl, subj, eyes, 0, seed=SEED)
    cnn = IrisDeepNetResNet18(n_classes=len(clsR)).to(DEV)
    dbn = DBNStyleCNN(n_classes=len(clsL)).to(DEV)
    if ckpt.exists():
        print("[all] loading cached encoders", flush=True)
        z = torch.load(ckpt, map_location=DEV); cnn.load_state_dict(z["cnn"]); dbn.load_state_dict(z["dbn"])
    else:
        E.set_seed(SEED); cnn = E.train_net(cnn, XRg, yRg, XRp, yRp, "cnnR")
        E.set_seed(SEED + 1000); dbn = E.train_net(dbn, XLg, yLg, XLp, yLp, "dbnL")
        torch.save({"cnn": cnn.state_dict(), "dbn": dbn.state_dict()}, ckpt)
        print("[all] encoders cached", flush=True)

    # CASIA full embeddings for matched references
    gR, gRy, pR, pRy = E.embed(cnn, XRg, yRg), yRg, E.embed(cnn, XRp, yRp), yRp
    gL, gLy, pL, pLy = E.embed(dbn, XLg, yLg), yLg, E.embed(dbn, XLp, yLp), yLp

    results = {}
    res_path = OUT / "cross_all_results.json"
    TESTS = [
        ("Interval", DS / "CASIA-Iris-Interval", {"R": "R", "L": "L"}, "*.jpg"),
        ("Lamp",     DS / "CASIA-Iris-Lamp",     {"R": "R", "L": "L"}, "*.jpg"),
        ("MMU",      DS / "MMU-Iris-Database",   {"R": "right", "L": "left"}, "*.bmp"),
    ]
    for name, root, eyemap, ext in TESTS:
        print(f"[all] === {name}: segmenting ===", flush=True)
        by = load_db_bysubject(root, eyemap, ext)
        cross = eval_db(name, cnn, dbn, by)
        ref = casia_matched(gR, gRy, pR, pRy, gL, gLy, pL, pLy, cross["subjects"])
        drop = {k: round(ref[k][0] - cross[k], 2) for k in ["CNN", "DBN", "HR", "BC", "LR"] if k in cross and k in ref}
        results[name] = dict(cross=cross, casia_matched=ref, drop=drop)
        res_path.write_text(json.dumps(results, indent=2))
        print(f"[all] {name} drop={drop}", flush=True)

    print(f"[all] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
