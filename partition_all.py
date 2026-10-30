# -*- coding: utf-8 -*-
"""Disagreement 4-cell partition (pi11/pi10/pi01/pi00) of the CNN & DBN matchers
on EVERY database, via subject-aware right/left probe pairing. CASIA + MMU use
cached embeddings; Lamp + Interval are segmented + embedded with the frozen
CASIA encoders. Output: partition_all.json."""
import json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as cfg
import exp_multiseed_pairs as E
import cross_all as X
from stage_3_feature_extraction.iris_dataset import make_gallery_probe_split
from stage_3_feature_extraction.irisdeepnet_resnet18 import IrisDeepNetResNet18
from stage_3_feature_extraction.dbn_style_cnn import DBNStyleCNN

OUT = Path(r"E:\Strudin\Jamal Nasir Al-khafaji\Paper3_Complementarity_PR\experiments")
DS = HERE / "dataset"
DEV = cfg.DEVICE


def part(csR, csL, pyc):
    a = csR.argmax(1) == pyc
    b = csL.argmax(1) == pyc
    n = len(pyc)
    return dict(p11=round(float((a & b).mean()) * 100, 2),
                p10=round(float((a & ~b).mean()) * 100, 2),
                p01=round(float((~a & b).mean()) * 100, 2),
                p00=round(float((~a & ~b).mean()) * 100, 2),
                n=int(n), D=round(float(((a & ~b) | (~a & b)).mean()) * 100, 2))


def aligned_pairs(pR, pRy, pL, pLy):
    subs = np.unique(pRy)
    iR, iL, py = [], [], []
    for s in subs:
        ri = np.where(pRy == s)[0]; li = np.where(pLy == s)[0]
        k = min(len(ri), len(li))
        iR += list(ri[:k]); iL += list(li[:k]); py += [s] * k
    return np.array(iR), np.array(iL), np.array(py)


def main():
    t0 = time.time()
    # encoders (cached)
    tmpl = np.load(cfg.NORM_NPZ)["templates"]; L = np.load(cfg.LBL_NPZ)
    subj, eyes = L["subjects"], L["eyes"]
    E.set_seed(42)
    (_, _), (_, _), clsR = make_gallery_probe_split(tmpl, subj, eyes, 1, seed=42)
    (_, _), (_, _), clsL = make_gallery_probe_split(tmpl, subj, eyes, 0, seed=42)
    cnn = IrisDeepNetResNet18(n_classes=len(clsR)).to(DEV)
    dbn = DBNStyleCNN(n_classes=len(clsL)).to(DEV)
    z = torch.load(OUT / "casia_encoders.pt", map_location=DEV)
    cnn.load_state_dict(z["cnn"]); dbn.load_state_dict(z["dbn"])
    print("[part] encoders loaded", flush=True)

    res = {}
    # ---- CASIA (cached embeddings) ----
    e = np.load(OUT / "xm_embeddings.npz")
    iR, iL, py = aligned_pairs(e["pR"], e["pRy"], e["pL"], e["pLy"])
    csR, clR = X.class_sims(e["gR"], e["gRy"], e["pR"][iR])
    csL, clL = X.class_sims(e["gL"], e["gLy"], e["pL"][iL])
    pyc = X.remap(py, clR)
    res["CASIA-Iris-Thousand"] = part(csR, csL, pyc)
    print("[part] CASIA:", res["CASIA-Iris-Thousand"], flush=True)

    # ---- MMU (cached embeddings) ----
    iR, iL, py = aligned_pairs(e["pRm"], e["pRym"], e["pLm"], e["pLym"])
    csR, clR = X.class_sims(e["gRm"], e["gRym"], e["pRm"][iR])
    csL, clL = X.class_sims(e["gLm"], e["gLym"], e["pLm"][iL])
    pyc = X.remap(py, clR)
    res["MMU"] = part(csR, csL, pyc)
    print("[part] MMU:", res["MMU"], flush=True)

    # ---- Lamp & Interval (segment + embed) ----
    TESTS = [("CASIA-Iris-Lamp", DS / "CASIA-Iris-Lamp", {"R": "R", "L": "L"}, "*.jpg"),
             ("CASIA-Iris-Interval", DS / "CASIA-Iris-Interval", {"R": "R", "L": "L"}, "*.jpg")]
    for name, root, eyemap, ext in TESTS:
        print(f"[part] segmenting {name} ...", flush=True)
        by = X.load_db_bysubject(root, eyemap, ext)
        gR, gRy, gL, gLy, pR, pL, pY, nv = X.build_eval(by)
        geR = E.embed(cnn, gR, gRy); peR = E.embed(cnn, pR, pY)
        geL = E.embed(dbn, gL, gLy); peL = E.embed(dbn, pL, pY)
        csR, clR = X.class_sims(geR, gRy, peR)
        csL, clL = X.class_sims(geL, gLy, peL)
        pyc = X.remap(pY, clR)
        res[name] = part(csR, csL, pyc)
        print(f"[part] {name}:", res[name], flush=True)
        (OUT / "partition_all.json").write_text(json.dumps(res, indent=2))

    (OUT / "partition_all.json").write_text(json.dumps(res, indent=2))
    print(f"[part] done in {time.time()-t0:.0f}s -> partition_all.json", flush=True)


if __name__ == "__main__":
    main()
