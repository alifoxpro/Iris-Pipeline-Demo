"""
Paper #3 quick-win experiments (real training, no fabrication):
  (1) Multi-seed robustness of the CNN+DBN fusion and recovery ratio.
  (2) Architectural-diversity control: CNN+CNN vs CNN+DBN, by also
      training a CNN on the LEFT eye and pairing it with the right-eye CNN.

For each seed in SEEDS, three networks are trained from the saved
normalised templates (segmentation/normalisation reused):
  - cnnR : IrisDeepNet (ResNet-18)  on the RIGHT eye   (shared matcher A)
  - dbnL : DBN-style CNN             on the LEFT  eye   (pair B1 = CNN+DBN)
  - cnnL : IrisDeepNet (ResNet-18)  on the LEFT  eye   (pair B2 = CNN+CNN)

Results (per seed + aggregate mean/std) are written as JSON to the
paper's experiments folder. Thesis checkpoints are NOT touched
(best weights kept in memory).
"""
import json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg
from stage_3_feature_extraction.iris_dataset import IrisDataset, make_gallery_probe_split
from stage_3_feature_extraction.irisdeepnet_resnet18 import IrisDeepNetResNet18
from stage_3_feature_extraction.dbn_style_cnn import DBNStyleCNN
from stage_4_rank_fusion.rule_highest_rank import highest_rank, sims_for_highest_rank
from stage_4_rank_fusion.rule_borda_count import borda_count, sims_for_borda_count
from stage_4_rank_fusion.rule_logistic_regression import logistic_regression_fusion

SEEDS = [42, 1, 2, 3, 4]
OUTDIR = Path(r"E:\Strudin\Jamal Nasir Al-khafaji\Paper3_Complementarity_PR\experiments")
OUTDIR.mkdir(parents=True, exist_ok=True)
DEV = cfg.DEVICE


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def run_epoch(model, loader, crit, opt=None):
    train = opt is not None
    model.train(train)
    tot_l = tot_c = tot_n = 0
    for x, y in loader:
        x = x.to(DEV); y = y.to(DEV)
        out = model(x); loss = crit(out, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot_l += loss.item() * x.size(0)
        tot_c += (out.argmax(1) == y).sum().item()
        tot_n += x.size(0)
    return tot_l / tot_n, tot_c / tot_n


def train_net(model, Xg, yg, Xp, yp, tag):
    tr = DataLoader(IrisDataset(Xg, yg, augment=True), batch_size=cfg.BATCH, shuffle=True, num_workers=0)
    pr = DataLoader(IrisDataset(Xp, yp, augment=False), batch_size=cfg.BATCH, shuffle=False, num_workers=0)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTH)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS)
    best_acc, best_sd = 0.0, None
    t0 = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        run_epoch(model, tr, crit, opt)
        with torch.no_grad():
            _, va = run_epoch(model, pr, crit, None)
        sch.step()
        if va > best_acc:
            best_acc = va; best_sd = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_sd)
    print(f"    {tag}: best probe acc={best_acc*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    return model


def embed(model, X, y):
    ld = DataLoader(IrisDataset(X, y, augment=False), batch_size=cfg.BATCH, shuffle=False)
    model.eval(); E = []
    with torch.no_grad():
        for x, _ in ld:
            _, e = model(x.to(DEV), return_embedding=True)
            E.append(e.cpu().numpy())
    return np.concatenate(E, 0)


def class_sims_of(g_emb, g_y, p_emb):
    sims = p_emb @ g_emb.T
    classes = np.unique(g_y)
    cs = np.full((sims.shape[0], len(classes)), -1.0)
    for i, c in enumerate(classes):
        cs[:, i] = sims[:, g_y == c].max(1)
    return cs  # (N, C), column index == class index (0..999)


def sims_to_ranks(cs):
    order = np.argsort(-cs, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(cs.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, cs.shape[1] + 1)
    return ranks


def rank1(ranks, py):
    N = len(py)
    return float((ranks[np.arange(N), py] == 1).mean() * 100)


def analyse_pair(csA, csB, py, label):
    """Disagreement cells + HR/BC/LR fusion + recovery ratio for a matcher pair."""
    rA, rB = sims_to_ranks(csA), sims_to_ranks(csB)
    N = len(py)
    okA = rA[np.arange(N), py] == 1
    okB = rB[np.arange(N), py] == 1
    p11 = float((okA & okB).mean() * 100)
    p10 = float((okA & ~okB).mean() * 100)
    p01 = float((~okA & okB).mean() * 100)
    p00 = float((~okA & ~okB).mean() * 100)
    rA1, rB1 = rank1(rA, py), rank1(rB, py)
    strong = max(rA1, rB1)
    oracle = 100 - p00
    gmax = oracle - strong
    # fusion
    hr = rank1(sims_to_ranks(-highest_rank(rA, rB).astype(float)), py)
    bc = rank1(sims_to_ranks(-borda_count(rA, rB).astype(float)), py)
    lr_sims, _ = logistic_regression_fusion(csA, csB, py)
    lr = rank1(sims_to_ranks(lr_sims), py)
    rho = lambda p: float((p - strong) / gmax * 100) if gmax > 1e-9 else float('nan')
    return dict(label=label, rA1=rA1, rB1=rB1, strong=strong,
                p11=p11, p10=p10, p01=p01, p00=p00,
                D=p10 + p01, oracle=oracle, gmax=gmax,
                HR=hr, BC=bc, LR=lr,
                rhoHR=rho(hr), rhoBC=rho(bc), rhoLR=rho(lr))


def main():
    print(f"[exp] device={DEV}", flush=True)
    templates = np.load(cfg.NORM_NPZ)["templates"]
    L = np.load(cfg.LBL_NPZ); subjects = L["subjects"]; eyes = L["eyes"]

    all_results = []
    out_json = OUTDIR / "multiseed_pairs_results.json"
    for si, seed in enumerate(SEEDS):
        print(f"\n=== seed {seed} ({si+1}/{len(SEEDS)}) ===", flush=True)
        set_seed(seed)
        # splits (seed-dependent gallery/probe assignment)
        (XRg, yRg), (XRp, yRp), clsR = make_gallery_probe_split(templates, subjects, eyes, eye_filter=1, seed=seed)
        (XLg, yLg), (XLp, yLp), clsL = make_gallery_probe_split(templates, subjects, eyes, eye_filter=0, seed=seed)

        set_seed(seed)
        cnnR = train_net(IrisDeepNetResNet18(n_classes=len(clsR)).to(DEV), XRg, yRg, XRp, yRp, "cnnR")
        set_seed(seed + 1000)
        dbnL = train_net(DBNStyleCNN(n_classes=len(clsL)).to(DEV), XLg, yLg, XLp, yLp, "dbnL")
        set_seed(seed + 2000)
        cnnL = train_net(IrisDeepNetResNet18(n_classes=len(clsL)).to(DEV), XLg, yLg, XLp, yLp, "cnnL")

        csR = class_sims_of(embed(cnnR, XRg, yRg), yRg, embed(cnnR, XRp, yRp))
        csDL = class_sims_of(embed(dbnL, XLg, yLg), yLg, embed(dbnL, XLp, yLp))
        csCL = class_sims_of(embed(cnnL, XLg, yLg), yLg, embed(cnnL, XLp, yLp))

        # pair 1: CNN(right)+DBN(left) ; pair 2: CNN(right)+CNN(left)
        pair_cd = analyse_pair(csR, csDL, yRp, "CNN+DBN")
        pair_cc = analyse_pair(csR, csCL, yRp, "CNN+CNN")
        rec = dict(seed=seed, cnn_dbn=pair_cd, cnn_cnn=pair_cc)
        all_results.append(rec)
        out_json.write_text(json.dumps(all_results, indent=2))
        print(f"  CNN+DBN: CNN={pair_cd['rA1']:.2f} DBN={pair_cd['rB1']:.2f} LR={pair_cd['LR']:.2f} "
              f"rhoLR={pair_cd['rhoLR']:.1f}% D={pair_cd['D']:.2f}", flush=True)
        print(f"  CNN+CNN: CNN={pair_cc['rA1']:.2f} CNN2={pair_cc['rB1']:.2f} LR={pair_cc['LR']:.2f} "
              f"rhoLR={pair_cc['rhoLR']:.1f}% D={pair_cc['D']:.2f}", flush=True)

    # aggregate
    def agg(key_path):
        vals = []
        for r in all_results:
            d = r
            for k in key_path: d = d[k]
            vals.append(d)
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    summary = {}
    for pair in ["cnn_dbn", "cnn_cnn"]:
        summary[pair] = {m: agg([pair, m]) for m in
                         ["rA1", "rB1", "D", "oracle", "gmax", "HR", "BC", "LR", "rhoLR", "rhoBC", "rhoHR", "p11", "p10", "p01", "p00"]}
    (OUTDIR / "multiseed_pairs_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY (mean, std over seeds) ===", flush=True)
    for pair in ["cnn_dbn", "cnn_cnn"]:
        s = summary[pair]
        print(f"{pair}: LR={s['LR'][0]:.2f}±{s['LR'][1]:.2f}  rhoLR={s['rhoLR'][0]:.1f}±{s['rhoLR'][1]:.1f}%  "
              f"D={s['D'][0]:.2f}±{s['D'][1]:.2f}  oracle={s['oracle'][0]:.2f}", flush=True)
    print("[exp] done.", flush=True)


if __name__ == "__main__":
    main()
