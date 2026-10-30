"""
Stage 4 driver — apply Highest-Rank, Borda Count, and Logistic
Regression fusion to the IrisDeepNet + DBN-style per-class
similarity matrices produced by Stage 3, then compute identification
(CMC) and verification (ROC / EER / AUC) metrics for every method.

Writes:
    outputs/results/final_metrics.json
    outputs/results/verification_data.npz
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg
from stage_4_rank_fusion.rule_highest_rank        import (
    highest_rank, sims_for_highest_rank,
)
from stage_4_rank_fusion.rule_borda_count         import (
    borda_count,  sims_for_borda_count,
)
from stage_4_rank_fusion.rule_logistic_regression import (
    logistic_regression_fusion,
)


# ----- helpers ---------------------------------------------------------
def sims_to_ranks(class_sims):
    """Convert (N, C) similarity matrix to a rank matrix (1 = best)."""
    order = np.argsort(-class_sims, axis=1)
    ranks = np.empty_like(order)
    rows  = np.arange(class_sims.shape[0])[:, None]
    ranks[rows, order] = np.tile(np.arange(1, class_sims.shape[1] + 1),
                                  (class_sims.shape[0], 1))
    return ranks


def cmc_curve(ranks_per_probe, P_Y, K=None):
    K = K or cfg.CMC_K
    N = len(P_Y)
    true_ranks = np.array([ranks_per_probe[i, P_Y[i]] for i in range(N)])
    return np.array([(true_ranks <= k).mean() for k in range(1, K + 1)]) * 100


def split_gen_imp(scores, p_y):
    same = np.zeros_like(scores, dtype=bool)
    for i in range(scores.shape[0]):
        same[i, p_y[i]] = True
    gen, imp = scores[same], scores[~same]
    if len(imp) > cfg.IMP_SUBSAMPLE:
        rng = np.random.RandomState(cfg.SEED)
        imp = imp[rng.choice(len(imp), cfg.IMP_SUBSAMPLE, replace=False)]
    return gen, imp


def roc_metrics(gen, imp):
    y   = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    s   = np.concatenate([gen, imp])
    fpr, tpr, _ = roc_curve(y, s)
    auc_v = auc(fpr, tpr)
    fnr   = 1 - tpr
    idx   = np.argmin(np.abs(fpr - fnr))
    eer   = (fpr[idx] + fnr[idx]) / 2 * 100
    return fpr, tpr, auc_v, eer


# ----- driver ----------------------------------------------------------
def main():
    print("[Stage 4] Rank-level fusion + evaluation")
    cnn = np.load(cfg.OUT_FEAT / "cnn_eval.npz", allow_pickle=True)
    dbn = np.load(cfg.OUT_FEAT / "dbn_eval.npz", allow_pickle=True)
    assert np.array_equal(cnn["p_y"], dbn["p_y"]), \
        "CNN and DBN probe orderings must match."
    P_Y = cnn["p_y"]
    N   = len(P_Y)

    cnn_cs   = cnn["class_sims"]
    dbn_cs   = dbn["class_sims"]
    cnn_ranks = sims_to_ranks(cnn_cs)
    dbn_ranks = sims_to_ranks(dbn_cs)
    print(f"  Probes: {N}  Classes: {cnn_cs.shape[1]}")

    # ---- Fusion outputs ---------------------------------------------
    hr_ranks = sims_to_ranks(-highest_rank(cnn_ranks, dbn_ranks).astype(float))
    bc_ranks = sims_to_ranks(-borda_count (cnn_ranks, dbn_ranks).astype(float))
    lr_sims, _ = logistic_regression_fusion(cnn_cs, dbn_cs, P_Y)
    lr_ranks   = sims_to_ranks(lr_sims)

    rank_table = {
        "IrisDeepNet":         cnn_ranks,
        "DBN":                 dbn_ranks,
        "Highest Rank":        hr_ranks,
        "Borda Count":         bc_ranks,
        "Logistic Regression": lr_ranks,
    }
    sim_table  = {
        "IrisDeepNet":         cnn_cs,
        "DBN":                 dbn_cs,
        "Highest Rank":        sims_for_highest_rank(cnn_cs, dbn_cs),
        "Borda Count":         sims_for_borda_count (cnn_cs, dbn_cs),
        "Logistic Regression": lr_sims,
    }

    # ---- CMC, ROC, save --------------------------------------------
    results, verify_data = {}, {}
    for name in rank_table:
        cmc = cmc_curve(rank_table[name], P_Y)
        gen, imp = split_gen_imp(sim_table[name], P_Y)
        fpr, tpr, auc_v, eer = roc_metrics(gen, imp)
        results[name] = dict(
            rank1=float(cmc[0]), rank5=float(cmc[4]), rank20=float(cmc[19]),
            cmc=cmc.tolist(),
            eer=float(eer), auc=float(auc_v),
        )
        verify_data[name] = dict(fpr=fpr, tpr=tpr, gen=gen, imp=imp)
        print(f"  {name:22s} rank-1={cmc[0]:5.2f}%  rank-5={cmc[4]:5.2f}%  "
              f"EER={eer:5.2f}%  AUC={auc_v:.4f}")

    # ---- Disagreement profile (CNN vs DBN) --------------------------
    cnn_ok = cnn_ranks[np.arange(N), P_Y] == 1
    dbn_ok = dbn_ranks[np.arange(N), P_Y] == 1
    profile = dict(
        both_correct=int(( cnn_ok &  dbn_ok).sum()),
        cnn_only    =int(( cnn_ok & ~dbn_ok).sum()),
        dbn_only    =int((~cnn_ok &  dbn_ok).sum()),
        both_wrong  =int((~cnn_ok & ~dbn_ok).sum()),
        total       =int(N),
    )
    results["disagreement"] = profile
    print(f"  Disagreement   CNN-only={profile['cnn_only']}  "
          f"DBN-only={profile['dbn_only']}  both-wrong={profile['both_wrong']}")

    (cfg.OUT_RES / "final_metrics.json").write_text(json.dumps(results, indent=2))
    np.savez(cfg.OUT_RES / "verification_data.npz",
             **{f"{k}_fpr": v["fpr"] for k, v in verify_data.items()},
             **{f"{k}_tpr": v["tpr"] for k, v in verify_data.items()},
             **{f"{k}_gen": v["gen"] for k, v in verify_data.items()},
             **{f"{k}_imp": v["imp"] for k, v in verify_data.items()})
    print(f"\n[Stage 4] Done. Results -> {cfg.OUT_RES}")


if __name__ == "__main__":
    main()
