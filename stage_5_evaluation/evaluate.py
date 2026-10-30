"""
Stage 5 driver — runs every statistical test on the fused Stage 4
results and writes a single JSON summary.

Tests:
    - Wilson 95 % CI on rank-1 (every method)
    - Bootstrap 95 % CI on AUC (every method, B=1000)
    - McNemar's exact paired test  (LR vs each of the others, and a
      few other pairs of interest)
    - Cohen's h effect size        (same pairs as McNemar)

Writes:  outputs/results/significance.json
"""

import json
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg
from stage_4_rank_fusion.fuse_and_evaluate import sims_to_ranks
from stage_4_rank_fusion.rule_highest_rank import highest_rank
from stage_4_rank_fusion.rule_borda_count  import borda_count
from stage_4_rank_fusion.rule_logistic_regression import logistic_regression_fusion
from stage_5_evaluation.mcnemar_test  import mcnemar_test
from stage_5_evaluation.wilson_ci     import wilson_ci
from stage_5_evaluation.bootstrap_ci  import bootstrap_auc_ci
from stage_5_evaluation.cohen_h       import cohen_h, cohen_h_label


# ----- correctness vector per method ----------------------------------
def correctness_table():
    cnn = np.load(cfg.OUT_FEAT / "cnn_eval.npz", allow_pickle=True)
    dbn = np.load(cfg.OUT_FEAT / "dbn_eval.npz", allow_pickle=True)
    P_Y = cnn["p_y"]
    N   = len(P_Y)

    cnn_cs    = cnn["class_sims"]; dbn_cs = dbn["class_sims"]
    cnn_ranks = sims_to_ranks(cnn_cs)
    dbn_ranks = sims_to_ranks(dbn_cs)

    lr_sims, _ = logistic_regression_fusion(cnn_cs, dbn_cs, P_Y)
    table = {
        "IrisDeepNet":         cnn_ranks[np.arange(N), P_Y] == 1,
        "DBN":                 dbn_ranks[np.arange(N), P_Y] == 1,
        "Highest Rank":        sims_to_ranks(
            -highest_rank(cnn_ranks, dbn_ranks).astype(float)
        )[np.arange(N), P_Y] == 1,
        "Borda Count":         sims_to_ranks(
            -borda_count(cnn_ranks, dbn_ranks).astype(float)
        )[np.arange(N), P_Y] == 1,
        "Logistic Regression": sims_to_ranks(lr_sims)[np.arange(N), P_Y] == 1,
    }
    return table, N


# ----- driver --------------------------------------------------------
def main():
    print("[Stage 5] Statistical evaluation")
    correct, N = correctness_table()
    metrics    = json.loads((cfg.OUT_RES / "final_metrics.json").read_text())
    verify     = np.load(cfg.OUT_RES / "verification_data.npz")

    # 1) Wilson CI ----------------------------------------------------
    print("\n--- Wilson 95% CI on rank-1 ---")
    wilson = {}
    for name, c in correct.items():
        k = int(c.sum())
        lo, hi = wilson_ci(k, N, alpha=cfg.ALPHA)
        wilson[name] = dict(rank1=float(k/N*100),
                            ci95_low=lo, ci95_high=hi, n=N)
        print(f"  {name:22s} {k/N*100:6.2f}%  [{lo:5.2f}, {hi:5.2f}]")

    # 2) Bootstrap CI on AUC -----------------------------------------
    print(f"\n--- Bootstrap 95% CI on AUC (B={cfg.BOOTSTRAP_B}) ---")
    boot = {}
    for name in correct:
        gen = verify[f"{name}_gen"]; imp = verify[f"{name}_imp"]
        auc_v, lo, hi, sd = bootstrap_auc_ci(gen, imp)
        boot[name] = dict(auc=auc_v, auc_ci_low=lo, auc_ci_high=hi, auc_std=sd)
        print(f"  {name:22s} AUC = {auc_v:.4f}  [{lo:.4f}, {hi:.4f}]")

    # 3) McNemar pairs ----------------------------------------------
    pairs = [
        ("Logistic Regression", "IrisDeepNet"),
        ("Logistic Regression", "DBN"),
        ("Logistic Regression", "Highest Rank"),
        ("Logistic Regression", "Borda Count"),
        ("Borda Count",         "IrisDeepNet"),
        ("Borda Count",         "Highest Rank"),
        ("IrisDeepNet",         "DBN"),
    ]
    print("\n--- McNemar's test + Cohen's h ---")
    print(f"{'comparison':45s} {'B-only':>6s} {'A-only':>6s} "
          f"{'p':>10s} {'sig':>4s} {'h':>7s} {'mag':>8s}")
    mcnemar = []
    for A, B in pairs:
        res = mcnemar_test(correct[A], correct[B])
        pA  = correct[A].mean()
        pB  = correct[B].mean()
        h   = cohen_h(pA, pB)
        mag = cohen_h_label(h)
        mcnemar.append(dict(
            method_A=A, method_B=B,
            A_only_correct=res.c, B_only_correct=res.b,
            p_value=res.p_value, significance=res.significance,
            cohen_h=h, h_magnitude=mag,
        ))
        print(f"{A+'  vs  '+B:45s} {res.b:6d} {res.c:6d} "
              f"{res.p_value:10.4g} {res.significance:>4s} {h:+.4f} {mag:>8s}")

    out = dict(n_probes=N, wilson_ci=wilson,
               bootstrap_ci=boot, mcnemar=mcnemar)
    (cfg.OUT_RES / "significance.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {cfg.OUT_RES / 'significance.json'}")


if __name__ == "__main__":
    main()
