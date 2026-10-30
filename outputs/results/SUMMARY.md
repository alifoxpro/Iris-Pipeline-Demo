# Chapter 4 — Real Experimental Results: SUMMARY

**Dataset:** CASIA-Iris-Thousand · 1,000 subjects · 20,000 images
**Splits:** 6 gallery / 4 probe per subject per eye = 12,000 / 8,000
**Hardware:** NVIDIA RTX 5070 GPU · CUDA 12.8 · PyTorch 2.7
**Evaluation:** Nearest-neighbor cosine matching, max-aggregation per class

---

## Table 4.1 — Per-method Identification & Verification Performance

| Method | Rank-1 (%) | Rank-5 (%) | Rank-20 (%) | EER (%) | AUC |
|---|---:|---:|---:|---:|---:|
| IrisDeepNet | 70.73 | 80.33 | 87.42 | 7.69 | 0.9773 |
| DBN | 58.98 | 71.33 | 80.00 | 12.89 | 0.9349 |
| Highest Rank | 64.08 | 89.72 | 94.83 | 6.63 | 0.9781 |
| Borda Count | 69.35 | 79.97 | 87.50 | 5.08 | 0.9872 |
| Logistic Regression | 80.33 | 89.03 | 93.67 | 5.20 | 0.9877 |

## Table 4.2 — Disagreement Profile (CNN vs DBN, 4000 probes)

| Cell | Count | Fraction (%) |
|---|---:|---:|
| Both correct | 1792 | 44.80 |
| CNN-only correct | 1037 | 25.92 |
| DBN-only correct | 567 | 14.17 |
| Both wrong | 604 | 15.10 |

## Table 4.3 — Wilson 95% Confidence Intervals on Rank-1

| Method | Rank-1 (%) | 95% CI |
|---|---:|:---:|
| IrisDeepNet | 70.73 | [69.30, 72.11] |
| DBN | 58.98 | [57.44, 60.49] |
| Highest Rank | 64.08 | [62.58, 65.55] |
| Borda Count | 69.35 | [67.90, 70.76] |
| Logistic Regression | 80.33 | [79.06, 81.53] |

## Table 4.4 — Bootstrap 95% Confidence Intervals on AUC (B=1000)

| Method | AUC | 95% CI | Std |
|---|---:|:---:|---:|
| IrisDeepNet | 0.9773 | [0.9745, 0.9800] | 0.0013 |
| DBN | 0.9349 | [0.9299, 0.9396] | 0.0027 |
| Highest Rank | 0.9781 | [0.9751, 0.9807] | 0.0015 |
| Borda Count | 0.9872 | [0.9852, 0.9892] | 0.0010 |
| Logistic Regression | 0.9877 | [0.9857, 0.9894] | 0.0009 |

## Table 4.5 — McNemar's Tests (Pairwise Statistical Significance)

| Method A | Method B | A-only correct | B-only correct | p-value | Significance |
|---|---|---:|---:|:---:|:---:|
| Logistic Regression | IrisDeepNet | 453 | 69 | 3.00e-70 | *** |
| Logistic Regression | DBN | 1080 | 226 | 1.04e-133 | *** |
| Logistic Regression | Highest Rank | 795 | 145 | 3.36e-109 | *** |
| Logistic Regression | Borda Count | 516 | 77 | 9.59e-81 | *** |
| Borda Count | IrisDeepNet | 461 | 516 | 8.40e-02 | ns |
| Borda Count | Highest Rank | 574 | 363 | 5.62e-12 | *** |
| IrisDeepNet | DBN | 1037 | 567 | 4.10e-32 | *** |

> Significance: *** p<0.001 · ** p<0.01 · * p<0.05 · ns p≥0.05

## Table 4.6 — Ablation Study

| ID | Configuration | Rank-1 (%) | EER (%) | AUC | Δ Rank-1 |
|---|---|---:|---:|---:|---:|
| A1 | Full proposed system | 80.80 | 4.87 | 0.9892 | +0.00% |
| A2 | Replace LR fusion with Borda Count | 69.65 | 4.82 | 0.9886 | -11.15% |
| A3 | Replace LR fusion with Highest Rank | 66.30 | 6.17 | 0.9812 | -14.50% |
| A4 | Remove DBN branch (no fusion) | 71.92 | 7.40 | 0.9788 | -8.88% |
| A5 | Remove CNN branch (no fusion) | 58.88 | 13.02 | 0.9378 | -21.93% |
| A6 | CNN trained WITHOUT data augmentation (vs CNN-only baseline) | 71.90 | 7.84 | 0.9730 | -0.03% |
| A7 | CNN with random init (vs CNN-only baseline) | 60.65 | 18.19 | 0.9024 | -11.27% |

## Key Findings

1. **Logistic Regression rank-level fusion is the strongest configuration**, with rank-1 = 80.33%, EER = 5.20%, AUC = 0.9877.
2. **Architectural diversity matters:** removing the DBN branch drops rank-1 by 8.88%, removing the CNN branch drops it by 21.93%.
3. **Fusion-rule choice matters:** moving from LR to Borda Count costs 11.15%; from LR to Highest Rank costs 14.50%.
4. **ImageNet pretraining is essential** (random-init CNN drops 11.27% relative to the CNN baseline).
5. **Data augmentation has marginal effect** at this dataset scale (0.03% — likely because 6,000 training samples are sufficient for the ResNet-18 backbone).
6. **All paired comparisons against LR Fusion are highly significant** (p < 10⁻⁷⁰ via McNemar's exact test).