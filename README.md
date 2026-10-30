# methodology_implementation/

Reference implementation of the **Chapter 3 methodology** of the
UNITEN PhD thesis *"Iris Segmentation and Recognition Using Deep
Learning Approach"* (Hind).

Each sub-folder corresponds to one **stage** of the pipeline and
contains only the code for that stage.  Stages are executed in
order; each stage reads the previous stage's outputs from
`outputs/` and writes its own.

---

## Folder layout

```
methodology_implementation/
├── README.md                        <- this file
├── requirements.txt                 <- pinned dependency list
├── config.py                        <- single source of truth for
│                                       paths + hyperparameters
├── run_pipeline.py                  <- master runner (--from/--to)
│
├── dataset/
│   ├── CASIA-Iris-Thousand/         <- junction to the real dataset
│   │                                   (1000 subjects, 20 000 images,
│   │                                   538 MB; NOT duplicated)
│   └── IITD/                        <- cross-dataset validation set
│       └── README.md                   download + placement instructions
│
├── datasets/                        <- dataset-agnostic loaders
│   ├── __init__.py                     unified enumerate_images() API
│   ├── casia_thousand.py
│   └── iitd.py
│
├── stage_1_segmentation/            <- §3.3
│   ├── __init__.py
│   ├── s1_specular_removal.py       <- §3.3.1, Eqs. 3.1-3.2
│   ├── s2_pupil_detection.py        <- §3.3.2, Eq. 3.3
│   ├── s3_limbus_detection.py       <- §3.3.3, Eq. 3.4 + AHE
│   ├── s4_eyelid_detection.py       <- §3.3.4, Eqs. 3.5-3.6
│   └── segment_all.py               <- driver: runs all four on every image
│
├── stage_2_normalisation/           <- §3.4
│   ├── __init__.py
│   ├── daugman_unwrap.py            <- Eq. 3.7 rubber-sheet
│   └── normalise_all.py             <- driver
│
├── stage_3_feature_extraction/      <- §3.5
│   ├── __init__.py
│   ├── iris_dataset.py              <- torch Dataset + gallery/probe split
│   ├── irisdeepnet_resnet18.py      <- IrisDeepNet (Fig. 3.5)
│   ├── dbn_style_cnn.py             <- DBN-style 4-layer CNN (Fig. 3.6)
│   └── train.py                     <- driver: trains both nets +
│                                       extracts 256-D embeddings
│
├── stage_4_rank_fusion/             <- §3.6
│   ├── __init__.py
│   ├── rule_highest_rank.py         <- §3.6.1
│   ├── rule_borda_count.py          <- §3.6.2
│   ├── rule_logistic_regression.py  <- §3.6.3
│   └── fuse_and_evaluate.py         <- driver: CMC + ROC + EER + AUC
│
├── stage_5_evaluation/              <- §3.7
│   ├── __init__.py
│   ├── mcnemar_test.py              <- §3.7.1 paired exact test
│   ├── wilson_ci.py                 <- §3.7.2 95% CI on proportion
│   ├── bootstrap_ci.py              <- §3.7.3 stratified bootstrap AUC CI
│   ├── cohen_h.py                   <- §3.7.4 effect-size h
│   ├── evaluate.py                  <- driver: runs all tests
│   └── cross_dataset_iitd.py        <- §3.8 cross-dataset validation on IITD
│
├── scripts_aux/                     <- chapter-3 illustrative figures
│   ├── README.md
│   └── generate_*.py
│
└── outputs/                         <- everything the pipeline writes
    ├── segmentation/                <-   stage 1 output
    ├── normalised_templates/        <-   stage 2 output
    ├── checkpoints/                 <-   stage 3 weights
    ├── features/                    <-   stage 3 embeddings + sims
    └── results/                     <-   stage 4/5 metrics + JSON
```

---

## Setup

```bash
# (1) create a clean environment (recommended)
python -m venv venv
venv/Scripts/activate                # on Windows
# venv/bin/activate                  # on Linux/macOS

# (2) install dependencies
pip install -r requirements.txt

# (3) verify the dataset junction
python -c "import config; print(config.DATASET); print(list(config.DATASET.glob('000/L/*.jpg'))[:2])"
# expected: 2 .jpg paths under .../CASIA-Iris-Thousand/000/L/
```

The `dataset/CASIA-Iris-Thousand/` link is a **Windows junction** to
the original 538 MB CASIA folder.  No data is duplicated.  If you
move the dataset, recreate the junction:

```powershell
$link  = "methodology_implementation\dataset\CASIA-Iris-Thousand"
$target= "<new path to>\CASIA-Iris-Thousand"
if (Test-Path $link) { (Get-Item $link).Delete() }
New-Item -ItemType Junction -Path $link -Target $target
```

---

## Phased execution

### Run everything end-to-end

```bash
python run_pipeline.py
```

This runs stages 1 through 5 in order, printing per-stage timing.

### Run a single stage

```bash
python run_pipeline.py --only 1     # segmentation only
python run_pipeline.py --only 3     # training only
python run_pipeline.py --only 5     # statistics only
```

### Run a sub-range

```bash
python run_pipeline.py --from 3 --to 5   # re-train + re-fuse + re-test
python run_pipeline.py --from 4          # re-run fusion + significance
```

### Run any stage directly

```bash
python -m stage_1_segmentation.segment_all
python -m stage_2_normalisation.normalise_all
python -m stage_3_feature_extraction.train
python -m stage_4_rank_fusion.fuse_and_evaluate
python -m stage_5_evaluation.evaluate
python -m stage_5_evaluation.cross_dataset_iitd      # cross-dataset on IITD
```

### Cross-dataset validation on IITD

The implementation supports cross-dataset validation on the **IITD
Iris Database** (224 subjects, 1,120 NIR images).  Procedure:

1.  Follow `dataset/IITD/README.md` to download IITD and drop the
    extracted subject folders into `dataset/IITD/001/ ... 224/`.
2.  Make sure stages 1-3 have already run on CASIA so that
    `outputs/checkpoints/cnn_best.pth` and `dbn_best.pth` exist.
3.  Run:

    ```bash
    python -m stage_5_evaluation.cross_dataset_iitd
    # or equivalently:
    python run_pipeline.py --only 6
    ```

The script segments + unwraps IITD, loads the CASIA-trained networks
*without retraining* (the classifier head is rebuilt for IITD's
subject count; the feature extractor is frozen), then reports rank-1
+ Wilson CI, EER, AUC + bootstrap CI for every method, plus the
cross-dataset gap and Cohen's h vs the within-CASIA result.

Output:  `outputs/results/iitd_cross_dataset.json`

### Smoke-test a sub-stage

Each sub-stage file has a small `if __name__ == "__main__":` block:

```bash
python stage_1_segmentation/s1_specular_removal.py    # 1 image
python stage_1_segmentation/s3_limbus_detection.py    # 1 image
python stage_2_normalisation/daugman_unwrap.py        # 1 polar template
```

---

## What each stage produces

| Stage | Reads                                   | Writes                                                                              |
|-------|-----------------------------------------|-------------------------------------------------------------------------------------|
| 1     | `dataset/CASIA-Iris-Thousand/`          | `outputs/segmentation/segmentation_meta.json` + overlay PNGs                        |
| 2     | `outputs/segmentation/...`              | `outputs/normalised_templates/normalised.npz` + `labels.npz`                        |
| 3     | `outputs/normalised_templates/...`      | `outputs/checkpoints/{cnn,dbn}_best.pth` + `outputs/features/{cnn,dbn}_eval.npz`    |
| 4     | `outputs/features/...`                  | `outputs/results/final_metrics.json` + `verification_data.npz`                      |
| 5     | `outputs/features/`, `outputs/results/` | `outputs/results/significance.json`                                                 |
| 6     | `dataset/IITD/`, `outputs/checkpoints/` | `outputs/iitd/...` + `outputs/results/iitd_cross_dataset.json`                      |

---

## Mapping to Chapter 3

| Section in Ch. 3                  | Implementation module                                  |
|-----------------------------------|--------------------------------------------------------|
| §3.3.1 Specular removal           | `stage_1_segmentation/s1_specular_removal.py`          |
| §3.3.2 Pupil detection            | `stage_1_segmentation/s2_pupil_detection.py`           |
| §3.3.3 Limbus localisation (AHE)  | `stage_1_segmentation/s3_limbus_detection.py`          |
| §3.3.4 Eyelid suppression         | `stage_1_segmentation/s4_eyelid_detection.py`          |
| §3.4   Daugman rubber-sheet       | `stage_2_normalisation/daugman_unwrap.py`              |
| §3.5.1 IrisDeepNet (ResNet-18)    | `stage_3_feature_extraction/irisdeepnet_resnet18.py`   |
| §3.5.2 DBN-style 4-layer CNN      | `stage_3_feature_extraction/dbn_style_cnn.py`          |
| §3.5.3 Embedding + training       | `stage_3_feature_extraction/train.py`                  |
| §3.6.1 Highest Rank               | `stage_4_rank_fusion/rule_highest_rank.py`             |
| §3.6.2 Borda Count                | `stage_4_rank_fusion/rule_borda_count.py`              |
| §3.6.3 Logistic Regression        | `stage_4_rank_fusion/rule_logistic_regression.py`      |
| §3.7.1 McNemar's exact test       | `stage_5_evaluation/mcnemar_test.py`                   |
| §3.7.2 Wilson 95% CI              | `stage_5_evaluation/wilson_ci.py`                      |
| §3.7.3 Bootstrap 95% CI on AUC    | `stage_5_evaluation/bootstrap_ci.py`                   |
| §3.7.4 Cohen's h                  | `stage_5_evaluation/cohen_h.py`                        |

---

## Hyperparameter sheet

All values live in `config.py` and match the published tables of
Chapter 3 (Tables 3.4 + 3.5):

| Parameter                | Value          | From          |
|--------------------------|----------------|---------------|
| Input resize             | 240 x 320      | §3.2          |
| Specular threshold T     | 240            | Eq. 3.1       |
| Telea inpaint radius     | 3 px           | Eq. 3.2       |
| AHE clipLimit / tile     | 2.0 / 8 x 8    | §3.3.3        |
| Perona-Malik κ           | 30             | Eq. 3.5       |
| Perona-Malik iterations  | 50             | Eq. 3.5       |
| Polar template           | 64 x 256       | Eq. 3.7       |
| Embedding dim            | 256            | Fig. 3.5      |
| Batch size               | 64             | Table 3.5     |
| Epochs                   | 80             | Table 3.5     |
| Optimiser                | AdamW          | Table 3.5     |
| Learning rate            | 1e-3           | Table 3.5     |
| Weight decay             | 5e-4           | Table 3.5     |
| Label smoothing          | 0.1            | Table 3.5     |
| Dropout (CNN / DBN)      | 0.4 / 0.3      | §3.5          |
| Bootstrap B              | 1000           | §3.7.3        |
| Significance α           | 0.05           | §3.7          |

---

## Expected runtime

On a workstation with a single RTX-class GPU:

| Stage | Approx. runtime  |
|-------|------------------|
| 1     | ~ 5 - 8 minutes  |
| 2     | ~ 2 - 3 minutes  |
| 3     | ~ 60 - 90 minutes (80 epochs × 2 nets) |
| 4     | ~ 30 seconds     |
| 5     | ~ 30 seconds (incl. 1000-rep bootstrap on 5 methods) |
| **Total** | **~ 1.5 - 2 hours** |

On CPU only, stage 3 dominates (several hours).

---

## Reproducibility

* `config.SEED = 42` is set in PyTorch and NumPy at the top of
  every training script.
* Gallery/probe splits are deterministic given the seed.
* Logistic-regression fusion uses the same seed for its 50/50 split,
  so Stage 4 and Stage 5 see the same held-out probes.
* `outputs/results/training_history.json` records per-epoch loss
  + accuracy for both networks (used to plot Fig. 4.1).
