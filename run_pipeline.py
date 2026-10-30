"""
End-to-end runner — execute stages 1..5 of the Chapter 3 methodology
in order, with --from / --to flags so any sub-range can be re-run.

Usage:
    python run_pipeline.py                    # run all stages
    python run_pipeline.py --from 3 --to 5    # re-run training onward
    python run_pipeline.py --from 4           # just fusion + evaluation
    python run_pipeline.py --only 1           # only segmentation
"""

import argparse, time, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config as cfg


STAGES = [
    ("Stage 1 - Iris segmentation",
     "stage_1_segmentation.segment_all"),
    ("Stage 2 - Daugman normalisation",
     "stage_2_normalisation.normalise_all"),
    ("Stage 3 - Feature extraction (CNN + DBN)",
     "stage_3_feature_extraction.train"),
    ("Stage 4 - Rank-level fusion + verification",
     "stage_4_rank_fusion.fuse_and_evaluate"),
    ("Stage 5 - Statistical significance",
     "stage_5_evaluation.evaluate"),
    ("Stage 6 - Cross-dataset validation on IITD",
     "stage_5_evaluation.cross_dataset_iitd"),
]


def run_module(mod_path):
    """Import the module and call its main()."""
    import importlib
    mod = importlib.import_module(mod_path)
    mod.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", type=int, default=1)
    ap.add_argument("--to",   type=int, default=len(STAGES))
    ap.add_argument("--only", type=int, default=None,
                    help="run only this stage (overrides --from/--to)")
    args = ap.parse_args()

    cfg.show()

    if args.only is not None:
        lo = hi = args.only
    else:
        lo, hi = args.frm, args.to

    t0 = time.time()
    for i, (title, mod) in enumerate(STAGES, start=1):
        if not (lo <= i <= hi):
            continue
        print(f"\n{'='*72}\n>>> {title}\n{'='*72}")
        t_s = time.time()
        run_module(mod)
        print(f"<<< done in {time.time() - t_s:.1f}s")

    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
