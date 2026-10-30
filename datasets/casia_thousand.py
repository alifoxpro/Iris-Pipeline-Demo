"""CASIA-Iris-Thousand loader.

Folder layout on disk:

    dataset/CASIA-Iris-Thousand/
    ├── 000/
    │   ├── L/  S5000L00.jpg ... S5000L09.jpg
    │   └── R/  S5000R00.jpg ... S5000R09.jpg
    ├── 001/
    │   ├── L/ ...
    │   └── R/ ...
    ...
    └── 999/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg

name = "casia"


def _resolve_root():
    """Handle the common nested layout CASIA-Iris-Thousand/CASIA-Iris-Thousand."""
    root = cfg.DATASET
    if not (root / "000").exists():
        nested = root / "CASIA-Iris-Thousand"
        if (nested / "000").exists():
            return nested
    return root


def enumerate_images():
    """Return list[{'subject','eye','path','key'}] in deterministic order."""
    root = _resolve_root()
    records = []
    for sid in range(cfg.N_SUBJECTS):
        for eye in ("L", "R"):
            d = root / f"{sid:03d}" / eye
            if not d.exists():
                continue
            for jpg in sorted(d.glob("*.jpg")):
                records.append({
                    "subject": sid,
                    "eye":     eye,
                    "path":    jpg,
                    "key":     f"{sid:03d}_{eye}_{jpg.stem}",
                })
    return records


if __name__ == "__main__":
    recs = enumerate_images()
    print(f"CASIA images : {len(recs)}")
    print(f"first        : {recs[0]}")
    print(f"last         : {recs[-1]}")
