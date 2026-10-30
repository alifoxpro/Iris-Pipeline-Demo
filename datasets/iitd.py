"""IITD Iris Database loader.

Expected folder layout on disk (after extraction):

    dataset/IITD/
    ├── 001/
    │   ├── 01_L.bmp  02_L.bmp  03_L.bmp  04_L.bmp  05_L.bmp
    │   └── 06_R.bmp  07_R.bmp  08_R.bmp  09_R.bmp  10_R.bmp
    ├── 002/
    │   └── ...
    ...
    └── 224/

The loader is tolerant of two real-world variants seen in IITD
distributions:

    (a) numbered files with explicit suffix:   05_L.bmp / 06_R.bmp
    (b) plain numbered files without suffix:   05.bmp ... 10.bmp
        In this case files 01-05 are treated as left eye and 06-10
        as right eye (the original IITD convention).
"""

import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config as cfg

name = "iitd"

IITD_ROOT = cfg.ROOT / "dataset" / "IITD"


def _eye_from_filename(p: Path) -> str:
    """Return 'L' or 'R' based on filename / index convention."""
    stem = p.stem.upper()
    if stem.endswith("_L"):
        return "L"
    if stem.endswith("_R"):
        return "R"
    # Fallback: plain numeric file (01..10).  IITD convention puts
    # files 01-05 as left eye and 06-10 as right eye.
    m = re.match(r"(\d+)", stem)
    if m:
        idx = int(m.group(1))
        return "L" if idx <= 5 else "R"
    return "L"


def enumerate_images():
    """Return list[{'subject','eye','path','key'}] in deterministic order."""
    if not IITD_ROOT.exists():
        return []
    records = []
    for sub_dir in sorted(IITD_ROOT.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.isdigit():
            continue
        sid = int(sub_dir.name)
        for img in sorted(sub_dir.glob("*.bmp")):
            eye = _eye_from_filename(img)
            records.append({
                "subject": sid,
                "eye":     eye,
                "path":    img,
                "key":     f"IITD_{sid:03d}_{eye}_{img.stem}",
            })
    return records


if __name__ == "__main__":
    recs = enumerate_images()
    if not recs:
        print(f"IITD root not populated yet: {IITD_ROOT}")
        print("See dataset/IITD/README.md for download instructions.")
    else:
        print(f"IITD images : {len(recs)}")
        print(f"subjects    : {len(set(r['subject'] for r in recs))}")
        print(f"first       : {recs[0]}")
        print(f"last        : {recs[-1]}")
