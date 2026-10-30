"""
Unified dataset loaders.

Each loader exposes the same interface so that the segmentation /
normalisation / evaluation modules can iterate over images
identically regardless of dataset.

A loader has two functions:

    enumerate_images() -> list[ImageRecord]
        return a list of (subject_id, eye, path) tuples ordered
        deterministically.

    name -> str
        short identifier used in output filenames.

The records returned are simple typed dicts of the form::

    {
        "subject" : int,          # numeric subject id
        "eye"     : "L" | "R",    # eye flag
        "path"    : Path,         # absolute path to the image
        "key"     : str,          # unique key for outputs/...
    }
"""
from importlib import import_module
from typing import List, Dict, Any

# Public registry ---------------------------------------------------
LOADERS = {
    "casia": "datasets.casia_thousand",
    "iitd":  "datasets.iitd",
}


def get(name: str):
    """Return the loader module for a registered dataset name."""
    name = name.lower()
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. "
                       f"Available: {list(LOADERS)}")
    return import_module(LOADERS[name])


def enumerate_images(dataset: str) -> List[Dict[str, Any]]:
    """Convenience helper."""
    return get(dataset).enumerate_images()
